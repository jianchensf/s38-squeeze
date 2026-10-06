#!/usr/bin/env python3
"""
s38-squeeze — walk-forward optimisation of the two tunable parameters: Donchian lookback (16–32) and Chandelier
multiplier (2.0–3.2). Weekly job (deploy/s38-wfo.timer); `python3 wfo.py --report-only` to look without touching.

A trade log cannot evaluate other parameters — a different lookback produces different entries, a different
multiplier different exits — so this is a REPLAY of the live logic over the logged window:
  candles    HL candleSnapshot, cached in the `candles` table
  universe   the coins that were actually eligible at each hour, from the scanner's own `scans` rows (no survivorship)
  OI         the scanner's own samples (HL has no OI history); a bar without samples fails the OI test, as live
  funding    realized prints from `funding_hourly` (0 where nothing was logged, reported as coverage)
  costs      taker fee, entry and exit slippage calibrated from the trade log once ≥ 10 trades exist,
             otherwise 4.5 bps / 8 bps / 5 bps
  logic      entries: convex_engine.compute_breakout-equivalent arrays + the squeeze rule;
             exits: risk_manager.ConvexRiskManager — the very functions the live book calls
Window: the last `--days` logged days; in-sample = first part, out-of-sample = last `--oos-days`; trades belong
to the segment their entry falls in. The breaker is not simulated (a kill switch is not a parameter to optimise).

Gate — ALL of these to adopt a non-baseline point, otherwise the baseline (20, 2.5) is (re)installed:
  n_IS ≥ 30 · n_OOS ≥ 10 · Sharpe_IS > 0 · WFE = Sharpe_OOS / Sharpe_IS > 0.70 ·
  OOS max drawdown < 10 % at the current risk fraction · candidate OOS Sharpe ≥ baseline OOS Sharpe
Results → `wfo_runs`; chosen values → `params`, which the running scanner re-reads every cycle.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import aiohttp
import numpy as np

from convex_engine import BreakoutConfig
from risk_manager import ConvexRiskManager, Position, RiskConfig, stats_from_r_multiples
from squeeze_scanner import (HOUR_MS, Candle, ClientConfig, HLInfoClient, SqueezeConfig, Store, WeightLimiter,
                             last_closed_open_time, now_ms, sma, squeeze_bands, squeeze_recency, true_range, utc)

log = logging.getLogger("s38.wfo")
DAY_MS = 24 * HOUR_MS
BASELINE = {"donchian_len": 20, "trail_atr_mult": 2.5}


@dataclass(frozen=True)
class WFOConfig:
    days: int = 30
    oos_days: int = 10
    donchian_grid: tuple = tuple(range(16, 33, 2))                                   # 16 … 32
    trail_grid: tuple = tuple(sorted({round(2.0 + 0.2 * i, 1) for i in range(7)} | {2.5}))   # 2.0 … 3.2 + baseline
    min_is_trades: int = 30
    min_oos_trades: int = 10
    wfe_min: float = 0.70
    oos_dd_max: float = 0.10
    max_positions: int = 3
    cooldown_bars: int = 24
    warmup_bars: int = 200
    entry_slip_bps: float = 8.0
    exit_slip_bps: float = 5.0
    taker_fee: float = 0.00045
    risk_frac: float = 0.01          # equity curve / drawdown; replaced by the book's current fraction when run


# ───────────────────────────── per-coin arrays ─────────────────────────────


@dataclass
class CoinSeries:
    coin: str
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    atr: np.ndarray             # ATR(atr_len) per bar
    sq_ago: np.ndarray          # bars since the 1h squeeze was last on (−1 never)
    sq_run: np.ndarray          # length of that squeeze run
    sq4_ago: np.ndarray         # same, in 4h bars, from the last 4h bar closed by this 1h bar's close
    sq4_run: np.ndarray
    vol_z: np.ndarray           # bar volume vs the prior vol_len bars
    eligible: np.ndarray        # bool: coin was in the scanned universe in the hour after the bar closed
    oi_open: np.ndarray         # OI sample within tolerance of the bar open (NaN if none)
    oi_close: np.ndarray        # … of the bar close
    idx: dict                   # bar open ms -> index


def _nearest(samples_t: np.ndarray, samples_v: np.ndarray, at: np.ndarray, tol_ms: int) -> np.ndarray:
    out = np.full(len(at), np.nan)
    if len(samples_t) == 0:
        return out
    pos = np.searchsorted(samples_t, at)
    for k, (p, a) in enumerate(zip(pos, at)):
        best, dist = None, tol_ms + 1
        for j in (p - 1, p):
            if 0 <= j < len(samples_t) and abs(samples_t[j] - a) < dist:
                best, dist = j, abs(samples_t[j] - a)
        if best is not None:
            out[k] = samples_v[best]
    return out


def squeeze_4h_for_1h(t: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                      scfg: SqueezeConfig) -> tuple:
    """Aggregate 1h bars to the UTC 4h grid, run the squeeze on it, and map (ago, run) back to each 1h bar using
    the last 4h bar that had CLOSED when the 1h bar closed — what the live scanner's 4h metrics show at that time."""
    n = len(t)
    ago1h, run1h = np.full(n, -1, dtype=int), np.zeros(n, dtype=int)
    if n == 0:
        return ago1h, run1h
    g = (t // (4 * HOUR_MS)) * (4 * HOUR_MS)
    starts = np.unique(g)
    h4 = np.array([h[g == s].max() for s in starts]); l4 = np.array([l[g == s].min() for s in starts])
    c4 = np.array([c[g == s][-1] for s in starts])
    if len(starts) < scfg.length + 1:
        return ago1h, run1h
    ago4, run4 = squeeze_recency(squeeze_bands(h4, l4, c4, scfg)["on"])
    closes4 = starts + 4 * HOUR_MS
    j = np.searchsorted(closes4, t + HOUR_MS, side="right") - 1      # last 4h bar closed by this 1h close
    ok = j >= 0
    ago1h[ok], run1h[ok] = ago4[j[ok]], run4[j[ok]]
    return ago1h, run1h


def build_series(coin: str, candles: Sequence[Candle], eligible_hours: set, oi_samples: Sequence,
                 bcfg: BreakoutConfig, scfg: SqueezeConfig = SqueezeConfig()) -> Optional[CoinSeries]:
    n = len(candles)
    if n < max(scfg.length, bcfg.vol_len, bcfg.atr_len) + 2:
        return None
    t = np.array([c.t for c in candles], dtype=np.int64)
    o = np.array([c.o for c in candles]); h = np.array([c.h for c in candles])
    l = np.array([c.l for c in candles]); c_ = np.array([c.c for c in candles]); v = np.array([c.v for c in candles])
    atr = sma(true_range(h, l, c_), bcfg.atr_len)
    ago, run = squeeze_recency(squeeze_bands(h, l, c_, scfg)["on"])
    ago4, run4 = squeeze_4h_for_1h(t, o, h, l, c_, scfg)
    m = bcfg.vol_len
    vol_z = np.full(n, np.nan)
    if n > m:
        w = np.lib.stride_tricks.sliding_window_view(v, m)[:-1]          # w[k] = v[k:k+m] = prior window of bar k+m
        mean, std = w.mean(axis=1), w.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (v[m:] - mean) / std
        z[std <= 0] = np.nan
        vol_z[m:] = z
    eligible = np.array([(int(x) + HOUR_MS) in eligible_hours for x in t])
    st = np.array([s[0] for s in oi_samples], dtype=np.int64)
    sv = np.array([s[1] for s in oi_samples], dtype=float)
    tol = int(bcfg.oi_tolerance_s * 1000)
    return CoinSeries(coin, t, o, h, l, c_, v, atr, ago, run, ago4, run4, vol_z, eligible,
                      _nearest(st, sv, t, tol), _nearest(st, sv, t + HOUR_MS, tol), {int(x): i for i, x in enumerate(t)})


def donchian(s: CoinSeries, L: int) -> tuple:
    """Prior-L highest high / lowest low per bar (the bar itself excluded); NaN for the first L bars."""
    n = len(s.t)
    upper, lower = np.full(n, np.nan), np.full(n, np.nan)
    if n > L:
        upper[L:] = np.lib.stride_tricks.sliding_window_view(s.h, L)[:-1].max(axis=1)
        lower[L:] = np.lib.stride_tricks.sliding_window_view(s.l, L)[:-1].min(axis=1)
    return upper, lower


def signals_for(s: CoinSeries, L: int, bcfg: BreakoutConfig) -> np.ndarray:
    """Direction per bar (+1/−1/0) after the same filters the live engine applies (freshness aside)."""
    upper, lower = donchian(s, L)
    with np.errstate(invalid="ignore"):
        d = np.where(s.c > upper, 1, np.where(s.c < lower, -1, 0))
        if not bcfg.allow_shorts:
            d = np.where(d < 0, 0, d)
        ok = s.eligible & (s.vol_z >= bcfg.vol_z_min)
        if bcfg.require_oi:
            ok &= (s.oi_open > 0) & (s.oi_close / s.oi_open - 1.0 >= bcfg.oi_min_change)
        if bcfg.require_squeeze:
            comp = np.zeros(len(s.t), dtype=bool)
            for tf, ago, run in (("1h", s.sq_ago, s.sq_run), ("4h", s.sq4_ago, s.sq4_run)):
                if tf in bcfg.compression_tfs:
                    comp |= (ago >= 0) & (ago <= bcfg.max_bars_since_squeeze) & (run >= bcfg.min_squeeze_bars)
            ok &= comp
    return np.where(ok, d, 0)


# ───────────────────────────── replay ─────────────────────────────


@dataclass
class SimTrade:
    coin: str
    side: str
    open_ms: int
    close_ms: int
    entry: float
    exit: float
    r_unit: float
    r: float                    # net of fees (and funding when available), in R
    stage: int
    reason: str
    atr_net: float = float("nan")   # the same net return in units of the entry bar's ATR (comparable across exit rules)


def simulate(series: dict, L: int, trail: float, start_ms: int, end_ms: int, cfg: WFOConfig,
             bcfg: BreakoutConfig, rcfg: RiskConfig, funding: Optional[Callable] = None,
             signals: Optional[dict] = None) -> list:
    """Replay every eligible coin bar by bar with portfolio limits. Entries at the signal bar's close × (1 ± slip);
    the stop is first tested on the bar after the signal bar (the live fill happens minutes after the close), exits
    at the stop ∓ slip, no gap-through. `signals` (coin → ±1/0 per bar) replaces the breakout rule — used by the
    random-entry control, which also skips the breakout-only "next open back inside the range" check."""
    rm = ConvexRiskManager(replace(rcfg, trail_atr_mult=trail))
    injected = signals is not None
    sig = signals if injected else {coin: signals_for(s, L, bcfg) for coin, s in series.items()}
    bands = {coin: donchian(s, L) for coin, s in series.items()}
    hours = sorted({int(x) for s in series.values() for x in s.t if start_ms <= int(x) < end_ms})
    e_slip, x_slip, fee = cfg.entry_slip_bps / 1e4, cfg.exit_slip_bps / 1e4, cfg.taker_fee
    open_pos: dict = {}
    cooldown: dict = {}
    trades: list = []

    def close(pos: Position, px: float, t_close: int, reason: str) -> None:
        gross = pos.direction * (px - pos.entry_px)
        cost = fee * (pos.entry_px + px)
        fund = 0.0
        if funding is not None:
            fund = -pos.direction * pos.entry_px * sum(rate for _, rate in funding(pos.coin, pos.opened_ms, t_close))
        r = (gross - cost + fund) / pos.r_unit
        trades.append(SimTrade(pos.coin, pos.side, pos.opened_ms, t_close, pos.entry_px, px, pos.r_unit, r, pos.stage, reason,
                               (gross - cost + fund) / (pos.r_unit / rcfg.stop_atr_mult)))
        open_pos.pop(pos.coin, None)
        cooldown[pos.coin] = t_close + cfg.cooldown_bars * HOUR_MS

    for t in hours:
        for coin, pos in list(open_pos.items()):                     # 1. manage: the bar that opened at t has closed
            s = series[coin]
            i = s.idx.get(t)
            if i is None:
                continue
            if rm.stop_hit(pos, s.h[i], s.l[i]):
                close(pos, pos.stop * (1 - pos.direction * x_slip), t + HOUR_MS, f"stop stage {pos.stage}")
                continue
            rm.update_excursion(pos, s.h[i], s.l[i])
            atr = s.atr[i] if np.isfinite(s.atr[i]) else pos.r_unit / rcfg.stop_atr_mult
            pos.stop, pos.stage = rm.desired_stop(pos, atr)
        for coin, s in series.items():                               # 2. entries on the bar that just closed
            i = s.idx.get(t)
            if i is None or coin not in sig or sig[coin][i] == 0 or i + 1 >= len(s.t) or coin in open_pos:
                continue
            if cooldown.get(coin, 0) > t + HOUR_MS or len(open_pos) >= cfg.max_positions:
                continue
            d = int(sig[coin][i])
            upper, lower = bands[coin]
            if not injected and ((d > 0 and s.o[i + 1] <= upper[i]) or (d < 0 and s.o[i + 1] >= lower[i])):
                continue                                             # next open already back inside the range
            atr = s.atr[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = s.c[i] * (1 + d * e_slip)
            stop = rm.initial_stop(entry, d, atr)
            open_pos[coin] = Position(coin=coin, side="long" if d > 0 else "short", direction=d, entry_px=entry, size=1.0,
                                      initial_stop=stop, stop=stop, r_unit=rcfg.stop_atr_mult * atr, best_px=entry,
                                      opened_ms=t + HOUR_MS, entry_bar=t + HOUR_MS, last_bar_checked=t + HOUR_MS)
    for coin, pos in list(open_pos.items()):                         # still open at the window end: mark at last close
        s = series[coin]
        close(pos, float(s.c[-1]), int(s.t[-1]) + HOUR_MS, "open at window end")
    trades.sort(key=lambda x: x.close_ms)
    return trades


# ───────────────────────────── statistics ─────────────────────────────


@dataclass
class SetStats:
    n: int = 0
    wins: int = 0
    avg_r: float = float("nan")
    se_r: float = float("nan")
    payoff: float = float("nan")
    sum_r: float = 0.0
    sharpe: float = float("nan")      # annualised, from daily R sums (days without trades count as 0)
    sharpe_se: float = float("nan")
    max_dd: float = 0.0               # equity drawdown compounding risk_frac × R per trade
    max_dd_r: float = 0.0             # peak-to-trough drawdown of cumulative ΣR, in R
    avg_atr: float = float("nan")     # mean net return in ATR units (when the trades carry it)
    se_atr: float = float("nan")
    days: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")


def set_stats(trades: Sequence[SimTrade], start_ms: int, end_ms: int, risk_frac: float) -> SetStats:
    days = max(1, int(round((end_ms - start_ms) / DAY_MS)))
    st = SetStats(days=days)
    rs = np.array([t.r for t in trades], dtype=float)
    if len(rs) == 0:
        return st
    st.n, st.wins, st.sum_r = len(rs), int((rs > 0).sum()), float(rs.sum())
    st.avg_r = float(rs.mean())
    st.se_r = float(rs.std(ddof=1) / math.sqrt(len(rs))) if len(rs) > 1 else float("nan")
    s = stats_from_r_multiples(rs.tolist())
    st.payoff = s.payoff if math.isfinite(s.payoff) else float("nan")
    daily = np.zeros(days)
    for t in trades:
        daily[min(days - 1, max(0, (t.close_ms - start_ms) // DAY_MS))] += t.r
    sd = daily.std(ddof=1) if days > 1 else 0.0
    if sd > 0:
        s_d = daily.mean() / sd
        st.sharpe = float(s_d * math.sqrt(365))
        st.sharpe_se = float(math.sqrt((1 + 0.5 * s_d * s_d) / days) * math.sqrt(365))
    eq, peak, dd = 1.0, 1.0, 0.0
    cum, cpeak, ddr = 0.0, 0.0, 0.0
    for t in trades:
        eq *= 1 + risk_frac * t.r
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
        cum += t.r
        cpeak = max(cpeak, cum)
        ddr = max(ddr, cpeak - cum)
    st.max_dd, st.max_dd_r = float(dd), float(ddr)
    a = np.array([t.atr_net for t in trades], dtype=float)
    if np.isfinite(a).all() and len(a) > 1:
        st.avg_atr, st.se_atr = float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))
    return st


@dataclass
class GridPoint:
    L: int
    trail: float
    is_: SetStats
    oos: SetStats

    @property
    def wfe(self) -> float:
        if not math.isfinite(self.is_.sharpe) or self.is_.sharpe <= 0 or not math.isfinite(self.oos.sharpe):
            return float("nan")
        return self.oos.sharpe / self.is_.sharpe


def walk_forward(series: dict, cfg: WFOConfig, bcfg: BreakoutConfig, rcfg: RiskConfig, start_ms: int, split_ms: int,
                 end_ms: int, funding: Optional[Callable] = None, simulate_fn: Callable = simulate) -> tuple:
    """→ (grid points, chosen {donchian_len, trail_atr_mult}, adopted: bool, verdict text)."""
    points: list = []
    for L in cfg.donchian_grid:
        for m in cfg.trail_grid:
            trades = simulate_fn(series, L, m, start_ms, end_ms, cfg, bcfg, rcfg, funding)
            is_t = [t for t in trades if t.open_ms < split_ms]
            oos_t = [t for t in trades if t.open_ms >= split_ms]
            points.append(GridPoint(L, m, set_stats(is_t, start_ms, split_ms, cfg.risk_frac),
                                    set_stats(oos_t, split_ms, end_ms, cfg.risk_frac)))
    base = next(p for p in points if p.L == BASELINE["donchian_len"] and p.trail == BASELINE["trail_atr_mult"])
    cands = [p for p in points if p.is_.n >= cfg.min_is_trades and math.isfinite(p.is_.sharpe) and p.is_.sharpe > 0]
    if not cands:
        max_n = max(p.is_.n for p in points)
        return points, dict(BASELINE), False, (f"baseline: no grid point has ≥ {cfg.min_is_trades} in-sample trades with "
                                                f"positive Sharpe (max n_IS = {max_n})")
    best = max(cands, key=lambda p: p.is_.sharpe)
    fails = []
    if best.oos.n < cfg.min_oos_trades:
        fails.append(f"n_OOS {best.oos.n} < {cfg.min_oos_trades}")
    if not (math.isfinite(best.wfe) and best.wfe > cfg.wfe_min):
        fails.append(f"WFE {best.wfe:.2f} ≤ {cfg.wfe_min}")
    if best.oos.max_dd >= cfg.oos_dd_max:
        fails.append(f"OOS DD {best.oos.max_dd:.1%} ≥ {cfg.oos_dd_max:.0%}")
    if math.isfinite(base.oos.sharpe) and math.isfinite(best.oos.sharpe) and best.oos.sharpe < base.oos.sharpe:
        fails.append(f"OOS Sharpe {best.oos.sharpe:.2f} < baseline {base.oos.sharpe:.2f}")
    if best is base:
        return points, dict(BASELINE), False, (f"baseline is the in-sample best (IS Sharpe {best.is_.sharpe:.2f} ± "
                                                f"{best.is_.sharpe_se:.2f}, n={best.is_.n}; OOS {best.oos.sharpe:.2f}, n={best.oos.n})")
    if fails:
        return points, dict(BASELINE), False, (f"baseline: best IS point ({best.L}, {best.trail}) failed the gate — "
                                                + "; ".join(fails))
    return points, {"donchian_len": best.L, "trail_atr_mult": best.trail}, True, (
        f"adopted ({best.L}, {best.trail}): IS Sharpe {best.is_.sharpe:.2f} ± {best.is_.sharpe_se:.2f} (n={best.is_.n}), "
        f"OOS Sharpe {best.oos.sharpe:.2f} ± {best.oos.sharpe_se:.2f} (n={best.oos.n}), WFE {best.wfe:.2f}, "
        f"OOS DD {best.oos.max_dd:.1%}, baseline OOS Sharpe {base.oos.sharpe:.2f}")


def reconcile(logged: Sequence[dict], sim: Sequence[SimTrade], tol_ms: int = 20 * 60_000) -> dict:
    """Does the replay under the live parameters reproduce the logged trades? (coin + entry within 20 min)."""
    matched, dr = 0, []
    pool = list(sim)
    for lt in logged:
        hit = next((s for s in pool if s.coin == lt["coin"] and s.side == lt["side"] and abs(s.open_ms - lt["opened_ms"]) <= tol_ms), None)
        if hit:
            matched += 1
            dr.append(abs(hit.r - lt["r_multiple"]))
            pool.remove(hit)
    return {"logged": len(logged), "simulated": len(sim), "matched": matched,
            "mean_abs_r_diff": float(np.mean(dr)) if dr else None, "unmatched_sim": len(pool)}


# ───────────────────────────── job ─────────────────────────────


def fmt_stats(s: SetStats) -> str:
    if s.n == 0:
        return f"{'n=0':<34}"
    return (f"n={s.n:<3} avgR {s.avg_r:+.2f}±{s.se_r:.2f} Sh {s.sharpe:+5.2f}±{s.sharpe_se:.2f} DD {s.max_dd:5.1%}")


def render(points: Sequence[GridPoint], chosen: dict, verdict: str, header: str, top: int = 8) -> str:
    lines = [header, f"{'L':>3} {'trail':>5} | {'IN-SAMPLE':<46}| {'OUT-OF-SAMPLE':<46}| WFE"]
    ranked = sorted([p for p in points if p.is_.n], key=lambda p: (-(p.is_.sharpe if math.isfinite(p.is_.sharpe) else -9e9)))
    base = next(p for p in points if p.L == BASELINE["donchian_len"] and p.trail == BASELINE["trail_atr_mult"])
    shown = ranked[:top]
    if base not in shown:
        shown.append(base)
    for p in shown:
        tag = " ← baseline" if p is base else (" ← chosen" if (p.L, p.trail) == (chosen["donchian_len"], chosen["trail_atr_mult"]) else "")
        wfe = f"{p.wfe:.2f}" if math.isfinite(p.wfe) else "  —"
        lines.append(f"{p.L:>3} {p.trail:>5.1f} | {fmt_stats(p.is_):<46}| {fmt_stats(p.oos):<46}| {wfe}{tag}")
    n_traded = sum(1 for p in points if p.is_.n or p.oos.n)
    lines.append(f"{len(points)} grid points, {n_traded} with any trade · verdict: {verdict}")
    return "\n".join(lines)


async def fetch_candles(store: Store, client: HLInfoClient, coins: Sequence[str], start_ms: int, end_ms: int) -> dict:
    """1h closed bars per coin over [start_ms, end_ms], from the sqlite cache plus whatever it lacks at either end
    (the weekly WFO caches 30 days; a 200-day study must fetch the head, not just the tail). A coin listed after
    start_ms costs one empty head request per run. HL serves at most 5000 bars per request (≈ 208 days)."""
    out: dict = {}
    last_closed = last_closed_open_time("1h", end_ms)
    for coin in coins:
        have = store.read_candles(coin, start_ms, last_closed)
        asked = store.candle_request_span(coin)
        spans = []
        if not have:                                                 # nothing cached: ask unless this exact question
            if asked is None or asked[0] > start_ms or asked[1] < last_closed - DAY_MS:   # was asked within a day
                spans.append((start_ms, end_ms))
        else:
            if have[0].t > start_ms + HOUR_MS and (asked is None or asked[0] > start_ms):
                spans.append((start_ms, have[0].t - 1))          # head: only if never asked from this far back
            if have[-1].t + HOUR_MS <= last_closed:
                spans.append((have[-1].t + HOUR_MS, end_ms))
        for a, b in spans:
            try:
                new = [c for c in await client.candles(coin, "1h", a, b) if c.t <= last_closed]
                store.mark_candle_request(coin, a, b)
                if new:
                    store.write_candles(coin, new)
            except Exception as e:
                log.warning("candles %s: %s", coin, e)
        if spans:
            have = store.read_candles(coin, start_ms, last_closed)
        if have:
            out[coin] = have
    return out


def calibrate_costs(cfg: WFOConfig, trades: Sequence[dict]) -> tuple:
    """Costs from the trade log once there is enough of it; → (cfg, note)."""
    if len(trades) < 10:
        return cfg, f"defaults (n={len(trades)} logged trades < 10)"
    es = [t["entry_slip_bps"] for t in trades if t.get("entry_slip_bps") is not None]
    xs = [abs(t["exit_slip_bps"]) for t in trades if t.get("exit_slip_bps") is not None]
    fees = [t["fees_usd"] / (t["size"] * (t["entry_px"] + t["exit_px"])) for t in trades if t.get("size") and t.get("fees_usd") is not None]
    new = replace(cfg,
                  entry_slip_bps=float(np.mean(es)) if es else cfg.entry_slip_bps,
                  exit_slip_bps=float(np.mean(xs)) if xs else cfg.exit_slip_bps,
                  taker_fee=float(np.mean(fees)) if fees else cfg.taker_fee)
    return new, f"calibrated from {len(trades)} trades: fee {new.taker_fee * 1e4:.1f} bps, slip {new.entry_slip_bps:.1f}/{new.exit_slip_bps:.1f} bps"


async def run(args: argparse.Namespace) -> int:
    store = Store(args.db)
    t_end = now_ms()
    first, last = store.logged_span()
    if not first:
        print("no scan log yet — nothing to replay", file=sys.stderr)
        return 1
    window_start = max(first, t_end - args.days * DAY_MS)
    split = max(window_start, t_end - args.oos_days * DAY_MS)
    elig = store.eligibility(window_start, t_end)
    coins = sorted(elig)
    logged_days = (t_end - window_start) / DAY_MS
    bcfg = BreakoutConfig(allow_shorts=not args.long_only)
    live = store.active_params()
    rcfg = RiskConfig(trail_atr_mult=live["trail_atr_mult"])
    trades_logged = store.trades(args.mode, window_start)
    cfg, cost_note = calibrate_costs(WFOConfig(days=args.days, oos_days=args.oos_days), trades_logged)
    risk_frac, risk_note = ConvexRiskManager(rcfg).risk_fraction(stats_from_r_multiples([t["r_multiple"] for t in trades_logged]))
    cfg = replace(cfg, risk_frac=risk_frac if risk_frac > 0 else RiskConfig().bootstrap_risk_frac)

    async with aiohttp.ClientSession() as session:
        client = HLInfoClient(ClientConfig(base_url=args.base_url.rstrip("/"), weight_per_minute=args.weight), session,
                              WeightLimiter(args.weight))
        warm_start = window_start - cfg.warmup_bars * HOUR_MS
        log.info("window %s → %s (%.1f logged days, OOS from %s), %d coins, fetching candles at %d weight/min",
                 utc(window_start), utc(t_end), logged_days, utc(split), len(coins), args.weight)
        candles = await fetch_candles(store, client, coins, warm_start, t_end)

    series: dict = {}
    oi_cov = []
    for coin, cs in candles.items():
        s = build_series(coin, cs, elig[coin], store.oi_samples(coin, warm_start, t_end), bcfg)
        if s is None:
            continue
        series[coin] = s
        in_win = (s.t >= window_start) & s.eligible
        if in_win.any():
            oi_cov.append(float(np.isfinite(s.oi_open[in_win]).mean()))
    funding_lookup = store.funding_between
    points, chosen, adopted, verdict = walk_forward(series, cfg, bcfg, rcfg, window_start, split, t_end, funding_lookup)

    live_sim = simulate(series, int(live["donchian_len"]), live["trail_atr_mult"], window_start, t_end, cfg, bcfg, rcfg, funding_lookup)
    rec = reconcile(trades_logged, live_sim)
    header = (f"WFO {utc(t_end)} · window {utc(window_start)} → {utc(t_end)} ({logged_days:.1f} d, OOS from {utc(split)}) · "
              f"{len(series)} coins · OI coverage {np.mean(oi_cov) if oi_cov else 0:.0%} · costs {cost_note} · "
              f"risk {cfg.risk_frac:.2%} ({risk_note}) · live params ({int(live['donchian_len'])}, {live['trail_atr_mult']})")
    report = render(points, chosen, verdict, header)
    report += (f"\nreconcile live params vs log [{args.mode}]: {rec['matched']}/{rec['logged']} logged trades reproduced"
               f" ({rec['simulated']} simulated, {rec['unmatched_sim']} without a logged counterpart"
               + (f", mean |ΔR| {rec['mean_abs_r_diff']:.2f}" if rec['mean_abs_r_diff'] is not None else "") + ")")
    if logged_days < args.days * 0.8 or (oi_cov and np.mean(oi_cov) < 0.8):
        report += f"\nNOTE: only {logged_days:.1f} of {args.days} days logged / OI coverage incomplete — results are not meaningful yet"
    print(report)
    detail = {"points": [{"L": p.L, "trail": p.trail, "is": vars(p.is_), "oos": vars(p.oos), "wfe": p.wfe} for p in points],
              "reconcile": rec, "costs": cost_note, "risk_frac": cfg.risk_frac, "logged_days": logged_days,
              "oi_coverage": float(np.mean(oi_cov)) if oi_cov else 0.0}
    store.write_wfo_run({"ts": t_end, "window_start": window_start, "window_end": t_end, "split_ms": split,
                         "n_coins": len(series), "chosen_donchian": chosen["donchian_len"], "chosen_trail": chosen["trail_atr_mult"],
                         "adopted": adopted, "verdict": verdict, "detail": detail})
    if args.apply:
        store.set_params(chosen, f"wfo {utc(t_end)}: {verdict}", t_end)
        print(f"params set: donchian_len {chosen['donchian_len']}, trail_atr_mult {chosen['trail_atr_mult']} "
              f"({'adopted' if adopted else 'baseline'}) — the scanner picks this up within a minute")
    else:
        print("report only — params untouched")
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat and args.apply:
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                                   json={"chat_id": chat, "text": f"s38 WFO: {verdict}\n{rec['matched']}/{rec['logged']} logged trades reproduced"},
                                   timeout=aiohttp.ClientTimeout(total=10))
            except Exception as e:
                log.warning("telegram: %s", e)
    store.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward optimisation of Donchian lookback × Chandelier multiplier.")
    p.add_argument("--db", default="state/s38.db")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--oos-days", type=int, default=10)
    p.add_argument("--mode", default="dry", help="which trade log calibrates costs / risk fraction (dry|live)")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--apply", dest="apply", action="store_true", default=True, help="write the chosen params (default)")
    p.add_argument("--report-only", dest="apply", action="store_false", help="print the grid, change nothing")
    p.add_argument("--weight", type=int, default=300)
    p.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
