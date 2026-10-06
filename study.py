#!/usr/bin/env python3
"""
s38-squeeze — historical study: is there an edge in compression → Donchian breakout → volume anomaly on HL mid-caps?

Replays up to --days of 1h candles (HL keeps 5000 bars ≈ 208 days) with the same entry arrays and the same exit code
the live book uses, in nested variants so each filter's contribution is visible:
  D      Donchian breakout only — the base rate of a 20-bar close-out
  DV     + volume z ≥ 2.5
  DVC1   + 1h compression only (the rule as first specified)
  DVC    + 1h-or-4h compression (the rule now live)
  DVC/3  DVC with the book's limits (3 positions, 24-bar cooldown) — what would actually have traded
Reported per variant: n, win rate ± SE, avg R ± SE, payoff, Sharpe ± SE, max DD at 1 % risk, trades/day; per leg;
per half of the window (regime check). Costs: 3.5 bps fees and 6/5 bps slippage per leg as in the paper book.

Universe. Default: the coins of the CURRENT universe applied to their own past — survivorship-biased (coins grew into
the $2–40M band *through* their breakouts). --pit builds a point-in-time universe instead: 1h candles for every perp
HL has ever listed (delisted included), and a coin is eligible at bar t only if its trailing-24h notional
(Σ volume × close over the last 24 bars) was inside [--min-vol, --max-vol] at that bar. One-off fetch of ~230 series
(--weight 180 ≈ 9 calls/min ≈ 25 min), cached; later runs are --offline.

--diag adds the structural checks that separate "the entry carries no information" from "the exits strangle it":
(1) forward return of every signal bar at +1/+4/+12/+24/+48 bars in ATR units per leg, against the drift of all
eligible bars, with SEs clustered by coin × 48-bar block AND two-way (coin + time block: 25 coins breaking out on the
same afternoon are one observation, not 25), net excess σ, top-1/top-5 coin concentration and a leave-one-coin-out
jackknife; (2) a random-entry Monte Carlo control through the same exit engine and costs; (3) stop velocity.
GATE: long net excess at +48 ≥ +0.50 ATR at ≥ 2σ two-way AND top-1 coin < 30 % of the net excess. On PASS, step 2
runs in the same invocation: two pre-registered long-only exit rules (A: 48-bar time exit + 3×ATR disaster stop;
B: structure stop at the signal bar's low, +1.5R → break-even, 48-bar exit) against the current ratchet, each with
its own random-entry control. On FAIL the directional breakout engine is hard-killed (see xs_portfolio.py).
What it cannot measure: the OI-inflow filter (HL has no OI history). --offline: candle cache only, no API calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import aiohttp
import numpy as np

from convex_engine import BreakoutConfig
from risk_manager import ConvexRiskManager, Position, RiskConfig
from squeeze_scanner import HOUR_MS, Candle, ClientConfig, HLInfoClient, Store, WeightLimiter, last_closed_open_time, now_ms, utc
from wfo import DAY_MS, SimTrade, WFOConfig, build_series, donchian, fetch_candles, set_stats, signals_for, simulate

log = logging.getLogger("s38.study")

VARIANTS = (
    ("D", dict(require_squeeze=False, vol_z_min=-1e9)),
    ("DV", dict(require_squeeze=False)),
    ("DVC1", dict(compression_tfs=("1h",))),
    ("DVC", dict(compression_tfs=("1h", "4h"))),
)
EXCLUDE = ("BTC", "ETH", "SOL")


# ───────────────────────────── formatting ─────────────────────────────


def fmt(s) -> str:
    if s.n == 0:
        return "n=0"
    wr_se = math.sqrt(s.win_rate * (1 - s.win_rate) / s.n)
    sharpe = f"{s.sharpe:+.2f}±{s.sharpe_se:.2f}" if math.isfinite(s.sharpe) else "   —   "
    payoff = f"{s.payoff:.2f}" if math.isfinite(s.payoff) else " —"
    se_r = f"{s.se_r:.2f}" if math.isfinite(s.se_r) else "—"
    return (f"n={s.n:<4} win {s.win_rate:.2f}±{wr_se:.2f}  avgR {s.avg_r:+.3f}±{se_r}  payoff {payoff:>5}  "
            f"ΣR {s.sum_r:+7.1f}  Sharpe {sharpe}  DD {s.max_dd:5.1%}")


def verdict(s) -> str:
    if s.n < 20 or not math.isfinite(s.se_r) or s.se_r == 0:
        return "too few trades to say"
    t = s.avg_r / s.se_r
    if t >= 2:
        return f"positive at {t:.1f}σ"
    if t <= -2:
        return f"negative at {abs(t):.1f}σ"
    return f"not distinguishable from zero ({t:+.1f}σ)"


def report_variant(name: str, trades: Sequence, start_ms: int, end_ms: int, risk: float) -> tuple:
    days = (end_ms - start_ms) / DAY_MS
    mid = (start_ms + end_ms) // 2
    lines = []
    allst = set_stats(trades, start_ms, end_ms, risk)
    lines.append(f"[{name:<5}] all    {fmt(allst)}  ({allst.n / days:.2f}/day)  → {verdict(allst)}")
    for leg in ("long", "short"):
        sub = [t for t in trades if t.side == leg]
        st = set_stats(sub, start_ms, end_ms, risk)
        lines.append(f"[{name:<5}] {leg:<6} {fmt(st)}  → {verdict(st)}")
    for label, sub in (("1st half", [t for t in trades if t.open_ms < mid]), ("2nd half", [t for t in trades if t.open_ms >= mid])):
        st = set_stats(sub, start_ms if label == "1st half" else mid, mid if label == "1st half" else end_ms, risk)
        lines.append(f"[{name:<5}] {label:<6} {fmt(st)}  → {verdict(st)}")
    out = {"all": vars(allst), "long": vars(set_stats([t for t in trades if t.side == "long"], start_ms, end_ms, risk)),
           "short": vars(set_stats([t for t in trades if t.side == "short"], start_ms, end_ms, risk))}
    return "\n".join(lines), out


# ───────────────────────────── point-in-time universe ─────────────────────────────


def pit_eligible_hours(candles: Sequence[Candle], min_vol: float, max_vol: float, window: int = 24) -> set:
    """Hour-opens (bar open + 1h, the hour in which the bar's close is acted on) at which the coin's trailing
    `window`-bar notional Σ v·c lay inside [min_vol, max_vol]. The first window−1 bars are never eligible."""
    if len(candles) < window:
        return set()
    ntl = np.array([c.v * c.c for c in candles], dtype=float)
    roll = np.convolve(ntl, np.ones(window), mode="valid")          # roll[j] = Σ ntl[j : j+window]
    out = set()
    for j, x in enumerate(roll):
        if min_vol <= x <= max_vol:
            out.add(int(candles[j + window - 1].t) + HOUR_MS)
    return out


def universe_profile(series: dict, start_ms: int, end_ms: int, today: Sequence[str]) -> dict:
    """How many coins were eligible per hour, and how many of today's universe were eligible at the window start."""
    counts: dict = {}
    for s in series.values():
        for x, e in zip(s.t, s.eligible):
            if e and start_ms <= int(x) < end_ms:
                counts[int(x)] = counts.get(int(x), 0) + 1
    n = np.array(sorted(counts.values())) if counts else np.array([0])
    first_hours = [h for h in counts if h < start_ms + DAY_MS]
    at_start = set()
    for coin, s in series.items():
        if coin in today and any(e and start_ms <= int(x) < start_ms + DAY_MS for x, e in zip(s.t, s.eligible)):
            at_start.add(coin)
    return dict(median=float(np.median(n)), lo=int(n.min()), hi=int(n.max()), n_coins=len(series),
                today=len(today), today_at_start=len(at_start), hours=len(counts), first_hours=len(first_hours))


def volume_unit_check(store: Store, candles: dict) -> Optional[float]:
    """Median of Σ(last 24 bars v·c) / HL dayNtlVlm over the coins of the latest scan: ≈ 1 means candle `v` is in
    base units (which the PIT filter assumes); ≈ price would mean it is already notional."""
    ref = store.latest_day_ntl_vlm()
    ratios = []
    for coin, vol in ref.items():
        cs = candles.get(coin)
        if cs and len(cs) >= 24 and vol > 0:
            ratios.append(sum(c.v * c.c for c in cs[-24:]) / vol)
    return float(np.median(ratios)) if ratios else None


# ───────────────────────────── clustered statistics ─────────────────────────────


def cluster_var(dev: np.ndarray, keys: Sequence) -> float:
    """Σ_g (Σ_{i∈g} dev_i)² / n² — the variance of the mean when observations are correlated within clusters."""
    sums: dict = {}
    for d, k in zip(dev, keys):
        sums[k] = sums.get(k, 0.0) + float(d)
    return sum(v * v for v in sums.values()) / (len(dev) ** 2)


def cluster_se(x: np.ndarray, keys: Sequence) -> float:
    """SE of the mean clustered one-way. Equals the plain SE (up to n/(n−1)) when every cluster is one observation."""
    if len(x) < 2:
        return float("nan")
    return math.sqrt(cluster_var(x - x.mean(), keys))


def two_way_se(x: np.ndarray, a_keys: Sequence, b_keys: Sequence) -> float:
    """Cameron–Gelbach–Miller two-way clustered SE of the mean: V_A + V_B − V_{A∩B}. With A = coin and B = time
    block, serial correlation within a coin and the cross-section of coins moving together in the same block are
    both one observation each. Falls back to the larger one-way variance if the difference is not positive."""
    if len(x) < 2:
        return float("nan")
    dev = x - x.mean()
    va, vb = cluster_var(dev, a_keys), cluster_var(dev, b_keys)
    vab = cluster_var(dev, list(zip(a_keys, b_keys)))
    v = va + vb - vab
    if v <= 0:
        v = max(va, vb)
    return math.sqrt(v)


# ───────────────────────────── forward profile ─────────────────────────────

HORIZONS = (1, 4, 12, 24, 48)
BLOCK = 48                      # bars per time block for clustering
RECOVER_BARS = 48               # window for "reached +2R anyway" after an early stop


def forward_samples(series: dict, picks: dict, start_ms: int, end_ms: int) -> dict:
    """picks: coin → ±1/0 per bar. → {leg: {k: dict(atr, pct, coins, blocks)}} — the signed close-to-close return
    k bars after each picked bar, in units of that bar's ATR(14) and in %, with its coin and 48-bar time block."""
    out: dict = {}
    for leg, want in (("long", 1), ("short", -1)):
        out[leg] = {}
        for k in HORIZONS:
            vals, pcts, coins, blocks = [], [], [], []
            for coin, s in series.items():
                sig = picks.get(coin)
                if sig is None:
                    continue
                idx = np.nonzero(sig == want)[0]
                idx = idx[(s.t[idx] >= start_ms) & (s.t[idx] < end_ms) & (idx + k < len(s.t))]
                if len(idx) == 0:
                    continue
                atr = s.atr[idx]
                ok = np.isfinite(atr) & (atr > 0)
                idx, atr = idx[ok], atr[ok]
                d = want * (s.c[idx + k] - s.c[idx])
                vals.extend((d / atr).tolist())
                pcts.extend((d / s.c[idx]).tolist())
                coins.extend([coin] * len(idx))
                blocks.extend((int(x) // (BLOCK * HOUR_MS)) for x in s.t[idx])
            out[leg][k] = dict(atr=np.array(vals), pct=np.array(pcts), coins=coins, blocks=blocks)
    return out


def summarize(sample: dict) -> dict:
    x = sample["atr"]
    if len(x) == 0:
        return dict(n=0, atr=float("nan"), se_cb=float("nan"), se_2w=float("nan"), pct=float("nan"), hit=float("nan"))
    cb = list(zip(sample["coins"], sample["blocks"]))
    return dict(n=len(x), atr=float(x.mean()), se_cb=cluster_se(x, cb), se_2w=two_way_se(x, sample["coins"], sample["blocks"]),
                pct=float(sample["pct"].mean()), hit=float((x > 0).mean()))


def forward_profile(series: dict, picks: dict, start_ms: int, end_ms: int) -> dict:
    """{leg: {k: summary}} — see forward_samples / summarize."""
    raw = forward_samples(series, picks, start_ms, end_ms)
    return {leg: {k: summarize(raw[leg][k]) for k in HORIZONS} for leg in raw}


def all_bars(series: dict, start_ms: int, end_ms: int, direction: int = 1) -> dict:
    """Every eligible bar as a pick of `direction` — the unconditional drift of the (point-in-time) universe."""
    return {coin: np.where((s.t >= start_ms) & (s.t < end_ms) & s.eligible, direction, 0) for coin, s in series.items()}


def net_excess(sig: dict, drift: dict) -> dict:
    """Signal forward return minus the drift mean, with SEs (two-way, coin×block) combined in quadrature, the per-coin
    contributions to the total net excess (top-1 / top-5 share) and a leave-one-coin-out jackknife."""
    x = sig["atr"]
    if len(x) == 0 or len(drift["atr"]) == 0:
        return dict(n=0)
    d_mean = float(drift["atr"].mean())
    d_2w = two_way_se(drift["atr"], drift["coins"], drift["blocks"])
    d_cb = cluster_se(drift["atr"], list(zip(drift["coins"], drift["blocks"])))
    net = x - d_mean
    mean = float(net.mean())
    se_2w = math.sqrt(two_way_se(x, sig["coins"], sig["blocks"]) ** 2 + d_2w ** 2)
    se_cb = math.sqrt(cluster_se(x, list(zip(sig["coins"], sig["blocks"]))) ** 2 + d_cb ** 2)
    contrib: dict = {}
    for v, c in zip(net, sig["coins"]):
        contrib[c] = contrib.get(c, 0.0) + float(v)
    total = float(net.sum())
    ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
    top1 = ranked[0][1] / total if total > 0 else float("inf")
    top5 = sum(v for _, v in ranked[:5]) / total if total > 0 else float("inf")
    # jackknife over coins: θ_(c) = mean of the sample without coin c
    counts: dict = {}
    for c in sig["coins"]:
        counts[c] = counts.get(c, 0) + 1
    loo = {c: (total - contrib[c]) / (len(net) - counts[c]) for c in contrib if len(net) > counts[c]}
    g = len(loo)
    if g >= 2:
        th = np.array(list(loo.values()))
        se_jack = math.sqrt((g - 1) / g * float(((th - th.mean()) ** 2).sum()))
        worst = min(loo.items(), key=lambda kv: kv[1])
    else:
        se_jack, worst = float("nan"), ("", float("nan"))
    return dict(n=len(net), mean=mean, se_2w=se_2w, se_cb=se_cb, drift=d_mean, total=total, top1=ranked[0][0], top1_share=top1,
                top5_share=top5, n_coins=len(contrib), jack_min=worst[1], jack_min_coin=worst[0], se_jack=se_jack)


# ───────────────────────────── random control, stops ─────────────────────────────


def random_signals(series: dict, start_ms: int, end_ms: int, n_per_side: int, rng: np.random.Generator,
                   sides: Sequence[int] = (1, -1)) -> dict:
    """n_per_side random (coin, bar) entries per direction, uniform over the window's eligible bars with a usable ATR."""
    pool = [(coin, i) for coin, s in series.items()
            for i in np.nonzero((s.t >= start_ms) & (s.t < end_ms) & s.eligible & np.isfinite(s.atr) & (s.atr > 0))[0]
            if i + 1 < len(s.t)]
    take = min(len(sides) * n_per_side, len(pool))
    chosen = rng.choice(len(pool), size=take, replace=False) if take else []
    sig = {coin: np.zeros(len(s.t), dtype=int) for coin, s in series.items()}
    per = max(1, take // len(sides))
    for j, p in enumerate(chosen):
        coin, i = pool[p]
        sig[coin][i] = sides[min(len(sides) - 1, j // per)]
    return sig


def stop_velocity(trades: Sequence, series: dict) -> dict:
    """Full stops (stage 0): share of all trades, share hit within ≤ 3 bars of entry, median bars to stop, and how
    many of the early stops reached +2R within RECOVER_BARS of entry anyway (what a wider stop would have caught)."""
    full = [t for t in trades if t.reason == "stop stage 0"]
    bars = np.array([(t.close_ms - t.open_ms) // HOUR_MS for t in full], dtype=float)
    early = [t for t, b in zip(full, bars) if b <= 3]
    recovered = 0
    for t in early:
        s = series[t.coin]
        i0 = s.idx.get(t.open_ms)
        if i0 is None:
            continue
        d = 1 if t.side == "long" else -1
        seg_h, seg_l = s.h[i0:i0 + RECOVER_BARS], s.l[i0:i0 + RECOVER_BARS]
        best = (seg_h.max() - t.entry) if d > 0 else (t.entry - seg_l.min())
        if best >= 2.0 * t.r_unit:
            recovered += 1
    n = len(trades)
    return dict(n=n, full=len(full), full_frac=len(full) / n if n else float("nan"),
                early=len(early), early_frac=len(early) / len(full) if full else float("nan"),
                median_bars=float(np.median(bars)) if len(bars) else float("nan"),
                recovered=recovered, recovered_frac=recovered / len(early) if early else float("nan"))


def exit_breakdown(trades: Sequence) -> list:
    groups: dict = {}
    for t in trades:
        groups.setdefault(t.reason, []).append(t.r)
    order = ["stop stage 0", "stop stage 1", "stop stage 2", "open at window end", "stop", "be", "time"]
    rows = []
    for reason in order + [r for r in groups if r not in order]:
        if reason in groups:
            rs = np.array(groups[reason])
            rows.append((reason, len(rs), float(rs.mean()), len(rs) / len(trades)))
    return rows


def friction(trades: Sequence, cfg: WFOConfig) -> tuple:
    """(median stop width as % of price, median round-trip friction in R) over the given trades."""
    if not trades:
        return float("nan"), float("nan")
    width = np.array([t.r_unit / t.entry for t in trades])
    rt = 2 * cfg.taker_fee + (cfg.entry_slip_bps + cfg.exit_slip_bps) / 1e4
    return float(np.median(width)), float(np.median(rt / width))


# ───────────────────────────── step 2: pre-registered exit rules ─────────────────────────────


@dataclass(frozen=True)
class ExitRule:
    name: str
    time_bars: int = 48             # exit at the close of the n-th bar after entry
    stop_atr: float = 3.0           # initial stop distance in ATR (ignored when structure=True)
    structure: bool = False         # stop at the signal bar's low (long) / high (short), at least min_struct_atr away
    min_struct_atr: float = 0.5
    be_trigger_R: float = 0.0       # 0 = never move the stop; else ratchet to entry ± be_offset_R × R at this excursion
    be_offset_R: float = 0.1


RULE_A = ExitRule("A: 48-bar time exit, 3×ATR disaster stop", 48, 3.0)
RULE_B = ExitRule("B: structure stop at the signal bar's low, +1.5R → BE+0.1R, 48-bar exit", 48, 0.0, structure=True,
                  be_trigger_R=1.5)


def simulate_rule(series: dict, signals: dict, start_ms: int, end_ms: int, cfg: WFOConfig, rule: ExitRule,
                  band_check: bool = True) -> list:
    """Bar-by-bar replay of `signals` (coin → ±1/0 per bar) under an ExitRule, with the same entry timing, costs and
    portfolio limits as wfo.simulate: entry at the signal bar's close × (1 ± slip), the stop first tested on the next
    bar (intra-bar), stop fills at the stop ∓ slip, time exits at the close ∓ slip. R = the rule's initial risk."""
    hours = sorted({int(x) for s in series.values() for x in s.t if start_ms <= int(x) < end_ms})
    e_slip, x_slip, fee = cfg.entry_slip_bps / 1e4, cfg.exit_slip_bps / 1e4, cfg.taker_fee
    bands = {coin: donchian(s, 20) for coin, s in series.items()} if band_check else {}
    open_pos: dict = {}
    cooldown: dict = {}
    trades: list = []

    def close(pos: dict, px: float, t_close: int, reason: str) -> None:
        gross = pos["d"] * (px - pos["entry"])
        net = gross - fee * (pos["entry"] + px)
        trades.append(SimTrade(pos["coin"], "long" if pos["d"] > 0 else "short", pos["opened"], t_close, pos["entry"], px,
                               pos["r_unit"], net / pos["r_unit"], pos["stage"], reason, net / pos["atr"]))
        open_pos.pop(pos["coin"], None)
        cooldown[pos["coin"]] = t_close + cfg.cooldown_bars * HOUR_MS

    for t in hours:
        for coin, pos in list(open_pos.items()):                     # 1. manage the bar that just closed
            s = series[coin]
            i = s.idx.get(t)
            if i is None:
                continue
            d = pos["d"]
            hit = s.l[i] <= pos["stop"] if d > 0 else s.h[i] >= pos["stop"]
            if hit:
                close(pos, pos["stop"] * (1 - d * x_slip), t + HOUR_MS, "be" if pos["stage"] else "stop")
                continue
            pos["best"] = max(pos["best"], s.h[i]) if d > 0 else min(pos["best"], s.l[i])
            if rule.be_trigger_R > 0 and d * (pos["best"] - pos["entry"]) / pos["r_unit"] >= rule.be_trigger_R:
                cand = pos["entry"] + d * rule.be_offset_R * pos["r_unit"]
                if (d > 0 and cand > pos["stop"]) or (d < 0 and cand < pos["stop"]):
                    pos["stop"], pos["stage"] = cand, 1
            pos["bars"] += 1
            if pos["bars"] >= rule.time_bars:
                close(pos, float(s.c[i]) * (1 - d * x_slip), t + HOUR_MS, "time")
        for coin, s in series.items():                               # 2. entries on the bar that just closed
            i = s.idx.get(t)
            if i is None or coin not in signals or signals[coin][i] == 0 or i + 1 >= len(s.t) or coin in open_pos:
                continue
            if cooldown.get(coin, 0) > t + HOUR_MS or len(open_pos) >= cfg.max_positions:
                continue
            d = int(signals[coin][i])
            if band_check:
                upper, lower = bands[coin]
                if (d > 0 and s.o[i + 1] <= upper[i]) or (d < 0 and s.o[i + 1] >= lower[i]):
                    continue
            atr = s.atr[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = float(s.c[i]) * (1 + d * e_slip)
            if rule.structure:
                level = float(s.l[i]) if d > 0 else float(s.h[i])
                floor_px = entry - d * rule.min_struct_atr * atr
                stop = min(level, floor_px) if d > 0 else max(level, floor_px)
            else:
                stop = entry - d * rule.stop_atr * atr
            r_unit = d * (entry - stop)
            if r_unit <= 0:
                continue
            open_pos[coin] = dict(coin=coin, d=d, entry=entry, stop=stop, r_unit=r_unit, atr=float(atr), best=entry,
                                  stage=0, bars=0, opened=t + HOUR_MS)
    for coin, pos in list(open_pos.items()):
        s = series[coin]
        close(pos, float(s.c[-1]), int(s.t[-1]) + HOUR_MS, "open at window end")
    trades.sort(key=lambda x: x.close_ms)
    return trades


def fmt_rule(s) -> str:
    if s.n == 0:
        return "n=0"
    wr_se = math.sqrt(s.win_rate * (1 - s.win_rate) / s.n)
    sharpe = f"{s.sharpe:+.2f}±{s.sharpe_se:.2f}" if math.isfinite(s.sharpe) else "   —   "
    payoff = f"{s.payoff:.2f}" if math.isfinite(s.payoff) else " —"
    atr = f"{s.avg_atr:+.3f}±{s.se_atr:.3f}" if math.isfinite(s.avg_atr) else "—"
    return (f"n={s.n:<4} win {s.win_rate:.2f}±{wr_se:.2f}  payoff {payoff:>5}  avgR {s.avg_r:+.3f}±{s.se_r:.3f}  "
            f"avgATR {atr}  Sharpe {sharpe}  maxDD {s.max_dd_r:5.1f}R ({s.max_dd:.0%} at 1 %)")


def step2_report(series: dict, start_ms: int, end_ms: int, base: BreakoutConfig, rcfg: RiskConfig, free: WFOConfig,
                 book: WFOConfig, seeds: int, n_ctrl: int) -> tuple:
    """Long-only DVC signals under the two pre-registered exit rules and the current ratchet, free and with the book's
    limits, each with a random-entry control through the same rule."""
    text, out = [], {}
    live = replace(base, compression_tfs=("1h", "4h"), allow_shorts=False)
    sig = {c: signals_for(s, 20, live) for c, s in series.items()}
    days = (end_ms - start_ms) / DAY_MS
    mid = (start_ms + end_ms) // 2
    text.append("── STEP 2 · long-only DVC signals, point-in-time universe, pre-registered exit rules; R = each rule's own "
                "initial risk, avgATR = net return in entry-bar ATR units (comparable across rules); maxDD in ΣR ──")
    runs = [("A", RULE_A, None), ("B", RULE_B, None), ("C", None, "current ratchet: 1.5×ATR stop, +2R BE, +3R Chandelier 2.5×ATR")]
    ctrl_sides = (1,)
    for tag, rule, label in runs:
        name = rule.name if rule else label
        text.append(f"[{tag}] {name}")

        def run_rule(signals: dict, cfg: WFOConfig, band: bool) -> list:
            if rule is None:
                return simulate(series, 20, 2.5, start_ms, end_ms, cfg, live, rcfg, signals=None if band else signals)
            return simulate_rule(series, signals, start_ms, end_ms, cfg, rule, band_check=band)

        tr_free = run_rule(sig, free, True)
        tr_book = run_rule(sig, book, True)
        ctrl: list = []
        for seed in range(seeds):
            rs = random_signals(series, start_ms, end_ms, n_ctrl, np.random.default_rng(100 + seed), sides=ctrl_sides)
            ctrl.extend(run_rule(rs, free, False))
        st_free, st_book, st_ctrl = (set_stats(tr_free, start_ms, end_ms, 0.01), set_stats(tr_book, start_ms, end_ms, 0.01),
                                     set_stats(ctrl, start_ms, end_ms, 0.01))
        diff = st_free.avg_atr - st_ctrl.avg_atr
        dse = math.sqrt(st_free.se_atr ** 2 + st_ctrl.se_atr ** 2) if math.isfinite(st_free.se_atr) and math.isfinite(st_ctrl.se_atr) else float("nan")
        text.append(f"  free   {fmt_rule(st_free)}  ({st_free.n / days:.2f}/day)")
        text.append(f"  /3     {fmt_rule(st_book)}  ({st_book.n / days:.2f}/day)")
        text.append(f"  random {fmt_rule(st_ctrl)}")
        text.append(f"  signal − random: {diff:+.3f}±{dse:.3f} ATR → " + (f"{diff / dse:+.1f}σ" if math.isfinite(dse) and dse > 0 else "—"))
        for label2, sub, a, b in (("1st half", [t for t in tr_free if t.open_ms < mid], start_ms, mid),
                                  ("2nd half", [t for t in tr_free if t.open_ms >= mid], mid, end_ms)):
            text.append(f"  {label2:<6} {fmt_rule(set_stats(sub, a, b, 0.01))}")
        text.append("  exits: " + " · ".join(f"{r} n={n} meanR {m:+.2f} ({sh:.0%})" for r, n, m, sh in exit_breakdown(tr_free)))
        out[tag] = dict(name=name, free=vars(st_free), book=vars(st_book), control=vars(st_ctrl), diff_atr=diff, diff_se=dse,
                        exits=exit_breakdown(tr_free))
        text.append("")
    return "\n".join(text), out


# ───────────────────────────── diagnostics (--diag) ─────────────────────────────


def fmt_fp(e: dict) -> str:
    if e["n"] == 0:
        return f"{'n=0':<52}"
    return f"n={e['n']:<5} {e['atr']:+.3f} ±{e['se_cb']:.3f}cb ±{e['se_2w']:.3f}2w ATR {e['pct']:+6.2%} hit {e['hit']:.2f}"


def diag_report(series: dict, start_ms: int, end_ms: int, base: BreakoutConfig, rcfg: RiskConfig, free: WFOConfig,
                dvc_trades: Sequence, seeds: int, n_per_side: int, gate_atr: float = 0.5, gate_sigma: float = 2.0,
                gate_top1: float = 0.30) -> tuple:
    text, out = [], {}
    live = replace(base, compression_tfs=("1h", "4h"))
    picks = {"D": {c: signals_for(s, 20, replace(base, require_squeeze=False, vol_z_min=-1e9)) for c, s in series.items()},
             "DVC": {c: signals_for(s, 20, live) for c, s in series.items()}}
    drift_raw = forward_samples(series, all_bars(series, start_ms, end_ms), start_ms, end_ms)["long"]
    drift = {k: summarize(drift_raw[k]) for k in HORIZONS}
    text.append("── 1. forward close-to-close return of every signal bar at +k bars, signed by direction (short = −return), "
                "in units of the signal bar's ATR(14) and in %; mean ± SE clustered by coin×48-bar block (cb) and two-way "
                "coin + time block (2w); 'drift' = every eligible bar, long sign ──")
    out["forward"] = {"drift": drift}
    raw_by = {}
    for name, pk in picks.items():
        raw = forward_samples(series, pk, start_ms, end_ms)
        raw_by[name] = raw
        fp = {leg: {k: summarize(raw[leg][k]) for k in HORIZONS} for leg in raw}
        out["forward"][name] = fp
        for k in HORIZONS:
            dr = drift[k]
            text.append(f"[{name:<4}] +{k:<3} L {fmt_fp(fp['long'][k])} │ S {fmt_fp(fp['short'][k])} │ "
                        f"drift {dr['atr']:+.3f}±{dr['se_2w']:.3f} ATR {dr['pct']:+6.2%}")
        text.append("")
    width, fric = friction(dvc_trades, free)
    text.append(f"friction reference: median stop width {width:.2%} of price (= 1 R = 1.5 ATR); round trip "
                f"{(2 * free.taker_fee + (free.entry_slip_bps + free.exit_slip_bps) / 1e4):.2%} ≈ {fric:.2f} R ≈ {fric * 1.5:.2f} ATR")
    # net excess over drift, two-way σ, concentration, jackknife — DVC, both legs, +12/+24/+48
    text.append("── net excess over drift (signal − all-bars mean), σ on the two-way SE; share of the total net excess from the "
                "top-1 / top-5 coins; leave-one-coin-out minimum and jackknife SE ──")
    out["net"] = {}
    for leg in ("long", "short"):
        out["net"][leg] = {}
        for k in (12, 24, 48):
            ne = net_excess(raw_by["DVC"][leg][k], drift_raw[k])
            out["net"][leg][k] = ne
            if ne["n"] == 0:
                text.append(f"[DVC {leg:<5}] +{k:<3} n=0")
                continue
            conc = (f"top-1 {ne['top1']} {ne['top1_share']:.0%}  top-5 {ne['top5_share']:.0%}" if math.isfinite(ne["top1_share"])
                    else "concentration n/a (total net excess ≤ 0)")
            text.append(f"[DVC {leg:<5}] +{k:<3} net {ne['mean']:+.3f} ±{ne['se_cb']:.3f}cb ±{ne['se_2w']:.3f}2w ATR = "
                        f"{ne['mean'] / ne['se_2w']:+.1f}σ(2w)  {conc}  coins {ne['n_coins']}  "
                        f"LOO min {ne['jack_min']:+.3f} (drop {ne['jack_min_coin']})  SE_jack {ne['se_jack']:.3f}")
    ne48 = out["net"]["long"].get(48, dict(n=0))
    if ne48["n"]:
        ok_atr = ne48["mean"] >= gate_atr
        ok_sig = ne48["mean"] / ne48["se_2w"] >= gate_sigma
        ok_top = ne48["top1_share"] < gate_top1
        passed = ok_atr and ok_sig and ok_top
        top = f"{ne48['top1_share']:.0%}" if math.isfinite(ne48["top1_share"]) else "n/a"
        text.append(f"GATE: long net excess +48 = {ne48['mean']:+.3f} ATR ({'≥' if ok_atr else '<'} {gate_atr}), "
                    f"{ne48['mean'] / ne48['se_2w']:+.1f}σ two-way ({'≥' if ok_sig else '<'} {gate_sigma}), "
                    f"top-1 coin {top} ({'<' if ok_top else '≥'} {gate_top1:.0%}) → "
                    + ("PASS — the long breakout edge survives point-in-time membership and two-way clustering; step 2 follows"
                       if passed else "FAIL — hard-kill the directional breakout engine (see xs_portfolio.py for the next vector)"))
    else:
        passed = False
        text.append("GATE: no long signals → FAIL")
    out["gate"] = dict(passed=passed, net48=ne48)
    # 2. random-entry control through the same exits
    text.append("")
    text.append(f"── 2. random-entry control: {seeds} seeds × {n_per_side} random entries per side through the same exits "
                f"(stop 1.5×ATR14, +2R BE, +3R Chandelier 2.5×ATR), same costs, no portfolio limits ──")
    pooled, per_seed = [], []
    for seed in range(seeds):
        sig = random_signals(series, start_ms, end_ms, n_per_side, np.random.default_rng(seed))
        tr = simulate(series, 20, 2.5, start_ms, end_ms, free, base, rcfg, signals=sig)
        pooled.extend(tr)
        per_seed.append(set_stats(tr, start_ms, end_ms, 0.01).avg_r)
    out["control"] = {}
    for leg in ("all", "long", "short"):
        c_tr = pooled if leg == "all" else [t for t in pooled if t.side == leg]
        d_tr = list(dvc_trades) if leg == "all" else [t for t in dvc_trades if t.side == leg]
        cs, ds = set_stats(c_tr, start_ms, end_ms, 0.01), set_stats(d_tr, start_ms, end_ms, 0.01)
        diff = ds.avg_r - cs.avg_r
        dse = math.sqrt(ds.se_r ** 2 + cs.se_r ** 2) if math.isfinite(ds.se_r) and math.isfinite(cs.se_r) else float("nan")
        extra = f"  per-seed avgR {min(per_seed):+.3f} … {max(per_seed):+.3f}" if leg == "all" and per_seed else ""
        text.append(f"[random] {leg:<5} {fmt(cs)}{extra}")
        text.append(f"[DVC−random] {leg:<5} {diff:+.3f}±{dse:.3f} R → " + (f"{diff / dse:+.1f}σ" if math.isfinite(dse) and dse > 0 else "—"))
        out["control"][leg] = dict(control=vars(cs), dvc=vars(ds), diff=diff, diff_se=dse)
    # 3. stop velocity
    text.append("")
    text.append("── 3. stop velocity: full stops (stage 0), share hit within ≤ 3 bars of entry, and how many of those reached +2R "
                f"within {RECOVER_BARS} bars anyway ──")
    out["stops"] = {}
    for name, tr in (("DVC", dvc_trades), ("DVC long", [t for t in dvc_trades if t.side == "long"]), ("random", pooled)):
        sv = stop_velocity(tr, series)
        out["stops"][name] = sv
        text.append(f"[{name:<8}] trades {sv['n']:<5} full stops {sv['full']:<5} ({sv['full_frac']:.0%})  within ≤3 bars "
                    f"{sv['early']:<5} ({sv['early_frac']:.0%} of full stops, median {sv['median_bars']:.0f} bars to stop)  "
                    f"of those reached +2R within {RECOVER_BARS} bars anyway: {sv['recovered']} ({sv['recovered_frac']:.0%})")
    text.append("")
    text.append("── DVC exits by reason ──")
    out["exits"] = exit_breakdown(dvc_trades)
    for reason, n, mean_r, share in out["exits"]:
        text.append(f"  {reason:<20} n={n:<5} meanR {mean_r:+.3f}  ({share:.0%})")
    return "\n".join(text), out


# ───────────────────────────── main ─────────────────────────────


async def run(a: argparse.Namespace) -> int:
    store = Store(a.db)
    today = store.latest_universe()
    t_end = now_ms()
    end_closed = last_closed_open_time("1h", t_end) + HOUR_MS
    start_ms = end_closed - a.days * DAY_MS
    warm_start = start_ms - a.warmup_bars * HOUR_MS
    t0 = time.monotonic()
    if a.coins:
        coins = [c.strip().upper() for c in a.coins.split(",") if c.strip()]
    elif a.pit:
        coins = store.cached_coins() if a.offline else []
    else:
        coins = today
    if a.offline:
        if not coins:
            print("no coins: run the scanner once first, pass --coins, or fetch with --pit online", file=sys.stderr)
            return 1
        last_closed = last_closed_open_time("1h", t_end)
        candles = {c: store.read_candles(c, warm_start, last_closed) for c in coins}
        candles = {c: v for c, v in candles.items() if v}
        log.info("offline: %d of %d coins have cached candles", len(candles), len(coins))
    else:
        async with aiohttp.ClientSession() as session:
            client = HLInfoClient(ClientConfig(base_url=a.base_url.rstrip("/"), weight_per_minute=a.weight), session, WeightLimiter(a.weight))
            if a.pit and not a.coins:
                assets = await client.meta_and_asset_ctxs()
                coins = sorted(x.coin for x in assets)
                log.info("point-in-time universe: %d perps listed (incl. %d delisted)", len(coins), sum(x.is_delisted for x in assets))
            if not coins:
                print("no universe: run the scanner once first or pass --coins", file=sys.stderr)
                return 1
            log.info("fetching %d coins × up to %d days of 1h candles at %d weight/min (~%.0f min)", len(coins), a.days, a.weight,
                     len(coins) * 20 / a.weight)
            candles = await fetch_candles(store, client, coins, warm_start, t_end)
    series = {}
    pit_sets = {}
    for coin, cs in candles.items():
        if a.pit:
            if coin in EXCLUDE:
                continue
            elig = pit_eligible_hours(cs, a.min_vol, a.max_vol)
            pit_sets[coin] = elig
        else:
            elig = {int(c.t) + HOUR_MS for c in cs}
        s = build_series(coin, cs, elig, [], BreakoutConfig(require_oi=False))
        if s is not None and s.eligible.any():
            series[coin] = s
    if not series:
        print("no candle data", file=sys.stderr)
        return 1
    first_bar = min(int(s.t[0]) for s in series.values())
    start_ms = max(start_ms, first_bar + a.warmup_bars * HOUR_MS)
    days = (end_closed - start_ms) / DAY_MS
    if days <= 0:
        print(f"not enough history: {len(series)} coins, earliest bar {utc(first_bar)}, warmup {a.warmup_bars} bars", file=sys.stderr)
        return 1
    rcfg = RiskConfig()
    results, text = {}, []
    base = BreakoutConfig(require_oi=False, allow_shorts=not a.long_only)
    free = WFOConfig(max_positions=10_000, cooldown_bars=0, risk_frac=0.01)
    book = WFOConfig(max_positions=3, cooldown_bars=24, risk_frac=0.01)
    dvc_trades: list = []
    for name, over in VARIANTS:
        trades = simulate(series, 20, 2.5, start_ms, end_closed, free, replace(base, **over), rcfg)
        if name == "DVC":
            dvc_trades = trades
        block, out = report_variant(name, trades, start_ms, end_closed, 0.01)
        results[name] = out
        text.append(block)
    trades = simulate(series, 20, 2.5, start_ms, end_closed, book, replace(base, compression_tfs=("1h", "4h")), rcfg)
    block, out = report_variant("DVC/3", trades, start_ms, end_closed, 0.01)
    results["DVC/3"] = out
    text.append(block)
    uni = ""
    if a.pit:
        prof = universe_profile(series, start_ms, end_closed, today)
        unit = volume_unit_check(store, candles)
        uni = (f"point-in-time universe: {prof['n_coins']} coins ever eligible · median {prof['median']:.0f} eligible per hour "
               f"(range {prof['lo']}–{prof['hi']}) · of today's {prof['today']} coins, {prof['today_at_start']} were eligible in "
               f"the window's first day · band [{a.min_vol:,.0f}, {a.max_vol:,.0f}] on trailing-24h Σ v·c · "
               + (f"volume-unit check: median Σv·c / dayNtlVlm = {unit:.2f} (≈1 ⇒ candle v is base units)" if unit else
                  "volume-unit check: no scan to compare against"))
        if unit is not None and not (0.5 <= unit <= 2.0):
            uni += "  !! candle volume does not look like base units — PIT band is mis-scaled"
    header = (f"STUDY {utc(t_end)} · {len(series)} coins · {'POINT-IN-TIME' if a.pit else 'current'} universe · "
              f"{utc(start_ms)[:10]} → {utc(end_closed)[:10]} ({days:.0f} days) · Donchian 20 · vol z ≥ 2.5 · stop 1.5×ATR14 · "
              f"+2R BE · +3R Chandelier 2.5×ATR · costs 3.5 bps + 6/5 bps per leg · OI filter NOT applied (no history) · "
              f"{time.monotonic() - t0:.0f}s")
    legend = ("D = Donchian only · DV = + volume z · DVC1 = + 1h compression · DVC = + 1h-or-4h compression (live rule) · "
              "DVC/3 = DVC with the book's limits (3 positions, 24-bar cooldown)\n"
              "win/avgR ± standard error · Sharpe annualised from daily R sums ± SE · DD at 1 % risk per trade · "
              "σ = avgR / SE; 'not distinguishable from zero' below 2σ")
    print(header + "\n" + (uni + "\n" if uni else "") + legend + "\n" + "\n\n".join(text))
    diag = step2 = None
    if a.diag:
        t1 = time.monotonic()
        diag_text, diag = diag_report(series, start_ms, end_closed, base, rcfg, free, dvc_trades, a.seeds, a.control_n)
        print(f"\nDIAG ({time.monotonic() - t1:.0f}s)\n" + diag_text)
        if diag["gate"]["passed"] or a.step2:
            t2 = time.monotonic()
            s2_text, step2 = step2_report(series, start_ms, end_closed, base, rcfg, free, book, max(2, a.seeds // 3), a.control_n)
            print(f"\nSTEP 2 ({time.monotonic() - t2:.0f}s)" + (" — forced by --step2" if not diag["gate"]["passed"] else "") + "\n" + s2_text)
    print("Caveats: " + ("point-in-time membership from trailing-24h Σ v·c (no spread filter, delisted coins included)"
                         if a.pit else "survivorship (today's $2–40M universe applied to its past)")
          + "; paper fills at the IOC bound; no OI filter; one 200-day window is one regime.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ts": t_end, "pit": a.pit, "coins": sorted(series), "start_ms": start_ms, "end_ms": end_closed, "days": days,
                   "results": results, "diag": diag, "step2": step2}, f, indent=1,
                  default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else (x.tolist() if isinstance(x, np.ndarray) else str(x)))
    print(f"json → {a.out}")
    store.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Historical edge study of compression → Donchian → volume breakouts.")
    p.add_argument("--db", default="state/s38.db")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--warmup-bars", type=int, default=200, help="indicator warm-up before the first evaluated bar")
    p.add_argument("--coins", default="", help="comma-separated override of the universe")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--weight", type=int, default=300, help="API budget, weight/min (20 per call); 180 for the all-perps fetch")
    p.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    p.add_argument("--out", default=f"state/study_{time.strftime('%Y%m%d_%H%M', time.gmtime())}.json")
    p.add_argument("--pit", action="store_true", help="point-in-time universe over every perp HL lists (fetches ~230 series once)")
    p.add_argument("--min-vol", type=float, default=2e6, help="--pit: trailing-24h notional band, lower")
    p.add_argument("--max-vol", type=float, default=40e6, help="--pit: trailing-24h notional band, upper")
    p.add_argument("--diag", action="store_true", help="forward-return profile, random-entry control, stop velocity, gate")
    p.add_argument("--step2", action="store_true", help="run the step-2 exit rules even if the gate fails")
    p.add_argument("--offline", action="store_true", help="candle cache only, no API calls")
    p.add_argument("--seeds", type=int, default=10, help="--diag: random-entry control seeds")
    p.add_argument("--control-n", type=int, default=2000, help="--diag: random entries per side per seed")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
