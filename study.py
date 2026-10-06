#!/usr/bin/env python3
"""
s38-squeeze — historical study: is there an edge in compression → Donchian breakout → volume anomaly on HL mid-caps?

Replays up to --days of 1h candles (HL keeps 5000 bars ≈ 208 days) for the coins of the CURRENT universe with the
same entry arrays and the same exit code the live book uses, in nested variants so each filter's contribution is
visible:
  D      Donchian breakout only — the base rate of a 20-bar close-out
  DV     + volume z ≥ 2.5
  DVC1   + 1h compression only (the rule as first specified)
  DVC    + 1h-or-4h compression (the rule now live)
  DVC/3  DVC with the book's limits (3 positions, 24-bar cooldown) — what would actually have traded
Reported per variant: n, win rate ± SE, avg R ± SE, payoff, Sharpe ± SE, max DD at 1 % risk, trades/day; per leg;
per half of the window (regime check). Costs: 3.5 bps fees and 6/5 bps slippage per leg as in the paper book.

What it cannot measure: the OI-inflow filter (HL has no OI history). What biases it: survivorship — today's
$2–40M universe applied to its own past — and the paper fill assumption (full size at the IOC bound).

--diag adds the three structural checks that separate "the entry carries no information" from "the exits strangle
it": (1) forward return of every signal bar at +1/+4/+12/+24/+48 bars in ATR units per leg (SE clustered by
coin × 48-bar block, because signals of one coin overlap), against the drift of all eligible bars; (2) a random-entry
Monte Carlo control through the same exit engine and costs — the entry's contribution is DVC minus control;
(3) stop velocity: share of full stops hit within ≤ 3 bars, and how many of those reached +2R within 48 bars anyway.
--offline runs from the candle cache alone (no API calls).
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
from dataclasses import replace
from typing import Sequence

import aiohttp

from convex_engine import BreakoutConfig
from risk_manager import RiskConfig
import numpy as np

from squeeze_scanner import HOUR_MS, ClientConfig, HLInfoClient, Store, WeightLimiter, last_closed_open_time, now_ms, utc
from wfo import DAY_MS, WFOConfig, build_series, fetch_candles, set_stats, signals_for, simulate

log = logging.getLogger("s38.study")

VARIANTS = (
    ("D", dict(require_squeeze=False, vol_z_min=-1e9)),
    ("DV", dict(require_squeeze=False)),
    ("DVC1", dict(compression_tfs=("1h",))),
    ("DVC", dict(compression_tfs=("1h", "4h"))),
)


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


# ───────────────────────────── diagnostics (--diag) ─────────────────────────────

HORIZONS = (1, 4, 12, 24, 48)
BLOCK = 48                      # bars per cluster for the forward-return SE
RECOVER_BARS = 48               # window for "reached +2R anyway" after an early stop


def cluster_se(x: np.ndarray, keys: Sequence) -> float:
    """SE of the mean when observations are correlated within clusters: sqrt(Σ_g (Σ_{i∈g} (x_i − m))²) / n.
    Equals the plain SE (up to n/(n−1)) when every cluster is a single observation."""
    if len(x) < 2:
        return float("nan")
    m = float(x.mean())
    sums: dict = {}
    for xi, k in zip(x - m, keys):
        sums[k] = sums.get(k, 0.0) + float(xi)
    return math.sqrt(sum(v * v for v in sums.values())) / len(x)


def forward_profile(series: dict, picks: dict, start_ms: int, end_ms: int) -> dict:
    """picks: coin → array of ±1/0 per bar. → {leg: {k: dict(n, atr, atr_se, pct, hit)}}: the signed close-to-close
    return k bars after each picked bar, in units of that bar's ATR(14) and in %, SE clustered by coin × 48-bar block."""
    out = {}
    for leg, want in (("long", 1), ("short", -1)):
        out[leg] = {}
        for k in HORIZONS:
            vals, pcts, keys = [], [], []
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
                keys.extend((coin, int(i) // BLOCK) for i in idx)
            x = np.array(vals)
            out[leg][k] = dict(n=len(x), atr=float(x.mean()) if len(x) else float("nan"), atr_se=cluster_se(x, keys),
                               pct=float(np.mean(pcts)) if pcts else float("nan"),
                               hit=float((x > 0).mean()) if len(x) else float("nan"))
    return out


def all_bars(series: dict, start_ms: int, end_ms: int, direction: int = 1) -> dict:
    """Every eligible bar as a pick of `direction` — the unconditional drift of the universe."""
    return {coin: np.where((s.t >= start_ms) & (s.t < end_ms), direction, 0) for coin, s in series.items()}


def random_signals(series: dict, start_ms: int, end_ms: int, n_per_side: int, rng: np.random.Generator) -> dict:
    """n_per_side random (coin, bar) entries per direction, uniform over the window's bars with a usable ATR."""
    pool = [(coin, i) for coin, s in series.items()
            for i in np.nonzero((s.t >= start_ms) & (s.t < end_ms) & np.isfinite(s.atr) & (s.atr > 0))[0] if i + 1 < len(s.t)]
    take = min(2 * n_per_side, len(pool))
    chosen = rng.choice(len(pool), size=take, replace=False)
    sig = {coin: np.zeros(len(s.t), dtype=int) for coin, s in series.items()}
    for j, p in enumerate(chosen):
        coin, i = pool[p]
        sig[coin][i] = 1 if j < take // 2 else -1
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
    order = ["stop stage 0", "stop stage 1", "stop stage 2", "open at window end"]
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


def fmt_fp(e: dict) -> str:
    if e["n"] == 0:
        return f"{'n=0':<40}"
    return f"n={e['n']:<5} {e['atr']:+.3f}±{e['atr_se']:.3f} ATR {e['pct']:+6.2%} hit {e['hit']:.2f}"


def gate_line(name: str, leg: str, fp: dict) -> str:
    """Best horizon of a leg with its t-stat and the list of horizons positive at ≥ 2σ."""
    usable = [k for k in HORIZONS if fp[k]["n"] > 1 and math.isfinite(fp[k]["atr_se"]) and fp[k]["atr_se"] > 0]
    if not usable:
        return f"gate: {name} {leg} leg — no usable horizon"
    best_k = max(usable, key=lambda k: fp[k]["atr"] / fp[k]["atr_se"])
    e = fp[best_k]
    sig2 = [k for k in usable if fp[k]["atr"] / fp[k]["atr_se"] >= 2]
    return (f"gate: {name} {leg:<5} best horizon +{best_k}: {e['atr']:+.3f}±{e['atr_se']:.3f} ATR = {e['atr'] / e['atr_se']:+.1f}σ; "
            + ("positive at ≥ 2σ at " + ", ".join(f"+{k}" for k in sig2) if sig2 else "no horizon positive at 2σ"))


def diag_report(series: dict, start_ms: int, end_ms: int, base: BreakoutConfig, rcfg: RiskConfig, free: WFOConfig,
                dvc_trades: Sequence, seeds: int, n_per_side: int) -> tuple:
    text, out = [], {}
    live = replace(base, compression_tfs=("1h", "4h"))
    picks = {"D": {c: signals_for(s, 20, replace(base, require_squeeze=False, vol_z_min=-1e9)) for c, s in series.items()},
             "DVC": {c: signals_for(s, 20, live) for c, s in series.items()}}
    drift = forward_profile(series, all_bars(series, start_ms, end_ms), start_ms, end_ms)["long"]
    text.append("── 1. forward close-to-close return of every signal bar at +k bars, signed by direction (short = −return), "
                "in units of the signal bar's ATR(14) and in %; mean ± SE clustered by coin × 48-bar block; "
                "'drift' = every eligible bar, long sign ──")
    out["forward"] = {"drift": drift}
    for name, pk in picks.items():
        fp = forward_profile(series, pk, start_ms, end_ms)
        out["forward"][name] = fp
        for k in HORIZONS:
            dr = drift[k]
            text.append(f"[{name:<4}] +{k:<3} L {fmt_fp(fp['long'][k])} │ S {fmt_fp(fp['short'][k])} │ "
                        f"drift {dr['atr']:+.3f}±{dr['atr_se']:.3f} ATR {dr['pct']:+6.2%}")
        text.append("")
    width, fric = friction(dvc_trades, free)
    text.append(f"friction reference: median stop width {width:.2%} of price (= 1 R = 1.5 ATR); round trip "
                f"{(2 * free.taker_fee + (free.entry_slip_bps + free.exit_slip_bps) / 1e4):.2%} ≈ {fric:.2f} R ≈ {fric * 1.5:.2f} ATR")
    for leg in ("long", "short"):
        text.append(gate_line("DVC", leg, out["forward"]["DVC"][leg]))
    out["gate"] = {leg: gate_line("DVC", leg, out["forward"]["DVC"][leg]) for leg in ("long", "short")}
    # 2. random-entry control through the same exits
    text.append("")
    text.append(f"── 2. random-entry control: {seeds} seeds × {n_per_side} random entries per side through the same exits "
                f"(stop 1.5×ATR14, +2R BE, +3R Chandelier 2.5×ATR), same costs, no portfolio limits ──")
    pooled, per_seed = [], []
    for seed in range(seeds):
        sig = random_signals(series, start_ms, end_ms, n_per_side, np.random.default_rng(seed))
        tr = simulate(series, 20, 2.5, start_ms, end_ms, free, base, rcfg, signals=sig)
        pooled.extend(tr)
        st = set_stats(tr, start_ms, end_ms, 0.01)
        per_seed.append(st.avg_r)
    out["control"] = {}
    for leg in ("all", "long", "short"):
        c_tr = pooled if leg == "all" else [t for t in pooled if t.side == leg]
        d_tr = list(dvc_trades) if leg == "all" else [t for t in dvc_trades if t.side == leg]
        cs, ds = set_stats(c_tr, start_ms, end_ms, 0.01), set_stats(d_tr, start_ms, end_ms, 0.01)
        diff = ds.avg_r - cs.avg_r
        dse = math.sqrt(ds.se_r ** 2 + cs.se_r ** 2) if math.isfinite(ds.se_r) and math.isfinite(cs.se_r) else float("nan")
        extra = f"  per-seed avgR {min(per_seed):+.3f} … {max(per_seed):+.3f}" if leg == "all" else ""
        text.append(f"[random] {leg:<5} {fmt(cs)}{extra}")
        text.append(f"[DVC−random] {leg:<5} {diff:+.3f}±{dse:.3f} R → "
                    + (f"{diff / dse:+.1f}σ" if math.isfinite(dse) and dse > 0 else "—"))
        out["control"][leg] = dict(control=vars(cs), dvc=vars(ds), diff=diff, diff_se=dse)
    # 3. stop velocity
    text.append("")
    text.append("── 3. stop velocity: full stops (stage 0), share hit within ≤ 3 bars of entry, and how many of those reached +2R "
                f"within {RECOVER_BARS} bars anyway ──")
    out["stops"] = {}
    for name, tr in (("DVC", dvc_trades), ("random", pooled)):
        sv = stop_velocity(tr, series)
        out["stops"][name] = sv
        text.append(f"[{name:<6}] trades {sv['n']:<5} full stops {sv['full']:<5} ({sv['full_frac']:.0%})  within ≤3 bars "
                    f"{sv['early']:<5} ({sv['early_frac']:.0%} of full stops, median {sv['median_bars']:.0f} bars to stop)  "
                    f"of those reached +2R within {RECOVER_BARS} bars anyway: {sv['recovered']} ({sv['recovered_frac']:.0%})")
    text.append("")
    text.append("── DVC exits by reason ──")
    out["exits"] = exit_breakdown(dvc_trades)
    for reason, n, mean_r, share in out["exits"]:
        text.append(f"  {reason:<20} n={n:<5} meanR {mean_r:+.3f}  ({share:.0%})")
    return "\n".join(text), out


async def run(a: argparse.Namespace) -> int:
    store = Store(a.db)
    coins = [c.strip().upper() for c in a.coins.split(",") if c.strip()] or store.latest_universe()
    if not coins:
        print("no universe: run the scanner once first or pass --coins", file=sys.stderr)
        return 1
    t_end = now_ms()
    end_closed = last_closed_open_time("1h", t_end) + HOUR_MS
    start_ms = end_closed - a.days * DAY_MS
    warm_start = start_ms - a.warmup_bars * HOUR_MS
    t0 = time.monotonic()
    if a.offline:
        last_closed = last_closed_open_time("1h", t_end)
        candles = {c: store.read_candles(c, warm_start, last_closed) for c in coins}
        candles = {c: v for c, v in candles.items() if v}
        log.info("offline: %d of %d coins have cached candles", len(candles), len(coins))
    else:
        async with aiohttp.ClientSession() as session:
            client = HLInfoClient(ClientConfig(base_url=a.base_url.rstrip("/"), weight_per_minute=a.weight), session, WeightLimiter(a.weight))
            log.info("fetching %d coins × up to %d days of 1h candles at %d weight/min (~%.0fs)", len(coins), a.days, a.weight,
                     len(coins) * 20 / (a.weight / 60))
            candles = await fetch_candles(store, client, coins, warm_start, t_end)
    series = {}
    for coin, cs in candles.items():
        s = build_series(coin, cs, {int(c.t) + HOUR_MS for c in cs}, [], BreakoutConfig(require_oi=False))
        if s is not None:
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
    header = (f"STUDY {utc(t_end)} · {len(series)} coins of the current universe · {utc(start_ms)[:10]} → {utc(end_closed)[:10]} "
              f"({days:.0f} days) · Donchian 20 · vol z ≥ 2.5 · stop 1.5×ATR14 · +2R BE · +3R Chandelier 2.5×ATR · "
              f"costs 3.5 bps + 6/5 bps per leg · OI filter NOT applied (no history) · {time.monotonic() - t0:.0f}s")
    legend = ("D = Donchian only · DV = + volume z · DVC1 = + 1h compression · DVC = + 1h-or-4h compression (live rule) · "
              "DVC/3 = DVC with the book's limits (3 positions, 24-bar cooldown)\n"
              "win/avgR ± standard error · Sharpe annualised from daily R sums ± SE · DD at 1 % risk per trade · "
              "σ = avgR / SE; 'not distinguishable from zero' below 2σ")
    print(header + "\n" + legend + "\n" + "\n\n".join(text))
    diag = None
    if a.diag:
        t1 = time.monotonic()
        diag_text, diag = diag_report(series, start_ms, end_closed, base, rcfg, free, dvc_trades, a.seeds, a.control_n)
        print(f"\nDIAG ({time.monotonic() - t1:.0f}s)\n" + diag_text)
    print("\nCaveats: survivorship (today's $2–40M universe applied to its past); paper fills at the IOC bound; "
          "no OI filter; one 200-day window is one regime.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ts": t_end, "coins": sorted(series), "start_ms": start_ms, "end_ms": end_closed, "days": days,
                   "results": results, "diag": diag}, f, indent=1,
                  default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x)
    print(f"json → {a.out}")
    store.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Historical edge study of compression → Donchian → volume breakouts on the current universe.")
    p.add_argument("--db", default="state/s38.db")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--warmup-bars", type=int, default=200, help="indicator warm-up before the first evaluated bar")
    p.add_argument("--coins", default="", help="comma-separated override of the universe")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--weight", type=int, default=300)
    p.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    p.add_argument("--out", default=f"state/study_{time.strftime('%Y%m%d_%H%M', time.gmtime())}.json")
    p.add_argument("--diag", action="store_true", help="forward-return profile, random-entry control, stop velocity")
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
