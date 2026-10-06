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
from squeeze_scanner import HOUR_MS, ClientConfig, HLInfoClient, Store, WeightLimiter, last_closed_open_time, now_ms, utc
from wfo import DAY_MS, WFOConfig, build_series, fetch_candles, set_stats, simulate

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
    for name, over in VARIANTS:
        trades = simulate(series, 20, 2.5, start_ms, end_closed, free, replace(base, **over), rcfg)
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
    print("\nCaveats: survivorship (today's $2–40M universe applied to its past); paper fills at the IOC bound; "
          "no OI filter; one 200-day window is one regime.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ts": t_end, "coins": sorted(series), "start_ms": start_ms, "end_ms": end_closed, "days": days,
                   "results": results}, f, indent=1, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x)
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
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
