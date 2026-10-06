#!/usr/bin/env python3
"""
s38-squeeze — next vector, offline: cross-sectional (market-neutral) portfolios on the point-in-time candle cache.

Two pre-registered signals, no grid beyond what is listed here:
  mom    trailing close-to-close return over L bars, skipping the most recent bar (L ∈ 24, 72, 168)
  carry  mean hourly funding over the last 24 h — long the most negative (paid to hold), short the most positive;
         needs funding history (--fetch-funding, paged fundingHistory, cached in funding_hourly)
Portfolio: every R hours (R ∈ 8, 24), rank the coins that are point-in-time eligible (trailing-24h Σ v·c inside the
band, see study.py --pit), long the top decile and short the bottom decile (≥ 3 names per side, ≥ 6 eligible names
or the period is skipped), equal weight per side, 50 % of capital per side (dollar-neutral, gross 1×). Entry at the
rebalance bar's close ± slip, exit at the next rebalance close; costs 3.5 bps fee + 6 bps slip per unit of turnover
(a name kept across rebalances costs nothing); funding paid/received per hour when the data is cached.

Reported per config: periods, mean return per period ± SE (bps), annualised Sharpe ± SE, hit rate, max drawdown,
turnover per period, long and short legs as cross-sectional excess over the eligible-universe mean (what a
market-neutral book actually earns), both halves of the window. Decile table (mean forward return by signal decile,
± SE over periods) for the primary config — a real cross-sectional effect is monotone, not a top-decile fluke.
8 configs: a single 2σ is what chance produces about once in three runs; look for ≥ 3σ and consistency across L and R.
Everything is offline except --fetch-funding.
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
from dataclasses import dataclass
from typing import Optional, Sequence

import aiohttp
import numpy as np

from squeeze_scanner import HOUR_MS, ClientConfig, HLInfoClient, Store, WeightLimiter, last_closed_open_time, now_ms, utc
from study import EXCLUDE, pit_eligible_hours
from wfo import DAY_MS

log = logging.getLogger("s38.xs")

FEE, SLIP = 0.00035, 0.0006
MOM_L = (24, 72, 168)
REBAL = (8, 24)


@dataclass
class Grid:
    """Hour-aligned panel: bar open times × coins. close/funding are NaN where a coin has no bar."""
    t: np.ndarray                 # bar open ms, hourly, ascending
    coins: list
    close: np.ndarray             # (n_coins, n_hours)
    eligible: np.ndarray          # (n_coins, n_hours) bool: the bar's trailing-24h notional was inside the band
    funding: np.ndarray           # (n_coins, n_hours) hourly rate paid at the bar's close (NaN = unknown → 0)
    has_funding: bool


def build_grid(store: Store, coins: Sequence[str], start_ms: int, end_ms: int, min_vol: float, max_vol: float) -> Grid:
    t = np.arange(start_ms, end_ms, HOUR_MS, dtype=np.int64)
    pos = {int(x): i for i, x in enumerate(t)}
    names, close, elig, fund = [], [], [], []
    any_funding = False
    for coin in coins:
        if coin in EXCLUDE:
            continue
        cs = store.read_candles(coin, start_ms, end_ms - 1)
        if len(cs) < 48:
            continue
        c = np.full(len(t), np.nan)
        e = np.zeros(len(t), dtype=bool)
        for x in cs:
            i = pos.get(int(x.t))
            if i is not None:
                c[i] = x.c
        for h in pit_eligible_hours(cs, min_vol, max_vol):
            i = pos.get(int(h) - HOUR_MS)
            if i is not None:
                e[i] = True
        f = np.full(len(t), np.nan)
        for ft, rate in store.funding_between(coin, start_ms, end_ms):
            i = pos.get(int(ft) - HOUR_MS)        # a print at time T settles the bar that closed at T
            if i is not None:
                f[i] = rate
                any_funding = True
        names.append(coin)
        close.append(c)
        elig.append(e)
        fund.append(f)
    return Grid(t, names, np.array(close), np.array(elig), np.array(fund), any_funding)


def signal_matrix(g: Grid, kind: str, L: int) -> np.ndarray:
    """Signal known at the close of bar i, per coin. mom: close[i−1]/close[i−1−L] − 1 (skip the last bar).
    carry: −mean(funding over bars i−23..i) — positive when the coin pays its longs."""
    n_c, n_h = g.close.shape
    out = np.full((n_c, n_h), np.nan)
    if kind == "mom":
        if n_h > L + 1:
            out[:, L + 1:] = g.close[:, L:-1] / g.close[:, :-(L + 1)] - 1.0
    elif kind == "carry":
        f = np.nan_to_num(g.funding, nan=0.0)
        w = np.ones(24) / 24.0
        for k in range(n_c):
            out[k, 23:] = -np.convolve(f[k], w, mode="valid")
    else:
        raise ValueError(kind)
    return out


def forward_returns(close: np.ndarray, i: int, j: int) -> np.ndarray:
    """close[j]/close[i] − 1 per coin; a coin that stops printing before j (delisted) exits at its last close in
    (i, j], and one with no later print at all exits flat. NaN entry → 0."""
    seg = close[:, i + 1:j + 1]
    fin = np.isfinite(seg)
    last = np.where(fin, np.arange(seg.shape[1]), -1).max(axis=1)
    exit_px = np.where(last >= 0, seg[np.arange(len(close)), np.maximum(last, 0)], close[:, i])
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = exit_px / close[:, i] - 1.0
    return np.where(np.isfinite(fr), fr, 0.0)


@dataclass
class Period:
    t_ms: int
    ret: float            # portfolio return on capital this period, net
    long_x: float         # long leg cross-sectional excess (mean long return − universe mean), gross
    short_x: float        # short leg cross-sectional excess (universe mean − mean short return), gross
    univ: float           # equal-weight eligible-universe return (the beta a neutral book removes)
    turnover: float
    n_elig: int
    funding: float


def backtest(g: Grid, sig: np.ndarray, rebal: int, start_ms: int, end_ms: int, decile: float = 0.1, min_side: int = 3,
             fee: float = FEE, slip: float = SLIP) -> tuple:
    """→ (periods, decile table). Weights ±0.5/n_side per name; turnover costs (fee + slip) per unit traded."""
    i0 = int(np.searchsorted(g.t, start_ms))
    i_end = int(np.searchsorted(g.t, end_ms))
    prev_w = np.zeros(len(g.coins))
    periods: list = []
    dec_rets: dict = {d: [] for d in range(10)}
    for i in range(i0, i_end - rebal, rebal):
        j = i + rebal
        s = sig[:, i]
        ok = g.eligible[:, i] & np.isfinite(s) & np.isfinite(g.close[:, i])      # no look at whether the coin survives to j
        idx = np.nonzero(ok)[0]
        fwd = forward_returns(g.close, i, j)
        w = np.zeros(len(g.coins))
        if len(idx) >= 2 * min_side:
            order = idx[np.argsort(s[idx])]
            n_side = max(min_side, int(round(decile * len(idx))))
            shorts, longs = order[:n_side], order[-n_side:]
            w[longs] = 0.5 / n_side
            w[shorts] = -0.5 / n_side
            # decile table (gross forward returns by signal decile)
            fr = fwd[idx]
            ranks = np.argsort(np.argsort(s[idx]))
            dec = np.minimum(9, (ranks * 10) // len(idx))
            for d in range(10):
                m = dec == d
                if m.any():
                    dec_rets[d].append(float(fr[m].mean()))
        fund = np.nan_to_num(g.funding[:, i + 1:j + 1], nan=0.0).sum(axis=1) if g.has_funding else np.zeros(len(g.coins))
        turnover = float(np.abs(w - prev_w).sum())
        gross = float((w * fwd).sum())
        funding_pnl = float((-w * fund).sum())
        ret = gross + funding_pnl - turnover * (fee + slip)
        univ = float(fwd[idx].mean()) if len(idx) else 0.0
        long_x = float(fwd[w > 0].mean() - univ) if (w > 0).any() else 0.0
        short_x = float(univ - fwd[w < 0].mean()) if (w < 0).any() else 0.0
        periods.append(Period(int(g.t[i]) + HOUR_MS, ret, long_x, short_x, univ, turnover, int(len(idx)), funding_pnl))
        prev_w = w
    table = {d: (len(v), float(np.mean(v)) if v else float("nan"), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))
             for d, v in dec_rets.items()}
    return periods, table


@dataclass
class Stats:
    n: int = 0
    mean_bps: float = float("nan")
    se_bps: float = float("nan")
    sharpe: float = float("nan")
    sharpe_se: float = float("nan")
    hit: float = float("nan")
    max_dd: float = float("nan")
    turnover: float = float("nan")
    long_x_bps: float = float("nan")
    long_x_se: float = float("nan")
    short_x_bps: float = float("nan")
    short_x_se: float = float("nan")
    univ_bps: float = float("nan")
    funding_bps: float = float("nan")
    n_elig: float = float("nan")


def stats(periods: Sequence[Period], rebal: int) -> Stats:
    st = Stats(n=len(periods))
    if len(periods) < 2:
        return st
    r = np.array([p.ret for p in periods])
    per_year = 365.0 * 24.0 / rebal
    st.mean_bps, st.se_bps = float(r.mean() * 1e4), float(r.std(ddof=1) / math.sqrt(len(r)) * 1e4)
    sd = r.std(ddof=1)
    if sd > 0:
        s = r.mean() / sd
        st.sharpe = float(s * math.sqrt(per_year))
        st.sharpe_se = float(math.sqrt((1 + 0.5 * s * s) / len(r)) * math.sqrt(per_year))
    st.hit = float((r > 0).mean())
    cum = np.cumsum(r)
    st.max_dd = float((np.maximum.accumulate(cum) - cum).max())
    st.turnover = float(np.mean([p.turnover for p in periods]))
    lx = np.array([p.long_x for p in periods]); sx = np.array([p.short_x for p in periods])
    st.long_x_bps, st.long_x_se = float(lx.mean() * 1e4), float(lx.std(ddof=1) / math.sqrt(len(lx)) * 1e4)
    st.short_x_bps, st.short_x_se = float(sx.mean() * 1e4), float(sx.std(ddof=1) / math.sqrt(len(sx)) * 1e4)
    st.univ_bps = float(np.mean([p.univ for p in periods]) * 1e4)
    st.funding_bps = float(np.mean([p.funding for p in periods]) * 1e4)
    st.n_elig = float(np.mean([p.n_elig for p in periods]))
    return st


def fmt(st: Stats) -> str:
    if st.n < 2:
        return f"n={st.n}"
    t = st.mean_bps / st.se_bps if st.se_bps > 0 else float("nan")
    return (f"n={st.n:<4} mean {st.mean_bps:+6.1f}±{st.se_bps:4.1f} bps/period ({t:+.1f}σ)  Sharpe {st.sharpe:+.2f}±{st.sharpe_se:.2f}  "
            f"hit {st.hit:.2f}  maxDD {st.max_dd:5.1%}  turnover {st.turnover:.2f}/period  "
            f"long-x {st.long_x_bps:+5.1f}±{st.long_x_se:.1f}  short-x {st.short_x_bps:+5.1f}±{st.short_x_se:.1f}  "
            f"univ {st.univ_bps:+5.1f}  funding {st.funding_bps:+4.1f}  names {st.n_elig:.0f}")


async def fetch_funding(store: Store, client: HLInfoClient, coins: Sequence[str], start_ms: int, end_ms: int,
                        max_pages: int = 40) -> int:
    """Page fundingHistory from start_ms per coin (resuming after the newest cached print), cache in funding_hourly."""
    n = 0
    for coin in coins:
        have = store.funding_between(coin, start_ms, end_ms)
        cur = (have[-1][0] + 1) if have else start_ms
        pages = 0
        while cur < end_ms and pages < max_pages:
            try:
                pts = await client.funding_history(coin, cur)
            except Exception as e:
                log.warning("fundingHistory %s: %s", coin, e)
                break
            pages += 1
            pts = [p for p in pts if p.time >= cur]
            if not pts:
                break
            store.write_funding(coin, pts)
            n += len(pts)
            if pts[-1].time >= end_ms or len(pts) < 2:
                break
            cur = pts[-1].time + 1
    return n


async def run(a: argparse.Namespace) -> int:
    store = Store(a.db)
    t_end = now_ms()
    end_ms = last_closed_open_time("1h", t_end) + HOUR_MS
    start_ms = end_ms - a.days * DAY_MS
    grid_start = start_ms - (max(MOM_L) + 2) * HOUR_MS
    coins = [c.strip().upper() for c in a.coins.split(",") if c.strip()] or store.cached_coins()
    if not coins:
        print("no cached candles: run `python3 study.py --pit --weight 180` first", file=sys.stderr)
        return 1
    if a.fetch_funding:
        f_start = end_ms - a.funding_days * DAY_MS
        async with aiohttp.ClientSession() as session:
            client = HLInfoClient(ClientConfig(base_url=a.base_url.rstrip("/"), weight_per_minute=a.weight), session, WeightLimiter(a.weight))
            # only coins that are ever eligible need funding
            g0 = build_grid(store, coins, grid_start, end_ms, a.min_vol, a.max_vol)
            need = [c for c, e in zip(g0.coins, g0.eligible) if e.any()]
            pages = math.ceil(a.funding_days * 24 / 500)
            log.info("fetching funding for %d coins × %d days (~%d calls at %d weight/min ≈ %.0f min)", len(need), a.funding_days,
                     len(need) * pages, a.weight, len(need) * pages * 20 / a.weight)
            n = await fetch_funding(store, client, need, f_start, end_ms)
            log.info("cached %d funding prints", n)
    t0 = time.monotonic()
    g = build_grid(store, coins, grid_start, end_ms, a.min_vol, a.max_vol)
    if not g.coins:
        print("no usable coins", file=sys.stderr)
        return 1
    first_elig = int(g.t[np.nonzero(g.eligible.any(axis=0))[0][0]]) if g.eligible.any() else end_ms
    start_ms = max(start_ms, first_elig)
    mid = (start_ms + end_ms) // 2
    days = (end_ms - start_ms) / DAY_MS
    print(f"XS {utc(t_end)} · {len(g.coins)} coins with candles · point-in-time band [{a.min_vol:,.0f}, {a.max_vol:,.0f}] · "
          f"{utc(start_ms)[:10]} → {utc(end_ms)[:10]} ({days:.0f} days) · decile long/short, ≥3 per side, 50 % per side · "
          f"costs {(FEE + SLIP) * 1e4:.1f} bps per unit turnover · funding {'applied' if g.has_funding else 'NOT cached (run --fetch-funding)'}")
    print("mean ± SE per rebalance period in bps of capital (σ = mean/SE) · Sharpe annualised ± SE · maxDD of cumulative return · "
          "long-x / short-x = each leg's gross excess over the eligible-universe mean (bps) · univ = that mean · funding = "
          "net funding bps per period")
    results = {}
    configs = [("mom", L, R) for L in MOM_L for R in REBAL] + ([("carry", 24, R) for R in REBAL] if g.has_funding else [])
    tables = {}
    for kind, L, R in configs:
        sig = signal_matrix(g, kind, L)
        periods, table = backtest(g, sig, R, start_ms, end_ms)
        name = f"{kind}{L if kind == 'mom' else ''}/{R}h"
        st = stats(periods, R)
        h1 = stats([p for p in periods if p.t_ms < mid], R)
        h2 = stats([p for p in periods if p.t_ms >= mid], R)
        print(f"\n[{name:<10}] {fmt(st)}")
        print(f"[{name:<10}] 1st half {fmt(h1)}")
        print(f"[{name:<10}] 2nd half {fmt(h2)}")
        results[name] = dict(all=vars(st), h1=vars(h1), h2=vars(h2))
        tables[name] = table
    primary = "mom72/24h"
    if primary in tables:
        print(f"\ndecile table · {primary} · mean forward {24}h return by signal decile (bps), ± SE over periods; 1 = lowest signal, 10 = highest")
        print("  " + "  ".join(f"D{d + 1}: {m * 1e4:+6.1f}±{se * 1e4:4.1f}" for d, (n, m, se) in sorted(tables[primary].items())))
        results["decile_" + primary] = tables[primary]
    print("\nCaveats: point-in-time membership from trailing-24h Σ v·c only (no spread filter); fills at close ± slip (thin names will "
          "do worse); 8 configs ⇒ one 2σ by chance is routine — require ≥ 3σ and consistency across L and R; one window, one regime.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ts": t_end, "coins": g.coins, "start_ms": start_ms, "end_ms": end_ms, "results": results}, f, indent=1,
                  default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else str(x))
    print(f"json → {a.out}  ({time.monotonic() - t0:.0f}s)")
    store.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cross-sectional momentum / funding-carry portfolio backtest on the point-in-time cache.")
    p.add_argument("--db", default="state/s38.db")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--coins", default="", help="comma-separated override (default: every cached coin)")
    p.add_argument("--min-vol", type=float, default=2e6)
    p.add_argument("--max-vol", type=float, default=40e6)
    p.add_argument("--fetch-funding", action="store_true", help="page fundingHistory for the eligible coins (API)")
    p.add_argument("--funding-days", type=int, default=60)
    p.add_argument("--weight", type=int, default=180)
    p.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    p.add_argument("--out", default=f"state/xs_{time.strftime('%Y%m%d_%H%M', time.gmtime())}.json")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
