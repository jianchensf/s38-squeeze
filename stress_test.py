#!/usr/bin/env python3
"""
s38-squeeze — expectancy / losing-streak / drawdown stress test of the trade-expectancy assumptions.
Pure arithmetic + Monte Carlo on the stated profile; no market data. Re-run with your own assumptions:

  python3 stress_test.py --p 0.35 --win 4.5 --fee 0.00035 --slip 0.0006 --atr 0.015 --n 200
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from risk_manager import ConvexRiskManager, RiskConfig


def streak_prob(n: int, k: int, q: float) -> float:
    """P(at least one run of ≥ k losses in n trades) at loss probability q (exact, Markov chain)."""
    s = np.zeros(k + 1)
    s[0] = 1.0
    for _ in range(n):
        t = np.zeros(k + 1)
        t[k] = s[k] + q * s[k - 1]
        for j in range(k - 1):
            t[j + 1] += q * s[j]
        t[0] = (1 - q) * s[:k].sum()
        s = t
    return float(s[k])


def streak_drawdown(f: float, k: int, rm: ConvexRiskManager, throttle: bool) -> float:
    eq = 1.0
    for _ in range(k):
        eq *= 1 - f * (rm.drawdown_throttle(1 - eq) if throttle else 1.0)
    return 1 - eq


def monte_carlo(p: float, W: float, L: float, f: float, rm: ConvexRiskManager, throttle: bool, n: int, paths: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = np.where(rng.random((paths, n)) < p, W, -L)
    eq = np.ones(paths)
    peak = np.ones(paths)
    maxdd = np.zeros(paths)
    for i in range(n):
        dd = 1 - eq / peak
        fr = f * (np.maximum(rm.cfg.dd_throttle_floor, 1 - dd / rm.cfg.dd_throttle_at) if throttle else 1.0)
        eq = eq * (1 + fr * r[:, i])
        peak = np.maximum(peak, eq)
        maxdd = np.maximum(maxdd, 1 - eq / peak)
    return maxdd, eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.35, help="win rate")
    ap.add_argument("--win", type=float, default=4.5, help="target average win, gross R")
    ap.add_argument("--loss", type=float, default=1.0, help="average loss, net R")
    ap.add_argument("--fee", type=float, default=0.00035, help="taker fee per leg")
    ap.add_argument("--slip", type=float, default=0.0006, help="slippage per leg")
    ap.add_argument("--atr", type=float, default=0.015, help="ATR14 / price for the friction-in-R conversion")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--paths", type=int, default=20_000)
    a = ap.parse_args()
    rm = ConvexRiskManager(RiskConfig())
    friction = 2 * a.fee + 2 * a.slip
    stop = rm.cfg.stop_atr_mult * a.atr
    fR = friction / stop
    Wn = a.win - fR
    E = a.p * Wn - (1 - a.p) * a.loss
    var = a.p * Wn ** 2 + (1 - a.p) * a.loss ** 2 - E ** 2
    print(f"friction {friction:.2%} round trip on a {stop:.2%} stop = {fR:.3f}R → net win {Wn:.2f}R")
    print(f"E = {a.p:.2f}×{Wn:.2f} − {1 - a.p:.2f}×{a.loss:.1f} = {E:+.3f}R/trade; {a.n} trades: {a.n * E:+.1f}R ± {math.sqrt(var * a.n):.1f}R (1σ)")
    print(f"breakeven payoff at p={a.p:.0%}: {(1 - a.p) / a.p + fR:.2f}R gross; breakeven win rate at {a.win:.1f}R: {1 / (1 + Wn):.1%}")
    q = 1 - a.p
    print(f"\nP(7 straight losses in a given window) = {q ** 7:.4f}")
    for k in (5, 7, 10, 12, 15):
        print(f"P(a run of ≥{k:2d} losses somewhere in {a.n} trades) = {streak_prob(a.n, k, q):.3f}")
    print(f"\nDrawdown from k straight −{a.loss:.0f}R losses (cap {rm.cfg.max_risk_frac:.0%}, throttle to ×{rm.cfg.dd_throttle_floor} at {rm.cfg.dd_throttle_at:.0%} DD)")
    print(f"{'f':>6} {'throttle':>9} " + " ".join(f"{k:>5d}L" for k in (5, 7, 10, 12, 15)))
    for f in (0.01, 0.02, 0.03, 0.04):
        for th in (False, True):
            print(f"{f:6.2%} {'on' if th else 'off':>9} " + " ".join(f"{streak_drawdown(f, k, rm, th):6.1%}" for k in (5, 7, 10, 12, 15)))
    print(f"\nMonte Carlo, {a.paths} paths × {a.n} trades at p={a.p:.2f}, W={Wn:.2f}R net, L={a.loss:.1f}R")
    print(f"{'f':>6} {'throttle':>9} {'P(DD≥15%)':>10} {'P(DD≥25%)':>10} {'median DD':>10} {'95th DD':>8} {'median final':>13}")
    for f in (0.01, 0.02, 0.04):
        for th in (False, True):
            dd, fin = monte_carlo(a.p, Wn, a.loss, f, rm, th, a.n, a.paths)
            print(f"{f:6.2%} {'on' if th else 'off':>9} {np.mean(dd >= 0.15):10.1%} {np.mean(dd >= 0.25):10.1%} {np.median(dd):10.1%} {np.percentile(dd, 95):8.1%} {np.median(fin):12.2f}x")


if __name__ == "__main__":
    main()
