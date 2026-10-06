"""xs_portfolio.py: panel construction, signal alignment, delisting-aware forward returns, the backtest on injected
momentum, and paged funding fetch."""
import asyncio
import math

import numpy as np

from squeeze_scanner import HOUR_MS, Candle, FundingPoint, Store
from test_wfo import T0
from xs_portfolio import Grid, backtest, build_grid, fetch_funding, forward_returns, signal_matrix, stats


def grid_from(closes: dict, elig_all: bool = True, funding: dict = None) -> Grid:
    coins = sorted(closes)
    n = max(len(v) for v in closes.values())
    t = T0 + np.arange(n) * HOUR_MS
    close = np.full((len(coins), n), np.nan)
    for k, c in enumerate(coins):
        v = closes[c]
        close[k, :len(v)] = v
    elig = np.isfinite(close) if elig_all else np.zeros_like(close, dtype=bool)
    fund = np.full((len(coins), n), np.nan)
    if funding:
        for k, c in enumerate(coins):
            if c in funding:
                fund[k, :len(funding[c])] = funding[c]
    return Grid(t, coins, close, elig, fund, funding is not None)


def test_signal_alignment_skips_the_last_bar():
    g = grid_from({"A": np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])})
    m = signal_matrix(g, "mom", 2)
    assert np.isnan(m[0, :3]).all()
    assert math.isclose(m[0, 3], 4.0 / 1.0 - 1) and math.isclose(m[0, 5], 16.0 / 4.0 - 1)   # close[i−1]/close[i−1−L]
    f = {"A": np.array([0.0] * 23 + [1e-4] + [0.0] * 26)}
    g2 = grid_from({"A": np.ones(50)}, funding=f)
    c = signal_matrix(g2, "carry", 24)
    assert np.isnan(c[0, :23]).all() and math.isclose(c[0, 23], -1e-4 / 24) and math.isclose(c[0, 46], -1e-4 / 24)
    assert math.isclose(c[0, 47], 0.0)                                                   # the print has rolled out


def test_forward_returns_handle_delisting_mid_period():
    close = np.array([[1.0, 1.1, 1.2, np.nan, np.nan],                                 # delisted after bar 2
                      [2.0, 2.0, 2.0, 2.0, 3.0],
                      [np.nan, 1.0, 1.0, 1.0, 1.0]])                                   # no entry price
    fr = forward_returns(close, 0, 4)
    assert math.isclose(fr[0], 1.2 / 1.0 - 1) and math.isclose(fr[1], 0.5) and fr[2] == 0.0
    fr = forward_returns(np.array([[1.0, np.nan, np.nan]]), 0, 2)
    assert fr[0] == 0.0                                                                 # vanished at once: flat


def test_backtest_captures_injected_momentum_and_pays_turnover():
    rng = np.random.default_rng(4)
    n_c, n_h = 30, 1200
    drift = np.linspace(-0.004, 0.004, n_c)                                            # persistent per-coin drift
    r = rng.normal(0, 0.01, (n_c, n_h)) + drift[:, None]
    close = np.exp(np.cumsum(r, axis=1))
    g = Grid(T0 + np.arange(n_h) * HOUR_MS, [f"C{k}" for k in range(n_c)], close, np.ones((n_c, n_h), dtype=bool),
             np.full((n_c, n_h), np.nan), False)
    sig = signal_matrix(g, "mom", 72)
    periods, table = backtest(g, sig, 24, int(g.t[200]), int(g.t[-1]) + HOUR_MS)
    st = stats(periods, 24)
    assert st.n >= 40 and st.mean_bps > 0 and st.mean_bps / st.se_bps > 3                 # momentum is there by construction
    assert st.long_x_bps > 0 and st.short_x_bps > 0                                      # both legs earn cross-sectional excess
    assert 0 < st.turnover <= 2.0
    means = [table[d][1] for d in range(10)]
    assert means[9] > means[0]                                                           # top decile beats bottom decile
    # driftless: not distinguishable from zero and no better than −costs on average
    r0 = rng.normal(0, 0.01, (n_c, n_h))
    g0 = Grid(g.t, g.coins, np.exp(np.cumsum(r0, axis=1)), g.eligible, g.funding, False)
    p0, _ = backtest(g0, signal_matrix(g0, "mom", 72), 24, int(g.t[200]), int(g.t[-1]) + HOUR_MS)
    s0 = stats(p0, 24)
    assert abs(s0.mean_bps / s0.se_bps) < 3
    # every period's turnover cost shows up: the gross sum minus the net sum equals turnover × (fee + slip)
    gross = sum(p.ret + p.turnover * (0.00035 + 0.0006) for p in p0)
    assert math.isclose(gross - sum(p.ret for p in p0), sum(p.turnover for p in p0) * 0.00095)


def test_funding_sign_longs_pay_positive_funding():
    n_h = 100
    close = np.ones((6, n_h))
    fund = np.full((6, n_h), 1e-4)                                                        # +1 bp/h for everyone
    g = Grid(T0 + np.arange(n_h) * HOUR_MS, [f"C{k}" for k in range(6)], close, np.ones((6, n_h), dtype=bool), fund, True)
    sig = np.tile(np.arange(6, dtype=float)[:, None], (1, n_h))                         # fixed ranking → zero turnover after the 1st
    periods, _ = backtest(g, sig, 8, int(g.t[10]), int(g.t[-1]) + HOUR_MS)
    assert all(math.isclose(p.funding, 0.0, abs_tol=1e-12) for p in periods)            # dollar-neutral: funding nets to zero
    fund[:3] = 0.0                                                                       # only the top three (longs) pay
    periods, _ = backtest(g, sig, 8, int(g.t[10]), int(g.t[-1]) + HOUR_MS)
    assert all(math.isclose(p.funding, -0.5 * 8 * 1e-4, rel_tol=1e-9) for p in periods)   # longs pay 8 hours of +1 bp on half the capital


def test_build_grid_and_paged_funding_fetch(tmp_path):
    store = Store(str(tmp_path / "xs.db"))
    bars = [Candle(T0 + i * HOUR_MS, T0 + (i + 1) * HOUR_MS - 1, 1.0, 1.0, 1.0, 1.0, 200_000.0, 1) for i in range(60)]
    store.write_candles("AAA", bars)
    store.write_candles("BTC", bars)
    g = build_grid(store, ["AAA", "BTC"], T0, T0 + 60 * HOUR_MS, 2e6, 40e6)
    assert g.coins == ["AAA"] and g.close.shape == (1, 60) and not g.has_funding
    assert not g.eligible[0, :23].any() and g.eligible[0, 23:].all()                      # 24 × 200k = 4.8M inside the band

    class Client:
        def __init__(self):
            self.calls = []

        async def funding_history(self, coin, start_ms):
            self.calls.append(start_ms)
            pts = [FundingPoint(T0 + i * HOUR_MS, 1e-5 * i, 0.0) for i in range(60) if T0 + i * HOUR_MS >= start_ms]
            return pts[:25]                                                             # server page of 25

    cl = Client()
    n = asyncio.run(fetch_funding(store, cl, ["AAA"], T0, T0 + 60 * HOUR_MS))
    assert n == 60 and len(cl.calls) == 4 and cl.calls[1] == T0 + 24 * HOUR_MS + 1   # 3 pages (resume = last print + 1 ms) + one empty call
    g = build_grid(store, ["AAA"], T0, T0 + 60 * HOUR_MS, 2e6, 40e6)
    assert g.has_funding and math.isclose(g.funding[0, 9], 1e-5 * 10)                    # the print at T settles the bar that closed at T
    cl.calls.clear()
    asyncio.run(fetch_funding(store, cl, ["AAA"], T0, T0 + 60 * HOUR_MS))
    assert cl.calls == [T0 + 59 * HOUR_MS + 1]                                           # resumes after the newest cached print
    store.close()
