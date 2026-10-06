import asyncio
import math

import numpy as np

from convex_engine import BreakoutConfig, ConvexExecutor, ConvexRunner, ConvexSignalEngine, ExecConfig, OIHistory
from fakes import FakeInfo
from portfolio import PositionBook
from risk_manager import ConvexRiskManager, RiskConfig
from squeeze_scanner import (HOUR_MS, AccountConfig, AssetCtx, Candle, FuelMetrics, FundingPoint, ScanRow, SqueezeConfig,
                             Store, compute_squeeze)
from synth import UNIVERSE
from wfo import (BASELINE, SimTrade, WFOConfig, build_series, calibrate_costs, reconcile, set_stats, signals_for,
                 simulate, walk_forward)

T0 = 1_760_000_000_000 - (1_760_000_000_000 % HOUR_MS)


def mk(i, o, h, l, c, v):
    t = T0 + i * HOUR_MS
    return Candle(t, t + HOUR_MS - 1, o, h, l, c, v, 10)


def breakout_series() -> list:
    """60 flat bars in a squeeze, a volume breakout at bar 60, a four-bar rally, then a reversal that hits the Chandelier."""
    bars = []
    for i in range(60):
        c = 100.0 + (0.3 if i % 2 else -0.3)
        bars.append(mk(i, c, c + 1.0, c - 1.0, c, 900.0 if i % 2 else 1100.0))
    bars.append(mk(60, 100.3, 111.0, 99.5, 110.0, 5000.0))
    path = [(110.0, 113.0, 109.5, 112.0), (112.0, 116.0, 111.0, 115.0), (115.0, 120.0, 114.0, 119.0),
            (119.0, 124.0, 117.0, 118.0), (118.0, 119.0, 112.0, 113.0), (113.0, 114.0, 110.0, 111.0)]
    for k, (o, h, l, c) in enumerate(path, start=61):
        bars.append(mk(k, o, h, l, c, 1000.0))
    return bars


def oi_samples_for(bars):
    """One sample per bar close (+60 s), like the scanner: 1000 until the breakout bar closes, 1040 after."""
    return [(b.t + HOUR_MS + 60_000, 1040.0 if b.t >= bars[60].t else 1000.0) for b in bars]


def run(coro):
    return asyncio.run(coro)


def test_series_and_signals():
    bars = breakout_series()
    s = build_series("X", bars, {b.t + HOUR_MS for b in bars}, oi_samples_for(bars), BreakoutConfig())
    d = signals_for(s, 20, BreakoutConfig())
    assert list(np.nonzero(d)[0]) == [60] and d[60] == 1
    assert s.vol_z[60] > 30 and s.sq_ago[60] == 1 and s.sq_run[60] >= 3
    assert s.oi_open[60] == 1000.0 and s.oi_close[60] == 1040.0 and np.isnan(s.oi_open[0])   # no sample before the first close
    # a longer lookback still sees the same break; a bar outside the universe never signals
    assert signals_for(s, 32, BreakoutConfig())[60] == 1
    s.eligible[60] = False
    assert signals_for(s, 20, BreakoutConfig())[60] == 0


def test_replay_reproduces_the_live_book():
    """The WFO replay and the live dry flow (engine → executor → book) must produce the same trade."""
    bars = breakout_series()
    hours = {b.t + HOUR_MS for b in bars}
    s = build_series("X", bars, hours, oi_samples_for(bars), BreakoutConfig())
    sim = simulate({"X": s}, BASELINE["donchian_len"], BASELINE["trail_atr_mult"], bars[0].t, bars[-1].t + HOUR_MS,
                   WFOConfig(), BreakoutConfig(), RiskConfig())
    assert len(sim) == 1
    st = sim[0]
    assert st.side == "long" and st.open_ms == bars[61].t and st.stage == 2 and st.reason.startswith("stop stage 2")

    # live flow on the same bars, cycle by cycle after each close (OI recorded by the engine from the ctx)
    meta, ctx0 = next((m, c) for m, c in UNIVERSE if m["name"] == "AAA")
    acct = AccountConfig(equity_usd=1000.0, risk_pct=0.01)
    info = FakeInfo({"X": "100.0"})
    oi = OIHistory()
    engine = ConvexSignalEngine(BreakoutConfig(max_signal_age_s=1e9), oi)
    risk = ConvexRiskManager()
    book = PositionBook(risk, acct, info, None, live=False)
    ex = ConvexExecutor(ExecConfig(live=False), acct, info, None, risk=risk, book=book)
    runner = ConvexRunner(engine, ex, book)
    for i in range(30, len(bars)):
        closed = bars[:i + 1]
        t_ms = bars[i].t + HOUR_MS + 60_000
        oi_now = 1040.0 if bars[i].t >= bars[60].t else 1000.0
        ctx = AssetCtx.from_api(dict(meta, name="X"), dict(ctx0, markPx=str(bars[i].c), midPx=str(bars[i].c), openInterest=str(oi_now)))
        info.mids = {"X": bars[i].c}
        sq = compute_squeeze("1h", closed, SqueezeConfig(), 20)
        row = ScanRow(ts=t_ms, ctx=ctx, squeeze={"1h": sq} if sq else {}, fuel=FuelMetrics(0, 0, False, False, False, False, None, None, False), score=0.0)
        run(runner.on_cycle([row], {("X", "1h"): closed}, t_ms))
    assert book.positions == {} and book.stats().n == 1
    live_r = book._r_multiples[0]
    live_trade = next(r for r in ex.reports if r.status == "dry")
    assert abs(live_trade.avg_px - st.entry) < 0.02                      # 110.08 (tick-rounded bound) vs 110.088
    assert abs(live_r - st.r) < 0.01 and 1.0 < st.r < 1.4
    assert math.isclose(st.exit, 114.9 * 0.9995, rel_tol=1e-3)           # Chandelier 124 − 2.5 × ATR14(3.64), −5 bps


def test_set_stats_and_gate():
    base = T0
    trades = [SimTrade("X", "long", base, base + k * 24 * HOUR_MS + 3600_000, 100, 101, 1, r, 0, "")
              for k, r in enumerate([1.0, -1.0, 2.0, -1.0])]
    st = set_stats(trades, base, base + 4 * 24 * HOUR_MS, 0.01)
    assert st.n == 4 and st.wins == 2 and math.isclose(st.avg_r, 0.25) and math.isclose(st.sum_r, 1.0)
    assert math.isclose(st.sharpe, 0.25 / 1.5 * math.sqrt(365), rel_tol=1e-9) and math.isclose(st.payoff, 1.5)
    assert math.isclose(st.max_dd, 1 - 1.009699 / 1.019898, abs_tol=1e-6)
    assert set_stats([], base, base + 24 * HOUR_MS, 0.01).n == 0

    cfg = WFOConfig(min_is_trades=4, min_oos_trades=2)
    start, split, end = base, base + 20 * 24 * HOUR_MS, base + 30 * 24 * HOUR_MS

    def scripted(is_rs, oos_rs):
        def fn(series, L, m, s, e, c, b, r, f=None):
            if (L, m) == (20, 2.5):
                rs_i, rs_o = [0.5, -1.0, 0.5, -1.0], [0.2, -1.0]
            elif (L, m) == (24, 3.0):
                rs_i, rs_o = is_rs, oos_rs
            else:
                rs_i, rs_o = [], []
            out = [SimTrade("X", "long", start + k * 24 * HOUR_MS, start + k * 24 * HOUR_MS + HOUR_MS, 1, 1, 1, r, 0, "") for k, r in enumerate(rs_i)]
            out += [SimTrade("X", "long", split + k * 24 * HOUR_MS, split + k * 24 * HOUR_MS + HOUR_MS, 1, 1, 1, r, 0, "") for k, r in enumerate(rs_o)]
            return out
        return fn

    # candidate strong in- and out-of-sample → adopted
    pts, chosen, adopted, verdict = walk_forward({}, cfg, BreakoutConfig(), RiskConfig(), start, split, end,
                                                 simulate_fn=scripted([2.0, 1.5, -1.0, 2.5, 1.0, -0.5], [1.5, 2.0, -0.5, 1.0]))
    assert adopted and chosen == {"donchian_len": 24, "trail_atr_mult": 3.0} and verdict.startswith("adopted")
    assert len(pts) == len(cfg.donchian_grid) * len(cfg.trail_grid) and 2.5 in cfg.trail_grid
    # same in-sample, bad out-of-sample → WFE gate fails → baseline
    pts, chosen, adopted, verdict = walk_forward({}, cfg, BreakoutConfig(), RiskConfig(), start, split, end,
                                                 simulate_fn=scripted([2.0, 1.5, -1.0, 2.5, 1.0, -0.5], [-1.0, -1.0, 0.1, -1.0]))
    assert not adopted and chosen == BASELINE and "WFE" in verdict
    # too few out-of-sample trades → baseline
    pts, chosen, adopted, verdict = walk_forward({}, cfg, BreakoutConfig(), RiskConfig(), start, split, end,
                                                 simulate_fn=scripted([2.0, 1.5, -1.0, 2.5, 1.0, -0.5], [1.5]))
    assert not adopted and "n_OOS 1 < 2" in verdict
    # nothing trades enough → baseline with the reason
    pts, chosen, adopted, verdict = walk_forward({}, cfg, BreakoutConfig(), RiskConfig(), start, split, end,
                                                 simulate_fn=scripted([1.0], [1.0]))
    assert not adopted and "no grid point has" in verdict


def test_reconcile_and_cost_calibration():
    sim = [SimTrade("X", "long", T0 + 60_000, T0 + 5 * HOUR_MS, 100, 104, 2, 1.9, 2, "stop"),
           SimTrade("Y", "short", T0 + 3 * HOUR_MS, T0 + 9 * HOUR_MS, 50, 49, 1, 0.9, 1, "stop")]
    logged = [{"coin": "X", "side": "long", "opened_ms": T0 + 5 * 60_000, "r_multiple": 1.8},
              {"coin": "Z", "side": "long", "opened_ms": T0, "r_multiple": -1.0}]
    r = reconcile(logged, sim)
    assert (r["logged"], r["simulated"], r["matched"], r["unmatched_sim"]) == (2, 2, 1, 1)
    assert math.isclose(r["mean_abs_r_diff"], 0.1)
    cfg, note = calibrate_costs(WFOConfig(), [])
    assert cfg.entry_slip_bps == 8.0 and note.startswith("defaults")
    trades = [{"entry_slip_bps": 6.0, "exit_slip_bps": -7.0, "fees_usd": 0.09, "size": 1.0, "entry_px": 100.0, "exit_px": 100.0}] * 10
    cfg, note = calibrate_costs(WFOConfig(), trades)
    assert cfg.entry_slip_bps == 6.0 and cfg.exit_slip_bps == 7.0 and math.isclose(cfg.taker_fee, 0.00045) and "calibrated" in note


def test_params_store_and_hot_reload(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    assert store.active_params() == {"donchian_len": 20.0, "trail_atr_mult": 2.5}
    store.set_params({"donchian_len": 24, "trail_atr_mult": 3.0}, "test", T0)
    assert store.active_params() == {"donchian_len": 24.0, "trail_atr_mult": 3.0}
    info = FakeInfo({})
    risk = ConvexRiskManager()
    book = PositionBook(risk, AccountConfig(), info, None)
    ex = ConvexExecutor(ExecConfig(), AccountConfig(), info, None, risk=risk, book=book)
    runner = ConvexRunner(ConvexSignalEngine(BreakoutConfig(), OIHistory()), ex, book, store)
    runner.apply_params(store.active_params())
    assert runner.engine.cfg.donchian_len == 24 and risk.cfg.trail_atr_mult == 3.0 and book.cfg.trail_atr_mult == 3.0
    store.write_wfo_run({"ts": T0, "window_start": T0, "window_end": T0, "split_ms": T0, "n_coins": 1, "chosen_donchian": 24,
                         "chosen_trail": 3.0, "adopted": True, "verdict": "v", "detail": {"x": float("nan")}})
    assert store.con.execute("SELECT adopted, chosen_donchian FROM wfo_runs").fetchone() == (1, 24)
    store.close()


def test_trade_log_enrichment_and_events(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    bars = breakout_series()
    info = FakeInfo({"X": "100.0"})
    book = PositionBook(ConvexRiskManager(), AccountConfig(equity_usd=1000.0), info, store, live=False)
    t_open = bars[61].t + 60_000
    pos = book.register("X", 1, 110.08, 2.4, 2.68, t_open, bars[61].t, 1, ref_px=110.0)
    store.write_funding("X", [FundingPoint(hour, 1e-4, 0.0) for hour in range(bars[61].t, bars[66].t + 1, HOUR_MS)])  # +1e-4/h while open

    async def go():
        out = None
        for k in range(62, len(bars)):
            out = await book.on_cycle({("X", "1h"): bars[:k + 1]}, bars[k].t + HOUR_MS + 60_000, {"X": bars[k].c})
        return out

    s = run(go())
    tr = store.trades("dry")[0]
    assert s["open"] == 0 and tr["coin"] == "X" and tr["stage"] == 2
    assert tr["duration_s"] == (tr["closed_ms"] - t_open) / 1000 and tr["duration_s"] > 3600 * 4
    assert math.isclose(tr["entry_slip_bps"], (110.08 / 110.0 - 1) * 1e4, abs_tol=0.01)
    assert math.isclose(tr["exit_slip_bps"], 5.0, abs_tol=0.01)                 # paper exit = stop − 5 bps
    n_hours = len(store.funding_between("X", t_open, tr["closed_ms"]))
    assert n_hours >= 4 and math.isclose(tr["funding_usd"], -2.4 * 110.08 * 1e-4 * n_hours, abs_tol=1e-4)
    kinds = [k for (k,) in store.con.execute("SELECT kind FROM position_events ORDER BY ts, rowid")]
    assert kinds[0] == "open" and kinds[-1] == "close" and kinds.count("stop_move") >= 2
    assert pos.status == "closed"
    store.close()
