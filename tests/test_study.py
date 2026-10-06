"""study.py --diag: forward-return profile, random-entry control, stop velocity; and simulate() with injected signals."""
import math

import numpy as np

from convex_engine import BreakoutConfig
from risk_manager import RiskConfig
from squeeze_scanner import HOUR_MS, Candle
from study import (HORIZONS, RULE_A, RULE_B, all_bars, cluster_se, diag_report, exit_breakdown, forward_profile,
                   forward_samples, net_excess, pit_eligible_hours, random_signals, simulate_rule, step2_report,
                   stop_velocity, two_way_se)
from test_wfo import T0, breakout_series, mk
from wfo import SimTrade, WFOConfig, build_series, simulate

FREE = WFOConfig(max_positions=10_000, cooldown_bars=0, risk_frac=0.01)
BCFG = BreakoutConfig(require_oi=False)


def walk_universe(n_coins: int = 6, n_bars: int = 900, seed: int = 3) -> dict:
    """Driftless geometric random walks with volatility regimes (so squeezes and breakouts both occur)."""
    rng = np.random.default_rng(seed)
    series = {}
    for k in range(n_coins):
        vol = rng.uniform(0.006, 0.02)
        regime = np.repeat(rng.choice([0.4, 1.0, 1.8], size=n_bars // 48 + 1), 48)[:n_bars]
        r = rng.normal(0, vol, n_bars) * regime
        px = np.exp(np.cumsum(r))
        bars = []
        for i in range(n_bars):
            o, c = (px[i - 1] if i else 1.0), px[i]
            wick = abs(rng.normal(0, vol * regime[i])) * c
            bars.append(Candle(T0 + i * HOUR_MS, T0 + (i + 1) * HOUR_MS - 1, o, max(o, c) + wick, min(o, c) - wick,
                               c, abs(rng.normal(1000, 300)) * (1 + 6 * (abs(r[i]) > 2.5 * vol * regime[i])), 10))
        s = build_series(f"W{k}", bars, {b.t + HOUR_MS for b in bars}, [], BCFG)
        series[s.coin] = s
    return series


def test_cluster_se_reduces_to_the_plain_se_for_singleton_clusters():
    x = np.random.default_rng(0).normal(0, 1, 400)
    plain = x.std(ddof=0) / math.sqrt(len(x))
    assert math.isclose(cluster_se(x, list(range(len(x)))), plain, rel_tol=1e-9)
    assert cluster_se(x, [0] * len(x)) < 1e-12                          # one cluster: no independent information
    # perfectly duplicated observations in pairs: the clustered SE is √2 × the naive one
    y = np.repeat(x[:200], 2)
    assert math.isclose(cluster_se(y, [i // 2 for i in range(400)]), math.sqrt(2) * y.std(ddof=0) / math.sqrt(400), rel_tol=1e-9)


def test_forward_profile_signs_and_units():
    bars = breakout_series(160)
    s = build_series("AAA", bars, {b.t + HOUR_MS for b in bars}, [], BCFG)
    k = 160                                                              # the breakout bar (close 110)
    picks = {"AAA": np.zeros(len(bars), dtype=int)}
    picks["AAA"][k] = 1
    fp = forward_profile({"AAA": s}, picks, int(s.t[0]), int(s.t[-1]) + 1)
    e1, e4 = fp["long"][1], fp["long"][4]
    assert e1["n"] == 1 and math.isclose(e1["atr"], (112.0 - 110.0) / s.atr[k]) and math.isclose(e1["pct"], 2.0 / 110.0)
    assert math.isnan(e1["se_cb"]) and math.isnan(e1["se_2w"])                  # one observation: no SE
    assert e4["n"] == 1 and math.isclose(e4["atr"], (118.0 - 110.0) / s.atr[k]) and e4["hit"] == 1.0
    assert fp["long"][12]["n"] == 0 and fp["short"][1]["n"] == 0        # beyond the series / no shorts
    picks["AAA"][k] = -1                                                 # a short on the same bar: the sign flips
    fp = forward_profile({"AAA": s}, picks, int(s.t[0]), int(s.t[-1]) + 1)
    assert math.isclose(fp["short"][1]["atr"], -(112.0 - 110.0) / s.atr[k]) and fp["short"][1]["hit"] == 0.0
    drift = forward_profile({"AAA": s}, all_bars({"AAA": s}, int(s.t[0]), int(s.t[-1]) + 1), int(s.t[0]), int(s.t[-1]) + 1)
    assert drift["long"][1]["n"] == int(np.isfinite(s.atr[:-1]).sum())  # every bar with a next bar and a finite ATR


def test_simulate_honours_injected_signals_and_skips_the_band_check():
    bars = breakout_series(160)
    s = build_series("AAA", bars, {b.t + HOUR_MS for b in bars}, [], BCFG)
    series = {"AAA": s}
    sig = {"AAA": np.zeros(len(bars), dtype=int)}
    sig["AAA"][40] = 1                                                   # a flat bar: no breakout, next open inside the range
    tr = simulate(series, 20, 2.5, int(s.t[0]), int(s.t[-1]) + 1, FREE, BCFG, RiskConfig(), signals=sig)
    assert len(tr) == 1 and tr[0].open_ms == int(s.t[40]) + HOUR_MS and tr[0].side == "long"
    base = simulate(series, 20, 2.5, int(s.t[0]), int(s.t[-1]) + 1, FREE, BCFG, RiskConfig())
    assert [t.open_ms for t in base] == [int(s.t[160]) + HOUR_MS]        # the real rule still fires only on the breakout


def test_random_control_runs_through_the_same_exits_on_a_driftless_walk():
    series = walk_universe()
    start, end = T0 + 200 * HOUR_MS, T0 + 900 * HOUR_MS
    sig = random_signals(series, start, end, 150, np.random.default_rng(0))
    n_long = sum(int((v == 1).sum()) for v in sig.values())
    n_short = sum(int((v == -1).sum()) for v in sig.values())
    assert n_long == n_short == 150
    assert all(start <= s.t[i] < end for c, s in series.items() for i in np.nonzero(sig[c])[0])
    tr = simulate(series, 20, 2.5, start, end, FREE, BCFG, RiskConfig(), signals=sig)
    assert 30 < len(tr) <= 300                                           # entries during an open same-coin position are skipped
    assert {t.side for t in tr} == {"long", "short"}
    rs = np.array([t.r for t in tr])
    assert np.isfinite(rs).all() and rs.min() > -2.0                     # stop fills at the stop: a loss is ≈ −1R − friction
    reasons = {t.reason for t in tr}
    assert "stop stage 0" in reasons


def test_stop_velocity_counts_early_stops_and_recoveries():
    bars = [mk(i, 100.0, 101.0, 99.0, 100.0, 1000.0) for i in range(30)]
    bars[5] = mk(5, 100.0, 100.5, 90.0, 95.0, 1000.0)                   # bar 5 wicks to 90: a stop at 97 is hit
    bars[8] = mk(8, 95.0, 120.0, 94.0, 118.0, 1000.0)                   # then +2R (entry 100, r_unit 3 → 106) is reached
    s = build_series("AAA", bars, {b.t + HOUR_MS for b in bars}, [], BCFG)
    t_open = int(s.t[5])                                                 # entered at the close of bar 4 = open of bar 5
    early = SimTrade("AAA", "long", t_open, t_open + 1 * HOUR_MS, 100.0, 97.0, 3.0, -1.1, 0, "stop stage 0")
    late = SimTrade("AAA", "long", t_open, t_open + 12 * HOUR_MS, 100.0, 97.0, 3.0, -1.1, 0, "stop stage 0")
    win = SimTrade("AAA", "long", t_open, t_open + 20 * HOUR_MS, 100.0, 115.0, 3.0, 4.9, 2, "stop stage 2")
    sv = stop_velocity([early, late, win], {"AAA": s})
    assert sv["n"] == 3 and sv["full"] == 2 and sv["early"] == 1 and sv["recovered"] == 1
    assert math.isclose(sv["full_frac"], 2 / 3) and sv["early_frac"] == 0.5 and sv["recovered_frac"] == 1.0
    assert sv["median_bars"] == 6.5
    rows = exit_breakdown([early, late, win])
    assert rows[0][:2] == ("stop stage 0", 2) and math.isclose(rows[0][2], -1.1) and rows[1][0] == "stop stage 2"


def test_diag_report_end_to_end_on_a_synthetic_universe():
    series = walk_universe(n_coins=8, n_bars=1200, seed=5)
    start, end = T0 + 200 * HOUR_MS, T0 + 1200 * HOUR_MS
    live = BreakoutConfig(require_oi=False, compression_tfs=("1h", "4h"))
    dvc = simulate(series, 20, 2.5, start, end, FREE, live, RiskConfig())
    text, out = diag_report(series, start, end, BCFG, RiskConfig(), FREE, dvc, seeds=2, n_per_side=100)
    assert set(out["forward"]) == {"drift", "D", "DVC"} and set(out["forward"]["DVC"]["long"]) == set(HORIZONS)
    assert out["forward"]["D"]["long"][1]["n"] + out["forward"]["D"]["short"][1]["n"] > 0
    assert set(out["control"]) == {"all", "long", "short"} and out["control"]["all"]["control"]["n"] > 0
    assert "stop stage 0" in {r[0] for r in out["stops"]} or out["stops"]["random"]["full"] >= 0
    assert "GATE: long net excess +48" in text and "[DVC−random] all" in text and "[random  ] trades" in text
    assert set(out["net"]["long"]) == {12, 24, 48} and "passed" in out["gate"]
    import json
    json.dumps(out, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else
               (x.tolist() if isinstance(x, np.ndarray) else str(x)))


def test_pit_eligibility_is_the_trailing_24_bar_notional():
    bars = [mk(i, 1.0, 1.0, 1.0, 1.0, 100_000.0) for i in range(30)]          # 100k notional per bar → 2.4M per 24
    elig = pit_eligible_hours(bars, 2e6, 40e6)
    assert elig == {bars[i].t + HOUR_MS for i in range(23, 30)}             # first 23 bars never eligible
    bars[27] = mk(27, 1.0, 1.0, 1.0, 1.0, 50_000_000.0)                      # a 50M bar pushes the trailing sum over the cap
    elig = pit_eligible_hours(bars, 2e6, 40e6)
    assert bars[26].t + HOUR_MS in elig and bars[27].t + HOUR_MS not in elig and bars[29].t + HOUR_MS not in elig
    assert pit_eligible_hours(bars[:10], 2e6, 40e6) == set()
    assert pit_eligible_hours([mk(i, 1.0, 1.0, 1.0, 1.0, 10.0) for i in range(30)], 2e6, 40e6) == set()   # too thin


def test_two_way_se_properties():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 600)
    coins = [f"c{i % 30}" for i in range(600)]
    blocks = [i // 30 for i in range(600)]
    plain = x.std(ddof=0) / math.sqrt(len(x))
    # independent data: both clusterings are near the plain SE and so is the two-way combination
    assert 0.6 * plain < two_way_se(x, coins, blocks) < 1.5 * plain
    # a common shock per time block makes the two-way SE track the time-clustered SE, far above the plain one
    shock = np.array([3.0 * (b % 2) for b in blocks])
    y = x + shock
    assert two_way_se(y, coins, blocks) > 3 * (y.std(ddof=0) / math.sqrt(len(y)))
    assert math.isclose(two_way_se(y, coins, blocks), cluster_se(y, blocks), rel_tol=0.5)
    # a per-coin shock makes it track the coin-clustered SE
    z = x + np.array([2.0 * (int(c[1:]) % 2) for c in coins])
    assert math.isclose(two_way_se(z, coins, blocks), cluster_se(z, coins), rel_tol=0.5)


def test_net_excess_concentration_and_jackknife():
    sig = dict(atr=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 10.0]), pct=np.zeros(6),
               coins=["A", "A", "B", "B", "C", "D"], blocks=[0, 1, 2, 3, 4, 5])
    drift = dict(atr=np.zeros(100) + 0.5, pct=np.zeros(100), coins=[f"z{i}" for i in range(100)], blocks=list(range(100)))
    ne = net_excess(sig, drift)
    net = sig["atr"] - 0.5
    assert ne["n"] == 6 and math.isclose(ne["mean"], net.mean()) and math.isclose(ne["drift"], 0.5)
    assert ne["top1"] == "D" and math.isclose(ne["top1_share"], 9.5 / net.sum())
    assert math.isclose(ne["top5_share"], 1.0)                                # only four coins
    assert ne["jack_min_coin"] == "D" and math.isclose(ne["jack_min"], (net.sum() - 9.5) / 5)
    assert ne["se_jack"] > 0 and ne["n_coins"] == 4


def test_exit_rules_time_exit_structure_stop_and_break_even():
    bars = breakout_series(160)
    s = build_series("AAA", bars, {b.t + HOUR_MS for b in bars}, [], BCFG)
    k = 160
    sig = {"AAA": np.zeros(len(bars), dtype=int)}
    sig["AAA"][k] = 1
    span = (int(s.t[0]), int(s.t[-1]) + 1)
    # A: 3×ATR disaster stop never hit on this path; the series ends 6 bars after the signal → open at window end
    tr = simulate_rule({"AAA": s}, sig, *span, FREE, RULE_A)
    assert len(tr) == 1 and tr[0].reason == "open at window end" and math.isclose(tr[0].r_unit, 3.0 * s.atr[k])
    # B: stop at the signal bar's low (99.5, a wide breakout bar → R = 10.6, +1.5R never reached on this path)
    tr = simulate_rule({"AAA": s}, sig, *span, FREE, RULE_B)
    t = tr[0]
    entry = 110.0 * (1 + FREE.entry_slip_bps / 1e4)
    assert math.isclose(t.r_unit, entry - 99.5) and t.stage == 0 and t.reason == "open at window end"
    # B with a tight signal bar: the stop is floored 0.5 ATR below the entry, +1.5R triggers break-even, the pullback hits it
    tight = [mk(i, 100.0 + (0.3 if i % 2 else -0.3), 101.0, 99.0, 100.0 + (0.3 if i % 2 else -0.3), 900.0 if i % 2 else 1100.0)
             for i in range(60)]
    tight.append(mk(60, 100.3, 111.0, 109.5, 110.0, 5000.0))
    for j, (o, h, l, c) in enumerate([(110.0, 113.0, 109.5, 112.0), (112.0, 116.0, 111.0, 115.0), (115.0, 120.0, 114.0, 119.0),
                                      (119.0, 124.0, 117.0, 118.0), (118.0, 119.0, 112.0, 113.0), (113.0, 114.0, 110.0, 111.0)], start=61):
        tight.append(mk(j, o, h, l, c, 1000.0))
    s3 = build_series("TGT", tight, {b.t + HOUR_MS for b in tight}, [], BCFG)
    sig3 = {"TGT": np.zeros(len(tight), dtype=int)}
    sig3["TGT"][60] = 1
    tr = simulate_rule({"TGT": s3}, sig3, int(s3.t[0]), int(s3.t[-1]) + 1, FREE, RULE_B)
    t = tr[0]
    floor_px = entry - 0.5 * s3.atr[60]
    assert floor_px < 109.5 and math.isclose(t.r_unit, entry - floor_px)        # the floor binds, not the 109.5 low
    assert t.stage == 1 and t.reason == "be" and t.exit > entry                 # rallied > 1.5R, then fell back to the BE stop
    assert math.isclose(t.exit, (entry + 0.1 * t.r_unit) * (1 - FREE.exit_slip_bps / 1e4))
    # time exit: a flat path after the entry closes the trade at the 48th bar's close
    flat = [mk(i, 100.0, 100.5, 99.5, 100.0, 1000.0) for i in range(260)]
    s2 = build_series("FLT", flat, {b.t + HOUR_MS for b in flat}, [], BCFG)
    sig2 = {"FLT": np.zeros(len(flat), dtype=int)}
    sig2["FLT"][100] = 1
    tr = simulate_rule({"FLT": s2}, sig2, int(s2.t[0]), int(s2.t[-1]) + 1, FREE, RULE_A, band_check=False)
    assert len(tr) == 1 and tr[0].reason == "time" and tr[0].close_ms == int(s2.t[100]) + 49 * HOUR_MS
    assert tr[0].r < 0 and math.isclose(tr[0].atr_net * s2.atr[100], tr[0].r * tr[0].r_unit)   # costs only; same net in both units
    # book limits: a second signal while a position is open is skipped, cooldown after the exit is respected
    sig2["FLT"][110] = 1
    sig2["FLT"][160] = 1                                                        # 11 bars after the time exit at bar 149 → cooldown 24
    tr = simulate_rule({"FLT": s2}, sig2, int(s2.t[0]), int(s2.t[-1]) + 1, WFOConfig(max_positions=3, cooldown_bars=24), RULE_A, band_check=False)
    assert [t.open_ms for t in tr] == [int(s2.t[100]) + HOUR_MS]


def test_step2_report_and_gate_on_a_synthetic_universe():
    series = walk_universe(n_coins=8, n_bars=1200, seed=9)
    start, end = T0 + 200 * HOUR_MS, T0 + 1200 * HOUR_MS
    text, out = step2_report(series, start, end, BCFG, RiskConfig(), FREE, WFOConfig(max_positions=3, cooldown_bars=24), 2, 100)
    assert set(out) == {"A", "B", "C"} and all(out[k]["control"]["n"] > 0 for k in out)
    assert "signal − random" in text and "[C] current ratchet" in text
    # the gate reads the +48 long net excess: on a driftless walk it should not pass
    live = BreakoutConfig(require_oi=False, compression_tfs=("1h", "4h"))
    dvc = simulate(series, 20, 2.5, start, end, FREE, live, RiskConfig())
    _, d = diag_report(series, start, end, BCFG, RiskConfig(), FREE, dvc, seeds=2, n_per_side=100)
    g = d["gate"]
    assert isinstance(g["passed"], bool)
    if g["net48"].get("n"):
        ok = g["net48"]["mean"] >= 0.5 and g["net48"]["mean"] / g["net48"]["se_2w"] >= 2 and g["net48"]["top1_share"] < 0.3
        assert g["passed"] == ok
