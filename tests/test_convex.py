import asyncio
import math
import os

import pytest

from convex_engine import (
    IOC, BreakoutConfig, ConvexExecutor, ConvexSignal, ConvexSignalEngine, ExecConfig, OIHistory, compute_breakout,
    floor_sz, parse_order_response, round_px,
)
from squeeze_scanner import (
    HOUR_MS, AccountConfig, AssetCtx, Candle, FuelMetrics, ScanRow, SqueezeConfig, compute_squeeze,
    last_closed_open_time,
)
from synth import UNIVERSE, candles_for

END = 1_760_000_000_000                          # fixed "now" for deterministic bars
BAR = last_closed_open_time("1h", END)           # open time of the breakout bar in the synthetic series


def bars(coin: str) -> list:
    return [Candle.from_api(d) for d in candles_for(coin, "1h", 0, END) if d["t"] <= BAR]


def ctx_for(coin: str, **over) -> AssetCtx:
    m, c = next((m, c) for m, c in UNIVERSE if m["name"] == coin)
    c = dict(c, **{k: str(v) for k, v in over.items()})
    return AssetCtx.from_api(m, c)


def row_for(coin: str, t_ms: int, **over) -> ScanRow:
    ctx = ctx_for(coin, **over)
    sq = compute_squeeze("1h", bars(coin), SqueezeConfig(), 20)
    fuel = FuelMetrics(ctx.funding, ctx.funding * 800, False, False, False, False, None, None, False)
    return ScanRow(ts=t_ms, ctx=ctx, squeeze={"1h": sq}, fuel=fuel, score=0.0)


# ───────────────────────────── tick rules ─────────────────────────────


@pytest.mark.parametrize("px, sd, mode, want", [
    (110.08801, 1, "down", 110.08),      # 5 sig figs → 2 decimals, floored
    (110.08801, 1, "up", 110.09),
    (110.08801, 1, "nearest", 110.09),
    (0.0123456789, 0, "nearest", 0.012346),   # 6 decimals allowed, 5 sig figs
    (123456.7, 0, "down", 123450.0),     # 5 sig figs beats decimals
    (1.23456789, 2, "up", 1.2346),       # decimals cap 4 = sig-fig cap 4
    (60000.4, 5, "nearest", 60000.0),    # BTC-like: 1 decimal allowed, 5 sig figs → integer
    (99.99999, 1, "down", 99.999),
])
def test_round_px(px, sd, mode, want):
    got = round_px(px, sd, mode)
    assert got == want
    assert float(f"{got:.8f}") == got        # exact on the wire (float_to_wire rule)


def test_floor_sz():
    assert floor_sz(5.4999, 1) == 5.4
    assert floor_sz(0.123456789, 5) == 0.12345
    assert floor_sz(7.0, 0) == 7.0
    with pytest.raises(ValueError):
        round_px(0.0, 1)


# ───────────────────────────── breakout math ─────────────────────────────


def test_compute_breakout_regimes():
    cfg = BreakoutConfig()
    k = compute_breakout(bars("KKK"), cfg)
    assert k.direction == 1 and k.close == 110.0 and k.bar_time == BAR
    assert math.isclose(k.donchian_upper, 101.3) and k.close > k.donchian_upper   # prior flat highs: 100.3 + 1
    assert math.isclose(k.donchian_lower, 98.7)                                   # wick 25 bars back is outside the window
    assert k.vol_z > 30 and math.isclose(k.vol_mean, 1000.0) and math.isclose(k.vol_std, 100.0)
    g = compute_breakout(bars("GGG"), cfg)
    assert g.direction == 1 and math.isnan(g.vol_z)                      # flat volume ⇒ no dispersion ⇒ NaN ⇒ rejected
    b = compute_breakout(bars("BBB"), cfg)
    assert b.direction == 0                                              # trend inside its own channel step
    assert compute_breakout(bars("KKK")[:15], cfg) is None


def test_oi_history():
    oi = OIHistory(keep_ms=3 * HOUR_MS)
    oi.record("X", 1_000_000, 100.0)
    oi.record("X", 1_060_000, 101.0)
    oi.record("X", 1_000_000, 999.0)                 # out of order → ignored
    assert oi.at("X", 1_010_000, 60_000) == 100.0
    assert oi.at("X", 1_059_000, 60_000) == 101.0
    assert oi.at("X", 2_000_000, 60_000) is None
    assert oi.at("Y", 1_000_000, 60_000) is None
    oi.record("X", 1_000_000 + 4 * HOUR_MS, 105.0)   # old samples pruned
    assert oi.at("X", 1_000_000, 60_000) is None
    oi.load([(5, "Z", 1.0), (1, "Z", 0.5)])
    assert oi.at("Z", 1, 0) == 0.5 and oi.at("Z", 5, 0) == 1.0


# ───────────────────────────── signal engine ─────────────────────────────


def make_engine(**over):
    oi = OIHistory()
    return ConvexSignalEngine(BreakoutConfig(**over), oi), oi


def test_engine_accepts_full_confluence():
    t_ms = BAR + HOUR_MS + 90_000                   # 90 s after the breakout bar closed
    eng, oi = make_engine()
    oi.record("KKK", BAR + 30_000, 1000.0)          # OI at bar open
    row = row_for("KKK", t_ms, openInterest="1040")
    sigs = eng.evaluate([row], {("KKK", "1h"): bars("KKK")}, t_ms)
    assert len(sigs) == 1 and not eng.rejections
    s = sigs[0]
    assert s.side == "long" and s.direction == 1 and s.bar_time == BAR
    assert math.isclose(s.oi_change, 0.04) and s.vol_z > 30
    assert s.squeeze_run >= 3 and s.bars_since_squeeze == 1
    assert s.stop_px == min(s.low, s.ref_px - s.atr) and s.stop_px < s.ref_px
    # same bar is never evaluated twice
    assert eng.evaluate([row], {("KKK", "1h"): bars("KKK")}, t_ms + 60_000) == []


def test_engine_rejections():
    t_ms = BAR + HOUR_MS + 90_000
    eng, oi = make_engine()
    oi.record("KKK", BAR + 30_000, 1000.0)
    oi.record("GGG", BAR + 30_000, 1000.0)
    rows = [row_for("KKK", t_ms, openInterest="1020"), row_for("GGG", t_ms, openInterest="1100")]
    sigs = eng.evaluate(rows, {("KKK", "1h"): bars("KKK"), ("GGG", "1h"): bars("GGG")}, t_ms)
    assert sigs == []
    rej = {r.coin: r.reasons for r in eng.rejections}
    assert any(x.startswith("oi_change=+2.00%") for x in rej["KKK"])
    assert any(x.startswith("vol_z=nan") for x in rej["GGG"])

    # no OI history → insufficient
    eng2, _ = make_engine()
    eng2.evaluate([row_for("KKK", t_ms, openInterest="1040")], {("KKK", "1h"): bars("KKK")}, t_ms)
    assert eng2.rejections[0].reasons == ("oi_history_insufficient",)

    # stale evaluation (restart 20 min after the close)
    eng3, oi3 = make_engine()
    oi3.record("KKK", BAR + 30_000, 1000.0)
    late = BAR + HOUR_MS + 20 * 60_000
    eng3.evaluate([row_for("KKK", late, openInterest="1040")], {("KKK", "1h"): bars("KKK")}, late)
    assert any(x.startswith("stale(") for x in eng3.rejections[0].reasons)

    # squeeze precondition off → GGG only fails volume; shorts disabled flag
    eng4, oi4 = make_engine(require_squeeze=False, allow_shorts=False)
    oi4.record("KKK", BAR + 30_000, 1000.0)
    sigs4 = eng4.evaluate([row_for("KKK", t_ms, openInterest="1040")], {("KKK", "1h"): bars("KKK")}, t_ms)
    assert len(sigs4) == 1


# ───────────────────────────── execution ─────────────────────────────


def test_parse_order_response():
    ok = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": "5.4", "avgPx": "110.02", "oid": 77}}]}}}
    assert parse_order_response(ok) == ("filled", 5.4, 110.02, 77, None)
    rest = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 78}}]}}}
    assert parse_order_response(rest)[0] == "resting" and parse_order_response(rest)[3] == 78
    nomatch = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Order could not immediately match against any resting orders."}]}}}
    assert parse_order_response(nomatch)[0] == "unfilled"
    bad = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Insufficient margin to place order."}]}}}
    assert parse_order_response(bad)[0] == "error" and "margin" in parse_order_response(bad)[4]
    assert parse_order_response({"status": "err", "response": "boom"})[0] == "error"
    assert parse_order_response("garbage")[0] == "error"


class FakeInfo:
    def __init__(self, mids, positions=None, equity="5000.0"):
        self.mids, self.positions, self.equity = mids, positions or [], equity

    async def all_mids(self):
        return {k: float(v) for k, v in self.mids.items()}     # the real client returns floats

    async def user_state(self, address):
        return {"marginSummary": {"accountValue": self.equity},
                "assetPositions": [{"type": "oneWay", "position": {"coin": c, "szi": str(s)}} for c, s in self.positions]}


class FakeGateway:
    def __init__(self, fill_frac=1.0, fail_entry=None, fail_stop=False):
        self.calls, self.fill_frac, self.fail_entry, self.fail_stop = [], fill_frac, fail_entry, fail_stop
        self.oid = 100

    def update_leverage(self, leverage, coin, is_cross=True):
        self.calls.append(("lev", coin, leverage, is_cross))
        return {"status": "ok", "response": {"type": "default"}}

    def order(self, coin, is_buy, sz, limit_px, order_type, reduce_only=False):
        self.calls.append(("order", coin, is_buy, sz, limit_px, order_type, reduce_only))
        self.oid += 1
        if "trigger" in order_type:
            if self.fail_stop:
                return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Reduce only order would increase position."}]}}}
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": self.oid}}]}}}
        if self.fail_entry:
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": self.fail_entry}]}}}
        filled = round(sz * self.fill_frac, 1)
        avg = limit_px - 0.01 if is_buy else limit_px + 0.01
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": str(filled), "avgPx": str(avg), "oid": self.oid}}]}}}


def signal(coin="KKK", direction=1, ref=110.0, stop=None, t_ms=None):
    t_ms = t_ms or (BAR + HOUR_MS + 90_000)
    stop = stop if stop is not None else (ref - 5.0 if direction > 0 else ref + 5.0)
    return ConvexSignal(ts=t_ms, coin=coin, side="long" if direction > 0 else "short", direction=direction, bar_time=BAR,
                        ref_px=ref, close=ref, high=ref + 1, low=ref - 10.5, donchian_upper=101.0, donchian_lower=85.0,
                        vol_z=40.0, oi_open=1000.0, oi_close=1040.0, oi_change=0.04, squeeze_run=20,
                        bars_since_squeeze=1, atr=2.5, stop_px=stop)


def run(coro):
    return asyncio.run(coro)


def test_executor_live_fill_and_stop(tmp_path):
    gw = FakeGateway()
    ex = ConvexExecutor(ExecConfig(live=True), AccountConfig(equity_usd=5000, risk_pct=0.01), FakeInfo({"KKK": "110.0"}),
                        gateway=gw, account_address="0xabc")
    rows = {"KKK": row_for("KKK", 0)}
    s = signal()
    reps = run(ex.handle([s], rows, s.ts))
    assert len(reps) == 1
    r = reps[0]
    # sizing: risk $50 over a $5 stop distance → 10 coins → $1,100 notional, under both caps (25k, 7.5k)
    assert r.status == "filled" and r.sz == 10.0 and r.notional == 1100.0
    assert r.bound_px == round_px(110.0 * 1.0008, 1, "down") == 110.08
    assert r.filled_sz == 10.0 and r.avg_px == 110.07 and 6.0 < r.slippage_bps < 7.0
    assert r.oid == 101 and r.stop_oid == 102 and r.error is None
    kinds = [c[0] for c in gw.calls]
    assert kinds == ["lev", "order", "order"]
    assert gw.calls[0] == ("lev", "KKK", 5, True)
    entry = gw.calls[1]
    assert entry[1:] == ("KKK", True, 10.0, 110.08, IOC, False)
    stop = gw.calls[2]
    assert stop[1] == "KKK" and stop[2] is False and stop[3] == 10.0 and stop[6] is True
    assert stop[5] == {"trigger": {"triggerPx": 105.0, "isMarket": True, "tpsl": "sl"}}
    assert stop[4] == round_px(105.0 * 0.95, 1, "down")
    # same coin again → cooldown; another coin while 'in position' per the account → in_position
    ex.info = FakeInfo({"KKK": "110.0", "AAA": "100.3"}, positions=[("AAA", 3.0)])
    reps2 = run(ex.handle([signal(), signal(coin="AAA", ref=100.3, stop=97.0)], {**rows, "AAA": row_for("AAA", 0)}, s.ts + 60_000))
    assert [r.status for r in reps2] == ["skipped", "skipped"]
    assert "cooldown" in reps2[0].error and reps2[1].error == "in_position"
    assert len(gw.calls) == 3


def test_executor_partial_unfilled_and_stop_failure():
    rows = {"KKK": row_for("KKK", 0)}
    gw = FakeGateway(fill_frac=0.5)
    ex = ConvexExecutor(ExecConfig(live=True), AccountConfig(equity_usd=5000), FakeInfo({"KKK": "110.0"}), gateway=gw, account_address="0xabc")
    r = run(ex.handle([signal()], rows, signal().ts))[0]
    assert r.status == "partial" and r.filled_sz == 5.0 and gw.calls[-1][3] == 5.0     # stop sized to the fill

    gw = FakeGateway(fail_entry="Order could not immediately match against any resting orders.")
    ex = ConvexExecutor(ExecConfig(live=True), AccountConfig(equity_usd=5000), FakeInfo({"KKK": "110.0"}), gateway=gw, account_address="0xabc")
    r = run(ex.handle([signal()], rows, signal().ts))[0]
    assert r.status == "unfilled" and r.filled_sz == 0 and len([c for c in gw.calls if c[0] == "order"]) == 1
    assert ex._cooldown["KKK"] == signal().ts + HOUR_MS                                 # short cooldown, no chase

    gw = FakeGateway(fail_stop=True)
    ex = ConvexExecutor(ExecConfig(live=True), AccountConfig(equity_usd=5000), FakeInfo({"KKK": "110.0"}), gateway=gw, account_address="0xabc")
    r = run(ex.handle([signal()], rows, signal().ts))[0]
    assert r.status == "filled" and r.stop_oid is None and "UNPROTECTED" in r.error


def test_executor_guards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = {"KKK": row_for("KKK", 0), "GGG": row_for("GGG", 0), "AAA": row_for("AAA", 0)}
    # dry mode: nothing is sent, but the plan is complete and positions/cooldowns are simulated
    ex = ConvexExecutor(ExecConfig(live=False, max_positions=1), AccountConfig(equity_usd=1000, risk_pct=0.01), FakeInfo({"KKK": "110.0", "GGG": "110.0"}))
    reps = run(ex.handle([signal(), signal(coin="GGG")], rows, signal().ts))
    assert [r.status for r in reps] == ["dry", "skipped"] and "max_positions" in reps[1].error
    assert reps[0].mode == "dry" and reps[0].sz == 2.0 and reps[0].notional == 220.0 and '"tif": "Ioc"' in reps[0].raw
    # short side: bound rounds up, stop above entry
    ex2 = ConvexExecutor(ExecConfig(live=False), AccountConfig(equity_usd=5000), FakeInfo({"KKK": "110.0"}))
    plan = ex2.plan(signal(direction=-1, ref=110.0, stop=115.0), rows["KKK"].ctx, 5000.0, 110.0)
    assert plan.is_buy is False and plan.bound_px == round_px(110.0 * 0.9992, 1, "up") == 109.92
    assert plan.stop_px == 115.0 and plan.stop_limit_px == round_px(115.0 * 1.05, 1, "up")
    # price already back inside the range → skip; too small → skip
    r = run(ex2.handle([signal()], {"KKK": rows["KKK"]}, signal().ts))
    assert r[0].status == "dry"
    ex3 = ConvexExecutor(ExecConfig(live=False), AccountConfig(equity_usd=5000), FakeInfo({"KKK": "100.5"}))
    r = run(ex3.handle([signal()], {"KKK": rows["KKK"]}, signal().ts))
    assert r[0].status == "skipped" and "inside the range" in r[0].error
    ex4 = ConvexExecutor(ExecConfig(live=False), AccountConfig(equity_usd=50, risk_pct=0.001), FakeInfo({"KKK": "110.0"}))
    r = run(ex4.handle([signal()], {"KKK": rows["KKK"]}, signal().ts))
    assert r[0].status == "skipped" and r[0].error == "too_small"
    # kill switch
    open(os.path.join(tmp_path, "STOP"), "w").close()
    r = run(ex2.handle([signal(coin="AAA", ref=100.3, stop=97.0)], {"AAA": rows["AAA"]}, signal().ts))
    assert r[0].status == "skipped" and r[0].error == "kill_switch"
    with pytest.raises(ValueError):
        ConvexExecutor(ExecConfig(live=True), AccountConfig(), FakeInfo({}))
