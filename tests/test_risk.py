import asyncio
import math

from fakes import FakeGateway, FakeInfo
from portfolio import PositionBook
from risk_manager import (
    BreakerState, CircuitBreaker, ConvexRiskManager, Position, RiskConfig, TradeStats, stats_from_r_multiples,
)
from squeeze_scanner import HOUR_MS, AccountConfig, Candle, Store

T0 = 1_760_000_000_000 - (1_760_000_000_000 % HOUR_MS)   # an exact hour


def bar(i: int, h: float, l: float, c: float) -> Candle:
    t = T0 + i * HOUR_MS
    return Candle(t, t + HOUR_MS - 1, c, h, l, c, 1000.0, 10)


def run(coro):
    return asyncio.run(coro)


# ───────────────────────────── sizing ─────────────────────────────


def test_kelly_fraction_and_bootstrap():
    rm = ConvexRiskManager()
    frac, note = rm.risk_fraction(TradeStats())
    assert frac == 0.01 and note.startswith("bootstrap")
    frac, note = rm.risk_fraction(TradeStats(n=19, wins=10, avg_win_R=4.0, avg_loss_R=1.0))
    assert frac == 0.01                                            # still bootstrap below 20 trades
    # the target profile, 100 trades: p_lb = 0.35 − 0.0477, f* = 0.147, quarter = 3.68 %
    f = rm.kelly_fraction(TradeStats(n=100, wins=35, avg_win_R=4.5, avg_loss_R=1.0))
    assert math.isclose(f, 0.25 * (0.30230 - 0.69770 / 4.5), rel_tol=1e-3) and 0.036 < f < 0.037
    # 1,000 trades: quarter-Kelly 4.68 % → capped at 4 %
    assert rm.kelly_fraction(TradeStats(n=1000, wins=350, avg_win_R=4.5, avg_loss_R=1.0)) == 0.04
    # no edge → zero, and risk_fraction says so
    frac, note = rm.risk_fraction(TradeStats(n=100, wins=30, avg_win_R=2.0, avg_loss_R=1.0))
    assert frac == 0.0 and note.startswith("no edge")
    # only winners so far → payoff ∞ → f* = p_lb → capped; only losers → 0
    assert rm.kelly_fraction(TradeStats(n=25, wins=25, avg_win_R=2.0, avg_loss_R=0.0)) == 0.04
    assert rm.kelly_fraction(TradeStats(n=25, wins=0, avg_win_R=0.0, avg_loss_R=1.0)) == 0.0
    s = stats_from_r_multiples([2.0, -1.0, -1.0, 4.0, -0.5])
    assert s.n == 5 and s.wins == 2 and s.avg_win_R == 3.0 and math.isclose(s.avg_loss_R, 2.5 / 3) and s.win_rate == 0.4


def test_size_and_initial_stop():
    rm = ConvexRiskManager()
    assert rm.initial_stop(100.0, 1, 2.0) == 97.0 and rm.initial_stop(100.0, -1, 2.0) == 103.0
    d = rm.size(equity=1000.0, entry_px=100.0, atr=2.0, stats=TradeStats(), sz_decimals=1, max_notional=5000.0)
    assert d.size == 3.3 and d.notional == 330.0 and d.risk_usd == 9.9 and math.isclose(d.risk_frac, 0.0099)
    capped = rm.size(1000.0, 100.0, 2.0, TradeStats(), 1, max_notional=200.0)
    assert capped.size == 2.0 and "capped" in capped.note and capped.risk_usd == 6.0
    tiny = rm.size(50.0, 100.0, 2.0, TradeStats(), 1, 5000.0)                            # $0.50 risk → 0.16 → 0.1 coin = $10, the HL minimum
    assert tiny.size == 0.1 and tiny.notional == 10.0
    assert rm.size(20.0, 100.0, 2.0, TradeStats(), 1, 5000.0) is None                    # 0.2/3 = 0.06 → 0.0 → None
    assert rm.size(1000.0, 100.0, 2.0, TradeStats(n=100, wins=30, avg_win_R=2.0, avg_loss_R=1.0), 1, 5000.0) is None  # no edge


# ───────────────────────────── exits ─────────────────────────────


def test_ratchet_long_and_short():
    rm = ConvexRiskManager()
    p = Position("X", "long", 1, 100.0, 1.0, 97.0, 97.0, 3.0, 100.0)
    assert rm.desired_stop(p, 2.0) == (97.0, 0)
    rm.update_excursion(p, 105.9, 99.0)                      # +1.97R: nothing
    assert rm.desired_stop(p, 2.0) == (97.0, 0)
    rm.update_excursion(p, 106.0, 99.0)                      # +2.0R: break-even + 0.1R
    p.stop, p.stage = rm.desired_stop(p, 2.0)
    assert (p.stop, p.stage) == (100.3, 1)
    rm.update_excursion(p, 109.0, 101.0)                     # +3.0R: chandelier 109 − 2.5×2 = 104
    p.stop, p.stage = rm.desired_stop(p, 2.0)
    assert (p.stop, p.stage) == (104.0, 2)
    p.stop, p.stage = rm.desired_stop(p, 4.0)                # ATR doubles → 109 − 10 = 99 < 104: never lowers
    assert (p.stop, p.stage) == (104.0, 2)
    rm.update_excursion(p, 112.0, 103.0)
    p.stop, p.stage = rm.desired_stop(p, 2.0)
    assert (p.stop, p.stage) == (107.0, 2)
    assert rm.stop_hit(p, 110.0, 107.0) and not rm.stop_hit(p, 110.0, 107.01)
    assert rm.paper_exit_px(107.0, 1) == 107.0 * 0.9995 and rm.paper_exit_px(107.0, -1) == 107.0 * 1.0005
    assert rm.stop_moved_enough(104.0, 107.0) and not rm.stop_moved_enough(104.0, 104.05)

    s = Position("Y", "short", -1, 100.0, 1.0, 103.0, 103.0, 3.0, 100.0)
    rm.update_excursion(s, 101.0, 94.0)                      # +2R → 99.7
    s.stop, s.stage = rm.desired_stop(s, 2.0)
    assert (s.stop, s.stage) == (99.7, 1)
    rm.update_excursion(s, 101.0, 90.0)                      # +3.33R → 90 + 5 = 95
    s.stop, s.stage = rm.desired_stop(s, 2.0)
    assert (s.stop, s.stage) == (95.0, 2)
    assert rm.stop_hit(s, 95.0, 92.0) and not rm.stop_hit(s, 94.99, 92.0)


# ───────────────────────────── breaker ─────────────────────────────


def test_circuit_breaker():
    cfg = RiskConfig()
    cb = CircuitBreaker(cfg)
    t = T0
    assert cb.sample(t, 1000.0) is None
    assert cb.sample(t + HOUR_MS, 990.0) is None
    assert cb.sample(t + 2 * HOUR_MS, 950.0) is None and math.isclose(cb.drawdown(950.0), 0.05)
    trip = cb.sample(t + 3 * HOUR_MS, 940.0)
    assert trip and math.isclose(trip["drawdown"], 0.06) and trip["peak"] == 1000.0
    assert trip["halt_until_ms"] == t + 3 * HOUR_MS + 12 * HOUR_MS and cb.state.trips == 1
    assert cb.halted(t + 10 * HOUR_MS) and not cb.halted(t + 15 * HOUR_MS + 1)
    assert cb.sample(t + 10 * HOUR_MS, 500.0) is None            # no second trip while halted
    assert cb.sample(t + 16 * HOUR_MS, 940.0) is None            # window restarts at the resume equity …
    assert cb.sample(t + 17 * HOUR_MS, 890.0) is None            # … so −5.3 % from 940 does not trip
    assert cb.sample(t + 18 * HOUR_MS, 880.0)                    # −6.4 % does
    # rolling window: a peak older than 24 h rolls off
    cb2 = CircuitBreaker(cfg)
    cb2.sample(t, 1000.0)
    cb2.sample(t + 25 * HOUR_MS, 950.0)
    assert cb2.sample(t + 26 * HOUR_MS, 945.0) is None and cb2.peak() == 950.0
    # restored state keeps the halt
    cb3 = CircuitBreaker(cfg, BreakerState(halt_until_ms=t + HOUR_MS, tripped_ms=t, trips=1))
    assert cb3.halted(t + 1) and not cb3.halted(t + HOUR_MS)


# ───────────────────────────── paper book ─────────────────────────────


def test_paper_book_lifecycle(tmp_path):
    store = Store(str(tmp_path / "book.db"))
    rm = ConvexRiskManager()
    info = FakeInfo({"X": "100.0"})
    book = PositionBook(rm, AccountConfig(equity_usd=1000.0), info, store, live=False)
    t_ms = T0 + 5 * 60_000
    pos = book.register("X", 1, 100.0, 3.3, 2.0, t_ms, T0, 1)
    assert pos.stop == 97.0 and pos.r_unit == 3.0 and pos.risk_usd == 9.9 and book.positions == {"X": pos}
    assert store.open_positions("dry")[0].coin == "X"

    bars = [bar(1, 102.0, 99.0, 101.0), bar(2, 106.5, 101.0, 106.0), bar(3, 110.0, 104.0, 109.0), bar(4, 106.0, 104.0, 105.0)]
    stages = []

    async def go():
        out = []
        for k in range(1, 5):
            marks = {"X": bars[k - 1].c}
            out.append(await book.on_cycle({("X", "1h"): bars[:k]}, T0 + (k + 1) * HOUR_MS + 60_000, marks))
            stages.append((pos.stage, pos.stop))
        return out

    s1, s2, s3, s4 = run(go())
    # bar1 +0.67R: stage 0, stop 97 · bar2 +2.17R: break-even 100.3 · bar3 +3.33R: chandelier 110 − 2.5×2 = 105
    # bar4 low 104 ≤ 105 (tested BEFORE its own high is credited) → exit at 105 − 5 bps
    assert stages[:3] == [(0, 97.0), (1, 100.3), (2, 105.0)]
    assert s1["open"] == 1 and s1["closed"] == [] and s2["open"] == 1 and s3["open"] == 1
    assert s4["open"] == 0 and len(s4["closed"]) == 1
    trade = s4["closed"][0]
    assert trade["coin"] == "X" and trade["stage"] == 2 and trade["reason"].startswith("stop stage 2")
    assert math.isclose(trade["exit_px"], 105.0 * 0.9995)
    gross = (trade["exit_px"] - 100.0) * 3.3
    fees = 3.3 * 100.0 * 0.00045 + 3.3 * trade["exit_px"] * 0.00045
    assert math.isclose(trade["pnl_usd"], gross - fees, abs_tol=1e-3) and math.isclose(trade["r_multiple"], (gross - fees) / (3.3 * 3.0), abs_tol=1e-3)
    assert 1.5 < trade["r_multiple"] < 1.7
    assert book.stats().n == 1 and book.stats().wins == 1 and store.trade_r_multiples("dry") == [trade["r_multiple"]]
    assert math.isclose(s4["equity"], 1000.0 + trade["pnl_usd"], abs_tol=1e-3)
    assert store.open_positions("dry") == [] and store.realized_pnl("dry") == trade["pnl_usd"]

    # intra-bar exit on the mark, and a restart that restores the open position
    pos2 = book.register("Y", 1, 50.0, 10.0, 1.0, T0 + 6 * HOUR_MS, T0 + 5 * HOUR_MS, 1)
    book2 = PositionBook(rm, AccountConfig(equity_usd=1000.0), info, store, live=False)
    assert "Y" in book2.positions and book2.positions["Y"].stop == 48.5 and book2.stats().n == 1
    s = run(book2.on_cycle({}, T0 + 6 * HOUR_MS + 60_000, {"Y": 48.0}))
    assert s["closed"][0]["reason"] == "stop stage 0 (mark)" and math.isclose(s["closed"][0]["exit_px"], 48.0 * 0.9995)
    assert s["closed"][0]["r_multiple"] < -1.0                       # gap through the stop costs more than 1R
    assert book2.positions == {} and store.open_positions("dry") == [] and pos2.coin == "Y"
    store.close()


def test_paper_book_breaker_flattens_and_halts(tmp_path):
    store = Store(str(tmp_path / "book.db"))
    info = FakeInfo({"X": "100.0", "Z": "10.0"})
    book = PositionBook(ConvexRiskManager(), AccountConfig(equity_usd=1000.0), info, store, live=False)
    t = T0
    book.register("X", 1, 100.0, 10.0, 10.0, t, T0 - HOUR_MS, 1)          # wide stop (85) so the breaker, not the stop, acts
    book.register("Z", -1, 10.0, 20.0, 2.0, t, T0 - HOUR_MS, 1)
    s0 = run(book.on_cycle({}, t + 60_000, {"X": 100.0, "Z": 10.0}))
    assert s0["open"] == 2 and not s0["tripped"] and math.isclose(s0["equity"], 1000.0 - 0.45 - 0.09)
    s1 = run(book.on_cycle({}, t + 120_000, {"X": 93.0, "Z": 10.0}))     # −$70 open = −7.0 % → trip
    assert s1["tripped"] and s1["halted"] and s1["open"] == 0 and len(s1["closed"]) == 2
    assert {tr["reason"] for tr in s1["closed"]} == {"breaker"}
    ok, why = book.can_enter(t + 3 * HOUR_MS)
    assert not ok and why.startswith("breaker halt until")
    assert book.can_enter(t + 120_000 + 12 * HOUR_MS)[0]
    assert store.breaker_state().halt_until_ms == t + 120_000 + 12 * HOUR_MS
    # a restarted book honours the halt
    book2 = PositionBook(ConvexRiskManager(), AccountConfig(equity_usd=1000.0), info, store, live=False)
    assert not book2.can_enter(t + 3 * HOUR_MS)[0]
    store.close()


# ───────────────────────────── live book ─────────────────────────────


def test_live_book_replaces_stop_and_closes_from_fills():
    gw = FakeGateway()
    info = FakeInfo({"X": "100.0"}, positions=[("X", 3.3)], equity="1000.0")
    book = PositionBook(ConvexRiskManager(), AccountConfig(equity_usd=1000.0), info, None, gateway=gw,
                        account_address="0xabc", live=True)
    pos = book.register("X", 1, 100.0, 3.3, 2.0, T0 + 60_000, T0, 1, stop_oid=500)
    bars = [bar(1, 102.0, 99.0, 101.0), bar(2, 106.5, 101.0, 106.0)]
    s1 = run(book.on_cycle({("X", "1h"): bars[:1]}, T0 + 2 * HOUR_MS + 60_000, {"X": 101.0}))
    assert s1["stops_moved"] == 0 and pos.stop == 97.0 and gw.calls == []
    s2 = run(book.on_cycle({("X", "1h"): bars}, T0 + 3 * HOUR_MS + 60_000, {"X": 106.0}))
    # +2.17R → break-even stop 100.3: new reduce-only stop placed FIRST, then the old one cancelled
    assert s2["stops_moved"] == 1 and pos.stop == 100.3 and pos.stage == 1 and pos.stop_oid == 101
    kinds = [c[0] for c in gw.calls]
    assert kinds == ["order", "cancel"]
    o = gw.calls[0]
    assert o[1:4] == ("X", False, 3.3) and o[5] == {"trigger": {"triggerPx": 100.3, "isMarket": True, "tpsl": "sl"}} and o[6] is True
    assert o[4] == 95.285                                             # worst fill once triggered: 5 % through (5 sig figs allowed)
    assert gw.calls[1] == ("cancel", "X", 500)
    # the exchange shows the position gone → closed from the fills, stop cancelled, trade recorded
    info.positions = []
    info.fills = [{"coin": "X", "px": "101.4", "sz": "3.3", "side": "A", "time": T0 + 4 * HOUR_MS, "fee": "0.15"}]
    s3 = run(book.on_cycle({("X", "1h"): bars}, T0 + 4 * HOUR_MS + 60_000, {"X": 101.0}))
    assert s3["open"] == 0 and len(s3["closed"]) == 1
    tr = s3["closed"][0]
    assert tr["exit_px"] == 101.4 and tr["reason"].endswith("(fills)") and gw.calls[-1] == ("cancel", "X", 101)
    # entry fee from the model, exit fee from the actual fill ($0.15)
    assert math.isclose(tr["pnl_usd"], (101.4 - 100.0) * 3.3 - (3.3 * 100 * 0.00045 + 0.15), abs_tol=1e-3)
    assert s3["equity"] == 1000.0                                   # live equity comes from the account


def test_live_book_breaker_flattens_with_ioc_and_cancels_orders():
    gw = FakeGateway()
    info = FakeInfo({"X": "90.0"}, positions=[("X", 10.0)], equity="940.0",
                    open_orders=[{"coin": "X", "oid": 500}, {"coin": "Q", "oid": 7}])
    book = PositionBook(ConvexRiskManager(), AccountConfig(equity_usd=1000.0), info, None, gateway=gw,
                        account_address="0xabc", live=True)
    book.register("X", 1, 100.0, 10.0, 2.0, T0, T0 - HOUR_MS, 1, stop_oid=500)
    info.equity = "1000.0"
    run(book.on_cycle({}, T0 + 60_000, {"X": 100.0}))
    info.equity = "930.0"
    s = run(book.on_cycle({}, T0 + 120_000, {"X": 93.0}))
    assert s["tripped"] and s["open"] == 0
    kinds = [(c[0], c[1]) for c in gw.calls]
    # flatten: IOC reduce-only sell at ≤ 93 × 0.99, then cancel the stop, then every remaining open order
    assert kinds[0] == ("order", "X") and gw.calls[0][2] is False and gw.calls[0][6] is True and gw.calls[0][4] == 92.07
    assert ("cancel", "X") in kinds and ("cancel", "Q") in kinds
    assert s["closed"][0]["reason"].startswith("breaker flatten") and s["closed"][0]["exit_px"] == 92.08
    assert not book.can_enter(T0 + 3 * HOUR_MS)[0]
