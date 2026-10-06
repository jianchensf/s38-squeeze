import asyncio
import math
import os
import time

import aiohttp
from aiohttp.test_utils import TestServer

from convex_engine import (BreakoutConfig, ConvexExecutor, ConvexRunner, ConvexSignalEngine, ExecConfig, OIHistory,
                           compute_breakout)
from portfolio import PositionBook
from risk_manager import ConvexRiskManager, RiskConfig
from squeeze_scanner import (
    ARMED, COOLDOWN, IDLE, AssetCtx, CandleCache, FundingTracker, HLInfoClient, Scanner, StateEngine, Store,
    WeightLimiter, build_configs, filter_universe, last_closed_open_time, main, now_ms, parse_args,
)
from synth import UNIVERSE, make_app


class Capture:
    def __init__(self):
        self.events = []

    async def on_event(self, ev):
        self.events.append(ev)


def test_universe_filter_reasons():
    assets = [AssetCtx.from_api(m, c) for m, c in UNIVERSE]
    ucfg = build_configs(parse_args(["--once"]))[0]
    eligible, reasons = filter_universe(assets, ucfg)
    assert [a.coin for a in eligible] == ["AAA", "BBB", "GGG", "KKK"]
    assert reasons == {"BTC": "excluded", "ETH": "excluded", "SOL": "excluded", "CCC": "vol_low", "DDD": "vol_high",
                       "EEE": "delisted", "FFF": "spread_wide", "HHH": "no_mid"}
    aaa = next(a for a in assets if a.coin == "AAA")
    assert abs(aaa.impact_spread_bps - 9.97) < 0.1
    assert abs(aaa.day_change - (100.3 / 95 - 1)) < 1e-9


def test_weight_limiter_paces_and_adapts():
    async def go():
        lim = WeightLimiter(6000)                   # 100 weight/s ceiling, burst of two calls, no more
        t0 = time.monotonic()
        await lim.acquire(20)
        await lim.acquire(20)
        burst = time.monotonic() - t0
        t1 = time.monotonic()
        await lim.acquire(20)                       # third call needs 0.2s of refill
        paced = time.monotonic() - t1
        new_budget = lim.penalize(debt_s=0.2)       # 429: rate halves to 50/s and 0.2s of debt is owed
        t2 = time.monotonic()
        await lim.acquire(20)                       # debt 0.2s + 20/50 = 0.6s
        after_429 = time.monotonic() - t2
        lim._last_429 -= 120                        # pretend: 2 min since the 429, 5 min since the last refill
        lim._updated -= 300
        lim._refill(time.monotonic())
        return burst, paced, new_budget, after_429, lim.rate, lim.n_429
    burst, paced, new_budget, after_429, rate, n = asyncio.run(go())
    assert burst < 0.05 and 0.15 <= paced < 1.0
    assert new_budget == 3000 and 0.5 <= after_429 < 2.0 and n == 1
    assert rate == 100.0                            # recovered to the ceiling: 50 + 100 × 300/600


def test_client_survives_429_and_backs_off():
    async def go():
        app, calls = make_app(fail_429_first=2)
        async with TestServer(app) as server:
            base = f"http://{server.host}:{server.port}"
            args = parse_args(["--once", "--db", "", "--base-url", base, "--weight", "600000", "--quiet"])
            ucfg, scfg, fcfg, ecfg, acct, ccfg = build_configs(args)
            ccfg = type(ccfg)(**{**ccfg.__dict__, "penalty_s": 0.1})
            async with aiohttp.ClientSession() as session:
                lim = WeightLimiter(ccfg.weight_per_minute)
                client = HLInfoClient(ccfg, session, lim)
                scanner = Scanner(client, CandleCache(client, scfg), FundingTracker(client), StateEngine(ecfg, acct, []),
                                  None, ucfg, scfg, fcfg, quiet=True)
                rows = await scanner.cycle()
                return rows, dict(calls), lim.n_429, lim.rate, lim.max_rate, client.consecutive_429
    rows, calls, n429, rate, max_rate, consecutive = asyncio.run(go())
    assert calls["429"] == 2 and n429 == 2 and consecutive == 0        # the counter resets on the first success
    assert calls["candleSnapshot"] == 8 + 2                 # every series still loaded, two retries
    assert {r.ctx.coin for r in rows} == {"AAA", "BBB", "GGG", "KKK"}
    assert rate == max_rate / 4                             # halved twice, no recovery yet


def test_end_to_end_pipeline(tmp_path, capsys):
    db = str(tmp_path / "s38.db")

    async def go():
        app, calls = make_app()
        async with TestServer(app) as server:
            base = f"http://{server.host}:{server.port}"
            args = parse_args(["--once", "--db", db, "--base-url", base, "--weight", "600000", "--quiet",
                               "--equity", "5000", "--risk-pct", "1"])
            ucfg, scfg, fcfg, ecfg, acct, ccfg = build_configs(args)
            store = Store(db)
            cap = Capture()
            async with aiohttp.ClientSession() as session:
                client = HLInfoClient(ccfg, session, WeightLimiter(ccfg.weight_per_minute))
                engine = StateEngine(ecfg, acct, [cap, store], fuel_tf=fcfg.fuel_tf)
                cache = CandleCache(client, scfg)
                # convex engine in dry mode, with the OI sample at the breakout bar's open pre-seeded (1000 → ctx 1040)
                oi = OIHistory()
                oi.record("KKK", last_closed_open_time("1h", now_ms()) + 30_000, 1000.0)
                oi.record("GGG", last_closed_open_time("1h", now_ms()) + 30_000, 900.0)
                # the synthetic bars are anchored to the wall clock, so relax the freshness guard here
                # (it is unit-tested in test_convex); everything else runs with production defaults
                risk = ConvexRiskManager(RiskConfig(bootstrap_risk_frac=acct.risk_pct))
                book = PositionBook(risk, acct, client, store, live=False)
                runner = ConvexRunner(ConvexSignalEngine(BreakoutConfig(max_signal_age_s=4000), oi, store),
                                      ConvexExecutor(ExecConfig(live=False), acct, client, store, risk=risk, book=book), book)
                scanner = Scanner(client, cache, FundingTracker(client, store), engine, store, ucfg, scfg, fcfg,
                                  quiet=True, hooks=[runner.on_cycle])
                scanner.extra_coins = runner.position_coins
                rows1 = await scanner.cycle()
                calls1 = dict(calls)
                rows2 = await scanner.cycle()
                calls2 = dict(calls)
            restored = StateEngine(ecfg, acct, [], restore_fires=store.last_fire_bars())
            n_scans = store.con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            n_fund = store.con.execute("SELECT COUNT(*) FROM funding_hourly WHERE coin='AAA'").fetchone()[0]
            fires = store.con.execute("SELECT coin, direction, intent FROM events WHERE kind='FIRE' AND coin='GGG'").fetchall()
            sigs = store.con.execute("SELECT coin, side, accepted, reasons, oi_change FROM signals ORDER BY coin").fetchall()
            orders = store.con.execute("SELECT coin, side, mode, status, sz, notional, stop_px FROM orders").fetchall()
            oi_rows = store.con.execute("SELECT COUNT(*) FROM scans WHERE oi IS NOT NULL").fetchone()[0]
            open_pos = store.open_positions("dry")
            store.close()
            return (rows1, rows2, calls1, calls2, cap.events, engine, cache, restored, n_scans, n_fund, fires,
                    sigs, orders, oi_rows, runner, open_pos)

    (rows1, rows2, calls1, calls2, events, engine, cache, restored, n_scans, n_fund, fires,
     sigs, orders, oi_rows, runner, open_pos) = asyncio.run(go())

    # universe → rows
    assert {r.ctx.coin for r in rows1} == {"AAA", "BBB", "GGG", "KKK"}
    assert rows1[0].ctx.coin in {"AAA", "GGG"}  # squeezed names rank above the trend name
    by = {r.ctx.coin: r for r in rows1}
    assert by["AAA"].squeeze["1h"].squeeze_on and by["AAA"].squeeze["4h"].squeeze_on
    assert by["AAA"].fuel.negative and by["AAA"].fuel.near_zero and by["AAA"].fuel.price_near_upper
    assert by["AAA"].fuel.short_squeeze_fuel and by["AAA"].fuel.sustained_negative
    assert by["AAA"].fuel.neg_hours_24h is not None and by["AAA"].fuel.neg_hours_24h >= 6
    assert not by["BBB"].squeeze["1h"].squeeze_on and not by["BBB"].fuel.short_squeeze_fuel
    assert by["GGG"].squeeze["1h"].released and by["GGG"].squeeze["1h"].release_dir == 1

    # state engine
    assert engine.states["AAA"].state == ARMED
    assert engine.states["BBB"].state == IDLE
    assert engine.states["GGG"].state == COOLDOWN
    kinds = [(e.coin, e.kind) for e in events]
    assert ("AAA", "ARM") in kinds and ("GGG", "FIRE") in kinds
    assert kinds.count(("GGG", "FIRE")) == 1          # second cycle must not re-fire the same bar
    fire = next(e for e in events if e.kind == "FIRE" and e.coin == "GGG")
    assert fire.direction == 1 and fire.intent is not None
    i = fire.intent
    assert i.side == "long" and i.entry_px == 110.0 and i.stop_px < i.entry_px
    assert abs(i.size * 10 - round(i.size * 10)) < 1e-9        # floored to szDecimals=1
    assert i.notional_usd <= 5000 * 5 and i.notional_usd <= 20e6 * 0.0005
    assert i.risk_usd <= 50.0 + 1e-6 and i.leverage == round(i.notional_usd / 5000, 2)

    # cache: the open bar is dropped, and nothing was re-fetched on the second cycle
    bars = cache._bars[("GGG", "1h")]
    assert bars[-1].t == last_closed_open_time("1h", rows1[0].ts)
    assert calls1["candleSnapshot"] == 8 and calls2["candleSnapshot"] == 8
    assert calls2["metaAndAssetCtxs"] == 2
    assert calls1["fundingHistory"] >= 1 and calls2["fundingHistory"] == calls1["fundingHistory"] + 1   # +1: KKK's open position

    # convex engine: KKK has squeeze + Donchian break + volume spike + OI +4% → one dry order;
    # GGG breaks out on flat volume → rejected; the same bars are not re-evaluated on cycle 2
    assert calls1["allMids"] == 1 and calls2["allMids"] == 2       # cycle 2: one mids call to mark the open paper position
    assert [(c, s, ok) for c, s, ok, _, _ in sigs] == [("GGG", "long", 0), ("KKK", "long", 1)]
    assert "vol_z=nan" in sigs[0][3] and abs(sigs[1][4] - 0.04) < 1e-9
    assert len(orders) == 1
    coin, side, mode, status, sz, notional, stop_px = orders[0]
    assert (coin, side, mode, status) == ("KKK", "long", "dry", "dry")
    atr = compute_breakout(cache._bars[("KKK", "1h")], BreakoutConfig()).atr
    assert sz == math.floor(5000 * 0.01 / (1.5 * atr) * 10) / 10 and abs(notional - sz * 110.0) < 1e-6
    assert abs(stop_px - round(110.08 - 1.5 * atr, 2)) < 0.011                          # 1.5 × ATR14 below the paper fill
    assert runner.executor._dry_positions == {"KKK": rows1[0].ts}
    # the paper position lives in the book and survives a restart via the store; cycle 2 kept it open
    assert list(runner.book.positions) == ["KKK"] and runner.position_coins() == {"KKK"}
    assert len(open_pos) == 1 and open_pos[0].coin == "KKK" and open_pos[0].entry_px == 110.08 and open_pos[0].stage == 0
    assert abs(open_pos[0].stop - (110.08 - 1.5 * atr)) < 1e-9 and open_pos[0].size == sz

    # persistence + restore
    assert n_scans == 8 and n_fund >= 24 and oi_rows == 8
    assert len(fires) == 1 and fires[0][0] == "GGG" and fires[0][1] == 1 and '"side": "long"' in fires[0][2]
    assert restored.states["GGG"].state == COOLDOWN
    assert os.path.exists(db)

    # db viewers used by `make events` / `make scans` / `make signals` / `make orders`
    assert main(["--db", db, "--events", "10"]) == 0
    assert main(["--db", db, "--scans"]) == 0
    assert main(["--db", db, "--signals", "--orders", "--positions", "--trades"]) == 0
    out = capsys.readouterr().out
    assert "FIRE" in out and "GGG" in out and "AAA" in out and "ARMED" in out
    assert "vol_z=nan" in out and "KKK" in out and " dry " in out and "STOP_OID" in out
    assert main(["--db", str(tmp_path / "missing.db"), "--scans"]) == 1


def test_429_backoff_doubles_per_consecutive_hit_and_honours_retry_after():
    from squeeze_scanner import backoff_429
    assert [backoff_429(n, 30, 300, rnd=0.5) for n in (1, 2, 3, 4, 5, 6)] == [30, 60, 120, 240, 300, 300]
    assert backoff_429(1, 30, 300, retry_after="45", rnd=0.5) == 45       # server asks for longer: obey
    assert backoff_429(2, 30, 300, retry_after="45", rnd=0.5) == 60       # server asks for less: keep our schedule
    assert backoff_429(1, 30, 300, retry_after="soon", rnd=0.5) == 30     # unparseable header ignored
    assert backoff_429(1, 30, 300, rnd=0.0) == 24 and backoff_429(1, 30, 300, rnd=1.0) == 36   # ±20 % jitter
