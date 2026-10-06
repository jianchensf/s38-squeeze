import asyncio
import os
import time

import aiohttp
from aiohttp.test_utils import TestServer

from squeeze_scanner import (
    ARMED, COOLDOWN, IDLE, AssetCtx, CandleCache, FundingTracker, HLInfoClient, Scanner, StateEngine, Store,
    WeightLimiter, build_configs, filter_universe, last_closed_open_time, main, parse_args,
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
    assert [a.coin for a in eligible] == ["AAA", "BBB", "GGG"]
    assert reasons == {"BTC": "excluded", "ETH": "excluded", "SOL": "excluded", "CCC": "vol_low", "DDD": "vol_high",
                       "EEE": "delisted", "FFF": "spread_wide", "HHH": "no_mid"}
    aaa = next(a for a in assets if a.coin == "AAA")
    assert abs(aaa.impact_spread_bps - 9.97) < 0.1
    assert abs(aaa.day_change - (100.3 / 95 - 1)) < 1e-9


def test_weight_limiter_paces():
    async def go():
        lim = WeightLimiter(6000)        # 100 weight/s
        await lim.acquire(6000)          # drain the bucket
        t0 = time.monotonic()
        await lim.acquire(20)            # needs 0.2s of refill
        return time.monotonic() - t0
    waited = asyncio.run(go())
    assert 0.15 <= waited < 1.5


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
                scanner = Scanner(client, cache, FundingTracker(client, store), engine, store, ucfg, scfg, fcfg, quiet=True)
                rows1 = await scanner.cycle()
                calls1 = dict(calls)
                rows2 = await scanner.cycle()
                calls2 = dict(calls)
            restored = StateEngine(ecfg, acct, [], restore_fires=store.last_fire_bars())
            n_scans = store.con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            n_fund = store.con.execute("SELECT COUNT(*) FROM funding_hourly WHERE coin='AAA'").fetchone()[0]
            fires = store.con.execute("SELECT coin, direction, intent FROM events WHERE kind='FIRE'").fetchall()
            store.close()
            return rows1, rows2, calls1, calls2, cap.events, engine, cache, restored, n_scans, n_fund, fires

    rows1, rows2, calls1, calls2, events, engine, cache, restored, n_scans, n_fund, fires = asyncio.run(go())

    # universe → rows
    assert {r.ctx.coin for r in rows1} == {"AAA", "BBB", "GGG"}
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
    fire = next(e for e in events if e.kind == "FIRE")
    assert fire.direction == 1 and fire.intent is not None
    i = fire.intent
    assert i.side == "long" and i.entry_px == 110.0 and i.stop_px < i.entry_px
    assert abs(i.size * 10 - round(i.size * 10)) < 1e-9        # floored to szDecimals=1
    assert i.notional_usd <= 5000 * 5 and i.notional_usd <= 20e6 * 0.0005
    assert i.risk_usd <= 50.0 + 1e-6 and i.leverage == round(i.notional_usd / 5000, 2)

    # cache: the open bar is dropped, and nothing was re-fetched on the second cycle
    bars = cache._bars[("GGG", "1h")]
    assert bars[-1].t == last_closed_open_time("1h", rows1[0].ts)
    assert calls1["candleSnapshot"] == 6 and calls2["candleSnapshot"] == 6
    assert calls2["metaAndAssetCtxs"] == 2
    assert calls1["fundingHistory"] >= 1 and calls2["fundingHistory"] == calls1["fundingHistory"]

    # persistence + restore
    assert n_scans == 6 and n_fund >= 24
    assert len(fires) == 1 and fires[0][0] == "GGG" and fires[0][1] == 1 and '"side": "long"' in fires[0][2]
    assert restored.states["GGG"].state == COOLDOWN
    assert os.path.exists(db)

    # db viewers used by `make events` / `make scans`
    assert main(["--db", db, "--events", "10"]) == 0
    assert main(["--db", db, "--scans"]) == 0
    out = capsys.readouterr().out
    assert "FIRE" in out and "GGG" in out and "AAA" in out and "ARMED" in out
    assert main(["--db", str(tmp_path / "missing.db"), "--scans"]) == 1
