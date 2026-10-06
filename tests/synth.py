"""Deterministic synthetic candles + a fake Hyperliquid /info server for offline tests.

Regimes (all with h = c + 1, l = c − 1 unless noted):
  trend        sawtooth with slope 1/bar → close dispersion ≫ bar range → Bollinger OUTSIDE Keltner (no squeeze)
  squeeze_top  trend into 100, then 40 flat bars at 100 ± 0.3; one flush wick to 85 fifteen bars before the end
               → Bollinger inside Keltner (squeeze ON) with close in the top of the 20-bar range
  release_up   same flat base (wick 25 bars back), last closed bar breaks out: c = 110, h = 111
               → squeeze ON at bar[-2], OFF at bar[-1] ⇒ released, direction up (volume flat ⇒ vol_z NaN)
  convex_up    release_up with volume 900/1100 alternating and 5000 on the breakout bar ⇒ vol_z ≈ 40
"""
from __future__ import annotations

from collections import Counter

from aiohttp import web

from squeeze_scanner import TF_MS, last_closed_open_time

KINDS = {"AAA": "squeeze_top", "GGG": "release_up", "KKK": "convex_up"}  # everything else: trend


def closes_for(kind: str, n: int) -> list:
    out = []
    flat_start = n - 40
    for k in range(n):
        if kind == "trend" or k < flat_start:
            phase = (k - flat_start) % 80
            out.append(100.0 - min(phase, 80 - phase))          # triangle wave 60..100 ending at 100
        else:
            out.append(100.0 + (0.3 if k % 2 else -0.3))        # flat with ±0.3 alternation
    if kind in ("release_up", "convex_up"):
        out[-1] = 110.0
    return out


def volume_for(kind: str, j: int, n: int) -> str:
    if kind == "convex_up":
        return "5000" if j == n - 1 else ("900" if j % 2 else "1100")
    return "1000"


def candles_for(coin: str, interval: str, start_ms: int, end_ms: int, n_hist: int = 220) -> list:
    """Closed bars ending at the last closed bar for `end_ms`, plus one in-progress bar."""
    kind = KINDS.get(coin, "trend")
    ms = TF_MS[interval]
    last_open = last_closed_open_time(interval, end_ms)
    closes = closes_for(kind, n_hist)
    bars = []
    for j, c in enumerate(closes):
        t = last_open - (n_hist - 1 - j) * ms
        h, l = c + 1.0, c - 1.0
        back = n_hist - 1 - j
        if kind == "squeeze_top" and back == 15:
            l = 85.0                                            # flush wick inside the 20-bar range window
        if kind in ("release_up", "convex_up") and back == 25:
            l = 85.0                                            # wick outside the final 20-bar window
        if kind in ("release_up", "convex_up") and back == 0:
            h, l = 111.0, 99.5
        bars.append({"t": t, "T": t + ms - 1, "s": coin, "i": interval,
                     "o": str(bars[-1]["c"] if bars else c), "c": str(c), "h": str(h), "l": str(l),
                     "v": volume_for(kind, j, n_hist), "n": 50})
    nxt = last_open + ms
    bars.append({"t": nxt, "T": nxt + ms - 1, "s": coin, "i": interval, "o": str(closes[-1]), "c": str(closes[-1]),
                 "h": str(closes[-1] + 0.5), "l": str(closes[-1] - 0.5), "v": "10", "n": 3})   # open bar
    return [b for b in bars if b["t"] >= start_ms and b["t"] <= end_ms]


def meta(name: str, sz: int = 1, lev: int = 10, **kw) -> dict:
    return {"name": name, "szDecimals": sz, "maxLeverage": lev, **kw}


def ctx(vol: float, funding: str = "0.0000125", mark: str = "100.0", mid="100.0", imp=("99.98", "100.02"),
        prev: str = "95.0", oi: str = "1000") -> dict:
    return {"dayNtlVlm": str(vol), "funding": funding, "openInterest": oi, "prevDayPx": prev, "premium": "0.0001",
            "oraclePx": mark, "markPx": mark, "midPx": mid, "impactPxs": list(imp) if imp else None}


UNIVERSE = [
    (meta("BTC", sz=5, lev=40), ctx(5e9, mark="60000", mid="60000", imp=("59990", "60010"), prev="59000")),
    (meta("ETH", sz=4, lev=25), ctx(2e9, mark="3000", mid="3000", imp=("2999", "3001"), prev="2900")),
    (meta("SOL", sz=2, lev=20), ctx(30e6, mark="150", mid="150", imp=("149.9", "150.1"), prev="140")),
    (meta("AAA"), ctx(10e6, funding="-0.00002", mark="100.3", mid="100.3", imp=("100.25", "100.35"))),
    (meta("BBB"), ctx(5e6)),
    (meta("CCC"), ctx(1e6)),
    (meta("DDD"), ctx(50e6)),
    (meta("EEE", isDelisted=True), ctx(5e6)),
    (meta("FFF"), ctx(5e6, imp=("99", "101"))),
    (meta("GGG"), ctx(20e6, funding="0.00001", mark="110.0", mid="110.0", imp=("109.9", "110.1"))),
    (meta("HHH"), ctx(3e6, mid=None, imp=None)),
    (meta("KKK", sz=1, lev=10), ctx(15e6, funding="0.00001", mark="110.0", mid="110.0", imp=("109.9", "110.1"), oi="1040")),
]


def funding_history(coin: str, start_ms: int, end_ms: int) -> list:
    hour = (end_ms // 3_600_000) * 3_600_000
    rate = "-0.00002" if coin == "AAA" else "0.0000125"
    return [{"coin": coin, "fundingRate": rate, "premium": "0.0", "time": t}
            for t in range(hour - 24 * 3_600_000, hour + 1, 3_600_000) if t >= start_ms]


def make_app(fail_429_first: int = 0) -> tuple:
    """Return (app, calls) — `calls` counts /info requests by type. The first `fail_429_first` candle requests get 429."""
    calls: Counter = Counter()

    async def info(request: web.Request) -> web.Response:
        import time
        body = await request.json()
        kind = body.get("type")
        calls[kind] += 1
        now = int(time.time() * 1000)
        if kind == "candleSnapshot" and calls["candleSnapshot"] <= fail_429_first:
            calls["429"] += 1
            return web.Response(status=429, text="rate limited")
        if kind == "metaAndAssetCtxs":
            return web.json_response([{"universe": [m for m, _ in UNIVERSE]}, [c for _, c in UNIVERSE]])
        if kind == "candleSnapshot":
            r = body["req"]
            return web.json_response(candles_for(r["coin"], r["interval"], int(r["startTime"]), int(r["endTime"])))
        if kind == "fundingHistory":
            return web.json_response(funding_history(body["coin"], int(body["startTime"]), now))
        if kind == "allMids":
            return web.json_response({m["name"]: c["midPx"] for m, c in UNIVERSE if c["midPx"] is not None})
        if kind == "clearinghouseState":
            return web.json_response({"marginSummary": {"accountValue": "5000.0", "totalMarginUsed": "0.0"},
                                      "assetPositions": [], "withdrawable": "5000.0", "time": now})
        return web.json_response({"error": f"unknown type {kind}"}, status=422)

    app = web.Application()
    app.router.add_post("/info", info)
    return app, calls
