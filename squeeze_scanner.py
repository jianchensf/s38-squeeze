#!/usr/bin/env python3
"""
s38-squeeze — Hyperliquid volatility-compression scanner + execution state engine.

One cycle of the pipeline:

    POST /info metaAndAssetCtxs ──► universe filter ──► closed-bar candles (1h, 4h)
                                │                        └► Bollinger(20,2) vs Keltner(20,1.5) squeeze
                                └► current funding + 24h fundingHistory ──► short-squeeze fuel
    ──► ScanRow per asset ──► StateEngine  IDLE → WATCH → ARMED → FIRE → COOLDOWN
    ──► sinks: stdout table · sqlite · Telegram (optional) · sized dry-run intents

Nothing in this file places orders. FIRE events carry a sized ExecutionIntent for a
separate executor to consume; by design the executor here only logs.

Python 3.10+. Dependencies: aiohttp, numpy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol, Sequence

import aiohttp
import numpy as np

log = logging.getLogger("s38")

HOUR_MS = 3_600_000
TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": HOUR_MS, "2h": 2 * HOUR_MS, "4h": 4 * HOUR_MS, "8h": 8 * HOUR_MS,
    "12h": 12 * HOUR_MS, "1d": 24 * HOUR_MS,
}
# HL funding = premium + interest component of 0.01% per 8h ⇒ the "neutral" hourly rate.
NEUTRAL_HOURLY_FUNDING = 0.0001 / 8  # 1.25e-5


def now_ms() -> int:
    return int(time.time() * 1000)


def utc(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000)) + "Z"


# ───────────────────────────── configuration ──────────────────────────────


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "https://api.hyperliquid.xyz"
    # HL allows 1200 weight/min per IP. Every info call used here costs 20.
    # Default to half the budget so other processes on the same IP keep working; the limiter adapts down on 429.
    weight_per_minute: int = 600
    max_concurrency: int = 2
    timeout_s: float = 15.0
    max_retries: int = 4
    penalty_s: float = 30.0      # silence owed after the first 429 (the shared IP is saturated); doubles per consecutive 429
    penalty_cap_s: float = 300.0


@dataclass(frozen=True)
class UniverseConfig:
    exclude: frozenset = frozenset({"BTC", "ETH", "SOL"})
    min_day_ntl_vlm: float = 2_000_000.0
    max_day_ntl_vlm: float = 40_000_000.0
    # (impact ask − impact bid) / mid, in bps. Slippage proxy for small clips.
    max_impact_spread_bps: float = 60.0


@dataclass(frozen=True)
class SqueezeConfig:
    length: int = 20
    bb_mult: float = 2.0          # Bollinger: SMA(length) ± bb_mult·σ (population σ, as TradingView)
    kc_mult: float = 1.5          # Keltner:   EMA(length) ± kc_mult·SMA(TrueRange, length)
    pctile_lookback: int = 120    # bars used to rank the current BB width
    timeframes: tuple = ("1h", "4h")
    history_bars: int = 200       # closed bars kept per (coin, tf)
    min_refetch_s: float = 90.0   # don't re-hit candleSnapshot for the same key more often than this


@dataclass(frozen=True)
class FuelConfig:
    near_zero_frac: float = 0.5    # funding < frac × neutral counts as "near zero" (negative always does)
    range_lookback: int = 20       # bars for the range-position test
    upper_range_pos: float = 0.75  # close must sit in the top quarter of the range
    fuel_tf: str = "1h"            # timeframe whose range position gates the fuel flag
    sustained_neg_hours: int = 6   # ≥ this many negative hourly prints in 24h ⇒ "sustained"


@dataclass(frozen=True)
class EngineConfig:
    min_squeeze_bars: int = 3      # a release only fires if the squeeze lasted at least this many bars
    cooldown_bars: int = 12        # 1h bars after a FIRE before the coin may re-arm
    drop_after_s: float = 3600.0   # coin unseen in the universe for this long ⇒ state dropped


@dataclass(frozen=True)
class AccountConfig:
    equity_usd: float = 5_000.0
    risk_pct: float = 0.01         # risk per trade as a fraction of equity
    max_leverage: float = 3.0      # PORTFOLIO gross notional cap = equity × max_leverage (all open positions together)
    max_vlm_frac: float = 0.0005   # per-position notional cap = 24h notional volume × this (capacity constraint)
    min_notional_usd: float = 10.0 # HL minimum order size
    long_only: bool = False        # --long-only restricts intents and convex signals to UP breakouts


# ───────────────────────────────── types ──────────────────────────────────


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AssetCtx:
    """One perp: metaAndAssetCtxs universe[i] joined with assetCtxs[i]."""
    coin: str
    sz_decimals: int
    max_leverage: int
    only_isolated: bool
    is_delisted: bool
    funding: float                 # current hourly funding rate (decimal; 1.25e-5 = neutral)
    open_interest: float           # coin units
    day_ntl_vlm: float             # 24h notional volume, USDC
    premium: Optional[float]
    oracle_px: float
    mark_px: float
    mid_px: Optional[float]
    impact_bid: Optional[float]
    impact_ask: Optional[float]
    prev_day_px: float

    @classmethod
    def from_api(cls, meta: dict, ctx: dict) -> "AssetCtx":
        imp = ctx.get("impactPxs") or [None, None]
        return cls(
            coin=str(meta["name"]),
            sz_decimals=int(meta.get("szDecimals", 0)),
            max_leverage=int(meta.get("maxLeverage", 1)),
            only_isolated=bool(meta.get("onlyIsolated", False)),
            is_delisted=bool(meta.get("isDelisted", False)),
            funding=_f(ctx.get("funding")) or 0.0,
            open_interest=_f(ctx.get("openInterest")) or 0.0,
            day_ntl_vlm=_f(ctx.get("dayNtlVlm")) or 0.0,
            premium=_f(ctx.get("premium")),
            oracle_px=_f(ctx.get("oraclePx")) or 0.0,
            mark_px=_f(ctx.get("markPx")) or 0.0,
            mid_px=_f(ctx.get("midPx")),
            impact_bid=_f(imp[0]) if len(imp) > 0 else None,
            impact_ask=_f(imp[1]) if len(imp) > 1 else None,
            prev_day_px=_f(ctx.get("prevDayPx")) or 0.0,
        )

    @property
    def impact_spread_bps(self) -> Optional[float]:
        if self.impact_bid and self.impact_ask and self.mid_px:
            return (self.impact_ask - self.impact_bid) / self.mid_px * 1e4
        return None

    @property
    def day_change(self) -> float:
        return self.mark_px / self.prev_day_px - 1.0 if self.prev_day_px else 0.0

    @property
    def oi_notional(self) -> float:
        return self.open_interest * self.mark_px


@dataclass(frozen=True)
class Candle:
    t: int      # open time, ms
    T: int      # close time, ms
    o: float
    h: float
    l: float
    c: float
    v: float
    n: int

    @classmethod
    def from_api(cls, d: dict) -> "Candle":
        return cls(int(d["t"]), int(d["T"]), float(d["o"]), float(d["h"]), float(d["l"]),
                   float(d["c"]), float(d["v"]), int(d.get("n", 0)))


@dataclass(frozen=True)
class FundingPoint:
    time: int
    rate: float
    premium: float


@dataclass(frozen=True)
class SqueezeMetrics:
    tf: str
    bar_time: int            # open time of the last CLOSED bar these numbers describe
    close: float
    bb_basis: float
    bb_upper: float
    bb_lower: float
    kc_basis: float
    kc_upper: float
    kc_lower: float
    atr: float
    bb_width: float          # (upper − lower) / basis
    squeeze_ratio: float     # BB width / KC width; < 1 ⇒ Bollinger inside Keltner
    squeeze_on: bool         # both Bollinger bands strictly inside the Keltner channel
    squeeze_bars: int        # consecutive closed bars with squeeze_on, ending at the last bar
    prev_squeeze_bars: int   # same, ending at the bar before last (length of a squeeze that just released)
    bars_since_squeeze: int  # 0 = on now, 1 = on at bar[-2], …; −1 = never within the loaded history
    last_squeeze_run: int    # length of the most recent squeeze run (the one that ended bars_since_squeeze ago)
    released: bool           # squeeze_on at bar[-2] and not at bar[-1]
    release_dir: int         # +1 / −1 (TTM momentum sign) when released, else 0
    bbw_pctile: float        # fraction of lookback bars with BB width ≤ current (0 = tightest ever)
    range_pos: float         # (close − LL) / (HH − LL) over range_lookback bars
    n_bars: int


@dataclass(frozen=True)
class FundingStats:
    mean_24h: float
    neg_hours_24h: int
    n: int


@dataclass(frozen=True)
class FuelMetrics:
    funding_hourly: float
    funding_8h_pct: float          # hourly × 8 × 100 — the number traders are used to (neutral = 0.0100)
    near_zero: bool                # funding < near_zero_frac × neutral (includes negative)
    negative: bool
    price_near_upper: bool
    short_squeeze_fuel: bool       # near_zero AND price_near_upper
    funding_24h_mean: Optional[float]
    neg_hours_24h: Optional[int]
    sustained_negative: bool


@dataclass
class ScanRow:
    ts: int
    ctx: AssetCtx
    squeeze: dict                  # tf -> SqueezeMetrics
    fuel: FuelMetrics
    score: float


@dataclass(frozen=True)
class ExecutionIntent:
    coin: str
    side: str                      # "long" | "short"
    entry_px: float
    stop_px: float
    stop_pct: float
    risk_usd: float
    size: float                    # coin units, floored to szDecimals
    notional_usd: float
    leverage: float                # notional / equity
    reason: str


@dataclass(frozen=True)
class SignalEvent:
    ts: int
    coin: str
    kind: str                      # FIRE | ARM | WATCH | DISARM | RESET | COOLDOWN_END | DROP
    state_from: str
    state_to: str
    direction: int
    bar_time: int
    mark_px: float
    note: str = ""
    intent: Optional[ExecutionIntent] = None
    row: Optional[ScanRow] = None


class HLError(RuntimeError):
    pass


# ─────────────────────────── rate-limited client ──────────────────────────


class WeightLimiter:
    """Adaptive token bucket over Hyperliquid's per-IP weight budget (1200/min, shared by every process on the IP).

    - no start-up burst: capacity is two calls, so a cold start is paced from the first second
    - on HTTP 429 (`penalize`): rate halves (floor 10 % of the configured budget) and the bucket goes into debt
      for `debt_s` seconds — a 429 means the whole IP is saturated, which also hurts any live bot behind it
    - after 60 s without a 429 the rate recovers linearly to the configured budget over `recovery_s`
    FIFO, lock-serialised."""

    def __init__(self, weight_per_minute: int, burst_calls: int = 2, call_weight: int = 20,
                 floor_frac: float = 0.1, recovery_s: float = 600.0):
        self.max_rate = weight_per_minute / 60.0
        self.rate = self.max_rate
        self.floor = self.max_rate * floor_frac
        self.capacity = float(burst_calls * call_weight)
        self.tokens = self.capacity
        self.recovery_s = recovery_s
        self.n_429 = 0
        self._updated = time.monotonic()
        self._last_429 = -1e9
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self._updated)
        self._updated = now
        if self.rate < self.max_rate and now - self._last_429 > 60.0:
            self.rate = min(self.max_rate, self.rate + self.max_rate * dt / self.recovery_s)
        self.tokens = min(self.capacity, self.tokens + dt * self.rate)

    async def acquire(self, weight: int) -> None:
        async with self._lock:
            while True:
                self._refill(time.monotonic())
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                await asyncio.sleep((weight - self.tokens) / self.rate)

    def penalize(self, debt_s: float = 30.0) -> float:
        """Record a 429: halve the rate and owe `debt_s` seconds of silence. Returns the new budget in weight/min."""
        self._refill(time.monotonic())
        self.rate = max(self.floor, self.rate / 2.0)
        self.tokens = min(self.tokens, -self.rate * debt_s)
        self._last_429 = self._updated
        self.n_429 += 1
        return self.rate * 60.0


def backoff_429(consecutive: int, base_s: float, cap_s: float, retry_after: Optional[str] = None,
                jitter: float = 0.2, rnd: float = 0.5) -> float:
    """Pause after the n-th consecutive 429: base × 2^(n−1) capped, ±jitter, never below a numeric Retry-After."""
    pause = min(cap_s, base_s * 2 ** max(0, consecutive - 1))
    pause *= 1.0 + jitter * (2.0 * rnd - 1.0)
    try:
        if retry_after is not None:
            pause = max(pause, float(retry_after))
    except ValueError:
        pass
    return pause


class HLInfoClient:
    """Typed wrapper over POST /info. Every call here has weight 20.

    429 handling is process-wide, because a 429 means the IP's 1200/min is saturated (every process on the box shares
    it): the limiter halves its rate and owes a pause that doubles per consecutive 429 — 30 s, 60 s, 120 s, 240 s,
    300 s (cap), ±20 % jitter, or the server's Retry-After if longer — and resets on the first successful response.
    5xx and network errors back off 1.5 s × 2^attempt (cap 30 s) per call."""

    WEIGHT = 20

    def __init__(self, cfg: ClientConfig, session: aiohttp.ClientSession, limiter: WeightLimiter):
        self.cfg = cfg
        self.session = session
        self.limiter = limiter
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self.calls = Counter()
        self.consecutive_429 = 0

    async def _post(self, body: dict, weight: int = WEIGHT) -> Any:
        url = f"{self.cfg.base_url}/info"
        kind = body.get("type", "?")
        last_err: Optional[BaseException] = None
        for attempt in range(self.cfg.max_retries + 1):
            await self.limiter.acquire(weight)
            self.calls[kind] += 1
            try:
                async with self._sem:
                    async with self.session.post(
                        url, json=body, timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_s)
                    ) as r:
                        if r.status == 429:
                            last_err = HLError(f"HTTP 429 for {kind}")
                            self.consecutive_429 += 1
                            pause = backoff_429(self.consecutive_429, self.cfg.penalty_s, self.cfg.penalty_cap_s,
                                                r.headers.get("Retry-After"), rnd=time.monotonic() % 1.0)
                            budget = self.limiter.penalize(pause)
                            log.warning("HTTP 429 for %s — IP budget saturated; budget cut to %.0f weight/min, "
                                        "pausing %.0fs (consecutive 429 #%d, attempt %d)", kind, budget, pause,
                                        self.consecutive_429, attempt + 1)
                            continue                      # the limiter's debt paces the retry
                        if r.status >= 500:
                            last_err = HLError(f"HTTP {r.status} for {kind}")
                            log.warning("%s (attempt %d)", last_err, attempt + 1)
                        elif r.status >= 400:
                            raise HLError(f"HTTP {r.status} for {kind}: {(await r.text())[:200]}")
                        else:
                            self.consecutive_429 = 0
                            return await r.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                log.warning("%s: %s (attempt %d)", kind, e, attempt + 1)
            await asyncio.sleep(min(30.0, 1.5 * 2 ** attempt))
        raise HLError(f"{kind} failed after {self.cfg.max_retries + 1} attempts: {last_err}")

    async def meta_and_asset_ctxs(self) -> list:
        data = await self._post({"type": "metaAndAssetCtxs"})
        try:
            meta, ctxs = data[0], data[1]
            universe = meta["universe"]
        except (KeyError, IndexError, TypeError) as e:
            raise HLError(f"unexpected metaAndAssetCtxs shape: {e}")
        if len(universe) != len(ctxs):
            raise HLError(f"universe/ctx length mismatch {len(universe)} vs {len(ctxs)}")
        return [AssetCtx.from_api(m, c) for m, c in zip(universe, ctxs)]

    async def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list:
        data = await self._post({"type": "candleSnapshot",
                                 "req": {"coin": coin, "interval": interval,
                                         "startTime": int(start_ms), "endTime": int(end_ms)}})
        if not isinstance(data, list):
            raise HLError(f"unexpected candleSnapshot payload for {coin}")
        out = [Candle.from_api(d) for d in data]
        out.sort(key=lambda c: c.t)
        return out

    async def funding_history(self, coin: str, start_ms: int) -> list:
        data = await self._post({"type": "fundingHistory", "coin": coin, "startTime": int(start_ms)})
        if not isinstance(data, list):
            raise HLError(f"unexpected fundingHistory payload for {coin}")
        pts = [FundingPoint(int(d["time"]), float(d["fundingRate"]), float(d.get("premium") or 0.0)) for d in data]
        pts.sort(key=lambda p: p.time)
        return pts

    async def all_mids(self) -> dict:
        """coin -> mid price. Weight 2."""
        data = await self._post({"type": "allMids"}, weight=2)
        if not isinstance(data, dict):
            raise HLError("unexpected allMids payload")
        return {k: float(v) for k, v in data.items() if _f(v) is not None}

    async def user_state(self, address: str) -> dict:
        """clearinghouseState for an account / sub-account address. Weight 2."""
        data = await self._post({"type": "clearinghouseState", "user": address}, weight=2)
        if not isinstance(data, dict) or "marginSummary" not in data:
            raise HLError("unexpected clearinghouseState payload")
        return data

    async def open_orders(self, address: str) -> list:
        """Resting orders of an address (coin, oid, side, sz, limitPx, …). Weight 20."""
        data = await self._post({"type": "openOrders", "user": address})
        if not isinstance(data, list):
            raise HLError("unexpected openOrders payload")
        return data

    async def user_fills_by_time(self, address: str, start_ms: int) -> list:
        """Fills since start_ms (coin, px, sz, side 'B'/'A', fee, time, oid, …). Weight 20."""
        data = await self._post({"type": "userFillsByTime", "user": address, "startTime": int(start_ms)})
        if not isinstance(data, list):
            raise HLError("unexpected userFillsByTime payload")
        return data


# ───────────────────────────── universe filter ────────────────────────────


def filter_universe(assets: Sequence[AssetCtx], cfg: UniverseConfig) -> tuple:
    """Return (eligible assets, {coin: exclusion reason})."""
    eligible: list = []
    reasons: dict = {}
    for a in assets:
        if a.coin in cfg.exclude:
            reasons[a.coin] = "excluded"
        elif a.is_delisted:
            reasons[a.coin] = "delisted"
        elif a.mid_px is None or a.mark_px <= 0:
            reasons[a.coin] = "no_mid"
        elif a.day_ntl_vlm < cfg.min_day_ntl_vlm:
            reasons[a.coin] = "vol_low"
        elif a.day_ntl_vlm > cfg.max_day_ntl_vlm:
            reasons[a.coin] = "vol_high"
        elif (a.impact_spread_bps or 0.0) > cfg.max_impact_spread_bps:
            reasons[a.coin] = "spread_wide"
        else:
            eligible.append(a)
    return eligible, reasons


# ─────────────────────────────── candle cache ─────────────────────────────


def last_closed_open_time(tf: str, t_ms: int) -> int:
    """Open time of the most recent fully closed bar of `tf` at wall time t_ms (UTC-aligned grid)."""
    ms = TF_MS[tf]
    return (t_ms // ms) * ms - ms


class CandleCache:
    """Closed-bar history per (coin, tf) with incremental refresh. Never returns the open bar."""

    def __init__(self, client: HLInfoClient, cfg: SqueezeConfig):
        self.client = client
        self.cfg = cfg
        self._bars: dict = {}
        self._fetched_at: dict = {}
        self._seen: dict = {}

    async def get(self, coin: str, tf: str, t_ms: int) -> list:
        key = (coin, tf)
        self._seen[key] = t_ms
        ms = TF_MS[tf]
        want_last = last_closed_open_time(tf, t_ms)
        bars = self._bars.get(key, [])
        if bars and bars[-1].t >= want_last:
            return bars
        if time.monotonic() - self._fetched_at.get(key, -1e9) < self.cfg.min_refetch_s:
            return bars
        start = bars[-1].t + ms if bars else want_last - (self.cfg.history_bars - 1) * ms
        self._fetched_at[key] = time.monotonic()
        fetched = await self.client.candles(coin, tf, start, t_ms)
        closed = [c for c in fetched if c.t <= want_last]
        if closed:
            merged = {c.t: c for c in bars}
            merged.update({c.t: c for c in closed})
            bars = [merged[k] for k in sorted(merged)][-self.cfg.history_bars:]
            self._bars[key] = bars
        return bars

    def evict_unseen(self, t_ms: int, max_age_ms: int = 24 * HOUR_MS) -> None:
        for key in [k for k, seen in self._seen.items() if t_ms - seen > max_age_ms]:
            self._bars.pop(key, None)
            self._fetched_at.pop(key, None)
            self._seen.pop(key, None)


# ──────────────────────────────── indicators ──────────────────────────────


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        c = np.cumsum(np.concatenate(([0.0], x)))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Population standard deviation (ddof=0), matching TradingView ta.stdev."""
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).std(axis=1)
    return out


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1)
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with the SMA of the first n values (TradingView convention)."""
    out = np.full(x.shape, np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    tr = h - l
    if len(c) > 1:
        pc = c[:-1]
        tr = tr.copy()
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    return tr


def _run_length(flags: np.ndarray, end: int) -> int:
    """Number of consecutive True values ending at index `end` (inclusive)."""
    k = 0
    i = end
    while i >= 0 and flags[i]:
        k += 1
        i -= 1
    return k


def squeeze_bands(h: np.ndarray, l: np.ndarray, c: np.ndarray, cfg: SqueezeConfig) -> dict:
    """Full-series Bollinger / Keltner bands and the squeeze flag. Shared by the scanner and the WFO replay."""
    n = cfg.length
    basis = sma(c, n)
    sd = rolling_std(c, n)
    bb_u, bb_l = basis + cfg.bb_mult * sd, basis - cfg.bb_mult * sd
    kc_mid = ema(c, n)
    atr = sma(true_range(h, l, c), n)
    kc_u, kc_l = kc_mid + cfg.kc_mult * atr, kc_mid - cfg.kc_mult * atr
    with np.errstate(invalid="ignore", divide="ignore"):
        on = (bb_u < kc_u) & (bb_l > kc_l)
        bbw = (bb_u - bb_l) / basis
        ratio = (bb_u - bb_l) / (kc_u - kc_l)
    return {"basis": basis, "bb_u": bb_u, "bb_l": bb_l, "kc_mid": kc_mid, "kc_u": kc_u, "kc_l": kc_l,
            "atr": atr, "on": on, "bbw": bbw, "ratio": ratio}


def squeeze_recency(on: np.ndarray) -> tuple:
    """Per bar: (bars since the squeeze was last on, length of that run). −1/0 when never on so far."""
    n = len(on)
    ago = np.full(n, -1, dtype=int)
    run = np.zeros(n, dtype=int)
    last_j, last_run, cur = -1, 0, 0
    for i in range(n):
        if on[i]:
            cur += 1
            last_j, last_run = i, cur
        else:
            cur = 0
        if last_j >= 0:
            ago[i], run[i] = i - last_j, last_run
    return ago, run


def compute_squeeze(tf: str, candles: Sequence[Candle], cfg: SqueezeConfig,
                    range_lookback: int) -> Optional[SqueezeMetrics]:
    """BB(20,2) inside KC(20,1.5·ATR) squeeze on closed bars. None if not enough history."""
    n = cfg.length
    if len(candles) < max(n, range_lookback) + 2:
        return None
    h = np.array([c.h for c in candles], dtype=float)
    l = np.array([c.l for c in candles], dtype=float)
    c = np.array([c.c for c in candles], dtype=float)

    b = squeeze_bands(h, l, c, cfg)
    basis, bb_u, bb_l, kc_mid, kc_u, kc_l = b["basis"], b["bb_u"], b["bb_l"], b["kc_mid"], b["kc_u"], b["kc_l"]
    atr, on, bbw, ratio = b["atr"], b["on"], b["bbw"], b["ratio"]

    i = len(c) - 1
    bars = _run_length(on, i)
    prev_bars = _run_length(on, i - 1)
    released = bool(on[i - 1] and not on[i])
    j = i
    while j >= 0 and not on[j]:
        j -= 1
    since, last_run = (i - j, _run_length(on, j)) if j >= 0 else (-1, 0)

    hh_n, ll_n = rolling_max(h, n)[i], rolling_min(l, n)[i]
    # TTM / LazyBear momentum proxy: close vs midpoint of (Donchian mid, SMA)
    mom = c[i] - 0.5 * ((hh_n + ll_n) / 2 + basis[i])
    release_dir = (1 if mom >= 0 else -1) if released else 0

    hh_r, ll_r = rolling_max(h, range_lookback)[i], rolling_min(l, range_lookback)[i]
    range_pos = float((c[i] - ll_r) / (hh_r - ll_r)) if hh_r > ll_r else 0.5

    window = bbw[-cfg.pctile_lookback:]
    valid = window[~np.isnan(window)]
    pct = float((valid <= bbw[i]).sum() / len(valid)) if len(valid) and not np.isnan(bbw[i]) else float("nan")

    return SqueezeMetrics(
        tf=tf, bar_time=candles[i].t, close=float(c[i]),
        bb_basis=float(basis[i]), bb_upper=float(bb_u[i]), bb_lower=float(bb_l[i]),
        kc_basis=float(kc_mid[i]), kc_upper=float(kc_u[i]), kc_lower=float(kc_l[i]), atr=float(atr[i]),
        bb_width=float(bbw[i]), squeeze_ratio=float(ratio[i]), squeeze_on=bool(on[i]),
        squeeze_bars=bars, prev_squeeze_bars=prev_bars, bars_since_squeeze=since, last_squeeze_run=last_run,
        released=released, release_dir=release_dir, bbw_pctile=pct, range_pos=range_pos, n_bars=len(candles),
    )


# ─────────────────────────────── funding fuel ─────────────────────────────


class FundingTracker:
    """24h realized hourly funding per coin, refreshed at most once an hour per coin."""

    def __init__(self, client: HLInfoClient, store: "Optional[Store]" = None,
                 refresh_s: float = 3600.0, window_h: int = 24):
        self.client = client
        self.store = store
        self.refresh_s = refresh_s
        self.window_h = window_h
        self._hist: dict = {}
        self._last: dict = {}

    async def refresh(self, coins: Sequence[str], t_ms: int) -> None:
        due = [c for c in coins if t_ms / 1000 - self._last.get(c, 0.0) >= self.refresh_s]
        if not due:
            return
        start = t_ms - (self.window_h + 1) * HOUR_MS
        res = await asyncio.gather(*(self.client.funding_history(c, start) for c in due), return_exceptions=True)
        for coin, r in zip(due, res):
            if isinstance(r, BaseException):
                log.warning("fundingHistory %s: %s", coin, r)
                continue
            self._hist[coin] = r
            self._last[coin] = t_ms / 1000
            if self.store:
                self.store.write_funding(coin, r)

    def stats(self, coin: str, t_ms: int) -> Optional[FundingStats]:
        pts = [p for p in self._hist.get(coin, []) if p.time >= t_ms - self.window_h * HOUR_MS]
        if not pts:
            return None
        rates = [p.rate for p in pts]
        return FundingStats(mean_24h=sum(rates) / len(rates), neg_hours_24h=sum(1 for r in rates if r < 0), n=len(rates))


def compute_fuel(ctx: AssetCtx, sq: Optional[SqueezeMetrics], fstats: Optional[FundingStats],
                 cfg: FuelConfig) -> FuelMetrics:
    f = ctx.funding
    near_zero = f < cfg.near_zero_frac * NEUTRAL_HOURLY_FUNDING
    negative = f < 0.0
    near_upper = sq is not None and sq.range_pos >= cfg.upper_range_pos
    sustained = fstats is not None and fstats.neg_hours_24h >= cfg.sustained_neg_hours
    return FuelMetrics(
        funding_hourly=f, funding_8h_pct=f * 8 * 100,
        near_zero=near_zero, negative=negative, price_near_upper=near_upper,
        short_squeeze_fuel=near_zero and near_upper,
        funding_24h_mean=fstats.mean_24h if fstats else None,
        neg_hours_24h=fstats.neg_hours_24h if fstats else None,
        sustained_negative=sustained,
    )


def score_row(squeeze: dict, fuel: FuelMetrics) -> float:
    """Ranking heuristic for the table. Not a validated predictor — see README."""
    s = 0.0
    for tf, w in (("1h", 2.0), ("4h", 2.0)):
        m = squeeze.get(tf)
        if m is None:
            continue
        if m.squeeze_on:
            s += w + min(m.squeeze_bars, 12) / 12.0
        if not math.isnan(m.bbw_pctile):
            s += 1.0 - m.bbw_pctile
    if fuel.short_squeeze_fuel:
        s += 1.5
    if fuel.sustained_negative:
        s += 0.5
    return round(s, 2)


# ─────────────────────────── execution state engine ───────────────────────


IDLE, WATCH, ARMED, COOLDOWN = "IDLE", "WATCH", "ARMED", "COOLDOWN"


@dataclass
class AssetState:
    coin: str
    state: str = IDLE
    since_ms: int = 0
    last_seen_ms: int = 0
    last_fire_bar: int = -1
    cooldown_until_ms: int = 0


class ExecutionSink(Protocol):
    async def on_event(self, ev: SignalEvent) -> None: ...


def build_intent(row: ScanRow, direction: int, acct: AccountConfig, fuel_tf: str = "1h") -> Optional[ExecutionIntent]:
    """Risk-sized intent: stop at the Keltner basis (channel mean), capped by leverage and liquidity."""
    if direction < 0 and acct.long_only:
        return None
    sq = row.squeeze.get(fuel_tf) or next(iter(row.squeeze.values()))
    entry = row.ctx.mark_px
    stop = sq.kc_basis
    dist = (entry - stop) * direction
    if dist <= 0 or not math.isfinite(dist):
        stop = entry - direction * sq.atr          # fallback: one ATR
        dist = sq.atr
    if dist <= 0:
        return None
    risk_usd = acct.equity_usd * acct.risk_pct
    size = risk_usd / dist
    cap = min(acct.equity_usd * acct.max_leverage, row.ctx.day_ntl_vlm * acct.max_vlm_frac)
    if size * entry > cap:
        size = cap / entry
    q = 10 ** row.ctx.sz_decimals
    size = math.floor(size * q) / q
    notional = size * entry
    if notional < acct.min_notional_usd:
        return None
    return ExecutionIntent(
        coin=row.ctx.coin, side="long" if direction > 0 else "short", entry_px=entry, stop_px=stop,
        stop_pct=dist / entry, risk_usd=round(size * dist, 2), size=size, notional_usd=round(notional, 2),
        leverage=round(notional / acct.equity_usd, 2),
        reason=f"{fuel_tf} squeeze release after {sq.prev_squeeze_bars} bars; stop = KC basis",
    )


class StateEngine:
    """Per-coin FSM fed by ScanRows. Emits SignalEvents to sinks; keeps no orders."""

    def __init__(self, cfg: EngineConfig, acct: AccountConfig, sinks: Sequence[ExecutionSink],
                 fuel_tf: str = "1h", restore_fires: Optional[dict] = None, t_ms: Optional[int] = None):
        self.cfg = cfg
        self.acct = acct
        self.sinks = list(sinks)
        self.fuel_tf = fuel_tf
        self.states: dict = {}
        t_ms = t_ms or now_ms()
        for coin, bar in (restore_fires or {}).items():
            st = AssetState(coin, last_fire_bar=bar, cooldown_until_ms=bar + (cfg.cooldown_bars + 1) * HOUR_MS)
            if t_ms < st.cooldown_until_ms:
                st.state, st.since_ms = COOLDOWN, bar
            self.states[coin] = st

    async def update(self, rows: Sequence[ScanRow], t_ms: int) -> list:
        events: list = []
        for row in rows:
            events.extend(self._step(row, t_ms))
        seen = {r.ctx.coin for r in rows}
        for coin, st in list(self.states.items()):
            if coin not in seen and st.state != IDLE and t_ms - st.last_seen_ms > self.cfg.drop_after_s * 1000:
                events.append(SignalEvent(t_ms, coin, "DROP", st.state, IDLE, 0, 0, 0.0, "left universe"))
                st.state, st.since_ms = IDLE, t_ms
        for ev in events:
            for sink in self.sinks:
                try:
                    await sink.on_event(ev)
                except Exception:  # a sink must never take the scanner down
                    log.exception("sink %s failed on %s %s", type(sink).__name__, ev.kind, ev.coin)
        return events

    def _step(self, row: ScanRow, t_ms: int) -> list:
        coin = row.ctx.coin
        st = self.states.setdefault(coin, AssetState(coin, since_ms=t_ms))
        st.last_seen_ms = t_ms
        events: list = []
        sq = row.squeeze.get(self.fuel_tf)
        on_tfs = [tf for tf, m in row.squeeze.items() if m.squeeze_on]

        def go(kind: str, new: str, direction: int = 0, note: str = "", intent: Optional[ExecutionIntent] = None) -> None:
            events.append(SignalEvent(t_ms, coin, kind, st.state, new, direction,
                                      sq.bar_time if sq else 0, row.ctx.mark_px, note, intent, row))
            st.state, st.since_ms = new, t_ms

        if st.state == COOLDOWN:
            if t_ms >= st.cooldown_until_ms:
                go("COOLDOWN_END", IDLE)
            else:
                return events

        # FIRE: the gating TF just released a squeeze that lasted long enough, and we have not fired on this bar.
        if sq and sq.released and sq.prev_squeeze_bars >= self.cfg.min_squeeze_bars and st.last_fire_bar != sq.bar_time:
            direction = sq.release_dir or (1 if row.ctx.mark_px >= sq.kc_basis else -1)
            intent = build_intent(row, direction, self.acct, self.fuel_tf)
            note = (f"{'UP' if direction > 0 else 'DOWN'} release after {sq.prev_squeeze_bars} bars"
                    f"{' · fuel' if row.fuel.short_squeeze_fuel else ''}"
                    f"{' · 4h squeeze' if '4h' in on_tfs else ''}")
            st.last_fire_bar = sq.bar_time
            st.cooldown_until_ms = sq.bar_time + (self.cfg.cooldown_bars + 1) * HOUR_MS
            go("FIRE", COOLDOWN, direction, note, intent)
            return events

        gate_on = bool(sq and sq.squeeze_on)
        armed = gate_on and (row.fuel.short_squeeze_fuel or len(on_tfs) > 1)
        target = ARMED if armed else WATCH if on_tfs else IDLE
        if target != st.state:
            kind = {ARMED: "ARM", WATCH: "DISARM" if st.state == ARMED else "WATCH", IDLE: "RESET"}[target]
            go(kind, target, note=f"squeeze on: {','.join(on_tfs) or '-'}"
                                 f"{' · fuel' if row.fuel.short_squeeze_fuel else ''}")
        return events


# ────────────────────────────────── sinks ─────────────────────────────────


def format_event(ev: SignalEvent) -> str:
    head = f"[{ev.kind}] {ev.coin} {ev.state_from}→{ev.state_to} @ {ev.mark_px:g}"
    parts = [head]
    if ev.note:
        parts.append(ev.note)
    if ev.row is not None:
        f = ev.row.fuel
        parts.append(f"fund8h {f.funding_8h_pct:+.4f}%")
        for tf, m in sorted(ev.row.squeeze.items()):
            parts.append(f"{tf}: ratio {m.squeeze_ratio:.2f} bars {m.squeeze_bars} pct {m.bbw_pctile:.2f} rng {m.range_pos:.2f}")
    if ev.intent:
        i = ev.intent
        parts.append(f"INTENT {i.side} {i.size:g} {i.coin} @ {i.entry_px:g} stop {i.stop_px:.6g} "
                     f"({i.stop_pct:.2%}) notional ${i.notional_usd:,.0f} lev {i.leverage}x risk ${i.risk_usd}")
    return " | ".join(parts)


class LogSink:
    """Prints every event; FIRE at WARNING level. This is the stand-in executor: it never places orders."""

    async def on_event(self, ev: SignalEvent) -> None:
        log.log(logging.WARNING if ev.kind == "FIRE" else logging.INFO, "%s", format_event(ev))


class TelegramSink:
    def __init__(self, session: aiohttp.ClientSession, token: str, chat_id: str, kinds: Sequence[str] = ("FIRE", "ARM")):
        self.session, self.token, self.chat_id, self.kinds = session, token, chat_id, set(kinds)

    async def on_event(self, ev: SignalEvent) -> None:
        if ev.kind in self.kinds:
            await self.send(f"s38-squeeze {utc(ev.ts)}\n" + format_event(ev).replace(" | ", "\n"))

    async def send(self, text: str) -> None:
        try:
            async with self.session.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                         json={"chat_id": self.chat_id, "text": text},
                                         timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    log.warning("telegram HTTP %s: %s", r.status, (await r.text())[:200])
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("telegram: %s", e)


class Store:
    """sqlite persistence: per-cycle scan rows, events, realized hourly funding, universe counts."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS scans (
        ts INTEGER, coin TEXT, state TEXT, score REAL, mark_px REAL, mid_px REAL, day_change REAL,
        funding REAL, oi_notional REAL, day_ntl_vlm REAL, spread_bps REAL,
        sq1h_on INTEGER, sq1h_ratio REAL, sq1h_bbw REAL, sq1h_pct REAL, sq1h_bars INTEGER, sq1h_range_pos REAL, sq1h_bar_time INTEGER,
        sq4h_on INTEGER, sq4h_ratio REAL, sq4h_bbw REAL, sq4h_pct REAL, sq4h_bars INTEGER, sq4h_range_pos REAL, sq4h_bar_time INTEGER,
        fuel INTEGER, near_zero INTEGER, neg_hours_24h INTEGER, funding_24h_mean REAL, oi REAL,
        PRIMARY KEY (ts, coin));
    CREATE TABLE IF NOT EXISTS events (
        ts INTEGER, coin TEXT, kind TEXT, state_from TEXT, state_to TEXT, direction INTEGER,
        bar_time INTEGER, mark_px REAL, note TEXT, intent TEXT);
    CREATE INDEX IF NOT EXISTS events_coin_kind ON events (coin, kind, bar_time);
    CREATE TABLE IF NOT EXISTS funding_hourly (coin TEXT, time INTEGER, rate REAL, premium REAL, PRIMARY KEY (coin, time));
    CREATE TABLE IF NOT EXISTS universe (ts INTEGER PRIMARY KEY, n_total INTEGER, n_eligible INTEGER, reasons TEXT);
    CREATE TABLE IF NOT EXISTS signals (
        ts INTEGER, coin TEXT, side TEXT, bar_time INTEGER, ref_px REAL, close REAL, high REAL, low REAL,
        donchian_upper REAL, donchian_lower REAL, vol_z REAL, oi_open REAL, oi_close REAL, oi_change REAL,
        squeeze_run INTEGER, bars_since_squeeze INTEGER, atr REAL, stop_px REAL, accepted INTEGER, reasons TEXT,
        PRIMARY KEY (coin, bar_time));
    CREATE TABLE IF NOT EXISTS orders (
        ts INTEGER, coin TEXT, side TEXT, mode TEXT, status TEXT, sz REAL, bound_px REAL, ref_px REAL,
        filled_sz REAL, avg_px REAL, slippage_bps REAL, notional REAL, stop_px REAL, oid INTEGER, stop_oid INTEGER,
        error TEXT, raw TEXT);
    CREATE TABLE IF NOT EXISTS positions (
        coin TEXT, mode TEXT, opened_ms INTEGER, side TEXT, direction INTEGER, entry_px REAL, size REAL,
        initial_stop REAL, stop REAL, r_unit REAL, best_px REAL, stage INTEGER, entry_bar INTEGER,
        last_bar_checked INTEGER, stop_oid INTEGER, risk_usd REAL, notional REAL, fees_usd REAL, sz_decimals INTEGER,
        status TEXT, closed_ms INTEGER, PRIMARY KEY (coin, mode, opened_ms));
    CREATE TABLE IF NOT EXISTS trades (
        opened_ms INTEGER, closed_ms INTEGER, coin TEXT, mode TEXT, side TEXT, entry_px REAL, exit_px REAL, size REAL,
        r_unit REAL, pnl_usd REAL, r_multiple REAL, fees_usd REAL, stage INTEGER, reason TEXT,
        duration_s REAL, entry_slip_bps REAL, exit_slip_bps REAL, funding_usd REAL,
        PRIMARY KEY (coin, mode, opened_ms));
    CREATE TABLE IF NOT EXISTS breaker (ts INTEGER PRIMARY KEY, equity REAL, peak REAL, drawdown REAL, halt_until_ms INTEGER);
    CREATE TABLE IF NOT EXISTS position_events (
        ts INTEGER, coin TEXT, mode TEXT, opened_ms INTEGER, kind TEXT, px REAL, stop REAL, stage INTEGER, detail TEXT);
    CREATE TABLE IF NOT EXISTS params (key TEXT PRIMARY KEY, value REAL, since_ms INTEGER, source TEXT);
    CREATE TABLE IF NOT EXISTS wfo_runs (
        ts INTEGER PRIMARY KEY, window_start INTEGER, window_end INTEGER, split_ms INTEGER, n_coins INTEGER,
        chosen_donchian INTEGER, chosen_trail REAL, adopted INTEGER, verdict TEXT, detail TEXT);
    CREATE TABLE IF NOT EXISTS candles (coin TEXT, t INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (coin, t));
    CREATE TABLE IF NOT EXISTS candle_requests (coin TEXT PRIMARY KEY, start_ms INTEGER, end_ms INTEGER);
    CREATE TABLE IF NOT EXISTS equity_curve (ts INTEGER, mode TEXT, equity REAL, PRIMARY KEY (ts, mode));
    """

    MIGRATIONS = (
        ("scans", "oi", "REAL"),                    # v1 → OI in coin units for the convex engine
        ("trades", "duration_s", "REAL"),           # v3 → trade-log enrichment for the WFO module
        ("trades", "entry_slip_bps", "REAL"),
        ("trades", "exit_slip_bps", "REAL"),
        ("trades", "funding_usd", "REAL"),
        ("positions", "ref_px", "REAL"),
        ("signals", "squeeze_tf", "TEXT"),          # v5 → which timeframe's squeeze satisfied the compression rule
    )

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(self.SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, col, decl in self.MIGRATIONS:
            cols = {r[1] for r in self.con.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                self.con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.con.commit()

    def last_fire_bars(self) -> dict:
        cur = self.con.execute("SELECT coin, MAX(bar_time) FROM events WHERE kind='FIRE' GROUP BY coin")
        return {coin: int(bar) for coin, bar in cur.fetchall()}

    def write_universe(self, ts: int, n_total: int, n_eligible: int, reasons: dict) -> None:
        self.con.execute("INSERT OR REPLACE INTO universe VALUES (?,?,?,?)", (ts, n_total, n_eligible, json.dumps(reasons)))
        self.con.commit()

    def write_funding(self, coin: str, pts: Sequence[FundingPoint]) -> None:
        self.con.executemany("INSERT OR IGNORE INTO funding_hourly VALUES (?,?,?,?)",
                             [(coin, p.time, p.rate, p.premium) for p in pts])
        self.con.commit()

    def write_scans(self, rows: Sequence[ScanRow], states: dict) -> None:
        def tfcols(m: Optional[SqueezeMetrics]) -> tuple:
            if m is None:
                return (None,) * 7
            return (int(m.squeeze_on), m.squeeze_ratio, m.bb_width, m.bbw_pctile, m.squeeze_bars, m.range_pos, m.bar_time)
        recs = []
        for r in rows:
            c, f = r.ctx, r.fuel
            st = states.get(c.coin)
            recs.append((r.ts, c.coin, st.state if st else None, r.score, c.mark_px, c.mid_px, c.day_change,
                         c.funding, c.oi_notional, c.day_ntl_vlm, c.impact_spread_bps,
                         *tfcols(r.squeeze.get("1h")), *tfcols(r.squeeze.get("4h")),
                         int(f.short_squeeze_fuel), int(f.near_zero), f.neg_hours_24h, f.funding_24h_mean,
                         c.open_interest))
        self.con.executemany(f"INSERT OR REPLACE INTO scans VALUES ({','.join('?' * 30)})", recs)
        self.con.commit()

    async def on_event(self, ev: SignalEvent) -> None:
        self.con.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (ev.ts, ev.coin, ev.kind, ev.state_from, ev.state_to, ev.direction, ev.bar_time,
                          ev.mark_px, ev.note, json.dumps(asdict(ev.intent)) if ev.intent else None))
        self.con.commit()

    # ── convex engine ──
    def oi_history(self, since_ms: int) -> list:
        return self.con.execute("SELECT ts, coin, oi FROM scans WHERE ts >= ? AND oi IS NOT NULL", (since_ms,)).fetchall()

    def signal_keys(self) -> list:
        return [(c, int(b)) for c, b in self.con.execute("SELECT coin, bar_time FROM signals").fetchall()]

    def write_signal(self, s: Any, accepted: bool, reasons: Sequence[str]) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO signals (ts, coin, side, bar_time, ref_px, close, high, low, donchian_upper, donchian_lower, "
            "vol_z, oi_open, oi_close, oi_change, squeeze_run, bars_since_squeeze, atr, stop_px, accepted, reasons, squeeze_tf) "
            f"VALUES ({','.join('?' * 21)})",
            (s.ts, s.coin, s.side, s.bar_time, s.ref_px, s.close, s.high, s.low, s.donchian_upper, s.donchian_lower,
             s.vol_z, s.oi_open, s.oi_close, s.oi_change, s.squeeze_run, s.bars_since_squeeze, s.atr, s.stop_px,
             int(accepted), "; ".join(reasons), getattr(s, "squeeze_tf", "1h")))
        self.con.commit()

    def write_order(self, r: Any) -> None:
        self.con.execute(
            f"INSERT INTO orders VALUES ({','.join('?' * 17)})",
            (r.ts, r.coin, r.side, r.mode, r.status, r.sz, r.bound_px, r.ref_px, r.filled_sz, r.avg_px,
             r.slippage_bps, r.notional, r.stop_px, r.oid, r.stop_oid, r.error, r.raw))
        self.con.commit()

    # ── risk manager / position book ──
    POS_COLS = ("coin", "mode", "opened_ms", "side", "direction", "entry_px", "size", "initial_stop", "stop", "r_unit",
                "best_px", "stage", "entry_bar", "last_bar_checked", "stop_oid", "risk_usd", "notional", "fees_usd",
                "sz_decimals", "status", "closed_ms", "ref_px")

    def upsert_position(self, p: Any) -> None:
        self.con.execute(f"INSERT OR REPLACE INTO positions ({','.join(self.POS_COLS)}) VALUES ({','.join('?' * len(self.POS_COLS))})",
                         tuple(getattr(p, c) for c in self.POS_COLS))
        self.con.commit()

    def open_positions(self, mode: str) -> list:
        from risk_manager import Position
        cur = self.con.execute(f"SELECT {','.join(self.POS_COLS)} FROM positions WHERE mode=? AND status='open'", (mode,))
        return [Position(**dict(zip(self.POS_COLS, row))) for row in cur.fetchall()]

    TRADE_COLS = ("opened_ms", "closed_ms", "coin", "mode", "side", "entry_px", "exit_px", "size", "r_unit", "pnl_usd",
                  "r_multiple", "fees_usd", "stage", "reason", "duration_s", "entry_slip_bps", "exit_slip_bps", "funding_usd")

    def write_trade(self, t: dict) -> None:
        self.con.execute(f"INSERT OR REPLACE INTO trades ({','.join(self.TRADE_COLS)}) VALUES ({','.join('?' * len(self.TRADE_COLS))})",
                         tuple(t.get(c) for c in self.TRADE_COLS))
        self.con.commit()

    def trades(self, mode: str, since_ms: int = 0) -> list:
        cur = self.con.execute(f"SELECT {','.join(self.TRADE_COLS)} FROM trades WHERE mode=? AND closed_ms>=? ORDER BY closed_ms",
                               (mode, since_ms))
        return [dict(zip(self.TRADE_COLS, row)) for row in cur.fetchall()]

    def write_position_event(self, ts: int, coin: str, mode: str, opened_ms: int, kind: str, px: float,
                             stop: float, stage: int, detail: str = "") -> None:
        self.con.execute("INSERT INTO position_events VALUES (?,?,?,?,?,?,?,?,?)",
                         (ts, coin, mode, opened_ms, kind, px, stop, stage, detail))
        self.con.commit()

    def funding_between(self, coin: str, start_ms: int, end_ms: int) -> list:
        """Realized hourly funding prints (time, rate) for a coin inside [start, end]."""
        return self.con.execute("SELECT time, rate FROM funding_hourly WHERE coin=? AND time>=? AND time<=? ORDER BY time",
                                (coin, start_ms, end_ms)).fetchall()

    # ── parameters (written by the WFO job, read by the running scanner) ──
    DEFAULT_PARAMS = {"donchian_len": 20.0, "trail_atr_mult": 2.5}

    def active_params(self) -> dict:
        out = dict(self.DEFAULT_PARAMS)
        for key, value in self.con.execute("SELECT key, value FROM params"):
            if key in out:
                out[key] = float(value)
        return out

    def set_params(self, values: dict, source: str, ts: int) -> None:
        self.con.executemany("INSERT OR REPLACE INTO params VALUES (?,?,?,?)",
                             [(k, float(v), ts, source) for k, v in values.items()])
        self.con.commit()

    def write_wfo_run(self, r: dict) -> None:
        self.con.execute("INSERT OR REPLACE INTO wfo_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (r["ts"], r["window_start"], r["window_end"], r["split_ms"], r["n_coins"], r["chosen_donchian"],
                          r["chosen_trail"], int(r["adopted"]), r["verdict"], json.dumps(r["detail"])))
        self.con.commit()

    # ── history for the replay ──
    def write_candles(self, coin: str, candles: Sequence[Candle]) -> None:
        self.con.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)",
                             [(coin, c.t, c.o, c.h, c.l, c.c, c.v) for c in candles])
        self.con.commit()

    def read_candles(self, coin: str, start_ms: int, end_ms: int) -> list:
        rows = self.con.execute("SELECT t, o, h, l, c, v FROM candles WHERE coin=? AND t>=? AND t<=? ORDER BY t",
                                (coin, start_ms, end_ms)).fetchall()
        return [Candle(t, t + HOUR_MS - 1, o, h, l, c, v, 0) for t, o, h, l, c, v in rows]

    def candle_request_span(self, coin: str) -> Optional[tuple]:
        """(earliest start, latest end) ever asked of the API for this coin — so an empty head or a coin with no
        data is not re-requested on every run."""
        row = self.con.execute("SELECT start_ms, end_ms FROM candle_requests WHERE coin=?", (coin,)).fetchone()
        return (int(row[0]), int(row[1])) if row else None

    def mark_candle_request(self, coin: str, start_ms: int, end_ms: int) -> None:
        prev = self.candle_request_span(coin)
        a = min(start_ms, prev[0]) if prev else start_ms
        b = max(end_ms, prev[1]) if prev else end_ms
        self.con.execute("INSERT OR REPLACE INTO candle_requests VALUES (?,?,?)", (coin, int(a), int(b)))
        self.con.commit()

    def cached_coins(self) -> list:
        return [c for (c,) in self.con.execute("SELECT DISTINCT coin FROM candles ORDER BY coin")]

    def latest_day_ntl_vlm(self) -> dict:
        """coin -> HL dayNtlVlm at the latest scan (to check the unit of candle volume against)."""
        return {c: float(v) for c, v in self.con.execute(
            "SELECT coin, day_ntl_vlm FROM scans WHERE ts = (SELECT MAX(ts) FROM scans)") if v is not None}

    def eligibility(self, start_ms: int, end_ms: int) -> dict:
        """coin -> set of hour-open ms during which the coin was in the scanned universe (one scan row per cycle)."""
        out: dict = {}
        for coin, hour in self.con.execute(
                "SELECT DISTINCT coin, (ts / 3600000) * 3600000 FROM scans WHERE ts>=? AND ts<=?", (start_ms, end_ms)):
            out.setdefault(coin, set()).add(int(hour))
        return out

    def oi_samples(self, coin: str, start_ms: int, end_ms: int) -> list:
        return self.con.execute("SELECT ts, oi FROM scans WHERE coin=? AND ts>=? AND ts<=? AND oi IS NOT NULL ORDER BY ts",
                                (coin, start_ms, end_ms)).fetchall()

    def logged_span(self) -> tuple:
        row = self.con.execute("SELECT MIN(ts), MAX(ts) FROM scans").fetchone()
        return (int(row[0]), int(row[1])) if row and row[0] else (0, 0)

    def latest_universe(self) -> list:
        return [c for (c,) in self.con.execute("SELECT DISTINCT coin FROM scans WHERE ts = (SELECT MAX(ts) FROM scans) ORDER BY coin")]

    def trade_r_multiples(self, mode: str) -> list:
        return [r for (r,) in self.con.execute("SELECT r_multiple FROM trades WHERE mode=? ORDER BY closed_ms", (mode,))]

    def realized_pnl(self, mode: str) -> float:
        return float(self.con.execute("SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE mode=?", (mode,)).fetchone()[0])

    def write_breaker(self, trip: dict) -> None:
        self.con.execute("INSERT OR REPLACE INTO breaker VALUES (?,?,?,?,?)",
                         (trip["ts"], trip["equity"], trip["peak"], trip["drawdown"], trip["halt_until_ms"]))
        self.con.commit()

    def breaker_state(self) -> Any:
        from risk_manager import BreakerState
        row = self.con.execute("SELECT ts, halt_until_ms, (SELECT COUNT(*) FROM breaker) FROM breaker ORDER BY ts DESC LIMIT 1").fetchone()
        return BreakerState(halt_until_ms=int(row[1]), tripped_ms=int(row[0]), trips=int(row[2])) if row else BreakerState()

    def write_equity(self, ts: int, mode: str, equity: float) -> None:
        self.con.execute("INSERT OR REPLACE INTO equity_curve VALUES (?,?,?)", (ts, mode, equity))
        self.con.commit()

    def equity_samples(self, mode: str, since_ms: int) -> list:
        return self.con.execute("SELECT ts, equity FROM equity_curve WHERE mode=? AND ts>=? ORDER BY ts", (mode, since_ms)).fetchall()

    def equity_peak(self, mode: str, since_ms: int) -> float:
        row = self.con.execute("SELECT MAX(equity) FROM equity_curve WHERE mode=? AND ts>=?", (mode, since_ms)).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def last_close_times(self, mode: str) -> dict:
        return {coin: int(ts) for coin, ts in self.con.execute("SELECT coin, MAX(closed_ms) FROM trades WHERE mode=? GROUP BY coin", (mode,))}

    def close(self) -> None:
        self.con.close()


# ─────────────────────────────── presentation ─────────────────────────────


def render_table(rows: Sequence[ScanRow], states: dict, top: int, t_ms: int) -> str:
    hdr = (f"{'COIN':<9}{'PX':>11}{'24h%':>7}{'VOL$M':>7}{'OI$M':>7}{'F8h%':>9}  "
           f"{'1h':<22}{'4h':<22}{'RNG':>5} {'FUEL':<5}{'STATE':<9}{'SCORE':>6}")
    lines = [f"── s38-squeeze {utc(t_ms)} · {len(rows)} eligible · top {min(top, len(rows))} by score ──", hdr]

    def tfcell(m: Optional[SqueezeMetrics]) -> str:
        if m is None:
            return f"{'—':<22}"
        tag = "REL" + ("↑" if m.release_dir > 0 else "↓") if m.released else ("ON " if m.squeeze_on else "off")
        return f"{tag} r{m.squeeze_ratio:4.2f} b{m.squeeze_bars:<3d}p{m.bbw_pctile:4.2f} "

    for r in rows[:top]:
        c, f = r.ctx, r.fuel
        st = states.get(c.coin)
        s1, s4 = r.squeeze.get("1h"), r.squeeze.get("4h")
        rng = s1.range_pos if s1 else (s4.range_pos if s4 else float("nan"))
        fuel = "YES" if f.short_squeeze_fuel else ("nz" if f.near_zero else "-")
        if f.sustained_negative:
            fuel += "*"
        lines.append(f"{c.coin:<9}{c.mark_px:>11.6g}{c.day_change * 100:>+7.1f}{c.day_ntl_vlm / 1e6:>7.1f}"
                     f"{c.oi_notional / 1e6:>7.1f}{f.funding_8h_pct:>+9.4f}  {tfcell(s1)}{tfcell(s4)}"
                     f"{rng:>5.2f} {fuel:<5}{st.state if st else '-':<9}{r.score:>6.2f}")
    lines.append("1h/4h: ON=BB inside KC · REL↑/↓=released this bar · r=BBwidth/KCwidth · b=bars in squeeze · "
                 "p=BBW percentile (0=tightest) · RNG=20-bar range position · FUEL: YES=near-zero/neg funding & "
                 "upper range, nz=near-zero only, *=≥6 neg hours/24h")
    return "\n".join(lines)


# ───────────────────────────────── scanner ────────────────────────────────


class Scanner:
    def __init__(self, client: HLInfoClient, cache: CandleCache, funding: FundingTracker, engine: StateEngine,
                 store: Optional[Store], ucfg: UniverseConfig, scfg: SqueezeConfig, fcfg: FuelConfig,
                 top: int = 25, quiet: bool = False, coins: Optional[Sequence[str]] = None,
                 hooks: Optional[list] = None):
        self.client, self.cache, self.funding, self.engine, self.store = client, cache, funding, engine, store
        self.ucfg, self.scfg, self.fcfg = ucfg, scfg, fcfg
        self.top, self.quiet = top, quiet
        self.only = set(coins) if coins else None
        self.hooks = list(hooks or [])      # async fn(rows, candles, t_ms, ctx_all) — e.g. ConvexRunner.on_cycle
        self.extra_coins = lambda: set()    # coins to keep fetching even when out of the universe (open positions)
        self.ctx_all: dict = {}
        self.cycles = 0

    async def cycle(self) -> list:
        t_ms = now_ms()
        t0 = time.monotonic()
        assets = await self.client.meta_and_asset_ctxs()
        self.ctx_all = {a.coin: a for a in assets}
        eligible, reasons = filter_universe(assets, self.ucfg)
        if self.only:
            eligible = [a for a in eligible if a.coin in self.only]
        counts = Counter(reasons.values())
        log.info("universe: %d perps, %d eligible (%s)", len(assets), len(eligible),
                 " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if self.store:
            self.store.write_universe(t_ms, len(assets), len(eligible), dict(counts))

        keys = [(a.coin, tf) for a in eligible for tf in self.scfg.timeframes]
        held = {a.coin for a in eligible}
        for coin in sorted(self.extra_coins() - held):       # open positions whose coin left the universe
            if coin in self.ctx_all:
                keys.append((coin, self.scfg.timeframes[0]))
        if self.cycles == 0:
            est = len(keys) * HLInfoClient.WEIGHT / self.client.limiter.rate
            log.info("cold start: loading %d candle series (~%.0fs at %d weight/min)", len(keys), est,
                     int(self.client.limiter.rate * 60))
        t_fetch = time.monotonic()
        results = await asyncio.gather(*(self.cache.get(c, tf, t_ms) for c, tf in keys), return_exceptions=True)
        fetch_s = time.monotonic() - t_fetch
        if fetch_s > 600:
            log.warning("candle refresh took %.0fs — breakout bars older than 15 min are refused as stale; the IP budget "
                        "(%.0f/min after %d×429) is too small for this universe on this box", fetch_s,
                        self.client.limiter.rate * 60, self.client.limiter.n_429)
        candles: dict = {}
        for key, res in zip(keys, results):
            if isinstance(res, BaseException):
                log.warning("candles %s/%s: %s", key[0], key[1], res)
            else:
                candles[key] = res

        squeezes: dict = {}
        for a in eligible:
            sq = {}
            for tf in self.scfg.timeframes:
                bars = candles.get((a.coin, tf))
                m = compute_squeeze(tf, bars, self.scfg, self.fcfg.range_lookback) if bars else None
                if m:
                    sq[tf] = m
            if sq:
                squeezes[a.coin] = sq

        # Realized funding history for coins in a squeeze (fuel) and for coins with an open position (trade funding).
        await self.funding.refresh(sorted({c for c, sq in squeezes.items() if any(m.squeeze_on for m in sq.values())}
                                          | set(self.extra_coins())), t_ms)

        rows = []
        for a in eligible:
            sq = squeezes.get(a.coin)
            if not sq:
                continue
            fuel = compute_fuel(a, sq.get(self.fcfg.fuel_tf), self.funding.stats(a.coin, t_ms), self.fcfg)
            rows.append(ScanRow(ts=t_ms, ctx=a, squeeze=sq, fuel=fuel, score=score_row(sq, fuel)))
        rows.sort(key=lambda r: (r.score, r.ctx.day_ntl_vlm), reverse=True)

        events = await self.engine.update(rows, t_ms)
        if self.store:
            self.store.write_scans(rows, self.engine.states)
        for hook in self.hooks:
            try:
                await hook(rows, candles, t_ms, self.ctx_all)
            except Exception:       # a hook must never take the scanner down
                log.exception("hook %s failed", getattr(hook, "__qualname__", hook))
        self.cache.evict_unseen(t_ms)
        self.cycles += 1
        n_on = sum(1 for r in rows if any(m.squeeze_on for m in r.squeeze.values()))
        n_fuel = sum(1 for r in rows if r.fuel.short_squeeze_fuel)
        log.info("cycle %d: %d rows, %d squeezed, %d with fuel, %d events, %d api calls, %d×429, budget %.0f/min, %.1fs",
                 self.cycles, len(rows), n_on, n_fuel, len(events), sum(self.client.calls.values()),
                 self.client.limiter.n_429, self.client.limiter.rate * 60, time.monotonic() - t0)
        if not self.quiet:
            print(render_table(rows, self.engine.states, self.top, t_ms), flush=True)
        return rows


# ─────────────────────────────────── main ─────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperliquid BB/KC squeeze + funding-fuel scanner (read-only).")
    p.add_argument("--once", action="store_true", help="run one cycle and exit")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between cycles (default 60)")
    p.add_argument("--db", default="state/s38.db", help="sqlite path ('' to disable)")
    p.add_argument("--min-vol", type=float, default=2e6)
    p.add_argument("--max-vol", type=float, default=40e6)
    p.add_argument("--exclude", default="BTC,ETH,SOL", help="comma-separated coins to exclude")
    p.add_argument("--max-spread-bps", type=float, default=60.0)
    p.add_argument("--tfs", default="1h,4h")
    p.add_argument("--coins", default="", help="debug: restrict to these coins (comma-separated)")
    p.add_argument("--equity", type=float, default=5000.0, help="account equity for sizing (dry mode; live reads the account)")
    p.add_argument("--risk-pct", type=float, default=1.0, help="bootstrap risk per trade (%% of equity) until 20 realized trades exist")
    p.add_argument("--max-risk-pct", type=float, default=2.0, help="cap on quarter-Kelly risk per trade (%% of equity); 4%% means a 7-loss streak is −25%%")
    p.add_argument("--long-only", action="store_true", help="UP breakouts only (default: both sides)")
    p.add_argument("--weight", type=int, default=600, help="HL weight budget per minute (IP limit is 1200)")
    p.add_argument("--base-url", default="https://api.hyperliquid.xyz")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--quiet", action="store_true", help="no table, log lines only")
    p.add_argument("--log-level", default="INFO")
    g = p.add_argument_group("convex engine (Donchian breakout + volume z + OI inflow → IOC entry)")
    g.add_argument("--convex", action="store_true", help="run the ConvexSignalEngine on each cycle (dry-run orders)")
    g.add_argument("--live", action="store_true", help="send real IOC orders; needs HL_API_WALLET_KEY (+ HL_ACCOUNT_ADDRESS) in env")
    g.add_argument("--max-positions", type=int, default=3)
    g.add_argument("--max-slippage-bps", type=float, default=8.0, help="IOC bound off the fresh mid (default 0.08%%)")
    v = p.add_argument_group("db views (print and exit)")
    v.add_argument("--events", type=int, nargs="?", const=30, metavar="N", help="last N state-engine events")
    v.add_argument("--scans", action="store_true", help="latest scan rows ranked by score")
    v.add_argument("--signals", type=int, nargs="?", const=30, metavar="N", help="last N convex signals (accepted and rejected)")
    v.add_argument("--orders", type=int, nargs="?", const=30, metavar="N", help="last N order reports")
    v.add_argument("--positions", action="store_true", help="open positions (paper and live) with stop stage")
    v.add_argument("--trades", type=int, nargs="?", const=30, metavar="N", help="last N closed trades + R statistics")
    v.add_argument("--wfo", action="store_true", help="active parameters and the last walk-forward runs")
    return p.parse_args(argv)


def show_db(path: str, what: str, n: int = 30) -> None:
    """Read-only views over the sqlite store (no sqlite3 CLI needed on the box)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    g = lambda v, spec: format(v, spec) if v is not None else "—"  # noqa: E731
    if what == "events":
        print(f"{'UTC':<18}{'COIN':<9}{'KIND':<13}{'FROM':<9}{'TO':<9}{'DIR':>4}{'PX':>11}  NOTE")
        for ts, coin, kind, f, t, d, px, note in con.execute(
                "SELECT ts, coin, kind, state_from, state_to, direction, mark_px, note FROM events ORDER BY ts DESC LIMIT ?", (n,)):
            print(f"{utc(ts):<18}{coin:<9}{kind:<13}{f:<9}{t:<9}{d:>4}{px:>11.6g}  {note}")
    elif what == "signals":
        print(f"{'BAR (UTC)':<18}{'COIN':<9}{'SIDE':<6}{'CLOSE':>11}{'DONCH':>11}{'VOL_Z':>7}{'OI%':>7}{'SQZ':>7}{'AGO':>4}{'STOP':>11} OK REASONS")
        for bt, coin, side, close, du, dl, vz, oic, run, ago, stop, ok, reasons, sqtf in con.execute(
                "SELECT bar_time, coin, side, close, donchian_upper, donchian_lower, vol_z, oi_change, squeeze_run, "
                "bars_since_squeeze, stop_px, accepted, reasons, squeeze_tf FROM signals ORDER BY bar_time DESC LIMIT ?", (n,)):
            band = du if side == "long" else dl
            print(f"{utc(bt):<18}{coin:<9}{side:<6}{close:>11.6g}{band:>11.6g}{g(vz, '.1f'):>7}"
                  f"{g(oic * 100 if oic is not None else None, '+.1f'):>7}{str(run) + (sqtf or '1h')[:2]:>7}{ago:>4}{stop:>11.6g} {'Y' if ok else '-':>2} {reasons}")
    elif what == "orders":
        print(f"{'UTC':<18}{'COIN':<9}{'SIDE':<6}{'MODE':<5}{'STATUS':<9}{'SZ':>9}{'BOUND':>11}{'REF':>11}{'FILLED':>9}{'AVG':>11}{'SLIP':>6}{'STOP':>11} NOTE")
        for ts, coin, side, mode, st, sz, bound, ref, filled, avg, slip, stop, err in con.execute(
                "SELECT ts, coin, side, mode, status, sz, bound_px, ref_px, filled_sz, avg_px, slippage_bps, stop_px, error "
                "FROM orders ORDER BY ts DESC LIMIT ?", (n,)):
            print(f"{utc(ts):<18}{coin:<9}{side:<6}{mode:<5}{st:<9}{g(sz, 'g'):>9}{g(bound, '.6g'):>11}{g(ref, '.6g'):>11}"
                  f"{g(filled, 'g'):>9}{g(avg, '.6g'):>11}{g(slip, '+.1f'):>6}{g(stop, '.6g'):>11} {err or ''}")
    elif what == "positions":
        print(f"{'OPENED (UTC)':<18}{'COIN':<9}{'MODE':<5}{'SIDE':<6}{'SIZE':>9}{'ENTRY':>11}{'STOP':>11}{'BEST':>11}{'EXC_R':>6}{'STG':>4}{'RISK$':>8}{'STOP_OID':>10}")
        for coin, mode, side, size, entry, stop, best, d, r_unit, stage, opened, risk, soid in con.execute(
                "SELECT coin, mode, side, size, entry_px, stop, best_px, direction, r_unit, stage, opened_ms, risk_usd, stop_oid "
                "FROM positions WHERE status='open' ORDER BY opened_ms"):
            exc = d * (best - entry) / r_unit if r_unit else 0.0
            print(f"{utc(opened):<18}{coin:<9}{mode:<5}{side:<6}{size:>9g}{entry:>11.6g}{stop:>11.6g}{best:>11.6g}{exc:>6.2f}{stage:>4}{risk:>8.2f}{g(soid, 'd'):>10}")
        row = con.execute("SELECT ts, drawdown, halt_until_ms FROM breaker ORDER BY ts DESC LIMIT 1").fetchone()
        if row:
            print(f"breaker: last trip {utc(row[0])} dd {row[1]:.1%} halt until {utc(row[2])}")
    elif what == "wfo":
        for key, value, since, source in con.execute("SELECT key, value, since_ms, source FROM params ORDER BY key"):
            print(f"param {key} = {value:g}  (since {utc(since)}; {source[:90]})")
        if not con.execute("SELECT 1 FROM params").fetchone():
            print("params: defaults (donchian_len 20, trail_atr_mult 2.5) — no WFO run has written anything yet")
        print(f"{'RUN (UTC)':<18}{'WINDOW':<30}{'COINS':>6}{'CHOSEN':>12} ADOPTED VERDICT")
        for ts, ws, we, n, L, m, ad, verdict in con.execute(
                "SELECT ts, window_start, window_end, n_coins, chosen_donchian, chosen_trail, adopted, verdict FROM wfo_runs ORDER BY ts DESC LIMIT 10"):
            print(f"{utc(ts):<18}{utc(ws)[:10] + ' → ' + utc(we)[:10]:<30}{n:>6}{f'({L}, {m:g})':>12} {'yes' if ad else 'no':<7} {verdict}")
    elif what == "trades":
        print(f"{'CLOSED (UTC)':<18}{'COIN':<9}{'MODE':<5}{'SIDE':<6}{'ENTRY':>11}{'EXIT':>11}{'SIZE':>9}{'R':>7}{'PNL$':>9}{'FEES$':>7}{'STG':>4} REASON")
        rows_ = con.execute("SELECT closed_ms, coin, mode, side, entry_px, exit_px, size, r_multiple, pnl_usd, fees_usd, stage, reason "
                            "FROM trades ORDER BY closed_ms DESC LIMIT ?", (n,)).fetchall()
        for c_ms, coin, mode, side, e, x, size, r, pnl, fees, stage, reason in rows_:
            print(f"{utc(c_ms):<18}{coin:<9}{mode:<5}{side:<6}{e:>11.6g}{x:>11.6g}{size:>9g}{r:>+7.2f}{pnl:>+9.2f}{fees:>7.2f}{stage:>4} {reason}")
        for mode, in con.execute("SELECT DISTINCT mode FROM trades"):
            for leg in ("all", "long", "short"):
                q = "SELECT r_multiple FROM trades WHERE mode=?" + ("" if leg == "all" else " AND side=?")
                rs = [r for (r,) in con.execute(q, (mode,) if leg == "all" else (mode, leg))]
                if not rs:
                    continue
                wins, losses = [r for r in rs if r > 0], [-r for r in rs if r <= 0]
                n_ = len(rs)
                p = len(wins) / n_
                mean_r = sum(rs) / n_
                se_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / (n_ - 1) / n_) if n_ > 1 else float("nan")
                payoff = (sum(wins) / len(wins)) / (sum(losses) / len(losses)) if wins and losses else float("nan")
                print(f"[{mode} {leg:<5}] n={n_:<4} win rate {p:.2f} ± {math.sqrt(p * (1 - p) / n_):.2f}  "
                      f"avg R {mean_r:+.3f} ± {se_r:.3f} (SE)  payoff {payoff:.2f}  sum R {sum(rs):+.2f}")
    else:
        row = con.execute("SELECT MAX(ts) FROM scans").fetchone()
        print(f"latest scan {utc(row[0]) if row and row[0] else '—'}")
        print(f"{'COIN':<9}{'STATE':<9}{'SCORE':>6}{'PX':>11}{'F8h%':>9}{'1h':>5}{'r1h':>6}{'b1h':>5}{'4h':>5}{'r4h':>6}{'b4h':>5}{'FUEL':>5}")
        for coin, st, sc, px, fund, on1, r1, b1, on4, r4, b4, fuel in con.execute(
                "SELECT coin, state, score, mark_px, funding, sq1h_on, sq1h_ratio, sq1h_bars, sq4h_on, sq4h_ratio, sq4h_bars, fuel "
                "FROM scans WHERE ts = (SELECT MAX(ts) FROM scans) ORDER BY score DESC LIMIT ?", (n,)):
            print(f"{coin:<9}{st or '-':<9}{sc:>6.2f}{px:>11.6g}{fund * 800:>+9.4f}{g(on1, 'd'):>5}{g(r1, '.2f'):>6}"
                  f"{g(b1, 'd'):>5}{g(on4, 'd'):>5}{g(r4, '.2f'):>6}{g(b4, 'd'):>5}{fuel:>5}")
    con.close()


def build_configs(a: argparse.Namespace) -> tuple:
    ucfg = UniverseConfig(exclude=frozenset(x.strip().upper() for x in a.exclude.split(",") if x.strip()),
                          min_day_ntl_vlm=a.min_vol, max_day_ntl_vlm=a.max_vol, max_impact_spread_bps=a.max_spread_bps)
    scfg = SqueezeConfig(timeframes=tuple(x.strip() for x in a.tfs.split(",") if x.strip()))
    fcfg = FuelConfig(fuel_tf=scfg.timeframes[0])
    ecfg = EngineConfig()
    acct = AccountConfig(equity_usd=a.equity, risk_pct=a.risk_pct / 100.0, long_only=a.long_only)
    ccfg = ClientConfig(base_url=a.base_url.rstrip("/"), weight_per_minute=a.weight)
    return ucfg, scfg, fcfg, ecfg, acct, ccfg


def build_convex_hook(a: argparse.Namespace, acct: AccountConfig, client: HLInfoClient, store: Optional[Store],
                      telegram: "Optional[TelegramSink]") -> Optional[Any]:
    """ConvexSignalEngine + executor as a scanner hook. Returns None (and logs why) if live was asked for unsafely."""
    from convex_engine import (BreakoutConfig, ConvexExecutor, ConvexRunner, ConvexSignalEngine, ExecConfig,
                               HLGateway, OIHistory)
    from portfolio import PositionBook
    from risk_manager import ConvexRiskManager, RiskConfig
    oi = OIHistory()
    if store:
        oi.load(store.oi_history(now_ms() - 6 * HOUR_MS))
    bcfg = BreakoutConfig(allow_shorts=not acct.long_only)
    xcfg = ExecConfig(live=a.live, max_slippage=a.max_slippage_bps / 1e4, max_positions=a.max_positions)
    rcfg = RiskConfig(bootstrap_risk_frac=a.risk_pct / 100.0, max_risk_frac=a.max_risk_pct / 100.0)
    gateway, address = None, None
    if a.live:
        key = os.environ.get("HL_API_WALLET_KEY")
        master, sub = os.environ.get("HL_ACCOUNT_ADDRESS"), os.environ.get("HL_SUBACCOUNT_ADDRESS")
        if not key or not master:
            log.error("--live refused: HL_API_WALLET_KEY and HL_ACCOUNT_ADDRESS must be set in the environment")
            return None
        gateway = HLGateway(key, master, sub, a.base_url.rstrip("/"))
        address = sub or master
        log.warning("LIVE EXECUTION ENABLED — trading for %s via API wallet %s · max %d positions · risk %.2f%% bootstrap / "
                    "¼-Kelly ≤ %.1f%% per trade · stop 1.5×ATR14 · breaker %.0f%%/24h → flatten + %dh halt · "
                    "notional ≤ min(%.0fx equity, %.2f%% of 24h volume) · IOC bound %.0f bps · %s",
                    address, gateway.signer, xcfg.max_positions, rcfg.bootstrap_risk_frac * 100, rcfg.max_risk_frac * 100,
                    rcfg.breaker_dd * 100, int(rcfg.breaker_halt_s // 3600), acct.max_leverage,
                    acct.max_vlm_frac * 100, a.max_slippage_bps, "long only" if acct.long_only else "both sides")
    else:
        log.info("convex engine in DRY mode: paper fills at the IOC bound, exits by the risk manager, nothing is sent")
    notify = telegram.send if telegram else None
    risk = ConvexRiskManager(rcfg)
    book = PositionBook(risk, acct, client, store, gateway, address, live=a.live, notify=notify, limiter=client.limiter)
    executor = ConvexExecutor(xcfg, acct, client, store, gateway, address, notify=notify, limiter=client.limiter,
                              risk=risk, book=book)
    return ConvexRunner(ConvexSignalEngine(bcfg, oi, store), executor, book, store)


async def run(a: argparse.Namespace) -> None:
    ucfg, scfg, fcfg, ecfg, acct, ccfg = build_configs(a)
    store = Store(a.db) if a.db else None
    async with aiohttp.ClientSession() as session:
        client = HLInfoClient(ccfg, session, WeightLimiter(ccfg.weight_per_minute))
        sinks: list = [LogSink()]
        if store:
            sinks.append(store)
        telegram: Optional[TelegramSink] = None
        tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if tok and chat:
            telegram = TelegramSink(session, tok, chat)
            sinks.append(telegram)
            log.info("telegram sink enabled (FIRE, ARM, convex orders)")
        engine = StateEngine(ecfg, acct, sinks, fuel_tf=fcfg.fuel_tf,
                             restore_fires=store.last_fire_bars() if store else None)
        scanner = Scanner(client, CandleCache(client, scfg), FundingTracker(client, store), engine, store,
                          ucfg, scfg, fcfg, top=a.top, quiet=a.quiet,
                          coins=[c.strip().upper() for c in a.coins.split(",") if c.strip()] or None)
        if a.convex or a.live:
            runner = build_convex_hook(a, acct, client, store, telegram)
            if runner is None:
                if store:
                    store.close()
                return
            scanner.hooks.append(runner.on_cycle)
            scanner.extra_coins = runner.position_coins

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass

        while not stop.is_set():
            t0 = time.monotonic()
            try:
                await scanner.cycle()
            except HLError as e:
                log.error("cycle failed: %s", e)
            except Exception:
                log.exception("cycle crashed")
            if a.once:
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.0, a.interval - (time.monotonic() - t0)))
            except asyncio.TimeoutError:
                pass
    if store:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    views = {"events": a.events, "signals": a.signals, "orders": a.orders, "scans": a.top if a.scans else None,
             "positions": 0 if a.positions else None, "trades": a.trades, "wfo": 0 if a.wfo else None}
    wanted = [k for k, v in views.items() if v is not None]
    if wanted:
        if not a.db or not os.path.exists(a.db):
            print(f"no database at {a.db!r}", file=sys.stderr)
            return 1
        for k in wanted:
            show_db(a.db, k, views[k])
        return 0
    try:
        from dotenv import load_dotenv  # optional; real env vars work without it
        load_dotenv()
    except ImportError:
        pass
    try:
        asyncio.run(run(a))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
