#!/usr/bin/env python3
"""
s38-squeeze — ConvexSignalEngine + IOC executor: enter only when compression turns into expansion.

A ConvexSignal needs ALL of the following, evaluated once per newest CLOSED 1h bar:
  1. compression   the 1h BB/KC squeeze held ≥ min_squeeze_bars and ended ≤ max_bars_since_squeeze bars ago
                   (or is still on)
  2. breakout      bar close above the prior-20-bar Donchian high → long; below the prior-20-bar low → short
  3. volume        bar volume ≥ mean + 2.5·σ of the prior 20 bars (population σ) — no low-volume breakouts
  4. OI inflow     open interest in COIN UNITS at bar close ≥ (1 + 3%) × OI at bar open: fresh capital, not a
                   notional artefact of the price move and not passive liquidations
  5. freshness     evaluated within max_signal_age_s of the bar close (a restart cannot fire on an old bar)

ConvexExecutor turns a signal into one IOC limit order at fresh mid × (1 ± 0.08 %) — Hyperliquid's "market"
order IS a slippage-bounded IOC — keeps partial fills, never chases, then rests a reduce-only stop-market at the
signal's stop. Dry-run unless ExecConfig.live, which the CLI only enables with --live AND HL_API_WALLET_KEY set.
Kill switch: a STOP file in the working directory blocks new entries.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

from squeeze_scanner import HOUR_MS, AccountConfig, AssetCtx, Candle, ScanRow, sma, true_range, utc

log = logging.getLogger("s38.convex")

IOC: dict = {"limit": {"tif": "Ioc"}}


# ───────────────────────────── configuration ──────────────────────────────


@dataclass(frozen=True)
class BreakoutConfig:
    tf: str = "1h"
    donchian_len: int = 20          # prior-N highs/lows; the breakout bar itself is excluded
    vol_len: int = 20
    vol_z_min: float = 2.5
    atr_len: int = 20
    oi_min_change: float = 0.03     # OI(bar close) / OI(bar open) − 1
    oi_tolerance_s: float = 300.0   # an OI sample must sit within ±5 min of the bar open
    require_squeeze: bool = True
    min_squeeze_bars: int = 3
    max_bars_since_squeeze: int = 2
    allow_shorts: bool = True
    stop_atr_min: float = 1.0       # stop = breakout-bar extreme, at least this many ATR away
    max_signal_age_s: float = 900.0 # bar must have closed less than this long ago (a full candle refresh at
                                    # 400 weight/min takes ~4–7 min; a restart must not fire on an old bar)


@dataclass(frozen=True)
class ExecConfig:
    live: bool = False
    max_slippage: float = 0.0008    # 0.08 % — IOC bound off the fresh mid
    max_positions: int = 3
    cooldown_bars: int = 24         # per coin after a fill
    cooldown_bars_unfilled: int = 1 # per coin after an unfilled / failed attempt
    leverage: int = 5               # cross leverage requested per coin before its first order
    stop_limit_slip: float = 0.05   # stop-market worst price = trigger × (1 ∓ 5 %)
    stop_file: str = "STOP"


# ───────────────────────────── signal generation ──────────────────────────


@dataclass(frozen=True)
class BreakoutMetrics:
    bar_time: int
    close: float
    high: float
    low: float
    volume: float
    donchian_upper: float       # max high of the prior donchian_len bars
    donchian_lower: float       # min low of the prior donchian_len bars
    direction: int              # +1 close > upper, −1 close < lower, 0 inside
    vol_mean: float
    vol_std: float
    vol_z: float                # NaN when the prior volumes have zero dispersion
    atr: float


def compute_breakout(candles: Sequence[Candle], cfg: BreakoutConfig) -> Optional[BreakoutMetrics]:
    n, m = cfg.donchian_len, cfg.vol_len
    if len(candles) < max(n, m, cfg.atr_len) + 1:
        return None
    h = np.array([c.h for c in candles], dtype=float)
    l = np.array([c.l for c in candles], dtype=float)
    c = np.array([c.c for c in candles], dtype=float)
    v = np.array([c.v for c in candles], dtype=float)
    i = len(c) - 1
    upper, lower = float(h[i - n:i].max()), float(l[i - n:i].min())
    direction = 1 if c[i] > upper else (-1 if c[i] < lower else 0)
    vp = v[i - m:i]
    mean, std = float(vp.mean()), float(vp.std())
    z = (float(v[i]) - mean) / std if std > 0 else float("nan")
    atr = float(sma(true_range(h, l, c), cfg.atr_len)[i])
    return BreakoutMetrics(bar_time=candles[i].t, close=float(c[i]), high=float(h[i]), low=float(l[i]),
                           volume=float(v[i]), donchian_upper=upper, donchian_lower=lower, direction=direction,
                           vol_mean=mean, vol_std=std, vol_z=z, atr=atr)


class OIHistory:
    """Open-interest samples (coin units) per coin, one per scanner cycle, kept for a few hours."""

    def __init__(self, keep_ms: int = 6 * HOUR_MS):
        self.keep_ms = keep_ms
        self._h: dict = {}

    def record(self, coin: str, ts: int, oi: float) -> None:
        d = self._h.setdefault(coin, deque())
        if d and d[-1][0] >= ts:
            return
        d.append((ts, oi))
        while d and d[0][0] < ts - self.keep_ms:
            d.popleft()

    def load(self, samples: Iterable) -> None:
        for ts, coin, oi in sorted(samples, key=lambda s: s[0]):
            if oi is not None:
                self.record(coin, int(ts), float(oi))

    def at(self, coin: str, ts: int, tol_ms: int) -> Optional[float]:
        d = self._h.get(coin)
        if not d:
            return None
        t, oi = min(d, key=lambda p: abs(p[0] - ts))
        return oi if abs(t - ts) <= tol_ms else None


@dataclass(frozen=True)
class ConvexSignal:
    ts: int
    coin: str
    side: str                   # "long" | "short"
    direction: int
    bar_time: int
    ref_px: float               # mark at evaluation; the executor re-prices off a fresh mid
    close: float
    high: float
    low: float
    donchian_upper: float
    donchian_lower: float
    vol_z: float
    oi_open: float
    oi_close: float
    oi_change: float
    squeeze_run: int
    bars_since_squeeze: int
    atr: float
    stop_px: float


@dataclass(frozen=True)
class Rejection:
    coin: str
    bar_time: int
    direction: int
    reasons: tuple


class ConvexSignalEngine:
    """Evaluates every eligible coin's newest closed bar once; emits ConvexSignals, records rejections."""

    def __init__(self, cfg: BreakoutConfig, oi: OIHistory, store: Any = None):
        self.cfg = cfg
        self.oi = oi
        self.store = store
        self.seen: set = set(store.signal_keys()) if store else set()
        self.last: dict = {}            # coin -> BreakoutMetrics (for display)
        self.rejections: list = []      # last cycle

    def evaluate(self, rows: Sequence[ScanRow], candles: dict, t_ms: int) -> list:
        cfg = self.cfg
        signals: list = []
        self.rejections = []
        for row in rows:
            coin = row.ctx.coin
            self.oi.record(coin, t_ms, row.ctx.open_interest)
            bars = candles.get((coin, cfg.tf))
            bm = compute_breakout(bars, cfg) if bars else None
            if bm is None:
                continue
            self.last[coin] = bm
            if bm.direction == 0 or (coin, bm.bar_time) in self.seen:
                continue
            self.seen.add((coin, bm.bar_time))

            reasons: list = []
            if bm.direction < 0 and not cfg.allow_shorts:
                reasons.append("shorts_disabled")
            age_s = (t_ms - (bm.bar_time + HOUR_MS)) / 1000
            if age_s > cfg.max_signal_age_s:
                reasons.append(f"stale({age_s:.0f}s after close)")
            sq = row.squeeze.get(cfg.tf)
            run = sq.last_squeeze_run if sq else 0
            ago = sq.bars_since_squeeze if sq else -1
            if cfg.require_squeeze and (ago < 0 or ago > cfg.max_bars_since_squeeze or run < cfg.min_squeeze_bars):
                reasons.append(f"no_squeeze(run={run},ago={ago})")
            if not bm.vol_z >= cfg.vol_z_min:       # NaN fails too
                reasons.append(f"vol_z={bm.vol_z:.2f}<{cfg.vol_z_min}")
            tol = int(cfg.oi_tolerance_s * 1000)
            oi_open = self.oi.at(coin, bm.bar_time, tol)
            oi_close = self.oi.at(coin, bm.bar_time + HOUR_MS, tol)
            if oi_close is None:
                oi_close = row.ctx.open_interest
            if oi_open is None or oi_open <= 0:
                reasons.append("oi_history_insufficient")
                oi_change = float("nan")
            else:
                oi_change = oi_close / oi_open - 1.0
                if oi_change < cfg.oi_min_change:
                    reasons.append(f"oi_change={oi_change:+.2%}<{cfg.oi_min_change:.0%}")

            ref = row.ctx.mark_px
            if bm.direction > 0:
                stop = min(bm.low, ref - cfg.stop_atr_min * bm.atr)
            else:
                stop = max(bm.high, ref + cfg.stop_atr_min * bm.atr)
            sig = ConvexSignal(ts=t_ms, coin=coin, side="long" if bm.direction > 0 else "short", direction=bm.direction,
                               bar_time=bm.bar_time, ref_px=ref, close=bm.close, high=bm.high, low=bm.low,
                               donchian_upper=bm.donchian_upper, donchian_lower=bm.donchian_lower, vol_z=bm.vol_z,
                               oi_open=oi_open or float("nan"), oi_close=oi_close, oi_change=oi_change,
                               squeeze_run=run, bars_since_squeeze=ago, atr=bm.atr, stop_px=stop)
            if self.store:
                self.store.write_signal(sig, not reasons, reasons)
            if reasons:
                self.rejections.append(Rejection(coin, bm.bar_time, bm.direction, tuple(reasons)))
                log.info("convex reject %s %s bar %s: %s", coin, sig.side, utc(bm.bar_time), "; ".join(reasons))
            else:
                signals.append(sig)
                log.warning("CONVEX %s %s bar %s close %g (donchian %g/%g) vol_z %.1f OI %+.1f%% squeeze %d bars "
                            "ended %d ago stop %g", coin, sig.side.upper(), utc(bm.bar_time), bm.close,
                            bm.donchian_upper, bm.donchian_lower, bm.vol_z, oi_change * 100, run, ago, stop)
        return signals


# ─────────────────────────────── execution ────────────────────────────────


def round_px(px: float, sz_decimals: int, mode: str = "nearest") -> float:
    """Hyperliquid perp tick rule: ≤ 5 significant figures and ≤ (6 − szDecimals) decimals.
    mode 'down'/'up' rounds toward a stricter bound (buy bound down, sell bound up)."""
    if not (px > 0) or not math.isfinite(px):
        raise ValueError(f"bad price {px}")
    d = min(max(0, 6 - sz_decimals), 4 - math.floor(math.log10(px)))
    rounding = {"down": ROUND_FLOOR, "up": ROUND_CEILING}.get(mode, ROUND_HALF_EVEN)
    return float(Decimal(str(px)).quantize(Decimal(1).scaleb(-d), rounding=rounding))


def floor_sz(sz: float, sz_decimals: int) -> float:
    return float(Decimal(str(sz)).quantize(Decimal(1).scaleb(-sz_decimals), rounding=ROUND_FLOOR))


@dataclass(frozen=True)
class OrderPlan:
    coin: str
    side: str
    is_buy: bool
    sz: float
    bound_px: float             # IOC limit: ref × (1 ± max_slippage), rounded toward ref
    ref_px: float               # fresh mid at send time
    notional: float
    stop_px: float              # trigger
    stop_limit_px: float        # worst acceptable fill once triggered
    leverage: int
    is_cross: bool
    risk_usd: float
    equity: float


@dataclass
class ExecutionReport:
    ts: int
    coin: str
    side: str
    mode: str                   # dry | live
    status: str                 # dry | filled | partial | unfilled | error | skipped
    sz: float = 0.0
    bound_px: float = 0.0
    ref_px: float = 0.0
    filled_sz: float = 0.0
    avg_px: Optional[float] = None
    slippage_bps: Optional[float] = None
    notional: float = 0.0
    stop_px: Optional[float] = None
    oid: Optional[int] = None
    stop_oid: Optional[int] = None
    error: Optional[str] = None
    raw: Optional[str] = None


class Gateway(Protocol):
    """Signs and sends exchange requests. HLGateway wraps the official SDK; tests inject a fake."""
    def order(self, coin: str, is_buy: bool, sz: float, limit_px: float, order_type: dict, reduce_only: bool = False) -> Any: ...
    def update_leverage(self, leverage: int, coin: str, is_cross: bool = True) -> Any: ...


class HLGateway:
    def __init__(self, key: str, account_address: Optional[str], vault_address: Optional[str], base_url: str):
        from eth_account import Account                 # imported here so the scanner runs without the SDK
        from hyperliquid.exchange import Exchange
        wallet = Account.from_key(key)
        self.signer = wallet.address                    # public; the key lives only inside the wallet object
        self.trading_for = vault_address or account_address or wallet.address
        self.exchange = Exchange(wallet, base_url, vault_address=vault_address, account_address=account_address)

    def order(self, coin, is_buy, sz, limit_px, order_type, reduce_only=False):
        return self.exchange.order(coin, is_buy, sz, limit_px, order_type, reduce_only=reduce_only)

    def update_leverage(self, leverage, coin, is_cross=True):
        return self.exchange.update_leverage(leverage, coin, is_cross)


def parse_order_response(resp: Any) -> tuple:
    """→ (status, filled_sz, avg_px, oid, error) with status ∈ filled | resting | unfilled | error."""
    try:
        if resp.get("status") != "ok":
            return "error", 0.0, None, None, str(resp.get("response", resp))[:300]
        st = resp["response"]["data"]["statuses"][0]
    except (KeyError, IndexError, TypeError, AttributeError):
        return "error", 0.0, None, None, f"unexpected response: {str(resp)[:300]}"
    if "filled" in st:
        f = st["filled"]
        return "filled", float(f["totalSz"]), float(f["avgPx"]), int(f.get("oid") or 0) or None, None
    if "resting" in st:
        return "resting", 0.0, None, int(st["resting"]["oid"]), None
    err = str(st.get("error", st))
    return ("unfilled" if "immediately match" in err else "error"), 0.0, None, None, err


class ConvexExecutor:
    """One IOC entry per signal under position/cooldown/kill-switch limits, then a resting reduce-only stop."""

    def __init__(self, cfg: ExecConfig, acct: AccountConfig, info: Any, store: Any = None,
                 gateway: Optional[Gateway] = None, account_address: Optional[str] = None,
                 notify: Optional[Callable[[str], Awaitable[None]]] = None, limiter: Any = None):
        if cfg.live and gateway is None:
            raise ValueError("live execution needs a gateway")
        self.cfg, self.acct, self.info, self.store = cfg, acct, info, store
        self.gateway, self.account_address, self.notify, self.limiter = gateway, account_address, notify, limiter
        self._cooldown: dict = {}       # coin -> until ms
        self._dry_positions: dict = {}  # coin -> opened ms (dry mode stand-in for the account state)
        self._lev_set: set = set()
        self.reports: list = []

    # ── account state ──
    async def _account(self, t_ms: int) -> tuple:
        if self.cfg.live and self.account_address:
            st = await self.info.user_state(self.account_address)
            equity = float(st["marginSummary"]["accountValue"])
            positions = {}
            for ap in st.get("assetPositions", []):
                p = ap.get("position", {})
                szi = float(p.get("szi", 0) or 0)
                if szi:
                    positions[p["coin"]] = szi
            return equity, positions
        horizon = self.cfg.cooldown_bars * HOUR_MS
        self._dry_positions = {c: t for c, t in self._dry_positions.items() if t_ms - t < horizon}
        return self.acct.equity_usd, dict(self._dry_positions)

    # ── sizing ──
    def plan(self, s: ConvexSignal, ctx: AssetCtx, equity: float, ref_px: float) -> Optional[OrderPlan]:
        is_buy = s.direction > 0
        dist = (ref_px - s.stop_px) if is_buy else (s.stop_px - ref_px)
        if dist <= 0 or not math.isfinite(dist):
            return None
        risk_usd = equity * self.acct.risk_pct
        sz = risk_usd / dist
        cap = min(equity * self.acct.max_leverage, ctx.day_ntl_vlm * self.acct.max_vlm_frac)
        sz = floor_sz(min(sz, cap / ref_px), ctx.sz_decimals)
        notional = sz * ref_px
        if notional < self.acct.min_notional_usd:
            return None
        sd = ctx.sz_decimals
        bound = round_px(ref_px * (1 + self.cfg.max_slippage), sd, "down") if is_buy else \
            round_px(ref_px * (1 - self.cfg.max_slippage), sd, "up")
        stop = round_px(s.stop_px, sd)
        stop_limit = round_px(stop * (1 - self.cfg.stop_limit_slip), sd, "down") if is_buy else \
            round_px(stop * (1 + self.cfg.stop_limit_slip), sd, "up")
        lev = max(1, min(ctx.max_leverage, self.cfg.leverage))
        return OrderPlan(coin=s.coin, side=s.side, is_buy=is_buy, sz=sz, bound_px=bound, ref_px=ref_px,
                         notional=round(notional, 2), stop_px=stop, stop_limit_px=stop_limit, leverage=lev,
                         is_cross=not ctx.only_isolated, risk_usd=round(sz * dist, 2), equity=equity)

    # ── main entry ──
    async def handle(self, signals: Sequence[ConvexSignal], rows_by_coin: dict, t_ms: int) -> list:
        reports: list = []
        if not signals:
            return reports
        if os.path.exists(self.cfg.stop_file):
            log.warning("kill switch %s present — %d signal(s) not executed", self.cfg.stop_file, len(signals))
            reports = [self._skip(s, t_ms, "kill_switch") for s in signals]
        else:
            equity, positions = await self._account(t_ms)
            mids = await self.info.all_mids()
            opened = 0
            for s in signals:
                ctx = rows_by_coin[s.coin].ctx
                ref = float(mids.get(s.coin) or ctx.mid_px or ctx.mark_px)
                if s.coin in positions:
                    rep = self._skip(s, t_ms, "in_position")
                elif self._cooldown.get(s.coin, 0) > t_ms:
                    rep = self._skip(s, t_ms, f"cooldown until {utc(self._cooldown[s.coin])}")
                elif len(positions) + opened >= self.cfg.max_positions:
                    rep = self._skip(s, t_ms, f"max_positions={self.cfg.max_positions}")
                elif (s.direction > 0 and ref <= s.donchian_upper) or (s.direction < 0 and ref >= s.donchian_lower):
                    rep = self._skip(s, t_ms, f"ref {ref:g} back inside the range")
                else:
                    plan = self.plan(s, ctx, equity, ref)
                    if plan is None:
                        rep = self._skip(s, t_ms, "too_small")
                    else:
                        rep = await self._send(plan, s, t_ms)
                        if rep.status in ("dry", "filled", "partial"):
                            opened += 1
                reports.append(rep)
        for rep in reports:
            self.reports.append(rep)
            if self.store:
                self.store.write_order(rep)
            if self.notify and rep.status != "skipped":
                try:
                    await self.notify(format_report(rep))
                except Exception:
                    log.exception("notify failed")
        return reports

    def _skip(self, s: ConvexSignal, t_ms: int, why: str) -> ExecutionReport:
        log.info("convex skip %s %s: %s", s.coin, s.side, why)
        return ExecutionReport(ts=t_ms, coin=s.coin, side=s.side, mode="live" if self.cfg.live else "dry",
                               status="skipped", error=why, stop_px=s.stop_px)

    async def _send(self, plan: OrderPlan, s: ConvexSignal, t_ms: int) -> ExecutionReport:
        rep = ExecutionReport(ts=t_ms, coin=plan.coin, side=plan.side, mode="live" if self.cfg.live else "dry",
                              status="dry", sz=plan.sz, bound_px=plan.bound_px, ref_px=plan.ref_px,
                              notional=plan.notional, stop_px=plan.stop_px)
        payload = {"coin": plan.coin, "is_buy": plan.is_buy, "sz": plan.sz, "limit_px": plan.bound_px,
                   "order_type": IOC, "reduce_only": False, "leverage": plan.leverage, "cross": plan.is_cross,
                   "stop": {"triggerPx": plan.stop_px, "limit_px": plan.stop_limit_px, "isMarket": True, "tpsl": "sl"}}
        rep.raw = json.dumps(payload)
        if not self.cfg.live:
            self._dry_positions[plan.coin] = t_ms
            self._cooldown[plan.coin] = t_ms + self.cfg.cooldown_bars * HOUR_MS
            log.warning("DRY %s %s %g @ ≤%g (ref %g) notional $%.0f risk $%.2f stop %g — would send %s",
                        plan.side.upper(), plan.coin, plan.sz, plan.bound_px, plan.ref_px, plan.notional,
                        plan.risk_usd, plan.stop_px, rep.raw)
            return rep

        loop = asyncio.get_running_loop()
        gw = self.gateway
        if self.limiter:
            await self.limiter.acquire(2)
        if plan.coin not in self._lev_set:
            try:
                r = await loop.run_in_executor(None, gw.update_leverage, plan.leverage, plan.coin, plan.is_cross)
                log.info("leverage %s -> %dx (%s): %s", plan.coin, plan.leverage, "cross" if plan.is_cross else "isolated", str(r)[:120])
            except Exception as e:
                log.warning("update_leverage %s failed: %s", plan.coin, e)
            self._lev_set.add(plan.coin)
        try:
            resp = await loop.run_in_executor(None, gw.order, plan.coin, plan.is_buy, plan.sz, plan.bound_px, IOC, False)
        except Exception as e:
            rep.status, rep.error = "error", f"order exception: {e}"
            self._cooldown[plan.coin] = t_ms + self.cfg.cooldown_bars_unfilled * HOUR_MS
            log.error("ORDER %s %s failed: %s", plan.side, plan.coin, e)
            return rep
        status, filled, avg, oid, err = parse_order_response(resp)
        rep.raw = json.dumps({"sent": payload, "resp": resp}, default=str)[:4000]
        rep.oid, rep.error = oid, err
        if filled > 0:
            rep.filled_sz, rep.avg_px = filled, avg
            rep.slippage_bps = (avg / plan.ref_px - 1.0) * 1e4 * (1 if plan.is_buy else -1)
            rep.status = "filled" if filled >= plan.sz - 1e-12 else "partial"
            self._cooldown[plan.coin] = t_ms + self.cfg.cooldown_bars * HOUR_MS
            log.warning("FILLED %s %s %g/%g @ %g (slip %+.1f bps, oid %s)", plan.side.upper(), plan.coin, filled,
                        plan.sz, avg, rep.slippage_bps, oid)
            stop_type = {"trigger": {"triggerPx": plan.stop_px, "isMarket": True, "tpsl": "sl"}}
            try:
                sresp = await loop.run_in_executor(None, gw.order, plan.coin, not plan.is_buy, filled,
                                                   plan.stop_limit_px, stop_type, True)
                sstatus, _, _, soid, serr = parse_order_response(sresp)
                rep.stop_oid = soid
                if soid is None:
                    raise RuntimeError(f"stop not accepted: {sstatus} {serr}")
                log.warning("STOP %s %s %g @ %g (oid %s)", plan.coin, "sell" if plan.is_buy else "buy", filled, plan.stop_px, soid)
            except Exception as e:
                rep.error = f"POSITION UNPROTECTED — stop failed: {e}"
                log.critical("%s %s: %s", plan.coin, plan.side, rep.error)
        else:
            rep.status = status if status in ("unfilled", "error") else "error"
            self._cooldown[plan.coin] = t_ms + self.cfg.cooldown_bars_unfilled * HOUR_MS
            log.warning("%s %s %s: %s", rep.status.upper(), plan.side, plan.coin, err)
        return rep


def format_report(rep: ExecutionReport) -> str:
    head = f"s38 {rep.mode.upper()} {rep.status.upper()} {rep.side} {rep.coin}"
    if rep.status == "skipped":
        return f"{head}: {rep.error}"
    parts = [head, f"sz {rep.sz:g} bound {rep.bound_px:g} ref {rep.ref_px:g} notional ${rep.notional:,.0f}"]
    if rep.filled_sz:
        parts.append(f"filled {rep.filled_sz:g} @ {rep.avg_px:g} slip {rep.slippage_bps:+.1f}bps oid {rep.oid}")
    if rep.stop_px:
        parts.append(f"stop {rep.stop_px:g}" + (f" oid {rep.stop_oid}" if rep.stop_oid else ""))
    if rep.error:
        parts.append(rep.error)
    return " | ".join(parts)


class ConvexRunner:
    """Scanner cycle hook: evaluate → execute → log one summary line."""

    def __init__(self, engine: ConvexSignalEngine, executor: ConvexExecutor):
        self.engine, self.executor = engine, executor

    async def on_cycle(self, rows: Sequence[ScanRow], candles: dict, t_ms: int) -> None:
        signals = self.engine.evaluate(rows, candles, t_ms)
        n_new = len(signals) + len(self.engine.rejections)
        reports = await self.executor.handle(signals, {r.ctx.coin: r for r in rows}, t_ms) if signals else []
        if n_new:
            log.info("convex: %d new breakout bar(s) → %d signal(s), %d rejected, %d order report(s)",
                     n_new, len(signals), len(self.engine.rejections), len(reports))
