#!/usr/bin/env python3
"""
s38-squeeze — PositionBook: applies ConvexRiskManager to every open position once per scanner cycle.

dry   fills are simulated at the IOC bound (pessimistic); stops are tested against each new closed bar's adverse
      extreme BEFORE that bar's favourable extreme is credited, then against the live mark; exits fill at the stop
      (or mark) minus 5 bps; equity = start + realized + unrealized. This is the paper book the Kelly stats come from.
live  positions come from clearinghouseState; when the ratchet moves the stop by ≥ 0.1 % the resting reduce-only
      stop is replaced — NEW order first, THEN cancel the old one, so the position is never unprotected; a position
      that disappears from the account is closed from its userFillsByTime fills (fallback: the stop price).
breaker  equity is sampled every cycle; drawdown ≥ 6 % from the rolling 24 h peak → flatten every position (IOC,
      1 % bound), cancel every open order on the account, halt entries 12 h, log + Telegram. Halt survives restarts.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence

import numpy as np

from convex_engine import IOC, parse_order_response, round_px
from risk_manager import CircuitBreaker, ConvexRiskManager, Position, TradeStats, stats_from_r_multiples
from squeeze_scanner import AccountConfig, sma, true_range, utc

log = logging.getLogger("s38.book")


class PositionBook:
    def __init__(self, risk: ConvexRiskManager, acct: AccountConfig, info: Any, store: Any = None,
                 gateway: Any = None, account_address: Optional[str] = None, live: bool = False,
                 notify: Optional[Callable[[str], Awaitable[None]]] = None, limiter: Any = None, tf: str = "1h"):
        self.risk, self.cfg, self.acct, self.info, self.store = risk, risk.cfg, acct, info, store
        self.gateway, self.address, self.live, self.notify, self.limiter, self.tf = gateway, account_address, live, notify, limiter, tf
        self.mode = "live" if live else "dry"
        self.positions: dict = {}
        self.realized_usd = 0.0
        self._r_multiples: list = []
        self.last_equity = acct.equity_usd
        self._warned: set = set()
        state = None
        if store:
            for p in store.open_positions(self.mode):
                self.positions[p.coin] = p
            self._r_multiples = store.trade_r_multiples(self.mode)
            self.realized_usd = store.realized_pnl(self.mode)
            state = store.breaker_state()
        self.breaker = CircuitBreaker(self.cfg, state)
        if self.positions:
            log.info("restored %d open %s position(s): %s", len(self.positions), self.mode, ", ".join(sorted(self.positions)))
        if state and state.halt_until_ms:
            log.warning("breaker state restored: halted until %s", utc(state.halt_until_ms))

    # ── queries ──
    def stats(self) -> TradeStats:
        return stats_from_r_multiples(self._r_multiples)

    def can_enter(self, t_ms: int) -> tuple:
        if self.breaker.halted(t_ms):
            return False, f"breaker halt until {utc(self.breaker.state.halt_until_ms)}"
        return True, ""

    def equity_dry(self, marks: dict) -> float:
        unreal = sum(p.pnl_usd(marks.get(p.coin, p.entry_px)) - p.fees_usd for p in self.positions.values())
        return self.acct.equity_usd + self.realized_usd + unreal

    # ── lifecycle ──
    def register(self, coin: str, direction: int, fill_px: float, size: float, atr: float, t_ms: int, entry_bar: int,
                 sz_decimals: int, stop_oid: Optional[int] = None, notional: float = 0.0) -> Position:
        r_unit = self.cfg.stop_atr_mult * atr
        stop = self.risk.initial_stop(fill_px, direction, atr)
        pos = Position(coin=coin, side="long" if direction > 0 else "short", direction=direction, entry_px=fill_px,
                       size=size, initial_stop=stop, stop=stop, r_unit=r_unit, best_px=fill_px, stage=0, opened_ms=t_ms,
                       entry_bar=entry_bar, last_bar_checked=entry_bar, stop_oid=stop_oid, mode=self.mode,
                       risk_usd=round(size * r_unit, 2), notional=round(notional or size * fill_px, 2),
                       fees_usd=round(size * fill_px * self.cfg.taker_fee, 4), sz_decimals=sz_decimals)
        self.positions[coin] = pos
        if self.store:
            self.store.upsert_position(pos)
        log.warning("BOOK open %s %s %g @ %g stop %g (R %g/unit, risk $%.2f) [%s]", pos.side, coin, size, fill_px,
                    stop, r_unit, pos.risk_usd, self.mode)
        return pos

    def _close(self, pos: Position, exit_px: float, t_ms: int, reason: str, exit_fee_usd: Optional[float] = None) -> dict:
        exit_fee = pos.size * exit_px * self.cfg.taker_fee if exit_fee_usd is None else exit_fee_usd
        fees = pos.fees_usd + exit_fee
        pnl = pos.pnl_usd(exit_px) - fees
        r = pnl / (pos.size * pos.r_unit) if pos.size * pos.r_unit > 0 else 0.0
        pos.status, pos.closed_ms, pos.fees_usd = "closed", t_ms, round(fees, 4)
        self.positions.pop(pos.coin, None)
        self.realized_usd += pnl
        self._r_multiples.append(r)
        trade = {"opened_ms": pos.opened_ms, "closed_ms": t_ms, "coin": pos.coin, "mode": self.mode, "side": pos.side,
                 "entry_px": pos.entry_px, "exit_px": exit_px, "size": pos.size, "r_unit": pos.r_unit,
                 "pnl_usd": round(pnl, 4), "r_multiple": round(r, 4), "fees_usd": round(fees, 4), "stage": pos.stage,
                 "reason": reason}
        if self.store:
            self.store.upsert_position(pos)
            self.store.write_trade(trade)
        log.warning("BOOK close %s %s @ %g: %+.2fR $%+.2f (%s) [%s]", pos.side, pos.coin, exit_px, r, pnl, reason, self.mode)
        return trade

    @staticmethod
    def _atr_by_bar(bars: Sequence, n: int) -> dict:
        if len(bars) < n + 1:
            return {}
        h = np.array([b.h for b in bars]); l = np.array([b.l for b in bars]); c = np.array([b.c for b in bars])
        atr = sma(true_range(h, l, c), n)
        return {b.t: float(a) for b, a in zip(bars, atr) if a == a}

    def _advance(self, pos: Position, bars: Sequence, mark: Optional[float], t_ms: int) -> Optional[dict]:
        """Walk new closed bars, then the live mark. Dry: may close the position. Returns the trade if it did."""
        atr_by_bar = self._atr_by_bar(bars, self.cfg.atr_len)
        atr = atr_by_bar.get(pos.last_bar_checked) or (list(atr_by_bar.values())[-1] if atr_by_bar else pos.r_unit / self.cfg.stop_atr_mult)
        for b in bars:
            if b.t <= pos.last_bar_checked:
                continue
            if not self.live and self.risk.stop_hit(pos, b.h, b.l):
                return self._close(pos, self.risk.paper_exit_px(pos.stop, pos.direction), t_ms, f"stop stage {pos.stage} (bar {utc(b.t)})")
            self.risk.update_excursion(pos, b.h, b.l)
            atr = atr_by_bar.get(b.t, atr)
            pos.stop, pos.stage = self.risk.desired_stop(pos, atr)
            pos.last_bar_checked = b.t
        if mark:
            if not self.live and self.risk.stop_hit(pos, mark, mark):
                return self._close(pos, self.risk.paper_exit_px(mark, pos.direction), t_ms, f"stop stage {pos.stage} (mark)")
            self.risk.update_excursion(pos, mark, mark)
            pos.stop, pos.stage = self.risk.desired_stop(pos, atr)
        return None

    async def _gw(self, fn, *args):
        if self.limiter:
            await self.limiter.acquire(1)
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def _replace_stop(self, pos: Position, new_stop: float) -> bool:
        """Live: place the new reduce-only stop first, cancel the old one after. Never leaves the position naked."""
        sd = pos.sz_decimals
        trig = round_px(new_stop, sd)
        limit = round_px(trig * (1 - 0.05), sd, "down") if pos.direction > 0 else round_px(trig * (1 + 0.05), sd, "up")
        order_type = {"trigger": {"triggerPx": trig, "isMarket": True, "tpsl": "sl"}}
        try:
            resp = await self._gw(self.gateway.order, pos.coin, pos.direction < 0, pos.size, limit, order_type, True)
            status, _, _, oid, err = parse_order_response(resp)
            if oid is None:
                raise RuntimeError(f"{status} {err}")
        except Exception as e:
            log.error("stop replace %s failed, keeping old stop %s: %s", pos.coin, pos.stop_oid, e)
            return False
        old = pos.stop_oid
        pos.stop_oid, pos.stop = oid, trig
        if old:
            try:
                await self._gw(self.gateway.cancel, pos.coin, old)
            except Exception as e:
                log.warning("cancel old stop %s/%s: %s", pos.coin, old, e)
        log.warning("STOP %s moved → %g (stage %d, oid %s)", pos.coin, trig, pos.stage, oid)
        return True

    async def _close_live(self, pos: Position, t_ms: int, reason: str) -> dict:
        """The account no longer shows the position: price it from the closing fills, else assume the stop."""
        exit_px, fees = pos.stop, None
        try:
            fills = await self.info.user_fills_by_time(self.address, pos.last_bar_checked or pos.opened_ms)
            closing = [f for f in fills if f.get("coin") == pos.coin and
                       ((f.get("side") == "A") if pos.direction > 0 else (f.get("side") == "B"))]
            qty = sum(float(f["sz"]) for f in closing)
            if qty > 0:
                exit_px = sum(float(f["px"]) * float(f["sz"]) for f in closing) / qty
                fees = sum(float(f.get("fee", 0) or 0) for f in closing)
        except Exception as e:
            log.warning("userFillsByTime %s: %s — closing at the stop price", pos.coin, e)
        if pos.stop_oid:
            try:
                await self._gw(self.gateway.cancel, pos.coin, pos.stop_oid)   # harmless if the stop already filled
            except Exception:
                pass
        return self._close(pos, exit_px, t_ms, reason + (" (fills)" if fees is not None else " (assumed stop)"), fees)

    async def _flatten(self, pos: Position, mark: float, t_ms: int) -> Optional[dict]:
        if not self.live:
            return self._close(pos, self.risk.paper_exit_px(mark, pos.direction), t_ms, "breaker")
        sd = pos.sz_decimals
        bound = round_px(mark * (1 - self.cfg.flatten_slippage), sd, "down") if pos.direction > 0 else \
            round_px(mark * (1 + self.cfg.flatten_slippage), sd, "up")
        try:
            resp = await self._gw(self.gateway.order, pos.coin, pos.direction < 0, pos.size, bound, IOC, True)
            status, filled, avg, oid, err = parse_order_response(resp)
            if filled <= 0:
                raise RuntimeError(f"{status} {err}")
            if pos.stop_oid:
                try:
                    await self._gw(self.gateway.cancel, pos.coin, pos.stop_oid)
                except Exception:
                    pass
            return self._close(pos, avg, t_ms, f"breaker flatten (oid {oid})")
        except Exception as e:
            log.critical("BREAKER could not flatten %s %s: %s — stop %s stays in place", pos.side, pos.coin, e, pos.stop_oid)
            return None

    async def _trip(self, trip: dict, marks: dict, t_ms: int) -> list:
        msg = (f"CIRCUIT BREAKER: equity ${trip['equity']:,.2f} is {trip['drawdown']:.1%} below the 24h peak "
               f"${trip['peak']:,.2f} — flattening {len(self.positions)} position(s), cancelling orders, "
               f"no entries until {utc(trip['halt_until_ms'])}")
        log.critical(msg)
        trades = []
        for pos in list(self.positions.values()):
            tr = await self._flatten(pos, marks.get(pos.coin, pos.entry_px), t_ms)
            if tr:
                trades.append(tr)
        if self.live:
            try:
                for o in await self.info.open_orders(self.address):
                    try:
                        await self._gw(self.gateway.cancel, o["coin"], int(o["oid"]))
                    except Exception as e:
                        log.warning("cancel %s/%s: %s", o.get("coin"), o.get("oid"), e)
            except Exception as e:
                log.warning("openOrders: %s", e)
        if self.store:
            self.store.write_breaker(trip)
        if self.notify:
            try:
                await self.notify(msg)
            except Exception:
                log.exception("notify failed")
        return trades

    async def on_cycle(self, candles: dict, t_ms: int, marks: Optional[dict] = None) -> dict:
        """Manage every open position, sample equity, run the breaker. Returns a small summary."""
        acct_pos: dict = {}
        equity = self.last_equity
        if self.live:
            st = await self.info.user_state(self.address)
            equity = float(st["marginSummary"]["accountValue"])
            for ap in st.get("assetPositions", []):
                p = ap.get("position", {})
                szi = float(p.get("szi", 0) or 0)
                if szi:
                    acct_pos[p["coin"]] = szi
            for coin in acct_pos:
                if coin not in self.positions and coin not in self._warned:
                    self._warned.add(coin)
                    log.warning("account holds %s %+g that this book did not open — left alone", coin, acct_pos[coin])
        if marks is None:
            marks = await self.info.all_mids() if self.positions else {}
        closed: list = []
        moved = 0
        for coin, pos in list(self.positions.items()):
            bars = candles.get((coin, self.tf), [])
            mark = marks.get(coin)
            if self.live and coin not in acct_pos:
                closed.append(await self._close_live(pos, t_ms, "closed on exchange"))
                continue
            if self.live and abs(acct_pos[coin]) < pos.size - 1e-9:
                log.warning("%s size on exchange %g < book %g — partial close outside the book, size updated", coin, abs(acct_pos[coin]), pos.size)
                pos.size = abs(acct_pos[coin])
            old_stop = pos.stop
            trade = self._advance(pos, bars, mark, t_ms)
            if trade:
                closed.append(trade)
                continue
            if self.live and self.risk.stop_moved_enough(old_stop, pos.stop):
                new = pos.stop
                pos.stop = old_stop                      # only adopt the new level once the exchange holds it
                if await self._replace_stop(pos, new):
                    moved += 1
            if self.store:
                self.store.upsert_position(pos)
        if not self.live:
            equity = self.equity_dry(marks)
        trip = self.breaker.sample(t_ms, equity)
        if trip:
            closed.extend(await self._trip(trip, marks, t_ms))
        self.last_equity = equity
        return {"equity": equity, "open": len(self.positions), "closed": closed, "stops_moved": moved,
                "drawdown": self.breaker.drawdown(equity), "halted": self.breaker.halted(t_ms), "tripped": bool(trip)}


def format_book(summary: dict, mode: str) -> str:
    s = summary
    txt = f"book[{mode}] equity ${s['equity']:,.2f} dd {s['drawdown']:.1%} open {s['open']} stops moved {s['stops_moved']}"
    if s["closed"]:
        txt += " · closed " + ", ".join(f"{t['coin']} {t['r_multiple']:+.2f}R" for t in s["closed"])
    if s["halted"]:
        txt += " · HALTED"
    return txt


__all__ = ["PositionBook", "format_book"]
