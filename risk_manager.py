#!/usr/bin/env python3
"""
s38-squeeze — ConvexRiskManager: sizing, invalidation stop, ratcheting Chandelier exit, portfolio circuit breaker.

Pure logic, no I/O. portfolio.PositionBook applies it to paper (dry) and live positions every scanner cycle.

Sizing      quarter-Kelly on REALIZED trade statistics, win rate taken 1σ pessimistic, capped at 4 % of equity per
            trade on the initial stop distance; 1 % bootstrap until `min_trades` trades exist; zero when the
            realized edge is not positive (no trade).
Stop        initial invalidation at 1.5 × ATR(14) from the fill. R = that distance.
Exit        best excursion ≥ +2R → stop to entry + 0.1R (break-even after taker fees);
            best excursion ≥ +3R → Chandelier: 2.5 × ATR(14) behind the best excursion, recomputed every cycle;
            the stop only ever moves in the trade's favour.
Breaker     mark-to-market equity drawdown from the rolling 24 h peak ≥ 6 % → flatten, cancel, halt entries 12 h.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = 0.25
    max_risk_frac: float = 0.04        # cap: equity fraction risked per trade on the initial stop distance
    bootstrap_risk_frac: float = 0.01  # used until min_trades realized trades exist
    min_trades: int = 20
    win_rate_z: float = 1.0            # size on win rate − z·SE (pessimistic)
    atr_len: int = 14
    stop_atr_mult: float = 1.5         # initial invalidation stop
    be_trigger_R: float = 2.0          # best excursion that moves the stop to break-even
    be_offset_R: float = 0.1           # break-even offset covering taker fees in and out
    trail_trigger_R: float = 3.0       # best excursion that switches to the Chandelier trail
    trail_atr_mult: float = 2.5        # Chandelier distance behind the best excursion
    min_stop_move_frac: float = 0.001  # replace a resting stop only for ≥ 0.1 % moves (order churn)
    breaker_dd: float = 0.06
    breaker_window_s: float = 24 * 3600.0
    breaker_halt_s: float = 12 * 3600.0
    taker_fee: float = 0.00045         # per side, HL base tier
    paper_slippage: float = 0.0005     # dry exits: 5 bps adverse to the stop/mark
    flatten_slippage: float = 0.01     # IOC bound for breaker flattening


@dataclass(frozen=True)
class TradeStats:
    n: int = 0
    wins: int = 0
    avg_win_R: float = 0.0            # mean R of winning trades (> 0)
    avg_loss_R: float = 0.0           # mean |R| of losing trades (> 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def payoff(self) -> float:
        """Win/loss ratio; inf when no loss has been realized yet."""
        if self.avg_loss_R <= 0:
            return math.inf if self.avg_win_R > 0 else 0.0
        return self.avg_win_R / self.avg_loss_R


@dataclass
class Position:
    coin: str
    side: str                   # long | short
    direction: int
    entry_px: float
    size: float
    initial_stop: float
    stop: float
    r_unit: float               # initial risk per unit = stop_atr_mult × ATR at entry
    best_px: float              # best excursion since entry (highest high for longs, lowest low for shorts)
    stage: int = 0              # 0 initial · 1 break-even · 2 chandelier
    opened_ms: int = 0
    entry_bar: int = 0          # open time of the bar in which the entry happened
    last_bar_checked: int = 0
    stop_oid: Optional[int] = None
    mode: str = "dry"
    risk_usd: float = 0.0
    notional: float = 0.0
    fees_usd: float = 0.0
    sz_decimals: int = 0
    status: str = "open"
    closed_ms: int = 0
    ref_px: float = 0.0         # reference mid at send time (entry slippage = fill vs this)

    @property
    def excursion_R(self) -> float:
        return self.direction * (self.best_px - self.entry_px) / self.r_unit if self.r_unit > 0 else 0.0

    def pnl_usd(self, px: float) -> float:
        return self.direction * (px - self.entry_px) * self.size

    def r_multiple(self, px: float) -> float:
        return self.direction * (px - self.entry_px) / self.r_unit if self.r_unit > 0 else 0.0


@dataclass(frozen=True)
class SizeDecision:
    size: float
    risk_frac: float            # fraction of equity actually at risk after caps
    risk_usd: float
    notional: float
    note: str


class ConvexRiskManager:
    def __init__(self, cfg: RiskConfig = RiskConfig()):
        self.cfg = cfg

    # ── sizing ──
    def kelly_fraction(self, stats: TradeStats) -> float:
        """Fractional Kelly on a pessimistic win rate: f = k · (p_lb − (1 − p_lb) / b), clipped to [0, cap]."""
        c = self.cfg
        if stats.n <= 0:
            return 0.0
        p = stats.win_rate
        p_lb = max(0.0, p - c.win_rate_z * math.sqrt(p * (1 - p) / stats.n))
        b = stats.payoff
        if b == math.inf:
            f_star = p_lb
        elif b <= 0:
            f_star = -1.0
        else:
            f_star = p_lb - (1 - p_lb) / b
        return min(c.max_risk_frac, max(0.0, c.kelly_fraction * f_star))

    def risk_fraction(self, stats: TradeStats) -> tuple:
        """(fraction, note). Bootstrap until enough realized trades; then quarter-Kelly, which can be zero."""
        c = self.cfg
        if stats.n < c.min_trades:
            return min(c.bootstrap_risk_frac, c.max_risk_frac), f"bootstrap {c.bootstrap_risk_frac:.2%} (n={stats.n}<{c.min_trades})"
        f = self.kelly_fraction(stats)
        note = (f"¼-Kelly {f:.2%} (n={stats.n}, p={stats.win_rate:.2f}, b={stats.payoff:.2f})"
                if f > 0 else f"no edge: ¼-Kelly ≤ 0 (n={stats.n}, p={stats.win_rate:.2f}, b={stats.payoff:.2f})")
        return f, note

    def initial_stop(self, entry_px: float, direction: int, atr: float) -> float:
        return entry_px - direction * self.cfg.stop_atr_mult * atr

    def size(self, equity: float, entry_px: float, atr: float, stats: TradeStats, sz_decimals: int,
             max_notional: float, min_notional: float = 10.0) -> Optional[SizeDecision]:
        """Units to trade so that equity × risk_frac is lost at the initial stop, under the notional cap."""
        frac, note = self.risk_fraction(stats)
        r_unit = self.cfg.stop_atr_mult * atr
        if frac <= 0 or r_unit <= 0 or equity <= 0 or entry_px <= 0:
            return None
        size = equity * frac / r_unit
        capped = False
        if size * entry_px > max_notional:
            size, capped = max_notional / entry_px, True
        q = 10 ** sz_decimals
        size = math.floor(size * q + 1e-9) / q
        notional = size * entry_px
        if notional < min_notional:
            return None
        risk_usd = size * r_unit
        return SizeDecision(size=size, risk_frac=risk_usd / equity, risk_usd=round(risk_usd, 2), notional=round(notional, 2),
                            note=note + (f"; notional capped at ${max_notional:,.0f}" if capped else ""))

    # ── exits ──
    def update_excursion(self, pos: Position, high: float, low: float) -> None:
        pos.best_px = max(pos.best_px, high) if pos.direction > 0 else min(pos.best_px, low)

    def desired_stop(self, pos: Position, atr: float) -> tuple:
        """(stop, stage) after ratcheting: never moves against the position."""
        c = self.cfg
        exc = pos.excursion_R
        if exc >= c.trail_trigger_R:
            stage, cand = 2, pos.best_px - pos.direction * c.trail_atr_mult * atr
        elif exc >= c.be_trigger_R:
            stage, cand = 1, pos.entry_px + pos.direction * c.be_offset_R * pos.r_unit
        else:
            stage, cand = 0, pos.initial_stop
        stage = max(stage, pos.stage)
        stop = max(pos.stop, cand) if pos.direction > 0 else min(pos.stop, cand)
        return stop, stage

    def stop_hit(self, pos: Position, high: float, low: float) -> bool:
        return low <= pos.stop if pos.direction > 0 else high >= pos.stop

    def paper_exit_px(self, px: float, direction: int) -> float:
        return px * (1 - direction * self.cfg.paper_slippage)

    def stop_moved_enough(self, old: float, new: float) -> bool:
        return old > 0 and abs(new - old) / old >= self.cfg.min_stop_move_frac


@dataclass
class BreakerState:
    halt_until_ms: int = 0
    tripped_ms: int = 0
    trips: int = 0


class CircuitBreaker:
    """Rolling-window mark-to-market drawdown monitor. `sample()` returns a trip dict once per trip."""

    def __init__(self, cfg: RiskConfig = RiskConfig(), state: Optional[BreakerState] = None):
        self.cfg = cfg
        self.state = state or BreakerState()
        self._eq: deque = deque()      # (ts_ms, equity)

    def halted(self, t_ms: int) -> bool:
        return t_ms < self.state.halt_until_ms

    def reset_window(self) -> None:
        self._eq.clear()

    def peak(self) -> float:
        return max((e for _, e in self._eq), default=0.0)

    def drawdown(self, equity: float) -> float:
        pk = self.peak()
        return 1.0 - equity / pk if pk > 0 else 0.0

    def sample(self, t_ms: int, equity: float) -> Optional[dict]:
        if equity <= 0:
            return None
        window_ms = int(self.cfg.breaker_window_s * 1000)
        while self._eq and self._eq[0][0] < t_ms - window_ms:
            self._eq.popleft()
        if self.halted(t_ms):
            self._eq.clear()           # the window restarts from the resume point
            return None
        dd = self.drawdown(equity)
        self._eq.append((t_ms, equity))
        if dd >= self.cfg.breaker_dd:
            pk = self.peak()
            self.state = BreakerState(halt_until_ms=t_ms + int(self.cfg.breaker_halt_s * 1000), tripped_ms=t_ms,
                                      trips=self.state.trips + 1)
            self._eq.clear()
            return {"ts": t_ms, "equity": equity, "peak": pk, "drawdown": dd, "halt_until_ms": self.state.halt_until_ms}
        return None


def stats_from_r_multiples(rs) -> TradeStats:
    rs = list(rs)
    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r <= 0]
    return TradeStats(n=len(rs), wins=len(wins),
                      avg_win_R=sum(wins) / len(wins) if wins else 0.0,
                      avg_loss_R=sum(losses) / len(losses) if losses else 0.0)


__all__ = ["RiskConfig", "TradeStats", "Position", "SizeDecision", "ConvexRiskManager", "BreakerState",
           "CircuitBreaker", "stats_from_r_multiples"]
