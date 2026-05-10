"""Hybrid Position Manager -- Institutional-Grade Exit System.

Three tranches with fundamentally different exit philosophies:

1. RISK REDUCTION (30%) -- Math-based. Exits when locked profit covers
   remaining position risk. Goal: create a risk-free trade.

2. CORE (40%) -- Signal-based. Exits when the strategy's entry condition
   invalidates. The edge is gone = we're out. This is how quants do it.

3. RUNNER (30%) -- Adaptive trail. Trail distance responds to regime,
   volume, and time -- not fixed ATR multiples.

Key differences from retail tranche systems:
- Risk reduction is CALCULATED, not a fixed TP
- Core exits on SIGNAL INVALIDATION, not a price target
- Runner trail ADAPTS to market conditions in real-time
- All tranches share a single hard SL (bracket) as safety net
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)


# -- Exit Events ----------------------------------------------------------

class ExitReason(Enum):
    RISK_REDUCTION = auto()    # Locked enough profit to cover remaining risk
    SIGNAL_INVALID = auto()    # Strategy says the edge is gone
    ADAPTIVE_TRAIL = auto()    # Adaptive trailing stop hit
    TIME_DECAY = auto()        # Trade stalled too long
    HARD_STOP = auto()         # Bracket SL hit (emergency)


@dataclass
class TrancheExit:
    """A tranche exit event."""
    role: str           # "risk_reduce", "core", "runner"
    size: int           # Contracts to close
    reason: str         # Human-readable
    exit_type: ExitReason
    pnl_pts: float      # Unrealized P&L at exit


# -- Configuration --------------------------------------------------------

@dataclass
class HybridConfig:
    """Configuration for the hybrid position manager.

    BTC scale: 1 pt = $1 of BTC price; 1 contract = 0.01 BTC; point_value = $0.01.
    Original ES/MES defaults (point_value=5.0, all _pts in 0.5–4.0 range) were
    100–200× too tight for BTC's $50–1000 typical 3-min ATR.
    """
    # Instrument — BTC default; override at instantiation if porting back to ES/MES
    point_value: float = 0.01        # BTC: $0.01/pt/contract (1 contract = 0.01 BTC)
    instrument: str = "BTC"

    # Tranche allocation (fractions -- must sum to 1.0)
    risk_reduce_pct: float = 0.30
    core_pct: float = 0.40
    runner_pct: float = 0.30

    # -- Risk Reduction tranche --
    # Target: lock profit fast to significantly reduce net risk on remaining contracts.
    risk_reduce_target_pts: float = 300.0  # Exit risk-reduce at +$300 of BTC move
    risk_reduce_min_pts: float = 200.0     # Don't exit below +$200

    # -- Core tranche --
    core_time_decay_seconds: int = 300     # 5 min: if no progress, exit
    core_time_decay_min_pts: float = 300.0  # Must be below +$300 for time decay
    core_breakeven_seconds: int = 180      # 3 min: move to BE if profit < threshold
    core_breakeven_min_pts: float = 150.0

    # -- Runner tranche --
    runner_activation_pts: float = 400.0   # Don't start trailing until +$400
    runner_base_trail_pts: float = 250.0   # Base trail distance ($250)
    runner_min_stop_pts: float = 50.0      # Trail stop NEVER goes below $50
    # Regime modifiers (multiply base trail)
    runner_trending_mult: float = 1.5
    runner_choppy_mult: float = 0.6
    runner_volatile_mult: float = 1.2
    # Volume modifiers
    runner_volume_exhaustion_mult: float = 0.5
    # Time decay
    runner_stale_seconds: int = 900        # 15 min no new high = tighten
    runner_stale_trail_pts: float = 100.0  # Tight $100 trail after stale

    # -- Hard stop (bracket SL — safety net for all tranches) --
    max_stop_pts: float = 1000.0           # $1000 hard cap on stop distance

    # -- Breakeven buffer --
    fee_buffer_pts: float = 5.0            # $5 added to entry when moving to BE


def _allocate(total: int, pcts: list) -> list:
    """Allocate total contracts across tranches by percentage.

    Ensures every tranche gets at least 1 contract if total >= 3,
    and the sum equals total exactly (remainder goes to core).
    """
    if total <= 0:
        return [0] * len(pcts)
    if total < len(pcts):
        # Not enough for all tranches -- give everything to core (middle)
        result = [0] * len(pcts)
        result[1] = total  # core index
        return result

    raw = [max(1, round(total * p)) for p in pcts]
    # Fix rounding: adjust core (index 1) to make sum correct
    diff = total - sum(raw)
    raw[1] += diff
    if raw[1] < 1:
        raw[1] = 1
        # Steal from largest
        while sum(raw) > total:
            idx = max(range(len(raw)), key=lambda i: raw[i] if i != 1 else 0)
            raw[idx] -= 1
    return raw


# -- Tranche State --------------------------------------------------------

@dataclass
class RiskReduceTranche:
    """Math-based risk reduction tranche."""
    size: int
    closed: bool = False
    close_price: float = 0.0

    def should_exit(self, unrealized_pts: float, target_pts: float,
                    min_pts: float) -> bool:
        """Exit at fixed profit target to reduce position risk.

        Simple and reliable: when price reaches target, take profit on this
        tranche. The locked profit reduces effective risk on remaining contracts.
        
        Example: 10 MES, risk-reduce 3 at +2pt = $30 locked.
        Remaining 7 MES at -4pt SL = $140 loss - $30 locked = $110 net risk.
        That's a 45% risk reduction from $200 original.
        """
        if self.closed:
            return False
        if unrealized_pts < min_pts:
            return False
        return unrealized_pts >= target_pts


@dataclass
class CoreTranche:
    """Signal-invalidation-based core tranche."""
    size: int
    closed: bool = False
    close_price: float = 0.0
    _best_pts: float = 0.0
    _breakeven_active: bool = False
    _breakeven_price: float = 0.0

    def update(self, unrealized_pts: float) -> None:
        self._best_pts = max(self._best_pts, unrealized_pts)

    def should_exit_signal(self, signal_exit_reason: Optional[str]) -> Optional[str]:
        """Exit if the strategy says the edge is gone."""
        if self.closed:
            return None
        if signal_exit_reason:
            return f"Signal invalidation: {signal_exit_reason}"
        return None

    def should_exit_time(self, hold_seconds: float, unrealized_pts: float,
                         config: HybridConfig, entry_price: float,
                         direction: int) -> Optional[str]:
        """Time-decay exits for stalled trades."""
        if self.closed:
            return None

        # Move to breakeven after N seconds if profit < threshold
        if (hold_seconds > config.core_breakeven_seconds
                and unrealized_pts < config.core_breakeven_min_pts
                and not self._breakeven_active):
            self._breakeven_active = True
            self._breakeven_price = entry_price + (config.fee_buffer_pts * direction)
            logger.info(
                f"[CORE] Time-decay breakeven active after {hold_seconds:.0f}s "
                f"(profit only {unrealized_pts:.2f}pts)"
            )

        # Check breakeven stop
        if self._breakeven_active:
            if direction == 1 and (entry_price + unrealized_pts * direction) <= self._breakeven_price:
                return f"Time-decay breakeven hit after {hold_seconds:.0f}s"
            # Simpler: if unrealized_pts drops to ~0 or negative
            if unrealized_pts <= 0:
                return f"Time-decay breakeven hit after {hold_seconds:.0f}s"

        # Hard time decay: if no meaningful progress after long hold
        if (hold_seconds > config.core_time_decay_seconds
                and unrealized_pts < config.core_time_decay_min_pts):
            return f"Time-decay exit: {hold_seconds:.0f}s held, only {unrealized_pts:.2f}pts profit"

        return None


@dataclass
class RunnerTranche:
    """Adaptive trailing stop runner tranche."""
    size: int
    closed: bool = False
    close_price: float = 0.0
    _activated: bool = False       # Trail not active until activation threshold
    _best_pts: float = 0.0
    _best_pts_time: float = 0.0    # When best was achieved
    _trail_stop_pts: float = 0.0   # Current trail stop in pts from entry (positive = profit)

    def update(self, unrealized_pts: float, now: float,
               entry_price: float, direction: int,
               config: HybridConfig,
               regime: str = "unknown",
               volume_ratio: float = 1.0) -> Optional[str]:
        """Update adaptive trail. Returns exit reason or None."""
        if self.closed:
            return None

        # Track best profit
        if unrealized_pts > self._best_pts:
            self._best_pts = unrealized_pts
            self._best_pts_time = now

        # Don't activate trail until threshold
        if not self._activated:
            if self._best_pts >= config.runner_activation_pts:
                self._activated = True
                self._trail_stop_pts = max(config.runner_min_stop_pts, self._best_pts - config.runner_base_trail_pts)
                logger.info(
                    f"[RUNNER] Trail activated at +{self._best_pts:.2f}pts, "
                    f"initial trail stop at +{self._trail_stop_pts:.2f}pts "
                    f"(floor: +{config.runner_min_stop_pts:.1f}pts)"
                )
            else:
                # Not yet activated — floor: runners NEVER go negative once they've been in the green
                if self._best_pts >= config.runner_min_stop_pts and unrealized_pts <= 0:
                    return (
                        f"Runner floor hit before activation: profit={unrealized_pts:+.2f}pts, "
                        f"best=+{self._best_pts:.2f}pts — closing to prevent loss"
                    )
                return None  # Not yet activated

        # Calculate adaptive trail distance
        trail_dist = config.runner_base_trail_pts

        # Regime adjustment
        if regime == "trending":
            trail_dist *= config.runner_trending_mult
        elif regime == "ranging" or regime == "choppy":
            trail_dist *= config.runner_choppy_mult
        elif regime == "volatile":
            trail_dist *= config.runner_volatile_mult

        # Volume exhaustion: if volume ratio < 0.5, trend is dying -- tighten
        if volume_ratio < 0.5:
            trail_dist *= config.runner_volume_exhaustion_mult
            logger.debug(f"[RUNNER] Volume exhaustion ({volume_ratio:.2f}), trail tightened")

        # Time stale: no new high for N seconds -- tighten
        time_since_best = now - self._best_pts_time if self._best_pts_time > 0 else 0
        if time_since_best > config.runner_stale_seconds:
            trail_dist = min(trail_dist, config.runner_stale_trail_pts)
            logger.debug(f"[RUNNER] Stale {time_since_best:.0f}s, trail={trail_dist:.2f}pts")

        # Trail only moves UP (never widens), and NEVER below minimum floor
        new_trail_stop = max(config.runner_min_stop_pts, self._best_pts - trail_dist)
        if new_trail_stop > self._trail_stop_pts:
            self._trail_stop_pts = new_trail_stop

        # Check if price has fallen through trail
        if unrealized_pts <= self._trail_stop_pts:
            return (
                f"Adaptive trail hit: profit={unrealized_pts:+.2f}pts, "
                f"trail_stop=+{self._trail_stop_pts:.2f}pts, "
                f"best=+{self._best_pts:.2f}pts, "
                f"regime={regime}, vol_ratio={volume_ratio:.2f}"
            )

        return None


# -- Main Manager ---------------------------------------------------------

class HybridPositionManager:
    """Institutional-grade position exit manager.

    Usage:
        hpm = HybridPositionManager(
            entry_price=6900.0, direction=-1, total_contracts=10,
            stop_distance=4.0, config=HybridConfig()
        )

        # On each tick/bar:
        exits = hpm.update(
            current_price=6896.0,
            market_state=state,           # MarketState for regime/volume
            signal_exit_reason=strategy.should_exit(...)  # Signal invalidation
        )
        for exit in exits:
            if hpm.is_fully_closed():
                flatten_all()
            else:
                partial_close(exit.size)
    """

    def __init__(
        self,
        entry_price: float,
        direction: int,            # +1 LONG, -1 SHORT
        total_contracts: int,
        stop_distance: float,      # SL distance in points (bracket)
        config: HybridConfig = None,
    ):
        self.config = config or HybridConfig()
        self.entry_price = entry_price
        self.direction = direction
        self.total_contracts = total_contracts
        self.stop_distance = stop_distance
        self.entry_time = time.time()

        # Allocate tranches
        pcts = [self.config.risk_reduce_pct, self.config.core_pct, self.config.runner_pct]
        sizes = _allocate(total_contracts, pcts)

        self.risk_reduce = RiskReduceTranche(size=sizes[0])
        self.core = CoreTranche(size=sizes[1])
        self.runner = RunnerTranche(size=sizes[2])

        # Compute risk math for logging
        total_risk = total_contracts * stop_distance * self.config.point_value
        rr_locked = self.risk_reduce.size * self.config.risk_reduce_target_pts * self.config.point_value
        remaining_after_rr = total_contracts - self.risk_reduce.size
        net_risk = (remaining_after_rr * stop_distance * self.config.point_value) - rr_locked
        risk_pct_reduction = (1 - net_risk / total_risk) * 100 if total_risk > 0 else 0

        logger.info(
            f"[HYBRID] Position: {total_contracts} {self.config.instrument} "
            f"{'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.2f}"
        )
        logger.info(
            f"[HYBRID] Tranches: risk_reduce={sizes[0]}, core={sizes[1]}, runner={sizes[2]}"
        )
        logger.info(
            f"[HYBRID] Total risk: ${total_risk:.0f} | "
            f"Risk-reduce at +{self.config.risk_reduce_target_pts:.1f}pts -> "
            f"locks ${rr_locked:.0f}, net risk ${net_risk:.0f} "
            f"({risk_pct_reduction:.0f}% reduction)"
        )

    def update(
        self,
        current_price: float,
        signal_exit_reason: Optional[str] = None,
        regime: str = "unknown",
        volume_ratio: float = 1.0,
        atr: float = 2.0,
    ) -> List[TrancheExit]:
        """Update all tranches. Returns list of exits to execute.

        Args:
            current_price: Current market price
            signal_exit_reason: From strategy.should_exit() -- None if signal still valid
            regime: 'trending', 'ranging', 'volatile', 'unknown'
            volume_ratio: Current volume / avg volume (< 0.5 = exhaustion)
            atr: Current ATR for adaptive calculations
        """
        now = time.time()
        hold_seconds = now - self.entry_time
        unrealized_pts = (current_price - self.entry_price) * self.direction
        exits: List[TrancheExit] = []

        # -- 1. Risk Reduction --
        if not self.risk_reduce.closed:
            should_exit = self.risk_reduce.should_exit(
                unrealized_pts,
                self.config.risk_reduce_target_pts,
                self.config.risk_reduce_min_pts,
            )
            if should_exit:
                remaining_after = self.remaining_contracts - self.risk_reduce.size
                locked = self.risk_reduce.size * unrealized_pts * self.config.point_value
                remaining_max_loss = remaining_after * self.config.max_stop_pts * self.config.point_value
                net_risk = remaining_max_loss - locked
                self.risk_reduce.closed = True
                self.risk_reduce.close_price = current_price
                exits.append(TrancheExit(
                    role="risk_reduce",
                    size=self.risk_reduce.size,
                    reason=(
                        f"Risk reduction: +{unrealized_pts:.2f}pts, "
                        f"locked ${locked:.0f}, net risk now ${net_risk:.0f} "
                        f"(was ${self.total_contracts * self.config.max_stop_pts * self.config.point_value:.0f})"
                    ),
                    exit_type=ExitReason.RISK_REDUCTION,
                    pnl_pts=unrealized_pts,
                ))
                logger.info(
                    f"[HYBRID] RISK REDUCED: Closed {self.risk_reduce.size} {self.config.instrument} "
                    f"at +{unrealized_pts:.2f}pts, locked ${locked:.0f}. "
                    f"Remaining {remaining_after} contracts, net risk ${net_risk:.0f}."
                )

        # -- 2. Core (signal invalidation + time decay) --
        if not self.core.closed:
            self.core.update(unrealized_pts)

            # Signal invalidation -- primary exit
            sig_reason = self.core.should_exit_signal(signal_exit_reason)
            if sig_reason:
                self.core.closed = True
                self.core.close_price = current_price
                exits.append(TrancheExit(
                    role="core",
                    size=self.core.size,
                    reason=sig_reason,
                    exit_type=ExitReason.SIGNAL_INVALID,
                    pnl_pts=unrealized_pts,
                ))

            # Time decay -- fallback
            if not self.core.closed:
                time_reason = self.core.should_exit_time(
                    hold_seconds, unrealized_pts, self.config,
                    self.entry_price, self.direction,
                )
                if time_reason:
                    self.core.closed = True
                    self.core.close_price = current_price
                    exits.append(TrancheExit(
                        role="core",
                        size=self.core.size,
                        reason=time_reason,
                        exit_type=ExitReason.TIME_DECAY,
                        pnl_pts=unrealized_pts,
                    ))

        # -- 3. Runner (adaptive trail) --
        if not self.runner.closed:
            # If core just closed and runner never activated, follow core out.
            # Runner's job is to ride a winning move — if core says "done", there's no move to ride.
            if self.core.closed and not self.runner._activated:
                self.runner.closed = True
                self.runner.close_price = current_price
                exits.append(TrancheExit(
                    role="runner",
                    size=self.runner.size,
                    reason=f"Core exited, runner never activated (best: +{self.runner._best_pts:.2f}pts) — no move to ride",
                    exit_type=ExitReason.TIME_DECAY,
                    pnl_pts=unrealized_pts,
                ))
                logger.info(
                    f"[RUNNER] Following core exit — never activated "
                    f"(best: +{self.runner._best_pts:.2f}pts, current: {unrealized_pts:+.2f}pts)"
                )
            else:
                trail_reason = self.runner.update(
                    unrealized_pts, now, self.entry_price, self.direction,
                    self.config, regime, volume_ratio,
                )
                if trail_reason:
                    self.runner.closed = True
                    self.runner.close_price = current_price
                    exits.append(TrancheExit(
                        role="runner",
                        size=self.runner.size,
                        reason=trail_reason,
                        exit_type=ExitReason.ADAPTIVE_TRAIL,
                        pnl_pts=unrealized_pts,
                    ))

        return exits

    def is_fully_closed(self) -> bool:
        return self.risk_reduce.closed and self.core.closed and self.runner.closed

    @property
    def remaining_contracts(self) -> int:
        total = 0
        if not self.risk_reduce.closed:
            total += self.risk_reduce.size
        if not self.core.closed:
            total += self.core.size
        if not self.runner.closed:
            total += self.runner.size
        return total

    def update_entry_price(self, fill_price: float):
        """Update entry price when actual fill differs from signal."""
        if abs(fill_price - self.entry_price) > 0.01:
            old = self.entry_price
            self.entry_price = fill_price
            logger.info(f"[HYBRID] Entry updated: {old:.2f} -> {fill_price:.2f}")

    def get_status(self) -> dict:
        return {
            "entry_price": self.entry_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "total_contracts": self.total_contracts,
            "remaining": self.remaining_contracts,
            "hold_seconds": time.time() - self.entry_time,
            "risk_reduce": {"size": self.risk_reduce.size, "closed": self.risk_reduce.closed},
            "core": {
                "size": self.core.size, "closed": self.core.closed,
                "best_pts": self.core._best_pts,
                "breakeven_active": self.core._breakeven_active,
            },
            "runner": {
                "size": self.runner.size, "closed": self.runner.closed,
                "activated": self.runner._activated,
                "best_pts": self.runner._best_pts,
                "trail_stop_pts": self.runner._trail_stop_pts,
            },
        }
