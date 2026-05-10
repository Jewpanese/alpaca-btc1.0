"""Tranche State Machine — Professional Scale-In/Scale-Out Framework.

Architecture shift (2026-03-26): Instead of entering all contracts at once
and managing exits in tranches, we now scale INTO the position as the trade
proves itself, then scale OUT at R-multiples.

Lifecycle: FLAT → PROBE → BUILDING → FULL_SIZE → REDUCING → CLOSED

Each transition is driven by structure and R-multiples, not "one signal, one order."
The market must earn additional size as it proves the thesis.

Key principles:
  - Risk budget per idea, not per contract
  - Invalidation-based stops (structural, not fixed ATR)
  - Scale in on confirmation, not on first signal
  - Scale out at R-multiples
  - Global risk veto layer
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# State Machine
# ═══════════════════════════════════════════════════════════════════════

class TrancheState(Enum):
    FLAT = auto()         # No position
    PROBE = auto()        # T1 filled — testing the idea
    BUILDING = auto()     # T2 filled — idea confirmed, adding size
    FULL_SIZE = auto()    # T3 filled — maximum conviction
    REDUCING = auto()     # Taking partial profits
    CLOSED = auto()       # All contracts exited


class Direction(Enum):
    LONG = auto()
    SHORT = auto()


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrancheConfig:
    """Configuration for the tranche state machine."""
    
    # Instrument
    point_value: float = 5.0          # MES = $5/point
    tick_size: float = 0.25
    
    # ── Risk Budget ──────────────────────────────────────────────
    max_risk_per_idea_dollars: float = 125.0   # Max loss per idea ($125)
    max_concurrent_risk_dollars: float = 250.0  # Max total open risk
    daily_loss_limit_dollars: float = -1500.0   # Done for day
    
    # ── Tranche Sizing ──────────────────────────────────────────
    # Fractions of max contracts (computed from risk budget + stop distance)
    t1_fraction: float = 0.40     # Probe: 40% (2 of 5)
    t2_fraction: float = 0.40     # Confirmation: 40% (2 of 5)
    t3_fraction: float = 0.20     # Momentum: 20% (1 of 5)
    min_contracts: int = 1        # Minimum per tranche
    
    # ── Entry Conditions ────────────────────────────────────────
    # T2 conditions (confirmation)
    t2_require_new_extreme: bool = True       # Price must make new high/low
    t2_require_hold: bool = True              # Must hold the new extreme (not V-reverse)
    t2_hold_bars: int = 2                     # Bars to hold above/below
    t2_max_wait_seconds: float = 300.0        # 5 min max wait for T2 after T1

    # C8: Scale on pullbacks, not at the extreme.
    # Required pullback from the confirmed extreme before T2 adds. Prevents
    # pyramid-at-peak. Pre-C8 T2 fired on any new-extreme-held bar, which put
    # the add at the worst possible price before reversals.
    t2_min_pullback_atr: float = 0.30         # Must be at least 0.3*ATR below recent extreme
    t3_min_pullback_atr: float = 0.20         # T3 slightly looser (trend more confirmed)

    # T3 conditions (momentum)
    t3_require_momentum: bool = True          # Volume spike or tape speed
    t3_min_profit_atr: float = 0.5            # T1+T2 must be profitable by 0.5 ATR
    t3_max_wait_seconds: float = 300.0        # 5 min max wait for T3 after T2
    
    # ── Exit Rules (R-based) ────────────────────────────────────
    # 1R = stop distance from average entry to invalidation
    exit_1r_pct: float = 0.50     # Take 50% off at 1R
    exit_2r_pct: float = 0.30     # Take 30% off at 2R (optional)
    runner_trail_atr: float = 0.75  # Trail runner at 0.75 ATR behind best price
    
    # After 1R exit, move stop to breakeven (or slight profit)
    breakeven_buffer_pts: float = 0.25  # +1 tick above breakeven after 1R

    # C9: Regime-conditional runner trail. runner_trail_atr above is the
    # baseline (used when regime == UNKNOWN/RANGE). In strong TREND we
    # can give the runner more room; in CHOP we tighten to lock profit
    # before the next reversal bar. Set to 0 to fall back to the baseline.
    runner_trail_atr_trend: float = 1.00  # Wider in trend
    runner_trail_atr_chop: float = 0.50   # Tighter in chop
    
    # ── Hard Stop ───────────────────────────────────────────────
    hard_stop_atr: float = 1.2    # Safety net if structural stop isn't hit
    
    # ── Winsorizing (2026-04-01) ────────────────────────────────
    # ATR-scaled cap for PROBE only. Confirmed trades (BUILDING/FULL)
    # rely on structural stops + brackets. Fixed dollar caps killed
    # winners on normal pullbacks (see 2026-04-01 10:04 incident).
    # Disabled 2026-04-20 after 19-day backtest: winsorize cost -$3,165 across 30 probe exits
    # and cut trades that would have recovered. Disabling recovered +$730 over the 19-day sample
    # (-$773 -> -$43 on A0 vs A1 in backtest_matrix.py). Hard stop (2.5×ATR bracket) remains the
    # real loss boundary.
    winsorize_enabled: bool = False
    winsorize_probe_atr_mult: float = 1.0     # PROBE max loss = ATR * mult * contracts * point_value
    winsorize_confirmed: bool = False          # If True, also cap BUILDING/FULL (disabled by default)
    
    # ── Cooldowns ───────────────────────────────────────────────
    min_time_between_ideas_seconds: float = 120.0  # 2 min between ideas
    max_full_stops_per_direction: int = 2           # After 2 full stops in same direction, block it
    max_ideas_per_session: int = 20                 # Don't overtrade


# ═══════════════════════════════════════════════════════════════════════
# Tranche Entry/Exit Records
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrancheEntry:
    """Record of a single tranche entry."""
    tranche: str           # "T1", "T2", "T3"
    contracts: int
    fill_price: float
    fill_time: float       # time.time()
    direction: Direction


@dataclass
class TrancheExit:
    """Record of a single tranche exit."""
    contracts: int
    fill_price: float
    fill_time: float
    reason: str            # "1R_TARGET", "2R_TARGET", "TRAIL", "INVALIDATION", "HARD_STOP"
    pnl_pts: float
    pnl_dollars: float


# ═══════════════════════════════════════════════════════════════════════
# Idea — one complete trade thesis with scale-in/scale-out
# ═══════════════════════════════════════════════════════════════════════

class TradeIdea:
    """Represents a single trade idea with tranche-based entries and exits.
    
    One idea = one thesis = one invalidation level = one risk budget.
    Contracts are added in tranches as the market proves the thesis.
    """
    
    def __init__(
        self,
        config: TrancheConfig,
        direction: Direction,
        invalidation_price: float,
        entry_price: float,
        atr: float,
        strategy_name: str = "",
        max_contracts: int = 5,
    ):
        self.config = config
        self.direction = direction
        self.invalidation_price = invalidation_price
        self.atr = atr
        self.strategy_name = strategy_name
        self.created_time = time.time()
        
        # State
        self.state = TrancheState.FLAT
        self.entries: list[TrancheEntry] = []
        self.exits: list[TrancheExit] = []
        
        # Compute stop distance and R
        if direction == Direction.LONG:
            self.stop_distance_pts = entry_price - invalidation_price
        else:
            self.stop_distance_pts = invalidation_price - entry_price
        
        # Ensure stop distance is positive and reasonable
        if self.stop_distance_pts <= 0:
            self.stop_distance_pts = atr * config.hard_stop_atr
            logger.warning(
                f"[TRANCHE] Invalid stop distance, using ATR fallback: "
                f"{self.stop_distance_pts:.2f}pts"
            )
        
        # 1R = stop distance
        self.r_value = self.stop_distance_pts
        
        # Compute max contracts from risk budget
        dollar_risk_per_contract = self.stop_distance_pts * config.point_value
        if dollar_risk_per_contract > 0:
            budget_contracts = int(config.max_risk_per_idea_dollars / dollar_risk_per_contract)
        else:
            budget_contracts = 1
        
        # Use the smaller of budget-based and provided max
        self.max_contracts = min(max(budget_contracts, 1), max_contracts)
        
        # Compute tranche sizes
        self.t1_size = max(config.min_contracts, round(self.max_contracts * config.t1_fraction))
        self.t2_size = max(config.min_contracts, round(self.max_contracts * config.t2_fraction))
        self.t3_size = max(config.min_contracts, self.max_contracts - self.t1_size - self.t2_size)
        
        # Ensure total doesn't exceed max
        total = self.t1_size + self.t2_size + self.t3_size
        if total > self.max_contracts:
            self.t3_size = max(0, self.max_contracts - self.t1_size - self.t2_size)
        
        # Position tracking
        self.total_contracts = 0
        self.avg_entry_price = 0.0
        self.best_price = entry_price  # Best price seen (for trailing)
        self.worst_unrealized_pts = 0.0
        self.best_unrealized_pts = 0.0
        
        # T2/T3 confirmation tracking
        self._t2_extreme_seen = False
        self._t2_hold_count = 0
        self._t2_extreme_price = 0.0
        self._t3_eligible = False
        
        # Exit tracking
        self._1r_taken = False
        self._2r_taken = False
        self._runner_active = False
        self._runner_trail_stop = 0.0
        self._stop_moved_to_be = False  # Stop moved to breakeven after 1R

        # C9: Regime-conditional trail. Caller updates this each bar from
        # the bot's market-mode classifier (TREND / CHOP / RANGE / UNKNOWN).
        self.current_regime: str = "UNKNOWN"
        
        logger.info(
            f"[TRANCHE] New idea: {direction.name} {strategy_name} | "
            f"Invalidation: {invalidation_price:.2f} | "
            f"Stop: {self.stop_distance_pts:.2f}pts (1R) | "
            f"Max contracts: {self.max_contracts} "
            f"(T1={self.t1_size} T2={self.t2_size} T3={self.t3_size}) | "
            f"Risk budget: ${config.max_risk_per_idea_dollars:.0f} | "
            f"Risk per contract: ${dollar_risk_per_contract:.2f}"
        )
    
    # ── Properties ──────────────────────────────────────────────
    
    @property
    def is_active(self) -> bool:
        return self.state not in (TrancheState.FLAT, TrancheState.CLOSED)
    
    @property
    def remaining_contracts(self) -> int:
        return self.total_contracts - sum(e.contracts for e in self.exits)
    
    @property
    def total_risk_dollars(self) -> float:
        """Current total risk if invalidation is hit."""
        if self.remaining_contracts == 0:
            return 0.0
        if self._stop_moved_to_be:
            # After 1R, stop is at breakeven — minimal risk
            return self.remaining_contracts * self.config.breakeven_buffer_pts * self.config.point_value
        return self.remaining_contracts * self.stop_distance_pts * self.config.point_value
    
    def unrealized_pts(self, current_price: float) -> float:
        """Current unrealized P&L in points from average entry."""
        if self.total_contracts == 0 or self.avg_entry_price == 0:
            return 0.0
        if self.direction == Direction.LONG:
            return current_price - self.avg_entry_price
        else:
            return self.avg_entry_price - current_price
    
    def _update_avg_entry(self, fill_price: float, contracts: int):
        """Recalculate weighted average entry after a new tranche fill."""
        old_value = self.avg_entry_price * self.total_contracts
        self.total_contracts += contracts
        if self.total_contracts > 0:
            self.avg_entry_price = (old_value + fill_price * contracts) / self.total_contracts
    
    # ── Entry Transitions ───────────────────────────────────────
    
    def fill_t1(self, fill_price: float, actual_contracts: int = None) -> TrancheEntry:
        """Record T1 (probe) fill. Call after order is confirmed filled.

        actual_contracts: the real filled quantity (may differ from t1_size when
        probe sizing was scaled down by market engine conviction). If omitted,
        falls back to t1_size. Always pass the real fill size to prevent the
        state machine tracking more contracts than the exchange actually holds.
        """
        assert self.state == TrancheState.FLAT, f"T1 fill in wrong state: {self.state}"
        contracts = actual_contracts if actual_contracts is not None else self.t1_size

        entry = TrancheEntry(
            tranche="T1",
            contracts=contracts,
            fill_price=fill_price,
            fill_time=time.time(),
            direction=self.direction,
        )
        self.entries.append(entry)
        self._update_avg_entry(fill_price, contracts)
        self.state = TrancheState.PROBE
        self.best_price = fill_price

        logger.info(
            f"[TRANCHE] T1 PROBE filled: {self.direction.name} {contracts}x @ {fill_price:.2f} | "
            f"Avg: {self.avg_entry_price:.2f} | Risk: ${self.total_risk_dollars:.0f}"
        )
        return entry
    
    def should_add_t2(self, current_price: float, current_bar: int = 0) -> bool:
        """Check if T2 (confirmation) conditions are met."""
        if self.state != TrancheState.PROBE:
            return False
        if self.t2_size == 0:
            return False
        
        # Time limit
        elapsed = time.time() - self.entries[-1].fill_time
        if elapsed > self.config.t2_max_wait_seconds:
            logger.info(f"[TRANCHE] T2 window expired ({elapsed:.0f}s > {self.config.t2_max_wait_seconds:.0f}s)")
            return False
        
        # Check if price has made a new extreme in our direction
        if self.config.t2_require_new_extreme:
            if self.direction == Direction.SHORT:
                is_new_extreme = current_price < self._t2_extreme_price if self._t2_extreme_seen else current_price < self.entries[0].fill_price
            else:
                is_new_extreme = current_price > self._t2_extreme_price if self._t2_extreme_seen else current_price > self.entries[0].fill_price

            if is_new_extreme:
                self._t2_extreme_seen = True
                self._t2_extreme_price = current_price
                # Do NOT reset hold_count on new extremes — fast trends make new highs
                # every bar and the old reset logic prevented T2 from ever firing during
                # the strongest (most desirable) moves.

            if not self._t2_extreme_seen:
                return False

            # Must hold for N bars above entry. Only reset when price LOSES the extreme
            # (drops back below T1 entry), not when it makes a new one.
            if self.config.t2_require_hold:
                if self.direction == Direction.LONG and current_price <= self.entries[0].fill_price:
                    self._t2_hold_count = 0  # Lost the extreme — reset
                    return False
                elif self.direction == Direction.SHORT and current_price >= self.entries[0].fill_price:
                    self._t2_hold_count = 0  # Lost the extreme — reset
                    return False

                self._t2_hold_count += 1
                if self._t2_hold_count < self.config.t2_hold_bars:
                    return False

        # C8: Pullback gate — don't chase the extreme.
        # Require price to be at least t2_min_pullback_atr*ATR retraced from
        # the confirmed extreme. Fires on resumption of trend, not at the top.
        if self._t2_extreme_seen and self.config.t2_min_pullback_atr > 0:
            min_pullback_pts = self.atr * self.config.t2_min_pullback_atr
            if self.direction == Direction.LONG:
                if current_price > self._t2_extreme_price - min_pullback_pts:
                    return False  # too close to high — wait for pullback
            else:
                if current_price < self._t2_extreme_price + min_pullback_pts:
                    return False  # too close to low — wait for pullback

        # Risk check: would adding T2 exceed budget?
        future_contracts = self.total_contracts + self.t2_size
        future_risk = future_contracts * self.stop_distance_pts * self.config.point_value
        if future_risk > self.config.max_risk_per_idea_dollars:
            logger.info(
                f"[TRANCHE] T2 blocked by risk budget: "
                f"${future_risk:.0f} > ${self.config.max_risk_per_idea_dollars:.0f}"
            )
            return False

        return True
    
    def fill_t2(self, fill_price: float, actual_contracts: int = None) -> TrancheEntry:
        """Record T2 (confirmation) fill."""
        assert self.state == TrancheState.PROBE, f"T2 fill in wrong state: {self.state}"
        contracts = actual_contracts if actual_contracts is not None else self.t2_size

        entry = TrancheEntry(
            tranche="T2",
            contracts=contracts,
            fill_price=fill_price,
            fill_time=time.time(),
            direction=self.direction,
        )
        self.entries.append(entry)
        self._update_avg_entry(fill_price, contracts)
        self.state = TrancheState.BUILDING

        logger.info(
            f"[TRANCHE] T2 CONFIRM filled: {self.direction.name} +{contracts}x @ {fill_price:.2f} | "
            f"Total: {self.total_contracts}x | Avg: {self.avg_entry_price:.2f} | "
            f"Risk: ${self.total_risk_dollars:.0f}"
        )
        return entry
    
    def should_add_t3(self, current_price: float, volume_ratio: float = 1.0) -> bool:
        """Check if T3 (momentum) conditions are met."""
        if self.state != TrancheState.BUILDING:
            return False
        if self.t3_size == 0:
            return False
        
        # Time limit from T2
        elapsed = time.time() - self.entries[-1].fill_time
        if elapsed > self.config.t3_max_wait_seconds:
            logger.info(f"[TRANCHE] T3 window expired ({elapsed:.0f}s > {self.config.t3_max_wait_seconds:.0f}s)")
            # Don't return False — just skip T3, trade stays as BUILDING
            self.state = TrancheState.FULL_SIZE  # Treat as full (without T3)
            logger.info(f"[TRANCHE] Promoting to FULL_SIZE without T3 (window expired)")
            return False
        
        # Must be profitable by min threshold
        unreal = self.unrealized_pts(current_price)
        if self.config.t3_require_momentum and unreal < self.atr * self.config.t3_min_profit_atr:
            return False

        # C8: Pullback gate — don't chase the extreme. best_price tracks the
        # highest (LONG) / lowest (SHORT) print since the position opened.
        if self.config.t3_min_pullback_atr > 0 and self.best_price != 0:
            min_pullback_pts = self.atr * self.config.t3_min_pullback_atr
            if self.direction == Direction.LONG:
                if current_price > self.best_price - min_pullback_pts:
                    return False  # too close to high — wait for pullback
            else:
                if current_price < self.best_price + min_pullback_pts:
                    return False  # too close to low — wait for pullback

        # Risk check
        future_contracts = self.total_contracts + self.t3_size
        future_risk = future_contracts * self.stop_distance_pts * self.config.point_value
        if future_risk > self.config.max_risk_per_idea_dollars:
            logger.info(f"[TRANCHE] T3 blocked by risk budget: ${future_risk:.0f} > ${self.config.max_risk_per_idea_dollars:.0f}")
            return False

        return True
    
    def fill_t3(self, fill_price: float, actual_contracts: int = None) -> TrancheEntry:
        """Record T3 (momentum) fill."""
        assert self.state == TrancheState.BUILDING, f"T3 fill in wrong state: {self.state}"
        contracts = actual_contracts if actual_contracts is not None else self.t3_size

        entry = TrancheEntry(
            tranche="T3",
            contracts=contracts,
            fill_price=fill_price,
            fill_time=time.time(),
            direction=self.direction,
        )
        self.entries.append(entry)
        self._update_avg_entry(fill_price, contracts)
        self.state = TrancheState.FULL_SIZE

        logger.info(
            f"[TRANCHE] T3 MOMENTUM filled: {self.direction.name} +{contracts}x @ {fill_price:.2f} | "
            f"Total: {self.total_contracts}x | Avg: {self.avg_entry_price:.2f} | "
            f"Risk: ${self.total_risk_dollars:.0f}"
        )
        return entry
    
    # ── Exit Logic ──────────────────────────────────────────────
    
    def check_exits(self, current_price: float) -> list[dict]:
        """Check all exit conditions. Returns list of exit actions to execute.
        
        Each action: {"contracts": N, "reason": str, "type": "MARKET"|"LIMIT"}
        
        Exit priority:
          1. Invalidation / hard stop → flatten all
          2. 1R target → take 50%
          3. 2R target → take 30%
          4. Runner trail → exit rest
        """
        if not self.is_active or self.remaining_contracts <= 0:
            return []
        
        actions = []
        unreal = self.unrealized_pts(current_price)
        remaining = self.remaining_contracts
        
        # Track best/worst
        if unreal > self.best_unrealized_pts:
            self.best_unrealized_pts = unreal
        if unreal < self.worst_unrealized_pts:
            self.worst_unrealized_pts = unreal
        
        # Update best price for trailing
        if self.direction == Direction.LONG:
            self.best_price = max(self.best_price, current_price)
        else:
            self.best_price = min(self.best_price, current_price)
        
        # ── 0. WINSORIZE — ATR-Scaled Probe Cap (2026-04-01 revised) ─────────
        # Only caps PROBE (unconfirmed) trades. Confirmed trades rely on
        # structural invalidation + exchange brackets. Fixed dollar caps
        # killed winners on normal pullbacks. ATR scaling adapts to vol.
        if self.config.winsorize_enabled:
            idea_pnl_dollars = unreal * remaining * self.config.point_value
            should_winsorize = False
            max_loss = 0.0
            
            if self.state == TrancheState.PROBE:
                # ATR-scaled: tighter in quiet markets, wider in volatile
                max_loss = self.atr * self.config.winsorize_probe_atr_mult * remaining * self.config.point_value
                max_loss = max(max_loss, 10.0)  # Floor: never less than $10
                should_winsorize = True
            elif self.config.winsorize_confirmed and self.state in (TrancheState.BUILDING, TrancheState.FULL_SIZE):
                # Optional: cap confirmed trades too (disabled by default)
                max_loss = self.atr * 1.5 * remaining * self.config.point_value
                max_loss = max(max_loss, 25.0)
                should_winsorize = True
            
            if should_winsorize and idea_pnl_dollars <= -max_loss:
                state_label = self.state.name
                actions.append({
                    "contracts": remaining,
                    "reason": (
                        f"WINSORIZED ({state_label}) at {current_price:.2f} | "
                        f"P&L: ${idea_pnl_dollars:+.2f} hit -${max_loss:.0f} cap "
                        f"(ATR={self.atr:.2f})"
                    ),
                    "type": "MARKET",
                    "exit_type": "WINSORIZED",
                })
                self.state = TrancheState.CLOSED
                logger.warning(
                    f"[TRANCHE] 🔒 WINSORIZED: {self.direction.name} {remaining}x | "
                    f"State: {state_label} | P&L: ${idea_pnl_dollars:+.2f} | "
                    f"Cap: -${max_loss:.0f} (ATR={self.atr:.2f}) | Price: {current_price:.2f}"
                )
                return actions
        
        # ── 1. INVALIDATION / HARD STOP ─────────────────────────
        invalidation_hit = False
        if self.direction == Direction.LONG:
            invalidation_hit = current_price <= self.invalidation_price
        else:
            invalidation_hit = current_price >= self.invalidation_price
        
        # Also check hard stop (ATR-based safety net)
        hard_stop_pts = self.atr * self.config.hard_stop_atr
        hard_stop_hit = unreal <= -hard_stop_pts
        
        # After 1R, check breakeven stop instead of invalidation
        if self._stop_moved_to_be and not invalidation_hit:
            be_price = self.avg_entry_price
            if self.direction == Direction.LONG:
                be_price -= self.config.breakeven_buffer_pts
                if current_price <= be_price:
                    actions.append({
                        "contracts": remaining,
                        "reason": f"BREAKEVEN_STOP at {current_price:.2f} (BE={be_price:.2f})",
                        "type": "MARKET",
                        "exit_type": "BREAKEVEN",
                    })
                    self.state = TrancheState.CLOSED
                    return actions
            else:
                be_price += self.config.breakeven_buffer_pts
                if current_price >= be_price:
                    actions.append({
                        "contracts": remaining,
                        "reason": f"BREAKEVEN_STOP at {current_price:.2f} (BE={be_price:.2f})",
                        "type": "MARKET",
                        "exit_type": "BREAKEVEN",
                    })
                    self.state = TrancheState.CLOSED
                    return actions
        
        if invalidation_hit or hard_stop_hit:
            reason = "INVALIDATION" if invalidation_hit else f"HARD_STOP ({unreal:.2f}pts)"
            actions.append({
                "contracts": remaining,
                "reason": f"{reason} at {current_price:.2f}",
                "type": "MARKET",
                "exit_type": "STOP",
            })
            self.state = TrancheState.CLOSED
            return actions
        
        # ── 2. PROFIT TARGETS (R-based) ─────────────────────────
        
        # 1R target
        if not self._1r_taken and unreal >= self.r_value:
            exit_contracts = max(1, round(remaining * self.config.exit_1r_pct))
            exit_contracts = min(exit_contracts, remaining)
            
            actions.append({
                "contracts": exit_contracts,
                "reason": f"1R_TARGET at +{unreal:.2f}pts (1R={self.r_value:.2f})",
                "type": "MARKET",
                "exit_type": "1R",
            })
            self._1r_taken = True
            self._stop_moved_to_be = True
            self.state = TrancheState.REDUCING
            remaining -= exit_contracts
            
            logger.info(
                f"[TRANCHE] 1R HIT → taking {exit_contracts} contracts, "
                f"stop moved to breakeven, {remaining} remaining"
            )
            
            if remaining <= 0:
                self.state = TrancheState.CLOSED
                return actions
            
            # Activate runner trail on remaining
            self._runner_active = True
            self._update_runner_trail(current_price)
        
        # 2R target
        if not self._2r_taken and self._1r_taken and unreal >= self.r_value * 2:
            exit_contracts = max(1, round(remaining * self.config.exit_2r_pct))
            exit_contracts = min(exit_contracts, remaining)
            
            if exit_contracts > 0:
                actions.append({
                    "contracts": exit_contracts,
                    "reason": f"2R_TARGET at +{unreal:.2f}pts (2R={self.r_value * 2:.2f})",
                    "type": "MARKET",
                    "exit_type": "2R",
                })
                self._2r_taken = True
                remaining -= exit_contracts
                
                if remaining <= 0:
                    self.state = TrancheState.CLOSED
                    return actions
        
        # ── 3. RUNNER TRAIL ─────────────────────────────────────
        if self._runner_active and remaining > 0:
            self._update_runner_trail(current_price)
            
            trail_hit = False
            if self.direction == Direction.LONG:
                trail_hit = current_price <= self._runner_trail_stop
            else:
                trail_hit = current_price >= self._runner_trail_stop
            
            if trail_hit:
                actions.append({
                    "contracts": remaining,
                    "reason": f"RUNNER_TRAIL at {current_price:.2f} (trail={self._runner_trail_stop:.2f})",
                    "type": "MARKET",
                    "exit_type": "TRAIL",
                })
                self.state = TrancheState.CLOSED
        
        return actions
    
    def update_live_price(self, price: float):
        """Update best_price and runner trail from a live quote (intra-bar).

        Call this on every tick/quote while the runner is active so the trail
        advances with the live price rather than waiting for bar close.
        Only updates best_price and trail — never triggers exits (that stays
        in check_exits at bar close).
        """
        if not self.is_active or not self._runner_active:
            return
        if self.direction == Direction.LONG:
            if price > self.best_price:
                self.best_price = price
                self._update_runner_trail(price)
        else:
            if price < self.best_price:
                self.best_price = price
                self._update_runner_trail(price)

    def _update_runner_trail(self, current_price: float):
        """Update the runner trailing stop.

        C9: Trail multiplier is regime-conditional. TREND → wider (let it run),
        CHOP → tighter (lock profit before reversal), else → baseline.
        """
        trail_mult = self.config.runner_trail_atr
        if self.current_regime == "TREND" and self.config.runner_trail_atr_trend > 0:
            trail_mult = self.config.runner_trail_atr_trend
        elif self.current_regime == "CHOP" and self.config.runner_trail_atr_chop > 0:
            trail_mult = self.config.runner_trail_atr_chop
        trail_distance = self.atr * trail_mult

        if self.direction == Direction.LONG:
            new_trail = self.best_price - trail_distance
            # Trail only moves up, never down
            if self._runner_trail_stop == 0 or new_trail > self._runner_trail_stop:
                self._runner_trail_stop = new_trail
        else:
            new_trail = self.best_price + trail_distance
            # Trail only moves down (for shorts), never up
            if self._runner_trail_stop == 0 or new_trail < self._runner_trail_stop:
                self._runner_trail_stop = new_trail
    
    def record_exit(self, contracts: int, fill_price: float, reason: str) -> TrancheExit:
        """Record an actual exit fill."""
        unreal = self.unrealized_pts(fill_price)
        pnl_dollars = unreal * self.config.point_value * contracts
        
        exit_record = TrancheExit(
            contracts=contracts,
            fill_price=fill_price,
            fill_time=time.time(),
            reason=reason,
            pnl_pts=unreal,
            pnl_dollars=pnl_dollars,
        )
        self.exits.append(exit_record)
        
        if self.remaining_contracts <= 0:
            self.state = TrancheState.CLOSED
        
        logger.info(
            f"[TRANCHE] EXIT: {contracts}x @ {fill_price:.2f} | "
            f"Reason: {reason} | P&L: ${pnl_dollars:+.2f} ({unreal:+.2f}pts) | "
            f"Remaining: {self.remaining_contracts}"
        )
        return exit_record
    
    def force_close(self, reason: str = "FORCED"):
        """Mark idea as closed (caller handles the actual order)."""
        self.state = TrancheState.CLOSED
        logger.info(f"[TRANCHE] FORCE CLOSED: {reason}")
    
    def get_status(self) -> dict:
        """Current status summary."""
        return {
            "state": self.state.name,
            "direction": self.direction.name,
            "strategy": self.strategy_name,
            "total_contracts": self.total_contracts,
            "remaining": self.remaining_contracts,
            "avg_entry": self.avg_entry_price,
            "invalidation": self.invalidation_price,
            "r_value": self.r_value,
            "best_unreal": self.best_unrealized_pts,
            "worst_unreal": self.worst_unrealized_pts,
            "1r_taken": self._1r_taken,
            "2r_taken": self._2r_taken,
            "runner_active": self._runner_active,
            "runner_trail": self._runner_trail_stop,
            "risk_dollars": self.total_risk_dollars,
            "entries": len(self.entries),
            "exits": len(self.exits),
        }


# ═══════════════════════════════════════════════════════════════════════
# Risk Manager — Global veto layer
# ═══════════════════════════════════════════════════════════════════════

class TrancheRiskManager:
    """Global risk manager that sits above individual TradeIdeas.
    
    Tracks:
      - Daily P&L
      - Open risk across all ideas
      - Consecutive full stops per direction
      - Idea count per session
    
    Can veto any new tranche or force flattening.
    """
    
    def __init__(self, config: TrancheConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.ideas_today: int = 0
        self.full_stops_long: int = 0
        self.full_stops_short: int = 0
        self.last_idea_close_time: float = 0.0
        self.active_idea: Optional[TradeIdea] = None
    
    def can_start_new_idea(self, direction: Direction) -> tuple[bool, str]:
        """Check if a new idea is allowed."""
        
        # Daily loss limit
        if self.daily_pnl <= self.config.daily_loss_limit_dollars:
            return False, f"Daily loss limit hit: ${self.daily_pnl:.0f} <= ${self.config.daily_loss_limit_dollars:.0f}"
        
        # Max ideas per session
        if self.ideas_today >= self.config.max_ideas_per_session:
            return False, f"Max ideas per session: {self.ideas_today} >= {self.config.max_ideas_per_session}"
        
        # Cooldown between ideas
        elapsed = time.time() - self.last_idea_close_time
        if self.last_idea_close_time > 0 and elapsed < self.config.min_time_between_ideas_seconds:
            return False, f"Cooldown: {elapsed:.0f}s < {self.config.min_time_between_ideas_seconds:.0f}s"
        
        # Direction-specific full stop limit
        if direction == Direction.LONG and self.full_stops_long >= self.config.max_full_stops_per_direction:
            return False, f"Max full stops LONG: {self.full_stops_long} >= {self.config.max_full_stops_per_direction}"
        if direction == Direction.SHORT and self.full_stops_short >= self.config.max_full_stops_per_direction:
            return False, f"Max full stops SHORT: {self.full_stops_short} >= {self.config.max_full_stops_per_direction}"
        
        # Already have an active idea
        if self.active_idea and self.active_idea.is_active:
            return False, f"Already have active idea: {self.active_idea.state.name}"
        
        return True, "OK"
    
    def register_idea(self, idea: TradeIdea):
        """Register a new idea."""
        self.active_idea = idea
        self.ideas_today += 1
    
    def on_idea_closed(self, idea: TradeIdea):
        """Called when an idea is fully closed."""
        # Calculate total P&L for the idea
        total_pnl = sum(e.pnl_dollars for e in idea.exits)
        self.daily_pnl += total_pnl
        self.last_idea_close_time = time.time()
        
        # Track full stops
        is_full_stop = not idea._1r_taken  # Never hit 1R = full stop
        if is_full_stop and total_pnl < 0:
            if idea.direction == Direction.LONG:
                self.full_stops_long += 1
            else:
                self.full_stops_short += 1
            logger.warning(
                f"[TRANCHE RISK] Full stop #{self.full_stops_long + self.full_stops_short}: "
                f"${total_pnl:.2f} | Daily: ${self.daily_pnl:.2f}"
            )
        
        if self.active_idea is idea:
            self.active_idea = None
        
        logger.info(
            f"[TRANCHE RISK] Idea closed: ${total_pnl:+.2f} | "
            f"Daily P&L: ${self.daily_pnl:+.2f} | "
            f"Ideas today: {self.ideas_today} | "
            f"Full stops: L={self.full_stops_long} S={self.full_stops_short}"
        )
    
    def reset_daily(self):
        """Reset daily counters (call at session start)."""
        self.daily_pnl = 0.0
        self.ideas_today = 0
        self.full_stops_long = 0
        self.full_stops_short = 0
        self.last_idea_close_time = 0.0
        self.active_idea = None
        logger.info("[TRANCHE RISK] Daily counters reset")
