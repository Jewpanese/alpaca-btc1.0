"""Adaptive Exit Manager — ATR-Scaled Tranche Exits.

Evolution of hybrid_position_manager.py with one critical change:
ALL distances are ATR-relative, never fixed points.

Same 3-tranche philosophy:
  1. Risk Reduction (30%) — lock profit to reduce net risk
  2. Core (40%) — exit on signal invalidation or time decay
  3. Runner (30%) — adaptive trailing stop

But now:
  - Stop distances scale with ATR
  - Tranche targets scale with ATR
  - Trail widths scale with ATR
  - Time decays adjust based on volatility (high vol = more patience)
  - Breakeven thresholds are ATR-relative

Normal day (ATR=2): stop=3pt, risk_reduce target=1.3pt, runner trail=1.5pt
War gap (ATR=10): stop=15pt, risk_reduce target=6.5pt, runner trail=7.5pt

Same risk. Different markets. That's the whole point.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    RISK_REDUCTION = auto()
    SIGNAL_INVALID = auto()
    ADAPTIVE_TRAIL = auto()
    TIME_DECAY = auto()
    HARD_STOP = auto()
    BREAKEVEN = auto()


@dataclass
class TrancheExit:
    """A tranche exit event."""
    role: str
    size: int
    reason: str
    exit_type: ExitReason
    pnl_pts: float


@dataclass
class AdaptiveExitConfig:
    """All distances are in ATR multiples unless noted."""
    
    # Instrument — BTC defaults (1 contract = 0.01 BTC, point_value = $0.01/pt)
    # The ATR-multiple-based design means absolute scale doesn't matter much
    # for tranche logic; these defaults primarily affect the fixed-point fallbacks.
    point_value: float = 0.01
    tick_size: float = 0.01
    instrument: str = "BTC"
    
    # Tranche allocation
    risk_reduce_pct: float = 0.30
    core_pct: float = 0.40
    runner_pct: float = 0.30
    
    # Risk Reduction tranche (ATR multiples)
    risk_reduce_target_atr: float = 1.0     # Exit at +1.0 ATR profit (was 0.65 — too tight)
    risk_reduce_min_atr: float = 0.65       # Don't exit below +0.65 ATR (was 0.40)
    
    # Core tranche
    core_time_decay_base_seconds: int = 480       # 8 min base (was 5 — too impatient)
    core_time_decay_min_atr: float = 0.75         # Must be below 0.75 ATR for time decay (was 0.50)
    core_breakeven_base_seconds: int = 300         # 5 min base (was 3 — shaken out too fast)
    core_breakeven_min_atr: float = 0.50           # Below 0.50 ATR triggers breakeven (was 0.25)
    
    # Runner tranche (ATR multiples)
    runner_activation_atr: float = 0.75            # Start trailing at +0.75 ATR (lowered — was missing moves)
    runner_base_trail_atr: float = 0.75            # Base trail distance (tighter — capture more of the move)
    runner_min_stop_atr: float = 0.15              # Floor: runners never go below +0.15 ATR (~2 ticks)
    runner_ratchet_enabled: bool = True            # Tighten trail as profit grows
    runner_ratchet_step_atr: float = 0.5           # Every +0.5 ATR of profit, tighten trail by ratchet_tighten
    runner_ratchet_tighten: float = 0.85           # Multiply trail width by this per step (trail shrinks)
    
    # Regime modifiers for runner trail
    runner_trending_mult: float = 1.5
    runner_choppy_mult: float = 0.6
    runner_volatile_mult: float = 1.2
    runner_volume_exhaustion_mult: float = 0.5
    runner_stale_seconds: int = 900
    runner_stale_trail_atr: float = 0.25
    
    # Fixed point targets (override ATR-based when set)
    fixed_target_pts_t1: Optional[float] = None    # T1: fixed points (e.g., 1.5)
    fixed_target_pts_t2: Optional[float] = None    # T2: fixed points (e.g., 2.0)
    fixed_target_pts_t3: Optional[float] = None    # T3: fixed points (e.g., 2.0)
    
    # Hard stop (ATR multiple — safety net)
    hard_stop_atr: float = 1.5                     # Same as vol_sizing stop
    
    # Incubation — no tranche exits during this window (only LIL guardrail can exit)
    incubation_seconds: float = 90.0               # Must match LIL incubation; during this window, let the trade breathe
    
    # Breakeven buffer — minimum profit floor (covers commissions)
    fee_buffer_atr: float = 0.05
    min_profit_floor_ticks: int = 2                # Minimum +2 ticks on any "breakeven" exit (covers commissions)


def _allocate(total: int, pcts: list) -> list:
    """Allocate contracts across tranches."""
    if total <= 0:
        return [0] * len(pcts)
    if total < len(pcts):
        result = [0] * len(pcts)
        result[1] = total
        return result
    raw = [max(1, round(total * p)) for p in pcts]
    diff = total - sum(raw)
    raw[1] += diff
    if raw[1] < 1:
        raw[1] = 1
        while sum(raw) > total:
            idx = max(range(len(raw)), key=lambda i: raw[i] if i != 1 else 0)
            raw[idx] -= 1
    return raw


@dataclass
class RiskReduceTranche:
    size: int
    closed: bool = False
    close_price: float = 0.0
    
    def should_exit(self, unrealized_pts: float, atr: float, config: AdaptiveExitConfig) -> bool:
        if self.closed:
            return False
        # Fixed point target overrides ATR-based
        if config.fixed_target_pts_t1 is not None:
            return unrealized_pts >= config.fixed_target_pts_t1
        target = atr * config.risk_reduce_target_atr
        minimum = atr * config.risk_reduce_min_atr
        if unrealized_pts < minimum:
            return False
        return unrealized_pts >= target


@dataclass
class CoreTranche:
    """Core tranche — condition-based exits only, NO time decay.
    
    Exits on:
      1. Signal invalidation (trend reversal from strategy)
      2. Trailing stop after activation (wider than runner)
      3. Hard stop (handled by brackets, not here)
    
    NO breakeven timers, NO time decay. Let the trade work.
    """
    size: int
    closed: bool = False
    close_price: float = 0.0
    _best_pts: float = 0.0
    _activated: bool = False
    _trail_stop_pts: float = 0.0
    
    def update(self, unrealized_pts: float):
        self._best_pts = max(self._best_pts, unrealized_pts)
    
    def should_exit_signal(self, signal_exit_reason: Optional[str]) -> Optional[str]:
        if self.closed:
            return None
        return f"Signal invalidation: {signal_exit_reason}" if signal_exit_reason else None
    
    def should_exit_trail(self, unrealized_pts: float, atr: float, config: AdaptiveExitConfig) -> Optional[str]:
        """Trailing stop — wider than runner. Activates at 1.5x ATR profit."""
        if self.closed:
            return None
        
        # Core trail activation: 1.5x ATR (wider than runner's 0.75x)
        activation_pts = atr * 1.5
        # Core trail width: 1.5x ATR (wider than runner — gives more room)
        trail_width = atr * 1.5
        commission_floor = config.min_profit_floor_ticks * config.tick_size
        
        if not self._activated:
            if self._best_pts >= activation_pts:
                self._activated = True
                self._trail_stop_pts = max(commission_floor, self._best_pts - trail_width)
                logger.info(
                    f"[CORE] Trail activated at +{self._best_pts:.2f}pts "
                    f"(ATR={atr:.2f}, trail at +{self._trail_stop_pts:.2f}pts)"
                )
            return None
        
        # Update trail (only tightens, never loosens)
        new_trail = self._best_pts - trail_width
        if new_trail > self._trail_stop_pts:
            self._trail_stop_pts = new_trail
        
        # Check if price dropped below trail
        if unrealized_pts <= self._trail_stop_pts:
            return (
                f"Core trail stop: best={self._best_pts:+.2f}pts, "
                f"trail=+{self._trail_stop_pts:.2f}pts, now={unrealized_pts:+.2f}pts"
            )
        
        return None


@dataclass
class RunnerTranche:
    size: int
    closed: bool = False
    close_price: float = 0.0
    _activated: bool = False
    _best_pts: float = 0.0
    _best_pts_time: float = 0.0
    _trail_stop_pts: float = 0.0
    
    def update(self, unrealized_pts: float, now: float,
               atr: float, config: AdaptiveExitConfig,
               regime: str = "unknown",
               volume_ratio: float = 1.0) -> Optional[str]:
        if self.closed:
            return None
        
        if unrealized_pts > self._best_pts:
            self._best_pts = unrealized_pts
            self._best_pts_time = now
        
        activation_pts = atr * config.runner_activation_atr
        base_trail = atr * config.runner_base_trail_atr
        min_stop = atr * config.runner_min_stop_atr
        # Commission-cover floor: never exit below +2 ticks
        commission_floor = config.min_profit_floor_ticks * config.tick_size
        min_stop = max(min_stop, commission_floor / atr * config.runner_min_stop_atr) if atr > 0 else min_stop
        # Actually just use the raw tick floor
        min_stop_pts = max(atr * config.runner_min_stop_atr, commission_floor)
        
        if not self._activated:
            if self._best_pts >= activation_pts:
                self._activated = True
                self._trail_stop_pts = max(min_stop_pts, self._best_pts - base_trail)
                logger.info(
                    f"[RUNNER] Trail activated at +{self._best_pts:.2f}pts "
                    f"(ATR={atr:.2f}, trail at +{self._trail_stop_pts:.2f}pts)"
                )
            else:
                # Floor: runners never go negative once they've been meaningfully green
                if self._best_pts >= commission_floor and unrealized_pts <= 0:
                    return f"Runner floor hit: was +{self._best_pts:.2f}, now {unrealized_pts:.2f}"
                return None
        
        # Adaptive trail calculation
        trail = base_trail
        
        # ── Ratchet: tighten trail as profit grows ──────────────────
        if getattr(config, 'runner_ratchet_enabled', False) and config.runner_ratchet_step_atr > 0:
            step_pts = atr * config.runner_ratchet_step_atr
            if step_pts > 0:
                steps = int(self._best_pts / step_pts)
                if steps > 0:
                    trail *= config.runner_ratchet_tighten ** steps
                    # Don't ratchet below 2 ticks trail width
                    trail = max(trail, config.tick_size * 3)
        
        # Regime adjustment
        if regime in ("trending_fast", "trending_slow"):
            trail *= config.runner_trending_mult
        elif regime == "volatile":
            trail *= config.runner_volatile_mult
        elif regime == "ranging":
            trail *= config.runner_choppy_mult
        
        # Volume exhaustion
        if volume_ratio < 0.5:
            trail *= config.runner_volume_exhaustion_mult
        
        # Staleness — tighten if no new high for too long
        if now - self._best_pts_time > config.runner_stale_seconds:
            stale_trail = atr * config.runner_stale_trail_atr
            trail = min(trail, stale_trail)
        
        # Update trail stop (only moves up, never below commission floor)
        new_stop = max(min_stop_pts, self._best_pts - trail)
        self._trail_stop_pts = max(self._trail_stop_pts, new_stop)
        
        # Check if trail stop hit
        if unrealized_pts <= self._trail_stop_pts:
            return (
                f"Adaptive trail hit: best={self._best_pts:.2f}pts, "
                f"trail_stop={self._trail_stop_pts:.2f}pts, current={unrealized_pts:.2f}pts"
            )
        
        return None


class AdaptiveExitManager:
    """Manages 3-tranche exits with ATR-adaptive distances.
    
    Now includes Loser Intelligence Layer (LIL) from topstep3 Phase 8.40.
    LIL runs BEFORE tranche checks and can force-exit the entire position
    when failure patterns are detected (stale losers, giveback failures, etc.)
    
    Usage:
        mgr = AdaptiveExitManager(config, total_contracts=7, atr=8.5)
        # On each tick:
        exits = mgr.check_exits(unrealized_pts, hold_seconds, ...)
        for exit in exits:
            # execute partial close of exit.size contracts
    """
    
    def __init__(self, config: AdaptiveExitConfig, total_contracts: int, atr: float,
                 lil=None):
        self.config = config
        self.atr = atr
        self.entry_time = time.time()
        
        # Loser Intelligence Layer (optional — used for graduation state check only)
        # NOTE: Don't call on_trade_open here — caller manages LIL lifecycle
        self.lil = lil
        
        # Allocate tranches
        sizes = _allocate(
            total_contracts,
            [config.risk_reduce_pct, config.core_pct, config.runner_pct]
        )
        
        self.risk_reduce = RiskReduceTranche(size=sizes[0])
        self.core = CoreTranche(size=sizes[1])
        self.runner = RunnerTranche(size=sizes[2])
        
        self._remaining_contracts = total_contracts
        
        logger.info(
            f"[ADAPTIVE_EXIT] Initialized: {total_contracts} contracts "
            f"(RR={sizes[0]}, Core={sizes[1]}, Runner={sizes[2]}) "
            f"ATR={atr:.2f} → targets scaled accordingly"
        )
    
    def update_atr(self, atr: float):
        """Update ATR for live recalculation. Call periodically."""
        self.atr = atr
    
    @property
    def remaining(self) -> int:
        return self._remaining_contracts
    
    @property
    def all_closed(self) -> bool:
        return self._remaining_contracts <= 0
    
    def check_exits(
        self,
        unrealized_pts: float,
        signal_exit_reason: Optional[str] = None,
        regime: str = "unknown",
        volume_ratio: float = 1.0,
        session: str = "US_MIDDAY",
    ) -> list[TrancheExit]:
        """Check all tranches for exit conditions. Returns list of exits to execute."""
        
        now = time.time()
        hold_seconds = now - self.entry_time
        exits = []
        
        # ── LIL: Loser Intelligence Layer (pre-tranche) ──────────
        # If LIL detects a failure pattern, force-exit ALL remaining contracts
        if self.lil and self._remaining_contracts > 0:
            hard_stop_pts = self.atr * self.config.hard_stop_atr
            unrealized_dollars = unrealized_pts * self.config.point_value * self._remaining_contracts
            
            lil_exit, lil_reason, lil_phase = self.lil.check(
                unrealized_pts=unrealized_pts,
                unrealized_dollars=unrealized_dollars,
                atr=self.atr,
                session=session,
                hard_stop_pts=hard_stop_pts,
            )
            
            if lil_exit:
                # Force-close all remaining tranches
                for tranche, role in [(self.risk_reduce, "risk_reduce"), 
                                       (self.core, "core"), 
                                       (self.runner, "runner")]:
                    if not tranche.closed and tranche.size > 0:
                        tranche.closed = True
                        self._remaining_contracts -= tranche.size
                        exits.append(TrancheExit(
                            role=role,
                            size=tranche.size,
                            reason=f"LIL: {lil_reason}",
                            exit_type=ExitReason.HARD_STOP,
                            pnl_pts=unrealized_pts,
                        ))
                return exits
        
        # ── INCUBATION GATE (BEHAVIOR-BASED) ──────────────────────────
        # Incubation is controlled by GRADUATION, not just time.
        # Once the market proves the trade (LIL graduation), incubation ends
        # and all tranches can fire. Before graduation:
        #   - T1 (risk_reduce) can ALWAYS fire (it's profit-taking)
        #   - T2/T3 trails blocked until graduation or time expiry
        #
        # Graduation happens when: +1 ATR, +Npts, peak MFE >= threshold,
        # or 10min+ AND green. See LIL._update_phase() for full logic.
        
        # Check graduation state from LIL if available
        lil_graduated = self.lil.is_graduated if self.lil else False
        in_incubation = hold_seconds < self.config.incubation_seconds and not lil_graduated
        
        if in_incubation:
            # Still update tracking so tranches know their best/worst when incubation ends
            self.core.update(unrealized_pts)
            if unrealized_pts > self.runner._best_pts:
                self.runner._best_pts = unrealized_pts
                self.runner._best_pts_time = now
            
            # T1 (risk_reduce) can ALWAYS fire during incubation — it's profit-taking
            if not self.risk_reduce.closed and self.risk_reduce.size > 0:
                if self.risk_reduce.should_exit(unrealized_pts, self.atr, self.config):
                    self.risk_reduce.closed = True
                    self._remaining_contracts -= self.risk_reduce.size
                    exits.append(TrancheExit(
                        role="risk_reduce",
                        size=self.risk_reduce.size,
                        reason=f"Risk reduction during incubation at +{unrealized_pts:.2f}pts (target: {self.config.fixed_target_pts_t1 or self.atr * self.config.risk_reduce_target_atr:.2f}pts)",
                        exit_type=ExitReason.RISK_REDUCTION,
                        pnl_pts=unrealized_pts,
                    ))
            
            return exits  # T2/T3 still blocked until graduation or time expiry
        
        # ── Normal tranche checks (post-incubation only) ─────────
        
        # Risk Reduction
        if not self.risk_reduce.closed and self.risk_reduce.size > 0:
            if self.risk_reduce.should_exit(unrealized_pts, self.atr, self.config):
                self.risk_reduce.closed = True
                self._remaining_contracts -= self.risk_reduce.size
                exits.append(TrancheExit(
                    role="risk_reduce",
                    size=self.risk_reduce.size,
                    reason=f"Risk reduction at +{unrealized_pts:.2f}pts (target: {self.atr * self.config.risk_reduce_target_atr:.2f}pts)",
                    exit_type=ExitReason.RISK_REDUCTION,
                    pnl_pts=unrealized_pts,
                ))
        
        # Core (T2) — fixed target OR condition-based
        if not self.core.closed and self.core.size > 0:
            self.core.update(unrealized_pts)
            
            # Fixed point target (overrides signal/trail logic)
            if self.config.fixed_target_pts_t2 is not None and unrealized_pts >= self.config.fixed_target_pts_t2:
                self.core.closed = True
                self._remaining_contracts -= self.core.size
                exits.append(TrancheExit(
                    role="core",
                    size=self.core.size,
                    reason=f"Fixed target T2 at +{unrealized_pts:.2f}pts (target: {self.config.fixed_target_pts_t2}pts)",
                    exit_type=ExitReason.RISK_REDUCTION,
                    pnl_pts=unrealized_pts,
                ))
            else:
                # Signal invalidation (trend reversal)
                exit_reason = self.core.should_exit_signal(signal_exit_reason)
                if exit_reason:
                    self.core.closed = True
                    self._remaining_contracts -= self.core.size
                    exits.append(TrancheExit(
                        role="core",
                        size=self.core.size,
                        reason=exit_reason,
                        exit_type=ExitReason.SIGNAL_INVALID,
                        pnl_pts=unrealized_pts,
                    ))
                else:
                    # Trailing stop (wider than runner — 1.5x ATR activation, 1.5x ATR trail)
                    exit_reason = self.core.should_exit_trail(unrealized_pts, self.atr, self.config)
                    if exit_reason:
                        self.core.closed = True
                        self._remaining_contracts -= self.core.size
                        exits.append(TrancheExit(
                            role="core",
                            size=self.core.size,
                            reason=exit_reason,
                            exit_type=ExitReason.ADAPTIVE_TRAIL,
                            pnl_pts=unrealized_pts,
                        ))
        
        # Runner — if other tranches just exited, lock runner to commission-cover minimum
        if exits and not self.runner.closed and self.runner.size > 0:
            commission_floor = self.config.min_profit_floor_ticks * self.config.tick_size  # +2 ticks
            if not self.runner._activated:
                # Force-activate trail — floor at +2 ticks (not zero — covers commissions)
                self.runner._activated = True
                self.runner._trail_stop_pts = max(commission_floor, self.runner._trail_stop_pts)
                logger.info(
                    f"[RUNNER] Commission-cover lock: other tranches exited, "
                    f"runner stop raised to +{self.runner._trail_stop_pts:.2f}pts minimum "
                    f"(commission floor: +{commission_floor:.2f}pts)"
                )
            else:
                # Already trailing — ensure floor covers commissions
                self.runner._trail_stop_pts = max(commission_floor, self.runner._trail_stop_pts)
        
        if not self.runner.closed and self.runner.size > 0:
            # Fixed point target (T3) — overrides trailing logic
            if self.config.fixed_target_pts_t3 is not None and unrealized_pts >= self.config.fixed_target_pts_t3:
                self.runner.closed = True
                self._remaining_contracts -= self.runner.size
                exits.append(TrancheExit(
                    role="runner",
                    size=self.runner.size,
                    reason=f"Fixed target T3 at +{unrealized_pts:.2f}pts (target: {self.config.fixed_target_pts_t3}pts)",
                    exit_type=ExitReason.RISK_REDUCTION,
                    pnl_pts=unrealized_pts,
                ))
            else:
                exit_reason = self.runner.update(
                    unrealized_pts, now, self.atr, self.config,
                    regime=regime, volume_ratio=volume_ratio,
                )
                if exit_reason:
                    self.runner.closed = True
                    self._remaining_contracts -= self.runner.size
                    exits.append(TrancheExit(
                        role="runner",
                        size=self.runner.size,
                        reason=exit_reason,
                        exit_type=ExitReason.ADAPTIVE_TRAIL,
                        pnl_pts=unrealized_pts,
                    ))
        
        return exits
    
    def get_status(self) -> dict:
        """Get current status of all tranches."""
        return {
            "remaining": self._remaining_contracts,
            "atr": self.atr,
            "risk_reduce": {"size": self.risk_reduce.size, "closed": self.risk_reduce.closed},
            "core": {"size": self.core.size, "closed": self.core.closed, 
                     "best_pts": self.core._best_pts, "breakeven": self.core._breakeven_active},
            "runner": {"size": self.runner.size, "closed": self.runner.closed,
                       "activated": self.runner._activated, "best_pts": self.runner._best_pts,
                       "trail_stop": self.runner._trail_stop_pts},
        }
