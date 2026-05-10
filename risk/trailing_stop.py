"""Trailing Stop Manager — Adaptive, tiered exit management.

Replaces the simple breakeven stop in bot.py with a professional-grade
trailing stop system that adapts to volatility and locks in profits progressively.

Architecture:
    1. HARD RULES — non-negotiable safety rails (max stop, min stop)
    2. ATR-ADAPTIVE — stop distances scale with current volatility
    3. TIERED STAGES — progressive profit protection as trade moves in our favor
    4. TIME DECAY — kill dead trades that aren't moving

Stage progression:
    INITIAL  → entry, hard stop at max risk
    BREAKEVEN → price reached breakeven_trigger_pts → stop moves to entry + fee_buffer
    TRAILING  → price reached trailing_trigger_pts → trailing stop activates
    RUNNER    → price reached runner_trigger_pts → trail widens to let winners run

Each stage is one-way: once activated, we never go back to a less protective stage.
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class StopStage(Enum):
    """Progressive stop stages — only moves forward, never backward."""
    INITIAL = auto()
    BREAKEVEN = auto()
    TRAILING = auto()
    RUNNER = auto()


@dataclass
class TrailingStopConfig:
    """All trailing stop parameters.

    Points are in INSTRUMENT units. For BTC (this port): 1 point = $1 of BTC
    price move; 1 native contract = 0.01 BTC; point_value = $0.01/pt/contract.
    ATR multipliers scale these dynamically with volatility.

    Original ES defaults (point_value=50.0, _pts in 1-5 range) were ~100-200×
    too tight for BTC's $50-1000 3-min ATR profile. Defaults below are
    BTC-scaled; if porting back to ES/MES, override at instantiation.
    """
    # ── Hard Rules (non-negotiable) ─────────────────────────────────
    max_stop_pts: float = 800.0          # Never risk more than $800 of BTC move
    min_stop_pts: float = 50.0           # Never tighter than $50 (noise floor)

    # ── Stage Triggers (in $ of unrealized profit) ──────────────────
    breakeven_trigger_pts: float = 200.0   # Move to BE after +$200
    trailing_trigger_pts: float = 300.0    # Activate trailing after +$300
    runner_trigger_pts: float = 400.0      # Widen trail after +$400

    # ── Stop Distances ──────────────────────────────────────────────
    fee_buffer_pts: float = 5.0          # Added to BE to cover fees (Alpaca crypto: ~25bps; $5 is safe)
    trailing_distance_pts: float = 150.0  # Trail $150 behind in TRAILING stage
    runner_distance_pts: float = 250.0    # Wider $250 trail in RUNNER stage

    # ATR multipliers (used if ATR is available; clamped by hard rules)
    initial_stop_atr_mult: float = 2.0
    trailing_atr_mult: float = 1.0
    runner_atr_mult: float = 1.5

    # ── Time Decay ──────────────────────────────────────────────────
    time_breakeven_seconds: int = 480    # 8 min: if profit < threshold, move to BE
    time_tighten_seconds: int = 900      # 15 min: tighten trail
    time_breakeven_min_pts: float = 100.0  # Must be below +$100 profit for time-BE
    time_tighten_min_pts: float = 200.0   # Must be below +$200 for time-tighten
    time_tighten_trail_pts: float = 100.0  # Trail distance after time-tighten

    # ── Contract Specs ──────────────────────────────────────────────
    point_value: float = 0.01            # BTC: $0.01/pt/contract (1 contract = 0.01 BTC)


class TrailingStopManager:
    """Manages trailing stop for a single position.
    
    Create a new instance for each trade. Call update() on every tick/bar.
    The manager tracks the high watermark and current stage, and returns
    the current stop price.
    
    Usage:
        mgr = TrailingStopManager(config, entry_price, direction, atr)
        
        # On each tick:
        stop_price = mgr.update(current_price, current_atr)
        if stop_price is not None and price has crossed stop:
            exit_position()
    """
    
    def __init__(
        self,
        config: TrailingStopConfig,
        entry_price: float,
        direction: int,           # +1 for LONG, -1 for SHORT
        atr: float,               # ATR at time of entry
        entry_time: float = None, # Unix timestamp of entry
    ):
        self.config = config
        self.entry_price = entry_price
        self.direction = direction  # +1 LONG, -1 SHORT
        self.entry_time = entry_time or time.time()
        self.entry_atr = max(atr, 0.5)  # Floor ATR to prevent division issues
        
        # State tracking
        self.stage = StopStage.INITIAL
        self.high_watermark_pts: float = 0.0  # Best unrealized profit in points
        self.current_stop_price: float = 0.0  # The actual stop level
        
        # Calculate initial stop
        initial_distance = self._clamp_distance(
            self.entry_atr * config.initial_stop_atr_mult
        )
        if direction == 1:  # LONG
            self.current_stop_price = entry_price - initial_distance
        else:  # SHORT
            self.current_stop_price = entry_price + initial_distance
        
        logger.info(
            f"[TRAIL] Initialized: entry={entry_price:.2f} dir={'LONG' if direction == 1 else 'SHORT'} "
            f"ATR={atr:.2f} initial_stop={self.current_stop_price:.2f} "
            f"distance={initial_distance:.2f}pt stage=INITIAL"
        )
    
    def _clamp_distance(self, distance_pts: float) -> float:
        """Clamp stop distance between min and max hard rules."""
        return max(self.config.min_stop_pts, min(distance_pts, self.config.max_stop_pts))
    
    def _unrealized_pts(self, current_price: float) -> float:
        """Calculate unrealized P&L in points (positive = profit)."""
        return (current_price - self.entry_price) * self.direction
    
    def _move_stop(self, new_stop: float, reason: str) -> float:
        """Move stop only if it's MORE protective (closer to current price in profit direction).
        
        For LONG: stop can only move UP (higher price).
        For SHORT: stop can only move DOWN (lower price).
        
        Returns the (possibly unchanged) stop price.
        """
        if self.direction == 1:  # LONG — stop must go UP
            if new_stop > self.current_stop_price:
                old = self.current_stop_price
                self.current_stop_price = new_stop
                logger.info(
                    f"[TRAIL] Stop moved: {old:.2f} → {new_stop:.2f} "
                    f"(+{new_stop - old:.2f}pt) | {reason}"
                )
        else:  # SHORT — stop must go DOWN
            if new_stop < self.current_stop_price:
                old = self.current_stop_price
                self.current_stop_price = new_stop
                logger.info(
                    f"[TRAIL] Stop moved: {old:.2f} → {new_stop:.2f} "
                    f"({new_stop - old:.2f}pt) | {reason}"
                )
        
        return self.current_stop_price
    
    def _advance_stage(self, new_stage: StopStage):
        """Advance to a new stage (one-way only)."""
        if new_stage.value > self.stage.value:
            old = self.stage
            self.stage = new_stage
            logger.info(f"[TRAIL] Stage: {old.name} → {new_stage.name}")
    
    def update(self, current_price: float, current_atr: float = None) -> Optional[str]:
        """Update trailing stop with latest price.
        
        Args:
            current_price: Latest market price
            current_atr: Current ATR (updates adaptively; uses entry ATR if None)
        
        Returns:
            Exit reason string if stop is hit, None if position should stay open.
        """
        atr = max(current_atr or self.entry_atr, 0.5)
        unrealized = self._unrealized_pts(current_price)
        hold_seconds = time.time() - self.entry_time
        
        # Update high watermark
        self.high_watermark_pts = max(self.high_watermark_pts, unrealized)
        
        # ── Check if stop is hit FIRST ──────────────────────────────
        # For LONG: price <= stop = hit
        # For SHORT: price >= stop = hit
        stop_hit = False
        if self.direction == 1 and current_price <= self.current_stop_price:
            stop_hit = True
        elif self.direction == -1 and current_price >= self.current_stop_price:
            stop_hit = True
        
        if stop_hit:
            pnl_pts = unrealized
            pnl_dollars = pnl_pts * self.config.point_value
            return (
                f"Trailing stop hit @ {current_price:.2f} | "
                f"Stage: {self.stage.name} | "
                f"P&L: {pnl_pts:+.2f}pts (${pnl_dollars:+,.2f}) | "
                f"Watermark: {self.high_watermark_pts:.2f}pts | "
                f"Hold: {hold_seconds:.0f}s"
            )
        
        # ── Time-Based Adjustments ──────────────────────────────────
        # These fire based on elapsed time AND insufficient profit.
        # They don't change the stage — they just tighten the stop.
        
        # Time breakeven: 8 min in trade, profit < 1pt → move to breakeven
        if (hold_seconds >= self.config.time_breakeven_seconds
                and unrealized < self.config.time_breakeven_min_pts
                and self.stage == StopStage.INITIAL):
            be_price = self.entry_price + (self.config.fee_buffer_pts * self.direction)
            self._move_stop(be_price, 
                f"Time breakeven: {hold_seconds:.0f}s elapsed, only {unrealized:+.2f}pts profit")
            self._advance_stage(StopStage.BREAKEVEN)
        
        # Time tighten: 15 min in trade, profit < 2pt → tight 1pt trail
        if (hold_seconds >= self.config.time_tighten_seconds
                and unrealized < self.config.time_tighten_min_pts
                and unrealized > 0):
            trail_distance = self.config.time_tighten_trail_pts
            if self.direction == 1:
                tight_stop = current_price - trail_distance
            else:
                tight_stop = current_price + trail_distance
            self._move_stop(tight_stop,
                f"Time tighten: {hold_seconds:.0f}s elapsed, only {unrealized:+.2f}pts profit, "
                f"trail={trail_distance:.1f}pt")
        
        # ── Stage Progression ───────────────────────────────────────
        
        # INITIAL → BREAKEVEN: unrealized >= breakeven_trigger
        if (self.stage == StopStage.INITIAL
                and self.high_watermark_pts >= self.config.breakeven_trigger_pts):
            self._advance_stage(StopStage.BREAKEVEN)
            be_price = self.entry_price + (self.config.fee_buffer_pts * self.direction)
            self._move_stop(be_price,
                f"Breakeven trigger: watermark {self.high_watermark_pts:.2f}pts >= "
                f"{self.config.breakeven_trigger_pts}pt threshold")
        
        # BREAKEVEN → TRAILING: unrealized >= trailing_trigger
        if (self.stage == StopStage.BREAKEVEN
                and self.high_watermark_pts >= self.config.trailing_trigger_pts):
            self._advance_stage(StopStage.TRAILING)
        
        # TRAILING → RUNNER: unrealized >= runner_trigger
        if (self.stage == StopStage.TRAILING
                and self.high_watermark_pts >= self.config.runner_trigger_pts):
            self._advance_stage(StopStage.RUNNER)
        
        # ── Apply Trailing Logic Based on Current Stage ─────────────
        
        if self.stage == StopStage.TRAILING:
            # Use the WIDER of: fixed points or ATR-based
            # Gives the trade room to breathe through normal pullbacks
            atr_distance = atr * self.config.trailing_atr_mult
            fixed_distance = self.config.trailing_distance_pts
            trail = self._clamp_distance(max(atr_distance, fixed_distance))
            
            if self.direction == 1:
                new_stop = current_price - trail
            else:
                new_stop = current_price + trail
            
            self._move_stop(new_stop,
                f"Trailing: {trail:.2f}pt behind price "
                f"(ATR-based={atr_distance:.2f}, fixed={fixed_distance:.2f})")
        
        elif self.stage == StopStage.RUNNER:
            # Wider trail to let big winners breathe
            atr_distance = atr * self.config.runner_atr_mult
            fixed_distance = self.config.runner_distance_pts
            trail = self._clamp_distance(max(atr_distance, fixed_distance))
            
            if self.direction == 1:
                new_stop = current_price - trail
            else:
                new_stop = current_price + trail
            
            self._move_stop(new_stop,
                f"Runner: {trail:.2f}pt behind price "
                f"(ATR-based={atr_distance:.2f}, fixed={fixed_distance:.2f})")
        
        return None  # No exit
    
    def get_status(self) -> dict:
        """Get current trailing stop status for logging/dashboard."""
        return {
            "stage": self.stage.name,
            "entry_price": self.entry_price,
            "stop_price": self.current_stop_price,
            "high_watermark_pts": self.high_watermark_pts,
            "hold_seconds": time.time() - self.entry_time,
            "stop_distance_pts": abs(self.current_stop_price - self.entry_price),
        }
