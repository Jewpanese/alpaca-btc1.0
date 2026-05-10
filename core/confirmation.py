"""Entry Confirmation Timer — Don't Chase, Confirm.

The #1 lesson from the Iran war-gap (2026-03-01):
  Bot was RIGHT on direction but EARLY on timing.
  Every losing trade would've been a winner 30-60 seconds later.

How it works:
  1. Strategy fires a signal → signal enters "incubation"
  2. During incubation (15-30 seconds), we watch:
     - Does price hold above/below signal level? (not just dumping through)
     - Does a higher low (for longs) or lower high (for shorts) form?
     - Does volume confirm the move?
  3. If confirmed → enter
  4. If invalidated → discard signal, saved a loss

The incubation period scales with ATR:
  - Low vol (ATR < 2): 10 seconds (moves are slow, don't need much)
  - Normal vol (ATR 2-4): 20 seconds
  - High vol (ATR > 4): 30 seconds (more time for whipsaws to clear)
  - Extreme vol (ATR > 8): 45 seconds (war gaps, news events)

Fast-track bypass:
  - Signal strength > 0.9 AND regime is TRENDING_FAST → skip confirmation
  - This catches the obvious breakouts where waiting = missing the move
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
from strategies.base import Signal, Direction

logger = logging.getLogger(__name__)


@dataclass
class IncubatingSignal:
    """A signal in the confirmation window."""
    signal: Signal
    start_time: float
    start_price: float
    best_price: float          # Best price in signal direction during incubation
    worst_price: float         # Worst price (against signal) during incubation
    confirmation_seconds: float
    prices: list = field(default_factory=list)  # Price samples during incubation
    confirmed: bool = False
    invalidated: bool = False
    invalidation_reason: str = ""
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def is_expired(self) -> bool:
        return self.elapsed > self.confirmation_seconds * 2  # 2x window = stale


@dataclass
class ConfirmationConfig:
    """Configuration for entry confirmation."""
    
    # Base confirmation time (seconds)
    base_seconds: float = 20.0
    
    # ATR scaling
    low_vol_atr: float = 4.0          # Below this = low vol
    normal_vol_atr: float = 8.0       # Below this = normal
    high_vol_atr: float = 15.0        # Below this = high vol, above = extreme
    
    low_vol_seconds: float = 8.0
    normal_vol_seconds: float = 15.0
    high_vol_seconds: float = 25.0
    extreme_vol_seconds: float = 35.0
    
    # Confirmation criteria
    # For longs: price must make a higher low (not just a straight dump)
    # For shorts: price must make a lower high
    require_higher_low: bool = True
    
    # Maximum adverse excursion during confirmation (ATR multiple)
    # If price moves this far against the signal, invalidate
    max_adverse_atr: float = 1.0
    
    # Minimum favorable excursion to confirm early
    min_favorable_atr: float = 0.20
    
    # Fast-track: skip confirmation for very strong signals in trending regimes
    fast_track_strength: float = 0.90
    fast_track_regimes: list = field(default_factory=lambda: ["trending_fast"])
    
    # Max signals to incubate simultaneously
    max_incubating: int = 3


class ConfirmationTimer:
    """Manages signal incubation and confirmation."""
    
    def __init__(self, config: ConfirmationConfig = None):
        self.config = config or ConfirmationConfig()
        self._incubating: List[IncubatingSignal] = []
    
    def _calc_confirmation_time(self, atr: float) -> float:
        """Calculate confirmation window based on current ATR."""
        if atr < self.config.low_vol_atr:
            return self.config.low_vol_seconds
        elif atr < self.config.normal_vol_atr:
            return self.config.normal_vol_seconds
        elif atr < self.config.high_vol_atr:
            return self.config.high_vol_seconds
        else:
            return self.config.extreme_vol_seconds
    
    def should_fast_track(self, signal: Signal, regime: str) -> bool:
        """Check if signal qualifies for fast-track (skip confirmation)."""
        return (
            signal.strength >= self.config.fast_track_strength
            and regime in self.config.fast_track_regimes
        )
    
    def submit_signal(self, signal: Signal, current_price: float, atr: float,
                      regime: str = "unknown") -> Optional[Signal]:
        """Submit a signal for confirmation.
        
        Returns:
            Signal immediately if fast-tracked, None if incubating.
        """
        # Fast-track check
        if self.should_fast_track(signal, regime):
            logger.info(
                f"[CONFIRM] Fast-track: {signal.strategy_name} {signal.direction.name} "
                f"strength={signal.strength:.2f} in {regime}"
            )
            return signal
        
        # Check if we already have a signal from this strategy incubating
        for inc in self._incubating:
            if (inc.signal.strategy_name == signal.strategy_name 
                    and inc.signal.direction == signal.direction
                    and not inc.invalidated):
                # Already incubating same signal, update best/worst
                return None
        
        # Clean up stale/invalidated signals
        self._incubating = [
            s for s in self._incubating 
            if not s.is_expired and not s.invalidated
        ]
        
        # Limit concurrent incubations
        if len(self._incubating) >= self.config.max_incubating:
            # Drop oldest
            self._incubating.pop(0)
        
        conf_time = self._calc_confirmation_time(atr)
        
        inc = IncubatingSignal(
            signal=signal,
            start_time=time.time(),
            start_price=current_price,
            best_price=current_price,
            worst_price=current_price,
            confirmation_seconds=conf_time,
            prices=[current_price],
        )
        self._incubating.append(inc)
        
        logger.info(
            f"[CONFIRM] Incubating: {signal.strategy_name} {signal.direction.name} "
            f"@ {current_price:.2f} for {conf_time:.0f}s (ATR={atr:.2f})"
        )
        return None
    
    def update(self, current_price: float, atr: float) -> List[Signal]:
        """Update all incubating signals with current price. Returns confirmed signals."""
        confirmed = []
        
        for inc in self._incubating:
            if inc.confirmed or inc.invalidated:
                continue
            
            inc.prices.append(current_price)
            
            # Update best/worst
            if inc.signal.direction == Direction.LONG:
                inc.best_price = max(inc.best_price, current_price)
                inc.worst_price = min(inc.worst_price, current_price)
                adverse = inc.start_price - inc.worst_price
                favorable = inc.best_price - inc.start_price
            else:
                inc.best_price = min(inc.best_price, current_price)
                inc.worst_price = max(inc.worst_price, current_price)
                adverse = inc.worst_price - inc.start_price
                favorable = inc.start_price - inc.best_price
            
            # Check invalidation: too much adverse movement
            max_adverse = atr * self.config.max_adverse_atr
            if adverse > max_adverse:
                inc.invalidated = True
                inc.invalidation_reason = (
                    f"Adverse excursion {adverse:.2f}pts > {max_adverse:.2f}pts limit"
                )
                logger.info(
                    f"[CONFIRM] INVALIDATED: {inc.signal.strategy_name} "
                    f"{inc.signal.direction.name} — {inc.invalidation_reason}"
                )
                continue
            
            # Check early confirmation: strong favorable move
            min_favorable = atr * self.config.min_favorable_atr
            if favorable >= min_favorable and inc.elapsed >= 5.0:
                inc.confirmed = True
                logger.info(
                    f"[CONFIRM] EARLY CONFIRM: {inc.signal.strategy_name} "
                    f"{inc.signal.direction.name} — favorable {favorable:.2f}pts "
                    f"in {inc.elapsed:.1f}s"
                )
                # Update signal entry price to current (better fill after confirmation)
                inc.signal.entry_price = current_price
                confirmed.append(inc.signal)
                continue
            
            # Check time-based confirmation
            if inc.elapsed >= inc.confirmation_seconds:
                # Price held through the window — confirmed
                if inc.signal.direction == Direction.LONG:
                    held = current_price >= inc.start_price - (atr * 0.15)
                else:
                    held = current_price <= inc.start_price + (atr * 0.15)
                
                if held:
                    inc.confirmed = True
                    logger.info(
                        f"[CONFIRM] TIME CONFIRM: {inc.signal.strategy_name} "
                        f"{inc.signal.direction.name} — held for {inc.elapsed:.1f}s"
                    )
                    inc.signal.entry_price = current_price
                    confirmed.append(inc.signal)
                else:
                    inc.invalidated = True
                    inc.invalidation_reason = "Price didn't hold through confirmation window"
                    logger.info(
                        f"[CONFIRM] EXPIRED: {inc.signal.strategy_name} — didn't hold"
                    )
        
        # Clean up confirmed/invalidated
        self._incubating = [
            s for s in self._incubating
            if not s.confirmed and not s.invalidated and not s.is_expired
        ]
        
        return confirmed
    
    @property
    def pending_count(self) -> int:
        return len(self._incubating)
    
    def get_status(self) -> list:
        """Get status of all incubating signals."""
        return [
            {
                "strategy": s.signal.strategy_name,
                "direction": s.signal.direction.name,
                "elapsed": s.elapsed,
                "target_seconds": s.confirmation_seconds,
                "start_price": s.start_price,
                "best": s.best_price,
                "worst": s.worst_price,
            }
            for s in self._incubating
            if not s.confirmed and not s.invalidated
        ]
