"""Support/Resistance Breakout Strategy.

Trades breakouts of prior session high/low with volume confirmation.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class SRBreakout(Strategy):
    """Breakout of prior session S/R levels."""
    
    def __init__(self, hold_bars: int = 3, min_volume_ratio: float = 1.3,
                 atr_stop_mult: float = 0.5, atr_target_mult: float = 3.0,
                 max_hold_seconds: int = 1200):
        super().__init__("SR_BREAKOUT")
        self.hold_bars = hold_bars
        self.min_volume_ratio = min_volume_ratio
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_seconds = max_hold_seconds
        
        # S/R levels (from prior session)
        self._resistance: float = 0.0
        self._support: float = 0.0
        self._last_session: str = ""
        self._bars_above_resistance: int = 0
        self._bars_below_support: int = 0
    
    def _update_levels(self, state: MarketState):
        """Reset S/R levels on session change."""
        if state.session != self._last_session and self._last_session:
            # Prior session high/low become new S/R
            if state.daily_high > 0 and state.daily_low > 0:
                self._resistance = state.daily_high
                self._support = state.daily_low
            self._bars_above_resistance = 0
            self._bars_below_support = 0
        self._last_session = state.session
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.atr_14 <= 0:
            return None
        
        self._update_levels(state)
        
        if self._resistance == 0 or self._support == 0:
            return None
        
        # Count bars holding above/below levels
        if state.price > self._resistance:
            self._bars_above_resistance += 1
        else:
            self._bars_above_resistance = 0
        
        if state.price < self._support:
            self._bars_below_support += 1
        else:
            self._bars_below_support = 0
        
        # LONG breakout
        if (self._bars_above_resistance >= self.hold_bars
                and state.volume_ratio_5 > self.min_volume_ratio
                and state.price > state.vwap):
            stop = self._resistance - state.atr_14 * self.atr_stop_mult
            target = state.price + state.atr_14 * self.atr_target_mult
            self._bars_above_resistance = 0  # prevent re-entry
            return Signal(
                direction=Direction.LONG,
                strength=min(state.volume_ratio_5 / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"SR breakout LONG: broke {self._resistance:.2f}, vol={state.volume_ratio_5:.1f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        # SHORT breakdown
        if (self._bars_below_support >= self.hold_bars
                and state.volume_ratio_5 > self.min_volume_ratio
                and state.price < state.vwap):
            stop = self._support + state.atr_14 * self.atr_stop_mult
            target = state.price - state.atr_14 * self.atr_target_mult
            self._bars_below_support = 0
            return Signal(
                direction=Direction.SHORT,
                strength=min(state.volume_ratio_5 / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"SR breakdown SHORT: broke {self._support:.2f}, vol={state.volume_ratio_5:.1f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"Max hold exceeded ({hold_time_seconds:.0f}s)"
        
        if direction == Direction.LONG:
            # Breakout failed: price fell back below the resistance level
            if self._resistance > 0 and state.price < self._resistance - state.atr_14 * 0.25:
                return f"Failed breakout: price {state.price:.2f} back below resistance {self._resistance:.2f}"
            # Lost VWAP support
            if state.price < state.vwap and state.price < entry_price:
                return f"Lost VWAP ({state.vwap:.2f}) + below entry — breakout invalidated"
        
        elif direction == Direction.SHORT:
            # Breakdown failed: price reclaimed support level
            if self._support > 0 and state.price > self._support + state.atr_14 * 0.25:
                return f"Failed breakdown: price {state.price:.2f} back above support {self._support:.2f}"
            # Reclaimed VWAP
            if state.price > state.vwap and state.price > entry_price:
                return f"Reclaimed VWAP ({state.vwap:.2f}) + above entry — breakdown invalidated"
        
        return None
