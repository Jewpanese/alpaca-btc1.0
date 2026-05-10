"""Opening Range Breakout Strategy.

One of the oldest and most reliable intraday futures strategies.

WHY IT WORKS:
- First 15-30 min of RTH establishes institutional interest levels
- Breakout with volume = institutional commitment to direction
- Clear, mechanical entry/exit rules — no ambiguity
- Works because market structure concentrates orders at open

REFERENCES:
- Crabel, "Day Trading with Short-Term Price Patterns" (the original)
- Connors & Raschke, "Street Smarts" (variations)

PARAMETERS:
- range_minutes: Minutes after open to establish range (default 15)
- breakout_buffer_ticks: Extra ticks beyond range for entry (default 2)
- target_multiplier: Target as multiple of range size (default 2.0)
- stop_multiplier: Stop as fraction of range size (default 0.5)
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class OpeningRangeBreakout(Strategy):
    """Trade breakouts of the first N minutes range."""
    
    def __init__(
        self,
        range_minutes: int = 15,
        breakout_buffer_ticks: int = 2,
        target_multiplier: float = 2.0,
        stop_multiplier: float = 0.5,
        min_range_points: float = 1.0,   # Min range to be tradeable
        max_range_points: float = 10.0,  # Max range (too wide = too risky)
        min_volume_ratio: float = 1.2,   # Need above-avg volume on breakout
        max_hold_seconds: int = 600,     # 10 min max
        tick_size: float = 0.25,         # ES tick size
    ):
        super().__init__("ORB")
        self.range_minutes = range_minutes
        self.buffer = breakout_buffer_ticks * tick_size
        self.target_mult = target_multiplier
        self.stop_mult = stop_multiplier
        self.min_range = min_range_points
        self.max_range = max_range_points
        self.min_vol_ratio = min_volume_ratio
        self.max_hold = max_hold_seconds
        self._traded_today = False  # Only trade ORB once per day
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        """Enter on breakout of opening range."""
        
        # Only during NY session, after range is established
        if state.session != 'NY_OPEN':
            self._traded_today = False  # Reset for next day
            return None
        
        # Only trade ORB once per day
        if self._traded_today:
            return None
        
        # Need opening range to be set
        if state.opening_range_high is None or state.opening_range_low is None:
            return None
        
        # Must be after range period
        if state.minutes_since_open < self.range_minutes:
            return None
        
        # Don't trade too late (ORB is a morning strategy)
        if state.minutes_since_open > 90:  # Max 1.5 hours after open
            return None
        
        range_size = state.opening_range_high - state.opening_range_low
        
        # Range must be tradeable
        if range_size < self.min_range or range_size > self.max_range:
            return None
        
        # Volume confirmation
        if state.volume_ratio_5 < self.min_vol_ratio:
            return None
        
        # LONG breakout
        if state.price > state.opening_range_high + self.buffer:
            self._traded_today = True
            return Signal(
                direction=Direction.LONG,
                strength=min(state.volume_ratio_5 / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"ORB LONG: price {state.price} > range high {state.opening_range_high:.2f} + buffer",
                entry_price=state.price,
                stop_loss=state.price - (range_size * self.stop_mult),
                take_profit=state.price + (range_size * self.target_mult),
                max_hold_seconds=self.max_hold,
            )
        
        # SHORT breakout
        elif state.price < state.opening_range_low - self.buffer:
            self._traded_today = True
            return Signal(
                direction=Direction.SHORT,
                strength=min(state.volume_ratio_5 / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"ORB SHORT: price {state.price} < range low {state.opening_range_low:.2f} - buffer",
                entry_price=state.price,
                stop_loss=state.price + (range_size * self.stop_mult),
                take_profit=state.price - (range_size * self.target_mult),
                max_hold_seconds=self.max_hold,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        """Exit if price re-enters the range (failed breakout)."""
        
        if state.opening_range_high is None or state.opening_range_low is None:
            return None
        
        midpoint = (state.opening_range_high + state.opening_range_low) / 2
        
        # Failed breakout: price returns to midpoint of range
        if direction == Direction.LONG and state.price < midpoint:
            return "Failed breakout — price returned to range midpoint"
        
        if direction == Direction.SHORT and state.price > midpoint:
            return "Failed breakout — price returned to range midpoint"
        
        if hold_time_seconds > self.max_hold:
            return f"Max hold time exceeded ({hold_time_seconds:.0f}s)"
        
        return None
