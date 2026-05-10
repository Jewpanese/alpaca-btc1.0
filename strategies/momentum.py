"""Momentum / Trend Following Strategy.

Rides trends using EMA crossovers confirmed by ADX and VWAP.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class Momentum(Strategy):
    """Trend following with EMA cross + ADX confirmation."""
    
    def __init__(self, adx_entry: float = 25, adx_exit: float = 20,
                 atr_stop_mult: float = 2.0, atr_target_mult: float = 4.0,
                 atr_trail_mult: float = 1.5, max_hold_seconds: int = 1800):
        super().__init__("MOMENTUM")
        self.adx_entry = adx_entry
        self.adx_exit = adx_exit
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.atr_trail_mult = atr_trail_mult
        self.max_hold_seconds = max_hold_seconds
        self._last_cross_dir: Optional[str] = None  # 'long' or 'short'
        self._prev_ema_9: float = 0.0
        self._prev_ema_21: float = 0.0
        self._trail_high: float = 0.0
        self._trail_low: float = float('inf')
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.atr_14 <= 0 or state.ema_9 == 0 or state.ema_21 == 0:
            return None
        
        # First bar: just capture EMA values, don't detect crosses
        if self._prev_ema_9 == 0.0:
            self._prev_ema_9 = state.ema_9
            self._prev_ema_21 = state.ema_21
            return None
        
        # Detect fresh EMA cross
        cross_long = self._prev_ema_9 <= self._prev_ema_21 and state.ema_9 > state.ema_21
        cross_short = self._prev_ema_9 >= self._prev_ema_21 and state.ema_9 < state.ema_21
        
        self._prev_ema_9 = state.ema_9
        self._prev_ema_21 = state.ema_21
        
        if state.adx_14 < self.adx_entry:
            return None
        
        if cross_long and state.price > state.vwap and self._last_cross_dir != 'long':
            self._last_cross_dir = 'long'
            self._trail_high = state.price
            stop = state.price - state.atr_14 * self.atr_stop_mult
            target = state.price + state.atr_14 * self.atr_target_mult
            return Signal(
                direction=Direction.LONG, strength=min(state.adx_14 / 50, 1.0),
                strategy_name=self.name,
                reason=f"EMA cross LONG: ADX={state.adx_14:.0f}, price>VWAP",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        if cross_short and state.price < state.vwap and self._last_cross_dir != 'short':
            self._last_cross_dir = 'short'
            self._trail_low = state.price
            stop = state.price + state.atr_14 * self.atr_stop_mult
            target = state.price - state.atr_14 * self.atr_target_mult
            return Signal(
                direction=Direction.SHORT, strength=min(state.adx_14 / 50, 1.0),
                strategy_name=self.name,
                reason=f"EMA cross SHORT: ADX={state.adx_14:.0f}, price<VWAP",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"Max hold exceeded ({hold_time_seconds:.0f}s)"
        
        if state.adx_14 < self.adx_exit:
            return f"ADX dropped below {self.adx_exit} ({state.adx_14:.0f})"
        
        # EMA crossback
        if direction == Direction.LONG and state.ema_9 < state.ema_21:
            return "EMA crossback (bearish)"
        if direction == Direction.SHORT and state.ema_9 > state.ema_21:
            return "EMA crossback (bullish)"
        
        # Trailing stop
        atr = state.atr_14
        if direction == Direction.LONG:
            self._trail_high = max(self._trail_high, state.price)
            if state.price < self._trail_high - atr * self.atr_trail_mult:
                return f"Trailing stop hit ({state.price:.2f} < {self._trail_high:.2f} - trail)"
        else:
            self._trail_low = min(self._trail_low, state.price)
            if state.price > self._trail_low + atr * self.atr_trail_mult:
                return f"Trailing stop hit ({state.price:.2f} > {self._trail_low:.2f} + trail)"
        
        return None
