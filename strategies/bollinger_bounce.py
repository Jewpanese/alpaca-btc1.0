"""Bollinger Band Bounce — Mean Reversion Scalper.

Trades bounces off Bollinger Bands in ranging markets only.
Explicitly refuses to trade when ADX > 25 (trending).
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class BollingerBounce(Strategy):
    """Mean reversion off Bollinger Bands in ranging markets."""
    
    def __init__(self, adx_max: float = 28, atr_stop_mult: float = 1.5,
                 max_hold_seconds: int = 600):
        super().__init__("BB_BOUNCE")
        self.adx_max = adx_max
        self.atr_stop_mult = atr_stop_mult
        self.max_hold_seconds = max_hold_seconds
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.atr_14 <= 0 or state.bb_upper == 0:
            return None
        
        # Only trade in ranging markets
        if state.adx_14 > self.adx_max:
            return None
        
        # Trade in all sessions — ranging markets happen anytime
        # Risk manager handles session-specific filtering if needed
        
        # LONG: price at lower band + RSI oversold
        if state.price <= state.bb_lower and state.rsi_14 < 40:
            stop = state.price - state.atr_14 * self.atr_stop_mult
            target = state.bb_middle
            if target <= state.price:
                return None
            return Signal(
                direction=Direction.LONG,
                strength=min((40 - state.rsi_14) / 40, 1.0),
                strategy_name=self.name,
                reason=f"BB bounce LONG: price<=BB_lower, RSI={state.rsi_14:.0f}, ADX={state.adx_14:.0f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        # SHORT: price at upper band + RSI overbought
        if state.price >= state.bb_upper and state.rsi_14 > 60:
            stop = state.price + state.atr_14 * self.atr_stop_mult
            target = state.bb_middle
            if target >= state.price:
                return None
            return Signal(
                direction=Direction.SHORT,
                strength=min((state.rsi_14 - 60) / 40, 1.0),
                strategy_name=self.name,
                reason=f"BB bounce SHORT: price>=BB_upper, RSI={state.rsi_14:.0f}, ADX={state.adx_14:.0f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"Max hold exceeded ({hold_time_seconds:.0f}s)"
        
        # Exit at middle band
        if direction == Direction.LONG and state.price >= state.bb_middle:
            return "Reached BB middle (target)"
        if direction == Direction.SHORT and state.price <= state.bb_middle:
            return "Reached BB middle (target)"
        
        return None
