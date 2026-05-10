"""Delta Divergence Strategy — Order Flow.

Detects price/delta divergences for reversal entries.
"""

from typing import Optional
from collections import deque
from .base import Strategy, Signal, MarketState, Direction


class DeltaDivergence(Strategy):
    """Trade divergences between price and cumulative delta."""
    
    def __init__(self, lookback: int = 10, atr_stop_mult: float = 2.0,
                 atr_target_mult: float = 3.0, max_hold_seconds: int = 900):
        super().__init__("DELTA_DIV")
        self.lookback = lookback
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_seconds = max_hold_seconds
        self._price_history: deque = deque(maxlen=lookback)
        self._delta_history: deque = deque(maxlen=lookback)
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.atr_14 <= 0:
            return None
        
        self._price_history.append(state.price)
        self._delta_history.append(state.cumulative_delta)
        
        if len(self._price_history) < self.lookback:
            return None
        
        prices = list(self._price_history)
        deltas = list(self._delta_history)
        
        current_price = prices[-1]
        current_delta = deltas[-1]
        max_price = max(prices[:-1])
        min_price = min(prices[:-1])
        prev_delta_at_max = deltas[prices[:-1].index(max_price)]
        prev_delta_at_min = deltas[prices[:-1].index(min_price)]
        
        # BEARISH divergence: new price high but delta declining
        if (current_price > max_price
                and current_delta < prev_delta_at_max
                and state.rsi_14 > 60):
            stop = state.price + state.atr_14 * self.atr_stop_mult
            target = state.price - state.atr_14 * self.atr_target_mult
            return Signal(
                direction=Direction.SHORT,
                strength=0.6,
                strategy_name=self.name,
                reason=f"Bearish delta div: new high but delta falling, RSI={state.rsi_14:.0f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        # BULLISH divergence: new price low but delta rising
        if (current_price < min_price
                and current_delta > prev_delta_at_min
                and state.rsi_14 < 40):
            stop = state.price - state.atr_14 * self.atr_stop_mult
            target = state.price + state.atr_14 * self.atr_target_mult
            return Signal(
                direction=Direction.LONG,
                strength=0.6,
                strategy_name=self.name,
                reason=f"Bullish delta div: new low but delta rising, RSI={state.rsi_14:.0f}",
                entry_price=state.price, stop_loss=stop, take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"Max hold exceeded ({hold_time_seconds:.0f}s)"
        
        if direction == Direction.SHORT:
            # Thesis: overbought reversal. Dead if:
            # 1. RSI re-accelerating (buyers not exhausted)
            if state.rsi_14 > 75 and state.price > entry_price + state.atr_14 * 0.5:
                return f"RSI re-accelerating ({state.rsi_14:.0f}) above entry — short thesis dead"
            # 2. Price reclaimed entry + buffer and holding (failed breakdown)
            if state.price > entry_price + state.atr_14 * 0.75:
                return f"Price {state.price:.2f} reclaimed entry+0.75ATR ({entry_price + state.atr_14 * 0.75:.2f}) — failed breakdown"
            # 3. Delta flipped — buyers now in control (divergence resolved)
            if len(self._delta_history) >= 3:
                recent = list(self._delta_history)[-3:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    if state.price > entry_price:
                        return f"Delta rising for 3 bars + price above entry — divergence resolved"
        
        elif direction == Direction.LONG:
            # Thesis: oversold reversal. Dead if:
            if state.rsi_14 < 25 and state.price < entry_price - state.atr_14 * 0.5:
                return f"RSI re-declining ({state.rsi_14:.0f}) below entry — long thesis dead"
            if state.price < entry_price - state.atr_14 * 0.75:
                return f"Price {state.price:.2f} broke entry-0.75ATR ({entry_price - state.atr_14 * 0.75:.2f}) — failed breakout"
            if len(self._delta_history) >= 3:
                recent = list(self._delta_history)[-3:]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    if state.price < entry_price:
                        return f"Delta falling for 3 bars + price below entry — divergence resolved"
        
        return None
