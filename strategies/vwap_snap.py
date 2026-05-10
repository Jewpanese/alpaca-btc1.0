"""VWAP_SNAP — VWAP Snapback Strategy.

Buys extreme extensions below VWAP. Complementary to MR_DIP_BUY but
requires deeper oversold conditions and elevated volume.

Rules:
  1. Price below VWAP - 1.0*ATR (significantly extended)
  2. RSI(5) < 30 (deeply oversold)
  3. Bullish bar (close > open)
  4. Volume ratio > 1.0 (elevated activity = institutional interest)

LONG ONLY. Target is VWAP return.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class VWAPSnap(Strategy):
    """VWAP snapback — buy extreme extensions below VWAP."""

    def __init__(
        self,
        vwap_distance_atr: float = 1.0,
        rsi5_threshold: float = 30.0,
        min_volume_ratio: float = 1.0,
        atr_stop_mult: float = 1.0,
        max_hold_seconds: int = 1800,
    ):
        super().__init__("VWAP_SNAP")
        self.vwap_distance_atr = vwap_distance_atr
        self.rsi5_threshold = rsi5_threshold
        self.min_volume_ratio = min_volume_ratio
        self.atr_stop_mult = atr_stop_mult
        self.max_hold_seconds = max_hold_seconds

    def should_enter(self, state: MarketState) -> Optional[Signal]:
        atr = state.atr_14
        if atr <= 0 or state.vwap <= 0:
            return None

        price = state.price
        rsi_5 = getattr(state, 'rsi_5', 50.0)

        # 1. Price below VWAP - N*ATR
        if price > state.vwap - self.vwap_distance_atr * atr:
            return None

        # 2. RSI(5) deeply oversold
        if rsi_5 >= self.rsi5_threshold:
            return None

        # 3. Bullish bar
        if state.bar_close <= state.bar_open:
            return None

        # 4. Volume elevated
        if state.volume_ratio_5 <= self.min_volume_ratio:
            return None

        strength = 0.7 + (25 - rsi_5) / 100
        strength = min(max(strength, 0.5), 1.0)

        stop = price - atr * self.atr_stop_mult
        target = state.vwap  # Target: VWAP return

        return Signal(
            direction=Direction.LONG,
            strength=strength,
            strategy_name=self.name,
            reason=f"VWAP_SNAP: RSI5={rsi_5:.0f}, dist={((state.vwap - price)/atr):.1f}ATR, vol={state.volume_ratio_5:.1f}x",
            entry_price=price,
            stop_loss=stop,
            take_profit=target,
            max_hold_seconds=self.max_hold_seconds,
        )

    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"VWAP_SNAP time stop ({hold_time_seconds:.0f}s)"

        # Exit at VWAP
        if state.vwap > 0 and state.price >= state.vwap:
            return f"VWAP return ({state.price:.2f} >= {state.vwap:.2f})"

        return None
