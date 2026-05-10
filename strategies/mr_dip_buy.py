"""MR_DIP_BUY — Mean Reversion Dip Buy.

Star strategy from State Machine v3 backtest: +$44,609 across all 4 test periods.
Buys oversold bounces using IBS (Internal Bar Strength) + RSI(5) + VWAP.

Rules:
  1. Session IBS < 0.35 (price near session low)
  2. RSI(5) < 35 (short-term oversold)
  3. RSI(5) declining (current < previous)
  4. Price at or below VWAP (+ small ATR tolerance)
  5. Bullish bar (close > open — confirming bounce)

LONG ONLY. This is a mean reversion strategy — it buys dips.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class MRDipBuy(Strategy):
    """Mean reversion dip buy — buy oversold bounces."""

    def __init__(
        self,
        ibs_threshold: float = 0.35,
        rsi5_threshold: float = 35.0,
        vwap_tolerance_atr: float = 0.5,
        atr_stop_mult: float = 1.0,
        max_hold_seconds: int = 1800,  # 30 min — MR should resolve
    ):
        super().__init__("MR_DIP_BUY")
        self.ibs_threshold = ibs_threshold
        self.rsi5_threshold = rsi5_threshold
        self.vwap_tolerance_atr = vwap_tolerance_atr
        self.atr_stop_mult = atr_stop_mult
        self.max_hold_seconds = max_hold_seconds
        self._prev_rsi_5 = 50.0

    def should_enter(self, state: MarketState) -> Optional[Signal]:
        atr = state.atr_14
        if atr <= 0 or state.vwap <= 0:
            return None

        price = state.price
        rsi_5 = getattr(state, 'rsi_5', 50.0)

        # 1. Session IBS < threshold (price near session low)
        session_range = state.daily_high - state.daily_low
        if session_range <= 0.5:
            return None
        ibs = (price - state.daily_low) / session_range
        if ibs >= self.ibs_threshold:
            self._prev_rsi_5 = rsi_5
            return None

        # 2. RSI(5) < threshold (oversold)
        if rsi_5 >= self.rsi5_threshold:
            self._prev_rsi_5 = rsi_5
            return None

        # 3. RSI(5) declining
        if rsi_5 >= self._prev_rsi_5:
            self._prev_rsi_5 = rsi_5
            return None

        # 4. Price at or below VWAP (small tolerance)
        if price > state.vwap + self.vwap_tolerance_atr * atr:
            self._prev_rsi_5 = rsi_5
            return None

        # 5. Bullish bar (close > open)
        if state.bar_close <= state.bar_open:
            self._prev_rsi_5 = rsi_5
            return None

        # All conditions met — generate signal
        strength = 0.7 + (self.rsi5_threshold - rsi_5) / 100 + (self.ibs_threshold - ibs) / 0.5
        strength = min(strength, 1.0)

        stop = price - atr * self.atr_stop_mult
        # T1 target: VWAP return or 1.5 ATR, whichever is closer
        target_vwap = state.vwap
        target_atr = price + atr * 1.5
        target = min(target_vwap, target_atr) if target_vwap > price else target_atr

        self._prev_rsi_5 = rsi_5

        return Signal(
            direction=Direction.LONG,
            strength=strength,
            strategy_name=self.name,
            reason=f"MR_DIP_BUY: IBS={ibs:.2f}, RSI5={rsi_5:.0f}, VWAP={state.vwap:.2f}",
            entry_price=price,
            stop_loss=stop,
            take_profit=target,
            max_hold_seconds=self.max_hold_seconds,
        )

    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if hold_time_seconds > self.max_hold_seconds:
            return f"MR time stop ({hold_time_seconds:.0f}s)"

        rsi_5 = getattr(state, 'rsi_5', 50.0)

        # Exit when RSI(5) recovers above 50 (mean reversion complete)
        if rsi_5 > 50:
            return f"RSI(5) recovered to {rsi_5:.0f}"

        # Exit if price returns to VWAP
        if state.vwap > 0 and state.price >= state.vwap:
            return f"VWAP return ({state.price:.2f} >= {state.vwap:.2f})"

        return None
