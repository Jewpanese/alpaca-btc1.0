"""
TREND_FOLLOW — rides slow grinds and sustained directional moves.

Designed to work in ALL sessions including overnight/ETH where volume
is thin but price still trends. Fewer filters, wider stops, patience.

v2 — Loosened from original 7-filter AND-gate that never fired overnight.
"""

import logging
from typing import Optional
from .base import Strategy, Signal, MarketState, Direction

logger = logging.getLogger(__name__)


class TrendFollow(Strategy):
    """Trend following strategy for slow grinds and sustained moves.
    
    v2: Simplified entry logic. Uses 5m EMA stack as primary trend signal.
    Dropped dual-timeframe confirmation and VWAP requirement that blocked
    all overnight entries.
    """

    def __init__(
        self,
        max_ema_distance_atr: float = 2.5,     # How far from 5m EMA is too far (in ATR)
        min_adx: float = 18.0,                  # Lowered from 20 — slow grinds can have ADX 18-22
        atr_stop_multiplier: float = 2.5,
        atr_target_multiplier: float = 4.0,
        rsi_long_max: float = 78.0,
        rsi_short_min: float = 22.0,
        max_hold_seconds: int = 3600,           # 1 hour
        cooldown_bars: int = 10,
    ):
        super().__init__("TREND_FOLLOW")
        self.max_ema_distance_atr = max_ema_distance_atr
        self.min_adx = min_adx
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_target_mult = atr_target_multiplier
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.max_hold_seconds = max_hold_seconds
        self.cooldown_bars = cooldown_bars
        self._last_signal_bar = -cooldown_bars
        self._bar_count = 0
        self._diag_count = 0  # For periodic debug logging

    def _get_trend(self, state: MarketState) -> str:
        """Determine trend from 5m EMA stack. This is the ONLY trend gate."""
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0 and state.ema_5m_50 > 0:
            if state.ema_5m_9 > state.ema_5m_26 > state.ema_5m_50:
                return "bullish"
            elif state.ema_5m_9 < state.ema_5m_26 < state.ema_5m_50:
                return "bearish"
        # Partial trend: at least 9 vs 26 agree (catches early trends)
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0:
            spread = abs(state.ema_5m_9 - state.ema_5m_26)
            if spread > state.atr_14 * 0.3:  # Meaningful separation
                if state.ema_5m_9 > state.ema_5m_26:
                    return "bullish"
                else:
                    return "bearish"
        return "neutral"

    def should_enter(self, state: MarketState) -> Optional[Signal]:
        self._bar_count += 1
        self._diag_count += 1
        
        if state.atr_14 <= 0:
            return None
        
        # Cooldown
        if self._bar_count - self._last_signal_bar < self.cooldown_bars:
            return None
        
        trend = self._get_trend(state)
        
        # Periodic diagnostic logging (every 30 bars)
        if self._diag_count >= 30:
            self._diag_count = 0
            fast_ema = state.ema_5m_9 if state.ema_5m_9 > 0 else 0
            dist = abs(state.price - fast_ema) if fast_ema > 0 else 0
            max_dist = state.atr_14 * self.max_ema_distance_atr
            logger.info(
                f"[TREND_FOLLOW DIAG] trend={trend} | price={state.price:.2f} | "
                f"5m_9={state.ema_5m_9:.2f} 5m_26={state.ema_5m_26:.2f} 5m_50={state.ema_5m_50:.2f} | "
                f"dist_from_ema={dist:.2f}/{max_dist:.2f} | ADX={state.adx_14:.1f} | RSI={state.rsi_14:.1f}"
            )
        
        if trend == "neutral":
            return None
        
        # ADX check — need some directional movement, but lower bar than before
        if state.adx_14 > 0 and state.adx_14 < self.min_adx:
            return None
        
        atr = state.atr_14
        price = state.price
        
        fast_ema = state.ema_5m_9
        if fast_ema <= 0:
            return None
        
        distance_from_ema = abs(price - fast_ema)
        max_distance = atr * self.max_ema_distance_atr
        
        if trend == "bullish":
            # Price near the fast EMA — allows entries AT or slightly below EMA (pullback entries)
            # Only block if way too far in either direction
            if distance_from_ema > max_distance:
                return None
            
            # RSI check — wide band, trends run hot
            if state.rsi_14 > self.rsi_long_max:
                return None
            
            # 1m EMA sanity check — just make sure 9 EMA isn't deeply below 50 EMA
            # (softer than requiring full stack alignment)
            if state.ema_9 > 0 and state.ema_50 > 0:
                if state.ema_9 < state.ema_50 - atr * 0.5:
                    return None  # 1m structure strongly disagrees
            
            stop = price - (atr * self.atr_stop_mult)
            target = price + (atr * self.atr_target_mult)
            
            proximity_score = 1.0 - (distance_from_ema / max_distance)
            adx_score = min((state.adx_14 - self.min_adx) / 20.0, 0.5) if state.adx_14 > 0 else 0.2
            strength = min(0.3 + proximity_score * 0.4 + adx_score, 1.0)
            
            self._last_signal_bar = self._bar_count
            
            return Signal(
                direction=Direction.LONG,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"TREND_FOLLOW LONG: bullish grind, "
                    f"price={price:.2f} near 5m_9={fast_ema:.2f} "
                    f"({distance_from_ema:.1f}pts/{max_distance:.1f}max), "
                    f"ADX={state.adx_14:.0f}, RSI={state.rsi_14:.0f}"
                ),
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        elif trend == "bearish":
            # Same logic — near EMA, not overextended
            if distance_from_ema > max_distance:
                return None
            
            if state.rsi_14 < self.rsi_short_min:
                return None
            
            # 1m sanity — don't short if 1m structure is strongly bullish
            if state.ema_9 > 0 and state.ema_50 > 0:
                if state.ema_9 > state.ema_50 + atr * 0.5:
                    return None
            
            stop = price + (atr * self.atr_stop_mult)
            target = price - (atr * self.atr_target_mult)
            
            proximity_score = 1.0 - (distance_from_ema / max_distance)
            adx_score = min((state.adx_14 - self.min_adx) / 20.0, 0.5) if state.adx_14 > 0 else 0.2
            strength = min(0.3 + proximity_score * 0.4 + adx_score, 1.0)
            
            self._last_signal_bar = self._bar_count
            
            return Signal(
                direction=Direction.SHORT,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"TREND_FOLLOW SHORT: bearish grind, "
                    f"price={price:.2f} near 5m_9={fast_ema:.2f} "
                    f"({distance_from_ema:.1f}pts/{max_distance:.1f}max), "
                    f"ADX={state.adx_14:.0f}, RSI={state.rsi_14:.0f}"
                ),
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None

    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        """Exit on trend break — 5m EMAs uncross."""
        
        trend = self._get_trend(state)
        
        if direction == Direction.LONG and trend == "bearish":
            return "5m trend reversed to bearish — exiting trend follow"
        
        if direction == Direction.SHORT and trend == "bullish":
            return "5m trend reversed to bullish — exiting trend follow"
        
        # Early exit if 1m EMAs strongly disagree after 5 min
        if hold_time_seconds > 300 and state.ema_9 > 0 and state.ema_50 > 0:
            atr = state.atr_14 if state.atr_14 > 0 else 2.0
            if direction == Direction.LONG and state.ema_9 < state.ema_50 - atr * 0.3:
                return "1m EMAs flipped against LONG — early trend exit"
            if direction == Direction.SHORT and state.ema_9 > state.ema_50 + atr * 0.3:
                return "1m EMAs flipped against SHORT — early trend exit"
        
        return None
