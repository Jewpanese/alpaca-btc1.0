"""EMA Rejection Strategy — Ben's bread and butter.

Trades rejections off key EMAs in the direction of the trend.
Based on months of manual trading with consistent results.

Setup:
1. Trend established (EMAs stacked on higher timeframe)
2. Price retraces TO a key EMA (9 on 3m, or 26 on 1m)
3. Rejection confirmed (price moves back in trend direction)
4. Enter with stop beyond the EMA

This is a pullback strategy — buying dips in uptrends,
selling rips in downtrends. The EMA acts as dynamic S/R.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class EMAReject(Strategy):
    """EMA rejection / pullback strategy."""

    def __init__(
        self,
        # How close price must get to the EMA to count as a "touch" (in points)
        touch_threshold: float = 2.0,
        # How far price must bounce from EMA to confirm rejection (in points)
        reject_threshold: float = 1.5,
        # ATR multipliers for stops and targets
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 2.5,
        # Max hold
        max_hold_seconds: int = 600,
        # Min RSI conditions (don't buy overbought, don't sell oversold)
        rsi_long_max: float = 65.0,
        rsi_short_min: float = 35.0,
    ):
        super().__init__("EMA_REJECT")
        self.touch_threshold = touch_threshold
        self.reject_threshold = reject_threshold
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_target_mult = atr_target_multiplier
        self.max_hold_seconds = max_hold_seconds
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        
        # Confirmation bar state — two-bar pattern:
        # Bar N: price touches/pierces 3m_9 EMA (pullback detected)
        # Bar N+1: price closes back in trend direction (confirmation)
        self._pending_touch = False
        self._pending_direction = Direction.FLAT
        self._pending_ema_name = ""
        self._pending_ema_value = 0.0
        self._pending_ema_weight = 1.0
        self._pending_bar_count = 0   # bars since touch (0 = touch bar)
        self._max_confirm_bars = 3    # must confirm within 3 bars of touch

    def _get_trend(self, state: MarketState) -> str:
        """Determine trend from 5m EMA stack (primary timeframe)."""
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0 and state.ema_5m_50 > 0:
            if state.ema_5m_9 > state.ema_5m_26 > state.ema_5m_50:
                return "bullish"
            elif state.ema_5m_9 < state.ema_5m_26 < state.ema_5m_50:
                return "bearish"
        return "neutral"

    def _get_key_emas(self, state: MarketState) -> list:
        """Get key EMA levels to watch for rejections.
        
        Returns list of (name, value, weight) tuples.
        PRIMARY: 3m_9 EMA — the main rejection level. Most signals should come from here.
        SECONDARY: 3m_26 EMA — deeper pullback, higher conviction when it hits.
        
        1m EMAs removed — too noisy, generated low-quality signals.
        3m_50 removed — too deep, rarely a clean rejection (more of a trend break).
        """
        emas = []
        
        # 3-minute 9 EMA — PRIMARY rejection level
        if state.ema_levels and "3m_9" in state.ema_levels:
            val = state.ema_levels["3m_9"]
            if val and val > 0:
                emas.append(("3m_9", val, 1.0))  # Full weight
        
        # 3-minute 26 EMA — SECONDARY deeper pullback (higher conviction)
        if state.ema_levels and "3m_26" in state.ema_levels:
            val = state.ema_levels["3m_26"]
            if val and val > 0:
                emas.append(("3m_26", val, 1.15))  # Bonus weight for deeper pullback
        
        return emas

    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.atr_14 <= 0:
            return None
        
        trend = self._get_trend(state)
        if trend == "neutral":
            # Cancel any pending touch if trend dies
            self._pending_touch = False
            return None
        
        key_emas = self._get_key_emas(state)
        if not key_emas:
            return None
        
        # ── Phase 2: Check for CONFIRMATION of a pending touch ──────
        if self._pending_touch:
            self._pending_bar_count += 1
            
            # Expire if too many bars passed without confirmation
            if self._pending_bar_count > self._max_confirm_bars:
                self._pending_touch = False
                return None
            
            # Check confirmation: price closes back in trend direction
            signal = self._check_confirmation(state, trend)
            if signal:
                self._pending_touch = False
                return signal
            
            # Still pending — don't look for new touches while waiting
            return None
        
        # ── Phase 1: Look for new EMA TOUCH (pullback to EMA) ───────
        # Only use the primary 3m_9 EMA for touches
        for ema_name, ema_value, ema_weight in key_emas:
            touched = self._check_touch(state, trend, ema_name, ema_value, ema_weight)
            if touched:
                return None  # Don't enter yet — wait for confirmation bar
        
        return None
    
    def _check_touch(self, state: MarketState, trend: str,
                     ema_name: str, ema_value: float, ema_weight: float) -> bool:
        """Detect if price touched/pierced the EMA on this bar (Phase 1).
        
        Returns True if touch detected (sets pending state).
        """
        price = state.price
        
        if trend == "bullish":
            # Pullback in uptrend: bar low touches or pierces EMA from above
            bar_low_dist = state.bar_low - ema_value
            if bar_low_dist <= self.touch_threshold * 0.5 and bar_low_dist >= -self.touch_threshold:
                # RSI check — shouldn't be overbought
                if state.rsi_14 > self.rsi_long_max:
                    return False
                self._pending_touch = True
                self._pending_direction = Direction.LONG
                self._pending_ema_name = ema_name
                self._pending_ema_value = ema_value
                self._pending_ema_weight = ema_weight
                self._pending_bar_count = 0
                return True
        
        elif trend == "bearish":
            # Pullback in downtrend: bar high touches or pierces EMA from below
            bar_high_dist = ema_value - state.bar_high
            if bar_high_dist <= self.touch_threshold * 0.5 and bar_high_dist >= -self.touch_threshold:
                if state.rsi_14 < self.rsi_short_min:
                    return False
                self._pending_touch = True
                self._pending_direction = Direction.SHORT
                self._pending_ema_name = ema_name
                self._pending_ema_value = ema_value
                self._pending_ema_weight = ema_weight
                self._pending_bar_count = 0
                return True
        
        return False
    
    def _check_confirmation(self, state: MarketState, trend: str) -> Optional[Signal]:
        """Check if current bar confirms the rejection (Phase 2).
        
        Confirmation = price closes decisively back in trend direction
        with a strong body (close near high for longs, close near low for shorts).
        """
        price = state.price
        ema_value = self._pending_ema_value
        ema_name = self._pending_ema_name
        ema_weight = self._pending_ema_weight
        direction = self._pending_direction
        
        bar_range = state.bar_high - state.bar_low
        if bar_range <= 0:
            return None
        
        if direction == Direction.LONG:
            # Confirmation: close above EMA with strong body
            if price <= ema_value:
                return None  # Still below EMA — not confirmed
            
            # Body strength: close should be in upper 60% of bar range
            body_position = (state.bar_close - state.bar_low) / bar_range
            if body_position < 0.40:
                return None  # Weak close — not a convincing bounce
            
            distance = abs(price - ema_value)
            stop = ema_value - (state.atr_14 * self.atr_stop_mult)
            target = price + (state.atr_14 * self.atr_target_mult)
            
            strength = self._calc_strength(
                state, ema_name, ema_value, ema_weight, distance, Direction.LONG)
            # Bonus for strong confirmation bar
            strength = min(strength + (body_position - 0.4) * 0.2, 1.0)
            
            return Signal(
                direction=Direction.LONG,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"EMA reject LONG (confirmed): bounced off {ema_name}={ema_value:.2f}, "
                    f"trend={trend}, RSI={state.rsi_14:.0f}, "
                    f"body={body_position:.0%}, str={strength:.2f}"
                ),
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        elif direction == Direction.SHORT:
            # Confirmation: close below EMA with strong body
            if price >= ema_value:
                return None
            
            body_position = (state.bar_high - state.bar_close) / bar_range
            if body_position < 0.40:
                return None
            
            distance = abs(price - ema_value)
            stop = ema_value + (state.atr_14 * self.atr_stop_mult)
            target = price - (state.atr_14 * self.atr_target_mult)
            
            strength = self._calc_strength(
                state, ema_name, ema_value, ema_weight, distance, Direction.SHORT)
            strength = min(strength + (body_position - 0.4) * 0.2, 1.0)
            
            return Signal(
                direction=Direction.SHORT,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"EMA reject SHORT (confirmed): rejected off {ema_name}={ema_value:.2f}, "
                    f"trend={trend}, RSI={state.rsi_14:.0f}, "
                    f"body={body_position:.0%}, str={strength:.2f}"
                ),
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None

    def _calc_strength(self, state: MarketState, ema_name: str,
                       ema_value: float, ema_weight: float,
                       distance: float, direction: Direction) -> float:
        """Multi-factor signal strength for EMA rejection.
        
        Factors (each 0-1, weighted):
          1. Touch precision — how cleanly price hit the EMA (30%)
          2. Volume confirmation — was there a volume spike? (20%)
          3. ADX trend strength — strong trend = better rejection (20%)
          4. RSI sweet zone — not overbought/oversold (15%)
          5. Slope alignment — both slopes agree with direction (15%)
        
        Final score scaled by EMA weight (deeper pullbacks score higher).
        """
        atr = state.atr_14 if state.atr_14 > 0 else 2.5
        
        # 1. Touch precision (closer to EMA = better, 0-1)
        touch_score = max(0, 1.0 - (distance / self.touch_threshold))
        
        # 2. Volume confirmation (volume ratio > 1.2 = buyers/sellers stepping in)
        vol_score = min(state.volume_ratio_5 / 1.5, 1.0) if state.volume_ratio_5 > 0 else 0.3
        
        # 3. ADX trend strength (higher ADX = stronger trend = better rejection)
        adx = state.adx_14
        if adx >= 35:
            adx_score = 1.0
        elif adx >= 25:
            adx_score = 0.7
        elif adx >= 20:
            adx_score = 0.4
        else:
            adx_score = 0.2
        
        # 4. RSI sweet zone (longs: 35-55 ideal; shorts: 45-65 ideal)
        rsi = state.rsi_14
        if direction == Direction.LONG:
            if 35 <= rsi <= 55:
                rsi_score = 1.0
            elif 30 <= rsi <= 60:
                rsi_score = 0.6
            else:
                rsi_score = 0.3
        else:
            if 45 <= rsi <= 65:
                rsi_score = 1.0
            elif 40 <= rsi <= 70:
                rsi_score = 0.6
            else:
                rsi_score = 0.3
        
        # 5. Slope alignment (both slopes agreeing with direction)
        slope_12 = state.slope_12 if hasattr(state, 'slope_12') else 0
        slope_75 = state.slope_75 if hasattr(state, 'slope_75') else 0
        if direction == Direction.LONG:
            s12_ok = slope_12 > 0
            s75_ok = slope_75 > 0
        else:
            s12_ok = slope_12 < 0
            s75_ok = slope_75 < 0
        
        if s12_ok and s75_ok:
            slope_score = 1.0
        elif s75_ok:  # Long-term agrees, short-term pulling back (expected for rejection)
            slope_score = 0.7
        elif s12_ok:
            slope_score = 0.4
        else:
            slope_score = 0.15
        
        # Weighted combination
        raw = (touch_score * 0.30 + vol_score * 0.20 + adx_score * 0.20 +
               rsi_score * 0.15 + slope_score * 0.15)
        
        # Apply EMA weight (deeper pullbacks score higher)
        strength = min(raw * ema_weight, 1.0)
        
        return round(strength, 3)

    # _check_rejection removed — replaced by _check_touch + _check_confirmation

    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        """Exit on trend reversal or max hold time."""
        
        trend = self._get_trend(state)
        
        # Exit long if trend flips bearish
        if direction == Direction.LONG and trend == "bearish":
            return f"Trend reversed to bearish"
        
        # Exit short if trend flips bullish
        if direction == Direction.SHORT and trend == "bullish":
            return f"Trend reversed to bullish"
        
        # No max hold — let brackets manage the trade.
        # Trend-following positions should run until TP, SL, or trend reversal.
        
        return None
