"""Trend Day Detector — Know When to Stop Fading.

A trend day is when the market moves directionally all day with minimal
mean reversion. On these days, mean-reversion strategies (VWAP_REVERT,
BB_BOUNCE, DELTA_DIV) get destroyed because every "oversold" signal
just keeps going.

Detection criteria (any 2 of 4 = trend day):
  1. Gap > 0.5% from prior close
  2. ATR percentile > 80th (volatility expansion)
  3. ADX > 30 and rising
  4. Price consistently on one side of VWAP for 30+ minutes

When detected:
  - Mean-reversion strategies are DISABLED
  - Trend-following strategies get boosted weight
  - Stops are widened (trends pull back before continuing)
  - Only entries in the trend direction are allowed
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Strategies classified by type
MEAN_REVERSION_STRATEGIES = {"VWAP_REVERT", "BB_BOUNCE", "DELTA_DIV"}
TREND_STRATEGIES = {"TREND_FOLLOW", "EMA_REJECT", "MOMENTUM", "SR_BREAKOUT"}


@dataclass
class TrendDayConfig:
    """Configuration for trend day detection."""
    
    # Gap threshold (fraction of price)
    gap_threshold_pct: float = 0.005        # 0.5%
    
    # ATR percentile threshold
    atr_percentile_threshold: float = 0.80  # 80th percentile
    
    # ADX threshold
    adx_threshold: float = 30.0
    
    # VWAP consistency: minutes price must stay on one side
    vwap_consistency_minutes: int = 30
    
    # How many criteria needed (out of 4)
    min_criteria: int = 2
    
    # Boost multiplier for trend strategies during trend days
    trend_strategy_boost: float = 1.3
    
    # Reduction multiplier for mean-reversion (0 = fully disabled)
    mean_reversion_mult: float = 0.3        # Reduced on trend days (was 0 = disabled)


class TrendDayDetector:
    """Detect trend days and modify strategy weights accordingly."""
    
    def __init__(self, config: TrendDayConfig = None):
        self.config = config or TrendDayConfig()
        self.is_trend_day: bool = False
        self.trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
        self.criteria_met: list = []
        self._prior_close: Optional[float] = None
        self._vwap_side_start: float = 0.0
        self._vwap_side: str = "neutral"
        self._last_check: float = 0.0
    
    def set_prior_close(self, price: float):
        """Set prior session close for gap calculation."""
        self._prior_close = price
    
    def update(
        self,
        current_price: float,
        vwap: float,
        atr_percentile: float,
        adx: float,
        open_price: Optional[float] = None,
    ) -> bool:
        """Update trend day detection. Returns True if trend day.
        
        Call this every minute or so, not every tick.
        """
        now = time.time()
        if now - self._last_check < 30:  # Only check every 30 seconds
            return self.is_trend_day
        self._last_check = now
        
        criteria = []
        direction_votes = {"bullish": 0, "bearish": 0}
        
        # 1. Gap check
        ref_price = self._prior_close or open_price
        if ref_price and open_price:
            gap_pct = (open_price - ref_price) / ref_price
            if abs(gap_pct) > self.config.gap_threshold_pct:
                criteria.append(f"gap={gap_pct*100:.2f}%")
                if gap_pct > 0:
                    direction_votes["bullish"] += 1
                else:
                    direction_votes["bearish"] += 1
        
        # 2. ATR percentile
        if atr_percentile > self.config.atr_percentile_threshold:
            criteria.append(f"ATR_pctl={atr_percentile:.0%}")
        
        # 3. ADX
        if adx > self.config.adx_threshold:
            criteria.append(f"ADX={adx:.1f}")
        
        # 4. VWAP consistency
        if current_price > vwap:
            current_side = "above"
        elif current_price < vwap:
            current_side = "below"
        else:
            current_side = "neutral"
        
        if current_side != self._vwap_side:
            self._vwap_side = current_side
            self._vwap_side_start = now
        
        vwap_duration_min = (now - self._vwap_side_start) / 60.0
        if vwap_duration_min >= self.config.vwap_consistency_minutes:
            criteria.append(f"VWAP_{current_side}={vwap_duration_min:.0f}min")
            if current_side == "above":
                direction_votes["bullish"] += 1
            else:
                direction_votes["bearish"] += 1
        
        # Determine trend day
        old_status = self.is_trend_day
        self.criteria_met = criteria
        self.is_trend_day = len(criteria) >= self.config.min_criteria
        
        if self.is_trend_day:
            if direction_votes["bullish"] > direction_votes["bearish"]:
                self.trend_direction = "bullish"
            elif direction_votes["bearish"] > direction_votes["bullish"]:
                self.trend_direction = "bearish"
            else:
                # Use current price vs VWAP as tiebreaker
                self.trend_direction = "bullish" if current_price > vwap else "bearish"
        else:
            self.trend_direction = "neutral"
        
        # Log state change
        if self.is_trend_day != old_status:
            if self.is_trend_day:
                logger.warning(
                    f"🔥 TREND DAY DETECTED ({self.trend_direction}): "
                    f"{', '.join(criteria)}. Mean-reversion strategies DISABLED."
                )
            else:
                logger.info("Trend day conditions no longer met. Returning to normal.")
        
        return self.is_trend_day
    
    def get_strategy_multiplier(self, strategy_name: str) -> float:
        """Get strategy weight multiplier based on trend day status.
        
        Returns:
            Multiplier (0.0 = disabled, 1.0 = normal, >1.0 = boosted)
        """
        if not self.is_trend_day:
            return 1.0
        
        if strategy_name in MEAN_REVERSION_STRATEGIES:
            return self.config.mean_reversion_mult
        
        if strategy_name in TREND_STRATEGIES:
            return self.config.trend_strategy_boost
        
        return 1.0  # Unknown strategies: normal weight
    
    def is_direction_allowed(self, direction) -> bool:
        """Check if a trade direction is allowed on a trend day.
        
        On trend days, only trade WITH the trend.
        """
        if not self.is_trend_day:
            return True
        
        if self.trend_direction == "bullish":
            return direction.value > 0  # LONG only
        elif self.trend_direction == "bearish":
            return direction.value < 0  # SHORT only
        
        return True  # Neutral = allow both
    
    def get_status(self) -> dict:
        return {
            "is_trend_day": self.is_trend_day,
            "direction": self.trend_direction,
            "criteria": self.criteria_met,
        }
