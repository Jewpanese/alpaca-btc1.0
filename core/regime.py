"""
Regime Detection — Classify market conditions using relative metrics.

No absolute thresholds. Everything is normalized to the instrument's own behavior.
This is how real quant systems work — they adapt, not hardcode.

Regimes:
    TRENDING_FAST  — Strong directional move, high ATR, high ADX
    TRENDING_SLOW  — Steady grind, moderate ATR, stacked EMAs, low volatility
    RANGING        — Oscillating around mean, low ADX, ATR near median
    VOLATILE       — Choppy, high ATR, no clear direction
    DEAD           — Extremely low volatility, no opportunity (bottom 5% ATR)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING_FAST = "trending_fast"
    TRENDING_SLOW = "trending_slow"
    RANGING = "ranging"
    VOLATILE = "volatile"
    DEAD = "dead"


@dataclass
class RegimeState:
    """Current regime classification with confidence."""
    regime: Regime
    confidence: float          # 0.0-1.0
    atr_percentile: float      # Where current ATR sits in its own history
    adx: float                 # ADX value
    trend_direction: str       # 'bullish', 'bearish', 'neutral'
    ema_slope: float           # Normalized EMA slope (positive = rising)
    description: str           # Human-readable summary


class RegimeDetector:
    """Classify market regime using relative, adaptive metrics.
    
    All thresholds are percentile-based or ratio-based — never absolute points.
    This means the same logic works whether ATR is 0.5pts or 5pts.
    """
    
    def __init__(
        self,
        # ATR percentile thresholds
        dead_atr_percentile: float = 0.05,      # Bottom 5% = DEAD
        low_vol_percentile: float = 0.20,        # Bottom 20% = low vol
        high_vol_percentile: float = 0.75,       # Top 25% = high vol
        # ADX thresholds (these are universal, not instrument-specific)
        adx_trending: float = 25.0,              # ADX > 25 = trending
        adx_strong_trend: float = 35.0,          # ADX > 35 = strong trend
        adx_ranging: float = 20.0,               # ADX < 20 = ranging
        # EMA slope lookback (bars)
        slope_lookback: int = 10,
    ):
        self.dead_atr_pct = dead_atr_percentile
        self.low_vol_pct = low_vol_percentile
        self.high_vol_pct = high_vol_percentile
        self.adx_trending = adx_trending
        self.adx_strong_trend = adx_strong_trend
        self.adx_ranging = adx_ranging
        self.slope_lookback = slope_lookback
        
        self._last_regime: Optional[RegimeState] = None
        self._regime_bar_count: int = 0
    
    def detect(self, features, market_state=None) -> RegimeState:
        """Classify current market regime.
        
        Args:
            features: FeatureEngine instance
            market_state: Optional MarketState for additional context
            
        Returns:
            RegimeState with classification and metadata
        """
        # Gather metrics
        atr_pct = features.atr_percentile()
        adx = features.adx() if features.bar_count > 30 else 20.0
        current_atr = features.atr()
        
        # Trend direction from EMA stack
        trend_dir = self._detect_trend_direction(features, market_state)
        
        # EMA slope — is the trend accelerating or grinding?
        ema_slope = self._compute_ema_slope(features)
        
        # ── Classification Logic ──
        
        # DEAD: Extremely low volatility — no opportunity
        if atr_pct < self.dead_atr_pct:
            return RegimeState(
                regime=Regime.DEAD,
                confidence=0.9,
                atr_percentile=atr_pct,
                adx=adx,
                trend_direction=trend_dir,
                ema_slope=ema_slope,
                description=f"DEAD: ATR at {atr_pct:.0%} percentile — market asleep"
            )
        
        # VOLATILE: High ATR but no trend direction (choppy)
        if atr_pct > self.high_vol_pct and adx < self.adx_trending and trend_dir == "neutral":
            return RegimeState(
                regime=Regime.VOLATILE,
                confidence=min(atr_pct, 0.95),
                atr_percentile=atr_pct,
                adx=adx,
                trend_direction=trend_dir,
                ema_slope=ema_slope,
                description=f"VOLATILE: High ATR ({atr_pct:.0%} pct) but no trend (ADX={adx:.0f})"
            )
        
        # TRENDING_FAST: Strong trend + high volatility
        if adx > self.adx_strong_trend and trend_dir != "neutral" and atr_pct > self.low_vol_pct:
            confidence = min(0.5 + (adx - self.adx_strong_trend) / 30.0, 0.95)
            return RegimeState(
                regime=Regime.TRENDING_FAST,
                confidence=confidence,
                atr_percentile=atr_pct,
                adx=adx,
                trend_direction=trend_dir,
                ema_slope=ema_slope,
                description=f"TRENDING_FAST: ADX={adx:.0f}, {trend_dir}, ATR {atr_pct:.0%} pct"
            )
        
        # TRENDING_SLOW: EMAs stacked with direction but low/moderate volatility
        # This is the overnight grind regime — the one that was missing
        if trend_dir != "neutral" and adx > self.adx_ranging:
            # Confirm with EMA slope — slow trends have consistent small positive slope
            if abs(ema_slope) > 0.01:  # EMAs are actually moving, not flat
                confidence = min(0.4 + (adx - self.adx_ranging) / 20.0, 0.85)
                return RegimeState(
                    regime=Regime.TRENDING_SLOW,
                    confidence=confidence,
                    atr_percentile=atr_pct,
                    adx=adx,
                    trend_direction=trend_dir,
                    ema_slope=ema_slope,
                    description=(
                        f"TRENDING_SLOW: {trend_dir} grind, ADX={adx:.0f}, "
                        f"ATR {atr_pct:.0%} pct, slope={ema_slope:.3f}"
                    )
                )
        
        # RANGING: Low ADX, moderate volatility, no clear direction
        if adx < self.adx_trending:
            return RegimeState(
                regime=Regime.RANGING,
                confidence=min(0.5 + (self.adx_trending - adx) / 20.0, 0.85),
                atr_percentile=atr_pct,
                adx=adx,
                trend_direction=trend_dir,
                ema_slope=ema_slope,
                description=f"RANGING: ADX={adx:.0f}, no clear trend"
            )
        
        # Default: RANGING (conservative fallback)
        return RegimeState(
            regime=Regime.RANGING,
            confidence=0.4,
            atr_percentile=atr_pct,
            adx=adx,
            trend_direction=trend_dir,
            ema_slope=ema_slope,
            description=f"RANGING (default): ADX={adx:.0f}, mixed signals"
        )
    
    def _detect_trend_direction(self, features, market_state=None) -> str:
        """Detect trend from EMA alignment. Returns 'bullish', 'bearish', 'neutral'."""
        score = 0
        
        # 5-minute EMA stack (primary — weight 2)
        if market_state:
            e9 = getattr(market_state, 'ema_5m_9', 0)
            e26 = getattr(market_state, 'ema_5m_26', 0)
            e50 = getattr(market_state, 'ema_5m_50', 0)
            if e9 > 0 and e26 > 0 and e50 > 0:
                if e9 > e26 > e50:
                    score += 2
                elif e9 < e26 < e50:
                    score -= 2
        
        # 1-minute EMA stack (confirmation — weight 1)
        e9 = features.ema(9)
        e21 = features.ema(21)
        e50 = features.ema(50)
        if e9 > 0 and e21 > 0 and e50 > 0:
            if e9 > e21 > e50:
                score += 1
            elif e9 < e21 < e50:
                score -= 1
        
        if score >= 2:
            return "bullish"
        elif score <= -2:
            return "bearish"
        return "neutral"
    
    def _compute_ema_slope(self, features) -> float:
        """Compute normalized EMA slope to distinguish grinding vs flat.
        
        Returns slope of the 21 EMA over last N bars, normalized by ATR.
        Positive = rising, negative = falling, near-zero = flat.
        """
        if features.bar_count < self.slope_lookback + 25:
            return 0.0
        
        # Get EMA 21 value now vs N bars ago
        current_ema = features.ema(21)
        
        # Approximate EMA N bars ago by recomputing from closes
        closes = features.get_closes()
        if len(closes) < self.slope_lookback + 21:
            return 0.0
        
        # Compute EMA at (now - lookback)
        past_closes = closes[:-(self.slope_lookback)]
        if len(past_closes) < 21:
            return 0.0
        
        mult = 2 / 22  # period 21
        past_ema = past_closes[0]
        for val in past_closes[1:]:
            past_ema = (val - past_ema) * mult + past_ema
        
        # Normalize by ATR to make it relative
        atr = features.atr()
        if atr <= 0:
            return 0.0
        
        slope = (current_ema - past_ema) / (self.slope_lookback * atr)
        return slope
