"""
Range Detector — Identifies market structure: Trend vs Wide Range vs Tight Chop.

Core concept:
  - Find swing highs/lows from recent bars
  - Measure range width (highest swing - lowest swing)  
  - Classify: Trend (directional), Wide Range (>15pts, trade edges), Tight Chop (<10pts, no trade)
  - In range mode: identify support/resistance levels for edge-only entries

Usage:
  detector = RangeDetector()
  state = detector.update(bars, current_price)
  
  if state.structure == MarketStructure.TIGHT_CHOP:
      # Don't trade
  elif state.structure == MarketStructure.WIDE_RANGE:
      # Only trade near support (long) or resistance (short)
      if state.near_support and direction == LONG: trade
      if state.near_resistance and direction == SHORT: trade
  elif state.structure in (MarketStructure.TREND_UP, MarketStructure.TREND_DOWN):
      # Normal trend logic
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class MarketStructure(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    WIDE_RANGE = "wide_range"      # 15-40pt range — trade the edges
    TIGHT_CHOP = "tight_chop"      # <10pt range — don't trade
    UNKNOWN = "unknown"


@dataclass
class SupportResistance:
    """A support or resistance level with strength."""
    price: float
    strength: int        # Number of touches/rejections
    is_support: bool     # True=support, False=resistance
    last_touch_bar: int  # Bar index of last touch
    
    @property
    def is_resistance(self) -> bool:
        return not self.is_support


@dataclass 
class RangeState:
    """Current market structure assessment."""
    structure: MarketStructure
    range_high: float = 0.0           # Upper bound of range
    range_low: float = 0.0            # Lower bound of range
    range_width: float = 0.0          # range_high - range_low
    support_levels: List[float] = field(default_factory=list)    # Sorted ascending
    resistance_levels: List[float] = field(default_factory=list) # Sorted ascending
    near_support: bool = False        # Price within proximity of support
    near_resistance: bool = False     # Price within proximity of resistance
    nearest_support: float = 0.0      # Closest support level
    nearest_resistance: float = 0.0   # Closest resistance level
    mid_range: bool = False           # Price in the dead zone (middle of range)
    trend_strength: float = 0.0       # 0-1, how trendy vs ranging
    description: str = ""


@dataclass
class RangeConfig:
    """Configuration for range detection."""
    # Lookback periods
    swing_lookback: int = 60          # Bars to look back for swing detection
    structure_lookback: int = 30      # Bars for structure classification
    
    # Swing detection — BTC scale (1 pt = $1 BTC price); was 3.0/10/20/4/3 for MES.
    swing_window: int = 5
    min_swing_distance: float = 100.0    # $100 between distinct swings (BTC)

    # Range classification thresholds (in $ for BTC)
    tight_chop_threshold: float = 300.0  # Range < $300 = tight chop, no trade
    wide_range_min: float = 300.0        # Range >= $300 = wide range (trade edges)
    wide_range_max: float = 1500.0       # Range > $1500 = likely trending

    # Edge proximity
    edge_proximity_pts: float = 100.0    # Within $100 of S/R = near edge
    mid_zone_pct: float = 0.30           # RELATIVE — no change

    # Trend detection
    trend_slope_threshold: float = 30.0  # $30/bar slope to qualify as trend (was 0.3 MES)
    trend_consistency: float = 0.60      # RELATIVE — no change

    # S/R clustering
    sr_cluster_distance: float = 100.0   # Merge S/R levels within $100 (BTC; was 3.0 MES)


class RangeDetector:
    """Detects market structure and S/R levels from bar data."""
    
    def __init__(self, config: RangeConfig = None):
        self.config = config or RangeConfig()
        self._last_state: Optional[RangeState] = None
    
    def update(self, bars: list, current_price: float) -> RangeState:
        """Analyze bars and return current market structure.
        
        Args:
            bars: List of bar dicts with 'h', 'l', 'c', 'o' keys
            current_price: Current market price
            
        Returns:
            RangeState with structure classification and S/R levels
        """
        if len(bars) < 20:
            return RangeState(structure=MarketStructure.UNKNOWN, description="Insufficient data")
        
        # Use configured lookback
        recent = bars[-self.config.swing_lookback:]
        
        # Extract price arrays
        highs = np.array([b.get('h', b.get('high', 0)) for b in recent])
        lows = np.array([b.get('l', b.get('low', 0)) for b in recent])
        closes = np.array([b.get('c', b.get('close', 0)) for b in recent])
        
        # 1. Find swing highs and lows
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)
        
        # 2. Cluster into S/R levels
        resistance_levels = self._cluster_levels(swing_highs, is_support=False)
        support_levels = self._cluster_levels(swing_lows, is_support=True)
        
        # 3. Determine range bounds
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            range_high = float(np.max(swing_highs))
            range_low = float(np.min(swing_lows))
        else:
            range_high = float(np.max(highs[-self.config.structure_lookback:]))
            range_low = float(np.min(lows[-self.config.structure_lookback:]))
        
        range_width = range_high - range_low
        
        # 4. Detect trend vs range
        trend_strength, trend_direction = self._detect_trend(closes)
        
        # 5. Classify structure
        structure = self._classify_structure(range_width, trend_strength, trend_direction)
        
        # 6. Proximity checks
        nearest_support = self._find_nearest_below(support_levels, current_price)
        nearest_resistance = self._find_nearest_above(resistance_levels, current_price)
        
        near_support = (current_price - nearest_support) <= self.config.edge_proximity_pts if nearest_support > 0 else False
        near_resistance = (nearest_resistance - current_price) <= self.config.edge_proximity_pts if nearest_resistance > 0 else False
        
        # Mid-range detection
        mid_range = False
        if range_width > 0:
            position_in_range = (current_price - range_low) / range_width
            mid_pct = self.config.mid_zone_pct / 2
            mid_range = (0.5 - mid_pct) < position_in_range < (0.5 + mid_pct)
        
        state = RangeState(
            structure=structure,
            range_high=range_high,
            range_low=range_low,
            range_width=range_width,
            support_levels=sorted(support_levels),
            resistance_levels=sorted(resistance_levels),
            near_support=near_support,
            near_resistance=near_resistance,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            mid_range=mid_range,
            trend_strength=trend_strength,
            description=self._build_description(structure, range_width, range_high, range_low,
                                                 nearest_support, nearest_resistance, current_price),
        )
        
        self._last_state = state
        return state
    
    def _find_swing_highs(self, highs: np.ndarray) -> np.ndarray:
        """Find swing high points (local maxima)."""
        w = self.config.swing_window
        swing_points = []
        
        for i in range(w, len(highs) - w):
            if highs[i] == max(highs[i-w:i+w+1]):
                swing_points.append(highs[i])
        
        return np.array(swing_points) if swing_points else np.array([])
    
    def _find_swing_lows(self, lows: np.ndarray) -> np.ndarray:
        """Find swing low points (local minima)."""
        w = self.config.swing_window
        swing_points = []
        
        for i in range(w, len(lows) - w):
            if lows[i] == min(lows[i-w:i+w+1]):
                swing_points.append(lows[i])
        
        return np.array(swing_points) if swing_points else np.array([])
    
    def _cluster_levels(self, prices: np.ndarray, is_support: bool) -> list:
        """Cluster nearby price levels into S/R zones."""
        if len(prices) == 0:
            return []
        
        sorted_prices = np.sort(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for i in range(1, len(sorted_prices)):
            if sorted_prices[i] - current_cluster[-1] <= self.config.sr_cluster_distance:
                current_cluster.append(sorted_prices[i])
            else:
                # Average the cluster
                clusters.append(float(np.mean(current_cluster)))
                current_cluster = [sorted_prices[i]]
        
        if current_cluster:
            clusters.append(float(np.mean(current_cluster)))
        
        return clusters
    
    def _detect_trend(self, closes: np.ndarray) -> Tuple[float, str]:
        """Detect trend strength and direction from closes.
        
        Returns:
            (strength 0-1, direction 'up'/'down'/'neutral')
        """
        lookback = min(self.config.structure_lookback, len(closes))
        recent = closes[-lookback:]
        
        if len(recent) < 10:
            return 0.0, 'neutral'
        
        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]  # Points per bar
        
        # Consistency: what % of bars move in the slope direction?
        diffs = np.diff(recent)
        if slope > 0:
            consistency = np.sum(diffs > 0) / len(diffs)
        elif slope < 0:
            consistency = np.sum(diffs < 0) / len(diffs)
        else:
            consistency = 0.0
        
        # Normalize slope strength (0-1)
        slope_strength = min(abs(slope) / (self.config.trend_slope_threshold * 2), 1.0)
        
        # Combined trend strength
        strength = slope_strength * 0.6 + consistency * 0.4
        
        # Direction
        if slope > self.config.trend_slope_threshold and consistency > self.config.trend_consistency:
            direction = 'up'
        elif slope < -self.config.trend_slope_threshold and consistency > self.config.trend_consistency:
            direction = 'down'
        else:
            direction = 'neutral'
        
        return float(strength), direction
    
    def _classify_structure(self, range_width: float, trend_strength: float, 
                           trend_direction: str) -> MarketStructure:
        """Classify market structure from range width and trend data."""
        
        # Strong trend overrides range analysis
        if trend_strength > 0.65:
            if trend_direction == 'up':
                return MarketStructure.TREND_UP
            elif trend_direction == 'down':
                return MarketStructure.TREND_DOWN
        
        # Very wide range + trend = trending (just volatile)
        if range_width > self.config.wide_range_max and trend_strength > 0.4:
            if trend_direction == 'up':
                return MarketStructure.TREND_UP
            elif trend_direction == 'down':
                return MarketStructure.TREND_DOWN
        
        # Tight chop
        if range_width < self.config.tight_chop_threshold:
            return MarketStructure.TIGHT_CHOP
        
        # Wide range (tradeable at edges)
        if range_width >= self.config.wide_range_min:
            return MarketStructure.WIDE_RANGE
        
        return MarketStructure.UNKNOWN
    
    def _find_nearest_below(self, levels: list, price: float) -> float:
        """Find nearest S/R level below current price."""
        below = [l for l in levels if l < price]
        return max(below) if below else 0.0
    
    def _find_nearest_above(self, levels: list, price: float) -> float:
        """Find nearest S/R level above current price."""
        above = [l for l in levels if l > price]
        return min(above) if above else 0.0
    
    def _build_description(self, structure, width, high, low, 
                          nearest_sup, nearest_res, price) -> str:
        """Human-readable description of current state."""
        parts = [f"{structure.value}: {width:.1f}pt range [{low:.2f}-{high:.2f}]"]
        if nearest_sup > 0:
            parts.append(f"sup={nearest_sup:.2f} ({price - nearest_sup:.1f}pts away)")
        if nearest_res > 0:
            parts.append(f"res={nearest_res:.2f} ({nearest_res - price:.1f}pts away)")
        return " | ".join(parts)
    
    @property
    def last_state(self) -> Optional[RangeState]:
        return self._last_state
