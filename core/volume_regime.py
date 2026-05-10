"""
Volume-Spread Regime Classifier
================================
Combines Volume Spread Analysis (VSA), Wyckoff effort-vs-result,
and ADX/slope for regime detection.

Core principles:
  - Volume tells you if price movement is REAL or FAKE
  - Spread (bar range) vs volume reveals institutional activity
  - Effort (volume) vs Result (price) detects accumulation/distribution
  - ADX/slope confirms trend strength

Bar Classifications (VSA):
  DEAD     = tight spread + low volume   → genuine chop, no one's trading
  COILING  = tight spread + high volume  → accumulation/distribution, breakout imminent
  GENUINE  = wide spread + high volume   → real move, trade with it
  FAKE     = wide spread + low volume    → low conviction, likely reversal
  NORMAL   = average spread + average vol → defer to ADX/slope

Regime Output:
  CHOP       → no trade
  RANGE      → mean reversion at S/R edges
  WEAK_TREND → trade with reduced size / tighter stops
  TREND      → full conviction directional trade
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class BarClass(Enum):
    """VSA bar classification."""
    DEAD = "DEAD"          # Tight spread + low volume
    COILING = "COILING"    # Tight spread + high volume (accumulation)
    GENUINE = "GENUINE"    # Wide spread + high volume (real move)
    FAKE = "FAKE"          # Wide spread + low volume (trap)
    NORMAL = "NORMAL"      # Average — defer to other signals


class VolumeRegime(Enum):
    """Market regime based on volume-spread analysis."""
    CHOP = "CHOP"
    RANGE = "RANGE"
    WEAK_TREND = "WEAK_TREND"
    TREND = "TREND"


@dataclass
class VolumeRegimeConfig:
    """Configuration for volume-spread regime classifier."""
    
    # Volume averaging
    vol_avg_period: int = 20         # Bars for volume moving average
    
    # Spread thresholds (as multiple of ATR)
    spread_tight_mult: float = 0.5   # Spread < 0.5×ATR = tight
    spread_normal_mult: float = 0.8  # Spread < 0.8×ATR = normal
    spread_wide_mult: float = 1.0    # Spread >= 1.0×ATR = wide
    
    # Volume ratio thresholds (current vol / avg vol)
    vol_low: float = 0.7             # Below = low volume
    vol_normal_low: float = 0.9      # Below = below-average
    vol_normal_high: float = 1.3     # Above = above-average
    vol_high: float = 1.5            # Above = high volume
    vol_spike: float = 2.5           # Above = volume spike (event)
    
    # Bar classification streak thresholds
    dead_streak_for_chop: int = 5    # N consecutive DEAD bars → CHOP
    coiling_streak_for_range: int = 3  # N consecutive COILING bars → breakout alert
    
    # Effort vs Result (Weis-style wave tracking)
    wave_lookback: int = 20          # Bars to track up/down wave volumes
    effort_result_threshold: float = 1.5  # Effort/result ratio for divergence
    
    # ADX integration
    adx_strong_trend: float = 28.0   # ADX above = strong trend
    adx_weak_trend: float = 18.0     # ADX above = weak trend possible
    adx_dead: float = 12.0           # ADX below = truly dead
    
    # Slope integration
    slope_supremacy: float = 12.0    # |slope75| above = trend override
    
    # Delta thresholds
    delta_strong: float = 5.0        # Cumulative delta divergence threshold
    
    # Stickiness (prevent rapid regime flipping)
    min_bars_in_regime: int = 3      # Hold regime for at least N bars
    
    # Logging
    log_interval_bars: int = 30      # Log regime every N bars


@dataclass
class BarAnalysis:
    """Analysis of a single bar."""
    timestamp: float
    spread: float           # high - low
    spread_atr_ratio: float # spread / ATR
    volume: float
    vol_ratio: float        # volume / avg_volume
    bar_class: BarClass
    close_position: float   # 0-1, where close is within bar (0=low, 1=high)
    is_up: bool             # close > open
    delta: float            # buy - sell estimate


@dataclass
class WaveState:
    """Tracks cumulative volume in up vs down waves (Weis-style)."""
    direction: str = "neutral"  # "up", "down", "neutral"
    cum_volume: float = 0.0
    cum_price: float = 0.0
    bar_count: int = 0
    
    # History of completed waves
    up_waves: List[dict] = field(default_factory=list)
    down_waves: List[dict] = field(default_factory=list)


@dataclass 
class VolumeRegimeOutput:
    """Output of the volume regime classifier."""
    regime: VolumeRegime
    confidence: float            # 0.0-1.0
    bar_class: BarClass          # Current bar classification
    vol_ratio: float             # Current volume ratio
    spread_atr_ratio: float      # Current spread/ATR
    reason: str                  # Human-readable explanation
    
    # Effort vs Result
    effort_result_bias: Optional[str] = None  # "bullish_accumulation", "bearish_distribution", None
    
    # Wave info
    wave_direction: str = "neutral"
    wave_volume: float = 0.0
    
    # Streak info
    dead_streak: int = 0
    coiling_streak: int = 0
    genuine_streak: int = 0


class VolumeRegimeClassifier:
    """
    Volume-Spread Regime Classifier.
    
    Classifies market regime using:
    1. VSA bar classification (spread vs volume)
    2. Effort vs Result (Weis wave volume tracking)
    3. ADX/slope confirmation
    4. Delta (order flow) bias
    """
    
    def __init__(self, config: VolumeRegimeConfig = None):
        self.config = config or VolumeRegimeConfig()
        
        # Volume history for averaging
        self._volumes: deque = deque(maxlen=self.config.vol_avg_period)
        
        # Bar classification history
        self._bar_classes: deque = deque(maxlen=50)
        self._bar_analyses: deque = deque(maxlen=50)
        
        # Wave tracking (effort vs result)
        self._wave = WaveState()
        self._max_waves = 10  # Keep last N waves per direction
        
        # Regime state
        self._current_regime = VolumeRegimeOutput(
            regime=VolumeRegime.CHOP,
            confidence=0.0,
            bar_class=BarClass.NORMAL,
            vol_ratio=1.0,
            spread_atr_ratio=1.0,
            reason="startup",
        )
        self._bars_in_regime: int = 0
        self._bar_count: int = 0
        self._last_log_bar: int = 0
    
    @property
    def current(self) -> VolumeRegimeOutput:
        return self._current_regime
    
    def update(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        atr: float,
        adx: float,
        slope_75: float = 0.0,
        delta: float = 0.0,
        cumulative_delta: float = 0.0,
    ) -> VolumeRegimeOutput:
        """
        Process a new bar and return the current regime classification.
        
        Args:
            bar_open/high/low/close: OHLC of current bar
            bar_volume: Volume of current bar
            atr: Current ATR(14)
            adx: Current ADX (prefer 3-min ADX)
            slope_75: 75-bar momentum slope in bps
            delta: Current bar delta (buy - sell)
            cumulative_delta: Rolling cumulative delta
            
        Returns:
            VolumeRegimeOutput with regime, confidence, and analysis
        """
        self._bar_count += 1
        
        # ── 1. Track volume history ──
        self._volumes.append(bar_volume)
        
        # Need enough history for vol average
        if len(self._volumes) < 5:
            return self._current_regime
        
        # ── 2. Compute volume ratio ──
        avg_volume = np.mean(list(self._volumes)[:-1])  # Exclude current bar
        if avg_volume <= 0:
            avg_volume = 1.0
        vol_ratio = bar_volume / avg_volume
        
        # ── 3. Compute spread metrics ──
        spread = bar_high - bar_low
        spread_atr_ratio = spread / atr if atr > 0 else 1.0
        
        # Close position within bar (0 = at low, 1 = at high)
        close_position = (bar_close - bar_low) / spread if spread > 0 else 0.5
        is_up = bar_close >= bar_open
        
        # ── 4. VSA Bar Classification ──
        bar_class = self._classify_bar(spread_atr_ratio, vol_ratio)
        
        analysis = BarAnalysis(
            timestamp=time.time(),
            spread=spread,
            spread_atr_ratio=spread_atr_ratio,
            volume=bar_volume,
            vol_ratio=vol_ratio,
            bar_class=bar_class,
            close_position=close_position,
            is_up=is_up,
            delta=delta,
        )
        self._bar_classes.append(bar_class)
        self._bar_analyses.append(analysis)
        
        # ── 5. Update wave tracking (effort vs result) ──
        self._update_waves(bar_close, bar_volume, is_up)
        
        # ── 6. Compute streaks ──
        dead_streak = self._count_streak(BarClass.DEAD)
        coiling_streak = self._count_streak(BarClass.COILING)
        genuine_streak = self._count_streak(BarClass.GENUINE)
        
        # ── 7. Effort vs Result analysis ──
        effort_result_bias = self._analyze_effort_result()
        
        # ── 8. Classify regime ──
        regime, confidence, reason = self._classify_regime(
            bar_class=bar_class,
            vol_ratio=vol_ratio,
            spread_atr_ratio=spread_atr_ratio,
            adx=adx,
            slope_75=slope_75,
            delta=delta,
            cumulative_delta=cumulative_delta,
            dead_streak=dead_streak,
            coiling_streak=coiling_streak,
            genuine_streak=genuine_streak,
            effort_result_bias=effort_result_bias,
        )
        
        # ── 9. Apply stickiness ──
        if regime != self._current_regime.regime:
            if self._bars_in_regime < self.config.min_bars_in_regime:
                # Don't switch yet — hold current regime
                regime = self._current_regime.regime
                confidence = self._current_regime.confidence * 0.9
                reason = f"STICKY ({self._bars_in_regime}/{self.config.min_bars_in_regime}) | " + reason
            else:
                self._bars_in_regime = 0
        
        self._bars_in_regime += 1
        
        output = VolumeRegimeOutput(
            regime=regime,
            confidence=confidence,
            bar_class=bar_class,
            vol_ratio=vol_ratio,
            spread_atr_ratio=spread_atr_ratio,
            reason=reason,
            effort_result_bias=effort_result_bias,
            wave_direction=self._wave.direction,
            wave_volume=self._wave.cum_volume,
            dead_streak=dead_streak,
            coiling_streak=coiling_streak,
            genuine_streak=genuine_streak,
        )
        
        self._current_regime = output
        
        # ── 10. Periodic logging ──
        if self._bar_count - self._last_log_bar >= self.config.log_interval_bars:
            self._last_log_bar = self._bar_count
            logger.info(
                f"📊 [VOL REGIME] {regime.value} (conf={confidence:.0%}) | "
                f"Bar={bar_class.value} | Vol={vol_ratio:.2f}x | "
                f"Spread={spread_atr_ratio:.2f}×ATR | ADX={adx:.1f} | "
                f"Streaks: dead={dead_streak} coil={coiling_streak} gen={genuine_streak} | "
                f"{reason}"
            )
        
        return output
    
    def _classify_bar(self, spread_atr_ratio: float, vol_ratio: float) -> BarClass:
        """VSA bar classification based on spread vs volume."""
        cfg = self.config
        
        tight_spread = spread_atr_ratio < cfg.spread_tight_mult
        normal_spread = spread_atr_ratio < cfg.spread_wide_mult
        wide_spread = spread_atr_ratio >= cfg.spread_wide_mult
        
        low_vol = vol_ratio < cfg.vol_low
        high_vol = vol_ratio > cfg.vol_high
        
        if tight_spread and low_vol:
            return BarClass.DEAD
        elif tight_spread and high_vol:
            return BarClass.COILING
        elif wide_spread and high_vol:
            return BarClass.GENUINE
        elif wide_spread and low_vol:
            return BarClass.FAKE
        else:
            # Normal spread or normal volume — no strong VSA signal
            # But check edge cases
            if vol_ratio > cfg.vol_spike:
                return BarClass.GENUINE  # Volume spike = something real
            elif tight_spread and vol_ratio > cfg.vol_normal_high:
                return BarClass.COILING  # Slightly elevated vol + tight = coiling
            elif low_vol and normal_spread:
                return BarClass.DEAD     # Low vol + unremarkable spread = dead
            return BarClass.NORMAL
    
    def _update_waves(self, close: float, volume: float, is_up: bool):
        """Track Weis-style volume waves."""
        direction = "up" if is_up else "down"
        
        if direction == self._wave.direction:
            # Continue current wave
            self._wave.cum_volume += volume
            self._wave.cum_price += close - (self._bar_analyses[-2].timestamp if len(self._bar_analyses) > 1 else close)
            self._wave.bar_count += 1
        else:
            # Wave reversed — archive the completed wave
            if self._wave.direction != "neutral" and self._wave.bar_count > 0:
                wave_data = {
                    "direction": self._wave.direction,
                    "volume": self._wave.cum_volume,
                    "bars": self._wave.bar_count,
                }
                if self._wave.direction == "up":
                    self._wave.up_waves.append(wave_data)
                    if len(self._wave.up_waves) > self._max_waves:
                        self._wave.up_waves.pop(0)
                else:
                    self._wave.down_waves.append(wave_data)
                    if len(self._wave.down_waves) > self._max_waves:
                        self._wave.down_waves.pop(0)
            
            # Start new wave
            self._wave.direction = direction
            self._wave.cum_volume = volume
            self._wave.cum_price = 0.0
            self._wave.bar_count = 1
    
    def _analyze_effort_result(self) -> Optional[str]:
        """
        Analyze effort vs result across recent waves.
        
        If up-wave volume >> down-wave volume near support → bullish accumulation
        If down-wave volume >> up-wave volume near resistance → bearish distribution
        """
        up_waves = self._wave.up_waves
        down_waves = self._wave.down_waves
        
        if len(up_waves) < 2 or len(down_waves) < 2:
            return None
        
        # Compare last 3 waves of each direction
        recent_up_vol = np.mean([w["volume"] for w in up_waves[-3:]])
        recent_down_vol = np.mean([w["volume"] for w in down_waves[-3:]])
        
        if recent_up_vol <= 0 or recent_down_vol <= 0:
            return None
        
        ratio = recent_up_vol / recent_down_vol
        
        if ratio > self.config.effort_result_threshold:
            return "bullish_accumulation"
        elif ratio < (1.0 / self.config.effort_result_threshold):
            return "bearish_distribution"
        
        return None
    
    def _count_streak(self, target_class: BarClass) -> int:
        """Count consecutive bars of a given class (from most recent)."""
        count = 0
        for bc in reversed(self._bar_classes):
            if bc == target_class:
                count += 1
            else:
                break
        return count
    
    def _classify_regime(
        self,
        bar_class: BarClass,
        vol_ratio: float,
        spread_atr_ratio: float,
        adx: float,
        slope_75: float,
        delta: float,
        cumulative_delta: float,
        dead_streak: int,
        coiling_streak: int,
        genuine_streak: int,
        effort_result_bias: Optional[str],
    ) -> tuple:  # (VolumeRegime, confidence, reason)
        """
        Main regime classification logic.
        
        Priority:
        1. Hard overrides (slope supremacy, extreme ADX)
        2. VSA streak-based signals
        3. Volume-weighted ADX classification
        """
        cfg = self.config
        abs_slope = abs(slope_75)
        reasons = []
        
        # ═══════════════════════════════════════════════════════
        # PRIORITY 1: Hard overrides — these always win
        # ═══════════════════════════════════════════════════════
        
        # Slope supremacy + volume confirms → TREND (highest priority)
        if abs_slope >= cfg.slope_supremacy and vol_ratio > cfg.vol_normal_low:
            reasons.append(f"|slope75|={abs_slope:.1f}bps (supremacy)")
            reasons.append(f"vol={vol_ratio:.2f}x confirms")
            return VolumeRegime.TREND, min(1.0, 0.7 + vol_ratio * 0.1), " + ".join(reasons)
        
        # Slope supremacy but low volume → WEAK_TREND (move might be fake)
        if abs_slope >= cfg.slope_supremacy and vol_ratio <= cfg.vol_low:
            reasons.append(f"|slope75|={abs_slope:.1f}bps but LOW vol={vol_ratio:.2f}x")
            return VolumeRegime.WEAK_TREND, 0.5, " + ".join(reasons)
        
        # Strong ADX + volume → TREND
        if adx >= cfg.adx_strong_trend and vol_ratio > cfg.vol_normal_low:
            reasons.append(f"ADX={adx:.1f} (strong)")
            reasons.append(f"vol={vol_ratio:.2f}x confirms")
            conf = min(1.0, 0.6 + (adx - cfg.adx_strong_trend) / 20 + vol_ratio * 0.1)
            return VolumeRegime.TREND, conf, " + ".join(reasons)
        
        # Strong ADX but dying volume → WEAK_TREND (trend exhaustion)
        if adx >= cfg.adx_strong_trend and vol_ratio <= cfg.vol_low:
            reasons.append(f"ADX={adx:.1f} but DYING vol={vol_ratio:.2f}x (exhaustion?)")
            return VolumeRegime.WEAK_TREND, 0.5, " + ".join(reasons)
        
        # ═══════════════════════════════════════════════════════
        # PRIORITY 2: VSA streak-based signals
        # ═══════════════════════════════════════════════════════
        
        # Dead streak → CHOP (genuine no-activity)
        if dead_streak >= cfg.dead_streak_for_chop and adx < cfg.adx_weak_trend:
            reasons.append(f"DEAD×{dead_streak} bars")
            reasons.append(f"ADX={adx:.1f}")
            return VolumeRegime.CHOP, min(1.0, 0.5 + dead_streak * 0.1), " + ".join(reasons)
        
        # Coiling streak → RANGE (accumulation, trade edges, prepare for breakout)
        if coiling_streak >= cfg.coiling_streak_for_range:
            reasons.append(f"COILING×{coiling_streak} bars (accumulation)")
            if effort_result_bias:
                reasons.append(f"bias={effort_result_bias}")
            return VolumeRegime.RANGE, min(1.0, 0.6 + coiling_streak * 0.1), " + ".join(reasons)
        
        # Genuine streak → TREND or WEAK_TREND depending on ADX
        if genuine_streak >= 2:
            reasons.append(f"GENUINE×{genuine_streak} bars")
            if adx >= cfg.adx_weak_trend:
                reasons.append(f"ADX={adx:.1f} supports trend")
                conf = min(1.0, 0.5 + genuine_streak * 0.1 + (adx - cfg.adx_weak_trend) / 30)
                return VolumeRegime.TREND, conf, " + ".join(reasons)
            else:
                reasons.append(f"ADX={adx:.1f} (weak but volume is real)")
                return VolumeRegime.WEAK_TREND, 0.55, " + ".join(reasons)
        
        # ═══════════════════════════════════════════════════════
        # PRIORITY 3: Volume-weighted ADX classification
        # ═══════════════════════════════════════════════════════
        
        # Volume spike (regardless of ADX) → something is happening
        if vol_ratio > cfg.vol_spike:
            reasons.append(f"VOL SPIKE {vol_ratio:.2f}x")
            if adx >= cfg.adx_weak_trend:
                return VolumeRegime.TREND, 0.7, " + ".join(reasons)
            else:
                reasons.append("→ RANGE (high vol but no trend yet)")
                return VolumeRegime.RANGE, 0.6, " + ".join(reasons)
        
        # ADX weak trend zone (18-28) — volume decides
        if cfg.adx_weak_trend <= adx < cfg.adx_strong_trend:
            if vol_ratio > cfg.vol_normal_high:
                reasons.append(f"ADX={adx:.1f} mid-zone + vol={vol_ratio:.2f}x (above avg)")
                return VolumeRegime.WEAK_TREND, 0.55, " + ".join(reasons)
            elif vol_ratio < cfg.vol_low:
                reasons.append(f"ADX={adx:.1f} mid-zone + vol={vol_ratio:.2f}x (low)")
                return VolumeRegime.RANGE, 0.5, " + ".join(reasons)
            else:
                reasons.append(f"ADX={adx:.1f} mid-zone + vol={vol_ratio:.2f}x (normal)")
                return VolumeRegime.RANGE, 0.45, " + ".join(reasons)
        
        # ADX dead zone (12-18) — volume is the tiebreaker
        if cfg.adx_dead <= adx < cfg.adx_weak_trend:
            if vol_ratio > cfg.vol_high:
                reasons.append(f"ADX={adx:.1f} low but vol={vol_ratio:.2f}x HIGH (coiling?)")
                return VolumeRegime.RANGE, 0.55, " + ".join(reasons)
            elif vol_ratio > cfg.vol_normal_high:
                reasons.append(f"ADX={adx:.1f} low + vol={vol_ratio:.2f}x (some activity)")
                return VolumeRegime.RANGE, 0.4, " + ".join(reasons)
            else:
                reasons.append(f"ADX={adx:.1f} low + vol={vol_ratio:.2f}x (quiet)")
                return VolumeRegime.CHOP, 0.5, " + ".join(reasons)
        
        # ADX truly dead (< 12)
        if adx < cfg.adx_dead:
            if vol_ratio > cfg.vol_spike:
                reasons.append(f"ADX={adx:.1f} DEAD but VOL SPIKE {vol_ratio:.2f}x")
                return VolumeRegime.RANGE, 0.5, " + ".join(reasons)
            elif vol_ratio > cfg.vol_high:
                reasons.append(f"ADX={adx:.1f} dead but vol={vol_ratio:.2f}x elevated")
                return VolumeRegime.RANGE, 0.35, " + ".join(reasons)
            else:
                reasons.append(f"ADX={adx:.1f} dead + vol={vol_ratio:.2f}x dead")
                return VolumeRegime.CHOP, 0.7, " + ".join(reasons)
        
        # Fallback (shouldn't reach here often)
        reasons.append(f"ADX={adx:.1f} vol={vol_ratio:.2f}x bar={bar_class.value}")
        return VolumeRegime.RANGE, 0.3, "fallback: " + " + ".join(reasons)
    
    def get_market_mode(self) -> str:
        """Get regime as simple string for bot routing (CHOP/RANGE/TREND)."""
        regime = self._current_regime.regime
        if regime == VolumeRegime.CHOP:
            return "CHOP"
        elif regime == VolumeRegime.RANGE:
            return "RANGE"
        elif regime == VolumeRegime.WEAK_TREND:
            return "TREND"  # Weak trends still trade, just with awareness
        elif regime == VolumeRegime.TREND:
            return "TREND"
        return "RANGE"  # Safe default
    
    def is_weak_trend(self) -> bool:
        """Check if current regime is a weak trend (for position sizing)."""
        return self._current_regime.regime == VolumeRegime.WEAK_TREND
    
    def get_effort_result_bias(self) -> Optional[str]:
        """Get current effort-vs-result bias (bullish_accumulation / bearish_distribution / None)."""
        return self._current_regime.effort_result_bias
    
    def reset(self):
        """Reset classifier state (e.g., at session boundaries)."""
        self._volumes.clear()
        self._bar_classes.clear()
        self._bar_analyses.clear()
        self._wave = WaveState()
        self._bars_in_regime = 0
        self._bar_count = 0
        self._current_regime = VolumeRegimeOutput(
            regime=VolumeRegime.CHOP,
            confidence=0.0,
            bar_class=BarClass.NORMAL,
            vol_ratio=1.0,
            spread_atr_ratio=1.0,
            reason="reset",
        )
