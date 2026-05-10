"""
Session-Aware Persistence Regime Detector — Ported from Topstep3.

Uses persistence ratio (higher highs/lows) instead of Kaufman Efficiency
to correctly detect slow overnight grinds. Session-aware thresholds:

  - Asian (6pm-3am ET):  Strong trends only (persistence > 0.60)
  - London (3am-9:30am ET): Weak+ trends allowed (persistence > 0.50)
  - NY (9:30am-4pm ET):  Most permissive (persistence > 0.45)

This replaces the original regime detector's role as a signal gate.
The existing RegimeDetector in core/regime.py still handles ATR-based
regime classification. This module adds the persistence-based directional
filter and session-aware gating.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    BULLISH_STRONG = "bullish_strong"
    BULLISH_WEAK = "bullish_weak"
    BEARISH_STRONG = "bearish_strong"
    BEARISH_WEAK = "bearish_weak"
    CHOP = "chop"
    RANGING = "ranging"


class VolatilityState(Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


class StructuralPhase(Enum):
    IMPULSIVE = "impulsive"
    CONSOLIDATION = "consolidation"
    CORRECTIVE = "corrective"
    EXHAUSTION = "exhaustion"
    BREAKOUT = "breakout"
    RANGING = "ranging"


class RiskEnvironment(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    AVOID = "avoid"


class TradingSession(Enum):
    ASIAN = "asian"           # 6pm - 3am ET (low vol, slow grinds)
    LONDON = "london"         # 3am - 9:30am ET (ramping vol)
    NY = "ny"                 # 9:30am - 4pm ET (peak vol)
    TRANSITION = "transition" # Overlaps / edge cases


# Session-specific thresholds
SESSION_THRESHOLDS = {
    TradingSession.ASIAN: {
        'persistence_min': 0.53,      # Moderate — was 0.60 but blocked real trends
        'slope_threshold': 0.12,
        'allow_weak_trends': True,    # Allow weak trends — ADX 34 bearish_weak is real
        'allow_ranging_breakout': False,
    },
    TradingSession.LONDON: {
        'persistence_min': 0.50,      # Weak trends OK
        'slope_threshold': 0.10,
        'allow_weak_trends': True,
        'allow_ranging_breakout': True,
    },
    TradingSession.NY: {
        'persistence_min': 0.45,      # Most permissive
        'slope_threshold': 0.08,
        'allow_weak_trends': True,
        'allow_ranging_breakout': True,
    },
    TradingSession.TRANSITION: {
        'persistence_min': 0.55,
        'slope_threshold': 0.12,
        'allow_weak_trends': True,
        'allow_ranging_breakout': False,
    },
}


@dataclass
class PersistenceRegimeContext:
    """Complete persistence-based regime context."""
    trend_regime: TrendRegime
    volatility_state: VolatilityState
    structural_phase: StructuralPhase
    risk_environment: RiskEnvironment
    session: TradingSession
    
    # Scores
    trend_strength: float        # 0-1, ATR-adjusted
    persistence_ratio: float     # 0-1, higher highs/lows consistency
    chop_probability: float      # 1 - persistence
    
    # Trading decision
    should_trade: bool
    preferred_direction: int     # 1=long, -1=short, 0=both
    position_size_multiplier: float
    
    # Metadata
    confidence: float
    reason: str                  # Why we're allowing/blocking


class PersistenceRegimeDetector:
    """
    Session-aware persistence regime detector.
    
    Adapts strictness based on trading session:
    - Asian: conservative, strong trends only
    - London: moderate, weak trends OK
    - NY: permissive, more setups allowed
    """
    
    def __init__(self, lookback_period: int = 30):
        """
        Args:
            lookback_period: Bars to analyze (on whatever timeframe is passed in)
        """
        self.lookback_period = lookback_period
    
    def detect(self, bars: list, current_hour_et: int = None) -> PersistenceRegimeContext:
        """
        Detect regime from a list of bar dicts.
        
        Args:
            bars: List of bar dicts with keys {o, h, l, c, v}
            current_hour_et: Current hour in Eastern Time (0-23)
        
        Returns:
            PersistenceRegimeContext
        """
        if len(bars) < self.lookback_period:
            return self._avoid_context("Insufficient bars")
        
        session = self._get_session(current_hour_et) if current_hour_et is not None else TradingSession.NY
        thresholds = SESSION_THRESHOLDS[session]
        
        window = bars[-self.lookback_period:]
        
        highs = np.array([b['h'] for b in window])
        lows = np.array([b['l'] for b in window])
        closes = np.array([b['c'] for b in window])
        
        # Core metrics
        trend_regime, trend_strength, persistence = self._detect_trend(
            highs, lows, closes, thresholds['slope_threshold']
        )
        vol_state, atr_norm = self._detect_volatility(highs, lows, closes)
        structure = self._detect_structure(closes, persistence)
        risk = self._assess_risk(trend_regime, vol_state, structure)
        
        # Session-aware trading decision
        should_trade, direction, size_mult, reason = self._make_decision(
            trend_regime, vol_state, structure, risk,
            trend_strength, persistence, session, thresholds
        )
        
        confidence = min((trend_strength + persistence) / 2.0, 1.0)
        if vol_state == VolatilityState.EXTREME:
            confidence *= 0.5
        
        return PersistenceRegimeContext(
            trend_regime=trend_regime,
            volatility_state=vol_state,
            structural_phase=structure,
            risk_environment=risk,
            session=session,
            trend_strength=trend_strength,
            persistence_ratio=persistence,
            chop_probability=1.0 - persistence,
            should_trade=should_trade,
            preferred_direction=direction,
            position_size_multiplier=size_mult,
            confidence=confidence,
            reason=reason,
        )
    
    # ─── Session Detection ──────────────────────────────────────────
    
    @staticmethod
    def _get_session(hour_et: int) -> TradingSession:
        """Classify current hour (ET) into trading session."""
        if 18 <= hour_et <= 23 or 0 <= hour_et < 3:
            return TradingSession.ASIAN
        elif 3 <= hour_et < 9:
            return TradingSession.LONDON
        elif 10 <= hour_et < 16:
            return TradingSession.NY
        else:
            # 9-10 ET is overlap / transition
            return TradingSession.TRANSITION
    
    # ─── Trend Detection (Persistence-Based) ────────────────────────
    
    def _detect_trend(self, highs, lows, closes, slope_threshold) -> Tuple[TrendRegime, float, float]:
        """Detect trend using persistence ratio + linear regression."""
        N = len(closes)
        
        # Persistence ratio
        hh = sum(highs[i] > highs[i-1] for i in range(1, N))
        hl = sum(lows[i] > lows[i-1] for i in range(1, N))
        lh = sum(highs[i] < highs[i-1] for i in range(1, N))
        ll = sum(lows[i] < lows[i-1] for i in range(1, N))
        
        bull_persist = (hh + hl) / (2 * (N - 1))
        bear_persist = (lh + ll) / (2 * (N - 1))
        persistence = max(bull_persist, bear_persist)
        
        # Linear regression slope
        x = np.arange(N)
        slope = np.polyfit(x, closes, 1)[0]
        
        # ATR-adjusted trend strength
        atr = self._calc_atr(highs, lows, closes)
        if atr > 0:
            net_move = closes[-1] - closes[0]
            noise = atr * np.sqrt(N)
            trend_strength = min(abs(net_move) / noise, 1.0)
        else:
            trend_strength = 0.0
        
        slope_norm = slope / (atr if atr > 0 else 1.0)
        
        # Classify
        if persistence >= 0.60 and abs(slope_norm) > slope_threshold:
            regime = TrendRegime.BULLISH_STRONG if slope > 0 else TrendRegime.BEARISH_STRONG
        elif persistence >= 0.50 and abs(slope_norm) > slope_threshold * 0.5:
            regime = TrendRegime.BULLISH_WEAK if slope > 0 else TrendRegime.BEARISH_WEAK
        elif persistence < 0.40:
            regime = TrendRegime.CHOP
        else:
            regime = TrendRegime.RANGING
        
        return regime, trend_strength, persistence
    
    def _detect_volatility(self, highs, lows, closes) -> Tuple[VolatilityState, float]:
        """Classify volatility state."""
        atr = self._calc_atr(highs, lows, closes)
        atr_norm = atr / closes[-1] if closes[-1] > 0 else 0.0
        
        tr = np.maximum(highs - lows,
                       np.maximum(np.abs(highs - np.roll(closes, 1)),
                                 np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atr_p75 = np.percentile(tr, 75) if len(tr) > 10 else atr
        
        if atr > atr_p75 * 1.5:
            return VolatilityState.EXTREME, atr_norm
        elif atr > atr_p75 * 1.2:
            return VolatilityState.ELEVATED, atr_norm
        elif atr < atr_p75 * 0.5:
            return VolatilityState.LOW, atr_norm
        return VolatilityState.NORMAL, atr_norm
    
    def _detect_structure(self, closes, persistence) -> StructuralPhase:
        """Detect structural phase."""
        recent = len(closes) // 5
        recent_mom = (closes[-1] - closes[-recent]) / closes[-recent] if closes[-recent] > 0 else 0
        full_mom = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
        
        if persistence > 0.7:
            if abs(recent_mom) > abs(full_mom) * 0.5:
                return StructuralPhase.IMPULSIVE
            return StructuralPhase.EXHAUSTION
        elif persistence > 0.5:
            if np.sign(recent_mom) == np.sign(full_mom):
                return StructuralPhase.CONSOLIDATION
            return StructuralPhase.CORRECTIVE
        else:
            vol = np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0
            return StructuralPhase.BREAKOUT if vol > 0.01 else StructuralPhase.RANGING
    
    def _assess_risk(self, trend, vol, structure) -> RiskEnvironment:
        """Assess risk environment."""
        if vol == VolatilityState.EXTREME:
            return RiskEnvironment.AVOID
        if trend == TrendRegime.CHOP:
            return RiskEnvironment.AVOID
        if structure == StructuralPhase.EXHAUSTION:
            return RiskEnvironment.AVOID
        if vol == VolatilityState.ELEVATED or structure == StructuralPhase.CORRECTIVE:
            return RiskEnvironment.CAUTION
        if trend in (TrendRegime.BULLISH_WEAK, TrendRegime.BEARISH_WEAK):
            return RiskEnvironment.CAUTION
        return RiskEnvironment.SAFE
    
    # ─── Session-Aware Trading Decision ─────────────────────────────
    
    def _make_decision(self, trend, vol, structure, risk,
                       trend_strength, persistence, session, thresholds
                       ) -> Tuple[bool, int, float, str]:
        """
        Core decision logic — session-aware.
        
        Returns: (should_trade, preferred_direction, size_mult, reason)
        """
        # Hard blocks (all sessions)
        if risk == RiskEnvironment.AVOID:
            return False, 0, 0.0, f"Risk=AVOID ({trend.value}, {vol.value})"
        
        if trend == TrendRegime.CHOP:
            return False, 0, 0.0, "True chop (persistence < 0.40)"
        
        # Persistence check (session-specific threshold)
        if persistence < thresholds['persistence_min']:
            return False, 0, 0.0, (
                f"Persistence {persistence:.2f} < {thresholds['persistence_min']:.2f} "
                f"for {session.value} session"
            )
        
        # Weak trend check
        if trend in (TrendRegime.BULLISH_WEAK, TrendRegime.BEARISH_WEAK):
            if not thresholds['allow_weak_trends']:
                return False, 0, 0.0, f"Weak trend blocked in {session.value} session"
        
        # Ranging breakout check
        if trend == TrendRegime.RANGING:
            if thresholds['allow_ranging_breakout'] and structure == StructuralPhase.BREAKOUT:
                # Allow breakout trades in permissive sessions
                return True, 0, 0.8, f"Ranging breakout allowed in {session.value}"
            return False, 0, 0.0, f"Ranging blocked in {session.value}"
        
        # Structure check — allow impulsive, consolidation, and breakout
        if structure not in (StructuralPhase.IMPULSIVE, StructuralPhase.CONSOLIDATION,
                            StructuralPhase.BREAKOUT):
            return False, 0, 0.0, f"Structure={structure.value} not tradeable"
        
        # Direction
        if trend in (TrendRegime.BULLISH_STRONG, TrendRegime.BULLISH_WEAK):
            direction = 1  # Long only
        elif trend in (TrendRegime.BEARISH_STRONG, TrendRegime.BEARISH_WEAK):
            direction = -1  # Short only
        else:
            direction = 0  # Both
        
        # Size multiplier
        if trend in (TrendRegime.BULLISH_STRONG, TrendRegime.BEARISH_STRONG):
            size_mult = 1.5 if vol in (VolatilityState.NORMAL, VolatilityState.LOW) else 1.0
        else:
            size_mult = 1.0 if vol in (VolatilityState.NORMAL, VolatilityState.LOW) else 0.7
        
        reason = (
            f"{session.value}: {trend.value} | persist={persistence:.2f} | "
            f"str={trend_strength:.2f} | {structure.value} | size={size_mult:.1f}x"
        )
        
        return True, direction, size_mult, reason
    
    # ─── Helpers ────────────────────────────────────────────────────
    
    @staticmethod
    def _calc_atr(highs, lows, closes, period=14):
        """Calculate ATR."""
        tr = np.maximum(highs - lows,
                       np.maximum(np.abs(highs - np.roll(closes, 1)),
                                 np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        return np.mean(tr[-min(period, len(tr)):])
    
    def _avoid_context(self, reason: str) -> PersistenceRegimeContext:
        """Create a blocking context."""
        return PersistenceRegimeContext(
            trend_regime=TrendRegime.RANGING,
            volatility_state=VolatilityState.NORMAL,
            structural_phase=StructuralPhase.RANGING,
            risk_environment=RiskEnvironment.AVOID,
            session=TradingSession.NY,
            trend_strength=0.0,
            persistence_ratio=0.0,
            chop_probability=1.0,
            should_trade=False,
            preferred_direction=0,
            position_size_multiplier=0.0,
            confidence=0.0,
            reason=reason,
        )
