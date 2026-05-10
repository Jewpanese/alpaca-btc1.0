"""
RegimeClassifier — Single source of truth for market state.

Replaces the 15-layer veto hierarchy with one classifier that outputs
a regime + direction + confidence. Everything else flows from this.

Inputs absorbed from old system:
  - ADX Hard Gate → ADX is an input, not a gate
  - 5-vote DirectionalDetector → slopes/EMA/VWAP are inputs
  - HMM Regime → optional confidence boost
  - Range Detector → range_width drives RANGE state
  - Slope Supremacy → extreme slopes = high confidence TREND

Design:
  - Sticky transitions (2-bar confirmation, except VOLATILE = immediate)
  - Each regime has one playbook (see core/playbook.py)
  - No vetoes, no overrides, no layers fighting each other
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ─── Regime States ───────────────────────────────────────────────────

class Regime(Enum):
    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    GRIND_UP   = "GRIND_UP"    # slow directional drift — EMA pullback entries
    GRIND_DOWN = "GRIND_DOWN"
    RANGE      = "RANGE"
    CHOP       = "CHOP"
    VOLATILE   = "VOLATILE"


@dataclass
class RegimeOutput:
    """Output of the regime classifier — single source of truth per bar."""
    regime: Regime
    confidence: float                  # 0.0 - 1.0
    direction: Optional[str] = None   # "LONG", "SHORT", or None
    reason: str = ""                   # Human-readable classification reason
    
    # Raw inputs preserved for logging/debugging
    adx: float = 0.0
    adx_slope: float = 0.0
    slope_75: float = 0.0
    slope_12: float = 0.0
    atr: float = 0.0
    atr_median: float = 0.0
    range_width: float = 0.0
    hmm_state: Optional[str] = None
    hmm_confidence: float = 0.0
    
    @property
    def is_tradeable(self) -> bool:
        return self.regime in (Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE)
    
    @property
    def is_trend(self) -> bool:
        return self.regime in (Regime.TREND_UP, Regime.TREND_DOWN)

    def __str__(self) -> str:
        dir_str = f" {self.direction}" if self.direction else ""
        return f"{self.regime.value}{dir_str} ({self.confidence:.0%}) — {self.reason}"


# ─── Classifier Config ───────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """Thresholds for regime classification. Tunable."""
    
    # TREND detection
    trend_slope75_min: float = 8.0         # |slope75| > this = trend candidate
    trend_slope75_strong: float = 15.0     # |slope75| > this = high confidence trend (Slope Supremacy)
    trend_adx_min: float = 22.0            # ADX floor for trend (softened from old 25)
    trend_adx_strong: float = 30.0         # ADX > this = strong trend
    trend_ema_stack_bonus: float = 0.1     # Confidence boost for aligned EMA stack
    
    # RANGE detection
    range_width_min: float = 8.0           # Range must be at least 8pts
    range_width_max: float = 40.0          # Above this it's probably trending
    range_adx_max: float = 28.0            # ADX must be below this for range
    range_slope75_max: float = 8.0         # |slope75| must be below this for range
    
    # CHOP detection
    chop_range_width_max: float = 8.0      # Tight range = chop
    chop_adx_max: float = 15.0             # Low ADX = chop
    
    # VOLATILE detection
    volatile_atr_spike_mult: float = 2.0   # ATR > 2x median = volatile
    
    # Transition stickiness
    confirm_bars: int = 2                  # Bars to confirm regime change
    volatile_immediate: bool = True        # VOLATILE transitions immediately (safety)
    
    # HMM integration
    hmm_weight: float = 0.15              # How much HMM confidence affects overall confidence
    hmm_trend_boost_threshold: float = 0.7 # HMM confidence above this boosts trend detection


# ─── Classifier ──────────────────────────────────────────────────────

class RegimeClassifier:
    """Single regime classifier replacing all veto layers.
    
    Call update() once per bar. Returns RegimeOutput.
    Sticky transitions prevent whipsawing.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        
        # Current confirmed state
        self.current = RegimeOutput(regime=Regime.CHOP, confidence=0.0, reason="startup")
        
        # Pending transition
        self._pending_regime: Optional[Regime] = None
        self._pending_output: Optional[RegimeOutput] = None
        self._pending_count: int = 0
        
        # History for debugging
        self._history: List[RegimeOutput] = []
        self._max_history: int = 100
    
    def update(
        self,
        price: float,
        vwap: float,
        adx: float,
        slope_75: float,
        slope_12: float,
        atr: float,
        atr_median: float,
        ema_9: float = 0.0,
        ema_21: float = 0.0,
        ema_50: float = 0.0,
        range_width: float = 0.0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        rsi: float = 50.0,
        hmm_state: Optional[str] = None,
        hmm_confidence: float = 0.0,
        adx_slope: float = 0.0,
    ) -> RegimeOutput:
        """Classify current bar into a regime. Call once per bar.
        
        Returns RegimeOutput with regime, direction, and confidence.
        """
        cfg = self.config
        
        # ── Step 1: Detect candidate regime ──────────────────────
        candidate = self._classify_raw(
            price=price, vwap=vwap, adx=adx, 
            slope_75=slope_75, slope_12=slope_12,
            atr=atr, atr_median=atr_median,
            ema_9=ema_9, ema_21=ema_21, ema_50=ema_50,
            range_width=range_width, rsi=rsi,
            hmm_state=hmm_state, hmm_confidence=hmm_confidence,
            adx_slope=adx_slope,
        )
        
        # Preserve raw inputs for debugging
        candidate.adx = adx
        candidate.adx_slope = adx_slope
        candidate.slope_75 = slope_75
        candidate.slope_12 = slope_12
        candidate.atr = atr
        candidate.atr_median = atr_median
        candidate.range_width = range_width
        candidate.hmm_state = hmm_state
        candidate.hmm_confidence = hmm_confidence
        
        # ── Step 2: Apply sticky transitions ─────────────────────
        output = self._apply_stickiness(candidate)
        
        # ── Step 3: Store and return ─────────────────────────────
        self.current = output
        self._history.append(output)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return output
    
    def _classify_raw(
        self, price, vwap, adx, slope_75, slope_12,
        atr, atr_median, ema_9, ema_21, ema_50,
        range_width, rsi, hmm_state, hmm_confidence,
        adx_slope=0.0,
    ) -> RegimeOutput:
        """Raw classification without stickiness. Priority order:
        
        1. VOLATILE (safety first — immediate)
        2. TREND_UP / TREND_DOWN (strongest signal)
        3. RANGE (tradeable but non-directional)
        4. CHOP (default — don't trade)
        """
        cfg = self.config
        abs_slope75 = abs(slope_75)
        
        # ── 1. VOLATILE — ATR spike, get out ────────────────────
        if atr_median > 0 and atr > cfg.volatile_atr_spike_mult * atr_median:
            return RegimeOutput(
                regime=Regime.VOLATILE,
                confidence=min(1.0, atr / (atr_median * cfg.volatile_atr_spike_mult)),
                direction=None,
                reason=f"ATR spike {atr:.2f} > {cfg.volatile_atr_spike_mult}x median {atr_median:.2f}",
            )
        
        # ── 2. TREND detection ──────────────────────────────────
        # Multiple signals vote for trend. Confidence is cumulative.
        trend_score = 0.0
        trend_reasons = []
        
        # Slope Supremacy — extreme slope75 is the strongest trend signal
        if abs_slope75 >= cfg.trend_slope75_strong:
            trend_score += 0.45
            trend_reasons.append(f"slope75={slope_75:+.1f}bps (supremacy)")
        elif abs_slope75 >= cfg.trend_slope75_min:
            trend_score += 0.25
            trend_reasons.append(f"slope75={slope_75:+.1f}bps")
        
        # ADX confirms trend strength
        if adx >= cfg.trend_adx_strong:
            trend_score += 0.25
            trend_reasons.append(f"ADX={adx:.1f} (strong)")
        elif adx >= cfg.trend_adx_min:
            trend_score += 0.15
            trend_reasons.append(f"ADX={adx:.1f}")
        
        # EMA stack alignment
        ema_long = ema_9 > ema_21 > ema_50 > 0
        ema_short = 0 < ema_9 < ema_21 < ema_50
        if ema_long or ema_short:
            trend_score += cfg.trend_ema_stack_bonus
            trend_reasons.append("EMA stack aligned")
        
        # VWAP alignment
        if slope_75 > 0 and price > vwap > 0:
            trend_score += 0.05
        elif slope_75 < 0 and price < vwap and vwap > 0:
            trend_score += 0.05
        
        # Short-term momentum agrees with long-term
        if slope_12 * slope_75 > 0 and abs(slope_12) > 3.0:
            trend_score += 0.05
            trend_reasons.append(f"slope12={slope_12:+.1f}bps agrees")
        
        # HMM boost
        if hmm_state and "TREND" in hmm_state.upper() and hmm_confidence >= cfg.hmm_trend_boost_threshold:
            trend_score += cfg.hmm_weight
            trend_reasons.append(f"HMM={hmm_state}@{hmm_confidence:.0%}")
        
        # ADX slope adjustments — fading vs building trends
        if adx >= cfg.trend_adx_min and adx_slope < -0.5:
            trend_score -= 0.15
            trend_reasons.append("ADX_FADING")
        elif adx_slope > 0.5:
            trend_score += 0.10
            trend_reasons.append("ADX_BUILDING")
        
        # Trend threshold: need at least 0.4 to call it a trend
        if trend_score >= 0.4:
            # Determine direction from slope75 (strongest signal)
            if slope_75 > 0:
                direction = "LONG"
                regime = Regime.TREND_UP
            elif slope_75 < 0:
                direction = "SHORT"
                regime = Regime.TREND_DOWN
            else:
                # slope75 == 0 but we scored enough from other signals
                # Use slope12 as tiebreaker
                if slope_12 > 0:
                    direction = "LONG"
                    regime = Regime.TREND_UP
                elif slope_12 < 0:
                    direction = "SHORT"
                    regime = Regime.TREND_DOWN
                else:
                    # Truly ambiguous — fall through to RANGE/CHOP
                    trend_score = 0.0
            
            if trend_score >= 0.4:
                return RegimeOutput(
                    regime=regime,
                    confidence=min(1.0, trend_score),
                    direction=direction,
                    reason=" + ".join(trend_reasons),
                )
        
        # ── 3. RANGE detection ──────────────────────────────────
        if (range_width >= cfg.range_width_min 
            and range_width <= cfg.range_width_max
            and adx < cfg.range_adx_max
            and abs_slope75 < cfg.range_slope75_max):
            
            conf = 0.5
            range_reasons = [f"range={range_width:.1f}pts"]
            
            # ADX fading into range — boost confidence
            if adx_slope < -0.3:
                conf += 0.10
                range_reasons.append("ADX_FADING_INTO_RANGE")
            
            # Higher confidence if ADX is low and range is well-defined
            if adx < 20:
                conf += 0.15
                range_reasons.append(f"ADX={adx:.1f} (low)")
            if 15 <= range_width <= 30:
                conf += 0.1
                range_reasons.append("sweet spot width")
            
            return RegimeOutput(
                regime=Regime.RANGE,
                confidence=min(1.0, conf),
                direction=None,  # Both directions allowed in range
                reason=" + ".join(range_reasons),
            )
        
        # ── 4. CHOP — default ───────────────────────────────────
        chop_reasons = []
        if range_width > 0 and range_width < cfg.chop_range_width_max:
            chop_reasons.append(f"tight range {range_width:.1f}pts")
        if adx < cfg.chop_adx_max:
            chop_reasons.append(f"ADX={adx:.1f} dead")
        if not chop_reasons:
            chop_reasons.append("no regime detected")
        
        return RegimeOutput(
            regime=Regime.CHOP,
            confidence=0.3,
            direction=None,
            reason=" + ".join(chop_reasons),
        )
    
    def _apply_stickiness(self, candidate: RegimeOutput) -> RegimeOutput:
        """Apply sticky transitions — don't flip on one bar.
        
        VOLATILE transitions immediately (safety).
        Everything else needs confirm_bars consecutive bars.
        """
        cfg = self.config
        
        # VOLATILE always transitions immediately
        if candidate.regime == Regime.VOLATILE and cfg.volatile_immediate:
            self._pending_regime = None
            self._pending_count = 0
            return candidate
        
        # Same regime as current → reset pending, stay
        if candidate.regime == self.current.regime:
            self._pending_regime = None
            self._pending_count = 0
            # Update confidence/direction even if regime stays same
            candidate_copy = RegimeOutput(
                regime=self.current.regime,
                confidence=candidate.confidence,
                direction=candidate.direction,
                reason=candidate.reason,
                adx=candidate.adx,
                slope_75=candidate.slope_75,
                slope_12=candidate.slope_12,
                atr=candidate.atr,
                atr_median=candidate.atr_median,
                range_width=candidate.range_width,
                hmm_state=candidate.hmm_state,
                hmm_confidence=candidate.hmm_confidence,
            )
            return candidate_copy
        
        # Different regime — check pending
        if candidate.regime == self._pending_regime:
            self._pending_count += 1
            self._pending_output = candidate
            if self._pending_count >= cfg.confirm_bars:
                # Confirmed transition
                logger.info(
                    f"REGIME CHANGE: {self.current.regime.value} → {candidate.regime.value} "
                    f"(confirmed after {self._pending_count} bars) — {candidate.reason}"
                )
                self._pending_regime = None
                self._pending_count = 0
                return candidate
        else:
            # New pending regime
            self._pending_regime = candidate.regime
            self._pending_output = candidate
            self._pending_count = 1
        
        # Still in current regime (pending not confirmed)
        # But update confidence/reason from current inputs
        return RegimeOutput(
            regime=self.current.regime,
            confidence=self.current.confidence,
            direction=self.current.direction,
            reason=self.current.reason + f" [pending: {candidate.regime.value} {self._pending_count}/{cfg.confirm_bars}]",
            adx=candidate.adx,
            slope_75=candidate.slope_75,
            slope_12=candidate.slope_12,
            atr=candidate.atr,
            atr_median=candidate.atr_median,
            range_width=candidate.range_width,
            hmm_state=candidate.hmm_state,
            hmm_confidence=candidate.hmm_confidence,
        )
    
    def get_regime_summary(self) -> str:
        """One-line summary for logging."""
        r = self.current
        pending = ""
        if self._pending_regime:
            pending = f" → pending {self._pending_regime.value} ({self._pending_count}/{self.config.confirm_bars})"
        return f"[REGIME] {r.regime.value} {r.confidence:.0%} dir={r.direction}{pending}"


# ─── ER-Based Regime Classifier ──────────────────────────────────────────────
#
# Lightweight replacement for the inline ADX block in big_money_bot._process_bar.
# Uses Efficiency Ratio as the primary gate — ER resets in 8 bars vs ADX's 28+.
#
# Four regimes:
#   TREND  ER >= 0.35 + ADX >= 28 + |slope| >= 1.5 bps → scale-in, trail, big size
#   GRIND  ER >= 0.20 + |slope| >= 0.5 bps (not TREND) → EMA pullback, fixed target
#   RANGE  ER 0.12-0.25, oscillating                   → fade extremes at S/R levels
#   CHOP   ER < 0.12                                    → NO TRADE

class ERRegimeClassifier:
    """
    ER-first regime classifier. Call classify() once per bar.
    Returns one of: 'TREND', 'GRIND', 'RANGE', 'CHOP'.

    Hysteresis: requires PERSIST_BARS consecutive bars in the new regime
    before confirming a flip — prevents thrashing on transition candles.
    """

    # B5 revert (2026-04-24 PM): reverted to pre-B5 values — today's widening
    # killed the edge (market up 0.7%, essentially no trades). The 7-loser chop
    # day (2026-04-21) is better addressed by per-trade risk controls and LIL,
    # not by blanket regime-band widening. Yesterday's config was 0.12 / 0.20.
    CHOP_ER_MAX   = 0.12    # ER below this — every bar reverses, no edge
    TREND_ER_MIN  = 0.35    # ER above this — price moving efficiently in one direction
    TREND_ADX_MIN = 28.0    # ADX floor for trend (smoothed)
    TREND_SLP_MIN = 1.5     # |slope| bps/bar — ~1pt per 3-min bar of consistent drift
    GRIND_ER_MIN  = 0.20    # Moderate ER — directional but slow/noisy
    GRIND_SLP_MIN = 0.5     # Minimal slope confirms direction
    PERSIST_BARS  = 2       # Bars to confirm a regime flip

    def __init__(self):
        self._current      = "RANGE"
        self._candidate    = None
        self._cand_count   = 0
        self._bars_stable  = 0

    @property
    def current(self) -> str:
        return self._current

    def classify(self, er: float, adx: float, slope_bps: float) -> str:
        """
        Classify from current bar's ER, smoothed ADX, and 20-bar slope.

        Args:
            er:        Efficiency Ratio from MarketDirectionEngine (0–1).
            adx:       EMA-smoothed ADX (already smoothed by caller).
            slope_bps: 20-bar linear regression slope in basis-points/bar.

        Returns:
            'TREND' | 'GRIND' | 'RANGE' | 'CHOP'
        """
        abs_slope = abs(slope_bps)

        if er < self.CHOP_ER_MAX:
            raw = "CHOP"
        elif (er >= self.TREND_ER_MIN
              and adx >= self.TREND_ADX_MIN
              and abs_slope >= self.TREND_SLP_MIN):
            raw = "TREND"
        elif er >= self.GRIND_ER_MIN and abs_slope >= self.GRIND_SLP_MIN:
            raw = "GRIND"
        else:
            raw = "RANGE"

        # Hysteresis
        if raw == self._current:
            self._candidate   = None
            self._cand_count  = 0
            self._bars_stable += 1
        else:
            if raw == self._candidate:
                self._cand_count += 1
            else:
                self._candidate  = raw
                self._cand_count = 1

            if self._cand_count >= self.PERSIST_BARS:
                logger.info(
                    f"[ER REGIME] {self._current} -> {raw} | "
                    f"ER={er:.3f} ADX={adx:.1f} slope={slope_bps:+.2f}bps"
                )
                self._current     = raw
                self._candidate   = None
                self._cand_count  = 0
                self._bars_stable = 0

        return self._current

    def reset(self):
        self._current     = "RANGE"
        self._candidate   = None
        self._cand_count  = 0
        self._bars_stable = 0
