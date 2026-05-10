"""
Directional Detector — Lightweight rules-based direction oracle.

Replaces topstep3's 5-model ML ensemble for direction detection,
without the known SHORT bias and signal-flipping problems.

Uses a vote-based system across 5 independent indicators:
  1. ADX + DI+/DI- (trend strength + direction)
  2. 5m EMA stack alignment (9/26/50)
  3. Price vs VWAP
  4. RSI momentum
  5. ADX trend strength as tiebreaker (approximates momentum)

Votes are tallied: 4+ = HIGH confidence, 3 = MEDIUM, ≤2 = NEUTRAL.

Phase 8.1 port: DirectionalRegime — 7-state regime classifier using
momentum slopes + VWAP. Adds nuance beyond binary trend/no-trend.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from strategies.base import MarketState

logger = logging.getLogger(__name__)


# ── Directional Regime (ported from topstep3 Phase 8.1 + 8.30) ──────

class DirectionalRegime(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    GRINDING_UP = "GRINDING_UP"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGE = "RANGE"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    GRINDING_DOWN = "GRINDING_DOWN"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


def classify_directional_regime(
    slope_12: float,
    slope_75: float,
    price: float,
    vwap: float,
    atr: float = 10.0,
    atr_median: float = 10.0,
) -> DirectionalRegime:
    """Classify market into 7 directional regimes.

    Uses momentum slopes (bps) and VWAP relationship.
    Cheap — no ML, just comparisons.

    Args:
        slope_12: 12-bar momentum slope in basis points
        slope_75: 75-bar momentum slope in basis points
        price: Current price
        vwap: VWAP
        atr: Current ATR (ticks)
        atr_median: Rolling median ATR (for grinding detection)
    """
    STRONG_SLOPE = 15.0   # bps — above this = strong trend
    GRIND_SLOPE_MIN = 1.0
    GRIND_SLOPE_MAX = 15.0
    GRIND_ATR_RATIO = 0.85

    is_low_atr = atr < (atr_median * GRIND_ATR_RATIO)
    slope_12_small = GRIND_SLOPE_MIN < abs(slope_12) < GRIND_SLOPE_MAX

    # STRONG_UPTREND: Both slopes positive and large, price above VWAP
    if slope_12 > STRONG_SLOPE and slope_75 > 0 and price > vwap:
        return DirectionalRegime.STRONG_UPTREND

    # GRINDING_UP: Slow steady uptrend — low ATR, small slopes, above VWAP
    if (price > vwap and slope_12 > 0 and slope_75 >= -3.0
            and slope_12_small and is_low_atr):
        return DirectionalRegime.GRINDING_UP

    # WEAK_UPTREND: Short-term positive, above VWAP
    if slope_12 > 0 and price > vwap:
        return DirectionalRegime.WEAK_UPTREND

    # STRONG_DOWNTREND: Both slopes negative and large, price below VWAP
    if slope_12 < -STRONG_SLOPE and slope_75 < 0 and price < vwap:
        return DirectionalRegime.STRONG_DOWNTREND

    # GRINDING_DOWN: Slow steady downtrend
    if (price < vwap and slope_12 < 0 and slope_75 <= 3.0
            and slope_12_small and is_low_atr):
        return DirectionalRegime.GRINDING_DOWN

    # WEAK_DOWNTREND: Short-term negative, below VWAP
    if slope_12 < 0 and price < vwap:
        return DirectionalRegime.WEAK_DOWNTREND

    return DirectionalRegime.RANGE


def check_regime_entry_gate(
    direction: str,
    regime: DirectionalRegime,
    slope_75: float,
    slope_12: float,
) -> tuple:
    """Block counter-trend entries based on regime + slopes.

    Returns (allowed: bool, reason: str).
    Ported from topstep3 Phase 8.2/8.12/8.13/8.23 — simplified.
    """
    if direction == "LONG":
        # Block longs in any downtrend regime
        if regime in (DirectionalRegime.STRONG_DOWNTREND,
                      DirectionalRegime.WEAK_DOWNTREND,
                      DirectionalRegime.GRINDING_DOWN):
            return False, f"LONG blocked in {regime.value}"
        # Block longs when long-term slope is bearish (even in RANGE)
        if slope_75 < -3.0:
            return False, f"LONG blocked: slope_75={slope_75:+.1f}bps bearish"
        # Block longs when short-term momentum is strongly bearish
        if slope_12 < -5.0:
            return False, f"LONG blocked: slope_12={slope_12:+.1f}bps bearish"
        return True, ""

    if direction == "SHORT":
        # Block shorts in any uptrend regime
        if regime in (DirectionalRegime.STRONG_UPTREND,
                      DirectionalRegime.WEAK_UPTREND,
                      DirectionalRegime.GRINDING_UP):
            return False, f"SHORT blocked in {regime.value}"
        if slope_75 > 3.0:
            return False, f"SHORT blocked: slope_75={slope_75:+.1f}bps bullish"
        if slope_12 > 5.0:
            return False, f"SHORT blocked: slope_12={slope_12:+.1f}bps bullish"
        return True, ""

    return False, f"Invalid direction: {direction}"


def check_entry_location(
    direction: str,
    price: float,
    vwap: float,
    atr: float,
    rsi_14: float,
    slope_75: float,
    slope_75_prev: float = 0.0,
    high_20: float = 0.0,
    low_20: float = 0.0,
    efficiency_ratio: float = 1.0,
    session: str = "US_MIDDAY",
) -> tuple:
    """Entry Location Filter — ported from topstep3 Phase 8.33.

    Prevents entries at price extremes, exhaustion zones, and in choppy trends.
    Direction is already decided; this answers "is this a GOOD PLACE to enter?"

    Checks:
      1. VWAP Extension — don't chase if price extended > 0.25× ATR from VWAP
         (direction-specific: only penalizes extension IN the trade direction)
      2. RSI Exhaustion — momentum exhausted (>78 long, <22 short)
      3. Slope Deceleration — slope75 losing strength (fading momentum)
      4. Efficiency Ratio — < 40% means chop disguised as trend
      5. Price Extreme — too close to 20-bar high/low (within 0.25× ATR)

    Returns (allowed: bool, reason: str).
    """
    if atr <= 0:
        return True, ""  # Can't evaluate without ATR

    reasons_blocked = []

    # 1. VWAP Extension — DISABLED
    # Trend days naturally trade extended from VWAP — blocking this kills trend entries.
    # The exhaustion check (already in BigMoneyBot) handles extreme VWAP extension
    # with its multi-condition gate (VWAP dist + ADX decline + RSI + ATR spike).
    # Keeping this comment for reference if we want to re-enable with a wider threshold.

    # 2. RSI Exhaustion
    if direction == "LONG" and rsi_14 > 78:
        reasons_blocked.append(f"RSI exhausted {rsi_14:.1f} > 78")
    elif direction == "SHORT" and rsi_14 < 22:
        reasons_blocked.append(f"RSI exhausted {rsi_14:.1f} < 22")

    # 3. Slope Deceleration — momentum fading, not strengthening
    if slope_75_prev != 0:
        slope_delta = slope_75 - slope_75_prev
        if direction == "LONG" and slope_75 > 3.0 and slope_delta < -5.0:
            reasons_blocked.append(f"Slope decelerating Δ={slope_delta:+.1f}bps")
        elif direction == "SHORT" and slope_75 < -3.0 and slope_delta > 5.0:
            reasons_blocked.append(f"Slope decelerating Δ={slope_delta:+.1f}bps")

    # 4. Efficiency Ratio — chop disguised as trend
    if efficiency_ratio < 0.40:
        reasons_blocked.append(f"Efficiency {efficiency_ratio:.0%} < 40% (choppy)")

    # 5. Price Extreme — too close to 20-bar high/low
    if high_20 > 0 and low_20 > 0 and atr > 0:
        if direction == "LONG":
            dist_from_high = high_20 - price
            if dist_from_high < (0.5 * atr) and dist_from_high >= 0:
                reasons_blocked.append(f"Near 20-bar high ({dist_from_high:.2f}pts < 0.5×ATR)")
        elif direction == "SHORT":
            dist_from_low = price - low_20
            if dist_from_low < (0.5 * atr) and dist_from_low >= 0:
                reasons_blocked.append(f"Near 20-bar low ({dist_from_low:.2f}pts < 0.5×ATR)")

    if reasons_blocked:
        return False, f"ENTRY_LOCATION: {' | '.join(reasons_blocked)}"
    return True, ""


class BiasDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class DirectionalBias:
    """Output of the directional detector."""
    direction: BiasDirection
    confidence: float          # 0.0-1.0
    trend_strength: float      # ADX value
    reasons: list = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """True if confidence is MEDIUM or HIGH (>= 0.6)."""
        return self.confidence >= 0.6 and self.direction != BiasDirection.NEUTRAL


# ── Slope Supremacy (ported from topstep3 Phase 8.32/8.34) ──────────

# Session-weighted thresholds — overnight needs stronger slope to trust
# Fix 2026-03-26: Recalibrated for MES minute bars. Old thresholds (12-18bps)
# were from T3/ES and never triggered on MES (typical S75 = 0.3-0.5bps).
# New thresholds ~3-4x lower to match MES scale.
SUPREMACY_SESSION_THRESHOLDS = {
    'NY_OPEN': 3.0,
    'US_OPEN': 3.0,
    'LONDON': 2.5,
    'PRE_US': 3.0,
    'PRE_NY': 3.0,
    'US_MIDDAY': 4.0,
    'US_AFTERNOON': 3.5,
    'US_CLOSE': 3.5,
    'ASIAN': 5.0,
    'OVERNIGHT': 5.0,
    'AFTER_HOURS': 5.0,
    'LATE_GLOBEX': 5.0,
}


def check_slope_supremacy(
    slope_75_bps: float,
    slope_12_bps: float,
    session: str = 'US_MIDDAY',
) -> tuple[bool, Optional[str], str]:
    """Slope Supremacy — when |slope75| is extreme, slope dictates direction.

    Ported from topstep3 Phase 8.32/8.34.

    Returns:
        (is_active, direction_str_or_None, reason)
        direction_str is "LONG" or "SHORT" or None if blocked/inactive.
    """
    threshold = SUPREMACY_SESSION_THRESHOLDS.get(session, 15.0)
    slope_75_abs = abs(slope_75_bps)

    if slope_75_abs < threshold:
        return False, None, f"Slope75 {slope_75_bps:+.1f}bps < ±{threshold} for {session}"

    # Extreme slope — determine direction
    direction_str = "LONG" if slope_75_bps > 0 else "SHORT"
    slope_dir = 1 if slope_75_bps > 0 else -1
    slope_12_dir = 1 if slope_12_bps > 0 else (-1 if slope_12_bps < 0 else 0)

    # slope12 must agree — short-term divergence = wait
    if slope_12_dir != slope_dir:
        return True, None, (
            f"SUPREMACY BLOCKED: Slope75={slope_75_bps:+.1f}bps ({direction_str}) "
            f"but Slope12={slope_12_bps:+.1f}bps disagrees"
        )

    return True, direction_str, (
        f"SUPREMACY: Slope75={slope_75_bps:+.1f}bps → {direction_str} | "
        f"Slope12={slope_12_bps:+.1f}bps agrees"
    )


def check_short_override(
    regime: DirectionalRegime,
    price: float,
    vwap: float,
    atr: float,
    slope_75_bps: float,
    rsi_14: float,
) -> tuple[bool, str]:
    """Deterministic SHORT injection on failed bounces in strong downtrends.

    Simplified port from topstep3 Phase 8.8. No ML signal check needed
    since Big Money Bot doesn't use ML.

    Returns:
        (should_short, reason)
    """
    # Must be STRONG_DOWNTREND
    if regime != DirectionalRegime.STRONG_DOWNTREND:
        return False, f"Not STRONG_DOWNTREND (regime={regime.value})"

    # VWAP deviation check — must be well below VWAP
    if atr <= 0:
        return False, "ATR is zero"
    vwap_dev_atr = (price - vwap) / atr
    if vwap_dev_atr >= -0.5:
        return False, f"VWAP dev {vwap_dev_atr:.2f}x ATR >= -0.5 (not bearish enough)"

    # Sustained momentum — slope75 must be strongly negative
    if slope_75_bps >= -15.0:
        return False, f"Slope75 {slope_75_bps:+.1f}bps >= -15 (not trending down hard)"

    # RSI must show weakness (< 45)
    if rsi_14 >= 45:
        return False, f"RSI {rsi_14:.1f} >= 45 (too strong for short override)"

    # Exhaustion guard — don't short the capitulation bottom
    if rsi_14 < 25:
        return False, f"RSI {rsi_14:.1f} < 25 (capitulation zone, snapback risk)"
    if vwap_dev_atr < -3.0:
        return False, f"VWAP dev {vwap_dev_atr:.2f}x ATR < -3.0 (extremely extended)"

    return True, (
        f"SHORT OVERRIDE: regime=STRONG_DOWNTREND, vwap_dev={vwap_dev_atr:.2f}xATR, "
        f"slope75={slope_75_bps:+.1f}bps, RSI={rsi_14:.1f}"
    )


class DirectionalDetector:
    """Vote-based directional bias detector.

    Each indicator casts a LONG or SHORT vote. Votes are tallied
    to determine overall bias and confidence level.
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        rsi_bull_threshold: float = 55.0,
        rsi_bear_threshold: float = 45.0,
    ):
        self.adx_threshold = adx_threshold
        self.rsi_bull = rsi_bull_threshold
        self.rsi_bear = rsi_bear_threshold

    def detect(self, state: MarketState) -> DirectionalBias:
        """Determine directional bias from current market state.

        Returns DirectionalBias with direction, confidence, and reasons.
        """
        long_votes = 0
        short_votes = 0
        reasons: list[str] = []

        # ── Vote 1: ADX + DI direction (3-min ADX for less noise) ───
        adx = state.adx_3m if state.adx_3m > 0 else state.adx_14
        di_plus = state.di_plus_3m if state.adx_3m > 0 else state.di_plus_14
        di_minus = state.di_minus_3m if state.adx_3m > 0 else state.di_minus_14

        if adx > self.adx_threshold:
            if di_plus > di_minus:
                long_votes += 1
                reasons.append(f"ADX3m={adx:.1f}>25, DI+={di_plus:.1f}>DI-={di_minus:.1f} → LONG")
            elif di_minus > di_plus:
                short_votes += 1
                reasons.append(f"ADX3m={adx:.1f}>25, DI-={di_minus:.1f}>DI+={di_plus:.1f} → SHORT")
            else:
                reasons.append(f"ADX3m={adx:.1f}>25 but DI+=DI- → no vote")
        else:
            reasons.append(f"ADX3m={adx:.1f}<25 → weak trend, no DI vote")

        # ── Vote 2: 5m EMA stack alignment ──────────────────────────
        ema9 = state.ema_5m_9
        ema26 = state.ema_5m_26
        ema50 = state.ema_5m_50

        if ema9 > ema26 > ema50 and ema50 > 0:
            long_votes += 1
            reasons.append(f"EMA stack 9>{ema9:.1f} > 26>{ema26:.1f} > 50>{ema50:.1f} → LONG")
        elif ema9 < ema26 < ema50 and ema50 > 0:
            short_votes += 1
            reasons.append(f"EMA stack 9<{ema9:.1f} < 26<{ema26:.1f} < 50<{ema50:.1f} → SHORT")
        else:
            reasons.append("EMA stack not aligned → no vote")

        # ── Vote 3: Price vs VWAP ───────────────────────────────────
        if state.vwap > 0:
            vwap_dist = (state.price - state.vwap) / state.vwap * 100
            if state.price > state.vwap:
                long_votes += 1
                reasons.append(f"Price {state.price:.2f} > VWAP {state.vwap:.2f} (+{vwap_dist:.2f}%) → LONG")
            elif state.price < state.vwap:
                short_votes += 1
                reasons.append(f"Price {state.price:.2f} < VWAP {state.vwap:.2f} ({vwap_dist:.2f}%) → SHORT")

        # ── Vote 4: RSI momentum ────────────────────────────────────
        rsi = state.rsi_14
        if rsi > self.rsi_bull:
            long_votes += 1
            reasons.append(f"RSI={rsi:.1f}>{self.rsi_bull} → bullish momentum")
        elif rsi < self.rsi_bear:
            short_votes += 1
            reasons.append(f"RSI={rsi:.1f}<{self.rsi_bear} → bearish momentum")
        else:
            reasons.append(f"RSI={rsi:.1f} neutral ({self.rsi_bear}-{self.rsi_bull})")

        # ── Vote 5: Price vs EMA structure (momentum proxy) ────────
        # Use 1m EMA 9 vs EMA 50 as momentum proxy (replaces bar history)
        if state.ema_9 > 0 and state.ema_50 > 0:
            if state.ema_9 > state.ema_50:
                long_votes += 1
                reasons.append(f"1m EMA9={state.ema_9:.2f} > EMA50={state.ema_50:.2f} → momentum LONG")
            elif state.ema_9 < state.ema_50:
                short_votes += 1
                reasons.append(f"1m EMA9={state.ema_9:.2f} < EMA50={state.ema_50:.2f} → momentum SHORT")

        # ── Tally votes ─────────────────────────────────────────────
        total_votes = 5
        net = long_votes - short_votes

        if long_votes >= 4:
            direction = BiasDirection.LONG
            confidence = long_votes / total_votes
        elif short_votes >= 4:
            direction = BiasDirection.SHORT
            confidence = short_votes / total_votes
        elif long_votes >= 3 and long_votes > short_votes:
            direction = BiasDirection.LONG
            confidence = 0.6  # MEDIUM
        elif short_votes >= 3 and short_votes > long_votes:
            direction = BiasDirection.SHORT
            confidence = 0.6  # MEDIUM
        else:
            direction = BiasDirection.NEUTRAL
            confidence = max(long_votes, short_votes) / total_votes

        reasons.append(
            f"TALLY: {long_votes} LONG / {short_votes} SHORT → "
            f"{direction.value} (confidence={confidence:.0%})"
        )

        return DirectionalBias(
            direction=direction,
            confidence=confidence,
            trend_strength=adx,
            reasons=reasons,
        )
