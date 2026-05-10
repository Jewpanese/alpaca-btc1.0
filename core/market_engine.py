"""
Market Direction Engine — Two-layer momentum system.

Redesigned 2026-04-08 to separate direction from timing:

  Layer 1: Direction (20-bar slope)
    20 minutes of history anchors which way the market is moving.
    Stable enough to not flip on noise, responsive enough to catch
    a session move without being diluted by prior action.
    Answers: "Which direction?"

  Layer 2: Timing (Efficiency Ratio, 12 bars)
    ER = |net displacement| / sum(|bar changes|)
    ER near 1.0 = every bar moves the same direction = trending efficiently
    ER near 0.0 = price oscillates with no net progress = chop
    ER RISING = momentum building = trend initiating = early entry signal
    ER HIGH   = trend running = continuation entry
    ER FALLING = trend losing efficiency = sit out
    Answers: "Is NOW a good time to enter?"

  ADX Trajectory (supporting layer)
    Rising ADX = trend gaining energy → confirms entry
    Falling ADX = trend losing energy → blocks entry
    Answers: "Is the trend healthy?"

Conviction mapping:
  Direction + ER rising + ADX rising  → HIGH   (catching the beginning)
  Direction + ER high   + ADX any     → MEDIUM (trend running)
  Direction + ER low/falling           → NONE   (momentum lost, sit out)
  Direction + ADX falling hard         → NONE   (blocked)

All inputs are pre-computed values from FeatureEngine / MarketState.
Zero new indicators beyond what is already computed.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


class MarketBias(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


class ConvictionLevel(Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    NONE   = "NONE"


@dataclass
class DirectionResult:
    bias:        MarketBias
    conviction:  ConvictionLevel
    probe_mult:  float        # 1.0 = full probe, 0.5 = half, 0.0 = no trade
    direction_slope: float    # 20-bar slope in bps
    er:          float        # current Efficiency Ratio (0-1)
    er_prev:     float        # ER from previous bar (to detect rising/falling)
    adx_slope:   float        # ADX change per bar
    reason:      str


@dataclass
class MarketEngineConfig:
    # All windows below are in BARS. With 3-min bars (the bot's primary timeframe):
    #   20 bars = 60 min, 8 bars = 24 min, 3 bars = 9 min, 2 bars = 6 min.

    # --- Layer 1: Direction ---
    direction_window: int   = 20      # 60 min of direction context on 3-min bars — stable trend anchor
    direction_min_bps: float = 0.15   # very loose — just needs some slope to have a direction

    # --- Layer 2: Timing (Efficiency Ratio) ---
    er_window:    int   = 8           # 24 min ER window on 3-min bars (was 12=36min, too slow)
    er_high:          float = 0.40    # ER above this = trend running (lowered 0.45→0.40: 3-min bars
                                      # consolidate slightly within each bar, reducing ER naturally)
    er_high_min_slope: float = 1.50   # slope must be this strong (bps) to use ER-high path
                                      # raised 0.60→1.50 for 3-min bars: on 3-min, 1.50bps ≈ 0.43pts/bar
                                      # = slow but real drift. Prevents tiny corrective ticks from triggering.
    er_rising_min: float = 0.08       # ER must rise at least this much bar-over-bar to flag "rising"
    er_low:       float = 0.08        # ER below this = choppy. 0.08 matches yesterday's live config
                                      # which produced the good session — 0.15 was too strict and
                                      # killed the edge (2026-04-24 AM: blocked multiple valid setups).

    # --- ADX Trajectory ---
    adx_lookback:    int   = 3        # 9 min ADX trajectory window (OK for 3-min bars)
    adx_rising_min:  float = 0.15     # ADX rising >= this/bar = healthy trend
    adx_falling_max: float = -0.5     # ADX falling faster = block

    # --- Continuation (established trend) ---
    adx_continuation_level: float = 28.0  # lowered 30→28: 3-min ADX is stable, 28 is a real trend
    adx_continuation_er:    float = 0.20   # ER just needs to be above noise for continuation

    # --- High-ER override ---
    er_override_threshold: float = 0.55    # ER above this overrides ADX falling block

    # --- Short-term slope override ---
    # 3-bar (9-min) slope override: catches current momentum vs. the 20-bar (60-min) trend.
    # On 3-min bars, slope amplitudes are ~3× larger than 1-min (same pts/time = 3× pts/bar).
    # Threshold raised to match: 3.0bps on 3-min ≈ 1pt per 3-min bar of consistent movement.
    short_slope_window:  int   = 3     # 9 min short-term slope window
    short_slope_min_bps: float = 3.0   # raised 1.50→3.0 for 3-min bars: prevents slow pullbacks
                                       # from overriding the 60-min trend direction

    # --- Trend persistence ---
    # Require opposing direction to hold for N bars before flipping.
    # 2 bars = 6 min on 3-min bars (was 3=9min — too slow to catch real reversals).
    trend_persist_bars: int = 2

    # --- Probe sizing ---
    probe_high:   float = 1.0
    probe_medium: float = 0.5


class MarketDirectionEngine:
    """
    Evaluates market direction and entry timing on every bar.
    Call evaluate() once per bar — returns DirectionResult.
    """

    def __init__(self, config: MarketEngineConfig = None):
        self.config = config or MarketEngineConfig()
        self._prev_er: float = 0.0
        # Trend persistence state
        self._current_bias: MarketBias = MarketBias.FLAT
        self._flip_candidate: MarketBias = MarketBias.FLAT
        self._flip_bars: int = 0

    # ------------------------------------------------------------------ #
    # Layer 1: Direction                                                   #
    # ------------------------------------------------------------------ #

    def _direction_slope(self, closes: np.ndarray) -> float:
        """
        Linear regression slope of last direction_window closes, in bps/bar.
        """
        n = self.config.direction_window
        if len(closes) < n:
            return 0.0
        window = closes[-n:]
        x = np.arange(len(window), dtype=float)
        slope = np.polyfit(x, window, 1)[0]
        avg = np.mean(window)
        if avg == 0:
            return 0.0
        return (slope / avg) * 10000.0

    def _direction_slope_n(self, closes: np.ndarray, n: int) -> float:
        """Same as _direction_slope but over a custom window of n bars."""
        if len(closes) < n + 1:
            return 0.0
        window = closes[-(n + 1):]
        x = np.arange(len(window), dtype=float)
        slope = np.polyfit(x, window, 1)[0]
        avg = np.mean(window)
        if avg == 0:
            return 0.0
        return (slope / avg) * 10000.0

    def _get_bias(self, slope: float) -> MarketBias:
        cfg = self.config
        if slope > cfg.direction_min_bps:
            return MarketBias.LONG
        if slope < -cfg.direction_min_bps:
            return MarketBias.SHORT
        return MarketBias.FLAT

    # ------------------------------------------------------------------ #
    # Layer 2: Timing — Efficiency Ratio                                   #
    # ------------------------------------------------------------------ #

    def _efficiency_ratio(self, closes: np.ndarray) -> float:
        """
        Kaufman Efficiency Ratio over er_window bars.
        ER = |net displacement| / sum(|bar-to-bar changes|)
        1.0 = pure trend, 0.0 = pure chop.
        """
        n = self.config.er_window
        if len(closes) < n + 1:
            return 0.0
        window = closes[-(n + 1):]
        net  = abs(window[-1] - window[0])
        path = np.sum(np.abs(np.diff(window)))
        if path == 0:
            return 0.0
        return float(net / path)

    # ------------------------------------------------------------------ #
    # ADX Trajectory                                                        #
    # ------------------------------------------------------------------ #

    def _adx_trajectory(self, adx_history: list) -> tuple:
        """
        Returns (category: str, slope: float).
        'rising', 'flat', or 'falling'.
        """
        cfg = self.config
        n = cfg.adx_lookback
        if len(adx_history) < n + 1:
            return 'flat', 0.0
        now  = adx_history[-1]
        prev = adx_history[-(n + 1)]
        slope = (now - prev) / n
        if slope >= cfg.adx_rising_min:
            return 'rising', slope
        if slope <= cfg.adx_falling_max:
            return 'falling', slope
        return 'flat', slope

    # ------------------------------------------------------------------ #
    # Public: evaluate()                                                   #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        closes:      np.ndarray,
        adx_history: list,
        atr_now:     float = 0.0,
        atr_median:  float = 0.0,
        bars:        list  = None,   # kept for API compatibility, unused
    ) -> DirectionResult:
        """
        Evaluate direction + timing and return a DirectionResult.

        Args:
            closes:      numpy array of recent 1-min close prices
            adx_history: list of recent smoothed ADX values
            atr_now:     current ATR (unused, kept for compatibility)
            atr_median:  rolling median ATR (unused, kept for compatibility)
            bars:        raw bar dicts (unused, kept for compatibility)

        Returns:
            DirectionResult with bias, conviction, probe_mult, and diagnostics.
        """
        cfg = self.config

        # --- Layer 1: Direction (20-bar) ---
        slope = self._direction_slope(closes)
        bias  = self._get_bias(slope)

        # --- Layer 1A: Short-term slope override (3-bar / 9-min) ---
        # If the 3-bar slope is strong AND contradicts the 20-bar bias,
        # the current momentum wins. Fixes post-spike stale direction:
        # 20-bar says SHORT (dominated by the flush), 3-bar says LONG (recovery NOW).
        short_slope = self._direction_slope_n(closes, cfg.short_slope_window)
        short_bias  = self._get_bias(short_slope)
        short_override_active = False
        if (short_bias != MarketBias.FLAT
                and abs(short_slope) >= cfg.short_slope_min_bps
                and short_bias != bias):
            bias  = short_bias
            slope = short_slope
            short_override_active = True

        # --- Layer 1B: Trend persistence ---
        # Require the opposing direction to hold for N bars before flipping.
        # A single red bar in an uptrend is noise — N=2 bars of opposing signal
        # confirms a genuine reversal vs a pullback.
        # FLAT never blocks a flip — only direction-vs-direction triggers this.
        if bias != MarketBias.FLAT and bias != self._current_bias and self._current_bias != MarketBias.FLAT:
            # Direction is trying to flip
            if self._flip_candidate == bias:
                self._flip_bars += 1
            else:
                self._flip_candidate = bias
                self._flip_bars = 1
            if self._flip_bars < cfg.trend_persist_bars:
                # Not confirmed yet — hold previous bias
                bias = self._current_bias
                slope = slope  # keep computed slope for diagnostics
        else:
            # Bias is stable or starting fresh from FLAT
            if bias != MarketBias.FLAT:
                self._current_bias = bias
            self._flip_candidate = MarketBias.FLAT
            self._flip_bars = 0
            if bias != MarketBias.FLAT:
                self._current_bias = bias

        if bias == MarketBias.FLAT:
            er = self._efficiency_ratio(closes)
            self._prev_er = er
            self._current_bias = MarketBias.FLAT
            self._flip_candidate = MarketBias.FLAT
            self._flip_bars = 0
            return DirectionResult(
                bias=MarketBias.FLAT,
                conviction=ConvictionLevel.NONE,
                probe_mult=0.0,
                direction_slope=slope,
                er=er,
                er_prev=self._prev_er,
                adx_slope=0.0,
                reason=f"FLAT: slope={slope:+.3f}bps — no directional bias",
            )

        # --- ADX Trajectory ---
        adx_cat, adx_slope = self._adx_trajectory(adx_history)
        adx_level = adx_history[-1] if adx_history else 0.0

        # Hard block: ADX falling fast = trend losing energy
        # Exception: if ER is very high the move is too efficient to block —
        # post-spike ADX decay is normal while price grinds steadily in one direction.
        _adx_override_active = False
        if adx_cat == 'falling' and adx_level < cfg.adx_continuation_level:
            er      = self._efficiency_ratio(closes)
            er_prev = self._prev_er
            self._prev_er = er
            if er >= cfg.er_override_threshold:
                # High-ER override: fall through with probe capped at 0.5x
                _adx_override_active = True
            else:
                return DirectionResult(
                    bias=bias,
                    conviction=ConvictionLevel.NONE,
                    probe_mult=0.0,
                    direction_slope=slope,
                    er=er,
                    er_prev=er_prev,
                    adx_slope=adx_slope,
                    reason=(
                        f"BLOCKED: ADX falling {adx_slope:+.2f}/bar and below {cfg.adx_continuation_level} "
                        f"— trend losing energy"
                    ),
                )

        # --- Layer 2: Timing — Efficiency Ratio ---
        if not _adx_override_active:
            er      = self._efficiency_ratio(closes)
            er_prev = self._prev_er
            self._prev_er = er

        er_rising  = (er - er_prev) >= cfg.er_rising_min
        er_high    = er >= cfg.er_high
        er_low     = er < cfg.er_low

        # ER too low = choppy, no edge regardless of direction
        if er_low and adx_level < cfg.adx_continuation_level:
            return DirectionResult(
                bias=bias,
                conviction=ConvictionLevel.NONE,
                probe_mult=0.0,
                direction_slope=slope,
                er=er,
                er_prev=er_prev,
                adx_slope=adx_slope,
                reason=(
                    f"BLOCKED: ER={er:.3f} < {cfg.er_low} (choppy) | "
                    f"ADX={adx_level:.1f} not established"
                ),
            )

        # --- Conviction scoring ---
        # HIGH: ER rising (catching the beginning) + ADX rising
        if er_rising and adx_cat == 'rising':
            conviction = ConvictionLevel.HIGH
            probe_mult = cfg.probe_high
            reason = (
                f"HIGH: {bias.value} | slope={slope:+.3f}bps | "
                f"ER rising {er_prev:.3f}->{er:.3f} | ADX rising {adx_slope:+.2f}/bar"
            )

        # HIGH: ER rising strongly even with flat ADX
        elif er_rising and er >= cfg.er_high:
            conviction = ConvictionLevel.HIGH
            probe_mult = cfg.probe_high
            reason = (
                f"HIGH: {bias.value} | slope={slope:+.3f}bps | "
                f"ER rising {er_prev:.3f}->{er:.3f} (strong) | ADX={adx_level:.1f}"
            )

        # MEDIUM: ER high = trend running efficiently
        # Requires slope >= er_high_min_slope to prevent stale ER from a prior trend
        # triggering a new entry on a tiny corrective move.
        elif er_high and abs(slope) >= cfg.er_high_min_slope:
            conviction = ConvictionLevel.MEDIUM
            probe_mult = cfg.probe_medium
            reason = (
                f"MEDIUM: {bias.value} | slope={slope:+.3f}bps | "
                f"ER={er:.3f} >= {cfg.er_high} (trend running) | ADX={adx_level:.1f}"
            )

        # MEDIUM: ER rising (momentum building) even if not high yet
        elif er_rising:
            conviction = ConvictionLevel.MEDIUM
            probe_mult = cfg.probe_medium
            reason = (
                f"MEDIUM: {bias.value} | slope={slope:+.3f}bps | "
                f"ER rising {er_prev:.3f}->{er:.3f} | ADX={adx_level:.1f}"
            )

        # MEDIUM: Established trend (high ADX) with adequate ER and meaningful slope
        elif (adx_level >= cfg.adx_continuation_level and er >= cfg.adx_continuation_er
              and abs(slope) >= cfg.er_high_min_slope):
            conviction = ConvictionLevel.MEDIUM
            probe_mult = cfg.probe_medium
            reason = (
                f"MEDIUM (continuation): {bias.value} | slope={slope:+.3f}bps | "
                f"ADX={adx_level:.1f} >= {cfg.adx_continuation_level} | ER={er:.3f}"
            )

        else:
            # Direction present but no timing confirmation
            conviction = ConvictionLevel.NONE
            probe_mult = 0.0
            reason = (
                f"NONE: {bias.value} | slope={slope:+.3f}bps | "
                f"ER={er:.3f} (not rising, not high) | ADX={adx_level:.1f} — no timing edge"
            )

        # When high-ER override is active (ADX falling but ER >= threshold),
        # cap probe at 0.5x — don't go full size while ADX is decaying.
        if _adx_override_active and probe_mult > cfg.probe_medium:
            probe_mult = cfg.probe_medium
            reason += f" [ER-override: ADX falling, probe capped 0.5x]"

        # Tag reason when short-slope override fired
        if short_override_active:
            reason += f" [3bar-override: {short_slope:+.3f}bps overrides 20bar]"

        # Tag reason when trend persistence is holding a flip back
        if self._flip_bars > 0 and self._flip_bars < cfg.trend_persist_bars:
            reason += f" [persist: holding {self._current_bias.value}, flip {self._flip_bars}/{cfg.trend_persist_bars}]"

        return DirectionResult(
            bias=bias,
            conviction=conviction,
            probe_mult=probe_mult,
            direction_slope=slope,
            er=er,
            er_prev=er_prev,
            adx_slope=adx_slope,
            reason=reason,
        )
