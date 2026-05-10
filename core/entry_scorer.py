"""
EntryScorer — replacement for ADX-gated DIRECT ENTRY logic.

Status: SKETCH FOR REVIEW. Not yet wired into big_money_bot.py. Do not ship
without the validation steps listed at the bottom of this file.

Origin of weights
-----------------
All coefficients are the mean across 6 walk-forward folds on the 16-session
April 2026 MES 3-min sample (2026-03-30 through 2026-04-20 inclusive, 16,232
bars after 80-bar warmup). Full research trail and the specific artifact that
produced each weight set:

  research/feature_matrix.py           - Phase 1   (correlation + VIF)
    → research/features.csv
    → research/correlation_matrix.csv
    → research/vif.csv

  research/ic_analysis.py              - Phase 2   (IC per feature per horizon)
    → research/ic_verdict.csv

  research/conditional_ic.py           - Phase 2.5 (IC by ER-regime + session)
    → research/conditional_ic_er_signed.csv
    → research/conditional_ic_session_signed.csv

  research/walk_forward_refit.py       - Phase 3b  (OOS weight calibration)
    → research/walk_forward_refit_direction_raw.csv  [direction weights]
    → research/walk_forward_refit_size_raw.csv       [size weights]
    → research/walk_forward_refit_metrics.csv        [per-fold OOS metrics]

  research/score_distribution.py       - Phase 3c  (threshold calibration)
    → prints q05/q10/q50/q90/q95 of direction_pred used below

Evidence summary
----------------
  Dropped from entry logic (no OOS edge):
    ADX, slope_bps, vol_z, atr_z

  Direction signal (signed 5-bar forward return prediction, bps):
    - dist_vwap_atr          mean-reversion (stable across ALL ER regimes)
    - dist_ema9_atr_gated    short-horizon MR (only when ER >= 0.20)
    - dist_vwap_x_er         interaction — tempers MR in high-ER regimes

  Size signal (predicted |5-bar return| magnitude, bps):
    - er_20, hurst_64, body_ratio  — all three stable, all three positive

  Walk-forward refit metrics (6 folds, 10-session train / 1-session test):
    Mean OOS IC            +0.058
    Mean OOS hit rate      0.582
    Negative folds         0 of 6
    All weight sign consistencies: 83-100%
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ─── Coefficients (mean of 6 walk-forward folds) ──────────────────────────
# Direction: predicts signed 5-bar forward return in basis points.
_W_INT_DIR      = +0.359216
_W_VWAP         = -0.002869
_W_EMA9_GATED   = -0.805713
_W_VWAP_X_ER    = +0.023020

# Size: predicts |5-bar forward return| in basis points.
_W_INT_SIZE     = +1.570993
_W_ER           = +2.007387
_W_HURST        = +2.288165
_W_BODY         = +0.994383

# Regime gate — dist_ema9 only meaningful when ER says there's a direction.
_ER_GATE_EMA9   = 0.20
# CHOP gate — never trade below this (matches ERRegimeClassifier.CHOP_ER_MAX).
_ER_CHOP_MAX    = 0.12

# ─── Entry thresholds (empirical quantiles of direction_pred distribution) ──
# Derived from full 16-session sample after applying the production weights.
# These are STARTING POINTS — must be re-calibrated after paper-trading the
# scorer for 10-20 live sessions.
#
# Distribution of direction_pred (bps) on the research sample:
#   q05 = -0.48   q10 = -0.20   q50 = +0.36   q90 = +0.82   q95 = +1.14
#
# ──────────────────────────────────────────────────────────────────────
# ASYMMETRIC thresholds are INTENTIONAL and DATA-DRIVEN.
# The research-sample direction_pred distribution is not symmetric around
# zero: the upside tail is wider than the downside tail. That asymmetry is
# structural (MES drifts up, and VWAP mean-reversion is louder off up-
# extensions than off down-extensions in this window). Forcing symmetric
# thresholds would impose model error for no empirical benefit.
#
# Whether the asymmetry is structural vs sample-biased is a calibration
# question for shadow mode, NOT a reason to rewrite the scorer today.
# After 10-20 shadow sessions, recompute the live direction_pred quantiles
# and adjust these constants if the asymmetry shrinks or shifts.
# ──────────────────────────────────────────────────────────────────────
_LONG_PROBE     = +0.82   # q90
_LONG_FULL      = +1.14   # q95
_SHORT_PROBE    = -0.20   # q10
_SHORT_FULL     = -0.48   # q05

# Size multiplier calibration (from size_pred distribution):
#   q25 = 3.26   q50 = 3.56   q75 = 3.90
_SIZE_NORM      = 3.56    # divide by this to get multiplier


class EntrySignal(Enum):
    NONE         = "NONE"
    LONG_PROBE   = "LONG_PROBE"
    LONG_FULL    = "LONG_FULL"
    SHORT_PROBE  = "SHORT_PROBE"
    SHORT_FULL   = "SHORT_FULL"


@dataclass(frozen=True)
class ScorerInput:
    """Features required by the scorer. All derived from bars the bot already
    computes. Nothing new needs to be tracked."""
    er_20: float              # Kaufman Efficiency Ratio (20-bar)
    dist_vwap_atr: float      # (price - session_vwap) / ATR   — SIGNED
    dist_ema9_atr: float      # (price - ema9) / ATR           — SIGNED
    hurst_64: float           # Hurst exponent (64-bar R/S)
    body_ratio: float         # |close - open| / (high - low)

    # Optional debug fields — not used in scoring, just for logging.
    price: Optional[float] = None
    vwap: Optional[float] = None
    ema9: Optional[float] = None
    atr: Optional[float] = None


@dataclass(frozen=True)
class ScorerOutput:
    signal: EntrySignal
    direction_pred: float     # bps — signed forward-return prediction
    size_pred: float          # bps — magnitude prediction
    size_multiplier: float    # ratio to apply to base position size (~0.5 to 1.5x)
    reason: str               # human-readable trace


def score_entry(inp: ScorerInput) -> ScorerOutput:
    """Pure function. Returns entry signal + direction + size for one bar.

    Consumes raw feature values. Applies production weights from walk-forward
    refit. No I/O, no state. The caller is responsible for feature computation.
    """
    # ── Hard CHOP gate — no scoring below this ER floor ─────────────────
    if inp.er_20 < _ER_CHOP_MAX:
        return ScorerOutput(
            signal=EntrySignal.NONE,
            direction_pred=0.0,
            size_pred=0.0,
            size_multiplier=0.0,
            reason=f"CHOP gate: ER={inp.er_20:.3f} < {_ER_CHOP_MAX}",
        )

    # ── Direction prediction (bps) ──────────────────────────────────────
    gated_ema9 = inp.dist_ema9_atr if inp.er_20 >= _ER_GATE_EMA9 else 0.0
    vwap_x_er  = inp.dist_vwap_atr * inp.er_20

    direction_pred = (
        _W_INT_DIR
        + _W_VWAP       * inp.dist_vwap_atr
        + _W_EMA9_GATED * gated_ema9
        + _W_VWAP_X_ER  * vwap_x_er
    )

    # ── Size prediction (bps) ───────────────────────────────────────────
    size_pred = (
        _W_INT_SIZE
        + _W_ER    * inp.er_20
        + _W_HURST * inp.hurst_64
        + _W_BODY  * inp.body_ratio
    )
    size_multiplier = max(0.5, min(1.5, size_pred / _SIZE_NORM))

    # ── Signal classification ───────────────────────────────────────────
    # Order matters: check FULL tiers before PROBE tiers so a bar that exceeds
    # the FULL threshold is not mis-categorized as PROBE.
    if direction_pred >= _LONG_FULL:
        signal = EntrySignal.LONG_FULL
        tier_reason = f"LONG_FULL (pred>={_LONG_FULL:+.2f})"
    elif direction_pred >= _LONG_PROBE:
        signal = EntrySignal.LONG_PROBE
        tier_reason = f"LONG_PROBE (pred>={_LONG_PROBE:+.2f})"
    elif direction_pred <= _SHORT_FULL:
        signal = EntrySignal.SHORT_FULL
        tier_reason = f"SHORT_FULL (pred<={_SHORT_FULL:+.2f})"
    elif direction_pred <= _SHORT_PROBE:
        signal = EntrySignal.SHORT_PROBE
        tier_reason = f"SHORT_PROBE (pred<={_SHORT_PROBE:+.2f})"
    else:
        signal = EntrySignal.NONE
        tier_reason = (
            f"NONE (pred in neutral band "
            f"[{_SHORT_PROBE:+.2f}, {_LONG_PROBE:+.2f}])"
        )

    # Reason is prefixed with the chosen signal so a log reader immediately
    # sees the decision before the numeric justification.
    reason = (
        f"[{signal.value}] {tier_reason} | "
        f"dir_pred={direction_pred:+.3f}bps size={size_pred:.2f}bps "
        f"(mult={size_multiplier:.2f}) | "
        f"ER={inp.er_20:.2f} dVWAP={inp.dist_vwap_atr:+.2f}ATR "
        f"dEMA9={inp.dist_ema9_atr:+.2f}ATR hurst={inp.hurst_64:.2f} "
        f"body={inp.body_ratio:.2f}"
    )

    return ScorerOutput(
        signal=signal,
        direction_pred=direction_pred,
        size_pred=size_pred,
        size_multiplier=size_multiplier,
        reason=reason,
    )


# ─── Validation checklist before wiring into big_money_bot.py ─────────────
#
# 1. Feature parity: confirm dist_vwap_atr, dist_ema9_atr, hurst_64, and
#    body_ratio are computed identically in the live path vs the research
#    feature_matrix.py. Any divergence invalidates the weights.
#
# 2. Dry-run vs logs: replay 3-5 sessions of live logs through score_entry()
#    and verify the (signal, magnitude) output reconciles with what the
#    research pipeline produced for the same bars.
#
# 3. Threshold calibration: the quantile-based thresholds assume the live
#    direction_pred distribution matches research. Paper-trade for 10-20
#    sessions and recompute quantiles before enabling live size.
#
# 4. Conflict with existing paths: decide what to do with TREND_FOLLOW and
#    DIRECT_ENTRY if this scorer is added alongside them. Running three
#    entry paths in parallel is worse than the current two.
#
# 5. Exits are unchanged: this scorer only decides WHEN/SIZE of entry. All
#    exit logic (LIL, regime-change, trail, hard stop) stays as-is.
