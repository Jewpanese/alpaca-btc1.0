"""
Reversal Feature Scorers — bar-shape signals that indicate exhaustion.

This module produces the **volume-signature** layer of the reversal-detection
plan (project_reversal_detection_plan). The single public function
`score_volume_signature` reads one closed bar plus context (vol_z, ATR, recent
direction) and returns a 0..1 score reflecting how strongly the bar's
*shape* signals exhaustion.

The bar-shape signals we score:

  1. Pin bar / wick rejection
        After a directional run, a bar with a long opposite-side wick
        signals price tried to extend but got rejected. Specifically:
          - After RALLY: close in the lower 30% of bar range = bearish reject
          - After SELLOFF: close in the upper 30% of bar range = bullish reject

  2. Wide-range, small-body candle
        Range >= 1.5×ATR with body / range <= 0.3 — the bar attempted both
        directions and the close ended up near the open. Indecision at an
        extreme. Often the inflection bar.

  3. Volume z-score gate
        Both signals only count when vol_z >= 1.0 (real volume behind the
        rejection). Without volume confirmation, wicks are noise.

The score is a weighted maximum over the three components, with the volume
gate as a multiplier (low vol_z attenuates the score sharply).

ALL READ-ONLY. Returns features. Does not make trade decisions.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Literal

logger = logging.getLogger(__name__)

Direction = Literal["LONG", "SHORT", "NONE"]


@dataclass
class VolumeSignatureScore:
    """Output of the volume-signature scorer for one bar."""
    score: float                 # 0..1 — the layer's reversal vote
    direction: Direction         # which way the reversal (LONG = bullish reversal off lows)
    body_ratio: float            # 0..1
    range_atr: float             # range / ATR
    upper_wick_ratio: float      # 0..1, fraction of range above body
    lower_wick_ratio: float      # 0..1, fraction of range below body
    vol_z: Optional[float]       # passthrough for diagnostics
    note: str = ""


def score_volume_signature(
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
    vol_z: Optional[float],
    recent_direction: Direction = "NONE",
    *,
    pin_wick_min: float = 0.55,        # opposite-side wick must be >= this fraction of range
    pin_body_max: float = 0.40,        # body must be <= this fraction of range
    wide_range_atr_min: float = 1.5,   # range / ATR threshold for wide-range
    wide_body_max: float = 0.30,       # body / range threshold for indecision
    vol_z_floor: float = 1.0,          # below this, attenuate score
) -> VolumeSignatureScore:
    """Score one closed bar for reversal exhaustion shape.

    `recent_direction` is the direction of the move INTO this bar — used to
    decide whether a wick is a rejection (against the move) or a continuation
    (with the move). Pass "LONG" if the move into this bar was up, "SHORT"
    if down. "NONE" disables direction-aware scoring (only the indecision
    component contributes).
    """
    # Basic geometry
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    body_ratio = body / rng

    upper_wick = (high - max(open_, close)) / rng
    lower_wick = (min(open_, close) - low) / rng

    range_atr = rng / atr if atr > 0 else 0.0

    # ─── Component 1: pin-bar rejection (direction-aware) ─────────
    pin_score = 0.0
    pin_dir: Direction = "NONE"
    if body_ratio <= pin_body_max:
        # Top wick rejected = bearish reversal (signals SHORT entry)
        if recent_direction == "LONG" and upper_wick >= pin_wick_min:
            # Stronger when wick is bigger and body is smaller
            pin_score = min(1.0,
                            (upper_wick - pin_wick_min) / (1.0 - pin_wick_min) * 0.6
                            + (pin_body_max - body_ratio) / pin_body_max * 0.4)
            pin_dir = "SHORT"
        # Bottom wick rejected = bullish reversal (signals LONG entry)
        elif recent_direction == "SHORT" and lower_wick >= pin_wick_min:
            pin_score = min(1.0,
                            (lower_wick - pin_wick_min) / (1.0 - pin_wick_min) * 0.6
                            + (pin_body_max - body_ratio) / pin_body_max * 0.4)
            pin_dir = "LONG"

    # ─── Component 2: wide-range indecision ───────────────────────
    indecision_score = 0.0
    if range_atr >= wide_range_atr_min and body_ratio <= wide_body_max:
        # Stronger when range is wider and body is tighter
        indecision_score = min(1.0,
                               min(1.0, (range_atr - wide_range_atr_min) / 1.0) * 0.5
                               + (wide_body_max - body_ratio) / wide_body_max * 0.5)
        # Indecision direction inferred from which wick is bigger AND
        # opposite to recent direction
        if recent_direction == "LONG" and upper_wick > lower_wick * 1.2:
            pin_dir = pin_dir if pin_dir != "NONE" else "SHORT"
        elif recent_direction == "SHORT" and lower_wick > upper_wick * 1.2:
            pin_dir = pin_dir if pin_dir != "NONE" else "LONG"

    # ─── Combine + volume gate ────────────────────────────────────
    raw_score = max(pin_score, indecision_score)

    # Volume attenuator: 1.0 when vol_z >= floor + 1, scales down to 0.3 at floor,
    # 0 below floor. If vol_z is None (no baseline), don't gate.
    if vol_z is None:
        vol_mult = 0.7   # no info — be conservative
    elif vol_z < vol_z_floor:
        vol_mult = 0.3 * max(0.0, vol_z / vol_z_floor)
    else:
        vol_mult = min(1.0, 0.5 + 0.5 * (vol_z - vol_z_floor) / 1.0)

    final = raw_score * vol_mult

    # Direction defaults to whichever component fired
    direction = pin_dir if pin_dir != "NONE" else "NONE"

    note_parts = []
    if pin_score > 0:
        note_parts.append(f"pin={pin_score:.2f}")
    if indecision_score > 0:
        note_parts.append(f"indecision={indecision_score:.2f}")
    if vol_mult < 1.0:
        note_parts.append(f"vol_mult={vol_mult:.2f}")
    note = " ".join(note_parts) if note_parts else "no_signal"

    return VolumeSignatureScore(
        score=final,
        direction=direction,
        body_ratio=body_ratio,
        range_atr=range_atr,
        upper_wick_ratio=upper_wick,
        lower_wick_ratio=lower_wick,
        vol_z=vol_z,
        note=note,
    )
