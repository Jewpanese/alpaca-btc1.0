"""
Reversal Score Combiner — Phase C of the reversal-detection plan.

Combines the four orthogonal detection layers into one directional reversal
score. Each layer's contribution is weighted; weights default to:

    order_flow      0.35
    vol_signature   0.20
    vwap_reactions  0.20
    ml_reversion    0.25

The combiner produces SEPARATE long_score and short_score values (not just
a magnitude), because reversal direction matters. Each input layer
contributes to one side or both:

    Layer            → SHORT side                LONG side
    --------------------------------------------------------------------
    order_flow       → bearish_divergence        bullish_divergence
    vol_signature    → score (if dir=SHORT)      score (if dir=LONG)
    vwap_reactions   → bearish_score             bullish_score
    ml_reversion     → ml_short_prob             ml_long_prob

Where ml_*_prob are optional. If both are None (no ML wired), the ML weight
is redistributed proportionally to the other three layers — the score still
sums to a 0..1 range.

The combiner is PURE — no internal state. Called per bar with the layer
outputs, returns a snapshot. Logging is the caller's job.

ALL READ-ONLY. Returns a score. Does not make trade decisions.

Tradeable interpretation (post-validation):
    reversal_score >= 0.65 → high-conviction reversal signal
    direction = SHORT      → fade the rally / short the high
    direction = LONG       → fade the dip / long the low
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.order_flow import OrderFlowSnapshot
from core.reversal_features import VolumeSignatureScore
from core.vwap_reactions import VWAPReactionSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ReversalLayerWeights:
    """Per-layer weights. Should sum to 1.0 for a 0..1 output range.
    Defaults derived from the institutional design rationale in
    `memory/project_reversal_detection_plan.md`. Re-derive from per-layer
    IC on historical reversals once shadow data accumulates (Phase E)."""
    order_flow: float     = 0.35
    vol_signature: float  = 0.20
    vwap_reactions: float = 0.20
    ml_reversion: float   = 0.25


@dataclass
class ReversalScoreSnapshot:
    """Output of the combiner for one bar."""
    score: float                 # max(long_score, short_score) — overall conviction
    direction: str               # "LONG" / "SHORT" / "NONE"
    long_score: float            # composite vote for bullish reversal
    short_score: float           # composite vote for bearish reversal
    components: dict[str, dict] = field(default_factory=dict)  # per-layer breakdown
    ml_used: bool = False        # was ML included in this score?
    note: str = ""


def combine_reversal_score(
    order_flow:    Optional[OrderFlowSnapshot],
    vol_signature: Optional[VolumeSignatureScore],
    vwap:          Optional[VWAPReactionSnapshot],
    ml_long_prob:  Optional[float] = None,
    ml_short_prob: Optional[float] = None,
    weights:       Optional[ReversalLayerWeights] = None,
) -> ReversalScoreSnapshot:
    """Compute the composite reversal score for one bar.

    Returns a ReversalScoreSnapshot. All inputs are optional — missing
    layers contribute 0 and their weight is redistributed to whatever
    layers ARE available, preserving the 0..1 output range.
    """
    w = weights or ReversalLayerWeights()

    # Per-side raw contributions and per-layer-presence flags.
    contribs_long: list[tuple[str, float, float]] = []   # (layer, weight, value)
    contribs_short: list[tuple[str, float, float]] = []

    # Layer 1 — order flow divergence
    if order_flow is not None:
        contribs_long.append(("order_flow", w.order_flow, order_flow.bullish_divergence))
        contribs_short.append(("order_flow", w.order_flow, order_flow.bearish_divergence))

    # Layer 2 — volume signature (direction-tagged)
    if vol_signature is not None and vol_signature.score > 0:
        if vol_signature.direction == "LONG":
            contribs_long.append(("vol_signature", w.vol_signature, vol_signature.score))
            contribs_short.append(("vol_signature", w.vol_signature, 0.0))
        elif vol_signature.direction == "SHORT":
            contribs_long.append(("vol_signature", w.vol_signature, 0.0))
            contribs_short.append(("vol_signature", w.vol_signature, vol_signature.score))
        else:
            # Neutral / indecision — split half-and-half so it counts but
            # doesn't pick a side
            half = vol_signature.score * 0.5
            contribs_long.append(("vol_signature", w.vol_signature, half))
            contribs_short.append(("vol_signature", w.vol_signature, half))
    else:
        # Still register the layer with 0 contribution so weight math works
        if vol_signature is not None:
            contribs_long.append(("vol_signature", w.vol_signature, 0.0))
            contribs_short.append(("vol_signature", w.vol_signature, 0.0))

    # Layer 3 — VWAP reactions
    if vwap is not None:
        contribs_long.append(("vwap_reactions", w.vwap_reactions, vwap.bullish_score))
        contribs_short.append(("vwap_reactions", w.vwap_reactions, vwap.bearish_score))

    # Layer 4 — ML reversion (optional)
    ml_used = (ml_long_prob is not None) or (ml_short_prob is not None)
    if ml_used:
        contribs_long.append(("ml_reversion", w.ml_reversion, float(ml_long_prob or 0.0)))
        contribs_short.append(("ml_reversion", w.ml_reversion, float(ml_short_prob or 0.0)))

    # Renormalize weights — if some layers are missing, surviving layers'
    # weights scale up so total stays 1.0 (assuming all original weights
    # sum to ≤ 1.0). This keeps the score in 0..1 regardless of which
    # layers reported.
    long_score = _weighted_sum(contribs_long)
    short_score = _weighted_sum(contribs_short)

    # Direction picker — favour the larger side, NONE if both small or tied.
    if long_score < 0.05 and short_score < 0.05:
        direction = "NONE"
    elif long_score > short_score * 1.10:   # 10% margin to avoid ties
        direction = "LONG"
    elif short_score > long_score * 1.10:
        direction = "SHORT"
    else:
        direction = "NONE"   # close call — be neutral

    score = max(long_score, short_score)

    components = _compose_components(contribs_long, contribs_short)
    note = _summary_note(direction, score, ml_used)

    return ReversalScoreSnapshot(
        score=score,
        direction=direction,
        long_score=long_score,
        short_score=short_score,
        components=components,
        ml_used=ml_used,
        note=note,
    )


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _weighted_sum(contribs: list[tuple[str, float, float]]) -> float:
    """Sum of (weight × value) with weights renormalized to total 1.0
    over the layers actually present. Empty list → 0."""
    if not contribs:
        return 0.0
    total_weight = sum(w for _, w, _ in contribs)
    if total_weight <= 0:
        return 0.0
    return sum((w / total_weight) * v for _, w, v in contribs)


def _compose_components(longs: list[tuple[str, float, float]],
                        shorts: list[tuple[str, float, float]]) -> dict:
    """Build a per-layer breakdown dict for diagnostic / audit logging."""
    out: dict[str, dict] = {}
    for name, weight, value in longs:
        out.setdefault(name, {"weight": weight})["long"] = round(value, 3)
    for name, weight, value in shorts:
        out.setdefault(name, {"weight": weight})["short"] = round(value, 3)
    return out


def _summary_note(direction: str, score: float, ml_used: bool) -> str:
    qual = "weak"
    if score >= 0.65:
        qual = "STRONG"
    elif score >= 0.45:
        qual = "moderate"
    elif score >= 0.25:
        qual = "mild"
    return f"{qual} {direction} (ml={'on' if ml_used else 'off'})"
