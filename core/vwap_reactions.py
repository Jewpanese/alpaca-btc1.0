"""
VWAP Reaction Counter — third layer of the reversal-detection stack.

Tracks how price reacts to VWAP. A "rejection" is a tag-and-bounce pattern:
price comes within `tag_band_atr` of VWAP, then within `bounce_window` bars
moves ≥ `bounce_min_atr` away from VWAP. The DIRECTION of the bounce tells
us which side VWAP is acting as:
  - bounce UP off VWAP    → VWAP = support  → bullish reaction
  - bounce DOWN off VWAP  → VWAP = resistance → bearish reaction

We maintain a rolling count of confirmed rejections in each direction over
the last `count_window` bars. When ≥ `confirm_count` rejections have stacked
in the same direction, VWAP is a strong reference and the NEXT rejection in
that direction is high-conviction.

The per-bar output `score` is non-zero only on bars that confirm a rejection
AND the historical count meets the confirmation threshold. This makes the
score sparse and meaningful — when it fires, it's saying "VWAP just
rejected price for the Nth time, this is a real level."

Tradeable interpretation:
  - bullish_score >= 0.5 — VWAP is acting as support; LONG continuation
                           bias near VWAP, OR a fade-LONG if currently below
  - bearish_score >= 0.5 — VWAP is acting as resistance; SHORT continuation
                           bias near VWAP, OR a fade-SHORT if currently above

ALL READ-ONLY. Returns features. Does not make trade decisions.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Literal

logger = logging.getLogger(__name__)


@dataclass
class _PendingTag:
    """A bar where price was near VWAP — waiting to see if it bounces."""
    bar_index: int          # global bar counter at the tag
    tag_close: float        # close of the tag bar (for distance later)
    vwap_at_tag: float      # VWAP at the tag bar
    side_at_tag: int        # +1 if close > vwap (price above), -1 if below


@dataclass
class VWAPReactionSnapshot:
    """One bar's VWAP-reaction score."""
    bullish_score: float        # 0..1 — VWAP acting as support
    bearish_score: float        # 0..1 — VWAP acting as resistance
    bullish_count: int          # confirmed rejections in window (UP bounces)
    bearish_count: int          # confirmed rejections in window (DOWN bounces)
    confirmed_this_bar: Literal["NONE", "BULLISH", "BEARISH"]
    note: str = ""


class VWAPReactionTracker:
    """Maintains rolling state and emits a VWAPReactionSnapshot per bar.

    Caller supplies (close, vwap, atr) per closed bar. Tracker watches for
    tag-and-bounce patterns and counts confirmed rejections.
    """

    def __init__(self,
                 tag_band_atr: float = 0.15,
                 bounce_min_atr: float = 0.30,
                 bounce_window: int = 3,
                 count_window: int = 30,
                 confirm_count: int = 3,
                 score_floor_count: int = 2):
        """
        tag_band_atr: how close price must be to VWAP (in ATR units) to
                      count as "tagging" the level.
        bounce_min_atr: how far away the price must move post-tag to confirm
                        a rejection.
        bounce_window: bars after the tag in which the bounce must complete.
        count_window: rolling-window length for counting confirmed rejections.
        confirm_count: rejections needed in window for the score to saturate
                       at 1.0 (linear ramp from score_floor_count up).
        score_floor_count: rejections needed before any score is emitted.
        """
        self.tag_band_atr = tag_band_atr
        self.bounce_min_atr = bounce_min_atr
        self.bounce_window = bounce_window
        self.count_window = count_window
        self.confirm_count = confirm_count
        self.score_floor_count = score_floor_count

        # Pending tags awaiting bounce confirmation
        self._pending: list[_PendingTag] = []
        # Confirmed rejections — stored as (bar_index, direction) tuples
        # direction: +1 = bullish (UP bounce), -1 = bearish (DOWN bounce)
        self._confirmed: Deque[tuple[int, int]] = deque()
        self._bar_index: int = 0

    # ─── per-bar update ─────────────────────────────────────────────

    def on_bar(self,
               close: float,
               vwap: Optional[float],
               atr: float) -> VWAPReactionSnapshot:
        self._bar_index += 1
        confirmed_this_bar: Literal["NONE", "BULLISH", "BEARISH"] = "NONE"

        # Skip when VWAP / ATR not available — can't compute distance
        if vwap is None or atr <= 0 or vwap <= 0:
            self._gc_old_confirmed()
            bullish_count = sum(1 for _, d in self._confirmed if d > 0)
            bearish_count = sum(1 for _, d in self._confirmed if d < 0)
            return VWAPReactionSnapshot(
                bullish_score=0.0, bearish_score=0.0,
                bullish_count=bullish_count, bearish_count=bearish_count,
                confirmed_this_bar="NONE",
                note="missing vwap/atr",
            )

        dist_atr = abs(close - vwap) / atr

        # ─── 1. Check if any pending tag confirms its bounce on THIS bar ──
        # A pending tag at index P confirms if, by current bar, price has
        # moved >= bounce_min_atr * atr AWAY from vwap_at_tag, in a direction
        # consistent with side_at_tag (price above VWAP at tag → bounced UP
        # away from VWAP = bullish; price below VWAP at tag → bounced DOWN
        # away from VWAP = bearish).
        still_pending: list[_PendingTag] = []
        for p in self._pending:
            age = self._bar_index - p.bar_index
            if age > self.bounce_window:
                # Expired without confirming — drop it
                continue
            # Distance moved relative to vwap-at-tag, in current ATR units
            move_from_vwap = (close - p.vwap_at_tag) / max(atr, 1e-9)
            # Price was ABOVE vwap at tag → bounce UP = move_from_vwap >= +threshold
            # Price was BELOW vwap at tag → bounce DOWN = move_from_vwap <= -threshold
            if p.side_at_tag > 0 and move_from_vwap >= self.bounce_min_atr:
                # Bullish rejection (held above VWAP — VWAP is support)
                self._confirmed.append((self._bar_index, +1))
                confirmed_this_bar = "BULLISH"
                continue
            if p.side_at_tag < 0 and move_from_vwap <= -self.bounce_min_atr:
                # Bearish rejection (held below VWAP — VWAP is resistance)
                self._confirmed.append((self._bar_index, -1))
                confirmed_this_bar = "BEARISH"
                continue
            still_pending.append(p)
        self._pending = still_pending

        # ─── 2. Add a new pending tag if THIS bar tags VWAP ──────────────
        if dist_atr <= self.tag_band_atr:
            side = +1 if close >= vwap else -1
            self._pending.append(_PendingTag(
                bar_index=self._bar_index,
                tag_close=close,
                vwap_at_tag=vwap,
                side_at_tag=side,
            ))

        # ─── 3. GC old confirmed rejections outside the window ───────────
        self._gc_old_confirmed()

        # ─── 4. Compute scores ───────────────────────────────────────────
        bullish_count = sum(1 for _, d in self._confirmed if d > 0)
        bearish_count = sum(1 for _, d in self._confirmed if d < 0)
        bullish_score = self._count_to_score(bullish_count)
        bearish_score = self._count_to_score(bearish_count)

        # Slight bonus when a confirmation just happened — captures the
        # "this bar IS the rejection" moment vs. just "VWAP has been a level."
        if confirmed_this_bar == "BULLISH":
            bullish_score = min(1.0, bullish_score + 0.10)
        elif confirmed_this_bar == "BEARISH":
            bearish_score = min(1.0, bearish_score + 0.10)

        note = ""
        if bullish_count or bearish_count:
            note = f"reject_count: bull={bullish_count} bear={bearish_count}"
            if confirmed_this_bar != "NONE":
                note = f"{confirmed_this_bar}_CONFIRMED | " + note

        return VWAPReactionSnapshot(
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            confirmed_this_bar=confirmed_this_bar,
            note=note,
        )

    # ─── internals ──────────────────────────────────────────────────

    def _gc_old_confirmed(self) -> None:
        """Drop confirmed rejections outside count_window."""
        cutoff = self._bar_index - self.count_window
        while self._confirmed and self._confirmed[0][0] < cutoff:
            self._confirmed.popleft()

    def _count_to_score(self, count: int) -> float:
        if count < self.score_floor_count:
            return 0.0
        if count >= self.confirm_count:
            return 1.0
        # Linear ramp from floor → confirm
        span = max(1, self.confirm_count - self.score_floor_count)
        return 0.5 + 0.5 * (count - self.score_floor_count) / span
