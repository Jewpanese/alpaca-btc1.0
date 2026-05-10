"""
Order Flow Estimator — Tom Williams cumulative delta + divergence detection.

Without bid/ask trade-side data, we estimate per-bar buy/sell pressure from
OHLCV using the Twiggs Money Flow formula:

    money_flow = ((close - low) - (high - close)) / (high - low)
    bar_delta  = volume × money_flow

money_flow ranges -1..+1:
    +1 = bar closed at the high (max buying pressure)
     0 = bar closed at midpoint
    -1 = bar closed at the low (max selling pressure)

Cumulative delta = rolling sum of bar_delta. Tracked over a window (default 50
bars = 150 min on 3-min) so the integrator doesn't grow unbounded across a
session.

The high-value signal is **divergence**: price makes a new high while
cumulative delta makes a LOWER high (bearish exhaustion — buyers losing
conviction even as price extends). Reverse for bullish exhaustion.

Outputs per bar:
    bar_delta             — this bar's signed pressure estimate
    cum_delta             — rolling-window sum
    delta_z               — z-score of bar_delta vs recent window
    bearish_divergence    — 0..1 score; high when price↑ + cum_delta↓
    bullish_divergence    — 0..1 score; high when price↓ + cum_delta↑
    score                 — max(bearish, bullish), the layer's reversal vote

ALL READ-ONLY. This module computes features. It does not make trade decisions.
"""
from __future__ import annotations
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowSnapshot:
    """One bar's order-flow estimate."""
    bar_delta: float
    cum_delta: float
    delta_z: float
    bearish_divergence: float    # 0..1
    bullish_divergence: float    # 0..1
    score: float                 # max of the two divergences
    note: str = ""               # diagnostic string for logging


class OrderFlowEstimator:
    """Maintains rolling state and produces per-bar order-flow features.

    Caller supplies OHLCV per closed bar. Estimator returns an
    OrderFlowSnapshot. State persists across bars for cumulative delta and
    divergence detection.
    """

    def __init__(self,
                 cum_window: int = 50,
                 z_window: int = 30,
                 pivot_lookback: int = 5,
                 pivot_window: int = 30):
        """
        cum_window: rolling-window length for cumulative delta (bars).
        z_window: bars of history used for delta z-score.
        pivot_lookback: bars on each side that must be lower (or higher)
                        for a price pivot to confirm. 5 means a pivot needs
                        2 bars before + 2 bars after lower than the pivot
                        bar (using a 5-wide centered window).
        pivot_window: how far back (in bars) we look for a prior pivot to
                      compare against for divergence detection.
        """
        self.cum_window = cum_window
        self.z_window = z_window
        self.pivot_lookback = pivot_lookback
        self.pivot_window = pivot_window

        # Rolling buffers
        self._deltas: Deque[float] = deque(maxlen=cum_window)
        self._closes: Deque[float] = deque(maxlen=pivot_window)
        self._highs:  Deque[float] = deque(maxlen=pivot_window)
        self._lows:   Deque[float] = deque(maxlen=pivot_window)
        self._cum_at_bar: Deque[float] = deque(maxlen=pivot_window)
        self._z_buf:  Deque[float] = deque(maxlen=z_window)

    # ─── per-bar update ─────────────────────────────────────────────

    def on_bar(self,
               open_: float,
               high: float,
               low: float,
               close: float,
               volume: float) -> OrderFlowSnapshot:
        # 1. bar_delta from Twiggs money flow
        if high > low and volume > 0:
            mf = ((close - low) - (high - close)) / (high - low)
            bar_delta = float(volume) * mf
        else:
            bar_delta = 0.0

        self._deltas.append(bar_delta)
        self._closes.append(float(close))
        self._highs.append(float(high))
        self._lows.append(float(low))

        # 2. cumulative delta over rolling window
        cum_delta = float(sum(self._deltas))
        self._cum_at_bar.append(cum_delta)

        # 3. z-score of bar delta vs recent window
        self._z_buf.append(bar_delta)
        if len(self._z_buf) >= 5:
            mean = sum(self._z_buf) / len(self._z_buf)
            var = sum((d - mean) ** 2 for d in self._z_buf) / max(1, len(self._z_buf) - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            delta_z = (bar_delta - mean) / std if std > 0 else 0.0
        else:
            delta_z = 0.0

        # 4. divergence detection (needs enough history for pivots)
        bearish, bullish = 0.0, 0.0
        note = ""
        if len(self._closes) >= self.pivot_window:
            bearish, bullish, note = self._detect_divergence()

        score = max(bearish, bullish)

        return OrderFlowSnapshot(
            bar_delta=bar_delta,
            cum_delta=cum_delta,
            delta_z=delta_z,
            bearish_divergence=bearish,
            bullish_divergence=bullish,
            score=score,
            note=note,
        )

    # ─── divergence detection ───────────────────────────────────────

    def _detect_divergence(self) -> tuple[float, float, str]:
        """Compare the most-recent confirmed pivot in price vs cum_delta.

        Bearish divergence: current bar makes a higher high than the
            previous price-pivot-high, but cum_delta is LOWER than the
            cum_delta value at that prior pivot.

        Bullish divergence: current bar makes a lower low than the previous
            price-pivot-low, but cum_delta is HIGHER than at that prior pivot.

        Score scales by how confirmed the divergence is — bigger price gap
        and bigger delta gap = stronger signal.
        """
        highs = list(self._highs)
        lows = list(self._lows)
        cum = list(self._cum_at_bar)
        n = len(highs)

        # Find the most recent confirmed price pivot HIGH
        # A pivot at index i needs `pivot_lookback` lower bars on each side.
        # We can only confirm pivots that are NOT in the last pivot_lookback bars.
        prior_pivot_hi_idx = None
        for i in range(n - self.pivot_lookback - 1, self.pivot_lookback, -1):
            window = highs[i - self.pivot_lookback: i + self.pivot_lookback + 1]
            if highs[i] == max(window):
                prior_pivot_hi_idx = i
                break

        prior_pivot_lo_idx = None
        for i in range(n - self.pivot_lookback - 1, self.pivot_lookback, -1):
            window = lows[i - self.pivot_lookback: i + self.pivot_lookback + 1]
            if lows[i] == min(window):
                prior_pivot_lo_idx = i
                break

        bearish = 0.0
        bullish = 0.0
        note = ""

        # Bearish: current high > prior pivot high, but cum_delta < cum_delta at pivot
        if prior_pivot_hi_idx is not None:
            cur_high = highs[-1]
            piv_high = highs[prior_pivot_hi_idx]
            cur_cum = cum[-1]
            piv_cum = cum[prior_pivot_hi_idx]
            if cur_high > piv_high and cur_cum < piv_cum:
                # Score scales with how far apart price and delta are
                price_pct = (cur_high - piv_high) / max(piv_high, 1.0) * 10000.0  # bps
                delta_gap_pct = (piv_cum - cur_cum) / max(abs(piv_cum), 1.0)
                # Saturate around realistic ranges
                bearish = min(1.0, 0.4 + 0.3 * min(1.0, price_pct / 5.0)
                                       + 0.3 * min(1.0, delta_gap_pct))
                note = (f"BEARISH_DIV: price+{price_pct:.1f}bps cum_delta-"
                        f"{delta_gap_pct:.1%} (idx{prior_pivot_hi_idx})")

        # Bullish: current low < prior pivot low, but cum_delta > cum_delta at pivot
        if prior_pivot_lo_idx is not None:
            cur_low = lows[-1]
            piv_low = lows[prior_pivot_lo_idx]
            cur_cum = cum[-1]
            piv_cum = cum[prior_pivot_lo_idx]
            if cur_low < piv_low and cur_cum > piv_cum:
                price_pct = (piv_low - cur_low) / max(piv_low, 1.0) * 10000.0
                delta_gap_pct = (cur_cum - piv_cum) / max(abs(piv_cum), 1.0)
                bullish = min(1.0, 0.4 + 0.3 * min(1.0, price_pct / 5.0)
                                       + 0.3 * min(1.0, delta_gap_pct))
                if note:
                    note += f" | BULLISH_DIV: price-{price_pct:.1f}bps cum_delta+{delta_gap_pct:.1%}"
                else:
                    note = f"BULLISH_DIV: price-{price_pct:.1f}bps cum_delta+{delta_gap_pct:.1%}"

        return bearish, bullish, note
