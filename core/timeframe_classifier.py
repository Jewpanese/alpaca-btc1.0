"""
Multi-Timeframe Pullback vs Reversal Classifier

Reads two timeframes — the bot's primary 3-min bars (live, fed by caller) and
an aggregated 15-min view (built internally from the same 3-min stream). The
15-min timeframe is the **trend anchor**; the 3-min is the action-bar.

Outputs one of three labels per 3-min bar:
  - PULLBACK: anchor trend intact, action-bar retracing → ADD opportunity
  - REVERSAL: anchor trend flipped or anchor structure broken → CLOSE opportunity
  - NEUTRAL : not enough conviction either way (or no active position context)

V1 design: SIGNAL ONLY. The classifier emits the label and a one-line reason
to the log on every closed bar; the bot does not act on it. After 1-2 weeks of
log data we compare the labels to chart reality. If they agree, we wire the
signal into trade management (T2/T3 add on PULLBACK, force-close on REVERSAL).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Literal

logger = logging.getLogger(__name__)

Label = Literal["PULLBACK", "REVERSAL", "NEUTRAL"]


@dataclass
class TFClassifierConfig:
    """Tunables for the classifier. Held conservative for V1 — calibrated to
    fire NEUTRAL when in doubt rather than over-flag."""
    # 5 × 3-min = 15-min anchor.
    aggregation_factor: int = 5
    # 15-min EMA lookback — matches a typical 4hr trader's "anchor MA".
    anchor_ema_period: int = 20
    # Slope is computed over this many 15-min bars.
    anchor_slope_window: int = 5
    # Slope magnitude (bps/bar) for the anchor to be considered "directional".
    # Below this → neutral on the anchor. 0.50 bps on /MES at 7200 ≈ 0.36
    # pts/15-min bar — gentle but real drift.
    anchor_slope_min_bps: float = 0.50
    # ER on the 15-min — if anchor ER drops below this, anchor is decaying.
    anchor_er_min: float = 0.20
    # Bars to confirm an anchor flip (hysteresis) before we call REVERSAL on
    # anchor direction alone.
    anchor_flip_persist: int = 2


@dataclass
class TFState:
    # Aggregation buffer for the current under-construction 15-min bar
    agg_count: int = 0
    agg_open: Optional[float] = None
    agg_high: Optional[float] = None
    agg_low: Optional[float] = None
    agg_close: Optional[float] = None

    # Closed 15-min bars
    anchor_closes: deque = field(default_factory=lambda: deque(maxlen=64))
    anchor_highs: deque = field(default_factory=lambda: deque(maxlen=64))
    anchor_lows: deque = field(default_factory=lambda: deque(maxlen=64))

    # EMA state for the anchor close
    anchor_ema: Optional[float] = None
    # Last anchor direction call (LONG/SHORT/FLAT) and how long it has held
    last_anchor_dir: str = "FLAT"
    anchor_dir_bars: int = 0
    # Tentative flip candidate + counter (for hysteresis)
    anchor_flip_pending: str = "FLAT"
    anchor_flip_count: int = 0


@dataclass
class TFSignal:
    label: Label
    anchor_dir: str          # LONG / SHORT / FLAT
    anchor_slope_bps: float
    anchor_er: float
    anchor_ema: Optional[float]
    closed_anchor_bar: bool  # True only on bars where the 15-min just closed
    reason: str


class TFClassifier:
    """One instance per bot. Call `on_3min_bar(close, high, low,
    position_dir)` once per closed 3-min bar. Returns a TFSignal."""

    def __init__(self, config: Optional[TFClassifierConfig] = None):
        self.config = config or TFClassifierConfig()
        self.state = TFState()

    # ── Public API ────────────────────────────────────────────────────────

    def on_3min_bar(self,
                    close: float,
                    high: float,
                    low: float,
                    position_dir: Optional[str] = None) -> TFSignal:
        """Process one closed 3-min bar.

        position_dir: "LONG" / "SHORT" / None. Used only to flavor the label
        when the bot is in a position (PULLBACK is meaningful in-position;
        REVERSAL is meaningful in-position).
        """
        cfg = self.config
        st = self.state

        # 1. Update 15-min aggregation
        if st.agg_count == 0:
            st.agg_open = close  # use close as proxy if open isn't passed
            st.agg_high = high
            st.agg_low = low
        else:
            st.agg_high = max(st.agg_high or high, high)
            st.agg_low = min(st.agg_low or low, low)
        st.agg_close = close
        st.agg_count += 1

        anchor_just_closed = (st.agg_count >= cfg.aggregation_factor)
        if anchor_just_closed:
            # Commit the 15-min bar
            st.anchor_closes.append(st.agg_close)
            st.anchor_highs.append(st.agg_high or close)
            st.anchor_lows.append(st.agg_low or close)
            # Reset for next aggregation window
            st.agg_count = 0
            st.agg_open = None
            st.agg_high = None
            st.agg_low = None
            st.agg_close = None
            # Update anchor EMA on the new 15-min close
            new_close = st.anchor_closes[-1]
            alpha = 2.0 / (cfg.anchor_ema_period + 1)
            if st.anchor_ema is None:
                # Wait until we have enough closes to seed
                if len(st.anchor_closes) >= cfg.anchor_ema_period:
                    seed = sum(list(st.anchor_closes)[-cfg.anchor_ema_period:]) \
                           / cfg.anchor_ema_period
                    st.anchor_ema = seed
            else:
                st.anchor_ema = alpha * new_close + (1 - alpha) * st.anchor_ema

        # 2. Compute current anchor metrics (only meaningful once warmed)
        anchor_dir = "FLAT"
        anchor_slope_bps = 0.0
        anchor_er = 0.0
        if (len(st.anchor_closes) >= cfg.anchor_slope_window + 1
                and st.anchor_ema is not None):
            closes = list(st.anchor_closes)
            recent = closes[-cfg.anchor_slope_window - 1:]
            # Slope in bps/bar over the slope window
            n = len(recent) - 1
            price_change = recent[-1] - recent[0]
            anchor_slope_bps = (price_change / recent[0]) * 10000.0 / n if recent[0] else 0.0
            # ER on the anchor: |net move| / sum |bar moves|
            net = abs(recent[-1] - recent[0])
            path = sum(abs(recent[i + 1] - recent[i]) for i in range(n))
            anchor_er = (net / path) if path > 0 else 0.0

            # Direction call from slope + EMA position
            current = closes[-1]
            ema_break_long = current < st.anchor_ema
            ema_break_short = current > st.anchor_ema
            if (anchor_slope_bps >= cfg.anchor_slope_min_bps
                    and not ema_break_long):
                anchor_dir = "LONG"
            elif (anchor_slope_bps <= -cfg.anchor_slope_min_bps
                    and not ema_break_short):
                anchor_dir = "SHORT"
            else:
                anchor_dir = "FLAT"

        # 3. Hysteresis — require N anchor bars in the new direction before
        #    declaring a flip
        if anchor_just_closed:
            if anchor_dir != st.last_anchor_dir:
                if anchor_dir == st.anchor_flip_pending:
                    st.anchor_flip_count += 1
                else:
                    st.anchor_flip_pending = anchor_dir
                    st.anchor_flip_count = 1
                if st.anchor_flip_count >= cfg.anchor_flip_persist:
                    st.last_anchor_dir = anchor_dir
                    st.anchor_dir_bars = 0
                    st.anchor_flip_count = 0
                    st.anchor_flip_pending = "FLAT"
            else:
                st.anchor_flip_count = 0
                st.anchor_flip_pending = "FLAT"
                st.anchor_dir_bars += 1

        # 4. Label
        label, reason = self._classify(
            anchor_dir=st.last_anchor_dir,
            anchor_slope_bps=anchor_slope_bps,
            anchor_er=anchor_er,
            position_dir=position_dir,
        )

        return TFSignal(
            label=label,
            anchor_dir=st.last_anchor_dir,
            anchor_slope_bps=anchor_slope_bps,
            anchor_er=anchor_er,
            anchor_ema=st.anchor_ema,
            closed_anchor_bar=anchor_just_closed,
            reason=reason,
        )

    def _classify(self,
                  anchor_dir: str,
                  anchor_slope_bps: float,
                  anchor_er: float,
                  position_dir: Optional[str]) -> tuple[Label, str]:
        cfg = self.config
        # Not warmed up yet
        if anchor_dir == "FLAT" and abs(anchor_slope_bps) < cfg.anchor_slope_min_bps:
            return ("NEUTRAL", "anchor flat / warming up")

        # Anchor weak (low ER) — neutral regardless
        if anchor_er < cfg.anchor_er_min:
            return ("NEUTRAL",
                    f"anchor ER={anchor_er:.2f} < {cfg.anchor_er_min} — anchor decaying")

        # No position context — just describe anchor state
        if position_dir is None:
            return ("NEUTRAL", f"no position — anchor={anchor_dir}")

        # In position: align action-bar pullback or anchor flip
        if position_dir == "LONG":
            if anchor_dir == "LONG":
                return ("PULLBACK",
                        f"anchor LONG intact (slope={anchor_slope_bps:+.2f}bps, "
                        f"ER={anchor_er:.2f}) — retrace = ADD opportunity")
            elif anchor_dir == "SHORT":
                return ("REVERSAL",
                        f"anchor flipped to SHORT (slope={anchor_slope_bps:+.2f}bps) "
                        "— EXIT")
            else:
                return ("NEUTRAL", "anchor FLAT vs LONG position")

        if position_dir == "SHORT":
            if anchor_dir == "SHORT":
                return ("PULLBACK",
                        f"anchor SHORT intact (slope={anchor_slope_bps:+.2f}bps, "
                        f"ER={anchor_er:.2f}) — retrace = ADD opportunity")
            elif anchor_dir == "LONG":
                return ("REVERSAL",
                        f"anchor flipped to LONG (slope={anchor_slope_bps:+.2f}bps) "
                        "— EXIT")
            else:
                return ("NEUTRAL", "anchor FLAT vs SHORT position")

        return ("NEUTRAL", "fall-through")
