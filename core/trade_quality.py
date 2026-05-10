"""
Trade Quality Filter — quant-style pre-entry gate.

Replaces the pile of softened [ALLOWING ENTRY ANYWAY] checks with three
mathematically grounded conditions evaluated once per signal. All pre-computed
values from MarketState — zero additional indicator overhead.

Three conditions:

  1. Efficiency Ratio (Kaufman)
     ER = |net price displacement| / sum(|bar-to-bar changes|)
     Measures how directionally efficient recent price movement has been.
     ER near 1.0 = pure trend, every bar moves the same direction.
     ER near 0.0 = pure chop, price oscillates with no net progress.
     Threshold: > 0.25 to enter (below this = chop, no trend-follow edge).
     At 1-minute granularity this is more reliable than autocorrelation
     because noise-to-signal is too high for autocorr to be meaningful.

  2. Structural location
     Price must be within location_atr_mult * ATR of VWAP or EMA9.
     Enforces pullback entries and prevents chasing moves mid-range.
     VWAP = session anchor. EMA9 = short-term momentum anchor.
     Either one counts — we just need price near something meaningful.

  3. Rolling profit factor
     Gross wins / gross losses over the last pf_lookback closed trades.
     Tracks whether the edge is currently working in live market conditions.
     PF < pf_min    -> block all entries (edge has deteriorated)
     PF < pf_reduce -> reduce probe to 1 contract (edge is borderline)
     Not evaluated until pf_ignore_below trades have closed.

Adaptive probe sizing (applied on top of the above):
  consecutive_losses >= consec_loss_reduce -> probe drops to 1 contract
  consecutive_losses >= consec_loss_stop   -> done for session

Returns: (allowed: bool, probe_contracts: int, reason: str)
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TQFConfig:
    # --- Efficiency Ratio ---
    # NOTE: ER quality is handled by MarketDirectionEngine before TQF is called.
    # TQF ER check is disabled (er_min=0.0) to avoid double-blocking on a different
    # window/threshold. Engine uses 12-bar ER; TQF used 20-bar — they disagreed on
    # early trend initiations where ER is rising but not yet high on the longer window.
    er_bars: int = 20              # kept for reference, not used when er_min=0.0
    er_min: float = 0.0            # disabled — engine handles ER gating

    # --- Structural location ---
    location_atr_mult: float = 3.0   # price must be within N*ATR of VWAP or EMA9
                                      # raised from 1.5: trend days move far from VWAP but EMA9 tracks

    # --- Rolling profit factor ---
    pf_lookback: int = 6           # last N closed trades to evaluate
    pf_min: float = 0.50           # below this: block all entries
    pf_reduce: float = 0.70        # below this: reduce probe to 1 contract
    pf_ignore_below: int = 3       # do not evaluate PF until N trades have closed

    # --- Adaptive sizing thresholds ---
    consec_loss_reduce: int = 3    # drop probe to 1 contract after N consecutive losses
    consec_loss_stop: int = 5      # stop for session after N consecutive losses


class TradeQualityFilter:
    """
    Pre-entry gate. Evaluated once per signal before any order is sent.
    All inputs are pre-computed values already sitting in MarketState.
    """

    def __init__(self, config: TQFConfig = None):
        self.config = config or TQFConfig()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _efficiency_ratio(self, closes: np.ndarray) -> float:
        """
        Kaufman Efficiency Ratio over the last er_bars bars.

        ER = |closes[-1] - closes[0]| / sum(|closes[i] - closes[i-1]|)

        Interpretation:
          ER = 1.0  pure trend: every bar moved in the same direction
          ER = 0.0  pure chop: price oscillated with zero net progress
          ER > 0.25 sufficient directional efficiency to trend-follow
          ER < 0.25 chop disguised as movement — no edge

        Returns 0.5 (pass) if not enough data.
        """
        n = self.config.er_bars
        if len(closes) < n + 1:
            return 0.5

        window = closes[-(n + 1):]
        net = abs(window[-1] - window[0])
        path = np.sum(np.abs(np.diff(window)))

        if path == 0:
            return 0.0

        return float(net / path)

    def _near_structural_level(
        self,
        price: float,
        vwap: float,
        ema_9: float,
        atr: float,
    ) -> tuple:
        """
        Returns (ok: bool, reason: str).

        ok=True if price is within location_atr_mult * ATR of either VWAP or EMA9.
        Either anchor counts. The check is direction-agnostic: being near a level
        is what matters, not which side of it.
        """
        if atr <= 0:
            return True, "atr=0 skip"

        threshold = self.config.location_atr_mult * atr
        dist_vwap = abs(price - vwap)
        dist_ema9 = abs(price - ema_9)

        if dist_vwap <= threshold:
            return True, f"near VWAP {vwap:.2f} ({dist_vwap:.2f}pts < {threshold:.2f})"
        if dist_ema9 <= threshold:
            return True, f"near EMA9 {ema_9:.2f} ({dist_ema9:.2f}pts < {threshold:.2f})"

        return False, (
            f"price {price:.2f} | VWAP {vwap:.2f} ({dist_vwap:.2f}pts away) "
            f"EMA9 {ema_9:.2f} ({dist_ema9:.2f}pts away) | "
            f"threshold {threshold:.2f}pts ({self.config.location_atr_mult}x ATR={atr:.2f})"
        )

    def _profit_factor(self, recent_pnls: list) -> Optional[float]:
        """
        Gross wins / gross losses over the last pf_lookback trades.
        Returns None if fewer than pf_ignore_below trades have closed.
        Returns 2.0 if all trades in the window are winners (no losses).
        """
        if len(recent_pnls) < self.config.pf_ignore_below:
            return None

        window = recent_pnls[-self.config.pf_lookback:]
        gross_wins = sum(p for p in window if p > 0)
        gross_losses = abs(sum(p for p in window if p < 0))

        if gross_losses == 0:
            return 2.0

        return gross_wins / gross_losses

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def check(
        self,
        direction: str,
        closes: np.ndarray,
        price: float,
        vwap: float,
        ema_9: float,
        atr: float,
        recent_pnls: list,
        consecutive_losses: int,
        normal_probe_size: int,
        skip_location: bool = False,
    ) -> tuple:
        """
        Evaluate all three conditions plus adaptive sizing.

        Args:
            direction:          'LONG' or 'SHORT'
            closes:             numpy array of recent bar closes (>= autocorr_bars+1)
            price:              current price
            vwap:               session VWAP
            ema_9:              9-period EMA
            atr:                current ATR
            recent_pnls:        list of closed trade P&Ls (dollars, oldest first)
            consecutive_losses: current consecutive loss count
            normal_probe_size:  what T1 would be at full confidence

        Returns:
            (allowed: bool, probe_contracts: int, reason: str)
            allowed=False means skip this entry entirely.
            probe_contracts may be less than normal_probe_size.
        """
        cfg = self.config

        # Hard stop: session done
        if consecutive_losses >= cfg.consec_loss_stop:
            return False, 0, (
                f"DONE FOR SESSION: {consecutive_losses} consecutive losses "
                f">= stop threshold {cfg.consec_loss_stop}"
            )

        # --- Condition 1: Efficiency Ratio ---
        er = self._efficiency_ratio(closes)
        if er < cfg.er_min:
            return False, 0, (
                f"EFFICIENCY RATIO: {er:.3f} < {cfg.er_min} | "
                f"market is choppy, no directional edge"
            )

        # --- Condition 2: Structural location ---
        if skip_location:
            loc_ok, loc_reason = True, "location skipped (DIRECT ENTRY)"
        else:
            loc_ok, loc_reason = self._near_structural_level(price, vwap, ema_9, atr)
        if not loc_ok:
            return False, 0, f"LOCATION: {loc_reason}"

        # --- Condition 3: Rolling profit factor ---
        pf = self._profit_factor(recent_pnls)
        if pf is not None and pf < cfg.pf_min:
            return False, 0, (
                f"PROFIT FACTOR: {pf:.2f} < {cfg.pf_min} | "
                f"edge has deteriorated over last {cfg.pf_lookback} trades"
            )

        # --- Determine probe size ---
        probe = normal_probe_size

        if consecutive_losses >= cfg.consec_loss_reduce:
            probe = min(probe, 1)

        if pf is not None and pf < cfg.pf_reduce:
            probe = min(probe, 1)

        pf_str = f"{pf:.2f}" if pf is not None else "n/a (warming up)"
        reason = (
            f"PASS | ER={er:.3f} | {loc_reason} | "
            f"pf={pf_str} | consec_losses={consecutive_losses} | probe={probe}x"
        )
        return True, probe, reason
