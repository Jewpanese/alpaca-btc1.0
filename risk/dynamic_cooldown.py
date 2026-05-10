"""
DynamicCooldown — Market-State Driven Re-entry Gate

Replaces fixed-timer consecutive loss cooldown with two market-driven gates:

  LOSS GATE: After 2 consecutive losses, require the regime to fully reset
  (≥2 CHOP bars confirming the bad regime ended, then ≥2 TREND/GRIND bars
  confirming a new regime began) before allowing re-entry. After 3 losses:
  session pause — something is systematically wrong.

  WIN EXHAUSTION GATE: After 2 wins in the same direction, require price to
  pull back ≥1×ATR from the streak extreme before the next same-direction
  entry. Also raises the ADX threshold from 30→35 so only the strongest
  continuation signals can re-enter. Prevents chasing an exhausted move.

Why market-state over timers:
  A bad regime can persist for 2 hours (timer too short) or reset in 10 min
  (timer too long). The regime itself is the correct signal, not the clock.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DynamicCooldown:
    # Regime reset thresholds
    CHOP_BARS_REQUIRED  = 2   # CHOP bars needed to confirm regime died
    TREND_BARS_REQUIRED = 2   # TREND/GRIND bars needed to confirm new regime
    LOSS_GATE_THRESHOLD = 2   # consecutive losses before regime reset required
    SESSION_PAUSE_THRESHOLD = 3  # consecutive losses before session pause

    # Win exhaustion thresholds
    WIN_STREAK_THRESHOLD    = 2     # wins in same direction before exhaustion check
    PULLBACK_ATR_MULTIPLIER = 1.0   # price must retrace this × ATR from streak extreme
    WIN_STREAK_ADX_GATE     = 35.0  # raised ADX bar after win streak (vs normal 30)

    def __init__(self):
        # Loss tracking
        self._consecutive_losses: int  = 0
        self._regime_reset_needed: bool = False
        self._session_paused: bool      = False

        # Regime reset state machine
        self._saw_chop:         bool = False
        self._chop_count:       int  = 0
        self._new_trend_count:  int  = 0
        self._last_reset_regime: Optional[str] = None

        # Win streak tracking (per direction)
        self._wins_long:          int   = 0
        self._wins_short:         int   = 0
        self._streak_direction:   Optional[str] = None
        self._streak_high:        Optional[float] = None   # track for LONG pullback
        self._streak_low:         Optional[float] = None   # track for SHORT pullback

    # ── Bar-level update ──────────────────────────────────────────────────────

    def on_bar(self, regime: str, price: float):
        """
        Call once per bar after ERRegimeClassifier produces a regime.
        Updates regime reset state machine and streak price tracking.
        """
        # --- Regime reset state machine ---
        if self._regime_reset_needed:
            if regime == "CHOP":
                if not self._saw_chop:
                    self._saw_chop = True
                    self._chop_count = 0
                    self._new_trend_count = 0
                self._chop_count += 1

            elif regime in ("TREND", "GRIND") and self._saw_chop and self._chop_count >= self.CHOP_BARS_REQUIRED:
                self._new_trend_count += 1
                if self._new_trend_count >= self.TREND_BARS_REQUIRED:
                    self._regime_reset_needed = False
                    self._saw_chop            = False
                    self._chop_count          = 0
                    self._new_trend_count     = 0
                    logger.warning(
                        "[DynamicCooldown] REGIME RESET CONFIRMED — "
                        f"saw {self.CHOP_BARS_REQUIRED}+ CHOP bars then {self.TREND_BARS_REQUIRED}+ "
                        f"{regime} bars. Re-entry UNLOCKED."
                    )
            else:
                # RANGE or TREND without prior CHOP — doesn't count as reset
                self._new_trend_count = 0

        # --- Streak price tracking ---
        if self._streak_direction == "LONG":
            if self._streak_high is None or price > self._streak_high:
                self._streak_high = price
        elif self._streak_direction == "SHORT":
            if self._streak_low is None or price < self._streak_low:
                self._streak_low = price

    # ── Trade outcome callbacks ───────────────────────────────────────────────

    def on_win(self, direction: str, exit_price: float):
        """Call when a trade closes with P&L >= 0."""
        self._consecutive_losses   = 0
        self._regime_reset_needed  = False
        self._session_paused       = False
        # Clear regime reset state on any win
        self._saw_chop       = False
        self._chop_count     = 0
        self._new_trend_count = 0

        if direction == self._streak_direction:
            # Extending an existing streak
            if direction == "LONG":
                self._wins_long  += 1
            else:
                self._wins_short += 1
        else:
            # Direction changed — start fresh streak
            self._wins_long        = 1 if direction == "LONG" else 0
            self._wins_short       = 1 if direction == "SHORT" else 0
            self._streak_direction = direction
            self._streak_high      = exit_price
            self._streak_low       = exit_price

        wins = self._wins_long if direction == "LONG" else self._wins_short
        logger.info(
            f"[DynamicCooldown] WIN recorded — {direction} streak: {wins} | "
            f"streak_high={self._streak_high} streak_low={self._streak_low}"
        )

    def on_loss(self, direction: str):
        """Call when a trade closes with P&L < 0."""
        self._consecutive_losses += 1

        # Reset win streak — losses break any streak
        self._wins_long        = 0
        self._wins_short       = 0
        self._streak_direction = None
        self._streak_high      = None
        self._streak_low       = None

        if self._consecutive_losses >= self.SESSION_PAUSE_THRESHOLD:
            self._session_paused      = True
            self._regime_reset_needed = False
            logger.warning(
                f"[DynamicCooldown] SESSION PAUSED — {self._consecutive_losses} consecutive losses. "
                "Something is systematically wrong. No more entries this session."
            )
        elif self._consecutive_losses >= self.LOSS_GATE_THRESHOLD:
            self._regime_reset_needed = True
            self._saw_chop            = False
            self._chop_count          = 0
            self._new_trend_count     = 0
            logger.warning(
                f"[DynamicCooldown] REGIME RESET REQUIRED — {self._consecutive_losses} consecutive losses. "
                f"Need {self.CHOP_BARS_REQUIRED} CHOP bars then {self.TREND_BARS_REQUIRED} TREND bars to re-enable."
            )

    def on_session_reset(self):
        """Call at the start of each new trading session."""
        self._consecutive_losses   = 0
        self._regime_reset_needed  = False
        self._session_paused       = False
        self._saw_chop             = False
        self._chop_count           = 0
        self._new_trend_count      = 0
        self._wins_long            = 0
        self._wins_short           = 0
        self._streak_direction     = None
        self._streak_high          = None
        self._streak_low           = None
        logger.info("[DynamicCooldown] Session reset — all cooldown state cleared.")

    # ── Entry gate ────────────────────────────────────────────────────────────

    def can_enter(self, direction: str, adx: float, price: float, atr: float) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        Call this immediately before any entry attempt.
        """
        # 1. Session pause — hard stop, no entries until next session
        if self._session_paused:
            return False, (
                f"SESSION PAUSED — {self._consecutive_losses} consecutive losses. "
                "Waiting for session reset."
            )

        # 2. Regime reset gate — market must prove the bad regime ended
        if self._regime_reset_needed:
            if not self._saw_chop:
                return False, (
                    f"REGIME RESET NEEDED — {self._consecutive_losses} losses. "
                    f"Waiting for CHOP to confirm regime ended (0/{self.CHOP_BARS_REQUIRED} bars)."
                )
            elif self._chop_count < self.CHOP_BARS_REQUIRED:
                return False, (
                    f"REGIME RESET — CHOP confirmed {self._chop_count}/{self.CHOP_BARS_REQUIRED} bars. "
                    "Need more CHOP before new trend counts."
                )
            else:
                return False, (
                    f"REGIME RESET — CHOP done, waiting for new regime: "
                    f"{self._new_trend_count}/{self.TREND_BARS_REQUIRED} TREND/GRIND bars confirmed."
                )

        # 3. Win exhaustion gate
        wins_in_dir = self._wins_long if direction == "LONG" else self._wins_short
        if wins_in_dir >= self.WIN_STREAK_THRESHOLD:
            # Require price to pull back ≥1×ATR from the streak extreme
            if not self._pullback_confirmed(direction, price, atr):
                dist = self._distance_from_extreme(direction, price)
                needed = atr * self.PULLBACK_ATR_MULTIPLIER
                return False, (
                    f"WIN EXHAUSTION — {wins_in_dir} wins {direction}. "
                    f"Need {needed:.1f}pt pullback from extreme, "
                    f"only {dist:.1f}pts so far. Move may be played out."
                )
            # Even after pullback, require higher ADX to confirm a genuine new impulse
            if adx < self.WIN_STREAK_ADX_GATE:
                return False, (
                    f"WIN STREAK ADX GATE — {wins_in_dir} wins {direction}, "
                    f"pullback confirmed but ADX {adx:.1f} < {self.WIN_STREAK_ADX_GATE} required. "
                    "Need stronger momentum for post-streak re-entry."
                )

        return True, ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pullback_confirmed(self, direction: str, price: float, atr: float) -> bool:
        required = atr * self.PULLBACK_ATR_MULTIPLIER
        if direction == "LONG" and self._streak_high is not None:
            return (self._streak_high - price) >= required
        if direction == "SHORT" and self._streak_low is not None:
            return (price - self._streak_low) >= required
        return True  # No streak data yet — allow

    def _distance_from_extreme(self, direction: str, price: float) -> float:
        if direction == "LONG" and self._streak_high is not None:
            return self._streak_high - price
        if direction == "SHORT" and self._streak_low is not None:
            return price - self._streak_low
        return 0.0

    @property
    def status(self) -> str:
        if self._session_paused:
            return f"SESSION_PAUSED({self._consecutive_losses}L)"
        if self._regime_reset_needed:
            chop = f"CHOP{self._chop_count}" if self._saw_chop else "AWAITING_CHOP"
            trend = f"TREND{self._new_trend_count}" if self._saw_chop and self._chop_count >= self.CHOP_BARS_REQUIRED else ""
            return f"REGIME_RESET({chop}{'/' + trend if trend else ''})"
        return (
            f"OK — losses:{self._consecutive_losses} "
            f"streak:{self._streak_direction or 'none'} "
            f"WL:{self._wins_long} WS:{self._wins_short}"
        )
