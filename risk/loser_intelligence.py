"""
Loser Intelligence Layer (LIL) — Ported from topstep3 Phase 8.40.

State-aware loser management that distinguishes between:
  - Developing trades (noisy but valid)
  - Stale/dead-on-arrival trades (never showed promise)
  - Failed trades (showed promise then reversed)

Key insight: A good trade proves itself early, even if it later pulls back.
This layer PREEMPTS hard stops by recognizing failure patterns.

Adapted for MES ($5/pt) from ES ($50/pt) topstep3 implementation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TradePhase(Enum):
    """Three-phase trade lifecycle."""
    INCUBATION = auto()   # 0-45s: noise tolerance, wider stops
    VALIDATION = auto()   # 45-120s: direction confirmation
    EXPANSION = auto()    # 120s+: full profit protection


class TradeEdge(Enum):
    """Edge realization state (Phase 8.28 port)."""
    UNPROVEN = auto()     # No material profit yet
    PROVEN = auto()       # Hit profit threshold — be patient
    DECAYING = auto()     # Was proven, now giving back — cut faster


class LILExitReason(Enum):
    STALE_LOSER = auto()
    LOSS_TO_PEAK = auto()
    GIVEBACK_FAILURE = auto()
    DRAWDOWN_VELOCITY = auto()


@dataclass
class LILConfig:
    """Configuration for Loser Intelligence Layer."""
    
    # Instrument — BTC defaults (1 contract = 0.01 BTC, 1 pt = $1 of BTC price)
    point_value: float = 0.01   # BTC: $0.01/pt/contract
    tick_size: float = 0.01     # BTC tick

    # === Trade Phase Timing ===
    incubation_seconds: float = 45.0    # Phase 1: noise tolerance
    validation_seconds: float = 120.0   # Phase 2: direction confirmation

    # === Early Graduation ===
    # NOTE: Original ES values (1.0 / 1.5 pts) were tuned for MES. BTC scale
    # below; BigMoneyBot overrides these to 300/500 explicitly.
    graduation_atr_mult: float = 1.0     # +1x ATR = graduated (primary signal, RELATIVE)
    graduation_min_pts: float = 200.0    # OR +$200 BTC move = graduated (fallback)
    graduation_strong_pts: float = 300.0  # OR +$300 = strong graduation

    # === TIER 1: STALE_LOSER ===
    # Trade never proved itself: low MFE + old + underwater.
    # ATR-SCALED: adapts to volatility (preferred). Fallback _pts is BTC scale.
    stale_mfe_threshold_atr: float = 0.3
    stale_mfe_threshold_pts: float = 50.0   # Fallback (was 0.75 = MES scale)
    stale_bars: int = 10                    # Stale after 10 bars with no progress
    stale_time_default: float = 600.0       # Hard ceiling: 10 min max (safety net)
    stale_time_by_session: dict = field(default_factory=lambda: {
        'NY_OPEN': 300.0,
        'US_OPEN': 300.0,
        'LONDON': 420.0,
        'US_AFTERNOON': 420.0,
        'US_MIDDAY': 600.0,
        'US_CLOSE': 420.0,
        'OVERNIGHT': 600.0,
        'ASIAN': 600.0,
    })
    
    # === TIER 1: LOSS_TO_PEAK ===
    # Current loss exceeds N× the peak profit.
    # B6 revert (2026-04-28): reverted 2.5 → 4.0. The 2.5x tightening
    # was firing exits earlier than Thursday's known-good config; combined
    # with raised graduation thresholds it killed trades that should have
    # been allowed to stay open until invalidation.
    loss_to_peak_ratio: float = 4.0
    loss_to_peak_min_peak_atr: float = 0.5
    loss_to_peak_min_peak: float = 50.0     # Fallback minimum (BTC: $50; was 0.50 = MES)

    # === TIER 1: GIVEBACK_FAILURE ===
    # Promising trade that failed to reclaim. ATR-relative is preferred.
    giveback_min_peak_atr: float = 1.0
    giveback_min_peak: float = 100.0        # Fallback (BTC: $100; was 1.0 = MES)
    giveback_time_since_peak: float = 90.0  # 90s since peak
    giveback_min_age_default: float = 300.0 # 5 min base
    giveback_min_age_by_session: dict = field(default_factory=lambda: {
        'NY_OPEN': 180.0,
        'US_OPEN': 180.0,
        'LONDON': 240.0,
        'US_AFTERNOON': 240.0,
        'US_MIDDAY': 300.0,
        'US_CLOSE': 240.0,
        'OVERNIGHT': 300.0,
        'ASIAN': 300.0,
    })

    # === POST-GRADUATION GIVEBACK PROTECTION (added 2026-05-07) ===
    # Closes the gap between LIL graduation and 1R partial. Original
    # `_check_decaying_exit` was disabled for graduated trades (comment:
    # "small 1-2pt peaks where 75% giveback is just normal MES noise").
    # That premise breaks for trades with meaningful peaks — the 2026-05-06
    # trade had +8pt peak that gave back 100%+ and ran to -8pts before
    # invalidation. Gating on peak >= N×ATR keeps small winners untouched
    # while protecting big winners that start to roll over.
    #
    # Logic: once trade has graduated AND peak MFE >= post_grad_min_peak_atr,
    # exit when unrealized P&L drops below (peak × post_grad_keep_ratio).
    # i.e. lock in 40% of peak profit by default. Gives more room than the
    # 75% giveback default but tighter than the BE ratchet user disabled.
    post_grad_giveback_enabled: bool = True
    post_grad_min_peak_atr: float = 1.0     # arm only when peak >= N×ATR
    post_grad_keep_ratio: float = 0.40      # exit when current < peak × this
    
    # === TIER 2: DRAWDOWN_VELOCITY ===
    # Losing money too fast. velocity_threshold_per_min is in $/min and was
    # already in dollar units (not _pts), so it carries cleanly.
    velocity_threshold_per_min: float = 50.0  # $50/min loss rate trigger
    velocity_min_age: float = 90.0
    velocity_mfe_threshold: float = 100.0     # BTC: only fire if MFE < $100 (was 1.0 = MES)
    velocity_window: float = 60.0

    # === TIER 2: MOMENTUM_REVERSAL ===
    reversal_mfe_threshold: float = 100.0     # BTC: $100 (was 1.0 = MES)
    reversal_mae_tightener: float = 0.7       # RELATIVE multiplier — no change
    
    # === Edge State (Phase 8.28 port) ===
    proven_threshold_pct: float = 0.50       # PROVEN when MFE >= 50% of hard stop
    proven_min_pts: float = 1.5              # OR MFE >= this (absolute floor)
    decay_giveback_pct: float = 0.50         # DECAYING when giveback >= 50% of peak
    decay_exit_giveback_pct: float = 0.75    # EXIT in DECAYING when giveback >= 75% of peak
    
    # === Incubation Guardrail ===
    # Even during incubation, don't let losses run wild
    incubation_loss_threshold_pct: float = 0.75  # Exit if loss > 75% of hard stop during incubation


@dataclass 
class LILState:
    """Per-trade state for the Loser Intelligence Layer."""
    entry_time: float = 0.0
    peak_mfe_pts: float = 0.0           # Maximum favorable excursion (points)
    peak_mfe_time: float = 0.0          # When peak MFE was reached
    worst_mae_pts: float = 0.0          # Worst adverse excursion (negative)
    graduated: bool = False              # Early graduation from incubation
    current_phase: TradePhase = TradePhase.INCUBATION
    edge_state: TradeEdge = TradeEdge.UNPROVEN
    pnl_history: list = field(default_factory=list)  # [(time, pnl_dollars)]
    
    def reset(self):
        self.entry_time = time.time()
        self.peak_mfe_pts = 0.0
        self.peak_mfe_time = time.time()
        self.worst_mae_pts = 0.0
        self.graduated = False
        self.current_phase = TradePhase.INCUBATION
        self.edge_state = TradeEdge.UNPROVEN
        self.pnl_history = []


class LoserIntelligenceLayer:
    """
    State-aware loser management that preempts hard stops by recognizing
    failure patterns BEFORE they hit catastrophic levels.
    
    Designed to sit between entry and the tranche exit system:
    - Runs AFTER: entry confirmation
    - Runs BEFORE: tranche exits and hard stops
    
    Usage:
        lil = LoserIntelligenceLayer(config)
        lil.on_trade_open(atr)
        
        # On each bar:
        result = lil.check(unrealized_pts, unrealized_dollars, atr, session)
        if result.should_exit:
            # exit entire position
    """
    
    def __init__(self, config: LILConfig = None):
        self.config = config or LILConfig()
        self.state = LILState()
        self._last_phase_logged = None
    
    def on_trade_open(self, atr: float = 2.5):
        """Call when a new trade is opened."""
        self.state.reset()
        self._last_phase_logged = None
        self._atr_at_entry = atr
        self._logged_grad_suppress = False
        logger.info(f"🧠 [LIL] Position opened — Loser Intelligence Layer active (ATR={atr:.2f})")
    
    def on_trade_close(self):
        """Call when trade is fully closed."""
        self.state = LILState()
        self._last_phase_logged = None
    
    @property
    def is_graduated(self) -> bool:
        """Whether the trade has graduated (proved itself via market behavior)."""
        return self.state.graduated
    
    @property
    def is_in_incubation(self) -> bool:
        """Whether the trade is still in incubation (not graduated, under time limit)."""
        if self.state.graduated:
            return False
        age = time.time() - self.state.entry_time
        return age < self.config.incubation_seconds
    
    # ─── Phase Management ─────────────────────────────────────────
    
    def _update_phase(self, unrealized_pts: float, atr: float) -> TradePhase:
        """Determine current trade phase — BEHAVIOR-BASED graduation (topstep3 port).
        
        Graduation is NOT time-based. The market decides when a trade is validated:
          - +1x ATR favorable → graduated (noise absorbed, structure validated)
          - +N pts profit → graduated (concrete win, not noise)
          - +strong pts → graduated (strong move, well beyond noise)
          - 10+ min AND green → graduated (slow grind confirmation, e.g. ASIAN)
        
        Once graduated:
          - Incubation ends immediately (even if < incubation_seconds)
          - Tier 1 LIL checks (STALE, GIVEBACK, LOSS_TO_PEAK) are SUPPRESSED
          - Trailing stops / tranche system takes over exit authority
          - Only emergency checks remain (velocity bleedout, incubation guardrail)
        
        This is a STATE MACHINE, not a timer:
          INCUBATION → (graduation OR time expiry) → VALIDATION → EXPANSION
        """
        age = time.time() - self.state.entry_time
        cfg = self.config
        
        # Check graduation — behavior-based, can happen at ANY time
        if not self.state.graduated:
            graduation_reasons = []
            
            # Condition 1: ≥1 ATR favorable (noise absorbed, structure validated)
            if atr > 0 and unrealized_pts >= atr * cfg.graduation_atr_mult:
                graduation_reasons.append(f"+{unrealized_pts:.2f}pts >= {atr*cfg.graduation_atr_mult:.2f} ({cfg.graduation_atr_mult}x ATR)")
            
            # Condition 2: ≥N point profit (concrete win, not noise)
            if unrealized_pts >= cfg.graduation_min_pts:
                graduation_reasons.append(f"+{unrealized_pts:.2f}pts >= {cfg.graduation_min_pts}")
            
            # Condition 3: ≥strong pts profit (strong move, well beyond noise)
            if unrealized_pts >= cfg.graduation_strong_pts:
                graduation_reasons.append(f"+{unrealized_pts:.2f}pts >= {cfg.graduation_strong_pts} (strong)")
            
            # Condition 4: ≥10 min AND green (slow grind confirmation — ASIAN/overnight)
            if age >= 600 and unrealized_pts > 0:
                graduation_reasons.append(f"≥10min AND green ({age:.0f}s, +{unrealized_pts:.2f}pts)")
            
            # Use peak MFE too — if trade EVER proved itself, it graduated
            if atr > 0 and self.state.peak_mfe_pts >= atr * cfg.graduation_atr_mult:
                if not graduation_reasons:
                    graduation_reasons.append(f"peak MFE={self.state.peak_mfe_pts:.2f}pts >= {atr*cfg.graduation_atr_mult:.2f} ({cfg.graduation_atr_mult}x ATR)")
            
            if graduation_reasons:
                self.state.graduated = True
                logger.info(f"🎓 [LIL] GRADUATED at {age:.0f}s: {graduation_reasons[0]}")
                logger.info(f"   → Tier 1 LIL checks SUPPRESSED. Trailing stops own exits now.")
        
        # Determine phase — graduation overrides time
        if self.state.graduated or age >= cfg.incubation_seconds:
            if age < cfg.validation_seconds:
                phase = TradePhase.VALIDATION
            else:
                phase = TradePhase.EXPANSION
        else:
            phase = TradePhase.INCUBATION
        
        # Log transitions
        if phase != self._last_phase_logged:
            logger.info(f"[LIL-PHASE] → {phase.name} (age={age:.0f}s, graduated={self.state.graduated})")
            self._last_phase_logged = phase
        
        self.state.current_phase = phase
        return phase
    
    def _update_tracking(self, unrealized_pts: float, unrealized_dollars: float):
        """Update MFE/MAE tracking."""
        now = time.time()
        
        if unrealized_pts > self.state.peak_mfe_pts:
            self.state.peak_mfe_pts = unrealized_pts
            self.state.peak_mfe_time = now
        
        if unrealized_pts < self.state.worst_mae_pts:
            self.state.worst_mae_pts = unrealized_pts
        
        # Append to pnl history (throttled to ~1/sec in backtests)
        if not self.state.pnl_history or (now - self.state.pnl_history[-1][0]) >= 1.0:
            self.state.pnl_history.append((now, unrealized_dollars))
            # Keep last 5 minutes
            cutoff = now - 300
            self.state.pnl_history = [(t, p) for t, p in self.state.pnl_history if t >= cutoff]
    
    # ─── Edge State (Phase 8.28 port) ────────────────────────────
    
    def _update_edge_state(self, unrealized_pts: float, hard_stop_pts: float):
        """Update UNPROVEN → PROVEN → DECAYING one-way state machine."""
        cfg = self.config
        state = self.state
        
        if state.edge_state == TradeEdge.UNPROVEN:
            # PROVEN when MFE >= threshold
            proven_threshold = max(
                hard_stop_pts * cfg.proven_threshold_pct,
                cfg.proven_min_pts,
            )
            if state.peak_mfe_pts >= proven_threshold:
                state.edge_state = TradeEdge.PROVEN
                logger.info(
                    f"[LIL-EDGE] UNPROVEN -> PROVEN | "
                    f"peak={state.peak_mfe_pts:.2f}pts >= {proven_threshold:.2f}pts"
                )
        
        if state.edge_state == TradeEdge.PROVEN:
            # DECAYING when giveback >= 50% of peak
            if state.peak_mfe_pts > 0:
                giveback = state.peak_mfe_pts - unrealized_pts
                if giveback >= state.peak_mfe_pts * cfg.decay_giveback_pct:
                    state.edge_state = TradeEdge.DECAYING
                    logger.warning(
                        f"[LIL-EDGE] PROVEN -> DECAYING | "
                        f"peak={state.peak_mfe_pts:.2f}, now={unrealized_pts:.2f}, "
                        f"giveback={giveback:.2f}pts ({giveback/state.peak_mfe_pts:.0%})"
                    )
    
    def _check_decaying_exit(self, unrealized_pts: float) -> Tuple[bool, Optional[str]]:
        """In DECAYING state, exit if giveback exceeds 75% of peak."""
        if self.state.edge_state != TradeEdge.DECAYING:
            return False, None
        if self.state.peak_mfe_pts <= 0:
            return False, None
        
        giveback = self.state.peak_mfe_pts - unrealized_pts
        threshold = self.state.peak_mfe_pts * self.config.decay_exit_giveback_pct
        
        if giveback >= threshold:
            reason = (
                f"DECAY_EXIT: peak={self.state.peak_mfe_pts:+.2f}pts, "
                f"now={unrealized_pts:+.2f}pts, "
                f"giveback={giveback:.2f}pts >= {threshold:.2f}pts (75% of peak)"
            )
            logger.warning(f"[LIL-EDGE] {reason}")
            return True, reason

        return False, None

    def _check_post_grad_giveback(self,
                                   unrealized_pts: float,
                                   atr: float) -> Tuple[bool, Optional[str]]:
        """POST-GRADUATION GIVEBACK PROTECTION.

        Once the trade is graduated, the legacy DECAY_EXIT path is bypassed.
        This check fills the gap: when peak MFE has been meaningful
        (>= post_grad_min_peak_atr × ATR), exit if current P&L drops below
        (peak × post_grad_keep_ratio).

        Default: peak >= 1×ATR arms; exit when current < 40% of peak.

        Example (2026-05-06 trade, ATR≈2.6):
          peak +8.0 (≥ 2.6 → armed) | floor = 8.0×0.40 = +3.2pts
          When unrealized fell below +3.2pts, exit fires.
          Saves ~$200 vs running to invalidation.
        """
        cfg = self.config
        if not cfg.post_grad_giveback_enabled:
            return False, None

        state = self.state
        if state.peak_mfe_pts <= 0:
            return False, None

        # Arm only when peak is meaningful — small 1-2pt peaks aren't worth
        # protecting (their normal noise would trigger the exit).
        if atr > 0:
            min_peak = cfg.post_grad_min_peak_atr * atr
        else:
            min_peak = cfg.giveback_min_peak  # ATR-less fallback

        if state.peak_mfe_pts < min_peak:
            return False, None

        floor = state.peak_mfe_pts * cfg.post_grad_keep_ratio
        if unrealized_pts < floor:
            giveback = state.peak_mfe_pts - unrealized_pts
            reason = (
                f"POST_GRAD_GIVEBACK: peak={state.peak_mfe_pts:+.2f}pts "
                f">= {min_peak:.2f} (1x ATR); "
                f"now={unrealized_pts:+.2f}pts < {floor:.2f}pts "
                f"(keep {cfg.post_grad_keep_ratio:.0%}, "
                f"giveback={giveback:.2f}pts)"
            )
            logger.warning(f"🧠 [LIL-PGB] {reason}")
            return True, reason
        return False, None

    # ─── Tier 1 Checks ───────────────────────────────────────────
    
    def _check_stale_loser(self, unrealized_pts: float, session: str, atr: float = 0) -> Tuple[bool, Optional[str]]:
        """TIER 1: Trade never proved itself — low MFE + old + underwater.
        
        ATR-SCALED: threshold adapts to current volatility.
        In low-vol (Asian): ATR=1, threshold=0.3pts, time=600s
        In high-vol (US open): ATR=5, threshold=1.5pts, time=300s
        """
        if unrealized_pts >= 0:
            return False, None
        
        cfg = self.config
        age = time.time() - self.state.entry_time
        
        # ATR-scaled MFE threshold (adapts to session volatility)
        if atr > 0:
            mfe_threshold = max(atr * cfg.stale_mfe_threshold_atr, cfg.stale_mfe_threshold_pts * 0.5)
        else:
            mfe_threshold = cfg.stale_mfe_threshold_pts
        
        time_threshold = cfg.stale_time_by_session.get(session, cfg.stale_time_default)
        
        if (self.state.peak_mfe_pts < mfe_threshold and age > time_threshold):
            reason = (
                f"STALE_LOSER: peak={self.state.peak_mfe_pts:.2f}pts < {mfe_threshold:.2f} "
                f"({cfg.stale_mfe_threshold_atr}x ATR={atr:.2f}), "
                f"age={age:.0f}s > {time_threshold:.0f}s"
            )
            logger.warning(f"🧠 [LIL-T1] {reason} | P&L={unrealized_pts:+.2f}pts")
            return True, reason
        
        return False, None
    
    def _check_loss_to_peak(self, unrealized_pts: float, atr: float = 0) -> Tuple[bool, Optional[str]]:
        """TIER 1: Current loss exceeds N× the peak profit."""
        cfg = self.config
        
        # ATR-scaled min peak
        min_peak = max(cfg.loss_to_peak_min_peak, atr * cfg.loss_to_peak_min_peak_atr) if atr > 0 else cfg.loss_to_peak_min_peak
        if self.state.peak_mfe_pts < min_peak:
            return False, None
        if unrealized_pts >= 0:
            return False, None
        
        current_loss = abs(unrealized_pts)
        ratio = current_loss / self.state.peak_mfe_pts
        
        if ratio >= cfg.loss_to_peak_ratio:
            reason = (
                f"LOSS_TO_PEAK: {ratio:.1f}× peak "
                f"(peak={self.state.peak_mfe_pts:+.2f}, now={unrealized_pts:+.2f})"
            )
            logger.warning(f"🧠 [LIL-T1] {reason}")
            return True, reason
        
        return False, None
    
    def _check_giveback_failure(self, unrealized_pts: float, session: str) -> Tuple[bool, Optional[str]]:
        """TIER 1: Promising trade that failed to reclaim."""
        cfg = self.config
        now = time.time()
        age = now - self.state.entry_time
        
        # Must have had meaningful peak (ATR-scaled)
        min_peak = max(cfg.giveback_min_peak, self._atr_at_entry * cfg.giveback_min_peak_atr) if hasattr(self, '_atr_at_entry') and self._atr_at_entry > 0 else cfg.giveback_min_peak
        if self.state.peak_mfe_pts < min_peak:
            return False, None
        
        # Must be underwater
        if unrealized_pts >= 0:
            return False, None
        
        # Drawdown must have erased the peak (100%+ giveback)
        current_loss = abs(unrealized_pts)
        total_drawdown = self.state.peak_mfe_pts + current_loss
        if total_drawdown < self.state.peak_mfe_pts:
            return False, None
        
        # Time since peak
        time_since_peak = now - self.state.peak_mfe_time
        if time_since_peak < cfg.giveback_time_since_peak:
            return False, None
        
        # Session-aware minimum age
        min_age = cfg.giveback_min_age_by_session.get(session, cfg.giveback_min_age_default)
        if age < min_age:
            return False, None
        
        reason = (
            f"GIVEBACK_FAILURE: peak={self.state.peak_mfe_pts:+.2f}pts, "
            f"now={unrealized_pts:+.2f}pts, "
            f"drawdown={total_drawdown:.2f}pts, "
            f"time_since_peak={time_since_peak:.0f}s"
        )
        logger.warning(f"🧠 [LIL-T1] {reason}")
        return True, reason
    
    # ─── Tier 2 Checks ───────────────────────────────────────────
    
    def _check_drawdown_velocity(self, unrealized_dollars: float) -> Tuple[bool, Optional[str]]:
        """TIER 2: Losing money too fast."""
        cfg = self.config
        age = time.time() - self.state.entry_time
        
        if age < cfg.velocity_min_age:
            return False, None
        if self.state.peak_mfe_pts >= cfg.velocity_mfe_threshold:
            return False, None
        if len(self.state.pnl_history) < 2:
            return False, None
        
        now = time.time()
        window_start = now - cfg.velocity_window
        history = [(t, p) for t, p in self.state.pnl_history if t >= window_start]
        
        if len(history) < 2:
            return False, None
        
        time_diff = history[-1][0] - history[0][0]
        if time_diff < 30:
            return False, None
        
        pnl_change = history[-1][1] - history[0][1]
        velocity_per_min = (pnl_change / time_diff) * 60
        
        if velocity_per_min >= -cfg.velocity_threshold_per_min:
            return False, None
        
        reason = (
            f"VELOCITY: ${abs(velocity_per_min):.0f}/min bleedout "
            f"(threshold: ${cfg.velocity_threshold_per_min}/min)"
        )
        logger.warning(f"🧠 [LIL-T2] {reason}")
        return True, reason
    
    def _check_incubation_guardrail(
        self, unrealized_pts: float, hard_stop_pts: float
    ) -> Tuple[bool, Optional[str]]:
        """During incubation, still bail if loss exceeds threshold of hard stop."""
        if unrealized_pts >= 0:
            return False, None
        
        cfg = self.config
        threshold = hard_stop_pts * cfg.incubation_loss_threshold_pct
        
        if abs(unrealized_pts) > threshold:
            reason = (
                f"INCUBATION_GUARDRAIL: loss={unrealized_pts:+.2f}pts > "
                f"{threshold:.2f}pts ({cfg.incubation_loss_threshold_pct:.0%} of {hard_stop_pts:.2f}pt stop)"
            )
            logger.warning(f"🧠 [LIL] {reason}")
            return True, reason
        
        return False, None
    
    # ─── Master Check ─────────────────────────────────────────────
    
    def check(
        self,
        unrealized_pts: float,
        unrealized_dollars: float,
        atr: float,
        session: str = "US_MIDDAY",
        hard_stop_pts: float = 3.0,
    ) -> Tuple[bool, Optional[str], TradePhase]:
        """
        Master LIL check — runs all detectors in priority order.
        
        Args:
            unrealized_pts: Current unrealized P&L in points
            unrealized_dollars: Current unrealized P&L in dollars (total position)
            atr: Current ATR
            session: Trading session name
            hard_stop_pts: Hard stop distance in points
            
        Returns:
            (should_exit, reason, current_phase)
        """
        # Update tracking
        self._update_tracking(unrealized_pts, unrealized_dollars)
        phase = self._update_phase(unrealized_pts, atr)
        # Edge state: informational only (logged but NOT used for exits)
        self._update_edge_state(unrealized_pts, hard_stop_pts)
        
        # During INCUBATION: only check guardrail (wide tolerance)
        if phase == TradePhase.INCUBATION:
            should_exit, reason = self._check_incubation_guardrail(unrealized_pts, hard_stop_pts)
            if should_exit:
                return True, reason, phase
            return False, None, phase
        
        # ================================================================
        # POST-INCUBATION EXIT CHECKS
        # ================================================================
        # Architecture (from topstep3):
        #   - GRADUATED trades: Tier 1 checks SUPPRESSED. The trade proved
        #     itself — trailing stops / tranche system owns exits now.
        #     Only emergency checks (velocity bleedout) remain.
        #   - NON-GRADUATED trades: Full Tier 1 + Tier 2 checks active.
        #     These trades never proved themselves — LIL is the safety net.
        # ================================================================
        
        if self.state.graduated:
            # Trade proved itself — brackets own exits now.
            # Only velocity bleedout (catastrophic) remains as emergency check.
            # DECAY_EXIT DISABLED: backtest R:R of 1.56 was achieved without it.
            # With 1.2×ATR stops, peaks are small (1-2pts) and 75% giveback
            # is just normal MES noise — was choking winners before target.
            should_exit, reason = self._check_drawdown_velocity(unrealized_dollars)
            if should_exit:
                return True, f"POST-GRAD {reason}", phase

            # POST-GRADUATION GIVEBACK PROTECTION (added 2026-05-07).
            # Closes the gap between graduation and 1R partial. Only fires
            # when peak MFE has been meaningful (>= 1×ATR) — small peaks
            # aren't protected (the original concern that disabled DECAY_EXIT).
            # See `_check_post_grad_giveback` docstring for the math.
            should_exit, reason = self._check_post_grad_giveback(unrealized_pts, atr)
            if should_exit:
                return True, f"POST-GRAD {reason}", phase

            if not getattr(self, '_logged_grad_suppress', False):
                logger.info(f"    Tier 1 LIL checks SUPPRESSED. Trailing stops own exits now.")
                self._logged_grad_suppress = True
            return False, None, phase
        
        # NON-GRADUATED: Full LIL checks — trade never proved itself
        
        # TIER 1: STALE_LOSER
        should_exit, reason = self._check_stale_loser(unrealized_pts, session, atr)
        if should_exit:
            return True, reason, phase
        
        # TIER 1: LOSS_TO_PEAK
        should_exit, reason = self._check_loss_to_peak(unrealized_pts, atr)
        if should_exit:
            return True, reason, phase
        
        # TIER 1: GIVEBACK_FAILURE
        should_exit, reason = self._check_giveback_failure(unrealized_pts, session)
        if should_exit:
            return True, reason, phase
        
        # TIER 2: DRAWDOWN_VELOCITY
        should_exit, reason = self._check_drawdown_velocity(unrealized_dollars)
        if should_exit:
            return True, reason, phase
        
        return False, None, phase
