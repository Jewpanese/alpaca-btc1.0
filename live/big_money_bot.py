"""
Big Money Bot — Trend Day Specialist.

Inspired by the $4K+ winning day on Jan 6, 2026 where the bot:
  - Detected a trend day (market went +31pts)
  - Made ALL trades in the same direction (LONG)
  - Used large size (5-6 ES = 50-60 MES equivalent)
  - 75% win rate, avg winner +3.5pts vs avg loser -1.6pts
  - Re-entered after each profitable exit in same direction

Design principles:
  1. Detect trend days early (ADX + EMA stack + VWAP alignment)
  2. Only trade WITH the trend — NEVER counter-trend
  3. Use large size (10 MES contracts) when trend confirmed
  4. Wider ATR-based stops (2x ATR stop, 3x ATR target)
  5. Re-enter after profitable exits in same direction
  6. Stop after 3 consecutive losses

Inherits from AlphaBot for connection, order execution, and position management.
Overrides strategy selection + sizing to be purely trend-focused.
"""

import logging
import time
import os
import sys
import numpy as np
from typing import Optional
from dataclasses import dataclass

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from live.alpha_bot import AlphaBot
from strategies.base import MarketState, Direction, Signal
from risk.vol_sizing import VolatilitySizer, VolSizingConfig
from core.ml_signal import MLSignalProvider, MLSignal
from core.direction_detector import (
    DirectionalDetector, BiasDirection,
    DirectionalRegime, classify_directional_regime, check_regime_entry_gate,
    check_slope_supremacy, check_short_override, check_entry_location,
)
from core.trade_quality import TradeQualityFilter, TQFConfig
from core.market_engine import MarketDirectionEngine, MarketEngineConfig, MarketBias
from core.regime_classifier import ERRegimeClassifier
from risk.loser_intelligence import LoserIntelligenceLayer, LILConfig
from risk.tranche_state_machine import (
    TrancheConfig, TradeIdea, TrancheRiskManager,
    TrancheState, Direction as TrancheDirection,
)
from core.adx_range_mapper import ADXRangeMapper, ADXRangeConfig, BreakoutSignal
from core.range_detector import MarketStructure
from risk.dynamic_cooldown import DynamicCooldown
from live.shadow_scorer import ShadowScorer
from live.shadow_variant import ShadowVariant, ShadowVariantConfig
from core.timeframe_classifier import TFClassifier
from core.volume_baseline import VolumeBaseline
from core.regime_matrix import RegimeMatrix, RegimeLabel
from core.order_flow import OrderFlowEstimator, OrderFlowSnapshot
from core.reversal_features import score_volume_signature, VolumeSignatureScore
from core.vwap_reactions import VWAPReactionTracker, VWAPReactionSnapshot
from core.reversal_score import combine_reversal_score, ReversalScoreSnapshot
logger = logging.getLogger(__name__)


@dataclass
class BigMoneyConfig:
    """Configuration for Big Money Bot trend detection and sizing."""
    # Trend detection thresholds
    adx_threshold: float = 25.0           # ADX > this = trending
    ema_stack_required: bool = True        # Require 9/26/50 EMA stack aligned
    vwap_alignment_required: bool = True   # Price on correct side of VWAP
    
    # Warmup period — collect data but block entries for N minutes after startup
    # With 3-min bars, 5 min = 2 bars. Indicators are seeded from 100 historical 3-min bars
    # (300 min = 5 hours of context) so warmup just needs to let the live ADX/ER settle.
    warmup_minutes: int = 5
    
    # Position sizing — cushion-scaled tiers
    # Wider stops at smaller size = trades breathe, same dollar risk
    non_trend_contracts: int = 0           # Don't trade if no trend (0 = skip)
    
    # Cushion-based sizing tiers (cushion_threshold, contracts, stop_atr_mult, target_atr_mult)
    # Evaluated top-down: first tier where cushion >= threshold wins
    # Below the lowest tier → don't trade at all
    cushion_tiers: list = None  # Initialized in __post_init__
    cushion_no_trade: float = 1500.0       # Below this → don't trade at all
    
    # Stop/target (ATR multiples) — defaults, overridden by tier
    stop_atr_mult: float = 2.0            # Wider stops for trend days
    target_atr_mult: float = 3.0          # Let winners run

    # 2026-04-30: BE ratchet at LIL graduation can fire BEFORE 1R partial,
    # which conflicts with the 1R-partial-then-BE structural exit ladder. The
    # ratchet was redundant with the post-1R BE stop and was killing winners
    # on shallow pullbacks before they could develop. Disabled by default;
    # set True to re-enable if shadow data shows it adds value.
    be_ratchet_enabled: bool = False

    # 2026-05-01: FAST_CONFIRM — fire on the FIRST bar that meets impulse
    # criteria, instead of waiting 2 bars (or for ADX≥30). Compensates for
    # the lag pattern observed 2026-05-01: 5 SHORT entries fired on bars
    # where ADX had finally crossed 30, but the move had already exhausted
    # — entries at the bottom of the dump, peak MFE = 0, all losers.
    # Conditions are intentionally STRICT — designed to fire ONLY on real
    # impulse bars with meaningful momentum and within reach of VWAP (not
    # chasing extended moves). All existing exit machinery (LIL, regime
    # exit, brackets, BE after 1R) is preserved — only entry timing changes.
    fast_confirm_enabled: bool = True
    fast_confirm_min_adx: float = 22.0           # Lower than DIRECT_ENTRY 30 — ADX is lagging
    fast_confirm_min_er_delta: float = 0.20      # ER must rise this much in this bar (impulse, not noise)
    fast_confirm_min_body_ratio: float = 0.50    # Bar body >= 50% of range (decisive directional candle)
    fast_confirm_max_vwap_atr: float = 3.0       # Don't fast-fire if price already 3+ATR from VWAP (chasing)

    # 2026-05-08 (Path A): Trend-day-conditional FAST_CONFIRM relaxation.
    # The bot already has `core.trend_day` which detects trend days from
    # ATR percentile + ADX + breadth. When that flag is active, the
    # FAST_CONFIRM gates are too strict — a trend day is by definition
    # "extended directional movement," so the chase-veto and impulse-bar
    # requirements veto the legitimate continuation entries.
    #
    # Relaxation rules ONLY apply when `trend_day.is_trend_day == True`:
    #   - er_delta:    0.20 → 0.05  (HIGH ER plateau is enough on a trend day)
    #   - vwap_atr:    3.0  → 5.0   (continuations on a trend day shouldn't
    #                                be vetoed for being extended)
    # Body ratio + ADX threshold + probe_mult requirements unchanged —
    # those still ensure quality.
    #
    # Kill switch: set fast_confirm_relax_on_trend_day = False to revert to
    # uniform thresholds. Diagnostic log distinguishes the relaxed path
    # ("FAST [TD]") from the standard FAST_CONFIRM path so we can tell
    # which fired, when, and at what success rate (Path B input data).
    fast_confirm_relax_on_trend_day: bool = True
    fast_confirm_min_er_delta_trend_day: float = 0.05
    fast_confirm_max_vwap_atr_trend_day: float = 5.0
    
    def __post_init__(self):
        if self.cushion_tiers is None:
            # (min_cushion, contracts, stop_atr_mult, target_atr_mult)
            # BTC port: 1 contract = 0.01 BTC. Tiers below assume ~$100k BTC price
            # so contract counts ~roughly equal $1k notional each.
            # Original MES values commented for reference.
            self.cushion_tiers = [
                (20000, 50, 2.5, 3.5),   # 0.50 BTC — full size      (was 13 MES)
                (15000, 40, 2.5, 3.5),   # 0.40 BTC                  (was 11 MES)
                (10000, 30, 2.5, 3.5),   # 0.30 BTC                  (was  9 MES)
                ( 5000, 20, 2.5, 3.5),   # 0.20 BTC                  (was  7 MES)
                ( 2500, 10, 2.5, 3.5),   # 0.10 BTC — base tier      (was  5 MES)
                ( 1500,  5, 2.5, 3.5),   # 0.05 BTC — survival mode  (was  2 MES)
            ]

    # Daily dollar loss limit — done for day if cumulative P&L drops below this.
    # User policy: 5% hard ($5k) of $100k account. Soft cap is in AccountConfig.
    daily_loss_limit: float = -5000.0     # BTC scale  (was -$1500 for Topstep MES)

    # Entry quality gates
    min_directional_confidence: float = 0.6  # 3/5 votes
    min_signal_strength: float = 0.45        # Skip weak strategy signals
    # min_atr_to_trade: BTC ATR is in $; on 3-min bars typical ATR ≈ $300-1000.
    # Setting floor at $50 keeps the gate meaningful but not blocking.
    min_atr_to_trade: float = 50.0           # BTC scale  (was 1.0 pt MES)
    
    # Slope Supremacy (Phase 8.32 port) — overrides vote system on extreme slopes
    slope_supremacy_enabled: bool = True
    
    # Short Override (Phase 8.8 port) — deterministic shorts in strong downtrends
    # Disabled: fires counter-trend right after winning LONG exits, catching reversal tops
    short_override_enabled: bool = False
    short_override_size_mult: float = 0.5    # 50% of normal size for override shorts
    short_override_cooldown_seconds: float = 180.0  # 3 min between override shorts
    
    # Re-entry
    re_entry_enabled: bool = True          # Re-enter after profitable exit
    re_entry_cooldown_seconds: float = 60  # 1 min before re-entering (reduced from 120s — was too conservative)
    
    # Consecutive LOSS cooldown (2026-03-25 — was tracked but never enforced!)
    consec_loss_cooldown_threshold: int = 2        # After 2 losses, pause
    consec_loss_cooldown_seconds: float = 180.0    # 3 min cooldown (reduced from 600s — 10 min was too long)
    
    # Consecutive winner cooldown — REMOVED 2026-04-03 (Optimizer's Curse kill list)
    # Was suppressing profitable streaks. Kept fields as dead code for reference.
    # consec_win_cooldown_threshold: int = 3
    # consec_win_cooldown_seconds: float = 300.0
    
    # ── RISK GUARD 1: Dollar Stop After Incubation ──────────────────
    # After incubation period, if down more than this, exit immediately
    dollar_stop_after_incubation: float = 500.0   # Max $ loss after incubation
    dollar_stop_incubation_seconds: float = 90.0  # Must wait this long first
    
    # ── RISK GUARD 2: Trend Exhaustion Detection (ADX/ATR-based) ──
    # Block entries when the trend move is exhausted
    exhaustion_adx_decline_bars: int = 5          # ADX must be declining over N bars
    exhaustion_vwap_distance_atr: float = 2.0     # Price > Nx ATR from VWAP = overextended
    exhaustion_atr_spike_mult: float = 1.5        # ATR > 1.5x median = blow-off
    exhaustion_rsi_extreme_long: float = 75.0     # RSI > this blocks new longs
    exhaustion_rsi_extreme_short: float = 25.0    # RSI < this blocks new shorts
    exhaustion_min_conditions: int = 4            # 2026-03-31: raised from 3→4 (was too sensitive, blocked with-trend entries)
    
    # ── RISK GUARD 3: Full Loss Cooldown ──────────────────────────
    # When ALL contracts hit stop (no tranche took profit), mandatory cooldown
    full_loss_cooldown_seconds: float = 180.0     # 3 min cooldown same direction (reduced from 300s)
    # Opposite direction still allowed during cooldown
    
    # ── ADX Range Mapper (breakout from consolidation) ──────────
    range_mapper_enabled: bool = True
    # BTC scale: original 10.0 pts (=$50 MES) is way too tight for BTC where
    # 1pt = $1 of price move. Allow up to $1500 stop on range breakouts.
    range_mapper_max_stop_pts: float = 1500.0     # BTC scale  (was 10.0 pts MES)
    range_mapper_trail_atr_mult: float = 3.0      # 3x ATR trailing stop
    range_mapper_min_strength: float = 0.6        # Min signal strength filter
    range_mapper_size_mult: float = 0.5           # 50% of normal tier size for breakouts
    
    # ── Entry Location Filter (Phase 8.33 port) ─────────────────
    # Blocks entries at price extremes — doesn't restrict direction, only location
    entry_location_filter_enabled: bool = True


class BigMoneyBot(AlphaBot):
    """Trend Day Specialist — only trades with confirmed trends, big size.
    
    Inherits all connection/execution/position management from AlphaBot.
    Overrides the decision-making to be purely trend-focused:
      - Waits for trend day confirmation before trading
      - Only takes signals in the trend direction
      - Uses 10 MES contracts (vs AlphaBot's dynamic sizing)
      - Wider stops (2x ATR) and targets (3x ATR)
      - Re-enters after profitable exits
      - Stops after 3 consecutive losses
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.bm_config = BigMoneyConfig()
        
        # Directional detector — replaces ML ensemble
        self._direction_detector = DirectionalDetector()
        
        # Fixed-point scalp tranche config (2026-03-26)
        # 60/20/20 allocation: T1=3 contracts @ 1.5pts, T2=1 @ 2pts, T3=1 @ 2pts
        # No runners, no trailing — pure fixed-target scalping.
        # On a 5-lot: full winner = 3×$7.50 + 1×$10 + 1×$10 = $42.50 gross
        from risk.adaptive_exits import AdaptiveExitConfig
        self._alpha_exit_config = AdaptiveExitConfig(
            point_value=self.instrument.point_value,
            tick_size=self.instrument.tick_size,
            
            # === 60/20/20 allocation — bulk exits fast, small rides a bit more ===
            risk_reduce_pct=0.60,   # Tranche 1: 3 contracts (quick scalp @ 1.5pts)
            core_pct=0.20,          # Tranche 2: 1 contract (fixed target @ 2pts)
            runner_pct=0.20,        # Tranche 3: 1 contract (fixed target @ 2pts)
            
            # === FIXED POINT TARGETS DISABLED (2026-04-02) ===
            # With 2.5x ATR stops (~7.5pts), fixed 1.5/2pt targets gave 0.2:1 R:R.
            # Now using ATR-scaled targets for proper R:R.
            fixed_target_pts_t1=0,           # Disabled — use ATR-based
            fixed_target_pts_t2=0,           # Disabled — use ATR-based
            fixed_target_pts_t3=0,           # Disabled — use ATR-based
            
            # ATR targets — now the primary exit method
            # T1 at 1.5x ATR, T2/T3 handled by runner trail
            risk_reduce_target_atr=1.5,      # T1: 1.5x ATR (~4.5pts) — quick partial
            risk_reduce_min_atr=0.75,        # Floor: at least 0.75x ATR
            
            # Runner config — trail the rest for bigger moves
            runner_activation_atr=1.0,       # Activate trail after 1x ATR profit
            runner_base_trail_atr=1.5,       # Trail at 1.5x ATR behind best price
            runner_min_stop_atr=0.15,
            runner_ratchet_enabled=False,    # No ratcheting — fixed targets
            runner_trending_mult=1.5,
            runner_choppy_mult=0.7,
            runner_volatile_mult=1.3,
            runner_volume_exhaustion_mult=0.6,
            runner_stale_seconds=1200,
            runner_stale_trail_atr=1.0,
            
            hard_stop_atr=2.5,              # Widened 2026-04-02 — 1.2x was inside noise band
            
            # Incubation — must match LIL config below
            incubation_seconds=90.0,
        )
        
        # Override vol sizer with wider stops and spot %-of-account sizing.
        # The %-sizing path uses balance × risk_per_trade_pct (no fixed-$ cap)
        # and respects a notional cap (spot can't auto-leverage). HybridBot pushes
        # balance into the sizer on each balance update.
        _trading = self.config.trading
        self.vol_sizer = VolatilitySizer(VolSizingConfig(
            # %-sizing path (spot BTC) — primary
            use_pct_sizing=True,
            risk_per_trade_pct=_trading.risk_per_trade_pct,
            risk_per_trade_pct_max=_trading.risk_per_trade_pct_max,
            notional_cap_pct=_trading.notional_cap_pct,
            contract_size_btc=self.instrument.contract_size_btc,
            drawdown_size_tiers=_trading.drawdown_size_tiers,

            # Stop / target shape (still ATR-relative)
            stop_atr_mult=self.bm_config.stop_atr_mult,
            target_atr_mult=self.bm_config.target_atr_mult,
            min_stop_pts=50.0,                   # BTC: $50 floor
            max_stop_pts=2000.0,                 # BTC: $2000 ceiling

            # Hard contract cap (the % path also clamps by notional)
            max_contracts=_trading.max_contracts,
            point_value=self.instrument.point_value,
            tick_size=self.instrument.tick_size,

            # Legacy fixed-$ knobs kept for futures portability (ignored when use_pct_sizing=True)
            risk_per_trade_dollars=_trading.max_loss_per_trade,
            daily_risk_budget=999_999.0,
            drawdown_full_reduce=2000.0,
            cushion_danger_zone=2000.0,
            cushion_critical=1000.0,
        ))
        
        # Override trend day detector to be more sensitive
        from core.trend_day import TrendDayConfig
        self.trend_day = self.trend_day.__class__(TrendDayConfig(
            gap_threshold_pct=0.003,             # Lower gap threshold
            atr_percentile_threshold=0.70,       # Lower ATR threshold
            adx_threshold=self.bm_config.adx_threshold,
            min_criteria=2,                      # 2 of 4 criteria
            mean_reversion_mult=0.0,             # Fully disable mean-reversion
        ))
        
        # Loser Intelligence Layer — BTC-tuned (1 pt = $1 BTC move; ATR ~$300-1000 on 3min)
        self._lil = LoserIntelligenceLayer(LILConfig(
            point_value=self.instrument.point_value,    # 0.01 for BTC (was 5.0 for MES)
            tick_size=self.instrument.tick_size,        # 0.01 for BTC (was 0.25 for MES)
            # Longer incubation — trend trades need room to develop
            incubation_seconds=90.0,
            validation_seconds=180.0,
            # Graduation: a real BTC move is meaningful only at $300+ price travel.
            # With ATR ~$500, graduation_min_pts=300 ≈ 0.6x ATR.
            graduation_atr_mult=1.5,
            graduation_min_pts=300.0,                   # BTC scale (was 8.0)
            graduation_strong_pts=500.0,                # BTC scale (was 12.0)
            # STALE_LOSER — ATR-scaled adapts naturally; absolute floor needs scaling
            stale_mfe_threshold_atr=0.3,
            stale_mfe_threshold_pts=50.0,               # BTC scale (was 0.75)
            stale_time_default=600.0,
            stale_time_by_session={
                'NY_OPEN': 600.0, 'US_OPEN': 600.0,
                'LONDON': 600.0, 'US_AFTERNOON': 480.0,
                'US_MIDDAY': 600.0, 'US_CLOSE': 480.0,
                'OVERNIGHT': 600.0, 'ASIAN': 600.0,
            },
            # LOSS_TO_PEAK — ATR-scaled; absolute mins bumped for BTC
            loss_to_peak_ratio=4.0,
            loss_to_peak_min_peak=50.0,                 # BTC scale (was 0.5)
            loss_to_peak_min_peak_atr=0.5,
            # GIVEBACK — ATR-scaled; absolute mins bumped for BTC
            giveback_min_peak=100.0,                    # BTC scale (was 1.0)
            giveback_min_peak_atr=1.0,
            giveback_time_since_peak=240.0,
            giveback_min_age_default=420.0,
            giveback_min_age_by_session={
                'NY_OPEN': 300.0, 'US_OPEN': 300.0,
                'LONDON': 360.0, 'US_AFTERNOON': 360.0,
                'US_MIDDAY': 420.0, 'US_CLOSE': 360.0,
                'OVERNIGHT': 420.0, 'ASIAN': 420.0,
            },
            # VELOCITY — for 50 contracts × $0.01 = $0.50 per pt-of-BTC. So $/min
            # threshold scales: original $250/min on MES (50ct × $5 × 1pt) maps to
            # 50ct × $0.01 × X pt → for X=$500/min, that's $250/min. Keep 250.
            velocity_threshold_per_min=250.0,
            velocity_min_age=120.0,
            velocity_mfe_threshold=100.0,               # BTC scale (was 1.5)
            velocity_window=60.0,
            # Incubation guardrail
            incubation_loss_threshold_pct=0.75,
        ))
        
        # Warmup tracking — block entries until indicators are hot
        self._bm_startup_time: float = time.time()
        self._bm_warmup_complete: bool = False
        
        # Directional regime (Phase 8.1 port)
        self._directional_regime = DirectionalRegime.RANGE
        
        # Big Money state
        self._last_directional_bias = None
        self._bm_trend_confirmed: bool = False
        self._bm_trend_direction: Optional[Direction] = None
        self._bm_trend_confirm_count: int = 0  # consecutive bars confirming same direction
        self._bm_trend_pending_direction: Optional[Direction] = None  # direction being confirmed
        self._bm_trend_confirm_required: int = 2  # need N consecutive bars to confirm trend
        self._bm_market_mode: str = "UNKNOWN"
        self._bm_consecutive_losses: int = 0
        self._bm_consec_loss_cooldown_until: float = 0.0  # legacy — superseded by DynamicCooldown
        self._dynamic_cooldown = DynamicCooldown()
        self._bm_chop_while_running: int = 0  # consecutive CHOP bars since runner activated
        self._bm_done_for_day: bool = False
        self._bm_session_ending: bool = False  # set True at 13:55 local — blocks new entries
        self._bm_session_end_shutdown_fired: bool = False  # idempotency guard for 14:15 shutdown
        self._bm_last_exit_time: float = 0.0
        self._bm_last_exit_profitable: bool = False
        self._bm_consec_wins_same_dir: int = 0
        self._bm_last_win_direction: Optional[Direction] = None
        self._bm_exhaustion_cooldown_until: float = 0.0
        self._bm_trade_count: int = 0
        self._bm_win_count: int = 0
        # _bm_total_pnl removed — now a read-through @property below (A3 fix).
        # Single source of truth is self.risk.daily_pnl.
        self._bm_last_short_override_exit: float = 0.0  # For short override cooldown
        
        # ── Tranche State Machine (2026-03-26) ─────────────────────
        self._tranche_config = TrancheConfig(
            point_value=self.instrument.point_value,
            tick_size=self.instrument.tick_size,
            # Raised 200→500: with 2.5x ATR stops and ATR≈7, old cap allowed only
            # 2 contracts total (200 / (7×2.5×5)=2). Cushion tier says 5-13 but
            # tranche budget overrode it. Now allows 5+ contracts in real market conditions.
            max_risk_per_idea_dollars=500.0,
            max_concurrent_risk_dollars=1000.0,
            daily_loss_limit_dollars=-1500.0,
            t1_fraction=0.40,     # 2 of 5 = probe
            t2_fraction=0.40,     # 2 of 5 = confirmation
            t3_fraction=0.20,     # 1 of 5 = momentum
            t2_hold_bars=2,
            t2_max_wait_seconds=900.0,    # Raised 300→900: 15 min. Fast trends need time
                                          # to confirm without racing against the expiry clock.
            t3_min_profit_atr=0.5,
            t3_max_wait_seconds=600.0,    # Raised 300→600: 10 min for T3 to develop.
            exit_1r_pct=0.50,
            exit_2r_pct=0.30,
            runner_trail_atr=2.5,              # Safety net only — regime-change exit is primary
            hard_stop_atr=2.5,             # Widened 2026-04-02 — 1.2x was inside noise band
            min_time_between_ideas_seconds=120.0,
            max_full_stops_per_direction=2,
            max_ideas_per_session=50,
        )
        self._tranche_risk_mgr = TrancheRiskManager(self._tranche_config)
        self._active_idea: Optional[TradeIdea] = None
        
        # Entry location filter state
        self._bm_slope_75_prev: float = 0.0   # Previous slope_75 for deceleration detection

        # Trade Quality Filter (2026-04-06)
        self._tqf = TradeQualityFilter(TQFConfig())
        self._tqf_recent_pnls: list = []       # rolling closed-trade P&Ls for profit factor
        self._tqf_probe_override: int = 0      # set by TQF, read by _bm_try_enter (0 = use tier)

        # Market Direction Engine (2026-04-06) — 3-layer: dual window + ADX trajectory + compression/expansion
        self._market_engine = MarketDirectionEngine()
        self._me_adx_history: list = []        # rolling ADX for market engine
        self._me_probe_mult: float = 1.0       # conviction-based probe multiplier from market engine
        self._me_er: float = 0.0               # latest Efficiency Ratio from market engine
        self._me_er_prev: float = 0.0          # prior bar's ER (for FAST_CONFIRM delta check)

        # ER-based regime classifier — replaces the inline ADX/slope block in _process_bar
        self._er_regime_clf = ERRegimeClassifier()

        # Shadow scorer — read-only observer of research-pipeline entry signals.
        # Writes JSONL per session to logs/shadow/ and takes NO trading action.
        # Wired at end of _process_bar (post-decision) on every closed bar.
        try:
            from pathlib import Path as _Path
            self._shadow_scorer = ShadowScorer(
                log_dir=_Path("logs/shadow"),
                symbol=self.instrument.instrument,
            )
            logger.info("[SHADOW] enabled — read-only observer")
        except Exception as _e:
            logger.warning(f"[SHADOW] failed to initialize — disabled: {_e}")
            self._shadow_scorer = None

        # Shadow variants — read-only A/B simulators with their own engine
        # config + simulated position lifecycle + per-trade P&L logging.
        # See `live/shadow_variant.py`. Variants run alongside the live bot;
        # they take NO trading action.
        self._shadow_variants: list[ShadowVariant] = []

        # Multi-timeframe pullback-vs-reversal classifier — SIGNAL ONLY (V1).
        # Reads 3-min bars, builds a 15-min anchor view internally, prints
        # PULLBACK / REVERSAL / NEUTRAL on every closed bar. NOT WIRED into
        # trade management yet; collecting log data for 1-2 weeks before
        # deciding whether the signal agrees with chart reality.
        try:
            self._tf_classifier = TFClassifier()
            logger.info("[TF CLASSIFIER] enabled — signal-only mode")
        except Exception as _e:
            logger.warning(f"[TF CLASSIFIER] init failed: {_e}")
            self._tf_classifier = None

        # Volume baseline — per-minute-of-day mean/std built from 5 years of
        # ES 1-min data (see research/build_volume_baseline.py). Provides
        # vol_z = (this_bar_volume - mean[NY_minute_of_day]) / std[…] which
        # is interpretable across the trading day (0=typical, +1=elevated,
        # +2=heavy). Available to downstream consumers via self._latest_vol_z.
        # Falls back to None per-bar if baseline file missing — bot still works.
        try:
            self._volume_baseline = VolumeBaseline()
            if self._volume_baseline.load():
                meta = self._volume_baseline.meta
                logger.info(
                    f"[VOL BASELINE] ready — {meta.get('n_buckets', '?')} buckets, "
                    f"~{meta.get('n_days_used', '?')} days, built {meta.get('built_at', '?')}"
                )
            else:
                logger.warning(
                    "[VOL BASELINE] file missing — vol_z will be None. "
                    "Run: python research/build_volume_baseline.py"
                )
        except Exception as _e:
            logger.warning(f"[VOL BASELINE] init failed: {_e}")
            self._volume_baseline = None
        self._latest_vol_z: Optional[float] = None

        # Regime Matrix — composite (ER, vol_z, time-of-day) regime labels
        # with strategy-veto / sizing-multiplier decisions. Phase 1 deployment
        # is VETO-ONLY: blocks entries in SWAMP cells (low ER + quiet vol),
        # CLIFF cells (climactic vol kills trend), and OVERNIGHT_QUIET. The
        # offensive layer (per-cell sizing multipliers, strategy promotion)
        # is deferred to Phase 4 once shadow + live data justifies per-cell
        # attribution. See `core/regime_matrix.py` and the threshold builder.
        try:
            self._regime_matrix = RegimeMatrix.from_thresholds_file()
            logger.info(
                f"[REGIME] matrix loaded — "
                f"ER buckets: <{self._regime_matrix.buckets.er_low_max} / "
                f"≥{self._regime_matrix.buckets.er_high_min}, "
                f"vol_z buckets: <{self._regime_matrix.buckets.vol_quiet_max} / "
                f"<{self._regime_matrix.buckets.vol_normal_max} / "
                f"<{self._regime_matrix.buckets.vol_elevated_max} / above"
            )
        except Exception as _e:
            logger.warning(f"[REGIME] matrix init failed: {_e}")
            self._regime_matrix = None
        self._latest_regime_label: Optional[RegimeLabel] = None

        # Reversal Detection — Phase A (2026-05-07): order-flow estimator +
        # volume-signature scorer. READ-ONLY — logs per-bar features. NOT
        # WIRED into entries. See `memory/project_reversal_detection_plan.md`
        # for the full 4-layer design and phase plan.
        try:
            self._order_flow = OrderFlowEstimator(
                cum_window=50, z_window=30,
                pivot_lookback=3, pivot_window=30,
            )
            self._latest_of_snapshot: Optional[OrderFlowSnapshot] = None
            self._latest_volsig: Optional[VolumeSignatureScore] = None
            # Phase B (2026-05-07): VWAP rejection tracker — third layer.
            # Counts confirmed tag-and-bounce rejections off VWAP over a
            # 30-bar window. Score saturates at 3 same-side rejections.
            self._vwap_reactions = VWAPReactionTracker(
                tag_band_atr=0.15, bounce_min_atr=0.30, bounce_window=3,
                count_window=30, confirm_count=3, score_floor_count=2,
            )
            self._latest_vwap_reaction: Optional[VWAPReactionSnapshot] = None
            # Phase C (2026-05-07): composite reversal score combining the
            # 3 detector layers (+ optional ML hook, currently None).
            self._latest_reversal_score: Optional[ReversalScoreSnapshot] = None
            # Last-N-bar close trend used to seed `recent_direction` for the
            # volume-signature scorer. Updated each bar.
            self._reversal_recent_closes: list[float] = []
            logger.info(
                "[REVERSAL DETECT] Phase A+B+C enabled — "
                "order-flow + vol-signature + vwap-reactions + composite score (ML hook open)"
            )
        except Exception as _e:
            logger.warning(f"[REVERSAL DETECT] init failed: {_e}")
            self._order_flow = None
            self._latest_of_snapshot = None
            self._latest_volsig = None
            self._vwap_reactions = None
            self._latest_vwap_reaction = None
            self._latest_reversal_score = None
            self._reversal_recent_closes = []
        try:
            from pathlib import Path as _Path
            shadow_log_dir = _Path("logs/shadow")
            # Variant A — control: same engine config the live bot uses.
            # Sanity-check: the variant should produce trade decisions broadly
            # similar to live (any divergence here means the variant's logic
            # differs from live in a way we should understand).
            self._shadow_variants.append(ShadowVariant(
                config=ShadowVariantConfig(
                    name="control",
                    engine_config=MarketEngineConfig(),
                ),
                log_dir=shadow_log_dir,
                symbol=self.instrument.instrument,
            ))
            # Variant B — low-slope: relaxes `er_high_min_slope` from 1.50 →
            # 0.50 to test whether the engine misses slow-grind trends. This
            # is the candidate change that user flagged Sunday/Monday after
            # watching multiple slow-grind moves run without entry.
            self._shadow_variants.append(ShadowVariant(
                config=ShadowVariantConfig(
                    name="low_slope",
                    engine_config=MarketEngineConfig(er_high_min_slope=0.50),
                ),
                log_dir=shadow_log_dir,
                symbol=self.instrument.instrument,
            ))
            # Variant C — mean-reversion. Fires SHORT when price extends >=2×ATR
            # above VWAP and ER drops <0.30 with a reversal candle (close <
            # prev close); LONG inverted. Same exit ladder as the trend variants
            # for an apples-to-apples sizing comparison. Engine config matches
            # control — variant differs in ENTRY rule, not engine tuning.
            self._shadow_variants.append(ShadowVariant(
                config=ShadowVariantConfig(
                    name="mean_revert",
                    engine_config=MarketEngineConfig(),
                    strategy="mean_revert",
                ),
                log_dir=shadow_log_dir,
                symbol=self.instrument.instrument,
            ))
        except Exception as _e:
            logger.warning(f"[SHADOW VARIANT] init failed — disabled: {_e}")

        # GRIND strategy state — EMA pullback tracking
        self._grind_ema_touch_long:  bool  = False   # prev bar's low touched 9 EMA from above
        self._grind_ema_touch_short: bool  = False   # prev bar's high touched 9 EMA from below
        self._grind_pullback_low:    float = 0.0     # low of the EMA-touch bar (structural stop)
        self._grind_pullback_high:   float = 0.0

        # Chart levels from ChartCapture vision analysis — used by RANGE strategy
        self._chart_support:    list  = []
        self._chart_resistance: list  = []
        self._chart_levels_ts:  float = 0.0    # timestamp levels were loaded
        self._chart_levels_regime: str = ""

        # Risk Guard state
        self._bm_adx_history: list[float] = []        # Rolling ADX for exhaustion detection
        self._bm_full_loss_cooldown_until: float = 0.0
        self._bm_full_loss_cooldown_direction: Optional[Direction] = None
        self._bm_last_trade_was_full_loss: bool = False
        
        # ADX Range Mapper — breakout from consolidation zones
        if self.bm_config.range_mapper_enabled:
            self._range_mapper = ADXRangeMapper(ADXRangeConfig(
                adx_dead_threshold=20.0,
                adx_alive_threshold=self.bm_config.adx_threshold,
                min_range_bars=10,
                # BTC scale (1 pt = $1 of price). Was 3.0 / 50.0 / 1.0 for MES.
                # Typical BTC 3-min ATR is $50–300; a tradeable consolidation
                # range is roughly 1–5× ATR = $50–1500.
                min_range_width_pts=50.0,
                max_range_width_pts=2000.0,
                breakout_buffer_pts=30.0,
                breakout_confirmation_bars=2,
                stop_at_midpoint=True,
                target_range_multiple=1.0,
            ))
            self._pending_breakout: Optional[BreakoutSignal] = None
        else:
            self._range_mapper = None
            self._pending_breakout = None
        
        # ── ML Signal Provider ──────────────────────────────────
        self._ml_provider: Optional[MLSignalProvider] = None
        self._ml_bar_accumulator = []  # accumulate 1-min bars to build 5-min bars
        self._ml_bar_count = 0
        self._ml_enabled = True  # set False to disable ML gating
        try:
            import pathlib
            model_dir = str(pathlib.Path(__file__).parent.parent / 'models' / 'production')
            self._ml_provider = MLSignalProvider.load(model_dir)
            self._ml_provider.warm_up_from_file()
            logger.info(f"ML Signal Provider loaded and warmed up from {model_dir}")
        except Exception as e:
            logger.warning(f"ML Signal Provider not available: {e}")
            self._ml_enabled = False
        
        logger.info("=" * 60)
        logger.info("💰 BIG MONEY BOT v4 — Slope Supremacy + Risk Guards + 60/20/20 Tranches + ML Signal")
        logger.info("=" * 60)
        _instr = self.instrument.instrument
        logger.info(f"  Cushion Tiers: {[(f'${t[0]:,}→{t[1]}{_instr} {t[2]}x/{t[3]}x') for t in self.bm_config.cushion_tiers]}")
        logger.info(f"  No-trade below: ${self.bm_config.cushion_no_trade:,.0f}")
        logger.info(f"  Directional confidence: {self.bm_config.min_directional_confidence:.0%} (3/5 votes)")
        logger.info(f"  Slope Supremacy: {'ON' if self.bm_config.slope_supremacy_enabled else 'OFF'}")
        logger.info(f"  Short Override: {'ON' if self.bm_config.short_override_enabled else 'OFF'}")
        logger.info(f"  ── Risk Guards ──")
        logger.info(f"  Dollar Stop: ${self.bm_config.dollar_stop_after_incubation:.0f} after {self.bm_config.dollar_stop_incubation_seconds:.0f}s incubation")
        logger.info(f"  Exhaustion: {self.bm_config.exhaustion_min_conditions} of [ADX↓, VWAP>{self.bm_config.exhaustion_vwap_distance_atr}xATR, ATR>{self.bm_config.exhaustion_atr_spike_mult}x, RSI extreme]")
        logger.info(f"  Full Loss Cooldown: {self.bm_config.full_loss_cooldown_seconds:.0f}s same-direction block")
        logger.info(f"  Daily Loss Limit: ${self.bm_config.daily_loss_limit:+,.0f}")
        logger.info(f"  ── Range Mapper ──")
        logger.info(f"  Enabled: {'ON' if self.bm_config.range_mapper_enabled else 'OFF'}")
        logger.info(f"  ── Warmup ──")
        logger.info(f"  Warmup Period: {self.bm_config.warmup_minutes} minutes (collecting data, no entries)")
        if self.bm_config.range_mapper_enabled:
            logger.info(f"  Max Stop: {self.bm_config.range_mapper_max_stop_pts}pts | Trail: {self.bm_config.range_mapper_trail_atr_mult}x ATR")
            logger.info(f"  Min Strength: {self.bm_config.range_mapper_min_strength} | Size: {self.bm_config.range_mapper_size_mult:.0%} of tier")
        logger.info("=" * 60)

    # ─── A3: Single source of P&L truth ─────────────────────────────
    @property
    def _bm_total_pnl(self) -> float:
        """Read-through view of risk.daily_pnl.

        Pre-A3 this was a second accumulator that double-counted tranche
        partials. All writes have been removed; reads now pull from the
        single authoritative value in RiskManager.
        """
        return self.risk.daily_pnl

    # ─── Risk Guard 2: Exhaustion Detection ─────────────────────────

    def _check_exhaustion(self, state: MarketState, direction: Direction) -> tuple[bool, str]:
        """Check if the trend is exhausted — should we block new entries?
        
        Uses ADX decline, VWAP overextension, ATR spike, and RSI extremes.
        Triggers when >= exhaustion_min_conditions are met.
        
        Returns:
            (is_exhausted, reason_string)
        """
        signals = []
        atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
        
        # 1. ADX declining — trend losing steam
        self._bm_adx_history.append(state.adx_14)
        if len(self._bm_adx_history) > 20:
            self._bm_adx_history = self._bm_adx_history[-20:]
        
        n = self.bm_config.exhaustion_adx_decline_bars
        if len(self._bm_adx_history) >= n:
            recent_adx = self._bm_adx_history[-1]
            prior_adx = self._bm_adx_history[-n]
            if prior_adx > 0 and recent_adx < prior_adx * 0.90:  # ADX dropped >10%
                signals.append(f"ADX declining {prior_adx:.1f}→{recent_adx:.1f}")
        
        # 2. Price overextended from VWAP
        if atr > 0 and state.vwap > 0:
            vwap_dist_atr = abs(state.price - state.vwap) / atr
            if vwap_dist_atr > self.bm_config.exhaustion_vwap_distance_atr:
                signals.append(f"VWAP dist {vwap_dist_atr:.1f}x ATR > {self.bm_config.exhaustion_vwap_distance_atr}")
        
        # 3. ATR spike (blow-off volatility)
        if state.atr_median > 0 and atr > 0:
            atr_ratio = atr / state.atr_median
            if atr_ratio > self.bm_config.exhaustion_atr_spike_mult:
                signals.append(f"ATR spike {atr_ratio:.1f}x median > {self.bm_config.exhaustion_atr_spike_mult}")
        
        # 4. RSI extreme (direction-specific)
        if direction == Direction.LONG and state.rsi_14 > self.bm_config.exhaustion_rsi_extreme_long:
            signals.append(f"RSI {state.rsi_14:.1f} > {self.bm_config.exhaustion_rsi_extreme_long} (overbought)")
        elif direction == Direction.SHORT and state.rsi_14 < self.bm_config.exhaustion_rsi_extreme_short:
            signals.append(f"RSI {state.rsi_14:.1f} < {self.bm_config.exhaustion_rsi_extreme_short} (oversold)")
        
        if len(signals) >= self.bm_config.exhaustion_min_conditions:
            reason = f"EXHAUSTION ({len(signals)}/{self.bm_config.exhaustion_min_conditions}): {' | '.join(signals)}"
            return True, reason
        
        return False, ""
    
    # ─── Tick-level runner trail update ──────────────────────────────

    def _check_tick_loss(self, price: float):
        """Override: also advance the runner trail on every live quote.

        The parent's _check_tick_loss handles emergency loss exits.
        This override additionally updates the TradeIdea's best_price so
        the runner trail tracks intra-bar highs/lows — preventing the trail
        from being initialized at bar-close price and missing a spike within
        the bar that then reverses before the next bar close.
        """
        # Update runner trail with live price before loss check
        if self._active_idea and self._active_idea.is_active:
            self._active_idea.update_live_price(price)

        # Delegate to parent for emergency loss check
        super()._check_tick_loss(price)

    # ─── Trend Detection ─────────────────────────────────────────────

    def _detect_trend(self, state: MarketState) -> tuple[bool, Optional[Direction]]:
        """Three-layer Market Direction Engine.

        Layer 1: Dual Rolling Window — fast (10-bar) + slow (75-bar) slopes must agree.
        Layer 2: ADX Trajectory    — rising = enter, falling = block, flat = reduce.
        Layer 3: Compression+Expansion — market coiling then releasing = conviction entry.

        Returns:
            (is_trend, direction) — direction is LONG or SHORT if engine fires
        """
        # Accumulate ADX history for trajectory layer — use smoothed 3-min ADX
        # 1-min ADX is too noisy (bar-to-bar swings); 3-min smoothed is the stable read
        adx_for_engine = self._adx3m_smooth if hasattr(self, '_adx3m_smooth') and self._adx3m_smooth > 0 else state.adx_14
        if adx_for_engine > 0:
            self._me_adx_history.append(adx_for_engine)
            if len(self._me_adx_history) > 50:
                self._me_adx_history = self._me_adx_history[-50:]

        closes = self.features.get_closes(100)

        result = self._market_engine.evaluate(
            closes      = closes,
            adx_history = self._me_adx_history,
        )

        logger.info(
            f"[MARKET ENGINE] {result.bias.value} | {result.conviction.value} | "
            f"probe={result.probe_mult:.1f}x | ER={result.er:.3f} prev={result.er_prev:.3f} | {result.reason}"
        )

        # Store ER every call (used by ERRegimeClassifier).
        # Also retain prev ER from the engine so the trend-confirmation block
        # can compute bar-over-bar ER delta for the FAST_CONFIRM impulse check.
        self._me_er_prev = result.er_prev
        self._me_er = result.er

        # Blocked — no trade
        if result.probe_mult == 0.0 or result.bias == MarketBias.FLAT:
            self._me_probe_mult = 0.0
            return False, None

        # Allowed — store probe multiplier, return direction
        self._me_probe_mult = result.probe_mult
        if result.bias == MarketBias.LONG:
            return True, Direction.LONG
        return True, Direction.SHORT
    
    # ─── Override: _process_bar ───────────────────────────────────────
    
    def _process_bar(self, bar: dict):
        """Wrapper: calls the real body in try/finally so the shadow hook
        runs on EVERY closed bar regardless of which early return the
        strategy code took. This is essential — the legacy body has many
        early returns (warmup, in-position, no signal, TQF-blocked, etc.),
        and without the finally the shadow log only captures the rare
        paths that reach the end of the method.

        Shadow is read-only, wrapped in try/except — no possible impact
        on live trading.
        """
        bar_ts = bar.get("t") or bar.get("time") or bar.get("timestamp")
        is_new_candle = (bar_ts != self._last_bar_timestamp) if bar_ts else True
        # Reset per-bar shadow state — populated inside _process_bar_impl
        # once the MarketState for THIS bar is built. Stays None if we
        # early-return before that (e.g., warmup, feature buffer too small).
        self._shadow_this_bar_state = None
        # Compute vol_z for this bar (None if baseline missing). Available
        # via self._latest_vol_z for downstream consumers (shadow variants,
        # TF classifier audit logs, future entry gates).
        if is_new_candle and self._volume_baseline is not None:
            try:
                bar_vol = float(bar.get("v", bar.get("volume", 0.0)))
                if bar_vol > 0 and bar_ts:
                    self._latest_vol_z = self._volume_baseline.z_score(bar_ts, bar_vol)
                else:
                    self._latest_vol_z = None
            except Exception:
                self._latest_vol_z = None
        # Compute regime label using ER (engine), vol_z (this bar), and ts.
        # Stored on instance for access in entry-veto + per-bar diagnostic logs.
        if is_new_candle and self._regime_matrix is not None:
            try:
                self._latest_regime_label = self._regime_matrix.label(
                    er=self._me_er,
                    vol_z=self._latest_vol_z,
                    ts=bar_ts,
                )
            except Exception:
                self._latest_regime_label = None

        # Reversal Detection Phase A — read-only per-bar features.
        # Order-flow divergence (Twiggs cum-delta) + volume-signature shape.
        # Logged at end of bar; NOT used in any trade decision yet.
        if is_new_candle and self._order_flow is not None:
            try:
                bar_o = float(bar.get("o", bar.get("open", 0.0)))
                bar_h = float(bar.get("h", bar.get("high", 0.0)))
                bar_l = float(bar.get("l", bar.get("low",  0.0)))
                bar_c = float(bar.get("c", bar.get("close", 0.0)))
                bar_v = float(bar.get("v", bar.get("volume", 0.0)))

                if bar_h > 0 and bar_v > 0:
                    self._latest_of_snapshot = self._order_flow.on_bar(
                        bar_o, bar_h, bar_l, bar_c, bar_v
                    )
                else:
                    self._latest_of_snapshot = None

                # Determine `recent_direction` from last 5 closes for the
                # volume-signature scorer's wick-rejection logic
                self._reversal_recent_closes.append(bar_c)
                if len(self._reversal_recent_closes) > 6:
                    self._reversal_recent_closes = self._reversal_recent_closes[-6:]
                rd = "NONE"
                if len(self._reversal_recent_closes) >= 5:
                    delta = self._reversal_recent_closes[-1] - self._reversal_recent_closes[-5]
                    atr_local = self._alpha_atr if self._alpha_atr > 0 else 1.0
                    if delta > 0.5 * atr_local:
                        rd = "LONG"
                    elif delta < -0.5 * atr_local:
                        rd = "SHORT"

                atr_for_sig = self._alpha_atr if self._alpha_atr > 0 else 0.0
                if bar_h > 0 and atr_for_sig > 0:
                    self._latest_volsig = score_volume_signature(
                        open_=bar_o, high=bar_h, low=bar_l, close=bar_c,
                        atr=atr_for_sig,
                        vol_z=self._latest_vol_z,
                        recent_direction=rd,
                    )
                else:
                    self._latest_volsig = None

                # Phase B — VWAP reaction tracker. Tag-and-bounce off VWAP.
                if self._vwap_reactions is not None and atr_for_sig > 0:
                    # MarketState VWAP is built inside _process_bar_impl;
                    # _shadow_this_bar_state may not be set yet on this hook
                    # path (it's set at end of bar). Use the last-known state
                    # if available, otherwise skip this bar gracefully.
                    state = self._shadow_this_bar_state
                    vwap_val = float(state.vwap) if (state is not None and state.vwap) else None
                    self._latest_vwap_reaction = self._vwap_reactions.on_bar(
                        close=bar_c, vwap=vwap_val, atr=atr_for_sig,
                    )
                else:
                    self._latest_vwap_reaction = None

                # Phase C — combine the 3 layers into a composite score.
                # ML hook (added 2026-05-07): convert the directional ML
                # signal into reversal probabilities. ML LONG only counts
                # as a reversal vote when recent direction was SHORT (and
                # vice versa). Returns None/None when ML agrees with recent
                # direction — combiner then redistributes the ML weight to
                # the other 3 layers.
                ml_long, ml_short = self._get_reversal_ml_probs(rd)
                self._latest_reversal_score = combine_reversal_score(
                    order_flow=self._latest_of_snapshot,
                    vol_signature=self._latest_volsig,
                    vwap=self._latest_vwap_reaction,
                    ml_long_prob=ml_long,
                    ml_short_prob=ml_short,
                )

                # Log only when at least one signal is non-trivial — avoids
                # filling logs with NEUTRAL noise but captures every potential
                # exhaustion bar for post-session review.
                composite = self._latest_reversal_score
                of_score = self._latest_of_snapshot.score if self._latest_of_snapshot else 0.0
                vs_score = self._latest_volsig.score if self._latest_volsig else 0.0
                vwap_score = 0.0
                if self._latest_vwap_reaction is not None:
                    vwap_score = max(
                        self._latest_vwap_reaction.bullish_score,
                        self._latest_vwap_reaction.bearish_score,
                    )
                vwap_just_confirmed = (
                    self._latest_vwap_reaction is not None
                    and self._latest_vwap_reaction.confirmed_this_bar != "NONE"
                )
                composite_score = composite.score if composite else 0.0
                if (of_score >= 0.30 or vs_score >= 0.30 or vwap_score >= 0.30
                        or vwap_just_confirmed or composite_score >= 0.25):
                    of_part = (
                        f"OF: bear={self._latest_of_snapshot.bearish_divergence:.2f} "
                        f"bull={self._latest_of_snapshot.bullish_divergence:.2f} "
                        f"d_z={self._latest_of_snapshot.delta_z:+.2f}"
                        if self._latest_of_snapshot else "OF: —"
                    )
                    vs_part = (
                        f"VS: score={self._latest_volsig.score:.2f} "
                        f"dir={self._latest_volsig.direction}"
                        if self._latest_volsig else "VS: —"
                    )
                    if self._latest_vwap_reaction is not None:
                        vw = self._latest_vwap_reaction
                        vwap_part = (
                            f"VW: bull={vw.bullish_score:.2f} bear={vw.bearish_score:.2f} "
                            f"counts({vw.bullish_count}/{vw.bearish_count})"
                            + (f" {vw.confirmed_this_bar}_NOW" if vw.confirmed_this_bar != "NONE" else "")
                        )
                    else:
                        vwap_part = "VW: —"
                    if composite is not None:
                        # Highlight strong signals so they're easy to find in logs
                        prefix = "[REVERSAL]"
                        if composite.score >= 0.65:
                            prefix = "[REVERSAL ⚡STRONG]"
                        elif composite.score >= 0.45:
                            prefix = "[REVERSAL ▲]"
                        ml_tag = f" ml=on" if composite.ml_used else " ml=off"
                        score_part = (
                            f"SCORE: {composite.score:.2f} dir={composite.direction} "
                            f"(L={composite.long_score:.2f} S={composite.short_score:.2f}){ml_tag}"
                        )
                        # ML contribution if present — useful for tuning
                        ml_part = ""
                        if composite.ml_used and "ml_reversion" in composite.components:
                            mc = composite.components["ml_reversion"]
                            ml_part = f" | ML: long={mc.get('long', 0):.2f} short={mc.get('short', 0):.2f}"
                        logger.info(
                            f"{prefix} rd={rd} | {score_part} | "
                            f"{of_part} | {vs_part} | {vwap_part}{ml_part}"
                        )
                    else:
                        logger.info(
                            f"[REVERSAL] rd={rd} | {of_part} | {vs_part} | {vwap_part}"
                        )
            except Exception as _re:
                logger.debug(f"[REVERSAL DETECT] per-bar compute failed: {_re}")
        try:
            self._process_bar_impl(bar)
        finally:
            if is_new_candle and self._shadow_scorer is not None:
                try:
                    self._fire_shadow_hook(bar)
                except Exception as _shadow_e:
                    logger.debug(f"[SHADOW] hook failed (ignored): {_shadow_e}")
            if is_new_candle and self._shadow_variants:
                try:
                    self._fire_shadow_variant_hook(bar)
                except Exception as _v_e:
                    logger.debug(f"[SHADOW VARIANT] hook failed (ignored): {_v_e}")
            if is_new_candle and self._tf_classifier is not None:
                try:
                    self._fire_tf_classifier_hook(bar)
                except Exception as _tf_e:
                    logger.debug(f"[TF CLASSIFIER] hook failed (ignored): {_tf_e}")

    def _fire_shadow_hook(self, bar: dict) -> None:
        """Build bot_context snapshot and hand the bar to ShadowScorer.

        Uses self._shadow_this_bar_state — None if state wasn't built for
        this bar (warmup / early feature-buffer return). In that case the
        state-derived context fields are None rather than stale.
        """
        state = self._shadow_this_bar_state
        pos_dir = None
        if self.position_direction and self.position_direction != Direction.FLAT:
            pos_dir = self.position_direction.name
        pos_qty = (
            self._active_idea.remaining_contracts
            if self._active_idea is not None else 0
        )
        shadow_bar = {
            'datetime': bar.get('t') or bar.get('time') or bar.get('timestamp'),
            'open':   float(bar.get('o', bar.get('open', 0.0))),
            'high':   float(bar.get('h', bar.get('high', 0.0))),
            'low':    float(bar.get('l', bar.get('low', 0.0))),
            'close':  float(bar.get('c', bar.get('close', 0.0))),
            'volume': float(bar.get('v', bar.get('volume', 0.0))),
        }
        bot_context = {
            "bar_closed":         True,
            "decision_phase":     "post",
            "bot_market_mode":    self._bm_market_mode,
            "bot_trend_confirmed": self._bm_trend_confirmed,
            "bot_trend_direction": (
                self._bm_trend_direction.name
                if self._bm_trend_direction else None
            ),
            "bot_directional_regime": (
                self._directional_regime.value
                if getattr(self, '_directional_regime', None) else None
            ),
            "bot_adx_3m_smooth":  getattr(self, '_adx3m_smooth', None),
            "bot_atr":            getattr(self, '_alpha_atr', None),
            "bot_er":             getattr(self, '_me_er', None),
            "bot_vwap":           state.vwap if state is not None else None,
            "bot_price":          state.price if state is not None else None,
            "bot_position_direction": pos_dir,
            "bot_position_qty":   pos_qty,
            "bot_consec_losses":  self._bm_consecutive_losses,
            "bot_state_built":    state is not None,
        }
        self._shadow_scorer.on_bar(shadow_bar, bot_context)

    def _fire_shadow_variant_hook(self, bar: dict) -> None:
        """Hand the closed bar to each ShadowVariant.

        Each variant maintains its own engine + state — we just pipe the bar
        and the same features the live bot computed (closes, ADX, ATR). No
        bot mutation; pure observers. Failures here MUST NOT affect the live
        bot, so we wrap each variant call individually.
        """
        state = self._shadow_this_bar_state
        if state is None:
            return  # No features built this bar (warmup / early return)

        variant_bar = {
            "datetime": bar.get("t") or bar.get("time") or bar.get("timestamp"),
            "open":   float(bar.get("o", bar.get("open", 0.0))),
            "high":   float(bar.get("h", bar.get("high", 0.0))),
            "low":    float(bar.get("l", bar.get("low", 0.0))),
            "close":  float(bar.get("c", bar.get("close", 0.0))),
            "volume": float(bar.get("v", bar.get("volume", 0.0))),
        }
        closes = self.features.get_closes(100)
        adx_smooth = float(getattr(self, "_adx3m_smooth", 0.0) or 0.0)
        atr = float(getattr(self, "_alpha_atr", 0.0) or state.atr_14 or 0.0)
        vwap = float(state.vwap) if state.vwap else None

        for variant in self._shadow_variants:
            try:
                variant.on_bar(variant_bar, list(closes), adx_smooth, atr, vwap,
                               vol_z=self._latest_vol_z)
            except Exception as _e:
                logger.debug(
                    f"[SHADOW VARIANT:{variant.config.name}] on_bar failed: {_e}"
                )

    def _get_reversal_ml_probs(self, recent_direction: str) -> tuple:
        """Convert the directional ML signal into reversal probabilities.

        The production ML model is a DIRECTION predictor (next-bar P(UP))
        — not a true reversion classifier. To use its output as a reversal
        layer, we condition on recent direction:

            ML predicts UP    + recent move was DOWN  → bullish reversal
            ML predicts DOWN  + recent move was UP    → bearish reversal
            ML agrees with recent direction           → continuation,
                                                        return (None, None)
                                                        so combiner skips
                                                        this layer
            ML FLAT or no recent direction            → (None, None)

        Returns (ml_long_prob, ml_short_prob) in [0,1], or (None, None)
        when ML should not contribute. Combiner redistributes the ML weight
        to the other 3 layers when both are None.

        Note: When user's XGBoost reversion model is swapped in for the
        direction model, this helper still works — a reversion-trained
        model will output P(reversion-up) / P(reversion-down) directly,
        which is even cleaner than the directional-model conditioning here.
        """
        if not getattr(self, '_ml_provider', None):
            return (None, None)
        if not getattr(self, '_ml_enabled', True):
            return (None, None)
        try:
            sig = self._ml_provider.get_signal()
        except Exception:
            return (None, None)
        if sig is None or not getattr(sig, 'tradeable', False):
            return (None, None)
        if recent_direction not in ("LONG", "SHORT"):
            # No clear recent move — ML's directional opinion isn't a
            # reversal signal in this context. Skip.
            return (None, None)

        # Reversal vote only when ML disagrees with recent direction
        if sig.direction == "LONG" and recent_direction == "SHORT":
            return (float(sig.confidence), 0.0)
        if sig.direction == "SHORT" and recent_direction == "LONG":
            return (0.0, float(sig.confidence))
        # ML agrees with recent direction or signal is FLAT → no reversal vote
        return (None, None)

    def _fire_tf_classifier_hook(self, bar: dict) -> None:
        """Hand the closed 3-min bar to the multi-timeframe classifier.

        SIGNAL ONLY — emits a single log line per bar with the PULLBACK /
        REVERSAL / NEUTRAL label, anchor direction, anchor slope, and a
        short reason. The bot does NOT act on this signal yet.
        """
        if self._tf_classifier is None:
            return
        close = float(bar.get("c", bar.get("close", 0.0)))
        high = float(bar.get("h", bar.get("high", close)))
        low = float(bar.get("l", bar.get("low", close)))
        if close == 0.0:
            return

        pos_dir = None
        if (self.position_direction is not None
                and self.position_direction != Direction.FLAT):
            pos_dir = self.position_direction.name

        signal = self._tf_classifier.on_3min_bar(
            close=close, high=high, low=low, position_dir=pos_dir,
        )
        # One concise log line per bar — quiet on NEUTRAL when no position
        # context (avoids spamming during overnight idle periods), verbose
        # when in-position so we can audit pullback/reversal calls vs the
        # bot's actual behavior.
        if pos_dir or signal.label != "NEUTRAL":
            anchor_str = (
                f"anchor={signal.anchor_dir} "
                f"slope={signal.anchor_slope_bps:+.2f}bps "
                f"ER={signal.anchor_er:.2f}"
            )
            ctx = f" pos={pos_dir}" if pos_dir else ""
            vol_str = (
                f" vol_z={self._latest_vol_z:+.2f}"
                if self._latest_vol_z is not None else ""
            )
            regime_str = (
                f" regime={self._latest_regime_label.cell}"
                if self._latest_regime_label is not None else ""
            )
            logger.info(
                f"[TF SIGNAL] {signal.label}{ctx} | {anchor_str}{vol_str}{regime_str} | "
                f"{signal.reason}"
            )

    def _process_bar_impl(self, bar: dict):
        """Override: Process bar with Big Money trend-only logic.

        Key differences from AlphaBot:
          - Checks trend confirmation before any trading
          - Only passes trend-aligned signals through
          - Forces 10 MES contracts on confirmed trends
          - Tracks consecutive losses for daily stop
          - Implements re-entry after profitable exits
        """
        # === Same setup as parent ===
        bar_ts = bar.get("t") or bar.get("time") or bar.get("timestamp")
        is_new_candle = (bar_ts != self._last_bar_timestamp) if bar_ts else True
        if bar_ts:
            self._last_bar_timestamp = bar_ts
        
        self._bar_count += 1
        if is_new_candle:
            self._live_bar_count += 1
            # Record bar to CSV for replay/backtesting
            if hasattr(self, 'bar_recorder') and self.bar_recorder:
                self.bar_recorder.record(bar)
        self.features.add_bar(bar)
        
        # ── Feed ML provider with 5-min aggregated bars ──
        if self._ml_provider and self._ml_enabled:
            self._ml_bar_accumulator.append(bar)
            # Every 5 bars (5 minutes), aggregate and feed to ML
            if len(self._ml_bar_accumulator) >= 5:
                chunk = self._ml_bar_accumulator[-5:]
                ml_bar = {
                    't': chunk[-1].get('t', chunk[-1].get('time', '')),
                    'o': chunk[0].get('o', chunk[0].get('open', 0)),
                    'h': max(b.get('h', b.get('high', 0)) for b in chunk),
                    'l': min(b.get('l', b.get('low', 0)) for b in chunk),
                    'c': chunk[-1].get('c', chunk[-1].get('close', 0)),
                    'v': sum(b.get('v', b.get('volume', 0)) for b in chunk),
                }
                self._ml_provider.update_bar(ml_bar)
                self._ml_bar_accumulator = []
                self._ml_bar_count += 1
        
        if self.features.bar_count < 20:
            return
        
        # Warmup gate
        if self._live_bar_count <= self._warmup_bars:
            if is_new_candle:
                warmup_elapsed = time.time() - self._startup_time if self._startup_time > 0 else 0
                logger.info(
                    f"[WARMUP] Fresh candle {self._live_bar_count}/{self._warmup_bars} "
                    f"({warmup_elapsed:.0f}s since start)"
                )
            return
        
        # Build market state
        from datetime import datetime as dt
        try:
            ts = bar.get("t", "")
            if isinstance(ts, str) and ts:
                bar_dt = dt.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                bar_dt = dt.now(__import__('datetime').timezone.utc)
        except:
            bar_dt = dt.now(__import__('datetime').timezone.utc)
        
        from core.market_data import detect_session, minutes_since_rth_open
        session = detect_session(bar_dt)

        # ── New Globex session reset ───────────────────────────────────────
        # CME maintenance break ends at ~5 PM ET daily. When we cross from
        # US_CLOSE into OVERNIGHT (4 PM ET), reset the TQF rolling window and
        # consecutive loss counter so morning chop doesn't block evening trades.
        prev_session = getattr(self, '_current_session', None)
        if prev_session in ('US_CLOSE', 'NY_OPEN', 'US_MIDDAY') and session == 'OVERNIGHT':
            if self._tqf_recent_pnls or self._bm_consecutive_losses > 0:
                logger.warning(
                    f"[SESSION RESET] RTH→Overnight: clearing TQF window "
                    f"({len(self._tqf_recent_pnls)} trades, "
                    f"{self._bm_consecutive_losses} consec losses) — fresh session"
                )
            self._tqf_recent_pnls = []
            self._bm_consecutive_losses = 0
            self._dynamic_cooldown.on_session_reset()

        self._current_session = session
        mins = minutes_since_rth_open(bar_dt)
        
        state_dict = self.features.build_market_state(session, mins)
        state = MarketState(**state_dict)
        self._last_bar_state = state
        # Mark that this bar's state is valid; consumed by the shadow hook
        # in the wrapper's finally block (so every closed bar logs fresh).
        self._shadow_this_bar_state = state
        
        # Update ATR
        if state.atr_14 > 0:
            self._alpha_atr = state.atr_14
        
        if self._alpha_open_price is None:
            self._alpha_open_price = state.price
        
        # Update regime
        if is_new_candle:
            self._current_regime = self.regime_detector.detect(self.features, state)
            # Track slope_75 history for entry location filter
            self._bm_slope_75_prev = getattr(self, '_bm_slope_75_current', 0.0)
            self._bm_slope_75_current = state.slope_75
            # Directional regime (7-state, from slopes + VWAP)
            self._directional_regime = classify_directional_regime(
                slope_12=state.slope_12,
                slope_75=state.slope_75,
                price=state.price,
                vwap=state.vwap,
                atr=state.atr_14,
                atr_median=state.atr_median,
            )
        
        # === HMM REGIME UPDATE — DISABLED 2026-03-31 ===
        # HMM was degenerate (stuck VOLATILITY_SPIKE 100%), removed.
        
        # Update parent's trend day detector
        atr_pctl = self._current_regime.atr_percentile if self._current_regime else 0.5
        self.trend_day.update(
            current_price=state.price,
            vwap=state.vwap,
            atr_percentile=atr_pctl,
            adx=state.adx_14,
            open_price=self._alpha_open_price,
        )
        
        # === BIG MONEY: Trend Detection (with hysteresis) ===
        if is_new_candle:
            was_confirmed = self._bm_trend_confirmed
            raw_trend, raw_direction = self._detect_trend(state)
            
            if raw_trend and raw_direction:
                if raw_direction == self._bm_trend_pending_direction:
                    self._bm_trend_confirm_count += 1
                else:
                    # Direction changed — reset counter
                    self._bm_trend_pending_direction = raw_direction
                    self._bm_trend_confirm_count = 1
                
                # Already confirmed in this direction — stay confirmed
                if self._bm_trend_confirmed and self._bm_trend_direction == raw_direction:
                    pass  # maintain confirmation
                # Strong ADX (smoothed 3m > 30) + HIGH conviction = instant confirmation
                # Requires probe_mult=1.0 (HIGH conviction) to prevent residual high-ADX
                # from a prior trend from instantly confirming a weak opposite-direction signal.
                # MEDIUM conviction (probe_mult=0.5) still needs 2 bars to confirm.
                elif (hasattr(self, '_adx3m_smooth') and self._adx3m_smooth >= 30.0
                      and self._me_probe_mult >= 1.0):
                    self._bm_trend_confirmed = True
                    self._bm_trend_direction = raw_direction
                    self._bm_trend_confirm_count = self._bm_trend_confirm_required
                    logger.warning(
                        f"BIG MONEY: TREND CONFIRMED (STRONG ADX+HIGH) — {raw_direction.name} | "
                        f"ADX3m(smooth)={self._adx3m_smooth:.1f} >= 30 | ADX3m(raw)={state.adx_3m:.1f} | "
                        f"ADX1m={state.adx_14:.1f} | ME probe={self._me_probe_mult:.1f}x"
                    )
                # FAST_CONFIRM (2026-05-01) — instant confirmation on impulse bars
                # below the ADX≥30 threshold. Trades the lag-vs-quality tradeoff:
                # accept lower ADX (≥22) BUT require explicit impulse evidence
                # (ER spike + decisive bar body + price still within VWAP reach).
                # Designed to catch the bar that LAUNCHES a move rather than
                # the one that confirms it after exhaustion.
                #
                # 2026-05-08 — Path A relaxation on trend days. When the
                # trend_day detector is active AND fast_confirm_relax_on_trend_day
                # is enabled, the impulse gates loosen because trend-day
                # continuations are by definition extended directional moves
                # — the chase-veto and bar-by-bar ER-rise requirements veto
                # legitimate continuation entries. ADX + body_ratio + probe_mult
                # gates still apply to keep quality up.
                elif (self.bm_config.fast_confirm_enabled
                      and self._me_probe_mult >= 1.0
                      and hasattr(self, '_adx3m_smooth')
                      and self._adx3m_smooth >= self.bm_config.fast_confirm_min_adx):
                    # Impulse signature: ER rose by enough THIS bar (not just plateau-high)
                    er_delta = self._me_er - self._me_er_prev
                    # Bar body / range — decisive directional candle, not a doji or wick
                    bar_open = float(bar.get("o", bar.get("open", state.price)))
                    bar_high = float(bar.get("h", bar.get("high", state.price)))
                    bar_low  = float(bar.get("l", bar.get("low",  state.price)))
                    bar_range = max(bar_high - bar_low, 0.001)
                    bar_body  = abs(state.price - bar_open)
                    body_ratio = bar_body / bar_range if bar_range > 0 else 0.0
                    # Distance from VWAP — don't fast-fire on already-extended moves
                    fc_atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                    dist_vwap_atr = (
                        abs(state.price - state.vwap) / fc_atr if fc_atr > 0 else 999.0
                    )

                    # Path A — choose threshold tier based on trend_day status.
                    # Use relaxed thresholds ONLY when trend_day is active AND
                    # the relaxation flag is enabled. Otherwise use the
                    # standard impulse-bar thresholds.
                    is_trend_day = bool(
                        self.bm_config.fast_confirm_relax_on_trend_day
                        and getattr(self, 'trend_day', None) is not None
                        and getattr(self.trend_day, 'is_trend_day', False)
                    )
                    if is_trend_day:
                        threshold_er_delta = self.bm_config.fast_confirm_min_er_delta_trend_day
                        threshold_vwap_atr = self.bm_config.fast_confirm_max_vwap_atr_trend_day
                        path_label = "FAST [TD]"   # diagnostic tag for log review
                    else:
                        threshold_er_delta = self.bm_config.fast_confirm_min_er_delta
                        threshold_vwap_atr = self.bm_config.fast_confirm_max_vwap_atr
                        path_label = "FAST"

                    # All impulse gates must pass with whichever threshold tier applies
                    if (er_delta >= threshold_er_delta
                            and body_ratio >= self.bm_config.fast_confirm_min_body_ratio
                            and dist_vwap_atr <= threshold_vwap_atr):
                        self._bm_trend_confirmed = True
                        self._bm_trend_direction = raw_direction
                        self._bm_trend_confirm_count = self._bm_trend_confirm_required
                        logger.warning(
                            f"BIG MONEY: TREND CONFIRMED ({path_label}) — {raw_direction.name} | "
                            f"ADX={self._adx3m_smooth:.1f} (>= {self.bm_config.fast_confirm_min_adx}) | "
                            f"ER_delta={er_delta:+.3f} (>= {threshold_er_delta}) | "
                            f"body={body_ratio:.2f} (>= {self.bm_config.fast_confirm_min_body_ratio}) | "
                            f"VWAP={dist_vwap_atr:.2f}xATR (<= {threshold_vwap_atr}) | "
                            f"ME probe={self._me_probe_mult:.1f}x"
                        )
                    elif is_new_candle:
                        # Diagnostic: log which fast_confirm gate failed so we can tune.
                        # Tags whether the bar was evaluated under standard or
                        # trend-day-relaxed thresholds — Path B back-test will
                        # use this to compute relaxation impact.
                        logger.info(
                            f"[{path_label} CONFIRM] {raw_direction.name} rejected — "
                            f"ER_delta={er_delta:+.3f}{'✓' if er_delta >= threshold_er_delta else '✗'}"
                            f"({threshold_er_delta}) "
                            f"body={body_ratio:.2f}{'✓' if body_ratio >= self.bm_config.fast_confirm_min_body_ratio else '✗'} "
                            f"VWAP={dist_vwap_atr:.2f}xATR{'✓' if dist_vwap_atr <= threshold_vwap_atr else '✗'}"
                            f"({threshold_vwap_atr})"
                        )
                # Need N consecutive bars to confirm NEW trend
                elif self._bm_trend_confirm_count >= self._bm_trend_confirm_required:
                    self._bm_trend_confirmed = True
                    self._bm_trend_direction = raw_direction
                    logger.warning(
                        f"BIG MONEY: TREND CONFIRMED — {raw_direction.name} | "
                        f"ADX1m={state.adx_14:.1f} ADX3m(smooth)={self._adx3m_smooth:.1f} | "
                        f"Confirmed after {self._bm_trend_confirm_count} bars | "
                        f"ME probe={self._me_probe_mult:.1f}x"
                    )
                else:
                    if is_new_candle and self._bm_trend_confirm_count == 1:
                        logger.info(
                            f"💰 BIG MONEY: Trend pending {raw_direction.name} "
                            f"({self._bm_trend_confirm_count}/{self._bm_trend_confirm_required} bars)"
                        )
            else:
                # No trend detected this bar
                if self._bm_trend_confirmed:
                    # Don't immediately drop confirmation — require sustained loss
                    self._bm_trend_confirm_count = max(0, self._bm_trend_confirm_count - 1)
                    if self._bm_trend_confirm_count <= 0:
                        self._bm_trend_confirmed = False
                        logger.info("💰 BIG MONEY: Trend conditions lost — waiting for re-confirmation")
                    else:
                        logger.debug(
                            f"💰 BIG MONEY: Trend weakening ({self._bm_trend_confirm_count} bars remain)"
                        )
                else:
                    self._bm_trend_confirm_count = 0
                    self._bm_trend_pending_direction = None
        
        # Daily dollar loss limit — done for day
        if self._bm_done_for_day:
            if is_new_candle and self._live_bar_count % 60 == 0:
                logger.info(f"💰 BIG MONEY: Done for day (P&L: ${self._bm_total_pnl:+,.2f})")
            return
        
        if self._bm_total_pnl <= self.bm_config.daily_loss_limit:
            self._bm_done_for_day = True
            logger.warning(
                f"🛑 BIG MONEY: DONE FOR DAY — P&L ${self._bm_total_pnl:+,.2f} "
                f"hit limit ${self.bm_config.daily_loss_limit:+,.2f}"
            )
            return
        
        # === If in position, manage it (TRANCHE STATE MACHINE) ===
        if self.position_direction and self.position_direction != Direction.FLAT:
            idea = self._active_idea
            
            if idea and idea.is_active:
                # ── Tranche-managed position ────────────────────────
                unreal_pts = idea.unrealized_pts(state.price)
                remaining = idea.remaining_contracts
                unreal_dollars = unreal_pts * self.instrument.point_value * remaining
                
                # Track worst/best unrealized
                if not hasattr(self, '_position_worst_unreal'):
                    self._position_worst_unreal = 0.0
                if not hasattr(self, '_position_best_unreal'):
                    self._position_best_unreal = 0.0
                self._position_worst_unreal = min(self._position_worst_unreal, unreal_dollars)
                self._position_best_unreal = max(self._position_best_unreal, unreal_dollars)
                
                # Periodic log
                position_age = time.time() - self.position_entry_time if self.position_entry_time else 0
                if is_new_candle or (hasattr(self, '_last_unreal_log') and time.time() - self._last_unreal_log >= 30):
                    logger.info(
                        f"💰 [TRANCHE] {idea.direction.name} {remaining}x ({idea.state.name}) | "
                        f"Avg: {idea.avg_entry_price:.2f} | "
                        f"Unrealized: ${unreal_dollars:+,.0f} ({unreal_pts:+.2f}pts) | "
                        f"1R={idea.r_value:.2f}pts | "
                        f"Best: ${self._position_best_unreal:+,.0f} | Worst: ${self._position_worst_unreal:+,.0f}"
                    )
                    self._last_unreal_log = time.time()
                
                # ── LIL Emergency Override (velocity bleedout, extreme decay) ──
                hard_stop_pts = self._alpha_atr * self.bm_config.stop_atr_mult
                should_exit, reason, phase = self._lil.check(
                    unrealized_pts=unreal_pts,
                    unrealized_dollars=unreal_dollars,
                    atr=self._alpha_atr,
                    session=self._current_session or "US_MIDDAY",
                    hard_stop_pts=hard_stop_pts,
                )
                if should_exit:
                    graduated_tag = "GRAD" if self._lil.is_graduated else "UNGRAD"
                    logger.warning(f"🧠💰 [LIL-{graduated_tag}] Emergency exit: {reason}")
                    idea.force_close(f"LIL:{reason}")
                    self._tranche_close_all(state.price, f"LIL:{reason}")
                    return

                # ── Breakeven ratchet on LIL graduation ───────────────────
                # When LIL graduates (trade proved itself with >= 1.5x ATR profit),
                # move the hard bracket stop to entry price so we can't lose on a
                # winner. Fires once per trade. Uses place_stop_order() to replace
                # the original bracket with a new stop at avg_entry_price.
                # 2026-04-30: gated on `be_ratchet_enabled` (default False) —
                # was firing pre-1R and killing winners on shallow pullbacks.
                # Post-1R BE stop in tranche state machine is the structural
                # equivalent and runs as part of the standard exit ladder.
                if (self.bm_config.be_ratchet_enabled
                        and self._lil.is_graduated
                        and not getattr(self, '_bm_be_ratchet_done', False)
                        and idea.avg_entry_price
                        and remaining > 0):
                    # Stop at entry - 1.0×ATR: gives the trade room to breathe after
                    # graduation. 0.5×ATR was too tight — normal pullbacks stopped out
                    # winning trades immediately after the ratchet fired.
                    atr_now = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                    noise_buffer = 1.0 * atr_now
                    if self.position_direction == Direction.LONG:
                        ratchet_price = idea.avg_entry_price - noise_buffer
                        stop_side = "SELL"
                    else:
                        ratchet_price = idea.avg_entry_price + noise_buffer
                        stop_side = "BUY"
                    # Round to nearest tick (0.25)
                    ratchet_price = round(ratchet_price / 0.25) * 0.25
                    # Cancel original bracket before placing ratchet stop —
                    # otherwise both stops are live and the bracket orphans after ratchet fills.
                    self.conn._cancel_all_working_orders(reason="BE ratchet replacing bracket")
                    ratchet_ok = self.conn.place_stop_order(
                        contract_id=self.config.topstep.contract_id,
                        side=stop_side,
                        size=remaining,
                        stop_price=ratchet_price,
                    )
                    peak = self._lil.state.peak_mfe_pts
                    if ratchet_ok:
                        self._bm_be_ratchet_done = True  # only lock once placement succeeds
                        logger.warning(
                            f"[BE RATCHET] Stop moved to {ratchet_price:.2f} "
                            f"(entry {idea.avg_entry_price:.2f} ± 1.0xATR {noise_buffer:.2f}pts) "
                            f"| peak was +{peak:.2f}pts | {remaining}x contracts"
                        )
                    else:
                        logger.warning(
                            f"[BE RATCHET] Stop placement FAILED — will retry next bar"
                        )

                # ── Check exits FIRST (C7: exit-before-add) ───────────
                # If price has already reached 1R / invalidation / trail this
                # bar, take those exits before any scale-in check. Scale-in
                # transitions state → REDUCING after a 1R partial, which
                # naturally blocks T2/T3 below. Pre-C7 the order was reversed,
                # so T3 could add at the 1R peak and round-trip (2026-04-22).

                # C9: Feed current regime into the idea so check_exits →
                # _update_runner_trail picks the regime-conditional trail.
                idea.current_regime = self._bm_market_mode or "UNKNOWN"

                exit_actions = idea.check_exits(state.price)
                for action in exit_actions:
                    exit_contracts = action["contracts"]
                    exit_reason = action["reason"]
                    exit_type = action.get("exit_type", "UNKNOWN")

                    if exit_contracts >= remaining:
                        # Full close — P&L for final contracts flows through
                        # _exit_position → risk.record_trade. Don't double-feed
                        # here or we'd count the final contracts twice.
                        idea.record_exit(remaining, state.price, exit_reason)
                        self._tranche_close_all(state.price, exit_reason)
                    else:
                        # Partial close — feed realized P&L to risk.daily_pnl
                        # right away (A3: single source of truth). Remaining
                        # contracts stay open; the final close will record_trade.
                        self._tranche_partial_exit(exit_contracts, state.price, exit_reason)
                        exit_rec = idea.record_exit(exit_contracts, state.price, exit_reason)
                        self.risk.add_tranche_pnl(exit_rec.pnl_dollars, label=exit_reason)

                # If exits fully closed the idea, skip regime-change + scale-in
                if idea.state == TrancheState.CLOSED and self.position_direction != Direction.FLAT:
                    self._tranche_close_all(state.price, "IDEA_CLOSED")
                    return

                # ── Regime-change exit (primary runner exit) ──────────
                # Once the runner is live, exit on 2 consecutive CHOP bars.
                # self._bm_market_mode holds the PREVIOUS bar's confirmed regime —
                # correct: we act on what the last closed bar proved.
                if is_new_candle and idea._runner_active:
                    if self._bm_market_mode == "CHOP":
                        self._bm_chop_while_running += 1
                    else:
                        self._bm_chop_while_running = 0

                    if self._bm_chop_while_running >= 2:
                        logger.warning(
                            f"🔴 [REGIME EXIT] {self._bm_chop_while_running} consecutive CHOP bars "
                            f"— trend confirmed dead. Exiting runner @ {state.price:.2f}"
                        )
                        idea.force_close("REGIME_CHOP")
                        self._tranche_close_all(state.price, "REGIME_CHOP_EXIT")
                        self._bm_chop_while_running = 0
                        return

                # ── T2/T3 Scale-In (AFTER exits — C7) ────────────────
                # Post-1R the state is REDUCING and these no-ops; pre-1R they
                # add only if their own conditions fire. Skip if session ending
                # because a T2/T3 add increases risk and needs fill time.
                if not self._bm_session_ending:
                    if idea.state == TrancheState.PROBE and is_new_candle:
                        if idea.should_add_t2(state.price):
                            self._tranche_add_contracts(idea, "T2", idea.t2_size, state)

                    if idea.state == TrancheState.BUILDING and is_new_candle:
                        if idea.should_add_t3(state.price):
                            self._tranche_add_contracts(idea, "T3", idea.t3_size, state)
                
                return
            
            else:
                # Legacy: non-tranche position (shouldn't happen after refactor)
                if self.position_entry_price and self._alpha_atr > 0:
                    mult = 1 if self.position_direction == Direction.LONG else -1
                    unreal_pts = (state.price - self.position_entry_price) * mult
                    contracts = getattr(self, 'position_contracts', 1)
                    unreal_dollars = unreal_pts * self.instrument.point_value * contracts
                    hard_stop_pts = self._alpha_atr * self.bm_config.stop_atr_mult
                    
                    should_exit, reason, phase = self._lil.check(
                        unrealized_pts=unreal_pts,
                        unrealized_dollars=unreal_dollars,
                        atr=self._alpha_atr,
                        session=self._current_session or "US_MIDDAY",
                        hard_stop_pts=hard_stop_pts,
                    )
                    if should_exit:
                        self._alpha_exit_position(state.price, f"LIL:{reason}")
                        return
                
                if self._alpha_exit_mgr:
                    self._alpha_check_exit(state)
                else:
                    self._check_exit(state)
                return
        
        if self._entering:
            return
        
        # === Not in position — look for entries ===
        
        # ── Feed ADX Range Mapper on every bar ──────────────────────
        if self._range_mapper and is_new_candle:
            adx_for_mapper = self._adx3m_smooth if hasattr(self, '_adx3m_smooth') else (state.adx_3m if state.adx_3m > 0 else state.adx_14)
            atr_for_mapper = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
            breakout = self._range_mapper.update(
                price=state.price,
                high=state.bar_high if hasattr(state, 'bar_high') else state.price,
                low=state.bar_low if hasattr(state, 'bar_low') else state.price,
                adx=adx_for_mapper,
                atr=atr_for_mapper,
            )
            if breakout:
                self._pending_breakout = breakout
                logger.warning(
                    f"📦 RANGE BREAKOUT: {breakout.direction} @ {breakout.breakout_price:.2f} | "
                    f"Range [{breakout.range_low:.2f}-{breakout.range_high:.2f}] "
                    f"({breakout.range_width:.1f}pts, {breakout.range_duration_bars} bars) | "
                    f"Strength: {breakout.strength:.2f}"
                )
        
        # Determine market mode — ADX3m smoothed + slope
        range_state = self._range_state  # from parent AlphaBot
        
        # Smooth ADX3m with EMA(3) to stop bar-to-bar flip-flopping (17→37→17)
        raw_adx = state.adx_3m if state.adx_3m > 0 else state.adx_14
        if not hasattr(self, '_adx3m_smooth'):
            self._adx3m_smooth = raw_adx
        else:
            alpha = 0.4  # responsive but stable — reacts in ~3 bars
            self._adx3m_smooth = alpha * raw_adx + (1 - alpha) * self._adx3m_smooth
        adx = self._adx3m_smooth
        
        if range_state:
            range_width = range_state.range_width
        else:
            range_width = 0
        
        # === REGIME CLASSIFICATION (ER-primary, 2026-04-18) ===
        # ERRegimeClassifier uses Efficiency Ratio as the primary gate.
        # ER resets in 8 bars; ADX lags 28+ bars after a trend ends.
        # This eliminates the overnight-chop-with-high-ADX false entries.
        market_mode = self._er_regime_clf.classify(
            er        = self._me_er,
            adx       = adx,             # already EMA-smoothed
            slope_bps = state.slope_75,  # 20-bar slope in bps/bar
        )
        self._bm_market_mode = market_mode  # Store for logging

        # Feed DynamicCooldown regime history and streak price tracking
        if is_new_candle:
            self._dynamic_cooldown.on_bar(market_mode, state.price)

        # Refresh chart levels from ChartCapture every 30 min
        if is_new_candle and (time.time() - self._chart_levels_ts) > 1800:
            try:
                from analyzers.chart_capture import get_latest_levels
                lvl = get_latest_levels()
                if lvl:
                    self._chart_support    = lvl.get("support", [])
                    self._chart_resistance = lvl.get("resistance", [])
                    self._chart_levels_regime = lvl.get("regime", "")
                    self._chart_levels_ts  = time.time()
                    logger.info(
                        f"[CHART LEVELS] support={self._chart_support} "
                        f"resistance={self._chart_resistance} "
                        f"regime={self._chart_levels_regime}"
                    )
            except Exception:
                pass
        
        # ── Session-end gate (TopstepX platform closes at 14:10 local) ──
        # SKIPPED for 24/7 crypto markets (Alpaca BTC) — see run_btc.py setting
        # `bot._crypto_24_7 = True`. ES/MES futures still run the original gates.
        if not getattr(self, "_crypto_24_7", False):
            # Bot runs 16:00 → next-day 14:15 (crosses midnight). The gates must
            # ONLY fire in the afternoon closing window, not whenever the clock
            # is past 14:15 (which is true all evening too).
            # 13:55-14:14 → stop new entries (blackout window).
            # 14:10 → platform auto-flattens open positions.
            # 14:15-14:59 → bot self-shutdown (constrained to the 14:xx hour).
            from datetime import datetime as _sess_dt
            _now_local = _sess_dt.now()
            _hm = (_now_local.hour, _now_local.minute)

            if _now_local.hour == 14 and _now_local.minute >= 15 and not self._bm_session_end_shutdown_fired:
                self._bm_session_end_shutdown_fired = True
                logger.warning(
                    f"[SESSION END] 14:15 local reached — self-shutdown initiated. "
                    f"Platform should have flattened at 14:10."
                )
                self.running = False
                return

            _in_blackout = (13, 55) <= _hm < (14, 15)
            if _in_blackout and not self._bm_session_ending:
                self._bm_session_ending = True
                logger.warning(
                    f"[SESSION END] 13:55 local reached — blocking new entries and T2/T3 scale-ins. "
                    f"Existing positions continue until platform close at 14:10."
                )

        # ── Warmup Gate: collect data but block entries ──
        if not self._bm_warmup_complete:
            elapsed_min = (time.time() - self._bm_startup_time) / 60.0
            if elapsed_min < self.bm_config.warmup_minutes:
                if is_new_candle and self._live_bar_count % 10 == 0:
                    remaining = self.bm_config.warmup_minutes - elapsed_min
                    logger.info(
                        f"🔥 WARMUP: {elapsed_min:.0f}/{self.bm_config.warmup_minutes}min "
                        f"({remaining:.0f}min left) | ADX={adx:.1f} ATR={self._alpha_atr:.1f} "
                        f"Mode={market_mode} | Collecting data, no entries"
                    )
                return
            else:
                self._bm_warmup_complete = True
                logger.warning(
                    f"✅ WARMUP COMPLETE — {self.bm_config.warmup_minutes}min elapsed | "
                    f"ADX={adx:.1f} ATR={self._alpha_atr:.1f} Mode={market_mode} | "
                    f"Entries now ENABLED"
                )
        
        # Extract bar OHLC for GRIND structural stop tracking
        _bar_high = float(bar.get("h", bar.get("high",  state.price)))
        _bar_low  = float(bar.get("l", bar.get("low",   state.price)))

        # Block all new entries during end-of-session window.
        # Position management above this point continues normally.
        if self._bm_session_ending:
            return

        # Route by market mode
        if market_mode == "CHOP":
            # Hard gate — ER < CHOP_ER_MAX means no strategy has statistical edge.
            # No mean-reversion either: in true chop both directions fail equally.
            if is_new_candle and self._live_bar_count % 30 == 0:
                logger.info(
                    f"[CHOP] No trade | ER={self._me_er:.3f} ADX={adx:.1f} "
                    f"slope={state.slope_75:+.1f}bps"
                )
            return

        elif market_mode == "GRIND":
            # Slow directional drift — EMA pullback entries, small size.
            # If trend is fully confirmed, fall through to DIRECT ENTRY for bigger size.
            if self._bm_trend_confirmed and self._bm_trend_direction:
                pass  # fall through — confirmed trend trumps grind
            else:
                if is_new_candle:
                    self._bm_grind_trade(state, _bar_high, _bar_low)
                return

        elif market_mode == "RANGE":
            # Oscillating market — fade extremes at S/R levels or VWAP.
            # If trend is confirmed, fall through to DIRECT ENTRY.
            if self._bm_trend_confirmed and self._bm_trend_direction:
                pass  # fall through
            else:
                if is_new_candle:
                    self._bm_range_trade(state, range_state)
                return

        if market_mode in ("TREND", "GRIND", "RANGE"):
            # Keep existing trend logic but make sure we have confirmed trend
            if not self._bm_trend_confirmed or not self._bm_trend_direction:
                # ── Short Override (Phase 8.8 port) ─────────────────────
                # Even without a confirmed trend, inject a SHORT in strong downtrends
                if self.bm_config.short_override_enabled and is_new_candle:
                    override_ok, override_reason = check_short_override(
                        regime=self._directional_regime,
                        price=state.price,
                        vwap=state.vwap,
                        atr=self._alpha_atr if self._alpha_atr > 0 else state.atr_14,
                        slope_75_bps=state.slope_75,
                        rsi_14=state.rsi_14,
                    )
                    if override_ok:
                        # Full loss cooldown blocks same-direction re-entry
                        if (self._bm_full_loss_cooldown_until > 0
                                and time.time() < self._bm_full_loss_cooldown_until
                                and self._bm_full_loss_cooldown_direction == Direction.SHORT):
                            remaining = self._bm_full_loss_cooldown_until - time.time()
                            logger.info(f"🛑 SHORT OVERRIDE blocked by FULL LOSS COOLDOWN ({remaining:.0f}s)")
                        # Consecutive loss cooldown (2026-03-25)
                        elif hasattr(self, '_bm_consec_loss_cooldown_until') and time.time() < self._bm_consec_loss_cooldown_until:
                            remaining = self._bm_consec_loss_cooldown_until - time.time()
                            logger.info(f"🛑 SHORT OVERRIDE blocked by CONSEC LOSS COOLDOWN ({remaining:.0f}s)")
                        # Exhaustion check
                        elif self._check_exhaustion(state, Direction.SHORT)[0]:
                            _, reason = self._check_exhaustion(state, Direction.SHORT)
                            logger.info(f"🔥 SHORT OVERRIDE blocked: {reason}")
                        # Short override cooldown
                        elif self._bm_last_short_override_exit > 0 and (time.time() - self._bm_last_short_override_exit) < self.bm_config.short_override_cooldown_seconds:
                            elapsed = time.time() - self._bm_last_short_override_exit
                            logger.info(f"🔻 SHORT OVERRIDE: Cooldown {elapsed:.0f}s / {self.bm_config.short_override_cooldown_seconds:.0f}s")
                        else:
                            # Entry location filter for short override
                            if self.bm_config.entry_location_filter_enabled:
                                atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                                loc_ok, loc_reason = check_entry_location(
                                    direction="SHORT",
                                    price=state.price,
                                    vwap=state.vwap,
                                    atr=atr,
                                    rsi_14=state.rsi_14,
                                    slope_75=state.slope_75,
                                    slope_75_prev=self._bm_slope_75_prev,
                                    high_20=getattr(state, 'high_20', 0.0),
                                    low_20=getattr(state, 'low_20', 0.0),
                                    efficiency_ratio=getattr(state, 'efficiency_ratio', 1.0),
                                    session=self._current_session or 'US_MIDDAY',
                                )
                                if not loc_ok:
                                    logger.info(f"📍 SHORT OVERRIDE blocked: {loc_reason}")
                                else:
                                    logger.warning(f"🔻 {override_reason}")
                                    self._bm_short_override_enter(state)
                                    if self.position_direction and self.position_direction != Direction.FLAT:
                                        return
                            else:
                                logger.warning(f"🔻 {override_reason}")
                                self._bm_short_override_enter(state)
                                if self.position_direction and self.position_direction != Direction.FLAT:
                                    return
                
                # ── ADX Range Mapper Breakout Entry ─────────────────────
                if (self._pending_breakout and self.bm_config.range_mapper_enabled
                        and is_new_candle):
                    breakout = self._pending_breakout
                    if breakout.strength >= self.bm_config.range_mapper_min_strength:
                        # Check cooldowns
                        direction = Direction.LONG if breakout.direction == "LONG" else Direction.SHORT
                        cooldown_blocked = (
                            self._bm_full_loss_cooldown_until > 0
                            and time.time() < self._bm_full_loss_cooldown_until
                            and self._bm_full_loss_cooldown_direction == direction
                        )
                        consec_loss_blocked = (
                            hasattr(self, '_bm_consec_loss_cooldown_until')
                            and time.time() < self._bm_consec_loss_cooldown_until
                        )
                        if cooldown_blocked:
                            remaining = self._bm_full_loss_cooldown_until - time.time()
                            logger.info(f"📦 RANGE BREAKOUT blocked by FULL LOSS COOLDOWN ({remaining:.0f}s)")
                        elif consec_loss_blocked:
                            remaining = self._bm_consec_loss_cooldown_until - time.time()
                            logger.info(f"📦 RANGE BREAKOUT blocked by CONSEC LOSS COOLDOWN ({remaining:.0f}s)")
                        else:
                            # TQF gate for range breakouts
                            atr_rb = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                            rb_dir = "LONG" if direction == Direction.LONG else "SHORT"
                            tqf_ok, _, tqf_reason = self._tqf.check(
                                direction=rb_dir,
                                closes=self.features.get_closes(25),
                                price=state.price,
                                vwap=state.vwap,
                                ema_9=state.ema_9,
                                atr=atr_rb,
                                recent_pnls=self._tqf_recent_pnls,
                                consecutive_losses=self._bm_consecutive_losses,
                                normal_probe_size=2,
                            )
                            if not tqf_ok:
                                logger.info(f"[TQF] RANGE BREAKOUT BLOCKED: {tqf_reason}")
                            else:
                                logger.warning(
                                    f"📦 RANGE BREAKOUT ENTRY: {breakout.direction} @ {state.price:.2f} | "
                                    f"Strength: {breakout.strength:.2f}"
                                )
                                self._bm_range_breakout_enter(breakout, state)
                            if self.position_direction and self.position_direction != Direction.FLAT:
                                self._pending_breakout = None
                                return
                    self._pending_breakout = None  # consume signal whether we traded or not
                
                if is_new_candle and self._live_bar_count % 30 == 0:
                    range_info = ""
                    if self._range_mapper:
                        rng = self._range_mapper.get_current_range()
                        if rng:
                            range_info = (f" | Range: [{rng['low']:.2f}-{rng['high']:.2f}] "
                                          f"({rng['width']:.1f}pts, {rng['bars']}bars, {rng['phase']})")
                    logger.info(
                        f"💰 BIG MONEY: TREND mode - Waiting for trend confirmation | "
                        f"ADX1m={state.adx_14:.1f} ADX3m(smooth)={self._adx3m_smooth:.1f} | "
                        f"Regime={self._directional_regime.value} | "
                        f"S75={state.slope_75:+.1f} S12={state.slope_12:+.1f} | "
                        f"Price vs VWAP={'above' if state.price > state.vwap else 'below'}"
                        f"{range_info}"
                    )
                return
        
        # ── MINIMUM ATR GATE — Don't trade dead markets ──────────
        # EXCEPTION: High ADX (>30) means strong trend — trade even with low ATR
        current_atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
        adx_for_gate = self._adx3m_smooth if hasattr(self, '_adx3m_smooth') else (state.adx_3m if state.adx_3m > 0 else state.adx_14)
        if current_atr < self.bm_config.min_atr_to_trade and adx_for_gate < 30:
            if is_new_candle and self._live_bar_count % 30 == 0:
                logger.info(
                    f"💰 BIG MONEY: ATR TOO LOW — {current_atr:.2f}pts < "
                    f"{self.bm_config.min_atr_to_trade}pts minimum | ADX={adx_for_gate:.1f} < 30 | Dead market"
                )
            return
        
        # Re-entry cooldown
        if self._bm_last_exit_time > 0:
            elapsed = time.time() - self._bm_last_exit_time
            if elapsed < self.bm_config.re_entry_cooldown_seconds:
                return
        
        # Consecutive winner exhaustion cooldown — REMOVED 2026-04-03
        # Was punishing profitable streaks. Never throttle edge expression.
        
        # ── RISK GUARD 3a: Dynamic Cooldown Gate ─────────────────────
        # Market-state driven: requires regime reset after losses, pullback after win streaks.
        if self._bm_trend_direction:
            dc_direction = "LONG" if self._bm_trend_direction == Direction.LONG else "SHORT"
            current_atr_dc = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
            dc_ok, dc_reason = self._dynamic_cooldown.can_enter(
                direction=dc_direction,
                adx=self._adx3m_smooth,
                price=state.price,
                atr=current_atr_dc,
            )
            if not dc_ok:
                if is_new_candle and self._live_bar_count % 5 == 0:
                    logger.info(f"🛑 [DYNAMIC COOLDOWN] {dc_reason} | Status: {self._dynamic_cooldown.status}")
                return
        
        # ── RISK GUARD 3b: Full Loss Cooldown ────────────────────────
        if (self._bm_full_loss_cooldown_until > 0
                and time.time() < self._bm_full_loss_cooldown_until
                and self._bm_trend_direction == self._bm_full_loss_cooldown_direction):
            if is_new_candle and self._live_bar_count % 10 == 0:
                remaining = self._bm_full_loss_cooldown_until - time.time()
                logger.info(
                    f"🛑 [FULL LOSS COOLDOWN] {self._bm_full_loss_cooldown_direction.name} "
                    f"blocked — {remaining:.0f}s remaining"
                )
            return
        
        # ── POST-COOLDOWN TREND RESET ───────────────────────────────
        # After full loss cooldown expires, force fresh trend re-evaluation
        # so the bot doesn't stay stuck in "Waiting for trend" forever
        if (self._bm_full_loss_cooldown_until > 0
                and time.time() >= self._bm_full_loss_cooldown_until
                and not self._bm_trend_confirmed):
            # Cooldown just expired and trend is unconfirmed — force re-detect
            self._bm_trend_confirmed, new_direction = self._detect_trend(state)
            if self._bm_trend_confirmed and new_direction:
                self._bm_trend_direction = new_direction
                logger.warning(
                    f"POST-COOLDOWN TREND RESET — {new_direction.name} | "
                    f"ADX={state.adx_14:.1f} | ADX3m(smooth)={self._adx3m_smooth:.1f} | "
                    f"ME probe={self._me_probe_mult:.1f}x"
                )
            # Clear the cooldown marker so this only fires once
            self._bm_full_loss_cooldown_until = 0
        
        # ── RISK GUARD 2: Exhaustion Detection ──────────────────────
        # 2026-03-31: Only block COUNTER-trend entries. With-trend entries pass.
        # Previously blocked ALL entries including with-trend, killing profitable setups.
        # Also raised min conditions from 3 to 4 (was too sensitive).
        if is_new_candle and self._bm_trend_direction:
            exhausted, exhaust_reason = self._check_exhaustion(state, self._bm_trend_direction)
            if exhausted:
                logger.info(f"🔥 {exhaust_reason} — WITH-TREND, allowing entry (counter-trend would be blocked)")
        
        # ── ADX FLOOR — REMOVED 2026-04-03 (duplicated ADX Hard Gate, compounded passivity) ──
        # ADX is context logged above, not an entry veto.
        
        # Pullback veto: block when price is on the wrong side of EMA9 for the trend direction.
        if self._bm_trend_confirmed and self._bm_trend_direction:
            if self._bm_trend_direction == Direction.LONG:
                side_ok = state.price > state.ema_9
            else:
                side_ok = state.price < state.ema_9
            if not side_ok:
                if is_new_candle:
                    logger.info(
                        f"[PULLBACK VETO] {self._bm_trend_direction.name} blocked: "
                        f"Price={state.price:.2f} on wrong side of EMA9={state.ema_9:.2f}"
                    )
                return
        
        # ── DIRECT ENTRY ─────────────────────────────────────────────────
        # When ADX is strong (>30) and trend just confirmed, don't wait for a strategy
        # signal — enter immediately with a synthetic TREND_FOLLOW signal.
        # Fast moves confirm and reverse before any strategy fires.
        # 30 matches adx_continuation_level in the engine — the floor for a real trend.
        if is_new_candle:
            logger.info(
                f"[ENTRY GATE] confirmed={self._bm_trend_confirmed} "
                f"dir={self._bm_trend_direction.name if self._bm_trend_direction else 'None'} "
                f"ADX_smooth={self._adx3m_smooth:.1f}(need>=30) "
                f"ME_probe={self._me_probe_mult:.1f} "
                f"entering={self._entering}"
            )
        if (is_new_candle
                and self._bm_trend_confirmed
                and self._bm_trend_direction
                and self._adx3m_smooth >= 30.0):
            try:
                atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                dist_vwap = abs(state.price - state.vwap)
                dist_ema9 = abs(state.price - state.ema_9)
                # VWAP chase veto: reject DIRECT ENTRY if price is >3×ATR from session VWAP.
                # Backtested 2026-04-20 (backtest_matrix.py B3): filters chase-into-resistance
                # entries while keeping legitimate trend continuations. TQF's OR-logic (VWAP or
                # EMA9) doesn't catch this because EMA9 tracks price in trends. +$537 over A1
                # on 19-day sample.
                # 2026-04-28: relaxed 3.0→5.0 after the 4/28 AM 40pt MES selloff
                # (price moved 8-10×ATR from VWAP while bot was vetoed every bar).
                # 3.0×ATR was calibrated for chase-entry rejection but it can't
                # distinguish "chasing froth" from "catching a real trending move"
                # — by the time ADX confirms, VWAP can't catch up fast enough on
                # sharp moves. 5.0×ATR still rejects extreme chase entries (8-10×)
                # but lets through genuine continuation entries during fast trends.
                if atr > 0 and dist_vwap > 5.0 * atr:
                    logger.info(
                        f"[DIRECT ENTRY VWAP VETO] {self._bm_trend_direction.name} blocked: "
                        f"price={state.price:.2f} is {dist_vwap/atr:.2f}×ATR from VWAP={state.vwap:.2f} "
                        f"(threshold=5.0×ATR={5.0*atr:.2f}pts, ATR={atr:.2f})"
                    )
                    return
                logger.info(
                    f"[DIRECT ENTRY] Checking TQF | price={state.price:.2f} "
                    f"VWAP={state.vwap:.2f}({dist_vwap:.1f}pts) EMA9={state.ema_9:.2f}({dist_ema9:.1f}pts) "
                    f"ATR={atr:.2f} threshold={3.0*atr:.1f}pts | consec_losses={self._bm_consecutive_losses}"
                )
                tqf_ok, tqf_probe, tqf_reason = self._tqf.check(
                    direction=self._bm_trend_direction.name,
                    closes=self.features.get_closes(25),
                    price=state.price,
                    vwap=state.vwap,
                    ema_9=state.ema_9,
                    atr=atr,
                    recent_pnls=self._tqf_recent_pnls,
                    consecutive_losses=self._bm_consecutive_losses,
                    normal_probe_size=2,
                    skip_location=True,  # DIRECT ENTRY fires on extended trend moves — location filter defeats the purpose
                )
                if tqf_ok:
                    from strategies.base import Signal
                    synthetic = Signal(
                        direction=self._bm_trend_direction,
                        strategy_name="TREND_DIRECT",
                        strength=0.8,
                        reason=f"ADX={self._adx3m_smooth:.1f} probe={self._me_probe_mult:.1f}x",
                        entry_price=state.price,
                        stop_loss=state.price + (atr * 2.5 if self._bm_trend_direction == Direction.SHORT else -atr * 2.5),
                        take_profit=state.price - (atr * 3.0 if self._bm_trend_direction == Direction.SHORT else -atr * 3.0),
                    )
                    self._tqf_probe_override = tqf_probe
                    logger.warning(
                        f"[DIRECT ENTRY] ADX={self._adx3m_smooth:.1f} >= 30 | "
                        f"{self._bm_trend_direction.name} @ {state.price:.2f} | {tqf_reason}"
                    )
                    self._bm_try_enter(synthetic, state)
                    if self.position_direction and self.position_direction != Direction.FLAT:
                        return
                else:
                    logger.info(f"[DIRECT ENTRY] TQF blocked: {tqf_reason}")
            except Exception as _de_exc:
                logger.error(f"[DIRECT ENTRY] EXCEPTION — entry aborted: {_de_exc}", exc_info=True)

        # Scan strategies for trend-aligned signals
        if True:
            for strategy in self.strategies:
                signal = strategy.should_enter(state)
                if signal is None:
                    continue
                
                # HARD FILTER: Only take signals in the trend direction
                if signal.direction != self._bm_trend_direction:
                    logger.debug(
                        f"💰 BIG MONEY: Blocked {signal.strategy_name} {signal.direction.name} -- "
                        f"trend is {self._bm_trend_direction.name}"
                    )
                    continue
                
                # REGIME ENTRY GATE: Block counter-slope entries
                gate_ok, gate_reason = check_regime_entry_gate(
                    direction=signal.direction.name,
                    regime=self._directional_regime,
                    slope_75=state.slope_75,
                    slope_12=state.slope_12,
                )
                if not gate_ok:
                    if is_new_candle:
                        logger.info(
                            f"💰 BIG MONEY: Regime gate blocked {signal.strategy_name} "
                            f"{signal.direction.name} -- {gate_reason} "
                            f"(regime={self._directional_regime.value})"
                        )
                    continue
                
                # Signal strength floor
                if signal.strength < self.bm_config.min_signal_strength:
                    logger.debug(
                        f"BIG MONEY: Skipped {signal.strategy_name} — "
                        f"strength {signal.strength:.2f} < {self.bm_config.min_signal_strength}"
                    )
                    continue

                # ── Trade Quality Filter (2026-04-06) ──────────────────────────────
                # Three hard conditions: autocorrelation, structural location, profit factor.
                # Returns probe size (may be reduced) or blocks entry entirely.
                atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
                normal_t1 = 2  # default probe size before tier/cushion adjustment
                tqf_ok, tqf_probe, tqf_reason = self._tqf.check(
                    direction=signal.direction.name,
                    closes=self.features.get_closes(25),
                    price=state.price,
                    vwap=state.vwap,
                    ema_9=state.ema_9,
                    atr=atr,
                    recent_pnls=self._tqf_recent_pnls,
                    consecutive_losses=self._bm_consecutive_losses,
                    normal_probe_size=normal_t1,
                )
                if not tqf_ok:
                    if is_new_candle:
                        logger.info(
                            f"[TQF BLOCKED] {signal.strategy_name} {signal.direction.name} "
                            f"@ {state.price:.2f} | {tqf_reason}"
                        )
                    continue

                self._tqf_probe_override = tqf_probe

                logger.info(
                    f"BIG MONEY: Signal! {signal.strategy_name} {signal.direction.name} "
                    f"@ {state.price:.2f} (trend-aligned, strength={signal.strength:.2f})"
                )
                logger.info(f"[TQF] {tqf_reason}")

                # Enter with big money sizing
                self._bm_try_enter(signal, state)
                if self.position_direction and self.position_direction != Direction.FLAT:
                    return
        
        # Diagnostic logging
        if is_new_candle and self._live_bar_count % 30 == 0:
            logger.info(
                f"💰 BIG MONEY: Bar #{self._live_bar_count} | {state.price:.2f} | "
                f"Trend={self._bm_trend_direction.name if self._bm_trend_direction else 'NONE'} | "
                f"Regime={self._directional_regime.value} | "
                f"S12={state.slope_12:+.1f} S75={state.slope_75:+.1f} | "
                f"ADX={state.adx_14:.1f} | ATR={self._alpha_atr:.2f} | "
                f"ConsecLoss={self._bm_consecutive_losses} | "
                f"Trades={self._bm_trade_count} W={self._bm_win_count} | "
                f"P&L=${self._bm_total_pnl:+,.2f}"
            )

    # ─── Grind Strategy ──────────────────────────────────────────────

    def _bm_grind_trade(self, state: MarketState, bar_high: float, bar_low: float):
        """
        GRIND regime: slow directional drift. Enter on EMA-9 pullback and reclaim.

        Setup (LONG):
          - 20-bar slope is positive (upward drift)
          - This bar's low touched or crossed below EMA-9 (pullback to the MA)
          - This bar's close is above EMA-9 (rejection — buyers stepped in)
          - Stop: 1 tick below the pullback low (structural, not ATR-based)
          - Target: 1.0x ATR above entry (modest fixed target for slow grind)
          - Size: 2 contracts max (probe only — T2 can still add if grind accelerates)

        Mirror logic for SHORT.
        """
        if not state or not getattr(state, 'ema_9', 0):
            return

        atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
        if atr < self.bm_config.min_atr_to_trade:
            return

        # Cooldowns still apply
        if (self._bm_full_loss_cooldown_until > 0
                and time.time() < self._bm_full_loss_cooldown_until):
            return
        if (hasattr(self, '_bm_consec_loss_cooldown_until')
                and time.time() < self._bm_consec_loss_cooldown_until):
            return
        if self._entering:
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            return

        ema = state.ema_9
        slope = state.slope_75  # 20-bar slope — determines grind direction

        direction = None
        stop_price = None

        # ── LONG setup ────────────────────────────────────────────
        if slope > 0:
            if self._grind_ema_touch_long and state.price > ema:
                # Previous bar touched EMA, this bar close is back above → entry
                direction  = Direction.LONG
                stop_price = self._grind_pullback_low - self.instrument.tick_size

            # Track touch for next bar
            if bar_low <= ema:
                self._grind_ema_touch_long = True
                # Record the deepest low of the pullback
                if (self._grind_pullback_low == 0.0
                        or bar_low < self._grind_pullback_low):
                    self._grind_pullback_low = bar_low
            elif state.price > ema:
                # Price moved away from EMA without a touch → reset
                self._grind_ema_touch_long = False
                self._grind_pullback_low   = bar_low

        # ── SHORT setup ───────────────────────────────────────────
        elif slope < 0:
            if self._grind_ema_touch_short and state.price < ema:
                direction  = Direction.SHORT
                stop_price = self._grind_pullback_high + self.instrument.tick_size

            if bar_high >= ema:
                self._grind_ema_touch_short = True
                if (self._grind_pullback_high == 0.0
                        or bar_high > self._grind_pullback_high):
                    self._grind_pullback_high = bar_high
            elif state.price < ema:
                self._grind_ema_touch_short = False
                self._grind_pullback_high   = bar_high

        if direction is None or stop_price is None:
            return

        # Validate structural stop
        stop_pts = abs(state.price - stop_price)
        if stop_pts < self.instrument.tick_size or stop_pts > 2.0 * atr:
            logger.debug(f"[GRIND] Stop distance {stop_pts:.2f}pts rejected (ATR={atr:.2f})")
            return

        target_pts = max(0.75 * atr, 1.0)
        take_profit = (
            state.price + target_pts if direction == Direction.LONG
            else state.price - target_pts
        )

        synthetic = Signal(
            direction     = direction,
            strategy_name = "GRIND_EMA",
            strength      = 0.65,
            reason        = (
                f"EMA pullback reclaim | slope={slope:+.2f}bps "
                f"ER={self._me_er:.3f} stop={stop_pts:.2f}pts"
            ),
            entry_price  = state.price,
            stop_loss    = stop_price,
            take_profit  = take_profit,
        )

        logger.warning(
            f"[GRIND] {direction.name} @ {state.price:.2f} | "
            f"EMA={ema:.2f} pullback_low={self._grind_pullback_low:.2f} "
            f"stop={stop_price:.2f} ({stop_pts:.2f}pts) target={take_profit:.2f}"
        )

        # 2 contracts max for grind probe
        self._tqf_probe_override = 2
        self._bm_try_enter(synthetic, state)

        # Reset touch state after entry attempt
        self._grind_ema_touch_long  = False
        self._grind_ema_touch_short = False
        self._grind_pullback_low    = 0.0
        self._grind_pullback_high   = 0.0

    # ─── Range Trading ───────────────────────────────────────────────

    def _bm_range_trade(self, state: MarketState, range_state):
        """
        RANGE regime: oscillating market, fade extremes at S/R levels.

        Entry source priority:
          1. Chart levels from ChartCapture vision analysis (most accurate — seen on chart)
          2. ADXRangeMapper's detected range boundaries (mechanical fallback)

        LONG when price is within 0.5x ATR of a support level (buy the bounce).
        SHORT when price is within 0.5x ATR of a resistance level (sell the rejection).
        Target: VWAP or the opposing chart level.
        Stop:   1x ATR beyond the entry level.
        Size:   2 contracts (small, mean-reversion loses big when wrong).
        """
        atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
        proximity = 0.5 * atr   # within this distance = "near the level"

        direction   = None
        stop_price  = None
        take_profit = None
        level_used  = None

        # ── Priority 1: Chart levels from vision analysis ──────────
        if self._chart_support or self._chart_resistance:
            price = state.price

            # Near a support level → LONG
            if self._chart_support:
                nearest_sup = min(self._chart_support, key=lambda l: abs(price - l))
                if price <= nearest_sup + proximity and price >= nearest_sup - proximity:
                    direction   = Direction.LONG
                    stop_price  = nearest_sup - atr
                    # Target: nearest resistance or VWAP
                    if self._chart_resistance:
                        take_profit = min(
                            r for r in self._chart_resistance if r > price
                        ) if any(r > price for r in self._chart_resistance) else (
                            state.vwap if state.vwap > price else price + atr
                        )
                    else:
                        take_profit = state.vwap if state.vwap > price else price + atr
                    level_used  = f"chart_support={nearest_sup:.2f}"

            # Near a resistance level → SHORT
            if direction is None and self._chart_resistance:
                nearest_res = min(self._chart_resistance, key=lambda l: abs(price - l))
                if price >= nearest_res - proximity and price <= nearest_res + proximity:
                    direction   = Direction.SHORT
                    stop_price  = nearest_res + atr
                    if self._chart_support:
                        take_profit = max(
                            s for s in self._chart_support if s < price
                        ) if any(s < price for s in self._chart_support) else (
                            state.vwap if state.vwap < price else price - atr
                        )
                    else:
                        take_profit = state.vwap if state.vwap < price else price - atr
                    level_used  = f"chart_resistance={nearest_res:.2f}"

        # ── Priority 2: ADXRangeMapper boundaries ──────────────────
        if direction is None and range_state:
            if range_state.mid_range:
                return   # dead zone — no edge at midpoint
            if range_state.near_support:
                direction   = Direction.LONG
                level_used  = "range_mapper_support"
            elif range_state.near_resistance:
                direction   = Direction.SHORT
                level_used  = "range_mapper_resistance"
            else:
                return   # not near any edge
        
        if direction is None:
            return

        # Common cooldown gates
        if (self._bm_full_loss_cooldown_until > 0
                and time.time() < self._bm_full_loss_cooldown_until
                and self._bm_full_loss_cooldown_direction == direction):
            return
        if (hasattr(self, '_bm_consec_loss_cooldown_until')
                and time.time() < self._bm_consec_loss_cooldown_until):
            return
        if self._bm_last_exit_time > 0:
            if (time.time() - self._bm_last_exit_time) < self.bm_config.re_entry_cooldown_seconds:
                return

        # Build synthetic signal from identified level, cap size at 2 contracts
        if stop_price and take_profit:
            # Chart-level path — use computed stop/target directly
            synthetic = Signal(
                direction     = direction,
                strategy_name = "RANGE_LEVEL",
                strength      = 0.70,
                reason        = f"Range fade @ {level_used}",
                entry_price  = state.price,
                stop_loss    = stop_price,
                take_profit  = take_profit,
            )
            logger.warning(
                f"[RANGE] {direction.name} @ {state.price:.2f} | "
                f"level={level_used} stop={stop_price:.2f} target={take_profit:.2f} | "
                f"support={self._chart_support} resistance={self._chart_resistance}"
            )
            self._tqf_probe_override = 2
            self._bm_try_enter(synthetic, state)
            return

        # ADXRangeMapper fallback — scan mean-reversion strategies
        if not range_state:
            return

        MEAN_REVERSION_STRATEGIES = {'VWAP_REVERT', 'BB_BOUNCE', 'MR_DIP_BUY', 'VWAP_SNAP'}
        for strategy in self.strategies:
            if strategy.name not in MEAN_REVERSION_STRATEGIES:
                continue
            signal = strategy.should_enter(state)
            if signal is None or signal.direction != direction:
                continue
            if signal.strength < 0.35:
                continue

            range_mid = (range_state.range_high + range_state.range_low) / 2
            if direction == Direction.LONG:
                signal.stop_loss   = range_state.range_low  - (atr * 0.5)
                signal.take_profit = range_mid
            else:
                signal.stop_loss   = range_state.range_high + (atr * 0.5)
                signal.take_profit = range_mid

            logger.warning(
                f"[RANGE] {signal.strategy_name} {direction.name} @ {state.price:.2f} | "
                f"boundaries [{range_state.range_low:.2f}-{range_state.range_high:.2f}] | "
                f"stop={signal.stop_loss:.2f} target={signal.take_profit:.2f}"
            )
            self._tqf_probe_override = 2
            self._bm_try_enter(signal, state)
            if self.position_direction and self.position_direction != Direction.FLAT:
                return

    # ─── Short Override Entry ─────────────────────────────────────────
    
    def _bm_short_override_enter(self, state: MarketState):
        """Enter a SHORT via Phase 8.8 override — reduced size, ATR stops."""
        if self._entering:
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            return
        
        # Build a synthetic SHORT signal
        signal = Signal(
            direction=Direction.SHORT,
            strategy_name="short_override",
            strength=0.7,
            entry_price=state.price,
        )
        
        # Reduced size for override trades
        atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
        stop_pts = max(2.0, min(10.0, atr * 0.5))  # Tighter stop (0.5x ATR)
        target_pts = max(2.0, min(10.0, atr * 1.0))  # 1x ATR target
        
        tick = self.instrument.tick_size
        stop_pts = round(stop_pts / tick) * tick
        target_pts = round(target_pts / tick) * tick
        
        signal.stop_loss = state.price + stop_pts
        signal.take_profit = state.price - target_pts
        
        # Cushion-based sizing at reduced multiplier
        balance = self.account_balance or self.mll_tracker.config.account_size
        cushion = self.mll_tracker.update(balance)
        
        if cushion < self.bm_config.cushion_no_trade:
            logger.warning(f"🔻 SHORT OVERRIDE: Cushion ${cushion:.0f} too thin — skipping")
            return
        
        # Use cushion tier for base contracts
        base_contracts = 5  # default smallest
        for min_cushion, tier_contracts, _, _ in self.bm_config.cushion_tiers:
            if cushion >= min_cushion:
                base_contracts = tier_contracts
                break
        contracts = max(2, int(base_contracts * self.bm_config.short_override_size_mult))
        
        logger.info("=" * 60)
        logger.info(f"🔻 SHORT OVERRIDE ENTERING: SELL {contracts}x {self.instrument.instrument}")
        logger.info(f"  Price: {state.price:.2f}")
        logger.info(f"  Stop: {signal.stop_loss:.2f} ({stop_pts:.2f}pts)")
        logger.info(f"  Target: {signal.take_profit:.2f} ({target_pts:.2f}pts)")
        logger.info(f"  Regime: {self._directional_regime.value}")
        logger.info(f"  Slope75: {state.slope_75:+.1f}bps | RSI: {state.rsi_14:.1f}")
        logger.info("=" * 60)
        
        self._entering = True
        try:
            success = self.conn.place_order(
                contract_id=self.config.topstep.contract_id,
                side="SELL",
                size=contracts,
                stop_loss_points=stop_pts,
                take_profit_points=target_pts,
                current_price=state.price,
            )
            
            if success:
                self.position_direction = Direction.SHORT
                self.position_entry_time = time.time()
                self.position_contracts = contracts
                self.position_strategy = "short_override"
                self._last_local_state_change = time.time()
                self._position_max_profit = 0.0
                self._position_worst_pnl = 0.0
                
                placed_order_id = success if isinstance(success, int) else None
                fill_price = self.conn.get_fill_price(
                    self.config.topstep.contract_id, order_id=placed_order_id
                )
                self.position_entry_price = fill_price if fill_price else state.price
                
                from risk.adaptive_exits import AdaptiveExitManager
                self._alpha_exit_mgr = AdaptiveExitManager(
                    config=self._alpha_exit_config,
                    total_contracts=contracts,
                    atr=self._alpha_atr if self._alpha_atr > 0 else state.atr_14,
                    lil=self._lil,
                )
                
                self._hybrid_pm = None
                self._trailing_stop = None
                self._position_manager = None
                self._lil.on_trade_open(atr=self._alpha_atr if self._alpha_atr > 0 else state.atr_14)
                
                self._bm_trade_count += 1
                self._position_worst_unreal = 0.0
                self._position_best_unreal = 0.0
                self._last_unreal_log = time.time()
                logger.info(f"🔻 SHORT OVERRIDE FILLED: SELL {contracts}x @ {self.position_entry_price:.2f}")
            else:
                logger.error(f"🔻 SHORT OVERRIDE ORDER FAILED")
        except Exception as e:
            logger.error(f"🔻 SHORT OVERRIDE entry error: {e}", exc_info=True)
        finally:
            self._entering = False
    
    # ─── Range Breakout Entry ─────────────────────────────────────────
    
    def _bm_range_breakout_enter(self, breakout: BreakoutSignal, state: MarketState):
        """Enter on ADX Range Mapper breakout — backtested params: 6pt stop, 3x ATR trail."""
        if self._entering:
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            return
        
        self._entering = True
        try:
            atr = self._alpha_atr if self._alpha_atr > 0 else state.atr_14
            direction = Direction.LONG if breakout.direction == "LONG" else Direction.SHORT
            
            # Calculate stop — use midpoint but cap at max_stop_pts
            max_stop = self.bm_config.range_mapper_max_stop_pts
            midpoint_dist = breakout.range_width / 2.0
            stop_pts = min(midpoint_dist, max_stop)
            
            # Target — range width projection
            target_pts = breakout.range_width
            
            # Round to tick size
            tick = self.instrument.tick_size
            stop_pts = round(stop_pts / tick) * tick
            target_pts = round(target_pts / tick) * tick
            stop_pts = max(tick * 4, stop_pts)   # minimum 4 ticks stop
            target_pts = max(tick * 8, target_pts)  # minimum 8 ticks target
            
            # Set signal prices
            if direction == Direction.LONG:
                stop_price = state.price - stop_pts
                target_price = state.price + target_pts
            else:
                stop_price = state.price + stop_pts
                target_price = state.price - target_pts
            
            # Cushion-based sizing at reduced multiplier
            balance = self.account_balance or self.mll_tracker.config.account_size
            cushion = self.mll_tracker.update(balance)
            
            if cushion < self.bm_config.cushion_no_trade:
                logger.warning(f"📦 RANGE BREAKOUT: Cushion ${cushion:.0f} too thin — skipping")
                self._entering = False
                return
            
            # Use cushion tier for base contracts, apply size_mult
            base_contracts = 5
            for min_cushion, tier_contracts, _, _ in self.bm_config.cushion_tiers:
                if cushion >= min_cushion:
                    base_contracts = tier_contracts
                    break
            contracts = max(2, int(base_contracts * self.bm_config.range_mapper_size_mult))
            
            side = "BUY" if direction == Direction.LONG else "SELL"
            
            logger.info("=" * 60)
            logger.info(f"📦 RANGE BREAKOUT ENTERING: {side} {contracts}x {self.instrument.instrument}")
            logger.info(f"  Strategy: range_breakout")
            logger.info(f"  Price: {state.price:.2f}")
            logger.info(f"  Stop: {stop_price:.2f} ({stop_pts:.2f}pts, capped at {max_stop:.1f})")
            logger.info(f"  Target: {target_price:.2f} ({target_pts:.2f}pts)")
            logger.info(f"  Range: [{breakout.range_low:.2f}-{breakout.range_high:.2f}] ({breakout.range_width:.1f}pts)")
            logger.info(f"  Duration: {breakout.range_duration_bars} bars | Strength: {breakout.strength:.2f}")
            logger.info(f"  Risk: ${stop_pts * self.instrument.point_value * contracts:.0f}")
            logger.info("=" * 60)
            
            success = self.conn.place_order(
                contract_id=self.config.topstep.contract_id,
                side=side,
                size=contracts,
                stop_loss_points=stop_pts,
                take_profit_points=target_pts,
                current_price=state.price,
            )
            
            if success:
                self.position_direction = direction
                self.position_entry_time = time.time()
                self.position_contracts = contracts
                self.position_strategy = "range_breakout"
                self._last_local_state_change = time.time()
                self._position_max_profit = 0.0
                self._position_worst_pnl = 0.0
                
                placed_order_id = success if isinstance(success, int) else None
                fill_price = self.conn.get_fill_price(
                    self.config.topstep.contract_id, order_id=placed_order_id
                )
                self.position_entry_price = fill_price if fill_price else state.price
                
                # Use adaptive exit manager with looser trail for breakout trades
                from risk.adaptive_exits import AdaptiveExitManager, AdaptiveExitConfig
                breakout_exit_config = AdaptiveExitConfig(
                    point_value=self.instrument.point_value,
                    tick_size=self.instrument.tick_size,
                    
                    # 20/20/60 allocation (2026-03-25 fix)
                    risk_reduce_pct=0.20,
                    core_pct=0.20,
                    runner_pct=0.60,
                    
                    # Tranche 1: Quick scalp
                    risk_reduce_target_atr=0.5,
                    risk_reduce_min_atr=0.25,
                    
                    # Tranche 2: Core — condition-based exits (no time decay)
                    
                    # Tranche 3: Loose runner — 3x ATR trail (backtested optimal)
                    runner_activation_atr=0.5,
                    runner_base_trail_atr=self.bm_config.range_mapper_trail_atr_mult,
                    runner_min_stop_atr=0.15,
                    runner_ratchet_enabled=True,
                    runner_ratchet_step_atr=0.5,
                    runner_ratchet_tighten=0.90,  # slower ratchet for breakout trades
                    
                    # Wider stops for trending
                    runner_trending_mult=1.5,
                    runner_choppy_mult=0.7,
                    runner_volatile_mult=1.3,
                    runner_volume_exhaustion_mult=0.6,
                    runner_stale_seconds=1800,  # 30 min stale
                    runner_stale_trail_atr=0.5,
                    
                    hard_stop_atr=1.5,
                )
                
                self._alpha_exit_mgr = AdaptiveExitManager(
                    config=breakout_exit_config,
                    total_contracts=contracts,
                    atr=atr,
                    lil=self._lil,
                )
                
                self._hybrid_pm = None
                self._trailing_stop = None
                self._position_manager = None
                self._lil.on_trade_open(atr=atr)
                
                self._bm_trade_count += 1
                self._position_worst_unreal = 0.0
                self._position_best_unreal = 0.0
                self._last_unreal_log = time.time()
                logger.info(
                    f"📦 RANGE BREAKOUT FILLED: {side} {contracts}x @ {self.position_entry_price:.2f}"
                )
            else:
                logger.error("📦 RANGE BREAKOUT ORDER FAILED")
        except Exception as e:
            logger.error(f"📦 RANGE BREAKOUT entry error: {e}", exc_info=True)
        finally:
            self._entering = False
    
    # ─── Big Money Entry ──────────────────────────────────────────────
    
    def _bm_try_enter(self, signal: Signal, state: MarketState):
        """Enter with tranche state machine — T1 probe only, T2/T3 added later."""
        if self._entering:
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            return
        
        self._entering = True
        
        try:
            atr = self._alpha_atr
            
            # Cushion check
            balance = self.account_balance or self.mll_tracker.config.account_size
            cushion = self.mll_tracker.update(balance)
            
            if cushion < self.bm_config.cushion_no_trade:
                logger.warning(
                    f"💰 BIG MONEY: Cushion ${cushion:.0f} < ${self.bm_config.cushion_no_trade:.0f} — NOT TRADING"
                )
                self._entering = False
                return
            
            # Find max contracts from cushion tier
            max_contracts = 5  # default
            stop_atr_mult = 1.2
            for min_cushion, tier_contracts, tier_stop_mult, tier_target_mult in self.bm_config.cushion_tiers:
                if cushion >= min_cushion:
                    max_contracts = tier_contracts
                    stop_atr_mult = tier_stop_mult
                    break
            
            # Compute structural invalidation price
            # Use recent swing high/low as invalidation (structural stop)
            tick = self.instrument.tick_size
            atr_stop = atr * stop_atr_mult
            atr_stop = max(2.0, min(20.0, atr_stop))
            atr_stop = round(atr_stop / tick) * tick
            
            if signal.direction == Direction.LONG:
                invalidation = state.price - atr_stop
            else:
                invalidation = state.price + atr_stop
            
            # Check tranche risk manager
            tranche_dir = TrancheDirection.LONG if signal.direction == Direction.LONG else TrancheDirection.SHORT
            can_start, veto_reason = self._tranche_risk_mgr.can_start_new_idea(tranche_dir)
            if not can_start:
                logger.info(f"💰 TRANCHE VETO: {veto_reason}")
                self._entering = False
                return

            # ── REGIME MATRIX VETO (Phase 1, defensive layer) ────────────
            # Blocks entries in cells where the strategy class has no edge:
            # SWAMP (low ER + quiet vol), CLIFF (climactic vol → trend
            # exhaustion), OVERNIGHT_QUIET (low ER overnight → trend
            # conviction weak). See core/regime_matrix.py.
            if (self._regime_matrix is not None
                    and self._latest_regime_label is not None):
                # Pass reversal context so the matrix can apply the new
                # REVERSAL_BLOCK / REVERSAL_OVERRIDE rules added 2026-05-07.
                # Falls back to score=0 when no reversal data → matrix
                # behaves as before (cell rules only).
                rsc = self._latest_reversal_score
                rev_score = rsc.score if rsc is not None else 0.0
                rev_dir = rsc.direction if rsc is not None else "NONE"
                trade_dir = signal.direction.name if signal.direction else "NONE"
                allowed, regime_reason = self._regime_matrix.can_fire(
                    strategy=signal.strategy_name,
                    label=self._latest_regime_label,
                    reversal_score=rev_score,
                    reversal_direction=rev_dir,
                    trade_direction=trade_dir,
                )
                if not allowed:
                    logger.warning(
                        f"[REGIME VETO] {signal.strategy_name} "
                        f"{signal.direction.name} blocked: {regime_reason}"
                    )
                    self._entering = False
                    return
                # Log the reversal-override case explicitly — these are the
                # bypass entries the matrix is now letting through that
                # would have been blocked by cell rules pre-2026-05-07.
                if "REVERSAL_OVERRIDE" in regime_reason:
                    logger.warning(
                        f"[REGIME OVERRIDE] {signal.strategy_name} "
                        f"{signal.direction.name} allowed via reversal: {regime_reason}"
                    )

            # Create the trade idea
            idea = TradeIdea(
                config=self._tranche_config,
                direction=tranche_dir,
                invalidation_price=invalidation,
                entry_price=state.price,
                atr=atr,
                strategy_name=signal.strategy_name,
                max_contracts=max_contracts,
            )
            
            # Enter T1 (probe) only — size scaled by market engine conviction, then TQF
            t1_contracts = idea.t1_size
            # Market engine conviction multiplier (MEDIUM=0.5x, HIGH=1.0x)
            if self._me_probe_mult < 1.0 and self._me_probe_mult > 0.0:
                scaled = max(1, round(t1_contracts * self._me_probe_mult))
                if scaled < t1_contracts:
                    logger.info(
                        f"[MARKET ENGINE] Probe scaled: {t1_contracts}x -> {scaled}x "
                        f"(conviction={self._me_probe_mult:.1f}x)"
                    )
                    t1_contracts = scaled
            # TQF may reduce further (consecutive losses, profit factor)
            if self._tqf_probe_override and self._tqf_probe_override < t1_contracts:
                logger.info(
                    f"[TQF] Probe reduced: {t1_contracts}x -> {self._tqf_probe_override}x"
                )
                t1_contracts = self._tqf_probe_override
            self._tqf_probe_override = 0  # reset after consuming

            side = "BUY" if signal.direction == Direction.LONG else "SELL"
            stop_pts = idea.stop_distance_pts

            logger.info("=" * 60)
            logger.info(f"TRANCHE T1 PROBE: {side} {t1_contracts}x {self.instrument.instrument}")
            logger.info(f"  Strategy: {signal.strategy_name}")
            logger.info(f"  Price: {state.price:.2f}")
            logger.info(f"  Invalidation: {invalidation:.2f} ({stop_pts:.2f}pts)")
            logger.info(f"  1R = {idea.r_value:.2f}pts")
            logger.info(f"  Plan: T1={idea.t1_size} → T2=+{idea.t2_size} → T3=+{idea.t3_size} = {idea.max_contracts} total")
            logger.info(f"  Probe risk: ${t1_contracts * stop_pts * self.instrument.point_value:.0f}")
            logger.info(f"  Full risk: ${idea.max_contracts * stop_pts * self.instrument.point_value:.0f}")
            logger.info("=" * 60)
            
            success = self.conn.place_order(
                contract_id=self.config.topstep.contract_id,
                side=side,
                size=t1_contracts,
                stop_loss_points=stop_pts,
                take_profit_points=None,  # No TP bracket — tranche state machine manages all exits
                current_price=state.price,
            )
            
            if success:
                placed_order_id = success if isinstance(success, int) else None
                fill_price = self.conn.get_fill_price(
                    self.config.topstep.contract_id, order_id=placed_order_id
                )
                
                # Record T1 fill — pass actual order size so state machine tracks
                # the real exchange position, not the idea's theoretical t1_size
                idea.fill_t1(fill_price or state.price, actual_contracts=t1_contracts)
                self._active_idea = idea
                self._tranche_risk_mgr.register_idea(idea)
                
                self.position_direction = signal.direction
                self.position_entry_time = time.time()
                self.position_contracts = t1_contracts
                self.position_strategy = signal.strategy_name
                self._position_max_profit = 0.0
                self._position_worst_pnl = 0.0
                self.position_entry_price = fill_price or state.price
                self._last_local_state_change = time.time()
                
                # Don't create AdaptiveExitManager — tranche handles exits
                self._alpha_exit_mgr = None
                
                # Disable parent's other exit systems
                self._hybrid_pm = None
                self._trailing_stop = None
                self._position_manager = None
                
                # Activate LIL for this trade (emergency override only)
                self._lil.on_trade_open(atr=self._alpha_atr)
                
                self._bm_trade_count += 1
                self._position_worst_unreal = 0.0
                self._position_best_unreal = 0.0
                self._last_unreal_log = time.time()
                logger.info(
                    f"💰 T1 PROBE FILLED: {side} {t1_contracts}x @ {self.position_entry_price:.2f} "
                    f"(trade #{self._bm_trade_count}) | State: {idea.state.name}"
                )
                # Chart capture — analyze market context at trade open
                try:
                    from analyzers.chart_capture import capture as _chart_capture
                    _chart_capture("trade_open", context={
                        "direction": side,
                        "price":     f"{self.position_entry_price:.2f}",
                        "strategy":  self.position_strategy,
                        "contracts": t1_contracts,
                        "adx":       f"{state.adx_14:.1f}",
                        "trade":     f"#{self._bm_trade_count}",
                    })
                except Exception as _ce:
                    logger.debug(f"Chart capture skipped: {_ce}")
            else:
                logger.error(f"💰 T1 PROBE ORDER FAILED: {side} {t1_contracts}x")
            
        except Exception as e:
            logger.error(f"💰 TRANCHE entry error: {e}", exc_info=True)
        finally:
            self._entering = False
    
    # ─── Tranche helpers: scale-in + partial/full exits ─────────────
    
    def _tranche_add_contracts(self, idea: TradeIdea, tranche_name: str, contracts: int, state: MarketState):
        """Add contracts for T2 or T3 tranche."""
        if self._entering:
            return
        self._entering = True
        try:
            side = "BUY" if self.position_direction == Direction.LONG else "SELL"
            stop_pts = idea.stop_distance_pts
            
            logger.info(
                f"💰 TRANCHE {tranche_name}: Adding {side} {contracts}x @ {state.price:.2f} | "
                f"Current: {idea.total_contracts}x | Avg: {idea.avg_entry_price:.2f}"
            )
            
            success = self.conn.place_order(
                contract_id=self.config.topstep.contract_id,
                side=side,
                size=contracts,
                stop_loss_points=stop_pts,
                take_profit_points=None,  # No TP bracket — tranche state machine manages all exits
                current_price=state.price,
            )
            
            if success:
                fill_price = self.conn.get_fill_price(
                    self.config.topstep.contract_id,
                    order_id=success if isinstance(success, int) else None,
                )
                actual_price = fill_price or state.price
                
                if tranche_name == "T2":
                    idea.fill_t2(actual_price, actual_contracts=contracts)
                elif tranche_name == "T3":
                    idea.fill_t3(actual_price, actual_contracts=contracts)

                self.position_contracts += contracts
                # 2026-04-30: re-arm grace window on T2/T3 add. Without this,
                # the phantom-FLAT event TopstepX emits a few seconds after
                # every fill triggers external-close on the whole position.
                # Cost on 4/30: 11:18 trade lost 3 contracts (T2 2x + T3 1x)
                # to phantom kill; would have ridden to +$390 instead of $0.
                self._last_local_state_change = time.time()
                # Update avg entry to match idea's calculation
                self.position_entry_price = idea.avg_entry_price
                
                logger.info(
                    f"💰 {tranche_name} FILLED: {side} +{contracts}x @ {actual_price:.2f} | "
                    f"Total: {idea.total_contracts}x | Avg: {idea.avg_entry_price:.2f} | "
                    f"State: {idea.state.name} | Risk: ${idea.total_risk_dollars:.0f}"
                )
            else:
                logger.error(f"💰 {tranche_name} ORDER FAILED")
        except Exception as e:
            logger.error(f"💰 {tranche_name} error: {e}", exc_info=True)
        finally:
            self._entering = False
    
    def _tranche_partial_exit(self, contracts: int, exit_price: float, reason: str):
        """Exit N contracts using the TopstepX partialCloseContract API endpoint.

        After the partial close, the platform adjusts the position size automatically.
        We then cancel old brackets (sized for the full position) and re-place a single
        correctly-sized stop for the remaining contracts.
        """
        logger.info(
            f"💰 TRANCHE PARTIAL EXIT: {contracts}x @ {exit_price:.2f} | Reason: {reason}"
        )

        try:
            # Step 1: Cancel existing stops — they are sized for the old total
            self.conn._cancel_all_working_orders(reason="tranche partial exit — resizing stops")

            # Step 2: Close N contracts via the proper Topstep partial close API
            success = self.conn.partial_close(
                contract_id=self.config.topstep.contract_id,
                size=contracts,
            )

            if success:
                self.position_contracts -= contracts
                self._last_local_state_change = time.time()
                remaining = self.position_contracts
                logger.info(
                    f"💰 PARTIAL EXIT FILLED: {contracts}x | Remaining: {remaining}x"
                )

                # Step 3: Re-place a single stop for the remaining position
                # Uses place_stop_order (type=4) — NOT place_order (which places a
                # market order + bracket and would close the remaining position).
                if remaining > 0 and self._active_idea:
                    stop_pts = self._active_idea.stop_distance_pts
                    import time as _time
                    _time.sleep(0.3)  # brief pause so platform processes the close first
                    if self.position_direction == Direction.LONG:
                        stop_price = exit_price - stop_pts
                        stop_side = "SELL"
                    else:
                        stop_price = exit_price + stop_pts
                        stop_side = "BUY"
                    stop_ok = self.conn.place_stop_order(
                        contract_id=self.config.topstep.contract_id,
                        side=stop_side,
                        size=remaining,
                        stop_price=stop_price,
                    )
                    if stop_ok:
                        logger.info(
                            f"💰 STOP RE-ANCHORED: {remaining}x | stop={stop_pts:.2f}pts @ {stop_price:.2f}"
                        )
                    else:
                        logger.error(
                            f"💰 STOP RE-ANCHOR FAILED: {remaining}x position now UNPROTECTED"
                        )
            else:
                logger.error(f"💰 PARTIAL EXIT FAILED: {contracts}x — position unchanged")
        except Exception as e:
            logger.error(f"💰 Partial exit error: {e}", exc_info=True)
    
    def _tranche_close_all(self, exit_price: float, reason: str):
        """Close entire position and clean up tranche state."""
        self._bm_chop_while_running = 0  # reset regime-exit counter on every close
        idea = self._active_idea

        if idea and idea.is_active:
            idea.force_close(reason)
            self._tranche_risk_mgr.on_idea_closed(idea)
        
        # Update daily tracking from idea
        if idea:
            # A3: total_pnl is computed from idea.exits for logging + TQF only.
            # Partials already flowed to risk.daily_pnl via add_tranche_pnl;
            # the final contracts will flow through _exit_position → record_trade.
            # _bm_total_pnl is now a @property reading risk.daily_pnl.
            total_pnl = sum(e.pnl_dollars for e in idea.exits)
            self._bm_last_exit_time = time.time()

            # Feed TQF rolling P&L window
            self._tqf_recent_pnls.append(total_pnl)
            if len(self._tqf_recent_pnls) > 20:
                self._tqf_recent_pnls = self._tqf_recent_pnls[-20:]

            if total_pnl > 0:
                self._bm_consecutive_losses = 0
                self._bm_win_count += 1
                self._bm_last_exit_profitable = True
                trade_dir = "LONG" if idea.direction == TrancheDirection.LONG else "SHORT"
                self._dynamic_cooldown.on_win(trade_dir, exit_price)
                logger.info(
                    f"TRANCHE WIN: ${total_pnl:+,.2f} | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f} | "
                    f"Cooldown: {self._dynamic_cooldown.status}"
                )
            elif total_pnl == 0:
                # Scratch — neither win nor loss. Don't touch streak counters
                # or cooldowns. (Pre-A3 this was mislabeled as a win.)
                self._bm_last_exit_profitable = False
                logger.info(
                    f"TRANCHE SCRATCH: ${total_pnl:+,.2f} | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f}"
                )
            else:
                self._bm_consecutive_losses += 1
                self._bm_last_exit_profitable = False

                trade_dir = "LONG" if idea.direction == TrancheDirection.LONG else "SHORT"
                self._dynamic_cooldown.on_loss(trade_dir)

                # Full loss = never hit 1R
                is_full_loss = not idea._1r_taken
                if is_full_loss:
                    self._bm_full_loss_cooldown_until = time.time() + self.bm_config.full_loss_cooldown_seconds
                    dir_for_cooldown = Direction.LONG if idea.direction == TrancheDirection.LONG else Direction.SHORT
                    self._bm_full_loss_cooldown_direction = dir_for_cooldown

                logger.warning(
                    f"💰 TRANCHE LOSS: ${total_pnl:+,.2f} | "
                    f"Full loss: {'YES' if is_full_loss else 'no'} | "
                    f"Consecutive: {self._bm_consecutive_losses} | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f} | "
                    f"Cooldown: {self._dynamic_cooldown.status}"
                )
        
        self._active_idea = None
        self._bm_be_ratchet_done = False  # reset for next trade

        # Clean up LIL
        self._lil.on_trade_close()
        self._alpha_exit_mgr = None
        
        # Use grandparent's exit to clear position state
        self._exit_position(exit_price, reason)
    
    # ─── Override: Exit tracking for consecutive losses + re-entry ──
    
    def _alpha_exit_position(self, exit_price: float, reason: str):
        """Override to track consecutive losses and enable re-entry."""
        if self.position_direction and self.position_direction != Direction.FLAT:
            mult = 1 if self.position_direction == Direction.LONG else -1
            pnl_pts = (exit_price - self.position_entry_price) * mult
            pnl_dollars = pnl_pts * self.instrument.point_value * self.position_contracts

            # A3: _bm_total_pnl is now a @property reading risk.daily_pnl.
            # The final close's P&L flows via super's _exit_position → record_trade.
            self._bm_last_exit_time = time.time()

            # Feed TQF rolling P&L window
            self._tqf_recent_pnls.append(pnl_dollars)
            if len(self._tqf_recent_pnls) > 20:
                self._tqf_recent_pnls = self._tqf_recent_pnls[-20:]

            if pnl_dollars > 0:
                # Winner
                self._bm_consecutive_losses = 0
                self._bm_consec_loss_cooldown_until = 0.0
                self._bm_win_count += 1
                self._bm_last_exit_profitable = True
                exit_dir = "LONG" if self.position_direction == Direction.LONG else "SHORT"
                self._dynamic_cooldown.on_win(exit_dir, exit_price)
                logger.info(
                    f"BIG MONEY WIN: ${pnl_dollars:+,.2f} ({pnl_pts:+.2f}pts) | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f} | "
                    f"Cooldown: {self._dynamic_cooldown.status}"
                )
            elif pnl_dollars == 0:
                # Scratch — neither win nor loss. Don't touch streak counters.
                self._bm_last_exit_profitable = False
                logger.info(
                    f"BIG MONEY SCRATCH: ${pnl_dollars:+,.2f} ({pnl_pts:+.2f}pts) | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f}"
                )
            else:
                # Loser
                self._bm_consecutive_losses += 1
                self._bm_last_exit_profitable = False
                self._bm_consec_wins_same_dir = 0
                exit_dir = "LONG" if self.position_direction == Direction.LONG else "SHORT"
                self._dynamic_cooldown.on_loss(exit_dir)
                
                # ── RISK GUARD 3: Detect full loss (all contracts stopped together) ──
                # A "full loss" = no tranche took partial profit before the stop
                # Detect by checking if all tranches are still open (nothing exited early)
                all_tranches_open = True
                if self._alpha_exit_mgr:
                    mgr = self._alpha_exit_mgr
                    if hasattr(mgr, 'risk_reduce') and mgr.risk_reduce.closed:
                        all_tranches_open = False
                    if hasattr(mgr, 'core') and mgr.core.closed:
                        all_tranches_open = False
                    if hasattr(mgr, 'runner') and mgr.runner.closed:
                        all_tranches_open = False
                
                is_full_loss = (all_tranches_open and pnl_dollars < -100)
                
                if is_full_loss:
                    self._bm_full_loss_cooldown_until = time.time() + self.bm_config.full_loss_cooldown_seconds
                    self._bm_full_loss_cooldown_direction = self.position_direction
                    self._bm_last_trade_was_full_loss = True
                    logger.warning(
                        f"🛑 [FULL LOSS] All {self.position_contracts} contracts stopped — "
                        f"${pnl_dollars:+,.2f} | {self.position_direction.name} blocked for "
                        f"{self.bm_config.full_loss_cooldown_seconds:.0f}s"
                    )
                else:
                    self._bm_last_trade_was_full_loss = False
                
                logger.warning(
                    f"💰 BIG MONEY LOSS: ${pnl_dollars:+,.2f} ({pnl_pts:+.2f}pts) | "
                    f"Consecutive losses: {self._bm_consecutive_losses} | "
                    f"Full loss: {'YES' if is_full_loss else 'no'} | "
                    f"Running P&L: ${self._bm_total_pnl:+,.2f}"
                )
                
                # No daily shutdown — risk guards handle protection
            
            self.vol_sizer.record_trade_pnl(pnl_dollars)
            
            # Track short override exits for cooldown
            if self.position_strategy == "short_override":
                self._bm_last_short_override_exit = time.time()
        
        # Clean up LIL + alpha state
        self._lil.on_trade_close()
        self._alpha_exit_mgr = None
        
        # Use grandparent's exit (HybridBot._exit_position)
        self._exit_position(exit_price, reason)

    def _exit_position(self, exit_price: float, reason: str):
        """Override to ensure tranche FSM is cleaned up on ANY exit path.

        Position sync (external close) calls _exit_position directly,
        bypassing _tranche_close_all. This override catches that case
        and resets the tranche state so future signals aren't blocked.

        Bug found 2026-03-27: stuck PROBE blocked 6 signals on a 70pt selloff.
        """
        # Defensive reset — catches orphan path where _tranche_close_all's
        # reset at line 2248 never ran (e.g., SignalR missed a fill and
        # position sync closed externally). Stale counter here would cause
        # next trade's runner to fire regime exit on 1st CHOP bar instead of 2nd.
        self._bm_chop_while_running = 0

        # If there's an active tranche idea that hasn't been cleaned up yet,
        # it means we got here via position sync or some other direct path.
        # Clean it up before the parent clears position state.
        if self._active_idea and self._active_idea.is_active:
            logger.warning(
                f"💰 TRANCHE ORPHAN CLEANUP: Active idea in {self._active_idea.state.name} "
                f"state during direct _exit_position (reason: {reason}). Forcing close."
            )
            self._active_idea.force_close(reason)
            self._tranche_risk_mgr.on_idea_closed(self._active_idea)
            self._active_idea = None
            self._lil.on_trade_close()
            self._alpha_exit_mgr = None
        elif self._active_idea and not self._active_idea.is_active:
            # Idea already closed (came from _tranche_close_all) but ref not cleared
            if self._active_idea is not None:
                self._active_idea = None

        # Call parent (HybridBot._exit_position)
        super()._exit_position(exit_price, reason)


def main():
    """Entry point for Big Money Bot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    
    import signal as sig
    from config.settings import Config
    
    config = Config.load()
    bot = BigMoneyBot(config)
    
    def handle_shutdown(s, frame):
        logger.info(f"💰 BIG MONEY: Shutdown signal {s}")
        import asyncio
        asyncio.ensure_future(bot.stop())
    
    sig.signal(sig.SIGINT, handle_shutdown)
    sig.signal(sig.SIGTERM, handle_shutdown)
    
    import asyncio
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
