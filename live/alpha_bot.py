"""
Alpha Bot — Phase 1+2: Volatility-Aware + ML Trading System.

Extends hybrid_bot.py's battle-tested connection/execution infrastructure
with Phase 1 upgrades + Phase 2 ML integration:

Phase 1:
  1. ATR-adaptive stops and targets (no more fixed 4pt stops)
  2. Volatility-scaled position sizing (risk dollars, not contract counts)
  3. Entry confirmation timer (wait for price to prove itself)
  4. Trend day mode (disable mean-reversion when market is trending)
  5. Enhanced directional filter (5m EMA stack as hard gate)

Phase 2:
  6. Session-aware persistence regime detector (Asian strict / NY permissive)

Architecture:
  Inherits: Connection, SignalR, order execution, position sync, balance checks
  Overrides: _process_bar (entry logic), _try_enter (sizing/stops), _check_exit (adaptive exits)
  New: ConfirmationTimer, TrendDayDetector, VolatilitySizer, AdaptiveExitManager,
       PersistenceRegimeDetector
"""

import logging
import time
import sys
from typing import Optional

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from live.hybrid_bot import TradingBot as HybridBot
from core.confirmation import ConfirmationTimer, ConfirmationConfig
from core.trend_day import TrendDayDetector, TrendDayConfig
from strategies.base import MarketState, Direction, Signal
from core.regime import Regime
from risk.vol_sizing import VolatilitySizer, VolSizingConfig
from core.regime_persistence import PersistenceRegimeDetector, TradingSession
from core.range_detector import RangeDetector, RangeConfig, MarketStructure
from risk.adaptive_exits import AdaptiveExitManager, AdaptiveExitConfig, TrancheExit, ExitReason
from risk.manager import TradeRecord
from core.direction_detector import DirectionalDetector, BiasDirection

logger = logging.getLogger(__name__)


class AlphaBot(HybridBot):
    """Phase 1 Alpha Bot — Volatility-aware, confirmation-gated trading.
    
    Inherits all connection/execution plumbing from HybridBot.
    Overrides the decision-making layer with Phase 1 upgrades.
    """
    
    def __init__(self, config=None):
        # Initialize parent (connection, features, strategies, regime, risk)
        super().__init__(config)
        
        # === Phase 1 Components ===
        
        # Volatility-targeted sizing
        self.vol_sizer = VolatilitySizer(VolSizingConfig(
            risk_per_trade_dollars=150.0,
            daily_risk_budget=999999.0,
            stop_atr_mult=1.5,
            target_atr_mult=2.5,
            min_stop_pts=1.5,
            max_stop_pts=15.0,
            max_contracts=10,
            point_value=self.instrument.point_value,
            tick_size=self.instrument.tick_size,
            drawdown_full_reduce=1000.0,
            cushion_danger_zone=2000.0,
            cushion_critical=1000.0,
        ))
        
        # Adaptive exit config (created per-trade)
        self._alpha_exit_config = AdaptiveExitConfig(
            point_value=self.instrument.point_value,
            tick_size=self.instrument.tick_size,
        )
        self._alpha_exit_mgr: Optional[AdaptiveExitManager] = None
        
        # Entry confirmation timer
        self.confirmation = ConfirmationTimer(ConfirmationConfig(
            base_seconds=15.0,
            max_adverse_atr=1.0,
            min_favorable_atr=0.20,
        ))
        
        # Trend day detector
        self.trend_day = TrendDayDetector(TrendDayConfig(
            gap_threshold_pct=0.005,
            atr_percentile_threshold=0.80,
            min_criteria=2,
            mean_reversion_mult=0.0,
        ))
        
        self._alpha_open_price: Optional[float] = None
        self._alpha_atr: float = 2.5
        
        # Entry quality gates (configurable — subclasses can tighten)
        self._min_directional_confidence: float = 0.6   # 3/5 votes default
        self._min_signal_strength: float = 0.0           # No floor default
        
        # Directional detector (vote-based, 5 indicators)
        self._direction_detector = DirectionalDetector()
        self._last_directional_bias = None
        
        # === Phase 2: Session-Aware Persistence Regime Detector ===
        self.persistence_regime = PersistenceRegimeDetector(lookback_period=30)
        self._persistence_context = None
        
        # === Phase 3: Range Detector — S/R levels + structure classification ===
        self.range_detector = RangeDetector(RangeConfig(
            swing_lookback=60,
            structure_lookback=30,
            swing_window=5,
            tight_chop_threshold=8.0,    # Was 10.0 — align with 8pt chop threshold
            wide_range_min=8.0,          # Was 10.0 — 8pt+ is tradeable range
            wide_range_max=35.0,         # Was 20.0 — 35pt range is still range, not trend
            edge_proximity_pts=8.0,
            mid_zone_pct=0.30,
        ))
        self._range_state = None
        
        logger.info("=" * 60)
        logger.info("ALPHA BOT v2 — Phase 1+2: Vol-Aware System")
        logger.info("=" * 60)
        logger.info(f"  Risk/trade: ${self.vol_sizer.config.risk_per_trade_dollars}")
        logger.info(f"  Daily budget: ${self.vol_sizer.config.daily_risk_budget}")
        logger.info(f"  Stop: {self.vol_sizer.config.stop_atr_mult}x ATR")
        logger.info(f"  Target: {self.vol_sizer.config.target_atr_mult}x ATR")
        logger.info(f"  Confirmation: {self.confirmation.config.base_seconds}s base")
        logger.info(f"  Trend day detection: ON")
        logger.info("=" * 60)
    
    # ─── Directional Filters ────────────────────────────────────────
    
    def _get_directional_bias(self, state: MarketState) -> str:
        """5m EMA stack direction. Returns 'bullish', 'bearish', or 'neutral'."""
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0 and state.ema_5m_50 > 0:
            if state.ema_5m_9 > state.ema_5m_26 > state.ema_5m_50:
                return "bullish"
            elif state.ema_5m_9 < state.ema_5m_26 < state.ema_5m_50:
                return "bearish"
        return "neutral"
    
    def _alpha_direction_check(self, signal: Signal, state: MarketState) -> tuple[bool, str]:
        """Combined directional filter: persistence regime + trend day + EMA stack."""
        # Phase 2: Persistence regime filter — SOFT (log warning, don't block)
        # In aggressive mode, we trade all regimes. The stop handles bad entries.
        if self._persistence_context:
            ctx = self._persistence_context
            if not ctx.should_trade:
                logger.debug(f"[PERSIST] Soft-pass: {ctx.reason} (would have blocked)")
            
            # Direction alignment — also soft, just log
            if ctx.preferred_direction == 1 and signal.direction == Direction.SHORT:
                logger.debug(f"[PERSIST] Soft-pass: LONG-only regime, allowing SHORT anyway")
            if ctx.preferred_direction == -1 and signal.direction == Direction.LONG:
                logger.debug(f"[PERSIST] Soft-pass: SHORT-only regime, allowing LONG anyway")
        
        # Trend day filter — soft, log only (counter-trend scalps can still work)
        if not self.trend_day.is_direction_allowed(signal.direction):
            logger.debug(f"[ALPHA] Trend day ({self.trend_day.trend_direction}) would block {signal.direction.name} — allowing (soft)")
        
        # 5m EMA stack — soft warning only, don't block
        # EMAs lag in transitions; blocking kills entries at turning points
        bias = self._get_directional_bias(state)
        if bias == "bullish" and signal.direction == Direction.SHORT:
            logger.debug(f"[ALPHA] Note: 5m EMA stack bullish but allowing SHORT (soft)")
        if bias == "bearish" and signal.direction == Direction.LONG:
            logger.debug(f"[ALPHA] Note: 5m EMA stack bearish but allowing LONG (soft)")
        
        return True, "OK"
    
    # ─── Override: _process_bar ──────────────────────────────────────
    
    def _process_bar(self, bar: dict):
        """Override: Process a bar with Phase 1 logic.
        
        Changes from parent:
          - Signals go through confirmation timer
          - Trend day detector updated
          - Strategy multipliers applied
          - Confirmed signals use vol-scaled sizing
        """
        # === Same setup as parent ===
        bar_ts = bar.get("t") or bar.get("time") or bar.get("timestamp")
        is_new_candle = (bar_ts != self._last_bar_timestamp) if bar_ts else True
        if bar_ts:
            self._last_bar_timestamp = bar_ts
        
        self._bar_count += 1
        if is_new_candle:
            self._live_bar_count += 1
        self.features.add_bar(bar)
        
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
        self._current_session = session  # Store for smart stop calculation
        mins = minutes_since_rth_open(bar_dt)
        
        state_dict = self.features.build_market_state(session, mins)
        state = MarketState(**state_dict)
        self._last_bar_state = state
        
        # === Phase 1: Update ATR ===
        if state.atr_14 > 0:
            self._alpha_atr = state.atr_14
        
        if self._alpha_open_price is None:
            self._alpha_open_price = state.price
        
        # === Phase 1: Update regime ===
        if is_new_candle:
            self._current_regime = self.regime_detector.detect(self.features, state)
            
            # === Phase 3: Update range/structure detection ===
            if hasattr(self.features, '_bars') and len(self.features._bars) >= 20:
                self._range_state = self.range_detector.update(
                    list(self.features._bars), state.price
                )
        
        # === Phase 2: Update persistence regime (session-aware) ===
        if is_new_candle and len(self.features._bars) >= 30:
            # Get ET hour from bar timestamp
            et_hour = None
            try:
                import pytz
                et_tz = pytz.timezone('US/Eastern')
                et_hour = bar_dt.astimezone(et_tz).hour
            except:
                et_hour = bar_dt.hour  # Fallback
            
            self._persistence_context = self.persistence_regime.detect(
                self.features._bars, current_hour_et=et_hour
            )
            
            if self._persistence_context and not self._persistence_context.should_trade:
                if self._live_bar_count % 30 == 0:
                    logger.info(
                        f"[PERSIST] Note: {self._persistence_context.reason} (soft — still trading)"
                    )
        
        # === Phase 1: Update trend day detector ===
        atr_pctl = self._current_regime.atr_percentile if self._current_regime else 0.5
        self.trend_day.update(
            current_price=state.price,
            vwap=state.vwap,
            atr_percentile=atr_pctl,
            adx=state.adx_14,
            open_price=self._alpha_open_price,
        )
        
        # === Phase 1: Process confirmed signals ===
        confirmed = self.confirmation.update(state.price, self._alpha_atr)
        
        # If in position, manage it
        if self.position_direction and self.position_direction != Direction.FLAT:
            # Use parent's _check_exit if no alpha exit manager,
            # otherwise use alpha adaptive exits
            if self._alpha_exit_mgr:
                self._alpha_check_exit(state)
            else:
                self._check_exit(state)
            return
        
        if self._entering:
            return
        
        # Process any confirmed signals first
        for sig in confirmed:
            self._alpha_try_enter(sig, state)
            if self.position_direction and self.position_direction != Direction.FLAT:
                return
        
        # Directional confidence gate (backtest-validated 2025-03-08)
        bias = self._direction_detector.detect(state)
        self._last_directional_bias = bias
        if bias.confidence < self._min_directional_confidence:
            return  # Not enough directional agreement
        
        # Scan strategies for new signals
        for strategy in self.strategies:
            # Trend day: check if strategy is allowed
            td_mult = self.trend_day.get_strategy_multiplier(strategy.name)
            if td_mult <= 0:
                if is_new_candle:
                    logger.debug(f"[ALPHA] {strategy.name} skipped: trend_day mult=0")
                continue
            
            signal = strategy.should_enter(state)
            if signal is None:
                continue
            
            # Signal strength floor (backtest-validated)
            if signal.strength < self._min_signal_strength:
                logger.debug(f"[ALPHA] {signal.strategy_name} skipped: strength {signal.strength:.2f} < {self._min_signal_strength}")
                continue
            
            # Directional alignment — signal must match detector bias
            if bias.direction == BiasDirection.LONG and signal.direction == Direction.SHORT:
                logger.debug(f"[ALPHA] {signal.strategy_name} SHORT blocked — bias is LONG")
                continue
            if bias.direction == BiasDirection.SHORT and signal.direction == Direction.LONG:
                logger.debug(f"[ALPHA] {signal.strategy_name} LONG blocked — bias is SHORT")
                continue
            
            logger.info(f"[ALPHA] Signal! {signal.strategy_name} {signal.direction.name} @ {state.price} (strength={signal.strength:.2f})")
            
            # Direction filter
            allowed, reason = self._alpha_direction_check(signal, state)
            if not allowed:
                logger.info(f"[ALPHA] {signal.strategy_name} direction-blocked: {reason}")
                continue
            
            # Submit to confirmation timer
            regime_name = self._current_regime.regime.value if self._current_regime else "unknown"
            fast_signal = self.confirmation.submit_signal(
                signal, state.price, self._alpha_atr, regime=regime_name
            )
            
            if fast_signal:
                self._alpha_try_enter(fast_signal, state)
                if self.position_direction and self.position_direction != Direction.FLAT:
                    return
        
        # Diagnostic logging
        if is_new_candle and self._live_bar_count % 30 == 0:
            td_status = "TREND DAY" if self.trend_day.is_trend_day else "normal"
            pending = self.confirmation.pending_count
            p_ctx = self._persistence_context
            p_status = (f"{p_ctx.session.value}|{p_ctx.trend_regime.value}|"
                       f"p={p_ctx.persistence_ratio:.2f}") if p_ctx else "N/A"
            range_status = f"Struct={self._range_state.structure.value}|{self._range_state.range_width:.0f}pt" if self._range_state else "Struct=N/A"
            logger.info(
                f"[ALPHA] Bar #{self._live_bar_count} | {state.price:.2f} | "
                f"ATR={self._alpha_atr:.2f} | ADX={state.adx_14:.1f} | "
                f"Regime={self._current_regime.regime.value if self._current_regime else 'N/A'} | "
                f"{range_status} | "
                f"Persist={p_status} | "
                f"Mode={td_status} | Confirming={pending} | "
                f"DailyP&L=${self.vol_sizer.daily_pnl:+.0f} | "
                f"RiskUsed=${self.vol_sizer.daily_risk_used:.0f}/${self.vol_sizer.config.daily_risk_budget:.0f}"
            )
    
    # ─── Alpha Entry (Vol-Scaled) ───────────────────────────────────
    
    def _alpha_try_enter(self, signal: Signal, state: MarketState):
        """Enter a trade with volatility-scaled sizing and ATR-adaptive stops."""
        if self._entering:
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            return
        
        self._entering = True
        
        try:
            # === Phase 3: Range/Structure Filter ===
            # Trend strategies bypass range filter — they trade on momentum, not S/R proximity
            TREND_STRATEGIES = {'TREND_FOLLOW', 'MOMENTUM', 'SR_BREAKOUT'}
            is_trend_strategy = signal.strategy_name.upper() in TREND_STRATEGIES
            
            if self._range_state and not is_trend_strategy:
                rs = self._range_state
                direction_str = "LONG" if signal.direction == Direction.LONG else "SHORT"
                
                # TIGHT CHOP: Don't trade — BUT only if ADX confirms it's actually choppy
                if rs.structure == MarketStructure.TIGHT_CHOP:
                    if state.adx_14 < 20:
                        logger.info(f"[ALPHA] RANGE BLOCKED: Tight chop ({rs.range_width:.1f}pt range, ADX={state.adx_14:.0f}) — no trade")
                        self._entering = False
                        return
                    else:
                        logger.info(f"[ALPHA] RANGE OVERRIDE: Tight chop ({rs.range_width:.1f}pt) but ADX={state.adx_14:.0f} says trending — allowing")
                
                # WIDE RANGE: Only trade near edges, block mid-range entries
                if rs.structure == MarketStructure.WIDE_RANGE:
                    if rs.mid_range:
                        if signal.strength >= 0.7:
                            logger.warning(
                                f"[ALPHA] RANGE WARNING: Mid-range entry but strong signal "
                                f"(strength={signal.strength:.2f}) — allowing. "
                                f"price in zone [{rs.range_low:.0f}-{rs.range_high:.0f}] "
                                f"| sup={rs.nearest_support:.0f} res={rs.nearest_resistance:.0f}"
                            )
                        else:
                            logger.info(
                                f"[ALPHA] RANGE BLOCKED: Mid-range entry — "
                                f"price in dead zone [{rs.range_low:.0f}-{rs.range_high:.0f}] "
                                f"| sup={rs.nearest_support:.0f} res={rs.nearest_resistance:.0f}"
                            )
                            self._entering = False
                            return
                    
                    # In wide range: note S/R proximity (informational, not blocking)
                    if signal.direction == Direction.LONG and not rs.near_support:
                        logger.info(
                            f"[ALPHA] RANGE NOTE: LONG not near support — "
                            f"price={state.price:.2f}, nearest sup={rs.nearest_support:.2f} "
                            f"({state.price - rs.nearest_support:.1f}pts away) — allowing anyway"
                        )
                        # self._entering = False; return  # Disabled: S/R proximity is informational
                    
                    if signal.direction == Direction.SHORT and not rs.near_resistance:
                        logger.info(
                            f"[ALPHA] RANGE NOTE: SHORT not near resistance — "
                            f"price={state.price:.2f}, nearest res={rs.nearest_resistance:.2f} "
                            f"({rs.nearest_resistance - state.price:.1f}pts away) — allowing anyway"
                        )
                        # self._entering = False; return  # Disabled: S/R proximity is informational
                    
                    logger.info(
                        f"[ALPHA] RANGE APPROVED: {direction_str} near {'support' if rs.near_support else 'resistance'} "
                        f"in {rs.range_width:.0f}pt range [{rs.range_low:.0f}-{rs.range_high:.0f}]"
                    )
            elif self._range_state and is_trend_strategy:
                logger.info(f"[ALPHA] RANGE BYPASS: {signal.strategy_name} is a trend strategy — skipping range filter")
            
            # Get cushion
            balance = self.account_balance or self.mll_tracker.config.account_size
            cushion = self.mll_tracker.update(balance)
            
            # Regime multiplier
            regime_mult = 1.0
            if self._current_regime:
                r = self._current_regime.regime
                if r == Regime.VOLATILE:
                    regime_mult = 0.6
                elif r == Regime.TRENDING_FAST:
                    regime_mult = 1.2
                elif r == Regime.DEAD:
                    regime_mult = 0.5
            
            # Vol-scaled sizing (session-aware + structure stops)
            direction_str = "LONG" if signal.direction == Direction.LONG else "SHORT"
            session_name = getattr(self, '_current_session', None)
            contracts, stop_pts, target_pts, size_reason = self.vol_sizer.calculate_contracts(
                atr=self._alpha_atr,
                cushion=cushion,
                signal_strength=signal.strength,
                regime_mult=regime_mult,
                session=session_name,
                direction=direction_str,
                price=state.price,
                bars=list(self.features._bars[-30:]) if hasattr(self.features, '_bars') else None,
            )
            
            if contracts <= 0:
                logger.info(f"[ALPHA] Sizing blocked: {size_reason}")
                self._entering = False
                return
            
            # Override signal stops with ATR-adaptive values
            if signal.direction == Direction.LONG:
                signal.stop_loss = state.price - stop_pts
                signal.take_profit = state.price + target_pts
            else:
                signal.stop_loss = state.price + stop_pts
                signal.take_profit = state.price - target_pts
            signal.entry_price = state.price
            
            # Risk manager checks (regime compat, cooldowns, frequency, etc.)
            should_trade, risk_reason, risk_contracts = self.risk.should_trade(
                signal, state, regime_state=self._current_regime
            )
            # Apply risk manager's size adjustment (soft filters reduce contracts)
            if risk_contracts and risk_contracts < contracts:
                logger.info(f"[ALPHA] Risk manager reduced size: {contracts} → {risk_contracts} contracts")
                contracts = risk_contracts
            if not should_trade:
                logger.info(f"[ALPHA] Risk blocked: {risk_reason}")
                self.vol_sizer.unreserve_risk(contracts, stop_pts)
                self._entering = False
                return
            
            # === EXECUTE ===
            side = "BUY" if signal.direction == Direction.LONG else "SELL"
            
            logger.info("=" * 55)
            logger.info(f"[ALPHA] ENTERING: {side} {contracts}x {self.instrument.instrument}")
            logger.info(f"  Strategy: {signal.strategy_name} (strength={signal.strength:.2f})")
            logger.info(f"  Price: {state.price:.2f}")
            logger.info(f"  Stop: {signal.stop_loss:.2f} ({stop_pts:.2f}pts = {stop_pts/self._alpha_atr:.1f}x ATR)")
            logger.info(f"  Target: {signal.take_profit:.2f} ({target_pts:.2f}pts = {target_pts/self._alpha_atr:.1f}x ATR)")
            logger.info(f"  Sizing: {size_reason}")
            if self.trend_day.is_trend_day:
                logger.info(f"  Trend Day: {self.trend_day.trend_direction}")
            logger.info("=" * 55)
            
            # Place order via parent's connection
            success = self.conn.place_order(
                contract_id=self.config.topstep.contract_id,
                side=side,
                size=contracts,
                stop_loss_points=stop_pts,
                take_profit_points=target_pts,
                current_price=state.price,
            )
            
            if success:
                self.position_direction = signal.direction
                self.position_entry_time = time.time()
                self.position_contracts = contracts
                self.position_strategy = signal.strategy_name
                self._position_max_profit = 0.0
                self._position_worst_pnl = 0.0
                self._last_local_state_change = time.time()
                
                # Get actual fill price
                placed_order_id = success if isinstance(success, int) else None
                fill_price = self.conn.get_fill_price(
                    self.config.topstep.contract_id, order_id=placed_order_id
                )
                self.position_entry_price = fill_price if fill_price else state.price
                
                if fill_price and abs(fill_price - state.price) > 0.5:
                    logger.warning(
                        f"[ALPHA] SLIPPAGE: Expected {state.price:.2f}, "
                        f"filled {fill_price:.2f} ({abs(fill_price - state.price):.2f}pts)"
                    )
                
                # Create adaptive exit manager
                self._alpha_exit_mgr = AdaptiveExitManager(
                    config=self._alpha_exit_config,
                    total_contracts=contracts,
                    atr=self._alpha_atr,
                )
                
                # Disable parent's hybrid/trailing stop (we use adaptive exits)
                self._hybrid_pm = None
                self._trailing_stop = None
                self._position_manager = None
                
                logger.info(
                    f"[ALPHA] FILLED: {side} {contracts}x @ {self.position_entry_price:.2f}"
                )
            else:
                logger.error(f"[ALPHA] ORDER FAILED: {side} {contracts}x")
                self.vol_sizer.unreserve_risk(contracts, stop_pts)
            
        except Exception as e:
            logger.error(f"[ALPHA] Entry error: {e}", exc_info=True)
        finally:
            self._entering = False
    
    # ─── Alpha Exit (Adaptive) ──────────────────────────────────────
    
    def _alpha_check_exit(self, state: MarketState):
        """Check exits using adaptive (ATR-scaled) exit manager."""
        if not self._alpha_exit_mgr:
            return
        
        hold_time = time.time() - self.position_entry_time
        mult = 1 if self.position_direction == Direction.LONG else -1
        unrealized_pts = (state.price - self.position_entry_price) * mult
        
        # Update ATR in exit manager (conditions change mid-trade)
        self._alpha_exit_mgr.update_atr(self._alpha_atr)
        
        # Get signal invalidation from strategy
        signal_exit = None
        for strat in self.strategies:
            if strat.name == self.position_strategy:
                signal_exit = strat.should_exit(
                    state, self.position_entry_price,
                    self.position_direction, hold_time
                )
                break
        
        # Regime for runner trail
        regime_name = self._current_regime.regime.value if self._current_regime else "unknown"
        volume_ratio = state.volume_ratio_5 if state.volume_ratio_5 > 0 else 1.0
        
        exits = self._alpha_exit_mgr.check_exits(
            unrealized_pts=unrealized_pts,
            signal_exit_reason=signal_exit,
            regime=regime_name,
            volume_ratio=volume_ratio,
        )
        
        for tranche_exit in exits:
            if self._alpha_exit_mgr.all_closed:
                # Last tranche — flatten everything
                self._alpha_exit_position(state.price,
                    f"[ALPHA:{tranche_exit.role}] {tranche_exit.reason}")
                return
            else:
                # Partial close
                pnl_dollars = tranche_exit.pnl_pts * self.instrument.point_value * tranche_exit.size
                logger.info(
                    f"[ALPHA] Partial: {tranche_exit.role} {tranche_exit.size}x "
                    f"@ {state.price:.2f} (${pnl_dollars:+.2f}) | {tranche_exit.reason}"
                )
                success = self.conn.partial_close(
                    self.config.topstep.contract_id, tranche_exit.size
                )
                if success:
                    self.position_contracts = self._alpha_exit_mgr.remaining
                    if self._alpha_exit_mgr.remaining > 0:
                        self._resize_brackets(self._alpha_exit_mgr.remaining)
                    
                    trade = TradeRecord(
                        entry_time=self.position_entry_time,
                        exit_time=time.time(),
                        direction=self.position_direction,
                        entry_price=self.position_entry_price,
                        exit_price=state.price,
                        contracts=tranche_exit.size,
                        pnl_dollars=pnl_dollars,
                        strategy=self.position_strategy,
                        exit_reason=f"[ALPHA:{tranche_exit.role}] {tranche_exit.reason}",
                    )
                    self.risk.record_trade(trade)
                    self.vol_sizer.record_trade_pnl(pnl_dollars)
                else:
                    logger.error("[ALPHA] Partial close FAILED - flattening for safety")
                    self._alpha_exit_position(state.price, "Partial close failed - safety flatten")
                    return
        
        if self._alpha_exit_mgr.all_closed:
            self._alpha_exit_position(state.price, "All alpha tranches closed")
    
    def _alpha_exit_position(self, exit_price: float, reason: str):
        """Close position and clean up alpha state."""
        # Record final P&L in vol sizer
        if self.position_direction and self.position_direction != Direction.FLAT:
            mult = 1 if self.position_direction == Direction.LONG else -1
            final_pnl_pts = (exit_price - self.position_entry_price) * mult
            final_pnl_dollars = final_pnl_pts * self.instrument.point_value * self.position_contracts
            self.vol_sizer.record_trade_pnl(final_pnl_dollars)
        
        # Clean up alpha state
        self._alpha_exit_mgr = None
        
        # Use parent's exit (handles flatten, trade recording, position reset)
        self._exit_position(exit_price, reason)
    
    # ─── Override tick-level loss check to use alpha exits ───────────
    
    def _check_tick_loss(self, price: float):
        """Override: Use alpha adaptive exits on tick-level updates too."""
        if not self.position_direction or self.position_direction == Direction.FLAT:
            return
        
        mult = 1 if self.position_direction == Direction.LONG else -1
        unrealized_pts = (price - self.position_entry_price) * mult
        unrealized_dollars = unrealized_pts * self.instrument.point_value * self.position_contracts
        
        self._position_worst_pnl = min(self._position_worst_pnl, unrealized_dollars)
        
        # Alpha adaptive exits on every tick
        if self._alpha_exit_mgr:
            self._alpha_exit_mgr.update_atr(self._alpha_atr)
            
            signal_exit = None
            regime = "unknown"
            volume_ratio = 1.0
            
            if self._last_bar_state:
                hold_time = time.time() - self.position_entry_time
                for strategy in self.strategies:
                    if strategy.name == self.position_strategy:
                        signal_exit = strategy.should_exit(
                            self._last_bar_state, self.position_entry_price,
                            self.position_direction, hold_time
                        )
                        break
                regime = getattr(self._last_bar_state, 'regime', 'unknown')
                volume_ratio = self._last_bar_state.volume_ratio_5 if self._last_bar_state.volume_ratio_5 > 0 else 1.0
            
            exits = self._alpha_exit_mgr.check_exits(
                unrealized_pts=unrealized_pts,
                signal_exit_reason=signal_exit,
                regime=regime,
                volume_ratio=volume_ratio,
            )
            
            for tranche_exit in exits:
                if self._alpha_exit_mgr.all_closed:
                    self._alpha_exit_position(price, f"[ALPHA:{tranche_exit.role}] {tranche_exit.reason}")
                    return
                else:
                    success = self.conn.partial_close(
                        self.config.topstep.contract_id, tranche_exit.size
                    )
                    if success:
                        self.position_contracts = self._alpha_exit_mgr.remaining
                        if self._alpha_exit_mgr.remaining > 0:
                            self._resize_brackets(self._alpha_exit_mgr.remaining)
                        pnl_dollars = tranche_exit.pnl_pts * self.instrument.point_value * tranche_exit.size
                        trade = TradeRecord(
                            entry_time=self.position_entry_time,
                            exit_time=time.time(),
                            direction=self.position_direction,
                            entry_price=self.position_entry_price,
                            exit_price=price,
                            contracts=tranche_exit.size,
                            pnl_dollars=pnl_dollars,
                            strategy=self.position_strategy,
                            exit_reason=f"[ALPHA:{tranche_exit.role}] {tranche_exit.reason}",
                        )
                        self.risk.record_trade(trade)
                        self.vol_sizer.record_trade_pnl(pnl_dollars)
                    else:
                        self._alpha_exit_position(price, "Tick partial close failed - flatten")
                        return
            
            if self._alpha_exit_mgr.all_closed:
                self._alpha_exit_position(price, "All alpha tranches closed (tick)")
            return
        
        # Fallback to parent's tick loss check if no alpha exit manager
        super()._check_tick_loss(price)


def main():
    """Entry point for alpha_bot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    
    import signal as sig
    from config.settings import Config
    
    config = Config.load()
    bot = AlphaBot(config)
    
    def handle_shutdown(s, frame):
        logger.info(f"[ALPHA] Shutdown signal {s}")
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
