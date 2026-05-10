"""
Topstep5 Live Trading Bot — The Orchestrator.

<500 lines. Connects data → strategies → risk → execution.
No ML. No magic. Just clean signal → filter → trade.
"""

import asyncio
import logging
import os
import time
import signal
import sys
import requests
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config.settings import Config
from core.connection import TopstepConnection
from core.features import FeatureEngine
from core.market_data import detect_session, minutes_since_rth_open, TickAggregator
from strategies.base import MarketState, Direction
from strategies.vwap_revert import VWAPReversion
from strategies.momentum import Momentum
from strategies.bollinger_bounce import BollingerBounce
from strategies.sr_breakout import SRBreakout
from strategies.delta_divergence import DeltaDivergence
from strategies.ema_reject import EMAReject
from strategies.trend_follow import TrendFollow
from core.regime import RegimeDetector, Regime
from risk.manager import RiskManager, RiskConfig, TradeRecord
from risk.trailing_stop import TrailingStopManager, TrailingStopConfig, StopStage
from risk.position_manager import PositionManager, TrancheExit
from risk.hybrid_position_manager import HybridPositionManager, HybridConfig, TrancheExit as HybridTrancheExit
from risk.topstep_mll import MLLTracker, TopstepAccountConfig

logger = logging.getLogger(__name__)


class TradingBot:
    """
    The quant. Connects to TopstepX, runs strategies, manages risk.
    
    Flow each tick/bar:
    1. Update features
    2. Build market state
    3. Check each strategy for signals
    4. Run signal through risk manager
    5. If approved → execute with bracket orders
    6. Monitor open position for exit signals
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config.load()
        self.running = False
        
        # Instrument config (must be before Features and Risk)
        self.instrument = self.config.instrument
        
        # Connection
        self.conn = TopstepConnection(
            username=self.config.topstep.username,
            api_key=self.config.topstep.api_key,
            account_id=self.config.topstep.account_id,
            api_endpoint=self.config.topstep.api_endpoint,
            signalr_hub=self.config.topstep.signalr_hub,
        )
        
        # Features
        self.features = FeatureEngine(tick_size=self.instrument.tick_size)
        self.tick_agg = TickAggregator(bar_minutes=3)
        
        # Strategies
        self.strategies = [
            VWAPReversion(),
            Momentum(),
            BollingerBounce(),
            SRBreakout(),
            DeltaDivergence(),
            EMAReject(),
            TrendFollow(),
        ]
        
        # Regime detector
        self.regime_detector = RegimeDetector()
        self._current_regime = None
        
        # MLL Tracker — cushion-based position scaling
        self.mll_tracker = MLLTracker(TopstepAccountConfig(
            account_size=self.config.account.starting_balance,
            initial_mll=self.config.account.mll_threshold,
            initial_cushion=self.config.account.starting_balance - self.config.account.mll_threshold,
            point_value=self.instrument.point_value,
            max_position_contracts=self.config.trading.max_contracts,
            instrument=self.instrument.instrument,
            mes_per_es=self.instrument.mes_per_es,
        ))
        
        # Risk
        self.risk = RiskManager(
            config=RiskConfig(
                max_daily_loss=self.config.account.daily_loss_hard_stop,
                daily_profit_target=self.config.account.daily_profit_target,
                max_contracts=self.config.trading.max_contracts,
                base_contracts=self.config.trading.base_contracts,
                max_loss_per_trade=self.config.trading.max_loss_per_trade,
                min_risk_reward=self.config.trading.min_risk_reward,
                max_trades_per_hour=self.config.trading.max_trades_per_hour,
                max_trades_per_day=self.config.trading.max_trades_per_day,
                min_seconds_between_trades=self.config.trading.min_seconds_between_trades,
                max_consecutive_losses=self.config.trading.max_consecutive_losses,
                loss_streak_cooldown_sec=self.config.trading.loss_streak_cooldown_seconds,
                point_value=self.instrument.point_value,
                tick_value=self.instrument.tick_value,
            ),
            mll_tracker=self.mll_tracker,
        )
        
        # Position state
        self.position_direction: Optional[Direction] = None
        self.position_entry_price: float = 0.0
        self.position_entry_time: float = 0.0
        self.position_contracts: int = 0
        self.position_strategy: str = ""
        # Grace-window anchor: wall-clock of the last local state transition
        # (entry fill OR exit). Used by the SignalR position-event reconciler
        # to avoid force-closing a freshly opened position when TopstepX emits
        # phantom `FLAT 0x @ 0.00` lifecycle events in the first seconds after
        # a fill. Updated on both entry and exit paths.
        self._last_local_state_change: float = 0.0
        
        # Account protection
        self.account_balance: Optional[float] = None
        self._last_balance_check: float = 0
        self._balance_check_interval: float = 60  # Check every 60s
        
        # Platform PNL (synced from API — source of truth)
        self.platform_pnl: dict = {}
        self.starting_balance: Optional[float] = None  # Captured at startup
        
        # Entry guard — prevents duplicate entries
        self._entering = False
        
        # Position sync — detect bracket fills and orphaned orders
        self._last_position_sync: float = 0
        self._position_sync_interval: float = 5  # Check every 5s
        
        # Real-time loss tracking per position
        self._position_worst_pnl: float = 0.0
        
        # Trailing stop manager (created per-position)
        self._trailing_stop: Optional[TrailingStopManager] = None
        
        # Position manager (multi-contract tranche system)
        self._position_manager: Optional[PositionManager] = None
        
        # Hybrid position manager (institutional-grade exits)
        self._hybrid_pm: Optional[HybridPositionManager] = None
        
        # Last market state from bar — used for tick-level signal invalidation
        self._last_bar_state: Optional[MarketState] = None
        
        # Stale data detection — if no quotes for N seconds, SignalR is dead
        self._last_quote_time: float = 0
        self._stale_threshold: float = 60  # No data for 60s = stale
        
        # Warmup — don't trade until we've seen N fresh candle closes after startup
        self._live_bar_count = 0
        self._warmup_bars = 2  # Must see 2 fresh candle closes before trading (~6 min on 3-min bars)
        self._startup_time = 0.0
        self._last_bar_timestamp = None  # Track bar timestamps to detect truly new candles
        
        # Stats
        self._bar_count = 0
        self._signal_count = 0
        self._quote_count = 0
        
        # Bar Recorder — saves 1-min bars to CSV for replay/backtesting
        from core.bar_recorder import BarRecorder
        self.bar_recorder = BarRecorder(
            data_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live"),
            instrument=self.instrument.instrument,
        )
    
    # ── Lifecycle ───────────────────────────────────────────────────────
    
    async def start(self):
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("  TOPSTEP5 HYBRID — Institutional Exit System")
        logger.info("  Risk Reduce / Signal Invalidation / Adaptive Trail")
        logger.info("=" * 60)
        
        # Auth
        if not self.conn.authenticate():
            logger.critical("Authentication failed — cannot start")
            return
        
        # Check account
        self.account_balance = self.conn.get_account_balance()
        self.starting_balance = self.account_balance
        self.sod_balance = self._load_or_set_sod_balance(self.account_balance)
        if self.account_balance:
            # Update MLL tracker with current balance — sets cushion, max contracts, daily loss limit
            self.risk.update_balance(self.account_balance)
            mll_status = self.mll_tracker.get_status(self.account_balance)
            # Push balance + SOD into the vol sizer for spot %-of-account sizing
            try:
                self.vol_sizer.account_balance = self.account_balance
                self.vol_sizer.sod_balance = self.sod_balance or self.account_balance
            except Exception:
                pass
            logger.info(f"Account balance: ${self.account_balance:,.2f}")
            logger.info(
                f"MLL: ${mll_status['mll']:,.2f} | Cushion: ${mll_status['cushion']:,.2f} | "
                f"{'LOCKED' if mll_status['mll_locked'] else 'TRAILING'} | "
                f"Max contracts: {mll_status['max_contracts']} | "
                f"Daily loss limit: ${mll_status['daily_loss_limit']:,.2f}"
            )
            if self.sod_balance and self.sod_balance != self.account_balance:
                startup_daily_pnl = self.account_balance - self.sod_balance
                logger.info(f"SOD balance: ${self.sod_balance:,.2f} | Today's P&L so far: ${startup_daily_pnl:+,.2f}")
                self.risk.daily_pnl = startup_daily_pnl
                self.risk.peak_pnl = max(0, startup_daily_pnl)
            if mll_status['cushion'] <= 0:
                logger.critical(
                    f"Balance ${self.account_balance:,.2f} is AT or BELOW MLL "
                    f"${mll_status['mll']:,.2f} — CANNOT TRADE"
                )
                return
            if mll_status['danger_zone']:
                logger.warning(
                    f"⚠️ DANGER ZONE: Cushion only ${mll_status['cushion']:,.2f} — "
                    f"trading with minimum size"
                )
        
        # Bootstrap historical bars
        logger.info("Loading historical bars...")
        try:
            bars = self.conn.get_bars(
                self.config.topstep.contract_id, bars_back=300, unit_number=3
            )
            for bar in bars:
                self.features.add_bar(bar)
            logger.info(f"Loaded {len(bars)} historical bars")
        except Exception as e:
            logger.error(f"Failed to load historical bars: {e}")
            logger.info("Continuing without historical data...")
        
        # Start status dashboard
        from live.status_server import start_status_server
        start_status_server(self)
        
        # Connect SignalR for real-time
        self.conn.set_callbacks(
            on_quote=self._on_quote,
            on_trade=self._on_trade_tick,
        )
        
        # Connect User Hub for real-time fills, orders, positions, balance
        self.conn.set_user_callbacks(
            on_user_trade=self._on_user_trade_event,
            on_user_order=self._on_user_order_event,
            on_user_position=self._on_user_position_event,
            on_user_account=self._on_user_account_event,
        )
        
        if getattr(self, '_poll_only', False):
            logger.info("Poll-only mode — skipping SignalR (no WebSocket session conflict)")
        else:
            signalr_ok = await self.conn.connect_signalr(self.config.topstep.contract_id)
            if not signalr_ok:
                logger.warning("No real-time data — running on polling mode")
            
            # User Hub — real-time order fills (no more guessing exit prices)
            user_hub_ok = await self.conn.connect_user_hub()
            if user_hub_ok:
                logger.info("[USER HUB] ✅ Connected — real-time fills active")
            else:
                logger.warning("[USER HUB] ❌ Failed to connect — falling back to position sync polling")
        
        # STARTUP SAFETY: Cancel any leftover bracket orders from previous sessions
        # Without this, old SL/TP brackets persist on the platform and can open phantom positions
        logger.info("[STARTUP] Cancelling any leftover working orders from previous sessions...")
        self.conn._cancel_all_working_orders(reason="startup cleanup - kill stale brackets")
        
        # Main loop
        self._startup_time = time.time()
        self.running = True
        logger.info("Bot is LIVE. Strategies active:")
        for s in self.strategies:
            logger.info(f"  → {s.name}")
        logger.info(
            f"Instrument: {self.instrument.instrument} ({self.instrument.contract_id}) | "
            f"${self.instrument.point_value}/point | "
            f"{'10 MES = 1 ES' if self.instrument.instrument == 'MES' else ('1 contract = 0.01 BTC' if self.instrument.instrument == 'BTC' else 'native ES')}"
        )
        logger.info(f"Risk: max {self.config.trading.max_contracts} {self.instrument.instrument} contracts, "
                    f"${self.risk.config.max_daily_loss:.0f} daily loss limit (dynamic)")

        # Chart capture — startup market review + hourly loop
        try:
            from analyzers.chart_capture import capture as _chart_capture, HourlyCapture
            _chart_capture("startup", blocking=False)
            self._hourly_capture = HourlyCapture(interval_minutes=60.0)
            self._hourly_capture.start()
        except Exception as _ce:
            logger.debug(f"Chart capture init skipped: {_ce}")

        await self._main_loop()
    
    async def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        # Close any open position
        if self.position_direction and self.position_direction != Direction.FLAT:
            logger.warning("Closing open position on shutdown...")
            self.conn.flatten_all(
                self.config.topstep.contract_id,
                known_side=self.position_direction.name,
                known_size=self.position_contracts
            )
        
        await self.conn.disconnect()
        
        # Fetch final balance from platform for accurate PNL
        final_balance = self.conn.get_account_balance()
        platform_session_pnl = 0.0
        if final_balance and self.starting_balance:
            platform_session_pnl = final_balance - self.starting_balance
        
        summary = self.risk.get_summary()
        logger.info("=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  Starting Balance: ${self.starting_balance:,.2f}" if self.starting_balance else "  Starting Balance: unknown")
        logger.info(f"  Final Balance:    ${final_balance:,.2f}" if final_balance else "  Final Balance: unknown")
        logger.info(f"  Platform P&L:     ${platform_session_pnl:+,.2f}")
        logger.info(f"  Bot-Tracked P&L:  ${summary['daily_pnl']:+,.2f} "
                    f"({summary['total_trades']} trades: {summary['winners']}W / {summary['losers']}L)")
        logger.info(f"  Bars processed: {self._bar_count}")
        if abs(platform_session_pnl - summary['daily_pnl']) > 10:
            logger.warning(f"  ⚠ PNL MISMATCH: Platform=${platform_session_pnl:+,.2f} vs Bot=${summary['daily_pnl']:+,.2f}")
        logger.info("=" * 60)
    
    # ── Main Loop ───────────────────────────────────────────────────────
    
    async def _main_loop(self):
        """Main trading loop. Polls for bars if no SignalR."""
        while self.running:
            try:
                # Auto-reconnect only if websocket actually dropped (skip in poll-only mode)
                if getattr(self.conn, '_needs_reconnect', False) and not getattr(self, '_poll_only', False):
                    self.conn._needs_reconnect = False
                    delays = self.conn._reconnect_delays
                    attempt = min(self.conn._reconnect_attempt, len(delays) - 1)
                    delay = delays[attempt]
                    self.conn._reconnect_attempt += 1
                    logger.info(f"Reconnecting SignalR in {delay}s (attempt {self.conn._reconnect_attempt})...")
                    await asyncio.sleep(delay)
                    # Close old connection cleanly
                    await self.conn.disconnect()
                    self.conn.session_token = None  # Force re-auth
                    if self.conn.authenticate():
                        ok = await self.conn.connect_signalr(self.config.topstep.contract_id)
                        if ok:
                            logger.info("SignalR reconnected successfully")
                            # Also reconnect User Hub
                            await self.conn.connect_user_hub()
                        else:
                            logger.warning("SignalR reconnect failed — will retry next loop")
                
                # Auto-reconnect User Hub independently
                if not self.conn._user_ws_connected and not getattr(self, '_poll_only', False):
                    await self.conn.reconnect_user_hub()
                
                # If no SignalR OR data is stale OR running poll-only, poll for new bars.
                # Poll-only throttled to every 15s to stay well under Alpaca's 200/min limit
                # (suitable for 3-min bars; in-position fast-poll handles tighter exit needs).
                data_stale = (self._last_quote_time > 0 and
                              time.time() - self._last_quote_time > self._stale_threshold)
                _now_for_poll = time.time()
                _poll_only = getattr(self, '_poll_only', False)
                _last_poll = getattr(self, '_last_poll_bars_time', 0.0)
                _poll_interval = getattr(self, '_poll_bars_interval', 15.0)
                _should_poll = (
                    (not self.conn.is_connected and (_now_for_poll - _last_poll) >= _poll_interval)
                    or data_stale
                    or (_poll_only and (_now_for_poll - _last_poll) >= _poll_interval)
                )
                if _should_poll:
                    if data_stale and self.conn.is_connected and not _poll_only:
                        logger.warning(
                            f"No quotes for {_now_for_poll - self._last_quote_time:.0f}s — "
                            f"SignalR likely stale, falling back to polling"
                        )
                        self.conn._needs_reconnect = True
                        self.conn.is_connected = False
                    await self._poll_bars()
                    self._last_poll_bars_time = _now_for_poll
                
                now = time.time()
                
                # Fast price poll when in a position (poll-only mode)
                # Gives ~1s trailing stop updates vs waiting for full bar polls
                if (self.position_direction and self.position_direction != Direction.FLAT
                        and (not self.conn.is_connected or getattr(self, '_poll_only', False))):
                    await self._poll_price_fast()
                
                # Position sync — catch orphaned brackets, bracket fills, stacking
                # When flat: poll every 30s (User Hub WebSocket handles real-time updates)
                # When in position: poll every 5s (need tight sync for exit detection)
                sync_interval = self._position_sync_interval if self.position_direction and self.position_direction.name != 'FLAT' else 30.0
                if now - self._last_position_sync > sync_interval:
                    await self._sync_position_state()
                    self._last_position_sync = now
                
                # Periodic balance check
                if now - self._last_balance_check > self._balance_check_interval:
                    await self._check_balance()
                    self._last_balance_check = now
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        await self.stop()
    
    async def _poll_bars(self):
        """Fallback: poll API for latest bars."""
        try:
            bars = self.conn.get_bars(
                self.config.topstep.contract_id, bars_back=5, unit_number=3
            )
            for bar in bars[-1:]:  # Just process latest
                self._process_bar(bar)
                # In poll mode, check loss on bar close price since we have no tick data
                price = bar.get("c") or bar.get("close")
                if price:
                    self._check_tick_loss(float(price))
        except Exception as e:
            logger.debug(f"Poll error: {e}")
    
    async def _poll_price_fast(self):
        """Fast price poll for trailing stop updates between bar polls.

        Fetches just 1 bar (lightweight) to get latest price. Runs every loop
        iteration when in a position, giving ~1s price updates for the trailing stop.
        """
        try:
            bars = self.conn.get_bars(
                self.config.topstep.contract_id, bars_back=1, unit_number=3
            )
            if bars:
                price = bars[-1].get("c") or bars[-1].get("close")
                if price:
                    self._check_tick_loss(float(price))
        except Exception:
            pass  # Silent — this is a best-effort fast poll
    
    # ── Data Handlers ───────────────────────────────────────────────────
    
    def _on_quote(self, *args):
        """Handle real-time quote from SignalR.
        
        GatewayQuote callback signature: (contractId, data)
        data fields: lastPrice, bestBid, bestAsk, change, open, high, low, volume, timestamp
        """
        self._quote_count += 1
        self._last_quote_time = time.time()
        try:
            # Parse args — SignalR sends (contractId, data) but signalrcore 
            # may wrap them differently depending on version
            quote = {}
            if args:
                if len(args) >= 2 and isinstance(args[1], dict):
                    # Standard: (contractId, data_dict)
                    quote = args[1]
                elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) >= 2:
                    # Wrapped: [[contractId, data_dict]]
                    quote = args[0][1] if isinstance(args[0][1], dict) else {}
                elif len(args) == 1 and isinstance(args[0], dict):
                    # Direct: (data_dict)
                    quote = args[0]
                elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], dict):
                    quote = args[0][0]
            
            price = (quote.get("lastPrice") or quote.get("lp") 
                     or quote.get("bestBid") or quote.get("bp") 
                     or quote.get("bestAsk") or quote.get("ap"))
            if price:
                price_f = float(price)
                completed_bar = self.tick_agg.on_tick(price_f)
                if completed_bar:
                    self._process_bar(completed_bar)
                
                # TICK-LEVEL LOSS MONITOR — catch runaway losses before bar completes
                self._check_tick_loss(price_f)
                
                # Log first few and then periodically
                if self._quote_count <= 3 or self._quote_count % 100 == 0:
                    logger.info(f"[LIVE] Quote #{self._quote_count} | Price: {price}")
        except Exception as e:
            if self._quote_count < 5:
                logger.error(f"Quote processing error: {e} | raw args: {str(args)[:300]}")
    
    def _on_trade_tick(self, *args):
        """Handle real-time trade tick from SignalR.
        
        GatewayTrade callback signature: (contractId, data)
        data fields: symbolId, price, timestamp, type (Buy=0/Sell=1), volume
        """
        try:
            trade = {}
            if args:
                if len(args) >= 2 and isinstance(args[1], dict):
                    trade = args[1]
                elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) >= 2:
                    trade = args[0][1] if isinstance(args[0][1], dict) else {}
                elif len(args) == 1 and isinstance(args[0], dict):
                    trade = args[0]
                elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], dict):
                    trade = args[0][0]
            
            price = trade.get("price") or trade.get("p") or trade.get("lp")
            vol = trade.get("volume") or trade.get("s") or trade.get("size") or 1
            if price:
                completed_bar = self.tick_agg.on_tick(float(price), int(vol))
                if completed_bar:
                    self._process_bar(completed_bar)
        except Exception as e:
            logger.debug(f"Trade tick error: {e}")
    
    # ── Core Logic ──────────────────────────────────────────────────────
    
    def _process_bar(self, bar: dict):
        """Process a completed bar — the heart of the bot."""
        # Detect truly new candles by timestamp (poll mode sends same bar repeatedly)
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
        
        # Need minimum bars for indicators
        if self.features.bar_count < 20:
            return
        
        # Warmup gate — don't trade until we have live market context
        if self._live_bar_count <= self._warmup_bars:
            if is_new_candle:
                warmup_elapsed = time.time() - self._startup_time if self._startup_time > 0 else 0
                logger.info(
                    f"[WARMUP] Fresh candle {self._live_bar_count}/{self._warmup_bars} detected — "
                    f"waiting for {self._warmup_bars} fresh candles before trading ({warmup_elapsed:.0f}s since start)"
                )
            return
        
        # Detect session
        try:
            from datetime import datetime as dt
            ts = bar.get("t", "")
            if isinstance(ts, str) and ts:
                bar_dt = dt.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                bar_dt = dt.now(timezone.utc)
        except:
            bar_dt = datetime.now(timezone.utc)
        
        session = detect_session(bar_dt)
        mins = minutes_since_rth_open(bar_dt)
        
        # Build market state
        state_dict = self.features.build_market_state(session, mins)
        state = MarketState(**state_dict)
        self._last_bar_state = state  # Cache for tick-level signal checks
        
        # If in position or currently entering → don't look for new entries
        if self.position_direction and self.position_direction != Direction.FLAT:
            self._check_exit(state)
            return
        if self._entering:
            return  # Already placing an order, skip
        
        # Anti-stacking: double-check we don't have an untracked platform position
        # (lightweight check — full sync happens in _sync_position_state)
        if hasattr(self, '_last_platform_pos_check'):
            if time.time() - self._last_platform_pos_check < 10:
                pass  # Skip if we checked recently
            else:
                self._last_platform_pos_check = time.time()
        else:
            self._last_platform_pos_check = time.time()
        
        # Not in position → detect regime, then check strategies for entry
        if is_new_candle:
            self._current_regime = self.regime_detector.detect(self.features, state)
        
        # Diagnostic logging every 30 bars (~30 min) to see why strategies aren't firing
        if is_new_candle and self._live_bar_count % 30 == 0:
            logger.info(
                f"[DIAG] Bar #{self._live_bar_count} | Price={state.price:.2f} | "
                f"ATR={state.atr_14:.2f} | ADX={state.adx_14:.1f} | RSI={state.rsi_14:.1f} | "
                f"EMA9={state.ema_9:.2f} EMA21={state.ema_21:.2f} EMA50={state.ema_50:.2f} | "
                f"5m_9={state.ema_5m_9:.2f} 5m_26={state.ema_5m_26:.2f} 5m_50={state.ema_5m_50:.2f} | "
                f"VWAP={state.vwap:.2f} vwap_std={state.vwap_std:.2f} | "
                f"vol_ratio={getattr(state, 'volume_ratio_5', 0):.2f} | "
                f"session={state.session} | regime={self._current_regime.regime.value if self._current_regime else 'N/A'}"
            )
        
        for strategy in self.strategies:
            signal = strategy.should_enter(state)
            if signal:
                self._signal_count += 1
                self._try_enter(signal, state)
                break  # One signal at a time
    
    def _try_enter(self, signal, market_state=None):
        """Attempt to enter a trade (strategy proposed, risk manager decides)."""
        try:
            self._try_enter_inner(signal, market_state)
        except Exception as e:
            logger.error(f"[ENTRY ERROR] {signal.strategy_name}: {e}", exc_info=True)
            self._entering = False

    def _try_enter_inner(self, signal, market_state=None):
        """Inner entry logic — wrapped by _try_enter for safety."""
        # Guard against duplicate entries from concurrent bar processing
        if self._entering:
            logger.debug(f"[SKIP] Already entering a trade, ignoring {signal.strategy_name}")
            return
        if self.position_direction and self.position_direction != Direction.FLAT:
            logger.debug(f"[SKIP] Already in position, ignoring {signal.strategy_name}")
            return
        
        self._entering = True
        should_trade, reason, contracts = self.risk.should_trade(
            signal, market_state, regime_state=self._current_regime
        )
        # NOTE: should_trade() already calls _calculate_size() internally.
        # Do NOT double-call it here — it was returning different values and
        # scaling up beyond what the risk manager approved.
        
        if not should_trade:
            logger.info(f"[BLOCKED] {signal.strategy_name} {signal.direction.name}: {reason}")
            self._entering = False
            return
        
        # Calculate stop/target in points
        stop_points = abs(signal.entry_price - signal.stop_loss)
        target_points = abs(signal.take_profit - signal.entry_price)
        
        # CAP BRACKET STOP — adaptive based on current ATR.
        # For BTC, ATR is in $-of-price (3-min ATR is typically $50–1000); the
        # cap bands are sized to BTC scale. In low vol (ATR ~$50), allow $200
        # min; in high vol (ATR ~$700), allow up to ~$2000.
        # Original ES values (4.0 / 6.0 pts) were $4–$6 on BTC — guaranteed
        # instant stop-out.
        current_atr = self.features.atr() if self.features.bar_count > 20 else 200.0
        MAX_BRACKET_SL_PTS = min(max(current_atr * 1.5, 200.0), 2000.0)
        if stop_points > MAX_BRACKET_SL_PTS:
            logger.warning(
                f"[BRACKET CAP] Strategy SL was {stop_points:.2f}pts, "
                f"capped to {MAX_BRACKET_SL_PTS}pt hard max"
            )
            stop_points = MAX_BRACKET_SL_PTS
            # Recalculate R:R with capped stop — if it's now too low, block
            if target_points > 0 and stop_points > 0:
                capped_rr = target_points / stop_points
                if capped_rr < self.config.trading.min_risk_reward:
                    logger.info(
                        f"[BLOCKED] R:R after bracket cap = {capped_rr:.2f} "
                        f"< {self.config.trading.min_risk_reward}. Skipping."
                    )
                    self._entering = False
                    return
        
        # Place order with bracket
        side = "BUY" if signal.direction == Direction.LONG else "SELL"
        success = self.conn.place_order(
            contract_id=self.config.topstep.contract_id,
            side=side,
            size=contracts,
            stop_loss_points=stop_points,
            take_profit_points=target_points,
            current_price=signal.entry_price,
        )
        
        # place_order returns order_id (int) on success, None/False on failure
        order_result = success
        if order_result:
            self.position_direction = signal.direction
            self.position_entry_time = time.time()
            self.position_contracts = contracts
            self.position_strategy = signal.strategy_name
            self._last_local_state_change = time.time()
            self._position_max_profit = 0.0  # Reset watermark
            self._position_worst_pnl = 0.0  # Reset loss tracker
            placed_order_id = order_result if isinstance(order_result, int) else None
            
            # Create position manager (multi-contract) or trailing stop (single)
            current_atr = self.features.atr() if self.features.bar_count > 20 else 2.0
            direction_mult = 1 if signal.direction == Direction.LONG else -1
            
            # HYBRID SYSTEM: Use HybridPositionManager for all multi-contract trades
            # Falls back to TrailingStop for single-contract only
            use_hybrid = contracts >= 3
            
            if use_hybrid:
                self._hybrid_pm = HybridPositionManager(
                    entry_price=signal.entry_price,
                    direction=direction_mult,
                    total_contracts=contracts,
                    stop_distance=stop_points,
                    config=HybridConfig(
                        point_value=self.instrument.point_value,
                        instrument=self.instrument.instrument,
                        max_stop_pts=min(stop_points, MAX_BRACKET_SL_PTS),
                    ),
                )
                self._trailing_stop = None
                self._position_manager = None
            else:
                # Single/tiny contract: use TrailingStopManager directly
                self._trailing_stop = TrailingStopManager(
                    config=TrailingStopConfig(
                        point_value=self.instrument.point_value,
                        max_stop_pts=min(
                            self.config.trading.max_loss_per_trade / self.instrument.point_value,
                            MAX_BRACKET_SL_PTS
                        ),
                    ),
                    entry_price=signal.entry_price,
                    direction=direction_mult,
                    atr=current_atr,
                )
                self._hybrid_pm = None
                self._position_manager = None
            
            # Get ACTUAL fill price from platform (not the signal's estimated price)
            fill_price = self.conn.get_fill_price(self.config.topstep.contract_id, order_id=placed_order_id)
            if fill_price:
                self.position_entry_price = fill_price
                # Update hybrid PM, position manager, or trailing stop with actual fill
                if self._hybrid_pm and abs(fill_price - signal.entry_price) > 0.01:
                    self._hybrid_pm.update_entry_price(fill_price)
                    logger.info(f"[HYBRID] Updated entry to fill price {fill_price:.2f}")
                elif self._position_manager and abs(fill_price - signal.entry_price) > 0.01:
                    self._position_manager.update_entry_price(fill_price, current_atr)
                    logger.info(f"[POSITION] Updated entry to fill price {fill_price:.2f}")
                elif self._trailing_stop and abs(fill_price - signal.entry_price) > 0.01:
                    self._trailing_stop.entry_price = fill_price
                    initial_distance = self._trailing_stop._clamp_distance(
                        current_atr * self._trailing_stop.config.initial_stop_atr_mult
                    )
                    if direction_mult == 1:
                        self._trailing_stop.current_stop_price = fill_price - initial_distance
                    else:
                        self._trailing_stop.current_stop_price = fill_price + initial_distance
                    logger.info(
                        f"[TRAIL] Updated entry to fill price {fill_price:.2f}, "
                        f"stop recalculated to {self._trailing_stop.current_stop_price:.2f}"
                    )
                if abs(fill_price - signal.entry_price) > 0.5:
                    logger.warning(
                        f"FILL SLIPPAGE: Signal estimated {signal.entry_price:.2f}, "
                        f"actual fill {fill_price:.2f} "
                        f"({abs(fill_price - signal.entry_price):.2f}pts slip)"
                    )
            else:
                # Fallback to signal price if we can't get fill
                self.position_entry_price = signal.entry_price
                logger.warning(f"Could not get fill price from platform, using signal price {signal.entry_price:.2f}")
            
            logger.info(
                f"ENTERED: {side} {contracts}x @ {self.position_entry_price:.2f} "
                f"| SL: {signal.stop_loss:.2f} ({stop_points:.1f}pts) "
                f"| TP: {signal.take_profit:.2f} ({target_points:.1f}pts) "
                f"| R:R: {signal.risk_reward:.2f} "
                f"| Strategy: {signal.strategy_name} "
                f"| Reason: {signal.reason}"
            )
            self._entering = False  # Clear after position state is set
        else:
            logger.error(f"ORDER FAILED: {side} {contracts}x — check connection")
            self._entering = False
    
    def _check_exit(self, state: MarketState):
        """Check if current position should be exited.
        
        HYBRID system (primary):
        - Risk reduction tranche: math-based, exits when profit covers remaining risk
        - Core tranche: signal-based, exits when strategy says edge is gone
        - Runner tranche: adaptive trail based on regime/volume/time
        
        Falls back to TrailingStopManager for tiny positions.
        """
        hold_time = time.time() - self.position_entry_time
        
        # ── Hybrid Position Manager ─────────────────────────────────
        if self._hybrid_pm:
            # Get signal invalidation from the active strategy
            signal_exit_reason = None
            for strategy in self.strategies:
                if strategy.name == self.position_strategy:
                    signal_exit_reason = strategy.should_exit(
                        state, self.position_entry_price,
                        self.position_direction, hold_time
                    )
                    break
            
            # Determine regime from market state
            regime = getattr(state, 'regime', 'unknown')
            if regime == 'unknown':
                # Derive from EMA spread
                if state.ema_5m_9 > 0 and state.ema_5m_50 > 0:
                    spread = abs(state.ema_5m_9 - state.ema_5m_50)
                    if spread < 3.0:
                        regime = "ranging"
                    elif state.atr_14 > 4.0:
                        regime = "volatile"
                    else:
                        regime = "trending"
            
            volume_ratio = state.volume_ratio_5 if state.volume_ratio_5 > 0 else 1.0
            
            exits = self._hybrid_pm.update(
                current_price=state.price,
                signal_exit_reason=signal_exit_reason,
                regime=regime,
                volume_ratio=volume_ratio,
                atr=state.atr_14 if state.atr_14 > 0 else 2.0,
            )
            
            for tranche_exit in exits:
                if self._hybrid_pm.is_fully_closed():
                    logger.info(
                        f"[HYBRID] Last tranche ({tranche_exit.role}) exiting -- flatten all"
                    )
                    self._exit_position(state.price,
                        f"[HYBRID:{tranche_exit.role}] {tranche_exit.reason}")
                    return
                else:
                    logger.info(
                        f"[HYBRID] Partial close: {tranche_exit.role} "
                        f"{tranche_exit.size} contracts | {tranche_exit.reason}"
                    )
                    success = self.conn.partial_close(
                        self.config.topstep.contract_id,
                        tranche_exit.size
                    )
                    if success:
                        self.position_contracts = self._hybrid_pm.remaining_contracts
                        # RESIZE BRACKETS after partial close — platform doesn't auto-adjust!
                        # Without this, brackets stay at original size and create phantom positions.
                        remaining = self._hybrid_pm.remaining_contracts
                        logger.info(f"[BRACKET] Post-partial: remaining={remaining}, calling resize...")
                        if remaining > 0:
                            self._resize_brackets(remaining)
                        pnl_dollars = tranche_exit.pnl_pts * self.instrument.point_value * tranche_exit.size
                        trade = TradeRecord(
                            entry_time=self.position_entry_time,
                            exit_time=time.time(),
                            direction=self.position_direction,
                            entry_price=self.position_entry_price,
                            exit_price=state.price,
                            contracts=tranche_exit.size,
                            pnl_dollars=pnl_dollars,
                            strategy=self.position_strategy,
                            exit_reason=f"[HYBRID:{tranche_exit.role}] {tranche_exit.reason}",
                        )
                        self.risk.record_trade(trade)
                        logger.info(
                            f"[HYBRID] {tranche_exit.role.upper()} closed: "
                            f"{tranche_exit.size}x @ {state.price:.2f} | "
                            f"P&L: ${pnl_dollars:+,.2f} | "
                            f"Remaining: {self._hybrid_pm.remaining_contracts}"
                        )
                    else:
                        logger.error(
                            f"[HYBRID] Partial close FAILED for {tranche_exit.role} -- "
                            f"flattening all for safety"
                        )
                        self._exit_position(state.price, "Partial close failed -- safety flatten")
                        return
            
            if self._hybrid_pm.is_fully_closed():
                self._exit_position(state.price, "All hybrid tranches closed")
                return
            
            return  # Hybrid handles everything -- don't fall through
        
        # ── Single-Contract: Trailing Stop System (Primary) ─────────
        if self._trailing_stop:
            current_atr = state.atr_14 if state.atr_14 > 0 else self._trailing_stop.entry_atr
            exit_reason = self._trailing_stop.update(state.price, current_atr)
            if exit_reason:
                self._exit_position(state.price, f"[TRAILING] {exit_reason}")
                return
        
        # ── EMA-Aware Exit (supplemental — only in profit) ──────────
        # Skip for trend-following strategies — let trailing stop + brackets manage
        if self.position_strategy not in ("EMA_REJECT",):
            ema_exit = self._check_ema_exit(state)
            if ema_exit:
                self._exit_position(state.price, ema_exit)
                return
        
        # ── Strategy-Specific Exit ──────────────────────────────────
        for strategy in self.strategies:
            if strategy.name == self.position_strategy:
                exit_reason = strategy.should_exit(
                    state, self.position_entry_price,
                    self.position_direction, hold_time
                )
                if exit_reason:
                    self._exit_position(state.price, exit_reason)
                break
    
    def _check_ema_exit(self, state: MarketState) -> Optional[str]:
        """Check if price is approaching an EMA that will bounce it.
        
        Logic:
        - SHORT: if price dropping toward an EMA below → that's support → take profit
        - LONG: if price rising toward an EMA above → that's resistance → take profit
        
        Only triggers when trade is in profit (don't exit losers at EMAs).
        Uses multi-timeframe EMAs (3m, 5m are stronger S/R than 1m).
        """
        if not state.ema_levels:
            return None
        
        mult = 1 if self.position_direction == Direction.LONG else -1
        unrealized_pts = (state.price - self.position_entry_price) * mult
        
        # Only manage winners — need at least 1 point profit
        if unrealized_pts < 1.0:
            return None
        
        atr = state.atr_14 if state.atr_14 > 0 else 2.0
        proximity_threshold = atr * 0.4  # Within 40% of ATR = "approaching"
        
        # Weight EMAs by timeframe significance (higher TF = stronger S/R)
        ema_weights = {
            "1m_": 0.5, "3m_": 1.0, "5m_": 1.5,
        }
        
        for label, ema_price in state.ema_levels.items():
            if ema_price <= 0:
                continue
            
            # Determine weight — higher TF EMAs are stronger
            weight = 0.5
            for prefix, w in ema_weights.items():
                if label.startswith(prefix):
                    weight = w
                    break
            
            # 200 EMAs are extra strong regardless of timeframe
            if "200" in label:
                weight *= 2.0
            
            distance = abs(state.price - ema_price)
            threshold = proximity_threshold / weight  # Stronger EMAs → exit earlier
            
            if self.position_direction == Direction.SHORT:
                # EMA below price = support = bounce point
                if ema_price < state.price and distance < threshold:
                    return (f"EMA exit: approaching {label} EMA @ {ema_price:.2f} "
                            f"(support, {distance:.1f}pts away, +{unrealized_pts:.1f}pts profit)")
            
            elif self.position_direction == Direction.LONG:
                # EMA above price = resistance = rejection point
                if ema_price > state.price and distance < threshold:
                    return (f"EMA exit: approaching {label} EMA @ {ema_price:.2f} "
                            f"(resistance, {distance:.1f}pts away, +{unrealized_pts:.1f}pts profit)")
    
    def _resize_brackets(self, new_size: int):
        """Modify bracket order sizes in-place via /api/Order/modify. Position never unprotected."""
        logger.info(f"[BRACKET RESIZE] Modifying brackets to {new_size} contracts...")
        try:
            orders = self.conn.get_working_orders()
            if not orders:
                logger.warning("[BRACKET RESIZE] No working orders found")
                return
            
            for o in orders:
                otype = o.get("type")
                oid = o.get("id")
                current_size = o.get("size", 0)
                
                if otype not in (1, 4) or not oid:  # Only modify SL (4) and TP (1)
                    continue
                
                if current_size == new_size:
                    continue  # Already correct
                
                label = "SL" if otype == 4 else "TP"
                payload = {
                    "accountId": int(self.conn.account_id),
                    "orderId": oid,
                    "size": new_size,
                }
                resp = requests.post(
                    f"{self.conn.api_endpoint}/api/Order/modify",
                    json=payload, headers=self.conn._headers(), timeout=15
                )
                result = resp.json() if resp.status_code == 200 else {}
                if result.get("success"):
                    logger.info(f"[BRACKET RESIZE] {label} {oid}: {current_size} → {new_size}")
                else:
                    logger.error(f"[BRACKET RESIZE] {label} modify failed: {result}")
                    
        except Exception as e:
            logger.error(f"[BRACKET RESIZE] Error: {e}", exc_info=True)

    def _exit_position(self, exit_price: float, reason: str):
        """Close current position and clean up ALL working orders."""
        if not self.position_direction or self.position_direction == Direction.FLAT:
            return  # Already flat, nothing to do
        
        logger.info(
            f"EXIT TRIGGERED: {self.position_strategy} {self.position_direction.name} "
            f"@ {exit_price:.2f} | Reason: {reason}"
        )
        
        # Flatten via API — cancels brackets, places opposite market order, cleans up
        self.conn.flatten_all(
            self.config.topstep.contract_id,
            known_side=self.position_direction.name,
            known_size=self.position_contracts
        )
        
        # Calculate P&L
        mult = 1 if self.position_direction == Direction.LONG else -1
        pnl_points = (exit_price - self.position_entry_price) * mult
        pnl_dollars = pnl_points * self.instrument.point_value * self.position_contracts
        
        # Record trade
        trade = TradeRecord(
            entry_time=self.position_entry_time,
            exit_time=time.time(),
            direction=self.position_direction,
            entry_price=self.position_entry_price,
            exit_price=exit_price,
            contracts=self.position_contracts,
            pnl_dollars=pnl_dollars,
            strategy=self.position_strategy,
            exit_reason=reason,
        )
        self.risk.record_trade(trade)
        
        logger.info(
            f"EXITED: {self.position_strategy} @ {exit_price:.2f} "
            f"| P&L: ${pnl_dollars:+,.2f} ({pnl_points:+.2f}pts) "
            f"| Worst unrealized: ${self._position_worst_pnl:+,.2f} "
            f"| Reason: {reason}"
        )
        
        # Log trailing stop final status
        if self._trailing_stop:
            status = self._trailing_stop.get_status()
            logger.info(
                f"[TRAIL] Final: stage={status['stage']} watermark={status['high_watermark_pts']:.2f}pts "
                f"stop={status['stop_price']:.2f} hold={status['hold_seconds']:.0f}s"
            )
        
        # Log position manager final status
        if self._position_manager:
            pm_status = self._position_manager.get_status()
            logger.info(f"[POSITION] Final status: {pm_status}")
        
        # Log hybrid position manager final status
        if self._hybrid_pm:
            hpm_status = self._hybrid_pm.get_status()
            logger.info(f"[HYBRID] Final status: {hpm_status}")
        
        # Reset position
        self.position_direction = None
        self.position_entry_price = 0.0
        self.position_entry_time = 0.0
        self.position_contracts = 0
        self.position_strategy = ""
        self._last_local_state_change = time.time()
        self._position_worst_pnl = 0.0
        self._trailing_stop = None
        self._position_manager = None
        self._hybrid_pm = None
        self._entering = False  # Ensure entry guard is cleared

        # Chart capture — analyze market context at trade close
        try:
            from analyzers.chart_capture import capture as _chart_capture
            _chart_capture("trade_close", context={
                "direction":  self.position_direction.name if self.position_direction else "?",
                "entry":      f"{self.position_entry_price:.2f}",
                "exit":       f"{exit_price:.2f}",
                "reason":     reason[:60],
                "pnl_pts":    f"{(exit_price - self.position_entry_price) * (1 if str(self.position_direction) == 'LONG' else -1):.2f}",
            })
        except Exception as _ce:
            logger.debug(f"Chart capture skipped: {_ce}")

        # Immediate orphan sweep — flatten_all cancels brackets, but if the position
        # was closed externally (bracket fill) any residual orders won't be caught
        # until the next sync cycle. Cancel now, don't wait.
        try:
            self.conn._cancel_all_working_orders(reason="post-exit orphan sweep")
        except Exception as e:
            logger.warning(f"Post-exit orphan sweep failed: {e}")

        # Delayed sweeps — catch race conditions where brackets are placed during
        # partial exits and aren't yet visible to the immediate cancel.
        # Sweep at 3s, 9s, and 21s (cumulative) to handle slow exchange ACKs.
        import threading
        import time as _time

        def _delayed_orphan_sweep():
            for sleep_secs, label in ((3.0, "3s"), (6.0, "9s"), (12.0, "21s")):
                _time.sleep(sleep_secs)
                try:
                    self.conn._cancel_all_working_orders(
                        reason=f"post-exit delayed orphan sweep ({label})"
                    )
                except Exception as e:
                    logger.warning(f"Delayed orphan sweep ({label}) failed: {e}")

        threading.Thread(target=_delayed_orphan_sweep, daemon=True).start()

    def _check_tick_loss(self, price: float):
        """Tick-level loss monitoring — emergency exit if unrealized loss exceeds max.
        
        This runs on EVERY quote, not just bar completion. Catches fast moves
        that could blow through the bar-level stop.
        """
        if not self.position_direction or self.position_direction == Direction.FLAT:
            return
        
        mult = 1 if self.position_direction == Direction.LONG else -1
        unrealized_pts = (price - self.position_entry_price) * mult
        unrealized_dollars = unrealized_pts * self.instrument.point_value * self.position_contracts
        
        # Track worst P&L for logging
        self._position_worst_pnl = min(self._position_worst_pnl, unrealized_dollars)
        
        # HYBRID POSITION MANAGER: update on every tick with full market awareness
        # Uses cached MarketState from last bar for signal invalidation + regime
        if self._hybrid_pm:
            signal_exit_reason = None
            regime = "unknown"
            volume_ratio = 1.0
            
            if self._last_bar_state:
                # Get signal invalidation from the active strategy using last bar state
                hold_time = time.time() - self.position_entry_time
                for strategy in self.strategies:
                    if strategy.name == self.position_strategy:
                        signal_exit_reason = strategy.should_exit(
                            self._last_bar_state, self.position_entry_price,
                            self.position_direction, hold_time
                        )
                        break
                
                # Derive regime from last bar state
                regime = getattr(self._last_bar_state, 'regime', 'unknown')
                if regime == 'unknown':
                    if self._last_bar_state.ema_5m_9 > 0 and self._last_bar_state.ema_5m_50 > 0:
                        spread = abs(self._last_bar_state.ema_5m_9 - self._last_bar_state.ema_5m_50)
                        if spread < 3.0:
                            regime = "ranging"
                        elif self._last_bar_state.atr_14 > 4.0:
                            regime = "volatile"
                        else:
                            regime = "trending"
                
                volume_ratio = self._last_bar_state.volume_ratio_5 if self._last_bar_state.volume_ratio_5 > 0 else 1.0
            
            exits = self._hybrid_pm.update(
                current_price=price,
                signal_exit_reason=signal_exit_reason,
                regime=regime,
                volume_ratio=volume_ratio,
            )
            for tranche_exit in exits:
                if self._hybrid_pm.is_fully_closed():
                    self._exit_position(price, f"[HYBRID:{tranche_exit.role}] {tranche_exit.reason}")
                    return
                else:
                    success = self.conn.partial_close(
                        self.config.topstep.contract_id, tranche_exit.size
                    )
                    if success:
                        self.position_contracts = self._hybrid_pm.remaining_contracts
                        # RESIZE BRACKETS after partial close (tick-level path)
                        remaining = self._hybrid_pm.remaining_contracts
                        logger.info(f"[BRACKET] Tick-level partial: remaining={remaining}, resizing...")
                        if remaining > 0:
                            self._resize_brackets(remaining)
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
                            exit_reason=f"[HYBRID:{tranche_exit.role}] {tranche_exit.reason}",
                        )
                        self.risk.record_trade(trade)
                    else:
                        self._exit_position(price, "Partial close failed -- safety flatten")
                        return
            if self._hybrid_pm.is_fully_closed():
                self._exit_position(price, "All hybrid tranches closed")
                return
            return  # Hybrid handles everything
        
        # TRAILING STOP: update on every tick for responsive exits (single contract)
        if self._trailing_stop:
            current_atr = self.features.atr() if self.features.bar_count > 20 else self._trailing_stop.entry_atr
            exit_reason = self._trailing_stop.update(price, current_atr)
            if exit_reason:
                self._exit_position(price, f"[TRAILING] {exit_reason}")
                return
        
        # HARD STOP: if unrealized loss exceeds max_loss_per_trade → emergency exit
        if unrealized_dollars <= -self.config.trading.max_loss_per_trade:
            logger.critical(
                f"EMERGENCY EXIT: Unrealized loss ${unrealized_dollars:,.2f} exceeds "
                f"max ${self.config.trading.max_loss_per_trade:.0f} | "
                f"Price: {price:.2f} | Entry: {self.position_entry_price:.2f}"
            )
            self._exit_position(price, f"Emergency tick-level stop: ${unrealized_dollars:,.2f} loss")
    
    def _load_or_set_sod_balance(self, current_balance: float) -> Optional[float]:
        """Load start-of-day balance from file. If it's a new CME session, save current balance as SOD.

        A2 (2026-04-24): anchored to the CME trading day (5pm ET boundary)
        rather than UTC midnight. Pre-A2 the UTC boundary crossed mid-session
        (00:00 UTC = 8pm EDT / 7pm EST) so evening P&L was mis-attributed
        across two "UTC days," which in turn double-seeded daily_pnl at the
        next session's startup. CME session labeling uses the trading day
        the session ENDS in: anything ≥ 17:00 ET belongs to tomorrow's session.
        """
        import json
        from datetime import datetime, timezone, timedelta
        try:
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            # Fallback: assume EDT (UTC-4). Wrong by 1hr in EST winter but
            # still correct for the session-boundary decision in practice.
            et_now = datetime.now(timezone.utc) + timedelta(hours=-4)

        if et_now.hour < 17:
            session_date = et_now.date()
        else:
            session_date = et_now.date() + timedelta(days=1)
        session_key = session_date.strftime("%Y-%m-%d")

        sod_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sod_balance.json")

        try:
            os.makedirs(os.path.dirname(sod_file), exist_ok=True)
            if os.path.exists(sod_file):
                with open(sod_file, "r") as f:
                    data = json.load(f)
                if data.get("session_date") == session_key:
                    logger.info(
                        f"Loaded SOD balance from file: ${data['balance']:,.2f} "
                        f"(session {session_key}, set at {data.get('time', '?')})"
                    )
                    return data["balance"]
                # Old UTC-keyed files had 'date' not 'session_date' — ignore
                # them so we re-seed cleanly the first time after the fix.
        except Exception as e:
            logger.warning(f"Error reading SOD file: {e}")

        # New session or no file — save current balance as SOD
        try:
            sod_data = {
                "session_date": session_key,
                "balance": current_balance,
                "time": datetime.now(timezone.utc).isoformat(),
                "session_anchor": "17:00 America/New_York",
            }
            with open(sod_file, "w") as f:
                json.dump(sod_data, f, indent=2)
            logger.info(f"Saved SOD balance: ${current_balance:,.2f} for CME session {session_key}")
        except Exception as e:
            logger.warning(f"Error saving SOD file: {e}")

        return current_balance

    async def _sync_position_state(self):
        """Sync local position state with platform reality.
        
        Fixes:
        - Orphaned brackets: if platform shows no position but bot thinks we're in one
        - Anti-stacking: if platform shows a position but bot thinks we're flat
        - Catches bracket fills the bot didn't record
        """
        try:
            positions = self.conn.get_open_positions()
            working_orders = self.conn.get_working_orders()

            # None = API returned 404 or error = unknown state, do not act
            # []   = API confirmed flat
            # [..] = confirmed position exists
            position_api_unknown = positions is None
            has_platform_position = bool(positions)  # False for None and []
            has_local_position = (self.position_direction is not None
                                  and self.position_direction != Direction.FLAT)

            # Case 1: Bot thinks we're in a position, but platform says no
            # → CRITICAL: Only act if position API gave a confident 200 response (not 404/error)
            # → If brackets still exist, position is still open regardless of position API
            time_in_position = time.time() - self.position_entry_time if self.position_entry_time > 0 else 0
            if has_local_position and not has_platform_position and time_in_position > 30:
                if position_api_unknown:
                    # API returned 404/error — do not interpret as flat, skip this sync cycle
                    logger.debug(
                        f"Position sync: position API returned unknown (404/error) — "
                        f"skipping sync to avoid false close of {self.position_direction.name} {self.position_contracts}x"
                    )
                elif working_orders:
                    # Confirmed flat by API but brackets exist — API is lagging
                    logger.debug(
                        f"Position sync: platform shows flat but {len(working_orders)} "
                        f"brackets exist — position API likely lagging, ignoring"
                    )
                else:
                    # Confirmed flat by API AND no brackets → position truly closed
                    logger.warning(
                        f"POSITION SYNC: Platform flat, no brackets, bot thinks "
                        f"{self.position_direction.name} {self.position_contracts}x "
                        f"(held {time_in_position:.0f}s). Position closed externally."
                    )
                    last_price = self.features.last_price if self.features.last_price > 0 else self.position_entry_price
                    self._exit_position(last_price, "Position sync — closed externally (no brackets remaining)")
            
            # Case 2: Bot thinks we're flat, but there are orphaned working orders
            # → Only cancel if bot has been flat for a while (avoid killing fresh brackets
            #   when position API is slower than order API)
            time_since_last_trade = time.time() - self.risk.last_trade_time if self.risk.last_trade_time > 0 else 999
            if not has_local_position and not has_platform_position and working_orders and time_since_last_trade > 30:
                logger.warning(
                    f"ORPHAN CLEANUP: Flat for {time_since_last_trade:.0f}s but "
                    f"{len(working_orders)} working orders found. Cancelling."
                )
                self.conn._cancel_all_working_orders(reason="sync case2 orphan cleanup")
            
            # Case 3: Bot thinks we're flat, but platform shows a position
            # → Something unexpected. Flatten for safety.
            if not has_local_position and has_platform_position:
                logger.warning(
                    f"POSITION SYNC: Platform shows position but bot is flat. "
                    f"Flattening for safety. Positions: {positions}"
                )
                self.conn.flatten_all(self.config.topstep.contract_id)
            
            # Sync platform PNL + balance → risk manager uses real numbers.
            # A4 (2026-04-24): session-anchored delta only. If we don't have
            # a SOD balance (first startup, corrupt file), we skip sync_platform_pnl
            # and only update the MLL tracker's balance. We do NOT pass platform
            # realizedPnL through — it spans the broker day and contaminates
            # session P&L. Better to trust local record_trade + add_tranche_pnl.
            pnl_data = self.conn.get_account_pnl()
            if pnl_data:
                self.platform_pnl = pnl_data
                current_bal = pnl_data.get("balance", 0)
                if current_bal and self.sod_balance:
                    balance_based_pnl = current_bal - self.sod_balance
                    self.risk.sync_platform_pnl(balance_based_pnl, current_balance=current_bal)
                elif current_bal:
                    # No SOD — just keep MLL tracker current; don't adopt
                    # platform realizedPnL (A4 invariant).
                    self.risk.update_balance(current_bal)
                
        except Exception as e:
            logger.error(f"Position sync error: {e}", exc_info=True)
    
    # ── User Hub Event Handlers (Real-Time Fills) ─────────────────────

    def _on_user_trade_event(self, data):
        """Handle GatewayUserTrade — real-time fill with exact price and P&L.
        
        This is the PRIMARY source of truth for bracket fills.
        Called when any trade executes on our account.
        
        data fields: id, accountId, contractId, creationTimestamp, price,
                     profitAndLoss, fees, side, size, voided, orderId
        side: 0=Bid(buy), 1=Ask(sell)
        profitAndLoss: null for opening trades, value for closing trades
        """
        try:
            if isinstance(data, list):
                data = data[0] if data else {}
            
            trade_id = data.get("id", "?")
            order_id = data.get("orderId", "?")
            price = data.get("price", 0)
            pnl = data.get("profitAndLoss")
            fees = data.get("fees", 0)
            side = data.get("side", -1)
            size = data.get("size", 0)
            voided = data.get("voided", False)
            side_str = "BUY" if side == 0 else "SELL" if side == 1 else f"SIDE={side}"
            
            if voided:
                logger.warning(f"[USER HUB] VOIDED TRADE: #{trade_id} | orderId={order_id}")
                return
            
            if pnl is not None:
                # Closing trade — bracket fill (SL or TP)
                logger.info(
                    f"[USER HUB] ✅ FILL (CLOSE): {side_str} {size}x @ {price:.2f} | "
                    f"P&L=${pnl:+,.2f} | Fees=${fees:.2f} | "
                    f"orderId={order_id} | tradeId={trade_id}"
                )
            else:
                # Opening trade — entry fill
                logger.info(
                    f"[USER HUB] ✅ FILL (OPEN): {side_str} {size}x @ {price:.2f} | "
                    f"Fees=${fees:.2f} | orderId={order_id} | tradeId={trade_id}"
                )
                
        except Exception as e:
            logger.error(f"[USER HUB] Trade event error: {e} | data={data}", exc_info=True)

    def _on_user_order_event(self, data):
        """Handle GatewayUserOrder — order status changes (fills, cancels, rejects).
        
        data fields: id, accountId, contractId, symbolId, creationTimestamp,
                     updateTimestamp, status, type, side, size, limitPrice,
                     stopPrice, fillVolume, filledPrice, customTag
        status: 0=None, 1=Open, 2=Filled, 3=Cancelled, 4=Expired, 5=Rejected, 6=Pending
        type: 1=Limit, 2=Market, 3=StopLimit, 4=Stop, 5=TrailingStop
        """
        try:
            if isinstance(data, list):
                data = data[0] if data else {}
            
            order_id = data.get("id", "?")
            status = data.get("status", 0)
            order_type = data.get("type", 0)
            side = data.get("side", -1)
            size = data.get("size", 0)
            filled_price = data.get("filledPrice")
            fill_volume = data.get("fillVolume", 0)
            limit_price = data.get("limitPrice")
            stop_price = data.get("stopPrice")
            
            status_names = {0: "NONE", 1: "OPEN", 2: "FILLED", 3: "CANCELLED", 
                          4: "EXPIRED", 5: "REJECTED", 6: "PENDING"}
            type_names = {0: "Unknown", 1: "Limit", 2: "Market", 3: "StopLimit",
                        4: "Stop", 5: "TrailingStop"}
            side_str = "BUY" if side == 0 else "SELL" if side == 1 else f"SIDE={side}"
            status_str = status_names.get(status, f"STATUS={status}")
            type_str = type_names.get(order_type, f"TYPE={order_type}")
            
            price_info = ""
            if filled_price:
                price_info = f" | filledPrice={filled_price:.2f}"
            elif limit_price:
                price_info = f" | limitPrice={limit_price:.2f}"
            elif stop_price:
                price_info = f" | stopPrice={stop_price:.2f}"
            
            if status == 2:  # Filled
                logger.info(
                    f"[USER HUB] 📋 ORDER FILLED: #{order_id} | {type_str} {side_str} {size}x"
                    f"{price_info} | fillVol={fill_volume}"
                )
            elif status == 5:  # Rejected
                logger.warning(
                    f"[USER HUB] ❌ ORDER REJECTED: #{order_id} | {type_str} {side_str} {size}x"
                    f"{price_info}"
                )
            elif status == 3:  # Cancelled
                logger.info(
                    f"[USER HUB] 🚫 ORDER CANCELLED: #{order_id} | {type_str} {side_str} {size}x"
                    f"{price_info}"
                )
            else:
                logger.debug(
                    f"[USER HUB] 📋 ORDER {status_str}: #{order_id} | {type_str} {side_str} {size}x"
                    f"{price_info}"
                )
                
        except Exception as e:
            logger.error(f"[USER HUB] Order event error: {e} | data={data}", exc_info=True)

    def _on_user_position_event(self, data):
        """Handle GatewayUserPosition — authoritative reconciler.

        Fires on every platform position change. Compares platform state to
        bot-local state and acts on divergence:
          - Platform has position, bot is FLAT → ghost position, flatten_all
          - Platform FLAT, bot holds position → bracket/external close, sync local
          - Direction or size mismatch → flatten_all for safety

        A short grace window after our own orders prevents racing with in-flight
        entries/scale-ins whose local state hasn't been committed yet.

        data fields: id, accountId, contractId, creationTimestamp, type, size, averagePrice
        type: 0=Undefined, 1=Long, 2=Short
        """
        try:
            if isinstance(data, list):
                data = data[0] if data else {}

            pos_type = data.get("type", 0)
            size = int(data.get("size", 0) or 0)
            avg_price = data.get("averagePrice", 0) or 0.0

            type_names = {0: "FLAT", 1: "LONG", 2: "SHORT"}
            type_str = type_names.get(pos_type, f"TYPE={pos_type}")

            logger.info(
                f"[USER HUB] 📊 POSITION: {type_str} {size}x @ {avg_price:.2f}"
            )

            # ── Grace window ─────────────────────────────────────────
            # Skip reconciliation within 10s of our own last local state
            # transition (entry fill OR exit). Anchors off _last_local_state_change
            # so it fires on the FIRST entry of the day too — risk.last_trade_time
            # is 0 until the first close, which previously bypassed the guard and
            # let phantom FLAT events force-close a just-opened position.
            now = time.time()
            t_state = (
                now - self._last_local_state_change
                if self._last_local_state_change > 0 else 999
            )
            t_trade = (
                now - self.risk.last_trade_time
                if self.risk.last_trade_time > 0 else 999
            )
            time_since_last_change = min(t_state, t_trade)
            if time_since_last_change < 10:
                return

            # Normalize both sides
            platform_flat = (size == 0 or pos_type == 0)
            local_flat = (
                not self.position_direction
                or self.position_direction == Direction.FLAT
                or self.position_contracts == 0
            )

            # ── Case A: GHOST — platform has position, bot is flat ──
            if local_flat and not platform_flat:
                logger.critical(
                    f"[USER HUB] 🚨 GHOST POSITION: Platform={type_str} {size}x "
                    f"@ {avg_price:.2f} but bot is FLAT "
                    f"(last state change {time_since_last_change:.0f}s ago). FLATTENING."
                )
                try:
                    self.conn.flatten_all(self.config.topstep.contract_id)
                except Exception as e:
                    logger.error(f"[USER HUB] Ghost flatten failed: {e}")
                return

            # ── Case B: Platform flat, bot holds position ───────────
            # External close (bracket fill) — sync local state so we don't
            # think we're still in a position that's already gone.
            if not local_flat and platform_flat:
                local_dir = (
                    self.position_direction.name
                    if self.position_direction else "?"
                )
                logger.warning(
                    f"[USER HUB] External close detected: platform FLAT but bot "
                    f"holds {local_dir} {self.position_contracts}x. "
                    f"Closing local state."
                )
                # 2026-04-30: query the platform for the ACTUAL close-side fill
                # price instead of using the bot's last quoted price. The last
                # quote can differ from the bracket fill by several points,
                # which made the bot's P&L diverge from broker truth on every
                # external close. Falls back to last_price if the lookup fails.
                exit_side = 1 if self.position_direction == Direction.LONG else 0
                actual_fill = None
                try:
                    actual_fill = self.conn.get_close_fill_price(
                        contract_id=self.config.topstep.contract_id,
                        exit_side=exit_side,
                    )
                except Exception as _e:
                    logger.warning(f"[USER HUB] Close-fill lookup failed: {_e}")

                if actual_fill and actual_fill > 0:
                    exit_price = actual_fill
                    logger.info(
                        f"[USER HUB] Using platform fill price {actual_fill:.2f} "
                        f"for P&L attribution (was last_price="
                        f"{self.features.last_price:.2f})"
                    )
                else:
                    exit_price = (
                        self.features.last_price
                        if self.features.last_price > 0
                        else self.position_entry_price
                    )
                    logger.info(
                        f"[USER HUB] Fill lookup returned no result — "
                        f"falling back to last_price {exit_price:.2f}"
                    )
                try:
                    self._exit_position(
                        exit_price,
                        "Platform reports FLAT — external bracket/close detected"
                    )
                except Exception as e:
                    logger.error(f"[USER HUB] External-close sync failed: {e}")
                return

            # ── Case C: Direction or size mismatch ──────────────────
            if not local_flat and not platform_flat:
                local_dir = (
                    "LONG" if self.position_direction == Direction.LONG
                    else "SHORT"
                )
                if local_dir != type_str or self.position_contracts != size:
                    logger.critical(
                        f"[USER HUB] 🚨 POSITION MISMATCH: "
                        f"Bot={local_dir} {self.position_contracts}x vs "
                        f"Platform={type_str} {size}x. FLATTENING."
                    )
                    try:
                        self.conn.flatten_all(self.config.topstep.contract_id)
                    except Exception as e:
                        logger.error(f"[USER HUB] Mismatch flatten failed: {e}")

        except Exception as e:
            logger.error(f"[USER HUB] Position event error: {e} | data={data}", exc_info=True)

    def _on_user_account_event(self, data):
        """Handle GatewayUserAccount — balance and account status updates.
        
        data fields: id, name, balance, canTrade, isVisible, simulated
        """
        try:
            if isinstance(data, list):
                data = data[0] if data else {}
            
            balance = data.get("balance", 0)
            can_trade = data.get("canTrade", True)
            
            logger.info(
                f"[USER HUB] 💰 ACCOUNT: balance=${balance:,.2f} | canTrade={can_trade}"
            )
            
            # Update balance in risk manager immediately
            if balance and balance > 0:
                self.risk.update_balance(balance)
                self.mll_tracker.update(balance)
            
            if not can_trade:
                logger.critical("[USER HUB] ⚠️ ACCOUNT CANNOT TRADE — canTrade=false!")
                
        except Exception as e:
            logger.error(f"[USER HUB] Account event error: {e} | data={data}", exc_info=True)

    async def _check_balance(self):
        """Check account balance against dynamic MLL."""
        balance = self.conn.get_account_balance()
        if balance is not None:
            self.account_balance = balance
            # Update MLL tracker — recalculates cushion, max contracts, daily loss limit
            self.risk.update_balance(balance)
            # Keep the vol sizer's balance fresh for spot %-of-account sizing
            try:
                self.vol_sizer.account_balance = balance
                if not self.vol_sizer.sod_balance:
                    self.vol_sizer.sod_balance = self.sod_balance or balance
            except Exception:
                pass
            cushion = self.mll_tracker.update(balance)
            
            if cushion <= 0:
                logger.critical(
                    f"ACCOUNT AT MLL! Balance ${balance:,.2f} <= "
                    f"MLL ${self.mll_tracker.current_mll:,.2f}"
                )
                logger.critical("EMERGENCY SHUTDOWN — Flattening all positions")
                if self.position_direction and self.position_direction != Direction.FLAT:
                    self.conn.flatten_all(
                        self.config.topstep.contract_id,
                        known_side=self.position_direction.name,
                        known_size=self.position_contracts
                    )
                else:
                    self.conn.flatten_all(self.config.topstep.contract_id)
                self.running = False
            elif cushion < 1000:
                logger.warning(
                    f"⚠️ CRITICAL CUSHION: ${cushion:,.0f} remaining above MLL — "
                    f"trading halted until cushion recovers"
                )


# ── Entry Point ─────────────────────────────────────────────────────────

async def main(poll_only: bool = False):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
                encoding="utf-8"
            ),
        ]
    )
    
    config = Config.load()
    bot = TradingBot(config)
    bot._poll_only = poll_only
    
    # Handle Ctrl+C
    def shutdown(sig, frame):
        logger.info("Ctrl+C received — shutting down")
        bot.running = False
    
    signal.signal(signal.SIGINT, shutdown)
    
    try:
        await bot.start()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        bot.conn.flatten_all(config.topstep.contract_id)


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    asyncio.run(main())
