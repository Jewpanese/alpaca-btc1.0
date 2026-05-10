"""Risk Manager - The Most Important Module.

"Risk management is more important than the trading strategy."
- Every successful trader ever

This module is SEPARATE from strategies on purpose.
Strategies say "there's a trade." Risk manager says "should we take it, and how big?"

Topstep3's fatal flaw: risk logic was intertwined with strategy logic,
creating 100+ conflicting parameters. Here, risk is clean and sovereign.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import time
import logging

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from strategies.base import Signal, Direction
from risk.topstep_mll import MLLTracker, TopstepAccountConfig
from core.regime import Regime, RegimeState

logger = logging.getLogger(__name__)

# ── Known High-Impact News Times (ET) ───────────────────────────────────
# These are recurring scheduled releases. Times in (hour, minute) ET.
# Updated periodically. The bot blocks entries within N minutes of these.
DAILY_NEWS_BLACKOUTS_ET = [
    # Pre-market economic data (most releases)
    (8, 30),   # NFP, CPI, PPI, GDP, Jobless Claims, Retail Sales, etc.
    (9, 45),   # S&P PMI Flash (varies, sometimes 9:45)
    (10, 0),   # ISM, JOLTS, Consumer Confidence, New Home Sales, UMich
    (10, 30),  # EIA Crude inventories (Wednesday)
    # FOMC
    (14, 0),   # FOMC rate decision
    (14, 30),  # FOMC press conference
]


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: float
    exit_time: float
    direction: Direction
    entry_price: float
    exit_price: float
    contracts: int
    pnl_dollars: float
    strategy: str
    exit_reason: str

    @property
    def pnl_ticks(self) -> float:
        mult = 1 if self.direction == Direction.LONG else -1
        return (self.exit_price - self.entry_price) * mult / 0.25

    @property
    def is_winner(self) -> bool:
        return self.pnl_dollars >= 0  # Breakeven = not a loss (don't count toward loss streak)

    @property
    def hold_seconds(self) -> float:
        return self.exit_time - self.entry_time


@dataclass
class RiskConfig:
    """All risk parameters in one place. ~20 params total."""

    # Daily limits (Topstep rules)
    max_daily_loss: float = 1500.0       # Stop trading if we lose this much
    daily_profit_target: float = 2000.0  # Optional: stop when target hit

    # Position sizing
    max_contracts: int = 3               # Max position size
    base_contracts: int = 1              # Default position size
    kelly_fraction: float = 0.25         # Fraction of Kelly to use (conservative)

    # Per-trade limits
    max_loss_per_trade: float = 500.0    # Hard stop per trade
    min_risk_reward: float = 1.1         # Minimum R:R to take a trade (was 1.3, too many marginal blocks)

    # Drawdown protection
    drawdown_reduce_threshold: float = 500.0   # Reduce size after this drawdown
    drawdown_stop_threshold: float = 1000.0    # Stop trading after this drawdown

    # Session open buffer - no trades for N minutes after futures open (6 PM ET / 4 PM MST)
    session_open_buffer_minutes: int = 30     # Wait 30min after session open before trading

    # Frequency limits
    max_trades_per_hour: int = 999        # No limit (was 6, caused missed trades 2026-02-27)
    max_trades_per_day: int = 999        # No limit (was 20, tranche exits inflated count)
    min_seconds_between_trades: int = 120 # Cooldown between trades (was 15, caused overtrading)

    # Loss streak protection
    max_consecutive_losses: int = 3      # Pause after N consecutive losses
    loss_streak_cooldown_sec: int = 300  # 5min cooldown after loss streak (was 60s - too short)
    
    # Win streak protection (prevent overconfidence / mean reversion)
    max_consecutive_wins: int = 5        # Pause after N consecutive wins
    win_streak_cooldown_sec: int = 180   # 3min cooldown after win streak

    # Chop detection - now relative (EMA spread / ATR ratio)
    # These legacy absolute thresholds are kept for backward compat but no longer used.
    # The risk manager now uses EMA_spread / ATR < 1.0 as the chop threshold.
    ema_chop_threshold: float = 3.0      # DEPRECATED - kept for config compat
    ema_chop_threshold_overnight: float = 5.0  # DEPRECATED - kept for config compat

    # Revenge trading prevention
    max_loss_velocity: float = 500.0     # Max $ lost per 10 minutes

    # News blackout - no new entries within N minutes of known high-impact releases
    # Times are in ET (Eastern Time) as (hour, minute) tuples
    news_blackout_minutes: int = 5       # Minutes before AND after to block entries

    # DEPRECATED: old balance-based scaling (replaced by MLLTracker cushion-based scaling)
    # scaling_thresholds are no longer used - see topstep_mll.py

    # Contract value (set by bot.py based on instrument config)
    tick_value: float = 1.25             # MES: $1.25 per tick (ES: $12.50)
    point_value: float = 5.0             # MES: $5 per point (ES: $50)


class RiskManager:
    """Sovereign risk management - strategies propose, risk manager disposes."""

    def __init__(self, config: RiskConfig = None, mll_tracker: MLLTracker = None):
        self.config = config or RiskConfig()
        self.mll = mll_tracker or MLLTracker()
        self.trades_today: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.last_trade_time: float = 0
        self.loss_streak_pause_until: float = 0
        self.win_streak_pause_until: float = 0
        self._regime_size_mult: float = 1.0
        self._session_start: float = time.time()
        self._current_balance: float = self.mll.config.account_size  # Updated via sync

    # ── Strategy-Regime Compatibility Map ──
    # Each strategy declares which regimes it can profitably trade in.
    # If a strategy fires in an incompatible regime, risk manager blocks it.
    STRATEGY_REGIMES = {
        "EMA_REJECT":     {Regime.TRENDING_FAST, Regime.TRENDING_SLOW},
        "TREND_FOLLOW":   {Regime.TRENDING_SLOW, Regime.TRENDING_FAST},
        "DELTA_DIV":      {Regime.RANGING, Regime.VOLATILE, Regime.TRENDING_SLOW},
        "BB_BOUNCE":      {Regime.RANGING},
        "VWAP_REVERT":    {Regime.RANGING, Regime.TRENDING_SLOW},
        "SR_BREAKOUT":    {Regime.TRENDING_FAST, Regime.VOLATILE},
        "MOMENTUM":       {Regime.TRENDING_FAST, Regime.VOLATILE},
    }

    def should_trade(self, signal: Signal, market_state=None,
                     regime_state: RegimeState = None) -> tuple[bool, str, int]:
        """Decide if we should take this trade and how many contracts.

        Args:
            signal: The trading signal from a strategy
            market_state: Current MarketState
            regime_state: Current RegimeState from regime detector (optional but recommended)

        Returns: (should_trade, reason, contracts)
        """
        now = time.time()

        # CLOSE CUTOFF + SESSION OPEN BUFFER — futures-only, neutralized for BTC.
        # On Alpaca crypto these gates would silently block ~3 hours/day for no
        # reason (BTC trades 24/7). They're skipped when `_crypto_24_7` is set
        # by run_btc.py; the original logic is preserved below for the futures
        # path so this manager stays portable to topstep5.
        if not getattr(self, "_crypto_24_7", False):
            # CLOSE CUTOFF — no new trades near Topstep's forced flatten (4:10 PM ET)
            if market_state and hasattr(market_state, 'minutes_since_open'):
                mins = market_state.minutes_since_open
                if 390 <= mins <= 420:
                    return False, f"Close cutoff: {mins} min since open (4:00-4:30 PM ET window), no new trades near close", 0

            # SESSION OPEN BUFFER — no trades in first N min after futures open (6 PM ET)
            now_utc = datetime.now(timezone.utc)
            now_et = now_utc.astimezone(ZoneInfo('America/New_York'))
            et_minutes = now_et.hour * 60 + now_et.minute
            session_open_et = 1080  # 6:00 PM ET
            if et_minutes >= session_open_et:
                mins_since_open = et_minutes - session_open_et
            elif et_minutes < 960:
                mins_since_open = (1440 - session_open_et) + et_minutes
            else:
                mins_since_open = 9999

            if mins_since_open < self.config.session_open_buffer_minutes:
                return False, (
                    f"Session open buffer: {mins_since_open}min since open "
                    f"(need {self.config.session_open_buffer_minutes}min). "
                    f"Indicators recalibrating."
                ), 0

        # REGIME-STRATEGY COMPATIBILITY - soft filter, reduce size instead of blocking
        # Strategies trade in any regime but get half-sized when out of their element
        self._regime_size_mult = 1.0
        if regime_state:
            allowed = self.STRATEGY_REGIMES.get(signal.strategy_name)
            if allowed and regime_state.regime not in allowed:
                self._regime_size_mult = 0.5
                logger.info(
                    f"[RISK] Regime soft-filter: {signal.strategy_name} in "
                    f"{regime_state.regime.value} - trading at 50% size"
                )

        # GLOBAL TREND FILTER - block weak counter-trend, reduce strong counter-trend
        if market_state:
            trend = self._detect_trend(market_state)
            is_counter_trend = (
                (signal.direction == Direction.LONG and trend == "bearish") or
                (signal.direction == Direction.SHORT and trend == "bullish")
            )
            if is_counter_trend:
                if signal.strength < 0.5:
                    logger.info(
                        f"[RISK] Trend HARD-BLOCK: {signal.direction.name} in {trend} trend "
                        f"with weak signal (strength={signal.strength:.2f} < 0.50) - BLOCKED"
                    )
                    return False, f"Counter-trend with weak signal ({signal.strength:.2f}) in {trend} trend", 0
                else:
                    self._regime_size_mult *= 0.5
                    logger.info(
                        f"[RISK] Trend soft-filter: {signal.direction.name} in {trend} trend "
                        f"with strong signal (strength={signal.strength:.2f}) - 50% size"
                    )

            # CHOP DETECTION - dead market is a hard block (sub-3 ATR is untradeable)
            # Other chop conditions reduce size
            is_choppy, chop_reason = self._check_chop(market_state, regime_state)
            if is_choppy:
                if regime_state and regime_state.regime == Regime.DEAD:
                    return False, chop_reason, 0
                else:
                    self._regime_size_mult *= 0.5
                    logger.info(f"[RISK] Chop soft-filter: {chop_reason} - trading at reduced size")

            # OVEREXTENSION FILTER - soft warning, reduce size instead of blocking
            # In trend days, price SHOULD be far from VWAP - blocking kills profits
            if signal.strategy_name not in ("EMA_REJECT", "TREND_FOLLOW"):
                overextended, ext_reason = self._check_overextension(market_state, signal.direction)
                if overextended:
                    self._regime_size_mult *= 0.5  # Stack with regime mult
                    logger.info(f"[RISK] Overextension soft-filter: {ext_reason} - trading at reduced size")

        # MLL CUSHION CHECK - if cushion is critical, stop everything
        cushion = self.mll.update(self._current_balance)
        if cushion < 1_000:
            return False, f"MLL CRITICAL: Cushion ${cushion:,.0f} < $1,000 - trading halted", 0

        # Daily loss limit (dynamically set by MLL tracker based on cushion)
        if self.daily_pnl <= -self.config.max_daily_loss:
            return False, f"Daily loss limit hit (${self.daily_pnl:.0f} / -${self.config.max_daily_loss:.0f})", 0

        # Daily profit target (optional soft stop)
        # Daily profit target - disabled, never stop trading on profit
        # if self.daily_pnl >= self.config.daily_profit_target:
        #     return False, f"Daily target reached (${self.daily_pnl:.0f})", 0

        # Loss streak cooldown
        if now < self.loss_streak_pause_until:
            remaining = self.loss_streak_pause_until - now
            return False, f"Loss streak cooldown ({remaining:.0f}s remaining)", 0

        # Win streak cooldown
        if now < self.win_streak_pause_until:
            remaining = self.win_streak_pause_until - now
            return False, f"Win streak cooldown ({remaining:.0f}s remaining)", 0

        # Consecutive losses - short pause, then back to full size
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.loss_streak_pause_until = now + self.config.loss_streak_cooldown_sec
            # Reset streak so cooldown only fires ONCE per streak, not infinitely
            streak_count = self.consecutive_losses
            self.consecutive_losses = 0
            return False, f"Max consecutive losses ({streak_count}), pausing {self.config.loss_streak_cooldown_sec}s", 0

        # Consecutive wins - pause to prevent overtrading on hot streak
        if self.consecutive_wins >= self.config.max_consecutive_wins:
            self.win_streak_pause_until = now + self.config.win_streak_cooldown_sec
            streak_count = self.consecutive_wins
            self.consecutive_wins = 0
            return False, f"Max consecutive wins ({streak_count}), pausing {self.config.win_streak_cooldown_sec}s", 0

        # Trade frequency
        if len(self.trades_today) >= self.config.max_trades_per_day:
            return False, "Max daily trades reached", 0

        trades_last_hour = sum(
            1 for t in self.trades_today
            if now - t.exit_time < 3600
        )
        if trades_last_hour >= self.config.max_trades_per_hour:
            return False, f"Max hourly trades reached ({trades_last_hour})", 0

        # Cooldown between trades
        if now - self.last_trade_time < self.config.min_seconds_between_trades:
            return False, "Trade cooldown active", 0

        # Risk/reward check
        if signal.risk_reward < self.config.min_risk_reward:
            return False, f"R:R too low ({signal.risk_reward:.2f} < {self.config.min_risk_reward})", 0

        # Loss velocity check (revenge trading prevention)
        recent_losses = sum(
            t.pnl_dollars for t in self.trades_today
            if now - t.exit_time < 600 and t.pnl_dollars < 0
        )
        if abs(recent_losses) > self.config.max_loss_velocity:
            return False, f"Loss velocity too high (${recent_losses:.0f} in 10min)", 0

        # Calculate position size (apply regime soft-filter)
        contracts = self._calculate_size(signal)
        if self._regime_size_mult < 1.0:
            contracts = max(1, int(contracts * self._regime_size_mult))

        # Verify max loss per trade won't be exceeded
        risk_per_contract = abs(signal.entry_price - signal.stop_loss) * self.config.point_value
        total_risk = risk_per_contract * contracts
        if total_risk > self.config.max_loss_per_trade:
            contracts = max(1, int(self.config.max_loss_per_trade / risk_per_contract))

        if contracts <= 0:
            return False, "Position size calculated as 0", 0

        logger.info(
            f"[RISK] APPROVED: {signal.strategy_name} {signal.direction.name} "
            f"x{contracts} | R:R={signal.risk_reward:.2f} | "
            f"Risk=${risk_per_contract * contracts:.0f} | "
            f"Daily P&L: ${self.daily_pnl:.0f}"
        )

        return True, "Approved", contracts

    def _calculate_size(self, signal: Signal) -> int:
        """Calculate position size using cushion-based MLL scaling.

        The MLLTracker determines max contracts based on the real cushion
        (balance - MLL). This is the ONLY way to scale safely with Topstep's
        trailing MLL mechanic.

        Additional modifiers:
        - Intraday drawdown reduces size
        - High-confidence signals can scale up (within cushion limits)
        """
        # Get cushion-based max from MLL tracker
        cushion_max = self.mll.get_max_contracts(self._current_balance)

        if cushion_max == 0:
            logger.warning("[RISK] MLL tracker says 0 contracts - cushion critically low")
            return 0

        # Start with base size (capped by cushion)
        contracts = min(self.config.base_contracts, cushion_max)

        # If in intraday drawdown, reduce size
        drawdown = self.peak_pnl - self.daily_pnl
        if drawdown > self.config.drawdown_reduce_threshold:
            contracts = max(1, contracts - 1)
            logger.info(f"[RISK] Reduced size due to intraday drawdown (${drawdown:.0f})")

        if drawdown > self.config.drawdown_stop_threshold:
            return 0

        # NOTE: Removed Kelly +1 scaling (2026-02-27). Adding 1 contract created
        # uneven tranche splits (e.g., 11 → 3/5/3 instead of 10 → 3/4/3).
        # Scaling is handled by MLL cushion tiers instead (10 → 20 → 30 etc.)

        return min(contracts, cushion_max)

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update state."""
        self.trades_today.append(trade)
        self.daily_pnl += trade.pnl_dollars
        self.peak_pnl = max(self.peak_pnl, self.daily_pnl)
        self.last_trade_time = trade.exit_time

        if trade.is_winner:
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        logger.info(
            f"[RISK] Trade #{len(self.trades_today)}: "
            f"{'WIN' if trade.is_winner else 'LOSS'} ${trade.pnl_dollars:+.0f} | "
            f"Daily: ${self.daily_pnl:+.0f} | "
            f"Streak: {self.consecutive_losses}L / {self.consecutive_wins}W"
        )

        # Persist to JSONL trade journal so analyzers/eod_summary.py can pick it
        # up tomorrow even if the bot crashed/restarted. Best-effort — must
        # never raise into the trading loop.
        try:
            from datetime import datetime as _dt, timezone as _tz
            from core import trade_journal as _tj
            _entry_iso = _dt.fromtimestamp(trade.entry_time, tz=_tz.utc).isoformat()
            _exit_iso = _dt.fromtimestamp(trade.exit_time, tz=_tz.utc).isoformat()
            # 1 contract = 0.01 BTC by convention in this port (point_value = $0.01)
            _btc_size = float(trade.contracts) * 0.01
            _pnl_pts = (
                (trade.exit_price - trade.entry_price)
                * (1 if getattr(trade.direction, "name", str(trade.direction)) == "LONG" else -1)
            )
            _tj.append_trade({
                "ts": _exit_iso,
                "entry_ts": _entry_iso,
                "direction": getattr(trade.direction, "name", str(trade.direction)),
                "contracts": int(trade.contracts),
                "btc_size": round(_btc_size, 6),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "pnl_pts": float(_pnl_pts),
                "pnl_dollars": float(trade.pnl_dollars),
                "exit_reason": str(trade.exit_reason),
                "regime": getattr(trade, "regime", "") or "",
                "session": getattr(trade, "session", "") or "",
                "strategy": str(trade.strategy),
                "duration_sec": int(max(0, trade.exit_time - trade.entry_time)),
                "mfe_pts": float(getattr(trade, "mfe_pts", 0.0) or 0.0),
                "mae_pts": float(getattr(trade, "mae_pts", 0.0) or 0.0),
            })
        except Exception as _e:
            logger.warning(f"[RISK] trade_journal append failed (non-fatal): {_e}")

    def add_tranche_pnl(self, pnl_dollars: float, label: str = "partial"):
        """Feed a tranche partial-exit's realized P&L into daily_pnl.

        Partials (1R take, BE exits, runner trails) are not completed trades —
        remaining contracts are still open. The final close will fire
        record_trade() for the last contracts. This keeps daily_pnl
        authoritative as the single P&L source (A3 fix).
        """
        self.daily_pnl += pnl_dollars
        self.peak_pnl = max(self.peak_pnl, self.daily_pnl)
        logger.info(
            f"[RISK] Tranche {label}: ${pnl_dollars:+.0f} | "
            f"Daily: ${self.daily_pnl:+.0f} | Peak: ${self.peak_pnl:+.0f}"
        )

    def _check_news_blackout(self) -> Optional[str]:
        """Block entries near scheduled high-impact news releases.

        Returns reason string if in blackout, None if clear.
        """
        now_utc = datetime.now(timezone.utc)
        # Convert to ET (UTC-5 standard, UTC-4 DST)
        # Approximate: March second Sunday to November first Sunday = EDT
        year = now_utc.year
        # DST starts second Sunday of March
        march_second_sunday = 14 - (datetime(year, 3, 1).weekday() + 1) % 7
        dst_start = datetime(year, 3, march_second_sunday, 7, 0, tzinfo=timezone.utc)
        # DST ends first Sunday of November
        nov_first_sunday = 7 - (datetime(year, 11, 1).weekday() + 1) % 7
        dst_end = datetime(year, 11, nov_first_sunday, 6, 0, tzinfo=timezone.utc)

        et_offset = timedelta(hours=-4) if dst_start <= now_utc < dst_end else timedelta(hours=-5)
        now_et = now_utc + et_offset

        blackout_mins = self.config.news_blackout_minutes

        for hour, minute in DAILY_NEWS_BLACKOUTS_ET:
            news_time = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
            diff_seconds = abs((now_et - news_time).total_seconds())
            if diff_seconds <= blackout_mins * 60:
                mins_to = (news_time - now_et).total_seconds() / 60
                if mins_to > 0:
                    return f"News blackout: {hour}:{minute:02d} ET release in {mins_to:.0f}min"
                else:
                    return f"News blackout: {hour}:{minute:02d} ET released {-mins_to:.0f}min ago, waiting for dust to settle"

        return None

    def _check_chop(self, state, regime_state: RegimeState = None) -> tuple[bool, str]:
        """Detect choppy/range-bound markets using regime-aware logic.

        Uses ATR percentile (relative) instead of absolute point floors.
        The regime detector classifies DEAD markets; this method checks
        EMA convergence for chop detection.

        Strategy-regime compatibility is checked in should_trade() - this
        method only blocks genuinely untradeable conditions.
        """
        # REGIME-BASED DEAD MARKET CHECK (replaces hardcoded 2.0pt ATR floor)
        if regime_state and regime_state.regime == Regime.DEAD:
            return True, (
                f"Dead market: ATR at {regime_state.atr_percentile:.0%} percentile "
                f"(bottom 5%). Session={getattr(state, 'session', 'unknown')}. "
                f"No strategy can profit here."
            )

        # EMA SPREAD CHOP DETECTION - normalized by ATR (relative, not absolute)
        atr = getattr(state, 'atr_14', 0)
        if atr <= 0:
            atr = 1.0  # Prevent division by zero

        # Check 5-minute EMA spread relative to ATR
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0 and state.ema_5m_50 > 0:
            emas_5m = [state.ema_5m_9, state.ema_5m_26, state.ema_5m_50]
            spread_5m = max(emas_5m) - min(emas_5m)
            spread_atr_ratio = spread_5m / atr

            # If EMA spread is less than 1x ATR, EMAs are bunched = chop
            # This is relative: works whether ATR is 1pt or 4pt
            if spread_atr_ratio < 1.0:
                return True, (
                    f"Chop detected: 5m EMA spread={spread_5m:.1f}pts "
                    f"({spread_atr_ratio:.1f}x ATR, < 1.0x threshold). "
                    f"EMAs: 9={state.ema_5m_9:.2f}, 26={state.ema_5m_26:.2f}, 50={state.ema_5m_50:.2f}"
                )

        # Fallback: check 1-minute EMA spread relative to ATR
        elif state.ema_9 > 0 and state.ema_21 > 0 and state.ema_50 > 0:
            emas_1m = [state.ema_9, state.ema_21, state.ema_50]
            spread_1m = max(emas_1m) - min(emas_1m)
            spread_atr_ratio = spread_1m / atr

            if spread_atr_ratio < 1.0:
                return True, (
                    f"Chop detected: 1m EMA spread={spread_1m:.1f}pts "
                    f"({spread_atr_ratio:.1f}x ATR, < 1.0x threshold). "
                    f"EMAs: 9={state.ema_9:.2f}, 21={state.ema_21:.2f}, 50={state.ema_50:.2f}"
                )

        return False, ""

    def _check_overextension(self, state, direction: Direction) -> tuple[bool, str]:
        """Block entries into already-exhausted moves.

        Checks:
        1. RSI: Don't short when RSI < 30 (already oversold), don't long when RSI > 70
        2. VWAP distance: Don't enter if price is already >2 ATR from VWAP in entry direction
        """
        # RSI overextension
        if direction == Direction.SHORT and state.rsi_14 < 30:
            return True, f"Overextended: RSI={state.rsi_14:.0f} already oversold, SHORT blocked"
        if direction == Direction.LONG and state.rsi_14 > 70:
            return True, f"Overextended: RSI={state.rsi_14:.0f} already overbought, LONG blocked"

        # VWAP distance - don't chase moves that are already stretched
        if state.vwap > 0 and state.atr_14 > 0:
            vwap_distance = abs(state.price - state.vwap)
            max_distance = state.atr_14 * 2.0  # 2x ATR from VWAP = overextended

            if direction == Direction.SHORT and state.price < state.vwap and vwap_distance > max_distance:
                return True, (f"Overextended: price {vwap_distance:.1f}pts below VWAP "
                            f"(>{max_distance:.1f}pts), SHORT blocked")
            if direction == Direction.LONG and state.price > state.vwap and vwap_distance > max_distance:
                return True, (f"Overextended: price {vwap_distance:.1f}pts above VWAP "
                            f"(>{max_distance:.1f}pts), LONG blocked")

        return False, ""

    def _detect_trend(self, state) -> str:
        """Detect macro trend from multi-timeframe EMA alignment.

        Returns: 'bullish', 'bearish', or 'neutral'

        Uses 5-minute EMAs as PRIMARY trend (stable, less whipsaw).
        Falls back to 1-minute EMAs if 5m not available.

        Scoring system:
        - 5m EMAs stacked: strong signal (weight 2)
        - 1m EMAs stacked: confirmation (weight 1)
        - Need score >= 2 to declare trend (prevents 1m whipsaw from controlling)
        """
        score = 0  # positive = bullish, negative = bearish

        # 5-minute EMA stack (primary - weight 2)
        if state.ema_5m_9 > 0 and state.ema_5m_26 > 0 and state.ema_5m_50 > 0:
            if state.ema_5m_9 > state.ema_5m_26 > state.ema_5m_50:
                score += 2
            elif state.ema_5m_9 < state.ema_5m_26 < state.ema_5m_50:
                score -= 2

        # 1-minute EMA stack (confirmation - weight 1)
        if state.ema_9 > 0 and state.ema_21 > 0 and state.ema_50 > 0:
            if state.ema_9 > state.ema_21 > state.ema_50:
                score += 1
            elif state.ema_9 < state.ema_21 < state.ema_50:
                score -= 1

        if score >= 2:
            return "bullish"
        elif score <= -2:
            return "bearish"
        return "neutral"

    def update_balance(self, current_balance: float):
        """Update the current account balance for MLL tracking.

        Call this on startup and after each trade with the platform balance.
        The MLL tracker will update highest_balance and recalculate cushion.
        """
        self._current_balance = current_balance
        cushion = self.mll.update(current_balance)

        # Dynamically adjust daily loss limit based on cushion
        self.config.max_daily_loss = self.mll.get_daily_loss_limit(current_balance)

        status = self.mll.get_status(current_balance)
        if status["danger_zone"]:
            logger.warning(
                f"[RISK] ⚠️ DANGER ZONE: Cushion ${cushion:,.0f} - "
                f"reduce risk or stop trading!"
            )

    def sync_platform_pnl(self, session_pnl: float, current_balance: float = None):
        """Sync daily P&L with platform reality using a session-anchored delta.

        A4 (2026-04-24): `session_pnl` must be (current_balance - session_SOD),
        NOT the platform's broker-day realizedPnL. Only the caller that computes
        a session-anchored value should invoke this — the fallback call sites
        that passed `realizedPnL` directly were removed. A $50 tolerance is kept
        so ordinary mid-trade unrealized fluctuations don't thrash daily_pnl.
        """
        if current_balance is not None:
            self.update_balance(current_balance)

        if abs(session_pnl - self.daily_pnl) > 50:  # $50 tolerance
            logger.warning(
                f"[RISK] PNL SYNC: Session (balance-SOD)=${session_pnl:+,.2f} vs "
                f"Local=${self.daily_pnl:+,.2f} — adopting session value"
            )
            self.daily_pnl = session_pnl
            self.peak_pnl = max(self.peak_pnl, self.daily_pnl)

    def reset_daily(self):
        """Reset for new trading day."""
        self.trades_today.clear()
        self.daily_pnl = 0.0
        self.peak_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.loss_streak_pause_until = 0
        self.win_streak_pause_until = 0
        logger.info("[RISK] Daily reset complete")

    def get_summary(self) -> dict:
        """Get current session summary."""
        winners = [t for t in self.trades_today if t.is_winner]
        losers = [t for t in self.trades_today if not t.is_winner]

        return {
            'total_trades': len(self.trades_today),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.trades_today) if self.trades_today else 0,
            'daily_pnl': self.daily_pnl,
            'peak_pnl': self.peak_pnl,
            'max_drawdown': self.peak_pnl - min(
                (sum(t.pnl_dollars for t in self.trades_today[:i+1])
                 for i in range(len(self.trades_today))),
                default=0
            ),
            'avg_win': sum(t.pnl_dollars for t in winners) / len(winners) if winners else 0,
            'avg_loss': sum(t.pnl_dollars for t in losers) / len(losers) if losers else 0,
            'consecutive_losses': self.consecutive_losses,
        }
