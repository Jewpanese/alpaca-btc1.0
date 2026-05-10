"""Volatility-Targeted Position Sizing — Think in Risk Dollars, Not Contracts.

How hedge funds size positions:
  1. Define risk budget per trade (e.g., $100)
  2. Measure current volatility (ATR)
  3. ATR * stop_multiplier = stop distance in points
  4. stop_distance * point_value = dollar risk per contract
  5. contracts = risk_budget / dollar_risk_per_contract

Result: In high vol, you trade fewer contracts with wider stops.
In low vol, you trade more contracts with tighter stops.
The DOLLAR RISK stays constant. That's the whole game.

Additional layers:
  - Daily risk budget (caps total daily exposure)
  - Drawdown curve (smooth deleveraging as losses mount)
  - Cushion-aware caps (never blow the funded account)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VolSizingConfig:
    """Volatility-targeted sizing configuration."""
    
    # Risk per trade
    risk_per_trade_dollars: float = 150.0     # Base risk budget — needs 3 cts minimum for runner strategy
    
    # Daily risk budget
    daily_risk_budget: float = 999999.0       # No cap — let the bot trade freely
    
    # Stop distance (in ATR multiples) — base, modified by session
    stop_atr_mult: float = 1.5               # Stop loss = entry ± (ATR * this) — base multiplier
    target_atr_mult: float = 2.5             # Take profit = entry ± (ATR * this)
    # BTC-scale floor/ceiling. Original (1.5/20.0) was MES points — useless on
    # BTC ($50-1000 typical 3-min ATR). BigMoneyBot overrides these explicitly;
    # the defaults below protect any code path that uses VolSizingConfig directly.
    min_stop_pts: float = 50.0               # Floor: never tighter than $50
    max_stop_pts: float = 2000.0             # Ceiling: never wider than $2000
    
    # Session-aware stop multiplier (applied on top of stop_atr_mult)
    # NY open is volatile — wider stops survive shakeouts
    session_stop_mult_overnight: float = 1.0  # Overnight/Asian: base ATR mult
    session_stop_mult_ny_open: float = 1.7    # NY open (8:30-10:30 ET): wider stops
    session_stop_mult_ny_day: float = 1.3     # NY day (10:30-15:00 ET): moderate
    session_stop_mult_ny_close: float = 1.0   # NY close (15:00-16:00 ET): tighten up
    
    # Structure-based stop floor
    swing_lookback_bars: int = 20             # Look back N bars for swing high/low
    swing_buffer_pts: float = 1.0             # Add buffer beyond swing level
    
    # Contract limits
    min_contracts: int = 1                    # Always trade at least 1
    max_contracts: int = 10                   # Hard cap regardless of math
    
    # Instrument
    point_value: float = 5.0                  # MES = $5/pt
    tick_size: float = 0.25                   # MES tick size

    # Drawdown curve — smooth deleveraging (futures path)
    drawdown_full_reduce: float = 1000.0       # At -$1000 daily P&L, trade minimum

    # Cushion safety (futures path)
    cushion_danger_zone: float = 2000.0       # Below this cushion, halve everything
    cushion_critical: float = 1000.0          # Below this, stop trading

    # ── Spot %-of-account sizing (BTC). When use_pct_sizing=True, the sizer
    # ignores risk_per_trade_dollars and uses balance × pct instead. The
    # cushion check still runs as a safety net but doesn't drive sizing.
    use_pct_sizing: bool = False
    risk_per_trade_pct: float = 0.02          # 2% of account at risk per trade
    risk_per_trade_pct_max: float = 0.05      # Hard ceiling
    notional_cap_pct: float = 1.0             # Spot: max position notional / balance
    contract_size_btc: float = 0.01           # 1 native contract = 0.01 BTC
    drawdown_size_tiers: list = field(default_factory=lambda: [
        (0.00, 1.00),
        (0.02, 0.75),
        (0.05, 0.50),
        (0.10, 0.25),
        (0.15, 0.00),
    ])


class VolatilitySizer:
    """Calculate position size based on volatility targeting.
    
    The core principle: normalize dollar risk across all market conditions.
    Whether ATR is 2 points or 12 points, you risk the same dollar amount.
    """
    
    def __init__(self, config: VolSizingConfig = None):
        self.config = config or VolSizingConfig()
        self.daily_risk_used: float = 0.0
        self.daily_pnl: float = 0.0
        self._last_reset_date: str = ""
        # Pushed by HybridBot on each balance update; required when use_pct_sizing=True
        self.account_balance: float = 0.0
        self.sod_balance: float = 0.0
        self.last_price: float = 0.0
    
    def reset_daily(self):
        """Reset daily counters. Called at session start or automatically at midnight."""
        self.daily_risk_used = 0.0
        self.daily_pnl = 0.0
        self._last_reset_date = str(date.today())
    
    def calculate_stop_distance(self, atr: float, session: str = None,
                                direction: str = None, price: float = None,
                                bars: list = None) -> float:
        """Calculate stop distance in points based on ATR + session + structure.
        
        Smart stops:
        1. Base: ATR * stop_atr_mult
        2. Session-aware: multiply by session factor (NY open = wider)
        3. Structure floor: never tighter than recent swing high/low + buffer
        
        Returns clamped stop distance: min_stop_pts <= stop <= max_stop_pts
        """
        # Session-aware multiplier
        session_mult = self.config.session_stop_mult_overnight  # default
        if session:
            s = session.upper()
            if s in ('NY_OPEN',):
                session_mult = self.config.session_stop_mult_ny_open
            elif s in ('US_MIDDAY', 'LONDON'):
                session_mult = self.config.session_stop_mult_ny_day
            elif s in ('US_CLOSE',):
                session_mult = self.config.session_stop_mult_ny_close
            # OVERNIGHT stays at default (1.0)
        
        raw_stop = atr * self.config.stop_atr_mult * session_mult
        
        # Structure-based floor: stop beyond recent swing high/low
        if bars and direction and price and len(bars) >= 5:
            lookback = min(self.config.swing_lookback_bars, len(bars))
            recent = bars[-lookback:]
            if direction == 'LONG':
                # Stop below recent swing low
                swing_low = min(b.get('l', b.get('low', price)) for b in recent)
                structure_stop = (price - swing_low) + self.config.swing_buffer_pts
            else:
                # Stop above recent swing high
                swing_high = max(b.get('h', b.get('high', price)) for b in recent)
                structure_stop = (swing_high - price) + self.config.swing_buffer_pts
            
            if structure_stop > raw_stop:
                logger.info(f"[VOL_SIZING] Structure stop {structure_stop:.2f}pts > ATR stop {raw_stop:.2f}pts — using structure")
                raw_stop = structure_stop
        
        stop = max(self.config.min_stop_pts, min(self.config.max_stop_pts, raw_stop))
        
        # Round to tick size
        stop = round(stop / self.config.tick_size) * self.config.tick_size
        return stop
    
    def calculate_target_distance(self, atr: float) -> float:
        """Calculate take-profit distance in points based on ATR."""
        raw_target = atr * self.config.target_atr_mult
        target = max(self.config.min_stop_pts * 1.5, raw_target)  # Target always > stop
        return round(target / self.config.tick_size) * self.config.tick_size
    
    def calculate_contracts(
        self,
        atr: float,
        cushion: float,
        signal_strength: float = 1.0,
        regime_mult: float = 1.0,
        session: str = None,
        direction: str = None,
        price: float = None,
        bars: list = None,
    ) -> tuple[int, float, float, str]:
        """Calculate position size, stop distance, and target distance.
        
        Args:
            atr: Current ATR (14-period, in points)
            cushion: Current MLL cushion in dollars
            signal_strength: 0.0-1.0 composite signal confidence
            regime_mult: Regime-based multiplier (e.g., 0.5 in volatile, 1.2 in trending)
        
        Returns:
            (contracts, stop_pts, target_pts, reason)
        """
        # Auto-reset at midnight — handles bots that run 24 hours
        today = str(date.today())
        if self._last_reset_date and self._last_reset_date != today:
            logger.info(
                f"[VOL_SIZING] New trading day ({today}). Resetting daily budget "
                f"(was ${self.daily_risk_used:.0f} used, P&L ${self.daily_pnl:+.0f})"
            )
            self.reset_daily()

        # Cushion checks
        if cushion <= self.config.cushion_critical:
            return 0, 0, 0, f"CRITICAL: Cushion ${cushion:.0f} below ${self.config.cushion_critical:.0f}"

        # Stop and target distances (session + structure aware)
        stop_pts = self.calculate_stop_distance(atr, session=session, direction=direction,
                                                 price=price, bars=bars)
        target_pts = self.calculate_target_distance(atr)

        # Dollar risk per contract
        dollar_risk_per_contract = stop_pts * self.config.point_value

        if dollar_risk_per_contract <= 0:
            return 0, stop_pts, target_pts, "Invalid: zero dollar risk per contract"

        # ───────────────────────────────────────────────────────────────
        # Spot %-of-account sizing path (BTC). Bypasses the futures-prop
        # fixed-$-risk + cushion-tier model. Position sizes auto-scale
        # with the account.
        # ───────────────────────────────────────────────────────────────
        if self.config.use_pct_sizing:
            return self._calc_contracts_pct(
                atr=atr,
                stop_pts=stop_pts,
                target_pts=target_pts,
                dollar_risk_per_contract=dollar_risk_per_contract,
                signal_strength=signal_strength,
                regime_mult=regime_mult,
                price=price,
            )

        # ── Futures path (fixed-$ risk budget) ─────────────────────────
        # Base contracts from risk budget
        risk_budget = self.config.risk_per_trade_dollars
        
        # Apply signal strength scaling (stronger signal → more size, but never > 1.0x)
        risk_budget *= min(1.0, max(0.3, signal_strength))
        
        # Apply regime multiplier
        risk_budget *= regime_mult
        
        # Drawdown curve — smooth reduction
        if self.daily_pnl < 0:
            drawdown_pct = min(1.0, abs(self.daily_pnl) / self.config.drawdown_full_reduce)
            # Linear: at max drawdown, risk budget drops to 30% of base
            drawdown_mult = 1.0 - (0.7 * drawdown_pct)
            risk_budget *= drawdown_mult
        
        # Cushion danger zone — halve risk
        if cushion < self.config.cushion_danger_zone:
            cushion_mult = cushion / self.config.cushion_danger_zone  # 0.0 to 1.0
            risk_budget *= max(0.3, cushion_mult)
        
        # Daily risk budget check
        remaining_budget = self.config.daily_risk_budget - self.daily_risk_used
        if remaining_budget <= 0:
            return 0, stop_pts, target_pts, f"Daily risk budget exhausted (${self.daily_risk_used:.0f} used)"
        risk_budget = min(risk_budget, remaining_budget)
        
        # Calculate contracts
        raw_contracts = risk_budget / dollar_risk_per_contract
        contracts = max(self.config.min_contracts, min(self.config.max_contracts, int(raw_contracts)))
        
        # Track risk used
        actual_risk = contracts * dollar_risk_per_contract
        self.daily_risk_used += actual_risk
        
        reason = (
            f"ATR={atr:.2f} → stop={stop_pts:.2f}pt (${dollar_risk_per_contract:.0f}/ct) "
            f"→ {contracts} contracts (risk=${actual_risk:.0f})"
        )
        
        logger.info(f"[VOL_SIZING] {reason}")
        return contracts, stop_pts, target_pts, reason
    
    # ──────────────────────────────────────────────────────────────────
    # Spot %-of-account sizing (BTC)
    # ──────────────────────────────────────────────────────────────────
    def _calc_contracts_pct(
        self,
        atr: float,
        stop_pts: float,
        target_pts: float,
        dollar_risk_per_contract: float,
        signal_strength: float,
        regime_mult: float,
        price: float,
    ) -> tuple[int, float, float, str]:
        """
        Quant-standard fixed-fractional sizing for spot accounts.

        risk_$  = balance × risk_per_trade_pct × dd_mult × signal_mult × regime_mult
        contracts = risk_$ / (stop_pts × point_value)
        capped by:
          - notional cap: contracts × contract_size_btc × price ≤ balance × notional_cap_pct
          - hard min/max contracts
        """
        balance = self.account_balance
        if balance <= 0:
            return 0, stop_pts, target_pts, (
                "PCT_SIZE: account_balance not set on vol_sizer "
                "(HybridBot must push balance via vol_sizer.account_balance)"
            )

        # Drawdown deleverage based on session P&L
        sod = self.sod_balance if self.sod_balance > 0 else balance
        # Fraction of starting balance lost so far today (≥0; clamped)
        dd_pct = max(0.0, -self.daily_pnl / sod)

        # Walk the tiers in descending order: largest matching DD wins
        dd_mult = 1.0
        tiers = sorted(self.config.drawdown_size_tiers, key=lambda t: t[0], reverse=True)
        for tier_pct, tier_mult in tiers:
            if dd_pct >= tier_pct:
                dd_mult = tier_mult
                break

        if dd_mult <= 0:
            return 0, stop_pts, target_pts, (
                f"PCT_SIZE STOPPED: daily DD {dd_pct:.1%} ≥ max-DD tier — no trading"
            )

        # Clamp risk pct to configured ceiling
        risk_pct = min(self.config.risk_per_trade_pct, self.config.risk_per_trade_pct_max)
        risk_pct = max(0.001, risk_pct)  # never below 0.1%

        signal_mult = min(1.0, max(0.3, signal_strength))
        risk_dollars = balance * risk_pct * dd_mult * signal_mult * regime_mult

        # Daily-budget guard — if daily_risk_budget is set finite, respect it
        if self.config.daily_risk_budget < 999_000:
            remaining = self.config.daily_risk_budget - self.daily_risk_used
            if remaining <= 0:
                return 0, stop_pts, target_pts, (
                    f"PCT_SIZE: daily risk budget exhausted "
                    f"(${self.daily_risk_used:.0f} used)"
                )
            risk_dollars = min(risk_dollars, remaining)

        # Contracts from risk budget
        raw_contracts = risk_dollars / dollar_risk_per_contract

        # Spot notional cap: position size × price ≤ balance × notional_cap_pct
        ref_price = price if price and price > 0 else self.last_price
        if ref_price > 0 and self.config.contract_size_btc > 0:
            max_notional = balance * self.config.notional_cap_pct
            max_ct_notional = max_notional / (ref_price * self.config.contract_size_btc)
            if raw_contracts > max_ct_notional:
                raw_contracts = max_ct_notional

        contracts = max(
            self.config.min_contracts,
            min(self.config.max_contracts, int(raw_contracts)),
        )
        actual_risk = contracts * dollar_risk_per_contract
        self.daily_risk_used += actual_risk

        if ref_price > 0:
            notional_used = contracts * self.config.contract_size_btc * ref_price
            notional_pct = notional_used / balance if balance > 0 else 0
            notional_str = f" notional=${notional_used:,.0f} ({notional_pct:.1%} of bal)"
        else:
            notional_str = ""

        reason = (
            f"PCT_SIZE: bal=${balance:,.0f} risk={risk_pct*100:.1f}% × dd_mult={dd_mult:.2f} "
            f"× sig={signal_mult:.2f} × regime={regime_mult:.2f} "
            f"= ${risk_dollars:.0f} target | ATR={atr:.1f} stop=${stop_pts:.0f} "
            f"=> {contracts} ct (${actual_risk:.0f} actual risk{notional_str})"
        )
        logger.info(f"[VOL_SIZING] {reason}")
        return contracts, stop_pts, target_pts, reason

    def record_trade_pnl(self, pnl_dollars: float):
        """Record a completed trade's P&L for drawdown tracking."""
        self.daily_pnl += pnl_dollars
    
    def unreserve_risk(self, contracts: int, stop_pts: float):
        """Unreserve risk budget when a trade is cancelled or stopped out.
        
        Call this so the daily budget doesn't get permanently consumed by
        trades that exited early (e.g., breakeven exits).
        """
        risk_to_return = contracts * stop_pts * self.config.point_value
        self.daily_risk_used = max(0, self.daily_risk_used - risk_to_return)
