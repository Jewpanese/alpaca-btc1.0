"""Position Manager — Multi-Contract Tranche System.

Manages a position split into tranches (scalp, core, runner) with
independent trailing stops and take-profit targets per tranche.
Enables partial exits via TopstepX's partialCloseContract API.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from risk.trailing_stop import TrailingStopManager, TrailingStopConfig, StopStage

logger = logging.getLogger(__name__)

# ── Tranche Allocation Tables ───────────────────────────────────────────
# Allocation = (scalp, core, runner). Scalp exits first to lock in a small win
# and de-risk the trade; Core is the bread & butter; Runner trails to capture
# extension. Native units depend on instrument (MES contract, ES contract, or
# 0.01-BTC contract for alpaca_btc).

# ES: 1 contract = 1 ES (full size)
TRANCHE_ALLOCATION_ES = {
    1: (0, 0, 1),
    2: (1, 0, 1),
    3: (1, 1, 1),
    4: (1, 2, 1),
    5: (2, 2, 1),
    6: (2, 2, 2),
    7: (2, 3, 2),
    8: (3, 3, 2),
}

# MES: 10 MES = 1 ES exposure
TRANCHE_ALLOCATION_MES = {
    10: (3, 4, 3),
    20: (6, 8, 6),
    30: (9, 12, 9),
    40: (12, 16, 12),
    50: (15, 20, 15),
    60: (18, 24, 18),
    80: (24, 32, 24),
    100: (30, 40, 30),
}

# BTC: 1 contract = 0.01 BTC (alpaca_btc convention). Same 30/40/30 ratio.
TRANCHE_ALLOCATION_BTC = {
    5:  (1, 3, 1),     # 0.05 BTC — survival-tier
    10: (3, 4, 3),     # 0.10 BTC
    20: (6, 8, 6),     # 0.20 BTC
    30: (9, 12, 9),    # 0.30 BTC
    40: (12, 16, 12),  # 0.40 BTC
    50: (15, 20, 15),  # 0.50 BTC — full size
}


def get_allocation(total: int, instrument: str = "BTC") -> tuple:
    """Get (scalp, core, runner) allocation for a given total contract count.

    BTC and MES both round to nearest 10-block then look up; otherwise apply
    a 30/40/30 fallback. ES uses its own small-count table.
    """
    if total <= 0:
        return (0, 0, 0)

    if instrument in ("MES", "BTC"):
        table = TRANCHE_ALLOCATION_BTC if instrument == "BTC" else TRANCHE_ALLOCATION_MES
        if total in table:
            return table[total]
        # Generic 30/40/30 fallback
        scalp = max(1, round(total * 0.30))
        runner = max(1, round(total * 0.30))
        core = total - scalp - runner
        if core < 1:
            core = 1
            scalp = max(1, (total - core) // 2)
            runner = total - scalp - core
        return (scalp, core, runner)
    else:
        # ES mode
        table = TRANCHE_ALLOCATION_ES
        if total in table:
            return table[total]
        scalp, core, runner = table[8]
        remaining = total - 8
        core += remaining
        return (scalp, core, runner)


@dataclass
class TrancheExit:
    """Represents a tranche that needs to be closed."""
    role: str       # "scalp", "core", "runner"
    size: int       # Number of contracts to close
    reason: str     # Why it's exiting
    pnl_pts: float  # Unrealized P&L at exit in points


@dataclass
class ThesisContext:
    """Captures the market state at entry for thesis invalidation checks."""
    entry_ema_name: str = ""        # e.g. "3m_9"
    entry_ema_value: float = 0.0    # EMA price at entry
    entry_trend: str = ""           # "bullish" or "bearish"
    entry_rsi: float = 0.0         # RSI at entry
    strategy_name: str = ""        # e.g. "EMA_REJECT"

    @staticmethod
    def from_signal_reason(reason: str, rsi: float = 0.0, trend: str = "") -> "ThesisContext":
        """Parse thesis context from a signal's reason string.
        
        Handles patterns like:
            'EMA reject SHORT: price rejected off 3m_9=6851.61, trend=bearish, RSI=44'
        """
        ctx = ThesisContext(entry_rsi=rsi)
        
        # Extract strategy name
        if "EMA reject" in reason or "EMA_REJECT" in reason:
            ctx.strategy_name = "EMA_REJECT"
        
        # Extract EMA name and value: "3m_9=6851.61" or "1m_26=6850.00"
        ema_match = re.search(r'(\d+m_\d+)=([\d.]+)', reason)
        if ema_match:
            ctx.entry_ema_name = ema_match.group(1)
            ctx.entry_ema_value = float(ema_match.group(2))
        
        # Extract trend
        trend_match = re.search(r'trend=(\w+)', reason)
        if trend_match:
            ctx.entry_trend = trend_match.group(1)
        elif trend:
            ctx.entry_trend = trend
        
        # Extract RSI if not passed
        if ctx.entry_rsi == 0.0:
            rsi_match = re.search(r'RSI=([\d.]+)', reason)
            if rsi_match:
                ctx.entry_rsi = float(rsi_match.group(1))
        
        return ctx


@dataclass
class Tranche:
    """A single tranche within a multi-contract position."""
    role: str                            # "scalp", "core", "runner"
    size: int                            # Number of contracts
    trailing_stop: TrailingStopManager   # Independent trailing stop
    tp_price: float                      # Take-profit price (0 = no TP)
    closed: bool = False
    close_reason: str = ""
    close_pnl_pts: float = 0.0


class PositionManager:
    """Manages a multi-contract position with independent tranches.
    
    Each tranche has its own trailing stop and TP target:
    - Scalp: standard TP (1x base), standard trail
    - Core: extended TP (2x base), standard trail
    - Runner: wide TP (5x base), WIDE trail (3.5pt runner distance, 2.5pt trailing)
    
    Usage (BTC):
        pm = PositionManager(entry_price=80000.0, direction=1, total_contracts=20,
                             base_tp_distance=400.0, base_sl_distance=800.0, atr=300.0, point_value=0.01)
        
        # On each tick:
        exits = pm.update(current_price, current_atr)
        for exit in exits:
            if pm.remaining_contracts == exit.size:
                flatten_all()  # Last tranche — clean up everything
            else:
                partial_close(exit.size)
    """
    
    def __init__(
        self,
        entry_price: float,
        direction: int,           # +1 LONG, -1 SHORT
        total_contracts: int,
        base_tp_distance: float,  # Strategy's base TP distance in points
        base_sl_distance: float,  # Strategy's base SL distance in points
        atr: float,
        point_value: float = 0.01,     # BTC default ($0.01/pt/contract; was 5.0 MES)
        instrument: str = "BTC",        # "BTC" | "MES" | "ES"
        thesis_context: Optional[ThesisContext] = None,
    ):
        self.entry_price = entry_price
        self.direction = direction
        self.total_contracts = total_contracts
        self.base_tp_distance = base_tp_distance
        self.base_sl_distance = base_sl_distance
        self.point_value = point_value
        self.instrument = instrument
        self.entry_time = time.time()
        self.thesis = thesis_context or ThesisContext()
        self._thesis_invalidated = False  # Only fire once
        
        # Build tranches
        self.tranches: List[Tranche] = []
        scalp_size, core_size, runner_size = get_allocation(total_contracts, instrument)
        
        max_sl = min(base_sl_distance, 4.0)
        
        # Scalp trailing stop config — TIGHT trail to lock profit fast
        scalp_config = TrailingStopConfig(
            point_value=point_value,
            max_stop_pts=max_sl,
            breakeven_trigger_pts=1.0,     # Breakeven after just +1pt
            trailing_trigger_pts=1.5,      # Start trailing at +1.5pt
            runner_trigger_pts=2.5,        # Runner mode at +2.5pt
            trailing_distance_pts=1.0,     # Tight 1pt trail
            runner_distance_pts=1.5,       # Tight runner trail
            trailing_atr_mult=0.5,         # 0.5x ATR trail
            runner_atr_mult=0.75,          # 0.75x ATR runner
            fee_buffer_pts=0.25,
        )
        
        # Core trailing stop config — standard trail
        core_config = TrailingStopConfig(
            point_value=point_value,
            max_stop_pts=max_sl,
            breakeven_trigger_pts=2.0,
            trailing_trigger_pts=3.0,
            runner_trigger_pts=4.0,
            trailing_distance_pts=1.5,
            runner_distance_pts=2.0,
            trailing_atr_mult=1.0,
            runner_atr_mult=1.25,
        )
        
        # Runner trailing stop config — WIDE trail to let winners run
        runner_config = TrailingStopConfig(
            point_value=point_value,
            max_stop_pts=max_sl,
            breakeven_trigger_pts=2.0,
            trailing_trigger_pts=3.0,
            runner_trigger_pts=5.0,
            trailing_distance_pts=2.5,
            runner_distance_pts=3.5,
            trailing_atr_mult=1.5,
            runner_atr_mult=2.0,
        )
        
        # TP multipliers: scalp is quick take, core is the meat, runner rides
        # Scalp: 0.5x base (half the strategy target — lock profit fast)
        # Core: 1.0x base (full strategy target)
        # Runner: 3.0x base (let it ride, trail stop does the work)
        SCALP_TP_MULT = 0.5
        CORE_TP_MULT = 1.0
        RUNNER_TP_MULT = 3.0
        
        # Create tranches in exit-priority order: scalp first, then core, then runner
        if scalp_size > 0:
            scalp_tp_dist = base_tp_distance * SCALP_TP_MULT
            tp_price = entry_price + (scalp_tp_dist * direction)
            self.tranches.append(Tranche(
                role="scalp",
                size=scalp_size,
                trailing_stop=TrailingStopManager(
                    config=scalp_config,
                    entry_price=entry_price,
                    direction=direction,
                    atr=atr,
                ),
                tp_price=tp_price,
            ))
            logger.info(
                f"[POSITION] Scalp tranche: {scalp_size} contracts, "
                f"TP={tp_price:.2f} ({SCALP_TP_MULT}x base={scalp_tp_dist:.2f}pts), "
                f"TIGHT trail (BE@1pt, trail@1pt)"
            )
        
        if core_size > 0:
            core_tp_dist = base_tp_distance * CORE_TP_MULT
            tp_price = entry_price + (core_tp_dist * direction)
            self.tranches.append(Tranche(
                role="core",
                size=core_size,
                trailing_stop=TrailingStopManager(
                    config=core_config,
                    entry_price=entry_price,
                    direction=direction,
                    atr=atr,
                ),
                tp_price=tp_price,
            ))
            logger.info(
                f"[POSITION] Core tranche: {core_size} contracts, "
                f"TP={tp_price:.2f} ({CORE_TP_MULT}x base={core_tp_dist:.2f}pts), "
                f"STD trail (BE@2pt, trail@1.5pt)"
            )
        
        if runner_size > 0:
            runner_tp_dist = base_tp_distance * RUNNER_TP_MULT
            tp_price = entry_price + (runner_tp_dist * direction)
            self.tranches.append(Tranche(
                role="runner",
                size=runner_size,
                trailing_stop=TrailingStopManager(
                    config=runner_config,
                    entry_price=entry_price,
                    direction=direction,
                    atr=atr,
                ),
                tp_price=tp_price,
            ))
            logger.info(
                f"[POSITION] Runner tranche: {runner_size} contracts, "
                f"TP={tp_price:.2f} ({RUNNER_TP_MULT}x base={runner_tp_dist:.2f}pts), "
                f"WIDE trail (BE@2pt, trail@2.5pt, runner@3.5pt)"
            )
        
        logger.info(
            f"[POSITION] Created: {total_contracts} contracts "
            f"(scalp={scalp_size}, core={core_size}, runner={runner_size}) "
            f"@ {entry_price:.2f} {'LONG' if direction == 1 else 'SHORT'}"
        )
    
    def check_thesis(self, price: float, ema_levels: Dict[str, float] = None,
                     rsi: float = 0.0, trend: str = "") -> List[TrancheExit]:
        """Check if the entry thesis has been invalidated by market state.
        
        Returns TrancheExit list for all remaining tranches if thesis is dead.
        Called from bot._check_exit() with live MarketState data.
        
        Invalidation conditions (any one triggers exit):
        1. EMA Reclaim: Price closes back on the wrong side of the entry EMA
           - Short rejected off EMA → price reclaims above EMA = thesis dead
           - Long bounced off EMA → price drops below EMA = thesis dead
        2. Trend Flip: The trend that justified the entry has reversed
        3. RSI Momentum Shift: RSI moves 15+ points against the entry direction
        """
        if self._thesis_invalidated:
            return []  # Already handled
        
        if not self.thesis.strategy_name:
            return []  # No thesis context captured — can't check
        
        invalidation_reason = None
        
        # ── 1. EMA Reclaim Check ─────────────────────────────────
        # This is the most powerful signal: the level we traded off is gone
        if self.thesis.entry_ema_name and ema_levels:
            current_ema = ema_levels.get(self.thesis.entry_ema_name)
            if current_ema and current_ema > 0:
                if self.direction == -1:  # SHORT
                    # We shorted on an EMA rejection. If price is now ABOVE the EMA
                    # by more than a small buffer, the rejection failed.
                    buffer = 0.50  # 0.50pt buffer to avoid noise
                    if price > current_ema + buffer:
                        invalidation_reason = (
                            f"EMA reclaim: price {price:.2f} > {self.thesis.entry_ema_name}"
                            f"={current_ema:.2f} (+{buffer}pt buffer). "
                            f"Entry EMA was {self.thesis.entry_ema_value:.2f}"
                        )
                elif self.direction == 1:  # LONG
                    # We went long on an EMA bounce. If price drops BELOW the EMA,
                    # the bounce failed.
                    buffer = 0.50
                    if price < current_ema - buffer:
                        invalidation_reason = (
                            f"EMA reclaim: price {price:.2f} < {self.thesis.entry_ema_name}"
                            f"={current_ema:.2f} (-{buffer}pt buffer). "
                            f"Entry EMA was {self.thesis.entry_ema_value:.2f}"
                        )
        
        # ── 2. Trend Flip Check ──────────────────────────────────
        if not invalidation_reason and self.thesis.entry_trend and trend:
            if self.direction == -1 and self.thesis.entry_trend == "bearish" and trend == "bullish":
                invalidation_reason = (
                    f"Trend flipped: entry was bearish, now bullish"
                )
            elif self.direction == 1 and self.thesis.entry_trend == "bullish" and trend == "bearish":
                invalidation_reason = (
                    f"Trend flipped: entry was bullish, now bearish"
                )
        
        # ── 3. RSI Momentum Shift ────────────────────────────────
        if not invalidation_reason and self.thesis.entry_rsi > 0 and rsi > 0:
            rsi_shift = rsi - self.thesis.entry_rsi
            rsi_threshold = 15.0
            if self.direction == -1 and rsi_shift > rsi_threshold:
                # Shorted with low RSI, now RSI surging = momentum against us
                invalidation_reason = (
                    f"RSI momentum shift: entry RSI={self.thesis.entry_rsi:.0f}, "
                    f"now RSI={rsi:.0f} (+{rsi_shift:.0f}pts)"
                )
            elif self.direction == 1 and rsi_shift < -rsi_threshold:
                # Longed with decent RSI, now RSI crashing = momentum against us
                invalidation_reason = (
                    f"RSI momentum shift: entry RSI={self.thesis.entry_rsi:.0f}, "
                    f"now RSI={rsi:.0f} ({rsi_shift:.0f}pts)"
                )
        
        if not invalidation_reason:
            return []
        
        # Thesis is dead — exit all remaining tranches
        self._thesis_invalidated = True
        unrealized_pts = (price - self.entry_price) * self.direction
        exits = []
        
        for tranche in self.tranches:
            if tranche.closed:
                continue
            exits.append(TrancheExit(
                role=tranche.role,
                size=tranche.size,
                reason=f"Thesis invalidated: {invalidation_reason}",
                pnl_pts=unrealized_pts,
            ))
            tranche.closed = True
            tranche.close_reason = "thesis_invalidated"
            tranche.close_pnl_pts = unrealized_pts
            logger.info(
                f"[THESIS] {tranche.role.upper()} EXIT: {tranche.size} contracts "
                f"@ {price:.2f} | {unrealized_pts:+.2f}pts | {invalidation_reason}"
            )
        
        return exits

    def update(self, current_price: float, current_atr: float = None) -> List[TrancheExit]:
        """Update all open tranches and return list of tranches that need closing.
        
        Checks each tranche's trailing stop and TP target.
        Returns TrancheExit objects for tranches that should be closed.
        """
        exits: List[TrancheExit] = []
        unrealized_pts = (current_price - self.entry_price) * self.direction
        
        for tranche in self.tranches:
            if tranche.closed:
                continue
            
            # Check take-profit first
            tp_hit = False
            if tranche.tp_price > 0:
                if self.direction == 1 and current_price >= tranche.tp_price:
                    tp_hit = True
                elif self.direction == -1 and current_price <= tranche.tp_price:
                    tp_hit = True
            
            if tp_hit:
                tp_pts = (tranche.tp_price - self.entry_price) * self.direction
                exits.append(TrancheExit(
                    role=tranche.role,
                    size=tranche.size,
                    reason=f"TP hit @ {current_price:.2f} (target={tranche.tp_price:.2f})",
                    pnl_pts=tp_pts,
                ))
                tranche.closed = True
                tranche.close_reason = "TP"
                tranche.close_pnl_pts = tp_pts
                logger.info(
                    f"[TRANCHE] {tranche.role.upper()} TP HIT: {tranche.size} contracts "
                    f"@ {current_price:.2f} | +{tp_pts:.2f}pts"
                )
                continue
            
            # Check trailing stop
            exit_reason = tranche.trailing_stop.update(current_price, current_atr)
            if exit_reason:
                exits.append(TrancheExit(
                    role=tranche.role,
                    size=tranche.size,
                    reason=f"Trail stop: {exit_reason}",
                    pnl_pts=unrealized_pts,
                ))
                tranche.closed = True
                tranche.close_reason = "trailing_stop"
                tranche.close_pnl_pts = unrealized_pts
                logger.info(
                    f"[TRANCHE] {tranche.role.upper()} TRAIL STOP: {tranche.size} contracts "
                    f"@ {current_price:.2f} | {unrealized_pts:+.2f}pts | {exit_reason}"
                )
        
        return exits
    
    def close_tranche(self, tranche: Tranche) -> int:
        """Mark a tranche as closed and return its size."""
        tranche.closed = True
        return tranche.size
    
    def is_fully_closed(self) -> bool:
        """Check if all tranches are closed."""
        return all(t.closed for t in self.tranches)
    
    @property
    def remaining_contracts(self) -> int:
        """Total contracts still open."""
        return sum(t.size for t in self.tranches if not t.closed)
    
    @property
    def open_tranches(self) -> List[Tranche]:
        """Get list of open tranches."""
        return [t for t in self.tranches if not t.closed]
    
    def update_entry_price(self, fill_price: float, atr: float):
        """Update entry price for all tranches (when actual fill differs from signal).
        
        Recalculates TP prices and trailing stop entry prices.
        """
        if abs(fill_price - self.entry_price) < 0.01:
            return
        
        old_entry = self.entry_price
        self.entry_price = fill_price
        
        for tranche in self.tranches:
            # Recalculate TP based on role multiplier
            if tranche.role == "scalp":
                mult = 0.5
            elif tranche.role == "core":
                mult = 1.0
            else:  # runner
                mult = 3.0
            tranche.tp_price = fill_price + (self.base_tp_distance * mult * self.direction)
            
            # Update trailing stop entry
            tsm = tranche.trailing_stop
            tsm.entry_price = fill_price
            initial_distance = tsm._clamp_distance(atr * tsm.config.initial_stop_atr_mult)
            if self.direction == 1:
                tsm.current_stop_price = fill_price - initial_distance
            else:
                tsm.current_stop_price = fill_price + initial_distance
        
        logger.info(
            f"[POSITION] Entry updated: {old_entry:.2f} → {fill_price:.2f}, "
            f"TPs and stops recalculated"
        )
    
    def get_status(self) -> dict:
        """Get current position status for logging/dashboard."""
        tranche_statuses = []
        for t in self.tranches:
            ts = t.trailing_stop.get_status()
            tranche_statuses.append({
                "role": t.role,
                "size": t.size,
                "closed": t.closed,
                "tp_price": t.tp_price,
                "trail_stage": ts["stage"],
                "trail_stop": ts["stop_price"],
                "watermark_pts": ts["high_watermark_pts"],
            })
        
        return {
            "entry_price": self.entry_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "total_contracts": self.total_contracts,
            "remaining_contracts": self.remaining_contracts,
            "tranches": tranche_statuses,
            "hold_seconds": time.time() - self.entry_time,
        }
