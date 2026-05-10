"""
ADX Range Mapper — Captures S/R levels during low-ADX (ranging) periods,
then provides breakout signals when ADX recovers.

Concept:
  1. When ADX < dead_threshold (e.g., 20), we're in "mapping mode"
     - Track range high/low from price action
     - Identify support/resistance via swing points
     - Build a "range box" with defined boundaries
  
  2. When ADX rises above alive_threshold (e.g., 25), we're in "breakout mode"
     - If price breaks above range_high → LONG breakout signal
     - If price breaks below range_low → SHORT breakout signal
     - Signal strength scales with how long the range lasted (longer = stronger breakout)
  
  3. After breakout, the range levels become S/R for trailing stops
     - Failed breakout (price re-enters range) → cancel signal

Usage:
    mapper = ADXRangeMapper()
    
    # On each bar:
    signal = mapper.update(price, high, low, adx_3m, atr)
    
    if signal:
        print(f"BREAKOUT {signal.direction} from {signal.range_low}-{signal.range_high}")
        print(f"Duration: {signal.range_duration_bars} bars, Strength: {signal.strength}")

Backtesting:
    mapper = ADXRangeMapper()
    signals = []
    for bar in bars:
        sig = mapper.update(bar['c'], bar['h'], bar['l'], adx_3m, atr)
        if sig:
            signals.append(sig)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class RangePhase(Enum):
    """Current phase of the range mapper."""
    IDLE = "idle"                    # ADX is alive, no range being mapped
    MAPPING = "mapping"              # ADX is dead, building the range box
    BREAKOUT_PENDING = "pending"     # ADX recovering, watching for breakout
    BREAKOUT_ACTIVE = "active"       # Breakout confirmed, signal fired


@dataclass
class RangeBox:
    """A captured range during dead ADX."""
    high: float                      # Highest price during range
    low: float                       # Lowest price during range
    start_time: float                # When range mapping started
    start_bar: int = 0               # Bar count when started
    bar_count: int = 0               # How many bars the range lasted
    touch_high: int = 0              # Times price tested the high
    touch_low: int = 0               # Times price tested the low
    midpoint: float = 0.0            # (high + low) / 2
    width: float = 0.0               # high - low
    
    # Internal swing tracking
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    
    def update(self, high: float, low: float, close: float, proximity_pts: float = 2.0):
        """Update range box with new bar data."""
        self.bar_count += 1
        
        # Expand range if needed
        if high > self.high:
            self.high = high
        if low < self.low:
            self.low = low
        
        self.width = self.high - self.low
        self.midpoint = (self.high + self.low) / 2
        
        # Count touches near boundaries
        if abs(high - self.high) <= proximity_pts:
            self.touch_high += 1
        if abs(low - self.low) <= proximity_pts:
            self.touch_low += 1
    
    @property
    def duration_minutes(self) -> float:
        return (time.time() - self.start_time) / 60.0


@dataclass
class BreakoutSignal:
    """Signal generated when price breaks out of a mapped range."""
    direction: str                   # "LONG" or "SHORT"
    breakout_price: float            # Price at breakout
    range_high: float                # Upper range boundary
    range_low: float                 # Lower range boundary
    range_width: float               # Width of the range
    range_duration_bars: int         # How long range lasted
    range_duration_minutes: float    # Duration in minutes
    strength: float                  # 0.0-1.0, signal quality
    timestamp: float                 # When breakout detected
    
    # Levels for trade management
    stop_level: float = 0.0          # Suggested stop (opposite side of range or midpoint)
    target_level: float = 0.0        # Suggested target (range width projection)
    
    @property
    def risk_pts(self) -> float:
        return abs(self.breakout_price - self.stop_level)
    
    @property
    def reward_pts(self) -> float:
        return abs(self.target_level - self.breakout_price)


@dataclass
class ADXRangeConfig:
    """Configuration for ADX Range Mapper."""
    # ADX thresholds
    adx_dead_threshold: float = 20.0      # Below this = "dead" → start mapping
    adx_alive_threshold: float = 25.0     # Above this = "alive" → watch for breakout
    
    # Range validation. BTC scale (1 pt = $1 of price); was 3.0/30.0/1.0/2.0 for MES.
    # BigMoneyBot overrides these; defaults below protect direct use.
    min_range_bars: int = 10                  # Minimum bars to form a valid range
    min_range_width_pts: float = 50.0         # Range must be at least $50 wide (BTC)
    max_range_width_pts: float = 2000.0       # >$2000 = not a useful range (BTC)
    min_touches: int = 2

    # Breakout confirmation
    breakout_buffer_pts: float = 30.0         # Need $30 above box edge to confirm
    breakout_confirmation_bars: int = 2
    failed_breakout_reentry_pts: float = 100.0  # Re-entry by $100 = breakout failed
    
    # Strength scoring
    duration_weight: float = 0.3          # Weight for range duration in strength calc
    width_weight: float = 0.2            # Weight for range width (wider = stronger breakout)
    touch_weight: float = 0.3            # Weight for boundary touches
    adx_recovery_weight: float = 0.2     # Weight for how fast ADX recovered
    
    # S/R proximity for touch counting
    touch_proximity_pts: float = 50.0  # BTC: count touches within $50 of edge (was 2.0 MES)
    
    # Targets
    target_range_multiple: float = 1.0    # Target = breakout + (range_width × multiple)
    stop_at_midpoint: bool = True         # Stop at range midpoint (tighter) vs opposite boundary


class ADXRangeMapper:
    """Maps S/R levels during dead ADX periods and generates breakout signals."""
    
    def __init__(self, config: ADXRangeConfig = None):
        self.config = config or ADXRangeConfig()
        
        self.phase = RangePhase.IDLE
        self.current_box: Optional[RangeBox] = None
        self.last_signal: Optional[BreakoutSignal] = None
        
        # History for backtesting
        self.completed_boxes: List[RangeBox] = []
        self.signals: List[BreakoutSignal] = []
        
        # Breakout tracking
        self._breakout_direction: Optional[str] = None
        self._breakout_start_bar: int = 0
        self._breakout_confirm_count: int = 0
        self._adx_at_recovery: float = 0.0
        self._bar_count: int = 0
        
        # ADX tracking for recovery speed
        self._adx_was_dead_at: float = 0.0
        self._adx_recovered_at: float = 0.0
    
    def update(self, price: float, high: float, low: float, 
               adx: float, atr: float = 0.0) -> Optional[BreakoutSignal]:
        """Process a new bar and return breakout signal if detected.
        
        Args:
            price: Current/close price
            high: Bar high
            low: Bar low
            adx: 3-minute ADX value
            atr: Current ATR (for dynamic thresholds)
            
        Returns:
            BreakoutSignal if breakout detected, None otherwise
        """
        self._bar_count += 1
        now = time.time()
        
        # ── Phase: IDLE — looking for dead ADX to start mapping ──
        if self.phase == RangePhase.IDLE:
            if adx < self.config.adx_dead_threshold:
                # ADX just died — start mapping
                self.phase = RangePhase.MAPPING
                self.current_box = RangeBox(
                    high=high,
                    low=low,
                    start_time=now,
                    start_bar=self._bar_count,
                )
                self._adx_was_dead_at = now
                logger.info(
                    f"📦 [RANGE MAPPER] ADX dead ({adx:.1f} < {self.config.adx_dead_threshold}) "
                    f"— started mapping range @ {price:.2f}"
                )
            return None
        
        # ── Phase: MAPPING — ADX is dead, building the range box ──
        elif self.phase == RangePhase.MAPPING:
            self.current_box.update(high, low, price, self.config.touch_proximity_pts)
            
            if adx >= self.config.adx_alive_threshold:
                # ADX woke up — evaluate if range is valid
                box = self.current_box
                self._adx_recovered_at = now
                self._adx_at_recovery = adx
                
                if self._is_valid_range(box, adx):
                    self.phase = RangePhase.BREAKOUT_PENDING
                    self._breakout_confirm_count = 0
                    self._breakout_direction = None
                    logger.warning(
                        f"📦 [RANGE MAPPER] ADX alive ({adx:.1f}) — valid range "
                        f"[{box.low:.2f}–{box.high:.2f}] ({box.width:.1f}pts, "
                        f"{box.bar_count} bars, {box.duration_minutes:.0f}min) — "
                        f"watching for breakout"
                    )
                else:
                    # Range too small/short — discard
                    reason = self._invalid_reason(box, adx)
                    logger.info(
                        f"📦 [RANGE MAPPER] ADX alive but range invalid: {reason} — discarding"
                    )
                    self.phase = RangePhase.IDLE
                    self.current_box = None
            
            elif adx < self.config.adx_dead_threshold:
                # Still dead — keep mapping, periodic log
                if self.current_box.bar_count % 30 == 0:
                    box = self.current_box
                    logger.info(
                        f"📦 [RANGE MAPPER] Mapping... [{box.low:.2f}–{box.high:.2f}] "
                        f"({box.width:.1f}pts, {box.bar_count} bars, "
                        f"H×{box.touch_high} L×{box.touch_low})"
                    )
            
            return None
        
        # ── Phase: BREAKOUT_PENDING — ADX alive, watching for price breakout ──
        elif self.phase == RangePhase.BREAKOUT_PENDING:
            box = self.current_box
            buffer = self.config.breakout_buffer_pts
            
            # Check for breakout above range
            if price > box.high + buffer:
                if self._breakout_direction == "LONG":
                    self._breakout_confirm_count += 1
                else:
                    self._breakout_direction = "LONG"
                    self._breakout_confirm_count = 1
                    self._breakout_start_bar = self._bar_count
            
            # Check for breakout below range
            elif price < box.low - buffer:
                if self._breakout_direction == "SHORT":
                    self._breakout_confirm_count += 1
                else:
                    self._breakout_direction = "SHORT"
                    self._breakout_confirm_count = 1
                    self._breakout_start_bar = self._bar_count
            
            else:
                # Price back inside range — reset confirmation
                if self._breakout_direction:
                    logger.info(
                        f"📦 [RANGE MAPPER] Breakout {self._breakout_direction} failed — "
                        f"price back in range @ {price:.2f}"
                    )
                self._breakout_direction = None
                self._breakout_confirm_count = 0
                
                # If ADX dies again, go back to mapping
                if adx < self.config.adx_dead_threshold:
                    self.phase = RangePhase.MAPPING
                    logger.info(f"📦 [RANGE MAPPER] ADX dead again ({adx:.1f}) — back to mapping")
                    return None
            
            # Confirmed breakout?
            if self._breakout_confirm_count >= self.config.breakout_confirmation_bars:
                signal = self._generate_signal(price, box)
                
                # Archive
                self.completed_boxes.append(box)
                self.signals.append(signal)
                self.last_signal = signal
                
                self.phase = RangePhase.BREAKOUT_ACTIVE
                
                logger.warning(
                    f"🚀 [RANGE MAPPER] BREAKOUT {signal.direction}! "
                    f"Price {price:.2f} broke {'above' if signal.direction == 'LONG' else 'below'} "
                    f"range [{box.low:.2f}–{box.high:.2f}] | "
                    f"Strength: {signal.strength:.2f} | "
                    f"Stop: {signal.stop_level:.2f} | Target: {signal.target_level:.2f} | "
                    f"R:R {signal.reward_pts:.1f}:{signal.risk_pts:.1f}"
                )
                
                return signal
            
            return None
        
        # ── Phase: BREAKOUT_ACTIVE — signal was fired, monitoring ──
        elif self.phase == RangePhase.BREAKOUT_ACTIVE:
            box = self.current_box
            
            # Check for failed breakout (price re-enters range significantly)
            reentry = self.config.failed_breakout_reentry_pts
            if self._breakout_direction == "LONG" and price < box.high - reentry:
                logger.warning(
                    f"📦 [RANGE MAPPER] FAILED breakout LONG — price {price:.2f} "
                    f"re-entered range (high was {box.high:.2f})"
                )
                self.phase = RangePhase.IDLE
                self.current_box = None
                self._breakout_direction = None
            elif self._breakout_direction == "SHORT" and price > box.low + reentry:
                logger.warning(
                    f"📦 [RANGE MAPPER] FAILED breakout SHORT — price {price:.2f} "
                    f"re-entered range (low was {box.low:.2f})"
                )
                self.phase = RangePhase.IDLE
                self.current_box = None
                self._breakout_direction = None
            
            # If ADX dies again after breakout, reset
            if adx < self.config.adx_dead_threshold:
                self.phase = RangePhase.IDLE
                self.current_box = None
                self._breakout_direction = None
            
            return None
        
        return None
    
    def _is_valid_range(self, box: RangeBox, adx: float = 0.0) -> bool:
        """Check if a range box is valid for breakout trading.
        
        When ADX < alive_threshold, skip max width check — if the market
        has been ranging, the range is valid no matter how wide.
        """
        if box.bar_count < self.config.min_range_bars:
            return False
        if box.width < self.config.min_range_width_pts:
            return False
        # Only enforce max width when ADX is high (trending) — 
        # during low ADX, wide ranges are genuine chop zones
        if adx >= self.config.adx_alive_threshold and box.width > self.config.max_range_width_pts:
            return False
        if box.touch_high + box.touch_low < self.config.min_touches:
            return False
        return True
    
    def _invalid_reason(self, box: RangeBox, adx: float = 0.0) -> str:
        """Explain why a range is invalid."""
        reasons = []
        if box.bar_count < self.config.min_range_bars:
            reasons.append(f"too short ({box.bar_count} < {self.config.min_range_bars} bars)")
        if box.width < self.config.min_range_width_pts:
            reasons.append(f"too narrow ({box.width:.1f} < {self.config.min_range_width_pts}pts)")
        if adx >= self.config.adx_alive_threshold and box.width > self.config.max_range_width_pts:
            reasons.append(f"too wide ({box.width:.1f} > {self.config.max_range_width_pts}pts)")
        if box.touch_high + box.touch_low < self.config.min_touches:
            reasons.append(f"too few touches ({box.touch_high + box.touch_low} < {self.config.min_touches})")
        return "; ".join(reasons) if reasons else "unknown"
    
    def _generate_signal(self, price: float, box: RangeBox) -> BreakoutSignal:
        """Generate a breakout signal with strength scoring."""
        direction = self._breakout_direction
        
        # ── Strength scoring ──
        cfg = self.config
        
        # Duration score (longer range = more energy stored = stronger breakout)
        # Normalize: 10 bars = 0.0, 100+ bars = 1.0
        duration_score = min(1.0, max(0.0, (box.bar_count - cfg.min_range_bars) / 90.0))
        
        # Width score (wider range = more significant)
        width_score = min(1.0, max(0.0, (box.width - cfg.min_range_width_pts) / 15.0))
        
        # Touch score (more touches = stronger S/R = more energy on break)
        total_touches = box.touch_high + box.touch_low
        touch_score = min(1.0, max(0.0, (total_touches - cfg.min_touches) / 8.0))
        
        # ADX recovery speed (fast recovery = strong momentum)
        if self._adx_recovered_at > self._adx_was_dead_at:
            recovery_minutes = (self._adx_recovered_at - self._adx_was_dead_at) / 60.0
            # Fast recovery (< 5 min) = 1.0, slow (> 30 min) = 0.0
            adx_score = max(0.0, 1.0 - (recovery_minutes - 5.0) / 25.0)
        else:
            adx_score = 0.5
        
        strength = (
            duration_score * cfg.duration_weight +
            width_score * cfg.width_weight +
            touch_score * cfg.touch_weight +
            adx_score * cfg.adx_recovery_weight
        )
        strength = min(1.0, max(0.0, strength))
        
        # ── Stop and target levels ──
        if direction == "LONG":
            if cfg.stop_at_midpoint:
                stop_level = box.midpoint
            else:
                stop_level = box.low
            target_level = price + (box.width * cfg.target_range_multiple)
        else:  # SHORT
            if cfg.stop_at_midpoint:
                stop_level = box.midpoint
            else:
                stop_level = box.high
            target_level = price - (box.width * cfg.target_range_multiple)
        
        return BreakoutSignal(
            direction=direction,
            breakout_price=price,
            range_high=box.high,
            range_low=box.low,
            range_width=box.width,
            range_duration_bars=box.bar_count,
            range_duration_minutes=box.duration_minutes,
            strength=strength,
            timestamp=time.time(),
            stop_level=stop_level,
            target_level=target_level,
        )
    
    def get_current_range(self) -> Optional[dict]:
        """Get the current range being mapped (if any). For display/debugging."""
        if self.current_box is None:
            return None
        box = self.current_box
        return {
            "phase": self.phase.value,
            "high": box.high,
            "low": box.low,
            "width": box.width,
            "midpoint": box.midpoint,
            "bars": box.bar_count,
            "duration_min": box.duration_minutes,
            "touch_high": box.touch_high,
            "touch_low": box.touch_low,
        }
    
    def reset(self):
        """Reset the mapper (e.g., at session boundaries)."""
        self.phase = RangePhase.IDLE
        self.current_box = None
        self._breakout_direction = None
        self._breakout_confirm_count = 0


# ── Backtest Helper ──────────────────────────────────────────────────

def backtest_range_mapper(bars: list, adx_values: list, atr_values: list = None,
                          config: ADXRangeConfig = None) -> dict:
    """Run range mapper over historical bars and return results.
    
    Args:
        bars: List of dicts with 'h', 'l', 'c' keys
        adx_values: List of 3-min ADX values (same length as bars)
        atr_values: Optional ATR values
        config: Optional config override
        
    Returns:
        dict with 'signals', 'ranges', 'stats'
    """
    mapper = ADXRangeMapper(config)
    
    results = {
        "signals": [],
        "ranges": [],
        "stats": {
            "total_ranges": 0,
            "valid_ranges": 0,
            "breakouts": 0,
            "long_breakouts": 0,
            "short_breakouts": 0,
            "avg_range_width": 0.0,
            "avg_range_duration_bars": 0.0,
            "avg_strength": 0.0,
        }
    }
    
    for i, bar in enumerate(bars):
        price = bar.get('c', bar.get('close', 0))
        high = bar.get('h', bar.get('high', 0))
        low = bar.get('l', bar.get('low', 0))
        adx = adx_values[i] if i < len(adx_values) else 0
        atr = atr_values[i] if atr_values and i < len(atr_values) else 0
        
        # Override time.time() for backtesting — use bar index as proxy
        mapper._bar_count = i
        
        signal = mapper.update(price, high, low, adx, atr)
        
        if signal:
            results["signals"].append({
                "bar_index": i,
                "direction": signal.direction,
                "breakout_price": signal.breakout_price,
                "range_high": signal.range_high,
                "range_low": signal.range_low,
                "range_width": signal.range_width,
                "range_duration_bars": signal.range_duration_bars,
                "strength": signal.strength,
                "stop_level": signal.stop_level,
                "target_level": signal.target_level,
                "risk_pts": signal.risk_pts,
                "reward_pts": signal.reward_pts,
            })
    
    # Compile stats
    results["ranges"] = [
        {"high": b.high, "low": b.low, "width": b.width, "bars": b.bar_count,
         "touch_high": b.touch_high, "touch_low": b.touch_low}
        for b in mapper.completed_boxes
    ]
    
    sigs = results["signals"]
    results["stats"]["total_ranges"] = len(mapper.completed_boxes)
    results["stats"]["breakouts"] = len(sigs)
    results["stats"]["long_breakouts"] = sum(1 for s in sigs if s["direction"] == "LONG")
    results["stats"]["short_breakouts"] = sum(1 for s in sigs if s["direction"] == "SHORT")
    
    if sigs:
        results["stats"]["avg_range_width"] = sum(s["range_width"] for s in sigs) / len(sigs)
        results["stats"]["avg_range_duration_bars"] = sum(s["range_duration_bars"] for s in sigs) / len(sigs)
        results["stats"]["avg_strength"] = sum(s["strength"] for s in sigs) / len(sigs)
    
    return results
