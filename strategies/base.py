"""Base strategy interface for Topstep5.

Every strategy must implement:
- should_enter(): Returns signal direction and strength
- should_exit(): Returns whether to close current position
- get_stops(): Returns stop loss and take profit levels

Strategies are SIMPLE. They answer one question: "Is there a trade here?"
Risk management is handled separately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """A trading signal from a strategy."""
    direction: Direction
    strength: float          # 0.0 to 1.0 (raw strategy confidence)
    strategy_name: str
    reason: str              # Human-readable reason for the signal
    entry_price: float
    stop_loss: float
    take_profit: float
    max_hold_seconds: int = 300  # Default 5 min max hold for scalps
    
    @property
    def risk_reward(self) -> float:
        """Risk/reward ratio. Must be > 1.0 to be worth trading."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0


@dataclass 
class MarketState:
    """Current market state passed to strategies.
    
    Strategies should NOT access global state or maintain complex internal state.
    Everything they need is in this object.
    """
    timestamp: float
    price: float
    bid: float
    ask: float
    spread: float
    
    # Bar data (1-min bars)
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    bar_volume: int
    
    # Key indicators (computed by core/features.py)
    vwap: float
    vwap_std: float
    atr_14: float
    rsi_14: float
    
    # Volume
    volume_ratio_5: float    # Current volume / 5-bar avg
    delta: float             # Buy volume - sell volume (order flow)
    cumulative_delta: float
    
    # Session context
    session: str             # 'LONDON', 'NY_OPEN', 'US_MIDDAY', 'US_CLOSE', 'OVERNIGHT'
    minutes_since_open: int
    daily_high: float
    daily_low: float
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    
    # Regime
    regime: str = 'unknown'  # 'trending', 'ranging', 'volatile'
    
    # EMAs (1-min)
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    
    # Multi-timeframe EMA levels (dict of label→price for dynamic S/R)
    ema_levels: dict = None
    
    # Multi-timeframe EMAs (for trend confirmation)
    ema_5m_9: float = 0.0
    ema_5m_26: float = 0.0
    ema_5m_50: float = 0.0
    
    # ADX and Directional Indicators
    adx_14: float = 0.0
    di_plus_14: float = 0.0
    di_minus_14: float = 0.0
    
    # 3-minute ADX (less noisy trend confirmation)
    adx_3m: float = 0.0
    di_plus_3m: float = 0.0
    di_minus_3m: float = 0.0
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0

    # RSI(5) for short-term mean reversion
    rsi_5: float = 50.0
    prev_rsi_5: float = 50.0  # Previous bar's RSI(5) for declining check

    # Momentum slopes (basis points per bar)
    slope_12: float = 0.0    # 12-bar momentum slope
    slope_75: float = 0.0    # 75-bar momentum slope
    atr_median: float = 0.0  # Rolling median ATR (for grinding detection)
    
    # Rolling price extremes (for entry location filter)
    high_20: float = 0.0     # Highest high over last 20 bars
    low_20: float = 0.0      # Lowest low over last 20 bars


class Strategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        """Check if there's a trade entry signal.
        
        Returns Signal if there's a trade, None otherwise.
        This should be FAST — called on every tick.
        """
        pass
    
    @abstractmethod
    def should_exit(self, state: MarketState, entry_price: float, 
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        """Check if current position should be exited.
        
        Returns exit reason string if should exit, None if should hold.
        Note: Stop loss and take profit are handled by risk manager,
        not the strategy. This is for strategy-specific exits only
        (e.g., VWAP reversion complete, signal reversed).
        """
        pass
    
    def __repr__(self):
        return f"Strategy({self.name})"
