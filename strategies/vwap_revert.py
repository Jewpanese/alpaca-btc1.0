"""VWAP Mean Reversion Strategy — with trend awareness.

Only mean-reverts in ranging markets or trades WITH the trend.
Uses EMA alignment and ADX to avoid shorting into uptrends.
"""

from typing import Optional
from .base import Strategy, Signal, MarketState, Direction


class VWAPReversion(Strategy):
    """Mean reversion to VWAP — trend-aware."""
    
    def __init__(
        self,
        vwap_entry_std: float = 1.5,
        vwap_exit_std: float = 0.3,
        min_volume_ratio: float = 0.8,
        atr_stop_multiplier: float = 2.0,
        atr_target_multiplier: float = 3.0,
        max_hold_seconds: int = 900,
    ):
        super().__init__("VWAP_REVERT")
        self.vwap_entry_std = vwap_entry_std
        self.vwap_exit_std = vwap_exit_std
        self.min_volume_ratio = min_volume_ratio
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_target_mult = atr_target_multiplier
        self.max_hold_seconds = max_hold_seconds
    
    def _ema_trend(self, state: MarketState) -> str:
        """Determine trend from EMA alignment."""
        if state.ema_9 > state.ema_21 > state.ema_50 and state.ema_50 > 0:
            return 'up'
        elif state.ema_9 < state.ema_21 < state.ema_50 and state.ema_50 > 0:
            return 'down'
        return 'mixed'
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        if state.vwap_std <= 0 or state.atr_14 <= 0:
            return None
        
        # Trade all sessions — overnight uses higher threshold via vwap_z naturally
        # (lower volume = wider VWAP std = fewer signals, which is correct)
        
        # Volume filter — relaxed during overnight/ETH sessions
        min_vol = self.min_volume_ratio
        if hasattr(state, 'session') and state.session in ('OVERNIGHT', 'LONDON'):
            min_vol = 0.3  # Overnight volume is naturally thin
        if state.volume_ratio_5 < min_vol:
            return None
        
        vwap_z = (state.price - state.vwap) / state.vwap_std
        trend = self._ema_trend(state)
        
        # SHORT: Price far above VWAP
        if vwap_z > self.vwap_entry_std:
            # EMA trend is king — NEVER short into an uptrend regardless of regime
            if trend == 'up':
                return None
            # In mixed/ranging: require delta confirmation (selling pressure)
            if trend == 'mixed':
                if state.delta >= 0 and state.rsi_14 < 70:
                    return None
            # trend == 'down' → short aligns with trend, proceed
            
            stop = state.price + (state.atr_14 * self.atr_stop_mult)
            target = state.vwap + (state.vwap_std * self.vwap_exit_std)
            
            return Signal(
                direction=Direction.SHORT,
                strength=min(abs(vwap_z) / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"VWAP revert SHORT: z={vwap_z:.2f}, trend={trend}, regime={state.regime}",
                entry_price=state.price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        # LONG: Price far below VWAP
        elif vwap_z < -self.vwap_entry_std:
            # EMA trend is king — NEVER long into a downtrend regardless of regime
            if trend == 'down':
                return None
            # In mixed/ranging: require delta confirmation (buying pressure)
            if trend == 'mixed':
                if state.delta <= 0 and state.rsi_14 > 30:
                    return None
            
            stop = state.price - (state.atr_14 * self.atr_stop_mult)
            target = state.vwap - (state.vwap_std * self.vwap_exit_std)
            
            return Signal(
                direction=Direction.LONG,
                strength=min(abs(vwap_z) / 3.0, 1.0),
                strategy_name=self.name,
                reason=f"VWAP revert LONG: z={vwap_z:.2f}, trend={trend}, regime={state.regime}",
                entry_price=state.price,
                stop_loss=stop,
                take_profit=target,
                max_hold_seconds=self.max_hold_seconds,
            )
        
        return None
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        if state.vwap_std <= 0:
            return None
        
        vwap_z = (state.price - state.vwap) / state.vwap_std
        
        if direction == Direction.SHORT and vwap_z < self.vwap_exit_std:
            return f"VWAP reversion complete (z={vwap_z:.2f})"
        
        if direction == Direction.LONG and vwap_z > -self.vwap_exit_std:
            return f"VWAP reversion complete (z={vwap_z:.2f})"
        
        if hold_time_seconds > self.max_hold_seconds:
            return f"Max hold time exceeded ({hold_time_seconds:.0f}s)"
        
        return None
