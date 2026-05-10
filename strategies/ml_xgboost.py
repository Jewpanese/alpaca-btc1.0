"""
ML XGBoost Strategy — Ported from Topstep3's V8.2 Lean Model.

Uses a pre-trained XGBoost model with 18 statistically significant features
to generate BUY/SELL/HOLD signals. Features are computed from raw OHLCV bars.

The model was trained on 5-minute bars. We aggregate the 1-min bars from
FeatureEngine into 5-min bars before computing features and predicting.

Architecture:
  - Load pre-trained XGBoost pipeline (scaler + SMOTE + XGBoost)
  - Aggregate 1-min bars → 5-min bars
  - Compute 18 features from 5-min OHLCV
  - Predict → BUY/SELL/HOLD with confidence
  - Convert to Signal with ATR-based stop/target
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from strategies.base import Strategy, Signal, Direction, MarketState

logger = logging.getLogger(__name__)

# The 18 statistically significant features (from topstep3 feature importance analysis)
TOP_18_FEATURES = [
    'market_sell_vol',
    'atr_pct_5',
    'tick_imbalance_3',
    'price_vs_ma_20',
    'near_support_20',
    'high_volume_bar',
    'is_us_close',
    'trend_strength_20',
    'price_vs_ma_50',
    'vol_regime_low',
    'vwap_slope_3',
    'price_vs_ma_10',
    'near_resistance_20',
    'vol_regime_high',
    'below_vwap_lower',
    'vwap_cumdev_10',
    'volume_ratio_20',
    'is_ny_premarket',
]


class MLXGBoost(Strategy):
    """XGBoost ML strategy using 18-feature lean model."""
    
    def __init__(self,
                 model_path: str = None,
                 signal_threshold: float = 0.10,
                 min_confidence: float = 0.40,
                 atr_stop_mult: float = 1.5,
                 atr_target_mult: float = 2.5,
                 bars_per_5min: int = 5,
                 lookback_5min_bars: int = 100):
        """
        Args:
            model_path: Path to trained XGBoost joblib file
            signal_threshold: Minimum (BUY_prob - SELL_prob) to generate signal
            min_confidence: Minimum max probability to act
            atr_stop_mult: Stop loss in ATR multiples
            atr_target_mult: Take profit in ATR multiples
            bars_per_5min: How many 1-min bars per aggregated bar (5 for 5-min)
            lookback_5min_bars: Number of 5-min bars to keep for feature computation
        """
        super().__init__("ml_xgboost")
        
        self.signal_threshold = signal_threshold
        self.min_confidence = min_confidence
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.bars_per_5min = bars_per_5min
        self.lookback_5min_bars = lookback_5min_bars
        
        # Model
        self.pipeline = None
        self.selected_features = None
        self._model_loaded = False
        
        # Bar aggregation buffer
        self._1min_buffer: list = []  # Accumulate 1-min bars
        self._5min_bars: list = []    # Aggregated 5-min bars
        self._bars_since_predict = 0
        self._last_signal_str: str = "HOLD"
        self._last_probs: list = [0.33, 0.34, 0.33]
        self._signal_proposed_this_cycle: bool = False  # Only propose once per 5-min bar
        
        # Load model
        if model_path is None:
            # Default: look in topstep3's model directory
            model_path = str(Path(__file__).parent.parent.parent / 
                           "topstep3" / "scalp_models_v8" / "xgboost_v8_2_lean.joblib")
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the pre-trained XGBoost pipeline."""
        try:
            import joblib
            model_data = joblib.load(model_path)
            self.pipeline = model_data['pipeline']
            self.selected_features = model_data.get('selected_features', TOP_18_FEATURES)
            self._model_loaded = True
            logger.info(f"[ML_XGBOOST] Model loaded: {model_path}")
            logger.info(f"[ML_XGBOOST] Using {len(self.selected_features)} features")
        except Exception as e:
            logger.error(f"[ML_XGBOOST] Failed to load model: {e}")
            logger.error(f"[ML_XGBOOST] Strategy will return HOLD for all signals")
            self._model_loaded = False
    
    # ─── Bar Aggregation ────────────────────────────────────────────
    
    def feed_bar(self, bar: dict):
        """Feed a 1-min bar for aggregation into 5-min bars.
        
        Call this from the bot's _process_bar before should_enter.
        """
        self._1min_buffer.append(bar)
        self._bars_since_predict += 1
        
        if len(self._1min_buffer) >= self.bars_per_5min:
            self._aggregate_buffer()
    
    def _aggregate_buffer(self):
        """Aggregate buffered 1-min bars into one 5-min bar."""
        buf = self._1min_buffer
        if not buf:
            return
        
        bar_5m = {
            't': buf[0].get('t', ''),
            'o': buf[0]['o'],
            'h': max(b['h'] for b in buf),
            'l': min(b['l'] for b in buf),
            'c': buf[-1]['c'],
            'v': sum(b.get('v', 0) for b in buf),
        }
        
        self._5min_bars.append(bar_5m)
        if len(self._5min_bars) > self.lookback_5min_bars:
            self._5min_bars = self._5min_bars[-self.lookback_5min_bars:]
        
        self._1min_buffer = []
    
    # ─── Feature Engineering (18 features) ──────────────────────────
    
    def _build_features_df(self) -> Optional[pd.DataFrame]:
        """Build a DataFrame of 5-min bars and compute all 18 features."""
        if len(self._5min_bars) < 55:  # Need ~50 bars for MA50 + buffer
            return None
        
        # Build OHLCV DataFrame
        df = pd.DataFrame(self._5min_bars)
        df.columns = [c if c != 'o' else 'open' for c in df.columns]
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})
        
        # Ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        features = pd.DataFrame(index=df.index)
        
        # --- 1. market_sell_vol: volume on down bars ---
        price_change = df['close'] - df['open']
        features['market_sell_vol'] = np.where(price_change < 0, df['volume'], 0)
        
        # --- 2. atr_pct_5: ATR(5) / close ---
        hl = df['high'] - df['low']
        atr_5 = hl.rolling(5).mean()
        features['atr_pct_5'] = atr_5 / df['close']
        
        # --- 3. tick_imbalance_3: uptick vs downtick (3-bar) ---
        tick_dir = np.sign(df['close'] - df['close'].shift(1))
        upticks = (tick_dir == 1).rolling(3).sum()
        downticks = (tick_dir == -1).rolling(3).sum()
        features['tick_imbalance_3'] = (upticks - downticks) / 3
        
        # --- 4,12,9. price_vs_ma_10/20/50 ---
        for window in [10, 20, 50]:
            ma = df['close'].rolling(window).mean()
            features[f'price_vs_ma_{window}'] = (df['close'] - ma) / ma
        
        # --- 5,13. near_support_20, near_resistance_20 ---
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        range_20 = high_20 - low_20 + 1e-9
        features['near_support_20'] = (df['close'] - low_20) / range_20
        features['near_resistance_20'] = (high_20 - df['close']) / range_20
        
        # --- 6. high_volume_bar ---
        vol_pctl = df['volume'].rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        features['high_volume_bar'] = (vol_pctl > 0.80).astype(int)
        
        # --- 7,18. Session flags (is_us_close, is_ny_premarket) ---
        # Parse timestamps to get hour (ET)
        try:
            ts = pd.to_datetime(df['timestamp'])
            hour = ts.dt.hour
            features['is_us_close'] = ((hour >= 15) & (hour < 16)).astype(int)
            features['is_ny_premarket'] = ((hour >= 8) & (hour < 10)).astype(int)  # ~8-9:30 ET
        except:
            features['is_us_close'] = 0
            features['is_ny_premarket'] = 0
        
        # --- 8. trend_strength_20 (ADX-like) ---
        plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
        minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
        tr = df['high'] - df['low']
        plus_di = plus_dm.rolling(20).sum() / (tr.rolling(20).sum() + 1e-9)
        minus_di = minus_dm.rolling(20).sum() / (tr.rolling(20).sum() + 1e-9)
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        features['trend_strength_20'] = dx.rolling(20).mean()
        
        # --- 10,14. vol_regime_low, vol_regime_high ---
        atr_14 = hl.rolling(14).mean()
        atr_pctl = atr_14.rolling(100, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        features['vol_regime_low'] = (atr_pctl < 0.25).astype(int)
        features['vol_regime_high'] = (atr_pctl > 0.75).astype(int)
        
        # --- 11. vwap_slope_3 ---
        typical = (df['high'] + df['low'] + df['close']) / 3
        cum_tpv = (typical * df['volume']).rolling(50).sum()
        cum_vol = df['volume'].rolling(50).sum()
        vwap = cum_tpv / (cum_vol + 1e-9)
        features['vwap_slope_3'] = vwap.diff(3) / vwap.shift(3)
        
        # --- 15. below_vwap_lower ---
        vwap_std = ((typical - vwap) ** 2).rolling(20).mean() ** 0.5
        lower_band = vwap - 2 * vwap_std
        features['below_vwap_lower'] = (df['close'] < lower_band).astype(int)
        
        # --- 16. vwap_cumdev_10 ---
        vwap_distance = (df['close'] - vwap) / vwap
        features['vwap_cumdev_10'] = vwap_distance.rolling(10).sum()
        
        # --- 17. volume_ratio_20 ---
        vol_ma_20 = df['volume'].rolling(20).mean()
        features['volume_ratio_20'] = df['volume'] / (vol_ma_20 + 1e-9)
        
        return features
    
    # ─── Prediction ─────────────────────────────────────────────────
    
    def _predict(self) -> dict:
        """Run XGBoost prediction on current 5-min bar data."""
        if not self._model_loaded:
            return {'signal': 'HOLD', 'confidence': 0.0, 'probs': [0.33, 0.34, 0.33]}
        
        features = self._build_features_df()
        if features is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'probs': [0.33, 0.34, 0.33]}
        
        try:
            # Select only the features the model expects
            X = features[self.selected_features]
            X_last = X.iloc[[-1]]
            
            if X_last.isnull().any().any():
                return {'signal': 'HOLD', 'confidence': 0.0, 'probs': [0.33, 0.34, 0.33]}
            
            probs = self.pipeline.predict_proba(X_last)[0]  # [SELL, HOLD, BUY]
            pred = self.pipeline.predict(X_last)[0]
            
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            return {
                'signal': signal_map[pred],
                'confidence': float(probs.max()),
                'probs': probs.tolist(),
                'strength': float(probs[2] - probs[0]),  # BUY - SELL
            }
        except Exception as e:
            logger.error(f"[ML_XGBOOST] Prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'probs': [0.33, 0.34, 0.33]}
    
    # ─── Strategy Interface ─────────────────────────────────────────
    
    def should_enter(self, state: MarketState) -> Optional[Signal]:
        """Check for ML-generated entry signal."""
        if not self._model_loaded:
            return None
        
        # Only predict on new 5-min bar completions
        if self._bars_since_predict < self.bars_per_5min:
            # Between 5-min bars — signal already proposed (or not), don't re-propose
            if self._signal_proposed_this_cycle:
                return None
        else:
            result = self._predict()
            self._last_signal_str = result['signal']
            self._last_probs = result['probs']
            self._bars_since_predict = 0
            self._signal_proposed_this_cycle = False  # New cycle, allow one proposal
        
        # Check signal threshold
        signal_strength = self._last_probs[2] - self._last_probs[0]  # BUY - SELL
        confidence = max(self._last_probs)
        
        if confidence < self.min_confidence:
            return None
        
        if abs(signal_strength) < self.signal_threshold:
            return None
        
        # Determine direction
        if signal_strength > self.signal_threshold:
            direction = Direction.LONG
        elif signal_strength < -self.signal_threshold:
            direction = Direction.SHORT
        else:
            return None
        
        # ATR-based stops
        atr = state.atr_14 if state.atr_14 > 0 else 2.5
        stop_pts = atr * self.atr_stop_mult
        target_pts = atr * self.atr_target_mult
        
        if direction == Direction.LONG:
            stop = state.price - stop_pts
            target = state.price + target_pts
        else:
            stop = state.price + stop_pts
            target = state.price - target_pts
        
        self._signal_proposed_this_cycle = True  # One proposal per 5-min bar
        
        return Signal(
            direction=direction,
            strength=min(abs(signal_strength), 1.0),
            strategy_name=self.name,
            reason=f"ML XGBoost: {self._last_signal_str} "
                   f"(BUY={self._last_probs[2]:.0%} SELL={self._last_probs[0]:.0%} "
                   f"strength={signal_strength:+.3f})",
            entry_price=state.price,
            stop_loss=stop,
            take_profit=target,
            max_hold_seconds=600,  # 10 min for ML trades
        )
    
    def should_exit(self, state: MarketState, entry_price: float,
                    direction: Direction, hold_time_seconds: float) -> Optional[str]:
        """ML-based exit: signal reversal."""
        if not self._model_loaded:
            return None
        
        signal_strength = self._last_probs[2] - self._last_probs[0]
        
        # Exit if signal flips strongly against us
        if direction == Direction.LONG and signal_strength < -self.signal_threshold * 1.5:
            return f"ML signal reversal (strength={signal_strength:+.3f})"
        if direction == Direction.SHORT and signal_strength > self.signal_threshold * 1.5:
            return f"ML signal reversal (strength={signal_strength:+.3f})"
        
        return None
