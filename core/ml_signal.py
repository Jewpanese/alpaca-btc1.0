"""
ML Signal Provider — LightGBM Direction Classifier for Live Trading.

Loads production models and provides real-time direction signals
from 5-minute bar data. Computes features incrementally and returns
calibrated probabilities.

Usage:
    provider = MLSignalProvider.load("models/production")
    provider.update_bar(bar_dict)  # call on every 5-min bar
    signal = provider.get_signal()
    if signal.tradeable:
        print(f"Direction: {signal.direction}, Confidence: {signal.confidence:.3f}")
"""

import os
import json
import pickle
import logging
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MLSignal:
    """ML model prediction for current bar."""
    direction: str  # "LONG", "SHORT", or "FLAT"
    confidence: float  # 0.0 to 1.0 (calibrated probability of predicted direction)
    raw_prob: float  # raw P(UP) from model
    tradeable: bool  # True if confidence exceeds threshold
    reason: str  # Human-readable reason


class MLSignalProvider:
    """
    Real-time ML signal provider using LightGBM production models.
    
    Maintains a rolling window of bars and computes features
    incrementally. Returns calibrated direction signals.
    """
    
    def __init__(self, direction_model, calibrator, config: dict):
        self.direction_model = direction_model
        self.calibrator = calibrator
        self.config = config
        
        self.bar_size = config.get('bar_size', 5)
        self.direction_threshold = config.get('direction_threshold', 0.55)
        self.direction_features = config.get('direction_features', [])
        
        # Rolling bar window (need ~150 bars for longest lookback features)
        self._max_bars = 300
        self._bars = deque(maxlen=self._max_bars)
        
        # Cached state
        self._last_signal: Optional[MLSignal] = None
        self._bar_count = 0
        self._warmup_bars = 120  # need this many bars before signals are reliable
        
        # Incremental EMA state
        self._ema_cache = {}  # {span: value}
        
        logger.info(f"ML Signal Provider loaded: {len(self.direction_features)} features, "
                     f"threshold={self.direction_threshold}")
    
    def warm_up_from_file(
        self,
        data_dir: str = r"C:\development\topstep3\data",
        symbol_glob: str = "ES *.Last.txt",
    ):
        """Pre-load recent historical bars so the model is ready immediately.

        Originally tuned for the topstep3/topstep5 ES data feed. For the BTC
        port there's no equivalent pre-built dataset on disk, so this method
        is effectively a no-op (the caller usually doesn't have the file).
        Pass `symbol_glob="BTC *.csv"` (or similar) if you build a BTC dataset.
        """
        import glob
        try:
            import pandas as pd
            pattern = os.path.join(data_dir, symbol_glob)
            files = sorted(glob.glob(pattern))
            if not files:
                logger.warning(
                    f"No data files for ML warmup at {pattern} — "
                    f"ML signal provider will warm up from live bars instead."
                )
                return
            
            # Load last file (most recent data)
            df = pd.read_csv(files[-1], sep=';', header=None,
                             names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
            df = df.set_index('datetime')
            
            # Resample to 5-min bars
            bars_5m = df.resample(f'{self.bar_size}min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()
            
            # Feed last 200 bars (more than enough for warmup)
            warmup_bars = bars_5m.tail(200).reset_index()
            for _, row in warmup_bars.iterrows():
                bar = {
                    't': str(row['datetime']),
                    'o': row['open'], 'h': row['high'],
                    'l': row['low'], 'c': row['close'], 'v': row['volume'],
                }
                self.update_bar(bar)
            
            sig = self.get_signal()
            logger.info(f"ML warmup complete: {self._bar_count} bars loaded, "
                        f"signal ready: {sig.direction} (conf={sig.confidence:.3f})")
        except Exception as e:
            logger.warning(f"ML warmup from file failed: {e}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'MLSignalProvider':
        """Load production models from directory."""
        config_path = os.path.join(model_dir, 'config.json')
        dir_model_path = os.path.join(model_dir, 'direction_model.txt')
        cal_path = os.path.join(model_dir, 'calibrator.pkl')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {model_dir}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        direction_model = lgb.Booster(model_file=dir_model_path)
        logger.info(f"Direction model loaded: {dir_model_path}")
        
        with open(cal_path, 'rb') as f:
            calibrator = pickle.load(f)
        logger.info(f"Calibrator loaded: {cal_path}")
        
        return cls(direction_model, calibrator, config)
    
    def update_bar(self, bar: dict):
        """
        Add a new bar and compute features.
        
        Bar format: {t: timestamp_str, o: float, h: float, l: float, c: float, v: float}
        or {datetime: str, open: float, high: float, low: float, close: float, volume: float}
        """
        # Normalize bar format
        normalized = {
            'close': bar.get('c', bar.get('close', 0)),
            'high': bar.get('h', bar.get('high', 0)),
            'low': bar.get('l', bar.get('low', 0)),
            'open': bar.get('o', bar.get('open', 0)),
            'volume': bar.get('v', bar.get('volume', 0)),
            'timestamp': bar.get('t', bar.get('datetime', '')),
        }
        
        self._bars.append(normalized)
        self._bar_count += 1
        
        # Update incremental EMAs
        close = normalized['close']
        for span in [9, 21, 50]:
            if span not in self._ema_cache:
                self._ema_cache[span] = close
            else:
                mult = 2.0 / (span + 1)
                self._ema_cache[span] = (close - self._ema_cache[span]) * mult + self._ema_cache[span]
        
        # Compute signal if we have enough bars
        if self._bar_count >= self._warmup_bars:
            self._compute_signal()
    
    def get_signal(self) -> MLSignal:
        """Get the latest ML signal."""
        if self._last_signal is None:
            return MLSignal(
                direction="FLAT", confidence=0.0, raw_prob=0.5,
                tradeable=False, reason=f"Warming up ({self._bar_count}/{self._warmup_bars} bars)"
            )
        return self._last_signal
    
    def _compute_signal(self):
        """Compute features from bar window and run model inference."""
        try:
            features = self._compute_features()
            if features is None:
                return
            
            # Model inference
            feature_array = np.array([features[f] for f in self.direction_features]).reshape(1, -1)
            
            # Check for NaN
            if np.any(np.isnan(feature_array)):
                nan_features = [f for i, f in enumerate(self.direction_features) 
                               if np.isnan(feature_array[0, i])]
                self._last_signal = MLSignal(
                    direction="FLAT", confidence=0.0, raw_prob=0.5,
                    tradeable=False, reason=f"NaN in features: {nan_features[:3]}"
                )
                return
            
            raw_prob = self.direction_model.predict(feature_array)[0]
            
            # Calibrate
            try:
                calibrated_prob = self.calibrator.predict(np.array([raw_prob]))[0]
            except Exception:
                calibrated_prob = raw_prob
            
            # Determine direction and confidence
            if calibrated_prob > 0.5:
                direction = "LONG"
                confidence = calibrated_prob
            else:
                direction = "SHORT"
                confidence = 1 - calibrated_prob
            
            tradeable = confidence >= self.direction_threshold
            
            self._last_signal = MLSignal(
                direction=direction if tradeable else "FLAT",
                confidence=confidence,
                raw_prob=raw_prob,
                tradeable=tradeable,
                reason=f"ML:{direction} conf={confidence:.3f} (raw={raw_prob:.3f})"
            )
            
        except Exception as e:
            logger.error(f"ML signal computation failed: {e}")
            self._last_signal = MLSignal(
                direction="FLAT", confidence=0.0, raw_prob=0.5,
                tradeable=False, reason=f"Error: {e}"
            )
    
    def _compute_features(self) -> Optional[dict]:
        """Compute all direction features from the bar window."""
        bars = list(self._bars)
        n = len(bars)
        if n < self._warmup_bars:
            return None
        
        closes = np.array([b['close'] for b in bars])
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        opens = np.array([b['open'] for b in bars])
        volumes = np.array([b['volume'] for b in bars])
        
        c = closes[-1]
        log_ret = np.diff(np.log(closes))
        
        features = {}
        
        # ── VWAP ──
        cum_vp = np.cumsum(closes[-20:] * volumes[-20:])
        cum_v = np.cumsum(volumes[-20:])
        vwap_mom = cum_vp[-1] / (cum_v[-1] + 1e-9) if len(cum_vp) > 0 else c
        features['vwap_mom'] = vwap_mom
        features['vwap_dev'] = (c - vwap_mom) / (vwap_mom + 1e-9) * 10000
        
        # ── Time (from timestamp) ──
        ts = bars[-1].get('timestamp', '')
        hour = 12.0  # default
        minute = 0.0
        if ts:
            try:
                from datetime import datetime
                if isinstance(ts, str):
                    # Try ISO format
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    hour = dt.hour
                    minute = dt.minute
                elif hasattr(ts, 'hour'):
                    hour = ts.hour
                    minute = ts.minute
            except Exception:
                pass
        
        features['mins_from_open'] = (hour + minute / 60.0 - 9.5) * 60
        features['hour'] = float(hour)
        
        # ── Volatility ──
        for w in [10, 20, 50]:
            w = min(w, len(log_ret))
            if w > 1:
                features[f'rvol_{w}'] = float(np.std(log_ret[-w:]) * np.sqrt(252 * 78))  # 78 five-min bars/day
            else:
                features[f'rvol_{w}'] = 0.0
        
        rvol_10 = np.std(log_ret[-10:]) if len(log_ret) >= 10 else 1e-9
        rvol_50 = np.std(log_ret[-50:]) if len(log_ret) >= 50 else 1e-9
        features['vol_ratio_10_50'] = float(rvol_10 / (rvol_50 + 1e-9))
        
        # ── ADX ──
        adx, di_diff = self._compute_adx(highs, lows, closes, 14)
        features['adx_14'] = adx
        features['di_diff'] = di_diff
        
        # ── EMAs ──
        ema_9 = self._ema_cache.get(9, c)
        ema_21 = self._ema_cache.get(21, c)
        ema_50 = self._ema_cache.get(50, c)
        
        features['ema_9_21_cross'] = (ema_9 - ema_21) / c * 10000
        features['ema_21_50_cross'] = (ema_21 - ema_50) / c * 10000
        
        # EMA slope (3-bar change)
        if n >= 4:
            # Approximate ema_9 three bars ago
            ema_9_prev = self._ema_at(closes, 9, -4)
            features['ema_9_slope'] = (ema_9 - ema_9_prev) / (ema_9_prev + 1e-9) * 10000
        else:
            features['ema_9_slope'] = 0.0
        
        features['ema_9_dist'] = (c - ema_9) / (ema_9 + 1e-9) * 10000
        
        # ── Trend rank ──
        if n >= 50:
            adx_window = self._compute_adx_series(highs[-64:], lows[-64:], closes[-64:], 14)
            if len(adx_window) >= 50:
                recent_adx = adx_window[-50:]
                rank = np.sum(recent_adx <= adx) / len(recent_adx)
                features['trend_rank_50'] = float(rank)
            else:
                features['trend_rank_50'] = 0.5
        else:
            features['trend_rank_50'] = 0.5
        
        # ── ATR ──
        tr = self._true_range(highs, lows, closes)
        features['atr_14'] = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0.0
        features['atr_28'] = float(np.mean(tr[-28:])) if len(tr) >= 28 else 0.0
        features['atr_14_pct'] = features['atr_14'] / c * 100 if c > 0 else 0.0
        features['atr_28_pct'] = features['atr_28'] / c * 100 if c > 0 else 0.0
        
        # ── Cumulative delta ──
        bar_range = highs - lows + 1e-9
        delta_pct = (closes - opens) / bar_range
        features['delta_pct'] = float(delta_pct[-1])
        features['cum_delta_20'] = float(np.sum(delta_pct[-20:]))
        
        # ── Distance from extremes ──
        for w in [60, 120]:
            w_actual = min(w, n)
            roll_h = np.max(highs[-w_actual:])
            roll_l = np.min(lows[-w_actual:])
            roll_range = roll_h - roll_l + 1e-9
            features[f'dist_high_{w}'] = float((c - roll_h) / roll_range)
            features[f'dist_low_{w}'] = float((c - roll_l) / roll_range)
        
        # ── Volume ratios ──
        for w in [10, 20]:
            w_actual = min(w, n)
            vol_ma = np.mean(volumes[-w_actual:]) + 1e-9
            features[f'vol_ratio_{w}'] = float(volumes[-1] / vol_ma)
        
        # ── Returns ──
        for w in [1, 3, 5]:
            if n > w:
                features[f'ret_{w}'] = float((c - closes[-(w+1)]) / (closes[-(w+1)] + 1e-9))
            else:
                features[f'ret_{w}'] = 0.0
        
        # ── RSI ──
        features['rsi_14'] = self._compute_rsi(closes, 14)
        
        # ── Bollinger %B ──
        if n >= 20:
            bb_mid = np.mean(closes[-20:])
            bb_std = np.std(closes[-20:])
            features['bb_pctB'] = float((c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9))
        else:
            features['bb_pctB'] = 0.5
        
        # ── Tick imbalance ──
        if n >= 21:
            tick_dirs = np.sign(np.diff(closes[-21:]))
            features['tick_imbalance_20'] = float(np.sum(tick_dirs) / 20)
        else:
            features['tick_imbalance_20'] = 0.0
        
        # ── Calendar ──
        try:
            from datetime import datetime
            if isinstance(ts, str) and ts:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                features['day_of_week'] = dt.weekday()
                features['month'] = dt.month
            else:
                features['day_of_week'] = 2  # default Wednesday
                features['month'] = 3
        except Exception:
            features['day_of_week'] = 2
            features['month'] = 3
        
        return features
    
    # ── Helper methods ──
    
    def _true_range(self, highs, lows, closes):
        """Compute true range array."""
        if len(closes) < 2:
            return np.array([highs[-1] - lows[-1]]) if len(highs) > 0 else np.array([0])
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _compute_adx(self, highs, lows, closes, period=14):
        """Compute current ADX and DI difference."""
        if len(closes) < period * 3:
            return 0.0, 0.0
        
        plus_dm = np.diff(highs)
        minus_dm = -np.diff(lows)
        
        plus_dm_clean = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm_clean = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr = self._true_range(highs, lows, closes)
        n = min(len(tr), len(plus_dm_clean))
        tr = tr[-n:]
        plus_dm_clean = plus_dm_clean[-n:]
        minus_dm_clean = minus_dm_clean[-n:]
        
        if n < period:
            return 0.0, 0.0
        
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm_clean[-period:]) / (atr + 1e-9)
        minus_di = 100 * np.mean(minus_dm_clean[-period:]) / (atr + 1e-9)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        
        # ADX is smoothed DX — use simple average of last `period` DX values
        # Simplified: just return current DX as proxy
        return float(dx), float(plus_di - minus_di)
    
    def _compute_adx_series(self, highs, lows, closes, period=14):
        """Compute ADX series for trend_rank calculation."""
        n = len(closes)
        if n < period * 2:
            return np.array([])
        
        plus_dm = np.diff(highs)
        minus_dm = -np.diff(lows)
        plus_dm_clean = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm_clean = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        tr = self._true_range(highs, lows, closes)
        
        m = min(len(tr), len(plus_dm_clean))
        tr = tr[-m:]
        plus_dm_clean = plus_dm_clean[-m:]
        minus_dm_clean = minus_dm_clean[-m:]
        
        adx_series = []
        for i in range(period, m):
            atr_val = np.mean(tr[i-period:i])
            pdi = 100 * np.mean(plus_dm_clean[i-period:i]) / (atr_val + 1e-9)
            mdi = 100 * np.mean(minus_dm_clean[i-period:i]) / (atr_val + 1e-9)
            dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
            adx_series.append(dx)
        
        return np.array(adx_series)
    
    def _compute_rsi(self, closes, period=14):
        """Compute RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period+1):])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = np.mean(-deltas[deltas < 0]) if np.any(deltas < 0) else 0
        if losses == 0:
            return 100.0
        rs = gains / losses
        return float(100 - (100 / (1 + rs)))
    
    def _ema_at(self, closes, span, offset):
        """Approximate EMA value at a historical offset (negative index)."""
        # Simple approximation: compute EMA of the subset
        subset = closes[:offset] if offset < 0 else closes[:offset+1]
        if len(subset) < span:
            return float(subset[-1]) if len(subset) > 0 else 0.0
        
        mult = 2.0 / (span + 1)
        ema = float(subset[0])
        for val in subset[1:]:
            ema = (float(val) - ema) * mult + ema
        return ema
