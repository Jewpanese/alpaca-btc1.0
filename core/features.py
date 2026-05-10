"""
Feature Engineering — Clean, documented, no magic numbers.

Computes everything strategies need from raw bar data.
Each feature has a clear reason for existing.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional
import logging

import ta.trend
import ta.momentum
import ta.volatility

logger = logging.getLogger(__name__)

# MAX_ATR caps the ATR value to prevent position-sizing math from exploding on a
# flash-spike. The original $15 cap was tuned for ES/MES (~3-7 pt typical ATR;
# >15 = real news spike like NFP/FOMC). For BTC at ~$80k this is wildly wrong:
# 3-min BTC ATR is normally $50–300 in calm conditions and $300–1000 in active
# regimes — values that the ES cap would misread as "news spikes" and clamp to
# $15, breaking every downstream stop / trade-quality / vol_z calculation.
# We still want a cap so a true 5-sigma event doesn't size positions on a $5000
# ATR reading; $2500 catches that without false-positiving on normal BTC moves.
MAX_ATR = 2500.0


def _safe_float(val, default: float = 0.0) -> float:
    """Return float, or default if val is NaN/None/inf."""
    try:
        f = float(val)
        return default if (f != f or f == float("inf") or f == float("-inf")) else f
    except (TypeError, ValueError):
        return default


class FeatureEngine:
    """Computes market features from OHLCV bar data."""

    def __init__(self, tick_size: float = 0.25):
        self.tick_size = tick_size
        self._bars: list = []          # Rolling window of bar dicts
        self._max_bars: int = 500

        # Internal DataFrame for ta library computations (DatetimeIndex, float64)
        self._df: pd.DataFrame = pd.DataFrame({
            "open":   pd.Series(dtype="float64"),
            "high":   pd.Series(dtype="float64"),
            "low":    pd.Series(dtype="float64"),
            "close":  pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        })

        # Running VWAP state (resets daily)
        self._cum_vol = 0.0
        self._cum_vwap = 0.0
        self._cum_vwap_sq = 0.0
        self._vwap_reset_date: Optional[str] = None

        # Incremental EMA cache: {period: current_ema_value}
        # Seeded on first call to ema(); updated in add_bar() each tick.
        self._ema_cache: dict = {}

        self._bar_count_total: int = 0   # Used as fallback timestamp counter

    # ── Bar ingestion ────────────────────────────────────────────────────

    def _parse_timestamp(self, t) -> pd.Timestamp:
        """Convert bar "t" field to pandas Timestamp for DataFrame index."""
        try:
            if isinstance(t, (int, float)):
                # Unix ms if > year-2001 threshold, else seconds
                return pd.Timestamp(t, unit="ms") if t > 1e12 else pd.Timestamp(t, unit="s")
            return pd.Timestamp(t)
        except Exception:
            return pd.Timestamp(self._bar_count_total, unit="s")

    def add_bar(self, bar: dict):
        """Add a new bar. Bar format: {t, o, h, l, c, v} (TopstepX format)."""
        self._bars.append(bar)
        if len(self._bars) > self._max_bars:
            self._bars = self._bars[-self._max_bars:]

        self._bar_count_total += 1
        close = bar["c"]

        # Append row to internal DataFrame
        ts = self._parse_timestamp(bar.get("t", self._bar_count_total))
        new_row = pd.DataFrame(
            {
                "open":   [float(bar["o"])],
                "high":   [float(bar["h"])],
                "low":    [float(bar["l"])],
                "close":  [float(bar["c"])],
                "volume": [float(bar.get("v", 0))],
            },
            index=[ts],
        )
        self._df = pd.concat([self._df, new_row])
        if len(self._df) > self._max_bars:
            self._df = self._df.iloc[-self._max_bars:]

        # Update incremental EMA cache (O(k) where k = number of cached periods)
        for period in list(self._ema_cache.keys()):
            mult = 2 / (period + 1)
            self._ema_cache[period] = (close - self._ema_cache[period]) * mult + self._ema_cache[period]

        # Update running VWAP
        typical = (bar["h"] + bar["l"] + close) / 3
        vol = max(bar.get("v", 0), 1)

        bar_date = str(bar.get("t", ""))[:10]
        if bar_date != self._vwap_reset_date:
            self._cum_vol = 0.0
            self._cum_vwap = 0.0
            self._cum_vwap_sq = 0.0
            self._vwap_reset_date = bar_date

        self._cum_vol += vol
        self._cum_vwap += typical * vol
        self._cum_vwap_sq += (typical ** 2) * vol

    # ── Properties / helpers ─────────────────────────────────────────────

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    @property
    def last_price(self) -> float:
        return self._bars[-1]["c"] if self._bars else 0.0

    def get_closes(self, n: int = None) -> np.ndarray:
        bars = self._bars[-n:] if n else self._bars
        return np.array([b["c"] for b in bars])

    def get_highs(self, n: int = None) -> np.ndarray:
        bars = self._bars[-n:] if n else self._bars
        return np.array([b["h"] for b in bars])

    def get_lows(self, n: int = None) -> np.ndarray:
        bars = self._bars[-n:] if n else self._bars
        return np.array([b["l"] for b in bars])

    def get_volumes(self, n: int = None) -> np.ndarray:
        bars = self._bars[-n:] if n else self._bars
        return np.array([b.get("v", 0) for b in bars])

    def _resample_df(self, timeframe_minutes: int) -> pd.DataFrame:
        """Resample 1-min DataFrame to a higher timeframe using pandas resample.

        Produces proper OHLCV candles — fixes the old chunk-boundary drift bug
        that caused mtf_adx to wildly overstate trend strength.
        """
        if len(self._df) < timeframe_minutes:
            return pd.DataFrame()
        rule = f"{timeframe_minutes}min"
        df_htf = self._df.resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])
        return df_htf

    # ── VWAP (kept as running cumulative — pandas_ta VWAP needs full-day data) ──

    def vwap(self) -> float:
        """Volume-Weighted Average Price (intraday, resets daily)."""
        if self._cum_vol <= 0:
            return self.last_price
        return self._cum_vwap / self._cum_vol

    def vwap_std(self) -> float:
        """Standard deviation around VWAP."""
        if self._cum_vol <= 0:
            return 1.0
        mean = self._cum_vwap / self._cum_vol
        var = (self._cum_vwap_sq / self._cum_vol) - (mean ** 2)
        return max(np.sqrt(max(var, 0)), 0.01)

    # ── ATR ──────────────────────────────────────────────────────────────

    def atr(self, period: int = 14) -> float:
        """Average True Range via ta.volatility.AverageTrueRange (Wilder's smoothing).
        Clamped at MAX_ATR (BTC scale: $2500) to protect against true 5-sigma
        spikes without false-positiving on normal BTC volatility.
        """
        if len(self._df) < period + 1:
            return 1.0

        indicator = ta.volatility.AverageTrueRange(
            high=self._df["high"],
            low=self._df["low"],
            close=self._df["close"],
            window=period,
            fillna=False,
        )
        val = _safe_float(indicator.average_true_range().iloc[-1], default=1.0)
        if val <= 0:
            val = 1.0

        if val > MAX_ATR:
            now = time.time()
            if not hasattr(self, "_last_atr_clamp_log") or now - self._last_atr_clamp_log > 60:
                logger.info(f"ATR clamped: {val:.2f} → {MAX_ATR:.2f} (news spike protection)")
                self._last_atr_clamp_log = now
        return min(val, MAX_ATR)

    def atr_percentile(self, period: int = 14, lookback: int = 200) -> float:
        """ATR percentile rank over rolling window (0.0–1.0).

        0.10 = 10th percentile (very low vol), 0.90 = 90th percentile (very high vol).
        """
        if len(self._df) < period + 1:
            return 0.5

        indicator = ta.volatility.AverageTrueRange(
            high=self._df["high"],
            low=self._df["low"],
            close=self._df["close"],
            window=period,
            fillna=False,
        )
        atr_series = indicator.average_true_range().dropna()
        if len(atr_series) < 2:
            return 0.5

        current = _safe_float(atr_series.iloc[-1], 0.5)
        history = atr_series.iloc[-lookback:-1].values
        if len(history) == 0:
            return 0.5

        return float(np.sum(current > history)) / len(history)

    def atr_median(self, period: int = 14, lookback: int = 100) -> float:
        """Rolling median ATR over lookback bars. Used for grinding detection."""
        if len(self._df) < period + 2:
            return self.atr(period)

        indicator = ta.volatility.AverageTrueRange(
            high=self._df["high"],
            low=self._df["low"],
            close=self._df["close"],
            window=period,
            fillna=False,
        )
        atr_series = indicator.average_true_range().dropna()
        if atr_series.empty:
            return self.atr(period)

        return float(atr_series.iloc[-lookback:].median())

    # ── RSI ───────────────────────────────────────────────────────────────

    def rsi(self, period: int = 14) -> float:
        """Relative Strength Index via ta.momentum.RSIIndicator (Wilder's smoothing).
        Fixes the degenerate 100.0/0.0 values from the old SMA-based hand-roll.
        """
        if len(self._df) < period + 1:
            return 50.0

        indicator = ta.momentum.RSIIndicator(
            close=self._df["close"],
            window=period,
            fillna=False,
        )
        return _safe_float(indicator.rsi().iloc[-1], default=50.0)

    # ── Volume ────────────────────────────────────────────────────────────

    def volume_ratio(self, period: int = 5) -> float:
        """Current volume / average volume over last N bars."""
        vols = self.get_volumes(period + 1)
        if len(vols) < 2:
            return 1.0
        avg = np.mean(vols[:-1])
        if avg <= 0:
            return 1.0
        return vols[-1] / avg

    # ── EMA (1-min, incremental cache) ───────────────────────────────────

    def ema(self, period: int, source: str = "close") -> float:
        """Exponential Moving Average — seeded via ta on first call, then incremental.

        The incremental update in add_bar() keeps this O(1) after the first bar.
        """
        if period in self._ema_cache:
            return self._ema_cache[period]

        if len(self._df) < period:
            return self.last_price if self._bars else 0.0

        indicator = ta.trend.EMAIndicator(
            close=self._df["close"],
            window=period,
            fillna=False,
        )
        val = _safe_float(indicator.ema_indicator().iloc[-1], default=self.last_price)

        # Seed incremental cache — add_bar() will update from here forward
        self._ema_cache[period] = val
        return val

    def ema_9(self) -> float:
        return self.ema(9)

    def ema_21(self) -> float:
        return self.ema(21)

    def ema_26(self) -> float:
        return self.ema(26)

    def ema_50(self) -> float:
        return self.ema(50)

    def ema_200(self) -> float:
        return self.ema(200)

    # ── EMA (multi-timeframe, via resample) ──────────────────────────────

    def mtf_ema(self, period: int, timeframe_minutes: int) -> float:
        """Multi-timeframe EMA — computed on properly resampled higher-TF candles.

        Uses pandas resample() to build correct OHLCV bars before computing EMA,
        fixing the old chunk-boundary alignment bug.
        """
        df_htf = self._resample_df(timeframe_minutes)
        if df_htf.empty or len(df_htf) < period:
            return self.last_price

        indicator = ta.trend.EMAIndicator(
            close=df_htf["close"],
            window=period,
            fillna=False,
        )
        return _safe_float(indicator.ema_indicator().iloc[-1], default=self.last_price)

    def ema_levels(self) -> dict:
        """All key EMA levels for dynamic S/R — 1-min and multi-timeframe."""
        return {
            "1m_9":   self.ema_9(),
            "1m_21":  self.ema_21(),
            "1m_26":  self.ema_26(),
            "1m_50":  self.ema_50(),
            "1m_200": self.ema_200(),
            "3m_9":   self.mtf_ema(9, 3),
            "3m_26":  self.mtf_ema(26, 3),
            "3m_50":  self.mtf_ema(50, 3),
            "5m_9":   self.mtf_ema(9, 5),
            "5m_26":  self.mtf_ema(26, 5),
            "5m_50":  self.mtf_ema(50, 5),
            "5m_200": self.mtf_ema(200, 5),
        }

    # ── Bollinger Bands ───────────────────────────────────────────────────

    def bollinger_bands(self, period: int = 20, std: float = 2.0) -> tuple:
        """Bollinger Bands → (upper, middle, lower) via ta.volatility.BollingerBands."""
        p = self.last_price
        if len(self._df) < period:
            return p, p, p

        indicator = ta.volatility.BollingerBands(
            close=self._df["close"],
            window=period,
            window_dev=std,
            fillna=False,
        )
        upper  = _safe_float(indicator.bollinger_hband().iloc[-1], p)
        middle = _safe_float(indicator.bollinger_mavg().iloc[-1], p)
        lower  = _safe_float(indicator.bollinger_lband().iloc[-1], p)
        return upper, middle, lower

    # ── ADX / Directional Indicators ─────────────────────────────────────

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Compute (ADX, DI+, DI-) from an OHLCV DataFrame via ta.trend.ADXIndicator."""
        if df.empty or len(df) < period * 2 + 1:
            return 0.0, 0.0, 0.0

        indicator = ta.trend.ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=period,
            fillna=False,
        )
        adx_val  = _safe_float(indicator.adx().iloc[-1],     0.0)
        di_plus  = _safe_float(indicator.adx_pos().iloc[-1], 0.0)
        di_minus = _safe_float(indicator.adx_neg().iloc[-1], 0.0)
        return adx_val, di_plus, di_minus

    def directional_indicators(self, period: int = 14) -> tuple:
        """Returns (ADX, DI+, DI-) on 1-min bars."""
        return self._compute_adx(self._df, period)

    def adx(self, period: int = 14) -> float:
        """Average Directional Index — trend strength (0–100).

        Side-effect: stores _di_plus_val / _di_minus_val for callers that
        need DI values without an extra indicator pass.
        """
        adx_val, di_plus, di_minus = self._compute_adx(self._df, period)
        self._di_plus_val  = di_plus
        self._di_minus_val = di_minus
        return adx_val

    def mtf_adx(self, period: int = 14, timeframe_minutes: int = 3) -> tuple:
        """Multi-timeframe ADX — computed on properly resampled higher-TF candles.

        Fixes the old broken chunk-boundary resampling that produced ADX readings
        of 35–67 when the real value was ~7.6.
        Returns (adx, di_plus, di_minus).
        """
        df_htf = self._resample_df(timeframe_minutes)
        return self._compute_adx(df_htf, period)

    # ── Regime detection ─────────────────────────────────────────────────

    def detect_regime(self) -> str:
        """Detect market regime: 'trending', 'ranging', 'volatile', or 'unknown'."""
        adx_val = self.adx()

        if len(self._bars) > 64:
            current_atr = self.atr(14)
            old_bars = self._bars[-64:-14]
            if len(old_bars) >= 14:
                trs = []
                for i in range(1, len(old_bars)):
                    h, l, pc = old_bars[i]["h"], old_bars[i]["l"], old_bars[i - 1]["c"]
                    trs.append(max(h - l, abs(h - pc), abs(l - pc)))
                avg_atr = np.mean(trs) if trs else current_atr
                if avg_atr > 0 and current_atr / avg_atr > 2.0:
                    return "volatile"

        if adx_val > 25:
            return "trending"
        if adx_val < 20:
            return "ranging"
        return "unknown"

    # ── Rolling range ─────────────────────────────────────────────────────

    def rolling_high(self, n: int = 20) -> float:
        """Highest high over last N bars."""
        if not self._bars:
            return 0.0
        return max(b["h"] for b in self._bars[-n:])

    def rolling_low(self, n: int = 20) -> float:
        """Lowest low over last N bars."""
        if not self._bars:
            return 0.0
        return min(b["l"] for b in self._bars[-n:])

    def daily_high(self) -> float:
        """Today's session high (all bars in rolling window)."""
        return float(max(self.get_highs())) if self._bars else 0.0

    def daily_low(self) -> float:
        """Today's session low."""
        return float(min(self.get_lows())) if self._bars else 0.0

    # ── Delta / order flow (custom approximation, no L2 data) ────────────

    def delta_estimate(self, n: int = 5) -> float:
        """Estimate buy/sell pressure from bar structure.
        Positive = buying pressure, negative = selling.
        """
        if len(self._bars) < n:
            return 0.0
        delta = 0.0
        for bar in self._bars[-n:]:
            rng = bar["h"] - bar["l"]
            vol = bar.get("v", 1)
            if rng > 0:
                buy_pct = (bar["c"] - bar["l"]) / rng
                delta += vol * (2 * buy_pct - 1)
        return delta

    # ── Opening range (custom, kept as-is) ───────────────────────────────

    def opening_range(self, minutes: int = 15) -> tuple:
        """Opening range (high, low) from first N bars of session.
        Returns (None, None) if not enough data.
        """
        if len(self._bars) < minutes:
            return None, None
        range_bars = self._bars[:minutes]
        return max(b["h"] for b in range_bars), min(b["l"] for b in range_bars)

    # ── Momentum slope (numpy polyfit, no ta equivalent) ─────────────────

    def momentum_slope(self, period: int = 12) -> float:
        """Linear regression slope of last N closes, expressed in basis points/bar."""
        closes = self.get_closes(period)
        if len(closes) < period:
            return 0.0
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        avg_price = np.mean(closes)
        if avg_price == 0:
            return 0.0
        return (slope / avg_price) * 10000

    # ── Market state ──────────────────────────────────────────────────────

    def build_market_state(self, session: str = "unknown",
                           minutes_since_open: int = 0) -> dict:
        """Build complete market state dict for strategies.

        Returns the same keys with the same semantics as the original implementation.
        All indicator computation happens here; callers receive a plain dict.
        """
        price        = self.last_price
        vwap_val     = self.vwap()
        vwap_std_val = self.vwap_std()
        atr_val      = self.atr()
        or_high, or_low = self.opening_range()
        bb           = self.bollinger_bands()

        # Compute ADX once; side-effect populates _di_plus_val/_di_minus_val
        adx_val    = self.adx()
        di_plus_14 = getattr(self, "_di_plus_val",  0.0)
        di_minus_14 = getattr(self, "_di_minus_val", 0.0)

        adx_3m_tuple = self.mtf_adx(14, 3)

        return {
            "timestamp":          self._bars[-1].get("t", 0) if self._bars else 0,
            "price":              price,
            "bid":                price - self.tick_size,
            "ask":                price + self.tick_size,
            "spread":             self.tick_size,
            "bar_open":           self._bars[-1]["o"]        if self._bars else 0,
            "bar_high":           self._bars[-1]["h"]        if self._bars else 0,
            "bar_low":            self._bars[-1]["l"]        if self._bars else 0,
            "bar_close":          price,
            "bar_volume":         self._bars[-1].get("v", 0) if self._bars else 0,
            "vwap":               vwap_val,
            "vwap_std":           vwap_std_val,
            "atr_14":             atr_val,
            "rsi_14":             self.rsi(14),
            "rsi_5":              self.rsi(5),
            "volume_ratio_5":     self.volume_ratio(),
            "delta":              self.delta_estimate(),
            "cumulative_delta":   self.delta_estimate(n=20),
            "session":            session,
            "minutes_since_open": minutes_since_open,
            "daily_high":         self.daily_high(),
            "daily_low":          self.daily_low(),
            "opening_range_high": or_high,
            "opening_range_low":  or_low,
            "ema_9":              self.ema_9(),
            "ema_21":             self.ema_21(),
            "ema_50":             self.ema_50(),
            "ema_200":            self.ema_200(),
            "ema_5m_9":           self.mtf_ema(9, 5),
            "ema_5m_26":          self.mtf_ema(26, 5),
            "ema_5m_50":          self.mtf_ema(50, 5),
            "adx_14":             adx_val,
            "di_plus_14":         di_plus_14,
            "di_minus_14":        di_minus_14,
            "adx_3m":             adx_3m_tuple[0],
            "di_plus_3m":         adx_3m_tuple[1],
            "di_minus_3m":        adx_3m_tuple[2],
            "bb_upper":           bb[0],
            "bb_middle":          bb[1],
            "bb_lower":           bb[2],
            "regime":             self.detect_regime(),
            "ema_levels":         self.ema_levels(),
            "slope_12":           self.momentum_slope(12),
            "slope_75":           self.momentum_slope(75),
            "atr_median":         self.atr_median(),
            "high_20":            self.rolling_high(20),
            "low_20":             self.rolling_low(20),
        }
