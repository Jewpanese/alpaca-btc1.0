"""
ShadowScorer — runs the EntryScorer in parallel with the live bot.

Purpose
-------
Shadow mode is a READ-ONLY observer. It computes canonical research-pipeline
features from the live bar stream, calls `core.entry_scorer.score_entry`, and
writes a JSONL log per session. It takes NO trading action.

Goals this enables
------------------
  1. Feature parity validation — canonical features vs bot-live features on
     the same bar, so any divergence (VWAP reset time, ATR period, etc.) is
     caught BEFORE weights are trusted.
  2. Threshold recalibration — after 10-20 sessions, live direction_pred
     quantiles can be recomputed and compared against the research sample.
  3. Signal-agreement audit — every bar's scorer signal vs what the bot
     actually did is on the same line, timestamped, for post-session diff.
  4. Hypothetical P&L scaffold — the log contains enough state to simulate
     what acting on the scorer would have returned.

Design choices
--------------
  - NO shared state with the bot. Shadow maintains its own feature buffers.
  - Feature definitions match research/feature_matrix.py EXACTLY (intentional
    duplication — this is the canonical source for the weights).
  - One JSONL file per session: logs/shadow_YYYYMMDD.jsonl
  - Safe on startup: if buffers aren't full yet, emits records with
    feature_status="warmup" and skips scoring.
"""
from __future__ import annotations
import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

from core.entry_scorer import EntrySignal, ScorerInput, score_entry

logger = logging.getLogger(__name__)


# ─── Canonical feature parameters (MUST match research/feature_matrix.py) ──
_ATR_PERIOD     = 14
_ER_WINDOW      = 20
_HURST_WINDOW   = 64
_EMA9_ALPHA     = 2.0 / (9 + 1)
_WARMUP_BARS    = 80   # matches the iloc[80:] slice in feature_matrix.py

# Stamped into every shadow log record. Bump when feature math or weights
# change so a log reader can tell which scorer revision produced the row.
SHADOW_VERSION  = "scorer_v1_20260421"


@dataclass
class ShadowState:
    """Rolling buffers + session accumulators. Kept small for cheap updates."""
    closes:      Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    highs:       Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    lows:        Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    trs:         Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    atr:         Optional[float] = None
    ema9:        Optional[float] = None
    prev_close:  Optional[float] = None
    # Session VWAP accumulators
    session_date: Optional[str] = None
    cum_pv:       float = 0.0
    cum_v:        float = 0.0
    vwap:         float = 0.0
    n_bars_seen:  int = 0


class ShadowScorer:
    """
    One instance per bot. Call `on_bar(bar, bot_context)` once per closed bar.
    `bot_context` is a plain dict with whatever the bot wants to record for
    side-by-side comparison (regime, current position, what action the legacy
    code took this bar, etc.). ShadowScorer does not interpret it.
    """

    def __init__(self, log_dir: Path, symbol: str = "BTC"):
        self.symbol = symbol
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state = ShadowState()
        self._log_fp = None
        self._log_date = None

    # ── Public API ─────────────────────────────────────────────────────

    def on_bar(self, bar: dict, bot_context: Optional[dict] = None) -> None:
        """
        bar: {
            'datetime': ISO8601 str or datetime,
            'open':  float, 'high': float, 'low': float, 'close': float,
            'volume': float,
        }
        bot_context: free-form dict, written through to the log for later diff.
        """
        try:
            self._update_features(bar)
            record = self._build_record(bar, bot_context)
            self._write(record)
        except Exception as e:
            logger.exception(f"[SHADOW] on_bar failed: {e}")

    def close(self) -> None:
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None

    # ── Feature computation (canonical, research-matching) ─────────────

    def _update_features(self, bar: dict) -> None:
        s = self.state
        o = float(bar['open']); h = float(bar['high'])
        lo = float(bar['low']); c = float(bar['close'])
        v  = float(bar.get('volume', 0.0))

        # Session VWAP reset — UTC date of the bar's timestamp
        ts = bar['datetime']
        if isinstance(ts, str):
            ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            ts_dt = ts
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        date_key = ts_dt.astimezone(timezone.utc).date().isoformat()
        if s.session_date != date_key:
            s.session_date = date_key
            s.cum_pv = 0.0
            s.cum_v  = 0.0

        typical = (h + lo + c) / 3.0
        s.cum_pv += typical * v
        s.cum_v  += v
        s.vwap    = s.cum_pv / s.cum_v if s.cum_v > 0 else typical

        # True range (research definition: max of h-l, |h-prev_close|, |l-prev_close|)
        if s.prev_close is None:
            tr = h - lo
        else:
            tr = max(h - lo, abs(h - s.prev_close), abs(lo - s.prev_close))
        s.trs.append(tr)

        # Wilder ATR — needs `period` TRs to seed
        if s.atr is None:
            if len(s.trs) >= _ATR_PERIOD:
                seed = sum(list(s.trs)[-_ATR_PERIOD:]) / _ATR_PERIOD
                s.atr = seed
        else:
            s.atr = (s.atr * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD

        # EMA9 — standard alpha-weighted EMA, seeded with first close
        if s.ema9 is None:
            s.ema9 = c
        else:
            s.ema9 = _EMA9_ALPHA * c + (1 - _EMA9_ALPHA) * s.ema9

        s.closes.append(c)
        s.highs.append(h)
        s.lows.append(lo)
        s.prev_close = c
        s.n_bars_seen += 1

    def _compute_er(self) -> Optional[float]:
        s = self.state
        if len(s.closes) < _ER_WINDOW + 1:
            return None
        closes = list(s.closes)[-(_ER_WINDOW + 1):]
        net = abs(closes[-1] - closes[0])
        gross = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        return (net / gross) if gross > 0 else 0.0

    def _compute_hurst(self) -> Optional[float]:
        s = self.state
        if len(s.closes) < _HURST_WINDOW:
            return None
        closes = list(s.closes)[-_HURST_WINDOW:]
        rets = [math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i - 1] > 0 and closes[i] > 0]
        if len(rets) < 8:
            return 0.5
        mean_r = sum(rets) / len(rets)
        dev = [r - mean_r for r in rets]
        cum = []
        acc = 0.0
        for d in dev:
            acc += d
            cum.append(acc)
        R = max(cum) - min(cum)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        S = math.sqrt(var)
        if S <= 0 or R <= 0:
            return 0.5
        h = math.log(R / S) / math.log(len(rets))
        return max(0.0, min(1.0, h))

    def _compute_body_ratio(self, bar: dict) -> float:
        o = float(bar['open']); h = float(bar['high'])
        lo = float(bar['low']); c = float(bar['close'])
        rng = max(h - lo, 1e-9)
        return abs(c - o) / rng

    # ── Record assembly ───────────────────────────────────────────────

    def _build_record(self, bar: dict, bot_context: Optional[dict]) -> dict:
        s = self.state

        er     = self._compute_er()
        hurst  = self._compute_hurst()
        body   = self._compute_body_ratio(bar)
        c      = float(bar['close'])
        atr    = s.atr
        ema9   = s.ema9
        vwap   = s.vwap

        warmup = (
            s.n_bars_seen < _WARMUP_BARS
            or er is None or hurst is None or atr is None or atr <= 0
        )

        features_canonical = {
            "er_20":          er,
            "hurst_64":       hurst,
            "body_ratio":     body,
            "atr":            atr,
            "ema9":           ema9,
            "vwap":           vwap,
            "dist_vwap_atr":  (c - vwap) / atr if (atr and atr > 0) else None,
            "dist_ema9_atr":  (c - ema9) / atr if (atr and atr > 0 and ema9) else None,
        }

        if warmup:
            return {
                "ts": bar['datetime'] if isinstance(bar['datetime'], str)
                      else bar['datetime'].isoformat(),
                "symbol": self.symbol,
                "shadow_version": SHADOW_VERSION,
                "bar": {k: float(bar[k]) for k in ("open", "high", "low", "close", "volume")
                        if k in bar},
                "feature_status": "warmup",
                "n_bars_seen": s.n_bars_seen,
                "features_canonical": features_canonical,
                "score": None,
                "bot_context": bot_context or {},
            }

        scorer_input = ScorerInput(
            er_20=features_canonical["er_20"],
            dist_vwap_atr=features_canonical["dist_vwap_atr"],
            dist_ema9_atr=features_canonical["dist_ema9_atr"],
            hurst_64=features_canonical["hurst_64"],
            body_ratio=features_canonical["body_ratio"],
            price=c, vwap=vwap, ema9=ema9, atr=atr,
        )
        out = score_entry(scorer_input)

        return {
            "ts": bar['datetime'] if isinstance(bar['datetime'], str)
                  else bar['datetime'].isoformat(),
            "symbol": self.symbol,
            "shadow_version": SHADOW_VERSION,
            "bar": {k: float(bar[k]) for k in ("open", "high", "low", "close", "volume")
                    if k in bar},
            "feature_status": "ok",
            "n_bars_seen": s.n_bars_seen,
            "features_canonical": features_canonical,
            "score": {
                "signal":          out.signal.value,
                "direction_pred":  out.direction_pred,
                "size_pred":       out.size_pred,
                "size_multiplier": out.size_multiplier,
                "reason":          out.reason,
            },
            "bot_context": bot_context or {},
        }

    # ── Logging ───────────────────────────────────────────────────────

    def _ensure_log_open(self, bar_date: str) -> None:
        if self._log_fp is None or self._log_date != bar_date:
            if self._log_fp is not None:
                self._log_fp.close()
            # YYYYMMDD from ISO YYYY-MM-DD
            date_compact = bar_date.replace("-", "")
            path = self.log_dir / f"shadow_{self.symbol}_{date_compact}.jsonl"
            self._log_fp = open(path, "a", encoding="utf-8")
            self._log_date = bar_date
            logger.info(f"[SHADOW] logging to {path}")

    def _write(self, record: dict) -> None:
        # Pull date for file rotation
        ts = record["ts"]
        try:
            bar_date = ts.split("T")[0] if "T" in ts else ts[:10]
        except Exception:
            bar_date = datetime.now(timezone.utc).date().isoformat()
        self._ensure_log_open(bar_date)
        self._log_fp.write(json.dumps(record, default=str) + "\n")
        self._log_fp.flush()
