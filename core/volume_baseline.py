"""
Volume Baseline & Z-Score — per-minute-of-day volume statistics.

What it does
------------
For every minute of the trading day (NY time) we maintain a baseline
mean and stddev of bar volume, computed from N+ trading days of historical
1-min data. At runtime, each new bar's volume is converted to a z-score:

    vol_z = (this_bar_volume - mean[minute_of_day]) / std[minute_of_day]

Interpretation:
    vol_z = 0    → typical volume for this time of day
    vol_z = +1   → ~1 stddev above typical (elevated)
    vol_z = +2   → significantly heavy
    vol_z = -1   → ~1 stddev below typical (quiet)

Why per-minute-of-day
---------------------
Volume is heavily time-of-day dependent. NY 9:30am open averages ~6x more
volume than the lunch period, and ~2x more than NY close. A flat 20-bar
rolling mean (topstep3's approach) compares high-volume morning bars to
low-volume midday bars and produces meaningless ratios. Per-minute-of-day
buckets fix that by comparing this 9:32 bar to the 9:32 baseline.

File format
-----------
JSON file at `data/volume_baseline.json`:

    {
        "version": 1,
        "instrument": "ES",
        "timeframe_seconds": 60,
        "tz": "America/New_York",
        "n_days_used": 1240,
        "built_at": "2026-05-03T...",
        "buckets": {
            "0930": {"mean": 5234.2, "std": 1840.5, "n": 1240},
            "0931": {"mean": 4920.1, "std": 1722.3, "n": 1240},
            ...
        }
    }

Bucket key is "HHMM" in NY local time (no DST adjustment — buckets follow
NY *clock* time, which is what session schedules anchor to).
"""
from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _NY_TZ = None

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_PATH = Path("data") / "volume_baseline.json"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _to_ny_minute_bucket(ts) -> str:
    """Convert a timestamp (datetime or ISO string) to a 'HHMM' NY-time key.

    Accepts:
      - datetime (naive → assumed UTC; aware → converted to NY)
      - ISO 8601 string (with or without tz)
      - epoch seconds / millis (int or float)

    Returns 4-char string 'HHMM' anchored to America/New_York clock time.
    """
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        # Heuristic: > 10^12 = ms, > 10^10 = seconds with tail, else seconds
        if ts > 1e12:
            ts = ts / 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, str):
        # Try ISO first, then NinjaTrader-style "YYYYMMDD HHMMSS"
        try:
            # Handle common Z-suffix
            iso = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
        except ValueError:
            dt = datetime.strptime(ts, "%Y%m%d %H%M%S")
            # Treat parsed datetime as UTC if no tz — caller should pass tz-aware
            # for live bars; this fallback is for the historical .txt format.
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        raise ValueError(f"Unsupported timestamp type: {type(ts)}")

    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if _NY_TZ is None:  # pragma: no cover — fallback to UTC bucket if zoneinfo missing
        ny = dt.astimezone(timezone.utc)
    else:
        ny = dt.astimezone(_NY_TZ)
    return f"{ny.hour:02d}{ny.minute:02d}"


# ──────────────────────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _BucketStats:
    mean: float
    std: float
    n: int


class VolumeBaseline:
    """Per-minute-of-day volume baseline for z-score computation.

    Two phases:
      1. BUILD — call build_from_bars() with a long history of bar volume
         and timestamps. Saves the baseline JSON to disk.
      2. USE — load the baseline at bot startup; call z_score(ts, volume)
         per live bar to get the normalized score.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_BASELINE_PATH
        self._buckets: dict[str, _BucketStats] = {}
        self._meta: dict = {}
        self._loaded = False

    # ── BUILD ────────────────────────────────────────────────────────────

    def build_from_bars(self,
                        bars: list[tuple],
                        instrument: str = "ES",
                        timeframe_seconds: int = 60,
                        save: bool = True) -> dict:
        """Build the baseline from a list of (timestamp, volume) tuples.

        Streaming aggregation (Welford-style) so we don't have to hold all
        samples per bucket in memory. Each bucket maintains running n / mean
        / M2 (sum of squared deviations); final std = sqrt(M2 / (n-1)).

        Returns a summary dict.
        """
        running: dict[str, list] = {}  # key -> [n, mean, M2]
        skipped = 0

        for ts, vol in bars:
            try:
                vol_f = float(vol)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if vol_f <= 0:
                skipped += 1
                continue
            try:
                key = _to_ny_minute_bucket(ts)
            except Exception:
                skipped += 1
                continue

            slot = running.setdefault(key, [0, 0.0, 0.0])
            slot[0] += 1
            n = slot[0]
            delta = vol_f - slot[1]
            slot[1] += delta / n            # new mean
            delta2 = vol_f - slot[1]
            slot[2] += delta * delta2       # M2

        # Finalize
        self._buckets = {}
        for key, (n, mean, m2) in running.items():
            if n < 2:
                # Need at least 2 samples for stddev; skip otherwise
                continue
            std = math.sqrt(m2 / (n - 1))
            self._buckets[key] = _BucketStats(mean=mean, std=std, n=n)

        # Estimate trading-day count: median bucket's n
        ns = sorted(b.n for b in self._buckets.values())
        n_days = ns[len(ns) // 2] if ns else 0

        self._meta = {
            "version": 1,
            "instrument": instrument,
            "timeframe_seconds": timeframe_seconds,
            "tz": "America/New_York",
            "n_days_used": n_days,
            "n_buckets": len(self._buckets),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "skipped_rows": skipped,
        }
        self._loaded = True

        if save:
            self.save()

        return {
            **self._meta,
            "sample_buckets": self._sample_for_log(),
        }

    def _sample_for_log(self) -> dict:
        """A few well-known time buckets for sanity-check logging."""
        return {
            k: {"mean": round(v.mean, 1), "std": round(v.std, 1), "n": v.n}
            for k, v in self._buckets.items()
            if k in {"0930", "1000", "1200", "1500", "1600", "1700", "0000"}
        }

    # ── PERSISTENCE ─────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self._meta,
            "buckets": {
                k: {"mean": v.mean, "std": v.std, "n": v.n}
                for k, v in self._buckets.items()
            },
        }
        target.write_text(json.dumps(payload, indent=2))
        logger.info(
            f"[VOL BASELINE] saved {len(self._buckets)} buckets "
            f"({payload.get('n_days_used', '?')} days) → {target}"
        )
        return target

    def load(self, path: Optional[Path] = None) -> bool:
        target = Path(path) if path else self.path
        if not target.exists():
            logger.info(f"[VOL BASELINE] no baseline at {target} — z_score will return None")
            return False
        try:
            payload = json.loads(target.read_text())
        except Exception as e:
            logger.warning(f"[VOL BASELINE] failed to read {target}: {e}")
            return False

        self._meta = {k: v for k, v in payload.items() if k != "buckets"}
        self._buckets = {
            k: _BucketStats(mean=v["mean"], std=v["std"], n=v["n"])
            for k, v in payload.get("buckets", {}).items()
        }
        self._loaded = True
        logger.info(
            f"[VOL BASELINE] loaded {len(self._buckets)} buckets, "
            f"n_days_used={self._meta.get('n_days_used', '?')}, "
            f"built_at={self._meta.get('built_at', '?')}"
        )
        return True

    # ── USE ──────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._loaded and len(self._buckets) > 0

    def z_score(self, ts, volume: Union[int, float]) -> Optional[float]:
        """Return the z-score of `volume` against the baseline for the
        minute-of-day bucket of `ts`, or None if no baseline is available
        for that bucket (or the baseline isn't loaded yet)."""
        if not self.is_ready():
            return None
        try:
            vol = float(volume)
        except (TypeError, ValueError):
            return None
        if vol <= 0:
            return None

        try:
            key = _to_ny_minute_bucket(ts)
        except Exception:
            return None

        b = self._buckets.get(key)
        if b is None or b.std <= 0:
            # No baseline for this minute — try ±1 minute as a graceful fallback
            for offset in (1, -1):
                hh = int(key[:2])
                mm = int(key[2:])
                total = hh * 60 + mm + offset
                total %= 1440
                fb_key = f"{total // 60:02d}{total % 60:02d}"
                b = self._buckets.get(fb_key)
                if b and b.std > 0:
                    break
            else:
                return None

        return (vol - b.mean) / b.std

    def bucket_stats(self, ts) -> Optional[_BucketStats]:
        """Return raw bucket stats for diagnostic / inspection use."""
        if not self.is_ready():
            return None
        try:
            key = _to_ny_minute_bucket(ts)
        except Exception:
            return None
        return self._buckets.get(key)

    @property
    def meta(self) -> dict:
        return dict(self._meta)
