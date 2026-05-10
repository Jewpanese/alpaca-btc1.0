"""Bar Recorder — Saves every 1-minute bar to CSV for replay/backtesting.

Usage:
    recorder = BarRecorder(data_dir="C:/Development/topstep5/data/live")
    recorder.record(bar)  # Call on every new candle

Output format: CSV with columns matching the backtest engine's expected format.
Files are named by date: MES_2026-03-11.csv
"""

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BarRecorder:
    """Records 1-minute bars to daily CSV files."""

    def __init__(self, data_dir: str = None, instrument: str = "BTC"):
        self.data_dir = Path(data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "live"
        ))
        self.instrument = instrument
        self._current_date: Optional[str] = None
        self._current_file = None
        self._writer = None
        self._bar_count = 0
        self._last_timestamp: Optional[str] = None

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📼 [BAR RECORDER] Initialized — saving to {self.data_dir}")

    def record(self, bar: dict) -> bool:
        """Record a single bar. Returns True if written, False if skipped (duplicate).

        Expected bar format from TopstepX:
            {"t": "2026-03-11T15:30:00Z", "o": 6765.25, "h": 6768.50, "l": 6764.00, "c": 6767.25, "v": 150}
        """
        try:
            # Extract timestamp
            ts = bar.get("t") or bar.get("time") or bar.get("timestamp")
            if not ts:
                logger.warning(f"📊 [BAR RECORDER] No timestamp in bar: {list(bar.keys())}")
                return False

            # Skip duplicates
            if ts == self._last_timestamp:
                return False
            self._last_timestamp = ts

            # Parse timestamp
            if isinstance(ts, str):
                try:
                    bar_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    bar_dt = datetime.now(timezone.utc)
            else:
                bar_dt = datetime.now(timezone.utc)

            # Check if we need a new file (new day)
            date_str = bar_dt.strftime("%Y-%m-%d")
            if date_str != self._current_date:
                self._rotate_file(date_str)

            # Extract OHLCV
            o = bar.get("o", bar.get("open", 0))
            h = bar.get("h", bar.get("high", 0))
            l = bar.get("l", bar.get("low", 0))
            c = bar.get("c", bar.get("close", 0))
            v = bar.get("v", bar.get("volume", 0))

            # Write row
            self._writer.writerow({
                "timestamp": bar_dt.strftime("%Y%m%d %H%M%S"),
                "datetime": bar_dt.isoformat(),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
            self._current_file.flush()  # Flush to disk immediately
            self._bar_count += 1

            if self._bar_count == 1:
                logger.info(f"📊 [BAR RECORDER] First bar written! ts={ts} close={c}")
            if self._bar_count % 100 == 0:
                logger.info(f"📼 [BAR RECORDER] {self._bar_count} bars saved to {self._current_date}")

            return True

        except Exception as e:
            logger.error(f"📼 [BAR RECORDER] Error recording bar: {e}")
            return False

    def _rotate_file(self, date_str: str):
        """Open a new CSV file for the given date."""
        self._close_file()
        self._current_date = date_str
        filepath = self.data_dir / f"{self.instrument}_{date_str}.csv"

        # Append mode — safe for restarts during the same day
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        self._current_file = open(filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._current_file,
            fieldnames=["timestamp", "datetime", "open", "high", "low", "close", "volume"],
        )
        if not file_exists:
            self._writer.writeheader()

        logger.info(f"📼 [BAR RECORDER] {'Appending to' if file_exists else 'Created'} {filepath}")

    def _close_file(self):
        """Close the current file if open."""
        if self._current_file:
            try:
                self._current_file.close()
            except Exception:
                pass
            self._current_file = None
            self._writer = None

    def close(self):
        """Clean shutdown."""
        self._close_file()
        logger.info(f"📼 [BAR RECORDER] Closed — {self._bar_count} total bars recorded")

    @property
    def total_bars(self) -> int:
        return self._bar_count

    def __del__(self):
        self._close_file()
