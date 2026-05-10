"""
Market Data Manager — Handles bar aggregation from ticks and session detection.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


def detect_session(dt_utc: datetime) -> str:
    """Determine trading session from UTC timestamp."""
    et = dt_utc.astimezone(ET)
    hour, minute = et.hour, et.minute
    t = hour * 60 + minute
    
    # RTH: 9:30 AM - 4:00 PM ET
    if 570 <= t < 600:       # 9:30 - 10:00 → NY_OPEN
        return "NY_OPEN"
    elif 600 <= t < 840:     # 10:00 - 2:00 → US_MIDDAY
        return "US_MIDDAY"
    elif 840 <= t < 960:     # 2:00 - 4:00 → US_CLOSE
        return "US_CLOSE"
    elif 180 <= t < 570:     # 3:00 AM - 9:30 AM → LONDON
        return "LONDON"
    else:
        return "OVERNIGHT"


def minutes_since_rth_open(dt_utc: datetime) -> int:
    """Minutes since 9:30 AM ET."""
    et = dt_utc.astimezone(ET)
    rth_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    diff = (et - rth_open).total_seconds() / 60
    return max(0, int(diff))


class TickAggregator:
    """Aggregates SignalR ticks into N-minute bars (default 3-min)."""

    def __init__(self, bar_minutes: int = 3):
        self._bar_minutes = bar_minutes
        self._current_bar = None
        self._current_period = None

    def on_tick(self, price: float, volume: int = 1,
                timestamp: datetime = None) -> Optional[dict]:
        """
        Process a tick. Returns completed bar dict when the N-minute period rolls over.
        Bar period key is floor-divided so bars align to clock boundaries
        (e.g. 3-min: 00, 03, 06, 09, ...).
        """
        ts = timestamp or datetime.now(timezone.utc)
        # Align to bar_minutes boundaries: floor(minute / bar_minutes)
        period_key = ts.strftime("%Y%m%d %H") + f"{(ts.minute // self._bar_minutes):02d}"

        if period_key != self._current_period:
            completed = self._current_bar
            self._current_bar = {
                "t": ts.isoformat(),
                "o": price,
                "h": price,
                "l": price,
                "c": price,
                "v": volume
            }
            self._current_period = period_key
            return completed  # None on first tick

        # Update current bar
        self._current_bar["h"] = max(self._current_bar["h"], price)
        self._current_bar["l"] = min(self._current_bar["l"], price)
        self._current_bar["c"] = price
        self._current_bar["v"] = self._current_bar.get("v", 0) + volume

        return None
