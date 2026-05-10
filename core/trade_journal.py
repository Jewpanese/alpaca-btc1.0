"""
Trade Journal — append-only JSONL log of closed trades for the EOD Claude analyzer.

The bot's RiskManager keeps trades in memory; this module persists them to disk so
they survive restarts and can be aggregated daily by `analyzers/eod_summary.py`.

Layout:
    data/journal/btc_trades_YYYY-MM-DD.jsonl   (one JSON object per line)
    data/journal/btc_summary_YYYY-MM-DD.json   (latest aggregate snapshot)

Each line in the JSONL is one closed trade record:
    {
        "ts":           "2026-05-09T20:43:11Z",  # exit time (UTC)
        "entry_ts":     "2026-05-09T20:25:04Z",
        "direction":    "LONG",                  # always LONG for spot BTC
        "contracts":    10,                      # native units (1 = 0.01 BTC)
        "btc_size":     0.10,
        "entry_price":  102341.50,
        "exit_price":   102998.20,
        "pnl_pts":      656.70,                  # USD price move
        "pnl_dollars":  65.67,                   # = pnl_pts * point_value * contracts
        "exit_reason":  "regime_change",
        "regime":       "TREND",                 # snapshot at exit
        "session":      "US_MIDDAY",
        "strategy":     "big_money_direct",
        "duration_sec": 1087,
        "mfe_pts":      720.0,
        "mae_pts":     -120.0,
    }
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "journal"


@dataclass
class TradeJournalRecord:
    ts: str
    entry_ts: str
    direction: str = "LONG"
    contracts: int = 0
    btc_size: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pts: float = 0.0
    pnl_dollars: float = 0.0
    exit_reason: str = ""
    regime: str = ""
    session: str = ""
    strategy: str = ""
    duration_sec: int = 0
    mfe_pts: float = 0.0
    mae_pts: float = 0.0
    extras: dict = field(default_factory=dict)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _journal_path(date_str: Optional[str] = None) -> Path:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    return JOURNAL_DIR / f"btc_trades_{date_str or _today_utc()}.jsonl"


def append_trade(record: TradeJournalRecord | dict[str, Any]) -> None:
    """Append a closed-trade record. Creates the file if needed (one per UTC day)."""
    if isinstance(record, TradeJournalRecord):
        payload = asdict(record)
    else:
        payload = dict(record)

    path = _journal_path()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.error(f"trade_journal append failed: {e}")


def read_trades(date_str: Optional[str] = None) -> list[dict[str, Any]]:
    """Read all trade records for the given UTC day (defaults to today)."""
    path = _journal_path(date_str)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"trade_journal: skipping malformed line: {e}")
    return out


def aggregate(date_str: Optional[str] = None) -> dict[str, Any]:
    """Compute daily summary stats for the EOD analyzer prompt."""
    trades = read_trades(date_str)
    if not trades:
        return {
            "date": date_str or _today_utc(),
            "trade_count": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "gross_pnl_dollars": 0.0,
            "avg_winner_dollars": 0.0,
            "avg_loser_dollars": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "total_btc_traded": 0.0,
            "trades": [],
        }

    winners = [t for t in trades if t.get("pnl_dollars", 0) > 0]
    losers = [t for t in trades if t.get("pnl_dollars", 0) <= 0]
    gross = sum(t.get("pnl_dollars", 0.0) for t in trades)
    avg_w = (sum(t["pnl_dollars"] for t in winners) / len(winners)) if winners else 0.0
    avg_l = (sum(t["pnl_dollars"] for t in losers) / len(losers)) if losers else 0.0
    best = max(trades, key=lambda t: t.get("pnl_dollars", 0)) if trades else None
    worst = min(trades, key=lambda t: t.get("pnl_dollars", 0)) if trades else None
    total_btc = sum(t.get("btc_size", 0.0) for t in trades)

    by_regime: dict[str, dict[str, Any]] = {}
    for t in trades:
        r = t.get("regime") or "UNKNOWN"
        slot = by_regime.setdefault(r, {"count": 0, "pnl_dollars": 0.0, "winners": 0})
        slot["count"] += 1
        slot["pnl_dollars"] += t.get("pnl_dollars", 0.0)
        if t.get("pnl_dollars", 0) > 0:
            slot["winners"] += 1

    by_session: dict[str, dict[str, Any]] = {}
    for t in trades:
        s = t.get("session") or "UNKNOWN"
        slot = by_session.setdefault(s, {"count": 0, "pnl_dollars": 0.0})
        slot["count"] += 1
        slot["pnl_dollars"] += t.get("pnl_dollars", 0.0)

    by_strategy: dict[str, dict[str, Any]] = {}
    for t in trades:
        s = t.get("strategy") or "UNKNOWN"
        slot = by_strategy.setdefault(s, {"count": 0, "pnl_dollars": 0.0, "winners": 0})
        slot["count"] += 1
        slot["pnl_dollars"] += t.get("pnl_dollars", 0.0)
        if t.get("pnl_dollars", 0) > 0:
            slot["winners"] += 1

    return {
        "date": date_str or _today_utc(),
        "trade_count": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
        "gross_pnl_dollars": gross,
        "avg_winner_dollars": avg_w,
        "avg_loser_dollars": avg_l,
        "best_trade": best,
        "worst_trade": worst,
        "total_btc_traded": total_btc,
        "by_regime": by_regime,
        "by_session": by_session,
        "by_strategy": by_strategy,
        "trades": trades,
    }
