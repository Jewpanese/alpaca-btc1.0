"""
End-of-Day BTC Trading Analyzer — powered by Claude Opus 4.7.

Reads today's trade journal (`data/journal/btc_trades_YYYY-MM-DD.jsonl`), pulls
account state from the bot's MLL snapshot, and asks Claude Opus 4.7 to produce a
candid daily review with:

  * What worked / what didn't (regime, session, strategy attribution).
  * Whether the bot followed the discipline rules (risk caps, regime gating).
  * Concrete tweaks for tomorrow.

Design notes:

  * Model: `claude-opus-4-7` (no `temperature` / `top_p` / `top_k` — removed on 4.7).
  * Adaptive thinking on: `thinking={"type": "adaptive"}` so Claude decides how
    deep to reason. We surface the reasoning by setting `display: "summarized"`.
  * Streaming: large outputs ≥ 16K need streaming to avoid SDK HTTP timeouts.
  * Prompt caching: the role / discipline blocks rarely change between days, so
    we mark them with `cache_control={"type": "ephemeral"}` to get ~10x cost
    reduction on repeated runs (5-min TTL by default).
  * Output is written to `analyzers/reports/btc_daily_<date>.md` and `.json`.

Usage:

    python run_eod_summary.py                # today (UTC)
    python run_eod_summary.py 2026-05-09     # specific date

Programmatic:

    from analyzers.eod_summary import run_eod_summary
    report_md, report_path = run_eod_summary()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core import trade_journal

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 16_000  # plenty of headroom; we stream so timeouts aren't a concern


# ---------------------------------------------------------------------------
# Prompt blocks — keep stable parts FIRST so prompt caching gets a clean prefix.
# ---------------------------------------------------------------------------

_ROLE_PROMPT = (
    "You are a senior quant crypto trader and a frank performance coach. The user "
    "runs an automated BTC/USD spot-trading bot on Alpaca paper. Each day they "
    "send you the trade journal plus the bot's current account state, and you "
    "produce a tight, opinionated review that compounds — your job is to make the "
    "bot smarter every day.\n\n"
    "Style:\n"
    " - Direct, no hedging, no filler.\n"
    " - Quantify where you can. Cite specific trades by entry/exit and PnL.\n"
    " - Distinguish discipline failures (rule violations) from quality failures "
    "(rule was followed, the rule itself is suspect).\n"
    " - Tomorrow's-plan items must be concrete and shippable as parameter tweaks "
    "or rule changes — no vague advice.\n"
)

_DISCIPLINE_RULES = (
    "BOT DISCIPLINE & ARCHITECTURE\n"
    "============================\n"
    "Account: $100,000 paper, longs-only spot BTC on Alpaca.\n"
    "Hard stops:\n"
    "  - Max daily loss HARD: $5,000 (5% of account) — bot must halt on this.\n"
    "  - Max daily loss SOFT: $2,500 (2.5%) — bot stops new entries, lets exits run.\n"
    "  - Max risk per trade: $500.\n"
    "  - Daily PROFIT target: $100 — once cumulative PnL ≥ $100 we expect the bot\n"
    "    to lock in and stop seeking (configurable).\n"
    "Position sizing: 1 native contract = 0.01 BTC. point_value = $0.01/pt where\n"
    "  1 'point' = $1 of BTC price move. PnL_dollars = price_move × 0.01 × contracts.\n"
    "Cushion-tier sizing (BTC): cushion ≥ $20k → 50 ct, $15k → 40, $10k → 30,\n"
    "  $5k → 20, $2.5k → 10, $1.5k → 5 ct (survival), < $1.5k → no trades.\n"
    "Tranche exits: 60% Risk-Reduce / 40% Core / 30% Runner (the bot may scale in\n"
    "  T2/T3 above 60% probe under confirmation rules).\n"
    "Regime gating: TREND / GRIND / RANGE / CHOP — only TREND fires direct big-size\n"
    "  entries; CHOP is HARD SKIP.\n"
    "Session-end gates: DISABLED (BTC is 24/7).\n"
    "ATR floor: only trade when ATR ≥ $50 unless ADX ≥ 30.\n"
    "Loser Intelligence Layer (LIL): pre-emptive exits on stale losers,\n"
    "  velocity bleedout, giveback after peak.\n"
    "Cooldowns: 180s after a full-loss exit (same direction blocked); 180s after\n"
    "  2 consecutive losses regardless of direction.\n"
    "Strategies layered on top of the BigMoney trend logic: VWAP_REVERT,\n"
    "  MOMENTUM, BB_BOUNCE, SR_BREAKOUT, DELTA_DIV, EMA_REJECT, TREND_FOLLOW.\n"
    "Architecture: HybridBot → AlphaBot → BigMoneyBot (ported from topstep5\n"
    "  ES/MES bot, broker layer swapped for Alpaca crypto).\n"
)

_REPORT_TEMPLATE = (
    "Produce a Markdown report with these sections (in order):\n\n"
    "## P&L Snapshot\n"
    "One paragraph: trades, win rate, gross PnL, vs daily target, vs daily loss\n"
    "limits, account-balance change.\n\n"
    "## Discipline Audit\n"
    "Did the bot follow its rules? Any breaches: oversize positions, trades during\n"
    "CHOP regime, entries below ATR floor, missed cooldowns, profit-target overshoot,\n"
    "loss-cap proximity. Specifically call out anything that looked like rule drift.\n\n"
    "## What Worked\n"
    "Best trades and why — by regime, session, and strategy. Pattern recognition\n"
    "across the day. If multiple strategies fired, which was carrying the day.\n\n"
    "## What Didn't\n"
    "Worst trades and the failure mode (chase, fakeout, late stop, bad regime).\n"
    "Distinguish discipline failures from quality failures.\n\n"
    "## Market Context\n"
    "From the bar/regime data provided: range-vs-trend day, volume profile, any\n"
    "structural levels that were respected or broken. Brief — one paragraph.\n\n"
    "## Tomorrow's Plan\n"
    "3–5 SPECIFIC items. Each must be one of: (a) a parameter tweak with the\n"
    "current value, proposed value, and rationale; (b) a rule change with the\n"
    "current rule and the proposed rule; (c) a watch-flag for a specific market\n"
    "condition with the action to take. NO vague items like 'be more careful'.\n\n"
    "## One-Line Summary\n"
    "A single sentence that captures the day. This is what the user reads first\n"
    "tomorrow morning.\n"
)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class EODData:
    date_str: str
    today: dict[str, Any]
    account_state: dict[str, Any]
    market_context: dict[str, Any]


def _load_account_state() -> dict[str, Any]:
    """Pull account snapshot from the bot's persisted MLL state."""
    state_file = (
        Path(__file__).resolve().parent.parent / "data" / "mll_state.json"
    )
    if not state_file.exists():
        return {"note": "mll_state.json not found — bot may not have run yet"}
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def _load_market_context(date_str: str) -> dict[str, Any]:
    """
    Quick market summary from the BarRecorder CSV (if present).
    Computes high/low/close, range, simple volatility — enough for the LLM
    to ground its analysis without bloating the prompt.
    """
    bar_dir = Path(__file__).resolve().parent.parent / "data" / "live"
    candidates = list(bar_dir.glob(f"*{date_str}*.csv"))
    if not candidates:
        return {"note": f"no bar recorder file for {date_str}"}
    path = candidates[0]
    try:
        import csv

        opens, highs, lows, closes, vols = [], [], [], [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    opens.append(float(row.get("o") or row.get("open") or 0))
                    highs.append(float(row.get("h") or row.get("high") or 0))
                    lows.append(float(row.get("l") or row.get("low") or 0))
                    closes.append(float(row.get("c") or row.get("close") or 0))
                    vols.append(float(row.get("v") or row.get("volume") or 0))
                except (TypeError, ValueError):
                    continue
        if not closes:
            return {"note": "bar file empty"}
        day_high = max(highs)
        day_low = min(lows)
        return {
            "bar_count": len(closes),
            "first_open": opens[0],
            "last_close": closes[-1],
            "day_high": day_high,
            "day_low": day_low,
            "range_dollars": day_high - day_low,
            "range_pct": (day_high - day_low) / opens[0] * 100 if opens[0] else 0,
            "total_volume": sum(vols),
            "net_change_pct": (closes[-1] - opens[0]) / opens[0] * 100 if opens[0] else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def _gather(date_str: Optional[str] = None) -> EODData:
    today = trade_journal.aggregate(date_str)
    return EODData(
        date_str=today["date"],
        today=today,
        account_state=_load_account_state(),
        market_context=_load_market_context(today["date"]),
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _build_messages(data: EODData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (system_blocks, message_list).

    System blocks are ordered stable→volatile so the cache prefix bites:
      1. role prompt (cached)
      2. discipline rules (cached)
      3. report template (cached)
    The actual day's data goes in the user message — varies per call, never cached.
    """
    system_blocks = [
        {"type": "text", "text": _ROLE_PROMPT},
        {"type": "text", "text": _DISCIPLINE_RULES},
        # Mark the LAST stable block — that single breakpoint caches everything
        # rendered before it (this block + the two above) together.
        {
            "type": "text",
            "text": _REPORT_TEMPLATE,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    user_payload = (
        f"# Daily Data — {data.date_str} (UTC)\n\n"
        f"## Account State\n```json\n{json.dumps(data.account_state, indent=2)}\n```\n\n"
        f"## Today's Aggregate\n```json\n{json.dumps({k: v for k, v in data.today.items() if k != 'trades'}, indent=2, default=str)}\n```\n\n"
        f"## Trade-by-trade\n```json\n{json.dumps(data.today.get('trades', []), indent=2, default=str)}\n```\n\n"
        f"## Market Context (BTC/USD bars for the day)\n```json\n{json.dumps(data.market_context, indent=2)}\n```\n\n"
        "Now produce the report in the exact section order specified."
    )

    messages = [{"role": "user", "content": user_payload}]
    return system_blocks, messages


def _call_claude(system_blocks: list[dict[str, Any]], messages: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """Stream a response from Claude Opus 4.7. Returns (text, usage_dict)."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic SDK is not installed — run `pip install anthropic` first"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is not set — add it to .env "
            "(or your shell) and rerun."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # We stream because:
    #  - max_tokens is large; non-streaming risks SDK HTTP timeouts.
    #  - lets us see Claude's reasoning surface as it goes (useful in dev).
    text_chunks: list[str] = []
    with client.messages.stream(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "high"},
        system=system_blocks,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            text_chunks.append(text)
        final = stream.get_final_message()

    full_text = "".join(text_chunks).strip()
    if not full_text:
        # text_stream only yields TextDeltas; rebuild from final.content as fallback
        full_text = "".join(
            block.text for block in final.content if getattr(block, "type", None) == "text"
        ).strip()

    usage = {
        "input_tokens": getattr(final.usage, "input_tokens", 0),
        "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0) or 0,
        "output_tokens": getattr(final.usage, "output_tokens", 0),
    }
    return full_text, usage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_eod_summary(
    date_str: Optional[str] = None,
    save: bool = True,
    print_to_stdout: bool = True,
) -> tuple[str, Optional[Path]]:
    """
    Generate the EOD report. Returns (markdown_text, saved_path_or_None).

    Args:
        date_str:        UTC day in YYYY-MM-DD form. None → today UTC.
        save:            Write to analyzers/reports/btc_daily_<date>.md/.json.
        print_to_stdout: Echo the report so it shows in the terminal.
    """
    data = _gather(date_str)

    if data.today["trade_count"] == 0:
        msg = (
            f"# BTC EOD — {data.date_str}\n\n"
            "No trades logged for this date. (Either the bot didn't run, or no "
            "signal qualified, or the trade journal at "
            "`data/journal/btc_trades_<date>.jsonl` was not written to.)\n\n"
            f"Account state snapshot:\n\n```json\n{json.dumps(data.account_state, indent=2)}\n```\n"
        )
        if print_to_stdout:
            print(msg)
        path = _save_report(data.date_str, msg, {"trades": 0}) if save else None
        return msg, path

    system_blocks, messages = _build_messages(data)
    logger.info(
        f"[EOD] Calling {MODEL_ID} for {data.date_str} | "
        f"trades={data.today['trade_count']} | "
        f"PnL=${data.today['gross_pnl_dollars']:+.2f}"
    )
    report, usage = _call_claude(system_blocks, messages)

    logger.info(
        f"[EOD] Claude returned | input={usage['input_tokens']}t "
        f"cache_read={usage['cache_read_input_tokens']}t "
        f"cache_create={usage['cache_creation_input_tokens']}t "
        f"output={usage['output_tokens']}t"
    )

    if print_to_stdout:
        print(report)

    saved_path: Optional[Path] = None
    if save:
        saved_path = _save_report(data.date_str, report, {
            "usage": usage,
            "model": MODEL_ID,
            "data_summary": {k: v for k, v in data.today.items() if k != "trades"},
        })

    return report, saved_path


def _save_report(date_str: str, report_md: str, meta: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / f"btc_daily_{date_str}.md"
    json_path = REPORTS_DIR / f"btc_daily_{date_str}.json"

    md_path.write_text(report_md, encoding="utf-8")

    payload = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_md": report_md,
        **meta,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(f"[EOD] Saved → {md_path}")
    return md_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    arg_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_eod_summary(date_str=arg_date)
