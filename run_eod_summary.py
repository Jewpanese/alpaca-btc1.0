#!/usr/bin/env python3
"""
End-of-Day BTC Trading Review — fires Claude Opus 4.7 over today's trade journal.

Usage:
    python run_eod_summary.py                # today (UTC)
    python run_eod_summary.py 2026-05-09     # explicit date
    python run_eod_summary.py --no-save      # don't write to disk

Reads:
    data/journal/btc_trades_<date>.jsonl   (written by RiskManager.record_trade)
    data/mll_state.json                    (account snapshot)
    data/live/<*date*>.csv                 (BarRecorder output for market context)

Writes:
    analyzers/reports/btc_daily_<date>.md
    analyzers/reports/btc_daily_<date>.json

Requires:
    ANTHROPIC_API_KEY in env (or .env at project root).
"""

import argparse
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Make sure .env is loaded so ANTHROPIC_API_KEY (and Alpaca creds) are available
import config.settings  # noqa: F401  side-effect: loads .env

from analyzers.eod_summary import run_eod_summary


def main() -> int:
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="BTC bot end-of-day review (Claude Opus 4.7)")
    parser.add_argument("date", nargs="?", default=None, help="UTC date YYYY-MM-DD; defaults to today")
    parser.add_argument("--no-save", action="store_true", help="Don't persist the report to disk")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout echo of the report body")
    parser.add_argument("--debug", action="store_true", help="Verbose SDK + module logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  BTC End-of-Day Review — Claude Opus 4.7")
    print(f"  Date: {args.date or 'today (UTC)'}")
    print("=" * 60)
    print()

    try:
        report, path = run_eod_summary(
            date_str=args.date,
            save=not args.no_save,
            print_to_stdout=not args.quiet,
        )
    except RuntimeError as e:
        # Friendly message for the most common setup mistakes
        print(f"\n[!] {e}\n", file=sys.stderr)
        return 2
    except Exception as e:
        logging.exception("EOD summary failed")
        print(f"\n[!] Unexpected error: {e}\n", file=sys.stderr)
        return 1

    if path:
        print()
        print(f"  Saved → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
