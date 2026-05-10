#!/usr/bin/env python3
"""
Run the BTC Big Money Bot on Alpaca paper account.

Usage:
    python run_btc.py
    python run_btc.py --scheduled-start    # adds 30-min warmup (skip otherwise)

Differences from topstep5/run_big_money.py:
    * No Topstep account selection prompt — Alpaca creds come from .env
    * 24/7 crypto: no RTH/EOD shutdown gates
    * BTC instrument config replaces MES
"""

import argparse
import asyncio
import io
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Config
from live.big_money_bot import BigMoneyBot


def main():
    # Force UTF-8 on Windows console
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="BTC Big Money Bot — Alpaca paper")
    parser.add_argument(
        "--scheduled-start",
        action="store_true",
        help="Add 30-min warmup before trading (default: 0 — assume bars warm).",
    )
    args = parser.parse_args()

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    current_log = log_dir / "btc_bot.log"
    if current_log.exists() and current_log.stat().st_size > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = log_dir / f"btc_bot_{ts}.log"
        try:
            current_log.rename(archive)
            print(f"  Archived previous log -> {archive.name}")
        except PermissionError:
            print("  Could not archive log (in use); will overwrite.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(current_log, mode="w", encoding="utf-8"),
        ],
    )

    config = Config.load()
    bot = BigMoneyBot(config)

    # Crypto-specific overrides:
    # 1) Force poll-only — Alpaca crypto doesn't fit the SignalR market-hub model;
    #    the adapter stubs the hubs, so the main loop must drive off bar polling.
    bot._poll_only = True
    # 2) Defeat ALL futures-time-of-day gates — BTC trades 24/7.
    #    Without these, the bot would silently block trades during ES close
    #    (4:00-4:30 PM ET) and futures session-open buffer (6:00-6:30 PM ET).
    bot._bm_session_end_shutdown_fired = True   # blocks the 14:15 self-shutdown branch
    bot._crypto_24_7 = True                     # consulted by _process_bar_impl
    bot.risk._crypto_24_7 = True                # neutralizes RiskManager close-cutoff + session-open-buffer
    bot.risk.config.session_open_buffer_minutes = 0  # belt+suspenders
    # 3) Warmup (only if a scheduled start; manual restart trades immediately on warm bars)
    bot.bm_config.warmup_minutes = 30 if args.scheduled_start else 0

    logging.info(
        f"[STARTUP] BTC Big Money Bot | symbol={config.instrument.symbol} | "
        f"contract_size={config.instrument.contract_size_btc} BTC | "
        f"poll_only={bot._poll_only} | warmup={bot.bm_config.warmup_minutes}min | "
        f"session_end_gates=disabled (24/7 crypto)"
    )

    def shutdown(sig, _frame):
        logging.info(f"[BTC] Shutdown signal received ({sig})")
        bot.running = False

    signal.signal(signal.SIGINT, shutdown)
    try:
        signal.signal(signal.SIGTERM, shutdown)
    except (AttributeError, ValueError):
        # SIGTERM not always available on Windows
        pass

    print("\n" + "=" * 60)
    print("  BTC BIG MONEY BOT — Alpaca paper trading")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
