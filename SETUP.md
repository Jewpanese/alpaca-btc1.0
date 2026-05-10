# alpaca_btc — Setup & Run

## First-time setup

```powershell
# 1. Clone
git clone https://github.com/Jewpanese/alpaca-btc1.0.git
cd alpaca-btc1.0

# 2. Create virtual env (Python 3.11 or 3.14, both verified)
python -m venv .venv

# 3. Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# Or on bash / Git Bash / WSL:
# source .venv/Scripts/activate

# 4. Install deps
pip install -r requirements.txt

# 5. Configure credentials
copy .env.example .env
# Then edit .env with your real keys:
#   ALPACA_PAPER_API_KEY=PK...
#   ALPACA_PAPER_SECRET_KEY=...
#   ANTHROPIC_API_KEY=sk-ant-...
```

If PowerShell rejects `Activate.ps1` with "running scripts is disabled" — run
once per session: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Daily run

```powershell
# Activate the venv (whenever you open a new PowerShell window)
.\.venv\Scripts\Activate.ps1

# Start the bot
python run_btc.py
```

You should see `(.venv)` in your prompt — that's how you know you're using
the project's pinned dependencies, not the system Python.

## End-of-day Claude review

```powershell
# Inside the venv
python run_eod_summary.py            # today (UTC)
python run_eod_summary.py 2026-05-10 # specific date
```

## Verifying you're in the right env

```powershell
# This must point at .venv\Scripts\python.exe, NOT global Python
python -c "import sys; print(sys.executable)"
```

If it shows a system path like `C:\Python314\python.exe`, the venv is not
activated. Run `.\.venv\Scripts\Activate.ps1` first.

## Why a venv

- Pins dep versions per the project's `requirements.txt` — no clashes with
  other Python projects on the same machine.
- Recreatable: nuke `.venv\` and rebuild any time. Nothing is in your global
  Python that the project secretly depends on.
- The `.gitignore` excludes `.venv/` so your environment is never committed.

## Status dashboard

While the bot is running: **http://localhost:8585**

## Logs and reports

- Bot log: `logs/btc_bot.log` (rotated each run)
- BarRecorder CSVs: `data/live/BTC_<date>.csv`
- Trade journal: `data/journal/btc_trades_<date>.jsonl`
- Account state: `data/mll_state.json`
- EOD reports: `analyzers/reports/btc_daily_<date>.md`
