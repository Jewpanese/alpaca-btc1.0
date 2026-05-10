"""Topstep Maximum Loss Limit (MLL) Tracker.

The MLL is a trailing floor that moves UP with your account balance,
capped at the account size. It never goes down.

Example ($150K account, initial MLL = $145,500, initial cushion = $4,500):
  Day 0: Balance $150,000 | MLL $145,500 | Cushion $4,500
  Day 1: Balance $152,000 | MLL $147,500 | Cushion $4,500  (MLL trails up)
  Day 2: Balance $155,000 | MLL $150,000 | Cushion $5,000  (MLL capped at account size)
  Day 3: Balance $153,000 | MLL $150,000 | Cushion $3,000  (MLL stays, cushion shrinks!)

Once MLL locks at $150K, every dollar lost comes directly out of your cushion.
Multi-contract scaling MUST be based on cushion, not raw balance.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Throttle MLL status logging to once per 60 seconds
_last_mll_log_time: float = 0.0
_last_mll_log_msg: str = ""

MLL_STATE_FILE = Path(__file__).parent.parent / "data" / "mll_state.json"


@dataclass
class TopstepAccountConfig:
    """Account parameters (originally Topstep MLL; here for the BTC port the
    MLL concept is reused as a 5%-from-peak drawdown floor).

    BTC defaults; HybridBot.__init__ overrides everything from AccountConfig
    + InstrumentConfig at startup, so these are the safety net for direct use.
    """
    account_size: float = 100_000.0        # BTC paper start (was 150k Topstep)
    initial_mll: float = 95_000.0          # 5% below start
    initial_cushion: float = 5_000.0       # account_size - initial_mll
    point_value: float = 0.01              # BTC: $0.01/pt/contract (was 5.0 MES)
    max_position_contracts: int = 50       # 50 contracts = 0.50 BTC max
    instrument: str = "BTC"                # was "MES"
    mes_per_es: int = 1                    # No futures contract-ratio for BTC (kept for API compat)


class MLLTracker:
    """Tracks the trailing MLL and calculates the real-time cushion.
    
    MLL formula:
        mll = min(account_size, max(initial_mll, highest_balance - initial_cushion))
    
    Cushion:
        cushion = current_balance - mll
    
    If cushion <= 0, the account is blown.
    """
    
    def __init__(self, config: TopstepAccountConfig = None):
        self.config = config or TopstepAccountConfig()
        self.highest_balance: float = self.config.account_size
        self.current_mll: float = self.config.initial_mll
        self.mll_locked: bool = False  # True once MLL reaches account_size
        
        # Try to restore state from disk
        self._load_state()
    
    def update(self, current_balance: float) -> float:
        """Update MLL based on current balance. Returns current cushion.
        
        Call this on every balance update (after each trade, on startup, etc.)
        """
        # Track highest balance ever seen
        if current_balance > self.highest_balance:
            self.highest_balance = current_balance
        
        # MLL trails up: highest_balance - initial_cushion, but never above account_size
        raw_mll = self.highest_balance - self.config.initial_cushion
        self.current_mll = min(self.config.account_size, max(self.config.initial_mll, raw_mll))
        self.mll_locked = self.current_mll >= self.config.account_size
        
        cushion = current_balance - self.current_mll
        
        # Persist state
        self._save_state()
        
        return cushion
    
    @property
    def cushion(self) -> float:
        """Current cushion (call update() first with latest balance)."""
        # Use last known balance
        return self.highest_balance - self.current_mll  # Approximation; real cushion needs current balance
    
    def get_max_contracts(self, current_balance: float) -> int:
        """Calculate max contracts allowed based on cushion.

        Returns contracts in NATIVE units. For BTC this port uses 1 contract = 0.01 BTC,
        so the tiers represent BTC notional at roughly $1,000 each (assuming BTC≈$100k).

        BTC cushion tiers (single risk-cap is `BigMoneyConfig.daily_loss_limit`):
            Cushion < $1,500  →  0 contracts (STOP)
            < $2,500          →  5 contracts (~0.05 BTC, ~$5k notional)
            < $5,000          → 10 contracts (~0.10 BTC, ~$10k notional)
            < $10,000         → 20 contracts (~0.20 BTC, ~$20k notional)
            < $15,000         → 30 contracts (~0.30 BTC, ~$30k notional)
            < $20,000         → 40 contracts (~0.40 BTC, ~$40k notional)
            ≥ $20,000         → 50 contracts (max — ~0.50 BTC, ~$50k notional)
        """
        cushion = self.update(current_balance)

        if cushion < 1_000:
            logger.warning(f"[MLL] DANGER: Cushion ${cushion:,.0f} < $1,000 — STOP TRADING")
            return 0

        # BTC-scale tiers (1 contract = 0.01 BTC). For futures (MES) the original
        # ramp was 0/2/5/7/9/11/13 — kept commented for reference.
        if cushion < 1_500:
            native_direct = 0
        elif cushion < 2_500:
            native_direct = 5
        elif cushion < 5_000:
            native_direct = 10
        elif cushion < 10_000:
            native_direct = 20
        elif cushion < 15_000:
            native_direct = 30
        elif cushion < 20_000:
            native_direct = 40
        else:
            native_direct = 50

        # Already in native contracts from graduated scaling
        native_cts = min(native_direct, self.config.max_position_contracts)

        # Build instrument-aware display: BTC shows the BTC notional;
        # MES shows MES + ES-equivalent; ES shows ES contracts.
        instr = self.config.instrument
        if instr == "MES":
            es_eq = native_cts / self.config.mes_per_es
            display = f"{native_cts} MES ({es_eq:.1f} ES-eq)"
        elif instr == "BTC":
            # Assumes 1 native contract = 0.01 BTC (alpaca_btc convention).
            btc_size = native_cts * 0.01
            display = f"{native_cts} ct ({btc_size:.2f} BTC)"
        else:
            display = f"{native_cts} {instr}"

        # Throttle MLL status logs: once per 60s or on change
        global _last_mll_log_time, _last_mll_log_msg
        msg = (f"[MLL] Balance=${current_balance:,.0f} | MLL=${self.current_mll:,.0f} | "
               f"Cushion=${cushion:,.0f} | {'LOCKED' if self.mll_locked else 'TRAILING'} | "
               f"Max={display}")
        now = time.time()
        if msg != _last_mll_log_msg or now - _last_mll_log_time >= 60:
            logger.info(msg)
            _last_mll_log_time = now
            _last_mll_log_msg = msg
        
        return native_cts
    
    def get_daily_loss_limit(self, current_balance: float) -> float:
        """Calculate how much we can afford to lose TODAY.

        BTC port: never risk more than 50% of cushion. Floor $500, cap $5,000
        (matches user-stated 5%-of-account hard stop on a $100k paper account).
        """
        cushion = self.update(current_balance)
        daily_limit = cushion * 0.50

        daily_limit = max(500.0, min(daily_limit, 5_000.0))

        logger.debug(f"[MLL] Daily loss limit: ${daily_limit:,.0f} (50% of ${cushion:,.0f} cushion)")
        return daily_limit
    
    def get_status(self, current_balance: float) -> dict:
        """Full status for logging/dashboard."""
        cushion = self.update(current_balance)
        max_cts = self.get_max_contracts(current_balance)
        
        return {
            "balance": current_balance,
            "mll": self.current_mll,
            "cushion": cushion,
            "mll_locked": self.mll_locked,
            "highest_balance": self.highest_balance,
            "max_contracts": max_cts,
            "daily_loss_limit": self.get_daily_loss_limit(current_balance),
            "danger_zone": cushion < 2_000,
        }
    
    def _save_state(self):
        """Persist MLL state to disk (survives restarts)."""
        try:
            MLL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "highest_balance": self.highest_balance,
                "current_mll": self.current_mll,
                "mll_locked": self.mll_locked,
                "config": {
                    "account_size": self.config.account_size,
                    "initial_mll": self.config.initial_mll,
                    "initial_cushion": self.config.initial_cushion,
                }
            }
            with open(MLL_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"[MLL] Failed to save state: {e}")
    
    def _load_state(self):
        """Restore MLL state from disk."""
        try:
            if MLL_STATE_FILE.exists():
                with open(MLL_STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.highest_balance = state.get("highest_balance", self.config.account_size)
                self.current_mll = state.get("current_mll", self.config.initial_mll)
                self.mll_locked = state.get("mll_locked", False)
                logger.info(
                    f"[MLL] Restored state: highest=${self.highest_balance:,.0f}, "
                    f"MLL=${self.current_mll:,.0f}, locked={self.mll_locked}"
                )
        except Exception as e:
            logger.warning(f"[MLL] Failed to load state (using defaults): {e}")
