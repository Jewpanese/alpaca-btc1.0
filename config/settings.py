"""
alpaca_btc — Configuration

Mirrors topstep5 Config shape so the inherited HybridBot/AlphaBot/BigMoneyBot logic
works unchanged. The `topstep` field carries Alpaca credentials (keeps the field
name to avoid touching call sites that do `config.topstep.contract_id` etc.).

BTC-vs-MES sizing convention:
    1 native "contract" = 0.01 BTC (configurable via InstrumentConfig.contract_size_btc)
    point_value = 0.01 ($/pt/contract — a $1 BTC move on 1 contract is $0.01)
    tick_size = 0.01 (USD)
    "points" everywhere in the inherited code = USD price moves of BTC

This means:
    - At BTC=$100k, 10 contracts = 0.1 BTC = $10k notional
    - $500 max risk with $1500 stop_pts (3xATR @ ATR=$500): contracts = 500/(0.01*1500) = 33
    - Cushion tiers retuned below for crypto scale
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Load .env at import time (use python-dotenv if installed; fall back to a tiny parser)
_env_path = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_env_path)
except ImportError:
    if _env_path.exists():
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------
# Alpaca credentials (named `TopstepConfig` for drop-in compat with parent code)
# ----------------------------------------------------------------------

@dataclass
class AlpacaConfig:
    """Pure Alpaca creds; used by core/connection.py."""
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True


@dataclass
class TopstepConfig:
    """
    Adapter shape — keeps the field name 'topstep' across the codebase.
    For Alpaca/BTC, 'username' carries the Alpaca API key and 'api_key' the Alpaca secret.
    `contract_id` is the BTC trading symbol.
    """
    username: str = ""              # = ALPACA api key
    api_key: str = ""               # = ALPACA secret key
    secret_key: str = ""            # = ALPACA secret key (preferred name)
    account_id: str = "alpaca-paper"
    api_endpoint: str = "https://paper-api.alpaca.markets"
    signalr_hub: str = ""           # unused for crypto polling
    contract_id: str = "BTC/USD"    # set from InstrumentConfig.contract_id at load()
    paper: bool = True

    @classmethod
    def from_env(cls, env_path: Optional[str] = None):
        cfg = cls()
        cfg.username = os.getenv("ALPACA_PAPER_API_KEY", "")
        cfg.api_key = os.getenv("ALPACA_PAPER_SECRET_KEY", "") or os.getenv("ALPACA_PAPER_API_SECRET", "")
        cfg.secret_key = cfg.api_key
        paper_env = os.getenv("USE_PAPER_TRADING", "True")
        cfg.paper = paper_env.lower() in ("true", "1", "yes")
        if not cfg.paper:
            cfg.api_endpoint = "https://api.alpaca.markets"
        return cfg


# ----------------------------------------------------------------------
# Account constraints (BTC-tuned)
# ----------------------------------------------------------------------

@dataclass
class AccountConfig:
    """
    Account constraints — non-negotiable. Tuned for $100k Alpaca paper account
    with the user's stated 5% max-risk policy.
    """
    starting_balance: float = 100_000.0
    mll_threshold: float = 95_000.0           # -5% from start = stop trading
    daily_loss_hard_stop: float = 5_000.0     # 5% — hard halt
    daily_loss_soft_stop: float = 2_500.0     # 2.5% — stop NEW entries; let exits run
    daily_profit_target: float = 100.0        # $100/day target — lock in & stop


# ----------------------------------------------------------------------
# Instrument: BTC spot
# ----------------------------------------------------------------------

@dataclass
class InstrumentConfig:
    """
    Drop-in replacement for topstep5 InstrumentConfig (which switched ES/MES).
    Here we only support 'BTC' on Alpaca crypto spot.

    The math used throughout the inherited bot logic:
        pnl_dollars = pnl_pts * point_value * contracts

    For BTC:
        pnl_pts = (exit_price - entry_price)        # USD per BTC
        point_value = contract_size_btc             # i.e. 0.01 if 1 contract = 0.01 BTC
        contracts = integer count of native units

    So pnl_dollars = (exit - entry) * 0.01 * contracts when contract_size_btc = 0.01.
    Verified: 10 contracts (=0.1 BTC) on a $1000 BTC move = $1000 * 0.01 * 10 = $100. Correct.
    """
    instrument: str = "BTC"
    symbol: str = "BTC/USD"
    contract_size_btc: float = 0.01           # 1 contract = 0.01 BTC
    btc_tick_size: float = 0.01               # USD tick
    btc_tick_value: float = 0.0001            # = tick_size * contract_size_btc
    btc_point_value: float = 0.01             # $/pt/contract = contract_size_btc

    # Compatibility shims for code that still references MES/ES paths
    es_contract_id: str = "BTC/USD"
    mes_contract_id: str = "BTC/USD"
    es_tick_size: float = 0.01
    mes_tick_size: float = 0.01
    es_tick_value: float = 0.0001
    mes_tick_value: float = 0.0001
    es_point_value: float = 0.01
    mes_point_value: float = 0.01
    mes_per_es: int = 1                       # no MES/ES split for crypto

    @property
    def contract_id(self) -> str:
        return self.symbol

    @property
    def tick_size(self) -> float:
        return self.btc_tick_size

    @property
    def tick_value(self) -> float:
        return self.btc_tick_value

    @property
    def point_value(self) -> float:
        return self.btc_point_value

    @property
    def symbol_id(self) -> str:
        return self.symbol

    def es_equivalent(self, contracts: int) -> float:
        return float(contracts)

    def to_mes(self, contracts: int) -> int:
        return int(contracts)


# ----------------------------------------------------------------------
# Trading parameters (BTC-tuned)
# ----------------------------------------------------------------------

@dataclass
class TradingConfig:
    """
    BTC-volatility-tuned parameters. Stop distances are USD price moves (BTC points).
    Contract counts are scaled to 0.01-BTC native units.

    Initial values are educated starts; tune once paper trading begins.
    """
    # Position sizing — hard caps. The %-sizing path below is the primary
    # mechanism; max_contracts is a safety ceiling that binds in low-ATR
    # regimes where the % target would otherwise size very large.
    # At max_contracts=50: 0.50 BTC = ~$40k notional at $80k BTC (~40% of $100k account).
    # To fully realize 2% risk on typical ATR ($300-500 stops), bump to ~150-200.
    base_contracts: int = 10                 # 0.10 BTC = ~$10k notional at $100k
    max_contracts: int = 50                  # safety cap — see note above

    # ── Spot %-of-account sizing (BTC) ──────────────────────────────
    # Standard quant fixed-fractional sizing: per-trade risk = balance × pct.
    # Replaces the futures-prop "fixed $500 risk" + "cushion-tier max contracts"
    # approach. Position size auto-scales with the account.
    risk_per_trade_pct: float = 0.02         # 2% of account at risk per trade (1-5% range)
    risk_per_trade_pct_max: float = 0.05     # hard ceiling — even if user sets higher, clamped here
    # Notional cap for spot (no leverage). 1.0 = can use 100% of cash on a single
    # position; lower for safety margin. The bot won't try to buy more BTC than
    # this fraction of the account would cover.
    notional_cap_pct: float = 1.0
    # Drawdown deleverage: at each daily-DD tier, scale per-trade size by mult.
    # Replaces cushion-tier sizing (which was a Topstep-prop concept).
    drawdown_size_tiers: list = field(default_factory=lambda: [
        (0.00, 1.00),   # 0%   DD → full size
        (0.02, 0.75),   # 2%   DD → 75%
        (0.05, 0.50),   # 5%   DD → half
        (0.10, 0.25),   # 10%  DD → quarter
        (0.15, 0.00),   # 15%  DD → stop trading
    ])

    # Risk per trade (legacy fixed-$ — kept for futures portability; spot uses pct)
    max_loss_per_trade: float = 500.0        # $500 hard stop per idea
    min_risk_reward: float = 1.1

    # Frequency
    max_trades_per_hour: int = 4             # crypto volume is steadier; pace it
    max_trades_per_day: int = 12
    min_seconds_between_trades: int = 120    # 2 min between trades (vs Topstep 15s)

    # Loss streak
    max_consecutive_losses: int = 3
    loss_streak_cooldown_seconds: int = 600  # 10 min after 3 losses

    # Sessions — kept for code compat, but BTC trades 24/7
    rth_open_hour: int = 0
    rth_open_minute: int = 0
    rth_close_hour: int = 23
    rth_close_minute: int = 59


# ----------------------------------------------------------------------
# Master config
# ----------------------------------------------------------------------

@dataclass
class Config:
    topstep: TopstepConfig = field(default_factory=TopstepConfig)   # Alpaca creds (legacy field name)
    account: AccountConfig = field(default_factory=AccountConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)

    @classmethod
    def load(cls, env_path: Optional[str] = None):
        config = cls(topstep=TopstepConfig.from_env(env_path))
        config.topstep.contract_id = config.instrument.contract_id
        return config


# ----------------------------------------------------------------------
# Helpers (kept for legacy imports)
# ----------------------------------------------------------------------

def load_settings() -> Config:
    """Legacy entrypoint."""
    return Config.load()


def mask_key(key: str) -> str:
    if not key:
        return "***"
    if len(key) <= 6:
        return "***"
    return key[:3] + "***" + key[-3:]
