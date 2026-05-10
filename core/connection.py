"""
Alpaca Crypto Connection — Adapter that exposes the TopstepConnection public API
so the inherited HybridBot/AlphaBot/BigMoneyBot logic works unchanged against Alpaca.

Mapping conventions (BTC vs MES):

    MES                                  BTC (this adapter)
    -------------------------------      -----------------------------------
    1 contract = 1 MES                   1 contract = 0.01 BTC (configurable)
    point_value = $5/pt                  point_value = $0.01/pt (1 BTC * 0.01 * $1 move = $0.01)
    tick_size = 0.25 pts                 tick_size = 0.01 (USD)
    points = price units (1 pt = 1 ES)   points = USD price move (1 pt = $1 BTC move)
    contract_id = "CON.F.US.MES.M26"     contract_id = "BTC/USD"
    SignalR market+user hubs             polling (bot already polls every bar)
    Native bracket orders                software-tracked SL/TP

Brackets are tracked in software inside this adapter; the actual broker only sees market
orders. When the underlying price hits a tracked stop or target, the adapter places the
exit order. This matches Alpaca crypto's lack of native OCO for spot.

Methods exposed (matching TopstepConnection):
    - authenticate, _headers
    - get_account_balance, get_account_pnl
    - get_open_positions, _get_all_positions
    - get_bars
    - place_order, place_stop_order
    - partial_close, flatten_all
    - get_working_orders, get_fill_price, get_close_fill_price
    - update_bracket_sizes, _cancel_all_working_orders, _place_manual_brackets
    - set_callbacks, set_user_callbacks
    - connect_signalr, connect_user_hub, reconnect_user_hub, disconnect
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import ccxt

from config.settings import AlpacaConfig, mask_key

logger = logging.getLogger("alpaca_btc.connection")

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


@dataclass
class _SoftBracket:
    """Software-tracked stop-loss / take-profit attached to an open position."""
    order_id: int
    side: str                      # "BUY" (long entry) — sells flatten longs
    contracts: int                 # native units (per InstrumentConfig.contract_size)
    btc_size: float                # actual BTC quantity
    entry_price: float
    stop_price: Optional[float]    # absolute price
    target_price: Optional[float]  # absolute price
    placed_ts: float = field(default_factory=time.time)


class AlpacaCryptoConnection:
    """
    Drop-in replacement for TopstepConnection that talks to Alpaca crypto via ccxt.
    Public method signatures match TopstepConnection so the rest of the bot code
    needs no changes.
    """

    def __init__(
        self,
        username: str = "",
        api_key: str = "",
        account_id: str = "",
        api_endpoint: str = "https://paper-api.alpaca.markets",
        signalr_hub: str = "",
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        symbol: Optional[str] = None,
        contract_size_btc: Optional[float] = None,
    ):
        # The HybridBot calls us with Topstep-style kwargs:
        #   username  = Alpaca API key  (mapped from ALPACA_PAPER_API_KEY)
        #   api_key   = Alpaca secret   (mapped from ALPACA_PAPER_SECRET_KEY)
        # Anything not passed is pulled from environment so this also works
        # if instantiated directly.
        import os as _os
        self.alpaca_api_key = username or _os.getenv("ALPACA_PAPER_API_KEY", "")
        self.alpaca_secret = secret_key or api_key or _os.getenv(
            "ALPACA_PAPER_SECRET_KEY", _os.getenv("ALPACA_PAPER_API_SECRET", "")
        )
        # Keep the original Topstep-shaped attrs around for any code that reads them
        self.username = username
        self.api_key = api_key
        self.account_id = account_id
        self.api_endpoint = api_endpoint
        self.signalr_hub = signalr_hub
        self.secret_key = self.alpaca_secret
        if paper is None:
            self.paper = _os.getenv("USE_PAPER_TRADING", "True").lower() in ("true", "1", "yes")
        else:
            self.paper = paper

        self.symbol = symbol or _os.getenv("BTC_SYMBOL", "BTC/USD")
        self.contract_size_btc = contract_size_btc if contract_size_btc is not None else 0.01

        self.exchange: Optional[ccxt.alpaca] = None
        self.session_token = "alpaca-ccxt"  # not used — keeps Topstep code paths happy
        self.connected = False
        # Compatibility flags read by HybridBot._main_loop
        self._needs_reconnect = False
        self._user_ws_connected = True          # we don't use a user hub; pretend connected
        self._reconnect_delays = [2, 5, 5, 10, 15, 30]
        self._reconnect_attempt = 0

        # Software bracket tracking: keyed by entry order_id
        self._brackets: Dict[int, _SoftBracket] = {}
        self._next_order_id = 100_000  # local order-id allocator; Alpaca returns its own too
        self._fills: Dict[int, dict] = {}      # order_id -> fill record
        self._fill_history: deque = deque(maxlen=500)  # for get_close_fill_price scanning

        # Callbacks (compat with TopstepConnection.set_callbacks / set_user_callbacks)
        self._on_quote: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_user_trade: Optional[Callable] = None
        self._on_user_order: Optional[Callable] = None
        self._on_user_position: Optional[Callable] = None
        self._on_user_account: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Auth / lifecycle
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        try:
            self.exchange = ccxt.alpaca({
                "apiKey": self.alpaca_api_key,
                "secret": self.alpaca_secret,
                "sandbox": self.paper,
                "options": {"defaultType": "spot"},
            })
            self.exchange.load_markets()
            self.connected = True
            logger.info(
                f"Alpaca authenticated ({'paper' if self.paper else 'live'}) | "
                f"key={mask_key(self.alpaca_api_key)} | symbol={self.symbol}"
            )
            return True
        except Exception as e:
            logger.error(f"Alpaca authenticate failed: {e}")
            self.connected = False
            return False

    @property
    def is_connected(self) -> bool:
        # HybridBot._main_loop reads `self.conn.is_connected`. Mirror our state.
        return self.connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self.connected = bool(value)

    def _headers(self) -> dict:
        # Topstep code occasionally builds custom HTTP requests with this; for crypto
        # we have no equivalent. Return an empty dict — any code path that hits this
        # against Alpaca will need rework upstream (logged for visibility).
        logger.warning("AlpacaCryptoConnection._headers() called — no equivalent for Alpaca; returning {}")
        return {}

    def _retry(self, fn, *args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                logger.warning(f"Network error (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    raise
            except ccxt.ExchangeError as e:
                logger.error(f"Alpaca exchange error: {e}")
                raise

    # ------------------------------------------------------------------
    # Account / balance
    # ------------------------------------------------------------------

    def get_account_balance(self) -> Optional[float]:
        try:
            bal = self._retry(self.exchange.fetch_balance)
            usd = bal.get("free", {}).get("USD", 0) or bal.get("USD", {}).get("free", 0) or 0
            return float(usd)
        except Exception as e:
            logger.error(f"get_account_balance failed: {e}")
            return None

    def get_account_pnl(self) -> Optional[dict]:
        """Approximate Topstep-shaped PnL summary from Alpaca balances and BTC mark."""
        try:
            bal = self._retry(self.exchange.fetch_balance)
            usd_free = float(bal.get("free", {}).get("USD", 0) or 0)
            btc_total = float(bal.get("total", {}).get("BTC", 0) or 0)
            mark = self._get_mark_price()
            unrealized = btc_total * mark
            return {
                "balance": usd_free + unrealized,
                "realizedPnl": 0.0,        # Alpaca doesn't separate realized intraday in this view
                "unrealizedPnl": unrealized,
                "totalPnl": unrealized,
                "openPnl": unrealized,
            }
        except Exception as e:
            logger.error(f"get_account_pnl failed: {e}")
            return None

    def _get_mark_price(self) -> float:
        try:
            ticker = self._retry(self.exchange.fetch_ticker, self.symbol)
            return float(ticker.get("last", 0) or 0)
        except Exception as e:
            logger.error(f"mark price fetch failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> Optional[List[dict]]:
        """
        Return positions in Topstep-shaped dicts. For BTC spot, position = BTC balance > 0.
        Returns [] on confirmed flat, None on error (matches TopstepConnection contract).
        """
        try:
            bal = self._retry(self.exchange.fetch_balance)
            btc = float(bal.get("total", {}).get("BTC", 0) or 0)
            if btc < 1e-6:
                return []
            avg_entry = self._estimate_avg_entry()
            net_size_contracts = int(round(btc / self.contract_size_btc))
            return [{
                "contractId": self.symbol,
                "size": net_size_contracts,           # native contract count
                "netSize": net_size_contracts,        # Topstep convention (>0 = long)
                "btcSize": btc,
                "averagePrice": avg_entry,
                "type": "LONG",
            }]
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return None

    def _get_all_positions(self) -> Optional[List[dict]]:
        return self.get_open_positions()

    def _estimate_avg_entry(self) -> float:
        # Use most recent BUY in our local fill history; fallback to mark.
        for rec in reversed(self._fill_history):
            if rec.get("side") == "BUY":
                return float(rec.get("price", 0) or 0)
        return self._get_mark_price()

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_bars(
        self,
        contract_id: str,
        bars_back: int = 400,
        from_time=None,
        to_time=None,
        unit: int = 2,         # Topstep: 2=Minute (we honor that)
        unit_number: int = 1,  # 1=1m, 3=3m, 5=5m, 15=15m
    ) -> List[dict]:
        """
        Returns Topstep-shaped bars: list of {"t","o","h","l","c","v"} oldest-first.
        Crypto trades 24/7; from_time/to_time honored if given.
        """
        if not self.exchange:
            return []
        # Map Topstep unit/unit_number to ccxt timeframe
        tf_map_minute = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        if unit == 2:
            tf = tf_map_minute.get(unit_number, "1m")
        elif unit == 3:
            tf = "1h"
        elif unit == 4:
            tf = "1d"
        else:
            tf = "1m"

        # NOTE on ccxt-Alpaca behavior (verified 2026-05-10):
        #   * Passing `since=<ms>` to fetch_ohlcv silently returns 0 bars for
        #     small windows — the parameter is unreliable on this venue.
        #   * Without `since`, fetch_ohlcv pages FORWARD from a fixed anchor
        #     (today's UTC midnight). `limit=5` returns the OLDEST 5 bars of
        #     the day, not the newest 5. To get the most recent bars we must
        #     fetch a wide window and slice the tail.
        # Strategy: ignore `from_time` (we have no reliable mapping), use a
        # generous limit (covers >1 trading day for any timeframe ≥ 1m), then
        # tail-slice to bars_back. Throws away a few KB per call but keeps the
        # poll loop on actually-fresh data.
        fetch_limit = max(bars_back * 4, 500)

        for attempt in range(MAX_RETRIES):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, tf, limit=fetch_limit
                )
                if not ohlcv:
                    if attempt + 1 < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    return []
                bars = [
                    {
                        "t": _ms_to_iso(b[0]),
                        "o": float(b[1]),
                        "h": float(b[2]),
                        "l": float(b[3]),
                        "c": float(b[4]),
                        "v": float(b[5]),
                    }
                    for b in ohlcv
                ]
                bars.sort(key=lambda x: x["t"])
                if len(bars) > bars_back:
                    bars = bars[-bars_back:]
                return bars
            except Exception as e:
                logger.warning(f"get_bars attempt {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(2 ** attempt)
        return []

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        contract_id: str,
        side: str,
        size: int,
        stop_loss_points: Optional[float] = None,
        take_profit_points: Optional[float] = None,
        current_price: float = 0.0,
        limit_price: Optional[float] = None,
    ):
        """
        Place a market or limit entry. SL/TP "points" are USD price moves for BTC.
        Brackets are stored in software and monitored by the bot loop or via
        check_software_brackets() called externally.

        Returns: order_id (int) on success, False on failure.
        """
        if side not in ("BUY", "SELL"):
            logger.error(f"place_order: invalid side {side}")
            return False
        if size <= 0:
            logger.error(f"place_order: size must be > 0, got {size}")
            return False

        btc_size = size * self.contract_size_btc
        ccxt_side = "buy" if side == "BUY" else "sell"
        order_type = "limit" if limit_price else "market"

        try:
            kwargs = {}
            if order_type == "limit":
                kwargs["price"] = limit_price
            order = self._retry(
                self.exchange.create_order,
                self.symbol, order_type, ccxt_side, btc_size, **kwargs
            )
            broker_id = order.get("id", "")
            local_id = self._alloc_order_id()
            fill_price = float(order.get("average") or order.get("price") or current_price or self._get_mark_price())

            self._fills[local_id] = {
                "order_id": local_id,
                "broker_id": broker_id,
                "side": side,
                "size": size,
                "btc_size": btc_size,
                "price": fill_price,
                "status": "FILLED",
                "ts": time.time(),
            }
            self._fill_history.append(self._fills[local_id])

            # Software bracket tracking (entry only — SL/TP are passive triggers)
            if side == "BUY" and (stop_loss_points or take_profit_points):
                stop_price = (fill_price - stop_loss_points) if stop_loss_points else None
                target_price = (fill_price + take_profit_points) if take_profit_points else None
                self._brackets[local_id] = _SoftBracket(
                    order_id=local_id, side=side, contracts=size, btc_size=btc_size,
                    entry_price=fill_price, stop_price=stop_price, target_price=target_price,
                )
                logger.info(
                    f"Bracket tracked: id={local_id} entry=${fill_price:.2f} "
                    f"SL=${stop_price:.2f if stop_price else 0} "
                    f"TP=${target_price:.2f if target_price else 0}"
                )

            logger.info(
                f"ORDER FILLED: {side} {size} contracts ({btc_size:.6f} BTC) @ ${fill_price:.2f} | "
                f"id={local_id} broker={broker_id}"
            )

            # Fire user callback so HybridBot reconciliation sees the fill
            if self._on_user_trade:
                try:
                    self._on_user_trade({
                        "orderId": local_id,
                        "contractId": contract_id,
                        "side": 0 if side == "BUY" else 1,
                        "size": size,
                        "price": fill_price,
                        "fillType": 2,
                    })
                except Exception as e:
                    logger.warning(f"on_user_trade callback raised: {e}")

            return local_id
        except Exception as e:
            logger.error(f"place_order failed: {side} {size} @ {self.symbol}: {e}")
            return False

    def place_stop_order(
        self, contract_id: str, side: str, size: int, stop_price: float
    ) -> bool:
        """
        Standalone stop-loss leg. For Alpaca crypto we register a software stop.
        Returns True on success.
        """
        # Find matching open bracket; update its stop. If none, create a passive stop.
        match = None
        for br in self._brackets.values():
            if br.contracts == size:
                match = br
                break
        if match:
            match.stop_price = stop_price
            logger.info(f"Stop updated to ${stop_price:.2f} for bracket id={match.order_id}")
            return True
        # No matching bracket — register a fresh passive stop
        local_id = self._alloc_order_id()
        self._brackets[local_id] = _SoftBracket(
            order_id=local_id, side="BUY", contracts=size,
            btc_size=size * self.contract_size_btc,
            entry_price=self._get_mark_price(), stop_price=stop_price, target_price=None,
        )
        logger.info(f"Standalone stop registered id={local_id} stop=${stop_price:.2f}")
        return True

    def _place_manual_brackets(
        self, contract_id, side, size, stop_loss_points, take_profit_points, entry_price
    ) -> bool:
        # For Alpaca we always use software brackets; the place_order call already
        # registered them. This method is a no-op kept for API compatibility.
        return True

    # ------------------------------------------------------------------
    # Position close / partial close
    # ------------------------------------------------------------------

    def partial_close(self, contract_id: str, size: int) -> bool:
        """Sell `size` contracts (BTC = size * contract_size_btc) at market."""
        try:
            btc_size = size * self.contract_size_btc
            order = self._retry(
                self.exchange.create_order,
                self.symbol, "market", "sell", btc_size
            )
            broker_id = order.get("id", "")
            fill_price = float(order.get("average") or order.get("price") or self._get_mark_price())
            local_id = self._alloc_order_id()
            self._fills[local_id] = {
                "order_id": local_id, "broker_id": broker_id, "side": "SELL",
                "size": size, "btc_size": btc_size, "price": fill_price,
                "status": "FILLED", "ts": time.time(),
            }
            self._fill_history.append(self._fills[local_id])
            logger.info(f"PARTIAL CLOSE: {size} contracts ({btc_size:.6f} BTC) @ ${fill_price:.2f}")

            # Update bracket bookkeeping — reduce contracts on remaining brackets FIFO
            self._reduce_brackets(size)

            # Fire user-trade callback
            if self._on_user_trade:
                try:
                    self._on_user_trade({
                        "orderId": local_id, "contractId": contract_id,
                        "side": 1, "size": size, "price": fill_price, "fillType": 2,
                    })
                except Exception as e:
                    logger.warning(f"on_user_trade callback raised: {e}")
            return True
        except Exception as e:
            logger.error(f"partial_close failed: {e}")
            return False

    def flatten_all(self, contract_id: str, reason: str = "") -> bool:
        """Cancel any working orders + sell entire BTC balance."""
        self._cancel_all_working_orders(reason=f"flatten_all: {reason}")
        try:
            bal = self._retry(self.exchange.fetch_balance)
            btc = float(bal.get("total", {}).get("BTC", 0) or 0)
            if btc < 1e-6:
                logger.info("flatten_all: already flat")
                self._brackets.clear()
                return True
            order = self._retry(
                self.exchange.create_order, self.symbol, "market", "sell", btc
            )
            fill_price = float(order.get("average") or order.get("price") or self._get_mark_price())
            local_id = self._alloc_order_id()
            self._fills[local_id] = {
                "order_id": local_id, "broker_id": order.get("id", ""),
                "side": "SELL", "size": int(round(btc / self.contract_size_btc)),
                "btc_size": btc, "price": fill_price, "status": "FILLED", "ts": time.time(),
            }
            self._fill_history.append(self._fills[local_id])
            logger.info(f"FLATTENED: {btc:.6f} BTC @ ${fill_price:.2f} | reason={reason}")
            self._brackets.clear()
            if self._on_user_trade:
                try:
                    self._on_user_trade({
                        "orderId": local_id, "contractId": contract_id,
                        "side": 1, "size": int(round(btc / self.contract_size_btc)),
                        "price": fill_price, "fillType": 2,
                    })
                except Exception as e:
                    logger.warning(f"on_user_trade callback raised: {e}")
            return True
        except Exception as e:
            logger.error(f"flatten_all failed: {e}")
            return False

    def update_bracket_sizes(self, new_size: int) -> bool:
        """After partial close, replace bracket bookkeeping with new size."""
        if not self._brackets:
            return True
        # Keep first bracket, drop the rest, set its contracts to new_size
        first_id = next(iter(self._brackets))
        keep = self._brackets[first_id]
        keep.contracts = new_size
        keep.btc_size = new_size * self.contract_size_btc
        self._brackets = {first_id: keep}
        logger.info(f"Bracket sizes updated to {new_size} contracts ({keep.btc_size:.6f} BTC)")
        return True

    def _cancel_all_working_orders(self, reason: str = "") -> None:
        """Cancel resting orders on Alpaca + clear our software brackets."""
        try:
            self._retry(self.exchange.cancel_all_orders, self.symbol)
            logger.info(f"All working orders cancelled | reason={reason}")
        except Exception as e:
            logger.warning(f"cancel_all_orders: {e}")
        self._brackets.clear()

    # ------------------------------------------------------------------
    # Order queries
    # ------------------------------------------------------------------

    def get_working_orders(self) -> List[dict]:
        try:
            orders = self._retry(self.exchange.fetch_open_orders, self.symbol)
            return [
                {
                    "orderId": o.get("id"),
                    "contractId": self.symbol,
                    "type": 1 if o.get("type") == "limit" else 2,
                    "side": 0 if o.get("side") == "buy" else 1,
                    "size": int(round(float(o.get("amount", 0) or 0) / self.contract_size_btc)),
                    "limitPrice": o.get("price"),
                    "status": 0,
                    "customTag": o.get("clientOrderId"),
                }
                for o in (orders or [])
            ]
        except Exception as e:
            logger.error(f"get_working_orders failed: {e}")
            return []

    def get_fill_price(
        self, contract_id: str, order_id: Optional[int] = None
    ) -> Optional[float]:
        # 1. Direct lookup in local fill cache
        if order_id is not None and order_id in self._fills:
            return float(self._fills[order_id]["price"])
        # 2. Most recent BUY fill within 5 minutes
        cutoff = time.time() - 300
        for rec in reversed(self._fill_history):
            if rec.get("side") == "BUY" and rec.get("ts", 0) >= cutoff:
                return float(rec["price"])
        # 3. Fall back to current mark
        mark = self._get_mark_price()
        return mark or None

    def get_close_fill_price(
        self, contract_id: str, exit_side: int, window_minutes: int = 5
    ) -> Optional[float]:
        cutoff = time.time() - window_minutes * 60
        target_side = "SELL" if exit_side == 1 else "BUY"
        for rec in reversed(self._fill_history):
            if rec.get("side") == target_side and rec.get("ts", 0) >= cutoff:
                return float(rec["price"])
        return None

    # ------------------------------------------------------------------
    # Real-time hubs (no-ops for crypto polling architecture)
    # ------------------------------------------------------------------

    def set_callbacks(self, on_quote=None, on_trade=None) -> None:
        self._on_quote = on_quote
        self._on_trade = on_trade

    def set_user_callbacks(
        self, on_user_trade=None, on_user_order=None,
        on_user_position=None, on_user_account=None
    ) -> None:
        self._on_user_trade = on_user_trade
        self._on_user_order = on_user_order
        self._on_user_position = on_user_position
        self._on_user_account = on_user_account

    async def connect_signalr(self, contract_id: str) -> bool:
        # Crypto path uses polling; market hub is a no-op.
        logger.info(f"connect_signalr stub (Alpaca crypto polls bars) | symbol={contract_id}")
        return True

    async def connect_user_hub(self) -> bool:
        # No user hub for Alpaca crypto in this adapter; fills fire from place_order callbacks.
        logger.info("connect_user_hub stub (fills fire inline from order placement)")
        return True

    async def reconnect_user_hub(self) -> bool:
        logger.info("reconnect_user_hub stub (no-op for polling architecture)")
        return True

    async def disconnect(self) -> None:
        self.connected = False
        logger.info("Alpaca connection closed")

    # ------------------------------------------------------------------
    # Software-bracket helpers (called by bot loop on each bar)
    # ------------------------------------------------------------------

    def check_software_brackets(self, mark_price: float) -> List[dict]:
        """
        Called by the bot main loop with the latest mark. Returns a list of trigger
        events: [{"order_id": id, "type": "STOP"|"TARGET", "price": x}, ...].
        The bot is responsible for executing the resulting partial_close/flatten.
        Stop/target triggers also auto-close in this adapter via flatten path
        for safety, but only when explicitly invoked.
        """
        triggers = []
        for br in list(self._brackets.values()):
            if br.stop_price and mark_price <= br.stop_price:
                triggers.append({"order_id": br.order_id, "type": "STOP", "price": mark_price})
            elif br.target_price and mark_price >= br.target_price:
                triggers.append({"order_id": br.order_id, "type": "TARGET", "price": mark_price})
        return triggers

    def _reduce_brackets(self, size: int) -> None:
        remaining = size
        for br in list(self._brackets.values()):
            if remaining <= 0:
                break
            take = min(br.contracts, remaining)
            br.contracts -= take
            br.btc_size = br.contracts * self.contract_size_btc
            remaining -= take
            if br.contracts <= 0:
                self._brackets.pop(br.order_id, None)

    def _alloc_order_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ms_to_iso(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# Back-compat alias so existing imports of `TopstepConnection` still work.
TopstepConnection = AlpacaCryptoConnection
