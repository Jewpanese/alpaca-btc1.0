"""
ShadowVariant — read-only simulator for engine/exit-config A/B testing.

Runs alongside the live bot. Each variant has its own MarketEngineConfig and
exit ladder. On every closed bar:
  1. Receives the same OHLC + features the live bot saw.
  2. Runs its own engine.evaluate() to get a bias.
  3. Walks its own 2-bar trend-confirmation state machine.
  4. Manages a single-contract simulated position with a realistic exit ladder
     (invalidation / hard stop / 1R partial / BE stop / runner trail).
  5. Writes one JSONL record per closed simulated trade to
     `logs/shadow/variant_{name}_trades_{YYYYMMDD}.jsonl`.

Daily / weekly summary scripts aggregate these per-variant JSONLs into P&L
attribution alongside the live bot. End of week, you compare equity curves.

Design choices for V1:
  - SINGLE CONTRACT per variant trade. No T2/T3 tranche complexity. The
    comparison is about ENTRY decisions, not sizing.
  - Mirrors live exit ladder structure (invalidation, 1R partial 50%, BE stop
    after 1R, regime-conditional ATR trail, hard stop ATR safety net).
  - No VWAP veto, no LIL, no TQF in V1. Each of those can be added later if
    we need closer parity. Without them, the variant fires a strict superset
    of what live takes — useful for "did we miss anything?" analysis.
  - Variant takes a fill at the ENTRY BAR's close. Exits also evaluated at
    bar close using the bar's high/low. Single-bar approximation; close enough
    for daily/weekly P&L comparison.

NO trading action. Only writes JSONL.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date as _date
from pathlib import Path
from typing import Optional

from core.market_engine import (
    MarketDirectionEngine, MarketEngineConfig, MarketBias, ConvictionLevel
)
from core.regime_classifier import ERRegimeClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Configs
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowVariantConfig:
    """Per-variant tunables. `engine_config` is the only knob you typically
    change between variants — the exit ladder is held constant so the A/B
    isolates the entry decision."""
    name: str
    engine_config: MarketEngineConfig

    # Entry gates (mirror the live bot's DIRECT_ENTRY path conceptually)
    trend_confirm_bars: int = 2
    adx_direct_entry_min: float = 30.0

    # Exit ladder (in ATR multiples)
    invalidation_atr_mult: float = 1.5    # 1R = invalidation distance
    hard_stop_atr_mult: float = 1.2       # safety net if invalidation is far
    partial_pct: float = 0.5              # take 50% at 1R (single contract: skipped, all or nothing)
    runner_trail_atr_trend: float = 1.0
    runner_trail_atr_chop: float = 0.5
    runner_trail_atr_baseline: float = 0.75
    breakeven_buffer_pts: float = 0.25    # +1 tick above entry as BE stop after 1R

    # Sizing / pricing
    contracts: int = 1                    # V1 is single-contract
    point_value: float = 5.0              # MES = $5/pt
    tick_size: float = 0.25

    # Strategy mode — selects entry logic.
    #   "trend"      : current (TREND_DIRECT-style: 2-bar confirm + ADX≥30)
    #   "mean_revert": extended-from-VWAP + ER stall + reversal candle
    strategy: str = "trend"

    # Mean-reversion specific tunables (only used when strategy="mean_revert")
    mr_vwap_extend_atr: float = 2.0    # Min distance from VWAP (in ATR) to trigger
    mr_er_max: float = 0.30            # ER must be BELOW this — momentum stalling
    mr_min_atr: float = 0.5            # Skip when ATR is too small to be meaningful
    mr_min_vol_z: float = 1.0          # Bar volume z-score must be >= this — exhaustion needs real volume,
                                       # not a low-volume drift to the extreme. Set 0.0 to disable gate.
                                       # When vol_z is None (no baseline), gate is skipped (logs warning).


@dataclass
class SimulatedTrade:
    """One simulated trade's lifecycle state."""
    entry_ts: str                         # ISO timestamp of entry bar
    entry_bar_index: int                  # index of entry bar in session
    direction: str                        # "LONG" or "SHORT"
    entry_price: float
    atr_at_entry: float
    invalidation_price: float
    one_r_target: float
    contracts: int
    # Tracking
    best_price: float = 0.0               # highest LONG / lowest SHORT seen
    worst_price: float = 0.0
    mfe_pts: float = 0.0
    mae_pts: float = 0.0
    # State flags
    one_r_taken: bool = False
    runner_active: bool = False
    be_stop_armed: bool = False
    runner_trail_stop: float = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Variant
# ──────────────────────────────────────────────────────────────────────────

class ShadowVariant:
    """Runs one engine/exit-config combo as a pure simulator.

    Caller must invoke `on_bar()` once per CLOSED bar with the same data the
    live bot just processed. The variant owns its own state — no shared
    mutables with the live bot.
    """

    def __init__(self, config: ShadowVariantConfig, log_dir: Path, symbol: str = "BTC"):
        self.config = config
        self.symbol = symbol
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.engine = MarketDirectionEngine(config.engine_config)
        self.regime = ERRegimeClassifier()

        # Engine inputs (the variant maintains its OWN buffers — must not share with live)
        self._closes: list[float] = []
        self._adx_history: list[float] = []

        # Trend confirmation state
        self._trend_pending: Optional[str] = None     # "LONG" or "SHORT"
        self._trend_count: int = 0
        self._trend_confirmed: bool = False
        self._trend_direction: Optional[str] = None

        # Position
        self._active: Optional[SimulatedTrade] = None
        self._bar_index: int = 0

        # Per-bar memory (for reversal-candle detection in mean-revert mode)
        self._prev_close: Optional[float] = None
        self._prev_er: Optional[float] = None

        # Output paths — keyed on session date so multiple bot starts in one
        # day append to the same file
        today = _date.today().strftime("%Y%m%d")
        self._fp_trades = log_dir / f"variant_{config.name}_trades_{today}.jsonl"
        self._fp_signals = log_dir / f"variant_{config.name}_signals_{today}.jsonl"

        # Closed-trade summary kept in memory for end-of-session reporting
        self.closed_trades: list[dict] = []

        logger.info(
            f"[SHADOW VARIANT:{config.name}] enabled — "
            f"er_high_min_slope={config.engine_config.er_high_min_slope}, "
            f"trades→{self._fp_trades.name}"
        )

    # ── Public entrypoint ────────────────────────────────────────────────

    def on_bar(self,
               bar: dict,
               closes: list[float],
               adx_smooth: float,
               atr: float,
               vwap: Optional[float] = None,
               vol_z: Optional[float] = None) -> None:
        """Process one closed bar.

        bar: dict with keys 'datetime', 'open', 'high', 'low', 'close', 'volume'.
             'datetime' is an ISO string or any timestamp the caller wants
             logged. The variant reads numeric fields from the dict directly.
        closes: sequence of recent closing prices (live bot's buffer; variant
                appends its own copy for engine.evaluate). Must include the
                CURRENT bar's close as the last element.
        adx_smooth: this bar's smoothed ADX (live bot's _adx3m_smooth).
        atr: this bar's ATR.
        vwap: session VWAP. Used by mean-revert entry as the reversion target.
        vol_z: bar volume z-score vs per-minute-of-day baseline. None if no
               baseline. mean_revert variant requires vol_z >= mr_min_vol_z to
               fire (extension on real volume = exhaustion confirmation).
        """
        try:
            self._bar_index += 1
            close = float(bar.get("close", bar.get("c", 0.0)))
            high = float(bar.get("high", bar.get("h", close)))
            low = float(bar.get("low", bar.get("l", close)))
            ts = bar.get("datetime") or bar.get("timestamp") or bar.get("t") or ""

            # Maintain own engine input buffers
            self._closes.append(close)
            if len(self._closes) > 200:
                self._closes = self._closes[-200:]
            if adx_smooth and adx_smooth > 0:
                self._adx_history.append(float(adx_smooth))
                if len(self._adx_history) > 50:
                    self._adx_history = self._adx_history[-50:]

            # 1. UPDATE EXISTING POSITION FIRST (mirrors live bot's check_exits-before-add)
            if self._active is not None:
                self._update_active_trade(high, low, close, atr, ts)

            # 2. RUN ENGINE
            if len(self._closes) < 25:
                # Warmup — log nothing, no decisions
                return
            result = self.engine.evaluate(self._closes, self._adx_history)
            bar_dir = (
                "LONG" if result.bias == MarketBias.LONG else
                "SHORT" if result.bias == MarketBias.SHORT else
                None
            )

            # 3. UPDATE TREND CONFIRMATION
            self._update_trend_confirmation(bar_dir, result.probe_mult, adx_smooth)

            # 4. ENTRY CHECK — mode-conditional
            if self._active is None:
                if self.config.strategy == "trend":
                    # DIRECT_ENTRY-style gate: ADX strong + trend confirmed
                    if (self._trend_confirmed
                            and adx_smooth >= self.config.adx_direct_entry_min
                            and self._trend_direction in ("LONG", "SHORT")
                            and atr > 0):
                        self._open_trade(self._trend_direction, close, atr, ts)
                elif self.config.strategy == "mean_revert":
                    self._mean_revert_entry_check(
                        close=close, high=high, low=low,
                        atr=atr, vwap=vwap, er=result.er, ts=ts,
                        vol_z=vol_z,
                    )

            # 5. LOG SIGNAL/STATE FOR THIS BAR (lightweight)
            self._write_signal_line({
                "ts": ts,
                "bar_index": self._bar_index,
                "close": close,
                "bias": bar_dir or "FLAT",
                "conviction": result.conviction.value,
                "probe_mult": result.probe_mult,
                "er": round(result.er, 4),
                "er_prev": round(result.er_prev, 4),
                "trend_confirmed": self._trend_confirmed,
                "trend_direction": self._trend_direction,
                "adx_smooth": round(adx_smooth, 2) if adx_smooth else None,
                "atr": round(atr, 4) if atr else None,
                "vwap": round(vwap, 4) if vwap else None,
                "vol_z": round(vol_z, 3) if vol_z is not None else None,
                "in_position": self._active is not None,
                "engine_reason": result.reason,
                "strategy": self.config.strategy,
            })

            # Persist per-bar memory for next-bar comparisons
            self._prev_close = close
            self._prev_er = result.er

        except Exception as e:
            # Shadow must never crash the live bot. Log and continue.
            logger.error(
                f"[SHADOW VARIANT:{self.config.name}] on_bar error: {e}",
                exc_info=True
            )

    # ── Mean-reversion entry logic ────────────────────────────────────────

    def _mean_revert_entry_check(self,
                                  close: float,
                                  high: float,
                                  low: float,
                                  atr: float,
                                  vwap: Optional[float],
                                  er: float,
                                  ts: str,
                                  vol_z: Optional[float] = None) -> None:
        """Mean-reversion entry rule.

        Fire a SHORT when:
          - Price is >= mr_vwap_extend_atr × ATR ABOVE VWAP (extended high)
          - ER has DROPPED below mr_er_max (momentum stalling)
          - Reversal hint: this bar's close is below previous bar's close
            (price ticking back toward VWAP)
          - Bar volume z-score >= mr_min_vol_z (extension on real volume —
            distinguishes exhaustion from low-volume drift)
          - ATR is meaningful (>= mr_min_atr)

        Fire a LONG when the same conditions hold inverted (extended below
        VWAP + reversal up).

        Stop: invalidation_atr_mult ATR away — same exit ladder as trend mode.
        Target: 1R partial at +1R (same machinery). Mean-reversion typically
        targets VWAP (often <1R), so 1R partial may not fire — most exits will
        be runner trail or hard stop. We accept this for V1; if shadow data
        shows the strategy has edge, we'll add a VWAP-target exit in V2.
        """
        cfg = self.config
        if vwap is None or atr is None or atr < cfg.mr_min_atr:
            return
        if self._prev_close is None:
            return  # need a prior bar for the reversal hint

        # Need ER stalling — ER below mr_er_max (low directional efficiency)
        if er > cfg.mr_er_max:
            return

        # Volume gate — exhaustion needs real volume. If vol_z is None
        # (baseline missing), skip the gate (graceful degradation) but log
        # so we know it's running without the volume confirmation.
        if cfg.mr_min_vol_z > 0.0:
            if vol_z is None:
                # No baseline available — proceed but flag in signal log
                pass
            elif vol_z < cfg.mr_min_vol_z:
                return

        dist = close - vwap
        threshold = cfg.mr_vwap_extend_atr * atr

        # Extended ABOVE VWAP → look for SHORT
        if dist >= threshold and close < self._prev_close:
            self._open_trade("SHORT", close, atr, ts)
            return

        # Extended BELOW VWAP → look for LONG
        if dist <= -threshold and close > self._prev_close:
            self._open_trade("LONG", close, atr, ts)
            return

    # ── Trend confirmation (mirrors big_money_bot's 2-bar logic) ─────────

    def _update_trend_confirmation(self,
                                    bar_dir: Optional[str],
                                    probe_mult: float,
                                    adx_smooth: float) -> None:
        if bar_dir is None:
            # NONE bias → counter decays
            if self._trend_confirmed:
                self._trend_count = max(0, self._trend_count - 1)
                if self._trend_count <= 0:
                    self._trend_confirmed = False
                    self._trend_direction = None
            else:
                self._trend_count = 0
                self._trend_pending = None
            return

        # Instant confirmation: HIGH conviction + ADX strong
        if probe_mult >= 1.0 and adx_smooth >= self.config.adx_direct_entry_min:
            if not (self._trend_confirmed and self._trend_direction == bar_dir):
                self._trend_confirmed = True
                self._trend_direction = bar_dir
                self._trend_count = self.config.trend_confirm_bars
                self._trend_pending = bar_dir
            return

        # Same direction continuation
        if bar_dir == self._trend_pending:
            self._trend_count += 1
            if self._trend_count >= self.config.trend_confirm_bars:
                self._trend_confirmed = True
                self._trend_direction = bar_dir
        else:
            # New direction candidate — reset counter
            self._trend_pending = bar_dir
            self._trend_count = 1

    # ── Position lifecycle ────────────────────────────────────────────────

    def _open_trade(self, direction: str, price: float, atr: float, ts: str) -> None:
        invalidation_dist = atr * self.config.invalidation_atr_mult
        if direction == "LONG":
            invalidation = price - invalidation_dist
            one_r = price + invalidation_dist
        else:
            invalidation = price + invalidation_dist
            one_r = price - invalidation_dist

        self._active = SimulatedTrade(
            entry_ts=str(ts),
            entry_bar_index=self._bar_index,
            direction=direction,
            entry_price=price,
            atr_at_entry=atr,
            invalidation_price=invalidation,
            one_r_target=one_r,
            contracts=self.config.contracts,
            best_price=price,
            worst_price=price,
        )

    def _update_active_trade(self, high: float, low: float, close: float,
                              atr: float, ts: str) -> None:
        t = self._active
        if t is None:
            return

        # Update high/low water
        if t.direction == "LONG":
            t.best_price = max(t.best_price, high)
            t.worst_price = min(t.worst_price, low) if t.worst_price else low
            t.mfe_pts = t.best_price - t.entry_price
            t.mae_pts = t.entry_price - t.worst_price
        else:
            t.best_price = min(t.best_price, low) if t.best_price else low
            t.worst_price = max(t.worst_price, high)
            t.mfe_pts = t.entry_price - t.best_price
            t.mae_pts = t.worst_price - t.entry_price

        # Exit priority:
        # 1. Invalidation / hard stop / BE stop
        # 2. 1R hit (arms BE + activates trail)
        # 3. Runner trail

        # ── 1. STOPS ──
        hard_stop_dist = atr * self.config.hard_stop_atr_mult
        if t.direction == "LONG":
            invalidation_hit = low <= t.invalidation_price
            hard_stop_hit = (t.entry_price - low) >= hard_stop_dist
        else:
            invalidation_hit = high >= t.invalidation_price
            hard_stop_hit = (high - t.entry_price) >= hard_stop_dist

        if invalidation_hit or hard_stop_hit:
            exit_price = t.invalidation_price if invalidation_hit else (
                t.entry_price - hard_stop_dist if t.direction == "LONG"
                else t.entry_price + hard_stop_dist
            )
            self._close_trade(exit_price, "INVALIDATION" if invalidation_hit else "HARD_STOP", ts)
            return

        # BE stop after 1R (only if armed)
        if t.be_stop_armed:
            be_price = (t.entry_price + self.config.breakeven_buffer_pts
                        if t.direction == "LONG"
                        else t.entry_price - self.config.breakeven_buffer_pts)
            be_hit = (low <= be_price) if t.direction == "LONG" else (high >= be_price)
            if be_hit:
                self._close_trade(be_price, "BREAKEVEN", ts)
                return

        # ── 2. 1R TARGET (arms BE + trail) ──
        if not t.one_r_taken:
            one_r_hit = (
                high >= t.one_r_target if t.direction == "LONG"
                else low <= t.one_r_target
            )
            if one_r_hit:
                t.one_r_taken = True
                t.be_stop_armed = True
                t.runner_active = True
                self._update_runner_trail(t, atr)
                # Don't return — runner trail still needs evaluation this bar

        # ── 3. RUNNER TRAIL ──
        if t.runner_active:
            self._update_runner_trail(t, atr)
            trail_hit = (
                low <= t.runner_trail_stop if t.direction == "LONG"
                else high >= t.runner_trail_stop
            )
            if trail_hit:
                self._close_trade(t.runner_trail_stop, "RUNNER_TRAIL", ts)
                return

    def _update_runner_trail(self, t: SimulatedTrade, atr: float) -> None:
        # V1: use baseline trail multiplier always. Could pipe in regime later
        # for full C9 parity; held simple for now.
        trail_dist = atr * self.config.runner_trail_atr_baseline
        if t.direction == "LONG":
            new_trail = t.best_price - trail_dist
            t.runner_trail_stop = max(t.runner_trail_stop, new_trail)
        else:
            new_trail = t.best_price + trail_dist
            if t.runner_trail_stop == 0.0:
                t.runner_trail_stop = new_trail
            else:
                t.runner_trail_stop = min(t.runner_trail_stop, new_trail)

    def _close_trade(self, exit_price: float, reason: str, ts: str) -> None:
        t = self._active
        if t is None:
            return

        if t.direction == "LONG":
            pnl_pts = exit_price - t.entry_price
        else:
            pnl_pts = t.entry_price - exit_price
        pnl_dollars = pnl_pts * self.config.point_value * t.contracts

        record = {
            "variant": self.config.name,
            "entry_ts": t.entry_ts,
            "exit_ts": str(ts),
            "direction": t.direction,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "contracts": t.contracts,
            "pnl_pts": round(pnl_pts, 4),
            "pnl_dollars": round(pnl_dollars, 2),
            "exit_reason": reason,
            "atr_at_entry": round(t.atr_at_entry, 4),
            "mfe_pts": round(t.mfe_pts, 4),
            "mae_pts": round(t.mae_pts, 4),
            "bars_in_trade": self._bar_index - t.entry_bar_index,
        }
        self.closed_trades.append(record)
        self._write_trade_line(record)
        self._active = None
        # Reset trend confirmation post-exit so we need a fresh setup.
        self._trend_confirmed = False
        self._trend_direction = None
        self._trend_count = 0
        self._trend_pending = None

    # ── JSONL writers ─────────────────────────────────────────────────────

    def _write_trade_line(self, record: dict) -> None:
        try:
            with self._fp_trades.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"[SHADOW VARIANT:{self.config.name}] write trade failed: {e}")

    def _write_signal_line(self, record: dict) -> None:
        try:
            with self._fp_signals.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug(f"[SHADOW VARIANT:{self.config.name}] write signal failed: {e}")
