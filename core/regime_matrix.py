"""
Regime Matrix — composite regime labels + strategy/sizing gating.

Three regime dimensions are computed per bar and combined into a cell label:

  - ER bucket    (LOW / MID / HIGH)         from market_engine.result.er
  - VOL bucket   (QUIET / NORMAL / ELEVATED / CLIMACTIC) from VolumeBaseline.z_score
  - TOD bucket   (OPEN / MORNING / LUNCH / AFTERNOON / CLOSE / OVERNIGHT)
                 from NY-time clock minute

Cell label format: "{ER}_{VOL}_{TOD}", e.g. "HIGH_NORMAL_MORNING".

The matrix exposes two decisions per bar:

  can_fire(strategy, label) -> (bool, reason)
      Veto layer. If False, no entry of this strategy may fire in this regime.
      Reason is a short human-readable string for the log line.

  sizing_multiplier(label) -> float
      Multiplier applied to nominal contract size. 0.0 = OFF, 1.0 = nominal,
      higher = aggressive. Phase 1 deployment uses {0.0, 1.0} only —
      offensive sizing scaling deferred to Phase 4 once per-cell P&L
      attribution justifies it.

Thresholds are data-driven: load from `data/regime_thresholds.json` (built
by `research/compute_regime_thresholds.py` over 5 years of ES history).
Defaults below are conservative starting values used if file is missing.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _NY_TZ = None

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = Path("data") / "regime_thresholds.json"


# ──────────────────────────────────────────────────────────────────────────
# Threshold definitions
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeBuckets:
    """Cutoffs for each regime dimension. Defaults are conservative initial
    values; the threshold-builder script overwrites these with empirical
    quantiles from 5 years of ES history."""
    # ER buckets (range 0..1 from MarketDirectionEngine)
    er_low_max:   float = 0.20    # ER below this = LOW
    er_high_min:  float = 0.40    # ER above this = HIGH; between = MID

    # Volume z-score buckets (signed, vs per-minute-of-day baseline)
    vol_quiet_max:    float = -0.5   # below = QUIET
    vol_normal_max:   float =  1.0   # below this and above quiet = NORMAL
    vol_elevated_max: float =  2.0   # below this and above normal = ELEVATED
    # vol_z > vol_elevated_max  → CLIMACTIC

    # Time-of-day buckets (NY clock minute boundaries)
    tod_open_start:    int =  9 * 60 + 30   # 09:30
    tod_morning_start: int = 10 * 60 + 30   # 10:30
    tod_lunch_start:   int = 12 * 60         # 12:00
    tod_afternoon_start: int = 13 * 60 + 30  # 13:30
    tod_close_start:   int = 15 * 60 + 30   # 15:30
    tod_close_end:     int = 16 * 60         # 16:00 (after this = OVERNIGHT until next 09:30)

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeBuckets":
        defaults = cls()
        kwargs = {k: d.get(k, getattr(defaults, k)) for k in defaults.__dataclass_fields__}
        return cls(**kwargs)


@dataclass(frozen=True)
class RegimeLabel:
    """Composite regime label for one bar."""
    er_bucket:  str   # LOW / MID / HIGH / UNKNOWN
    vol_bucket: str   # QUIET / NORMAL / ELEVATED / CLIMACTIC / UNKNOWN
    tod_bucket: str   # OPEN / MORNING / LUNCH / AFTERNOON / CLOSE / OVERNIGHT
    er_value:   Optional[float] = None
    vol_z:      Optional[float] = None

    @property
    def cell(self) -> str:
        return f"{self.er_bucket}_{self.vol_bucket}_{self.tod_bucket}"

    def is_complete(self) -> bool:
        return ("UNKNOWN" not in (self.er_bucket, self.vol_bucket))

    def __str__(self) -> str:
        parts = [self.cell]
        if self.er_value is not None:
            parts.append(f"er={self.er_value:.2f}")
        if self.vol_z is not None:
            parts.append(f"vol_z={self.vol_z:+.2f}")
        return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Veto rules
# ──────────────────────────────────────────────────────────────────────────

# Strategy classes — used by veto rules. Strategies in the bot map to one
# of these "kinds" so the veto can be specified at a higher level than per-
# strategy. (TREND_DIRECT, TREND_FOLLOW, GRIND_EMA → TREND. MEAN_REVERT,
# CHOP_REVERSION → REVERSION. RANGE_BREAKOUT, EMA_REJECT → RANGE.)
TREND_STRATS = {
    "TREND_DIRECT", "TREND_FOLLOW", "TREND_FAST",
    "GRIND_EMA", "MOMENTUM",
}
REVERSION_STRATS = {
    "MEAN_REVERT", "CHOP_REVERSION", "MEAN_REVERT_ML",
}
RANGE_STRATS = {
    "RANGE_BREAKOUT", "EMA_REJECT", "BREAKOUT",
}


def _strategy_kind(strategy: str) -> str:
    s = strategy.upper()
    if s in TREND_STRATS:
        return "TREND"
    if s in REVERSION_STRATS:
        return "REVERSION"
    if s in RANGE_STRATS:
        return "RANGE"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────────
# Matrix
# ──────────────────────────────────────────────────────────────────────────

class RegimeMatrix:
    """Computes regime labels and decides which strategies may fire."""

    def __init__(self, buckets: Optional[RegimeBuckets] = None,
                 thresholds_meta: Optional[dict] = None):
        self.buckets = buckets or RegimeBuckets()
        self.meta = thresholds_meta or {}

    # ── Labeling ─────────────────────────────────────────────────────────

    def label(self,
              er: Optional[float],
              vol_z: Optional[float],
              ts) -> RegimeLabel:
        """Compute a composite regime label from this bar's features.

        ts: datetime, ISO string, or epoch — anything `_resolve_minute` can
        convert to NY-time minute-of-day.
        """
        return RegimeLabel(
            er_bucket=self._bucket_er(er),
            vol_bucket=self._bucket_vol(vol_z),
            tod_bucket=self._bucket_tod(ts),
            er_value=er,
            vol_z=vol_z,
        )

    def _bucket_er(self, er: Optional[float]) -> str:
        if er is None:
            return "UNKNOWN"
        b = self.buckets
        if er < b.er_low_max:
            return "LOW"
        if er >= b.er_high_min:
            return "HIGH"
        return "MID"

    def _bucket_vol(self, vol_z: Optional[float]) -> str:
        if vol_z is None:
            return "UNKNOWN"
        b = self.buckets
        if vol_z < b.vol_quiet_max:
            return "QUIET"
        if vol_z < b.vol_normal_max:
            return "NORMAL"
        if vol_z < b.vol_elevated_max:
            return "ELEVATED"
        return "CLIMACTIC"

    def _bucket_tod(self, ts) -> str:
        minute = self._resolve_minute(ts)
        if minute is None:
            return "OVERNIGHT"
        b = self.buckets
        if minute < b.tod_open_start:
            return "OVERNIGHT"
        if minute < b.tod_morning_start:
            return "OPEN"
        if minute < b.tod_lunch_start:
            return "MORNING"
        if minute < b.tod_afternoon_start:
            return "LUNCH"
        if minute < b.tod_close_start:
            return "AFTERNOON"
        if minute < b.tod_close_end:
            return "CLOSE"
        return "OVERNIGHT"

    @staticmethod
    def _resolve_minute(ts) -> Optional[int]:
        """Return NY-time minute-of-day (0..1439) or None."""
        try:
            if isinstance(ts, datetime):
                dt = ts
            elif isinstance(ts, (int, float)):
                if ts > 1e12:
                    ts = ts / 1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                iso = ts.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(iso)
                except ValueError:
                    dt = datetime.strptime(ts, "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)
            else:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ny = dt.astimezone(_NY_TZ) if _NY_TZ else dt
            return ny.hour * 60 + ny.minute
        except Exception:
            return None

    # ── Decisions ────────────────────────────────────────────────────────

    # Threshold for treating a reversal_score as "strong" — calibrated to
    # match the reversal_score combiner's STRONG band (>= 0.65).
    REVERSAL_STRONG_THRESHOLD = 0.65

    def can_fire(self,
                 strategy: str,
                 label: RegimeLabel,
                 reversal_score: float = 0.0,
                 reversal_direction: str = "NONE",
                 trade_direction: str = "NONE") -> tuple[bool, str]:
        """Decision layer. Returns (allowed, reason).

        Inputs:
            strategy            — strategy_name (mapped to kind via _strategy_kind)
            label               — current regime cell label
            reversal_score      — composite reversal score [0..1] from
                                  core.reversal_score (0.0 if not wired)
            reversal_direction  — "LONG" / "SHORT" / "NONE" — direction the
                                  reversal_score points
            trade_direction     — "LONG" / "SHORT" / "NONE" — direction the
                                  proposed trade would go

        Rules in priority order:
          1. Reversal overrides (NEW 2026-05-07) — strong reversal can:
                a) BLOCK trend entries against the reversal direction
                b) ALLOW reversion entries in cells that would normally
                   block (the reversal score IS the edge)
          2. SWAMP — block all in low-ER/quiet-vol cells
          3. CLIFF — block trend in climactic-vol cells (reversion allowed)
          4. OVERNIGHT_NO_EDGE — block trend in calibrated no-edge overnight
                                 cells (per 5yr ES historical attribution)
        """
        # If we don't have full regime data, be permissive — let the
        # existing entry gates do their job.
        if not label.is_complete():
            return True, f"PARTIAL — {label.cell}"

        kind = _strategy_kind(strategy)

        # ─── 1. REVERSAL OVERRIDES (added 2026-05-07) ─────────────────────
        # Strong reversal_score (>= 0.65) modulates entry decisions:
        #
        #   a) Counter-trend block: if a strong reversal points opposite
        #      to a proposed trend trade, block it. We're not fighting
        #      a confirmed reversal — even in HIGHWAY cells.
        #
        #   b) Reversion override: if a strong reversal matches the
        #      proposed reversion trade's direction, allow it even in
        #      cells that would normally block (SWAMP, OVERNIGHT). The
        #      reversal score IS the edge in this cell — bypass the
        #      population-level "no edge" verdict.
        #
        # These rules require BOTH score and direction to be informative
        # (reversal_direction in {LONG, SHORT}). When the combiner returns
        # NONE direction, no override fires and the cell rules apply.
        if (reversal_score >= self.REVERSAL_STRONG_THRESHOLD
                and reversal_direction in ("LONG", "SHORT")):
            # 1a) Counter-trend block
            if (kind == "TREND"
                    and trade_direction in ("LONG", "SHORT")
                    and reversal_direction != trade_direction):
                return False, (
                    f"REVERSAL_BLOCK — score={reversal_score:.2f} "
                    f"reversal={reversal_direction} opposes trend "
                    f"{trade_direction} ({label.cell})"
                )
            # 1b) Reversion override
            if (kind == "REVERSION"
                    and trade_direction in ("LONG", "SHORT")
                    and reversal_direction == trade_direction):
                return True, (
                    f"REVERSAL_OVERRIDE — score={reversal_score:.2f} "
                    f"reversal={reversal_direction} ({label.cell})"
                )

        # ─── 2. SWAMP ─────────────────────────────────────────────────────
        # Low ER + quiet vol means the market isn't efficient AND isn't
        # expanding. Trends won't run, mean-reversion targets unreliable.
        # (Reversal override above bypasses this for high-conviction reversion.)
        if label.er_bucket == "LOW" and label.vol_bucket == "QUIET":
            return False, f"SWAMP — {label.cell} (low ER + quiet vol, no edge)"

        # ─── 3. CLIFF ─────────────────────────────────────────────────────
        # Climactic vol = terminal-extension behavior. Trend entries here
        # catch the exhaustion bar. Block trend, allow reversion.
        if label.vol_bucket == "CLIMACTIC":
            if kind == "TREND":
                return False, f"CLIFF — {label.cell} (climactic vol, trend likely exhausting)"
            # Reversion strategies allowed at CLIFF — that's their cell.

        # ─── 4. OVERNIGHT TREND BLOCK ─────────────────────────────────────
        # Calibrated 2026-05-02 on 647k 3-min bars (see
        # data/regime_calibration_report.json):
        #
        #   - LOW_ER  + any vol  → block (n≈195k, mean_fwd ≈ −0.4 ATR pulled
        #                          by occasional flash-crash tails)
        #   - MID_ER  + QUIET    → block (n=54k, mean_fwd=+0.05 ATR, no edge)
        #   - MID_ER  + NORMAL   → block (n=87k, mean_fwd=+0.03 ATR, no edge)
        #   - MID_ER  + ELEVATED → ALLOW (real overnight expansion)
        #   - HIGH_ER + any vol  → ALLOW ("alpha path")
        #
        # Reversion strategies allowed in any overnight cell.
        if label.tod_bucket == "OVERNIGHT" and kind == "TREND":
            if label.er_bucket == "LOW":
                return False, (
                    f"OVERNIGHT_LOW_ER — {label.cell} "
                    "(low ER overnight, trend conviction weak)"
                )
            if (label.er_bucket == "MID"
                    and label.vol_bucket in ("QUIET", "NORMAL")):
                return False, (
                    f"OVERNIGHT_NO_EDGE — {label.cell} "
                    "(5yr data: mean_fwd ~0.04 ATR, no directional edge)"
                )

        return True, f"OK — {label.cell}"

    def sizing_multiplier(self, label: RegimeLabel) -> float:
        """Per-cell sizing override. Phase 1 returns 0.0 (block) or 1.0
        (nominal). Aggressive sizing (>1.0) is gated on per-cell P&L
        attribution data — deferred to Phase 4.
        """
        # The veto already handles the OFF cells. For everything else
        # return nominal sizing. Future expansion: scale up in HIGHWAY,
        # scale down in BATTLE based on per-cell attribution.
        if not label.is_complete():
            return 1.0
        if label.er_bucket == "LOW" and label.vol_bucket == "QUIET":
            return 0.0
        return 1.0

    # ── Persistence ──────────────────────────────────────────────────────

    @classmethod
    def from_thresholds_file(cls, path: Optional[Path] = None) -> "RegimeMatrix":
        target = Path(path) if path else DEFAULT_THRESHOLDS_PATH
        if not target.exists():
            logger.warning(
                f"[REGIME] thresholds file missing at {target} — using defaults"
            )
            return cls()
        try:
            payload = json.loads(target.read_text())
        except Exception as e:
            logger.warning(f"[REGIME] failed to read {target}: {e} — using defaults")
            return cls()
        buckets = RegimeBuckets.from_dict(payload.get("buckets", {}))
        meta = {k: v for k, v in payload.items() if k != "buckets"}
        logger.info(
            f"[REGIME] loaded thresholds from {target.name} "
            f"(built {meta.get('built_at', '?')}, n_bars={meta.get('n_bars', '?')})"
        )
        return cls(buckets=buckets, thresholds_meta=meta)

    def save_thresholds(self, path: Optional[Path] = None,
                        extra_meta: Optional[dict] = None) -> Path:
        target = Path(path) if path else DEFAULT_THRESHOLDS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **(self.meta or {}),
            **(extra_meta or {}),
            "buckets": asdict(self.buckets),
        }
        target.write_text(json.dumps(payload, indent=2))
        return target
