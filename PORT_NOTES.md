# Alpaca BTC Bot — Topstep5 Port Status

This repo is now a port of `c:/Development/topstep5/` re-targeted at Alpaca crypto (BTC/USD spot)
instead of TopstepX futures (MES). The architecture is identical; the broker layer is swapped.

---

## Phase A — Foundation (DONE)

What was changed in this pass:

1. **Backed up the original lean ccxt bot** — moved to [`_legacy/`](_legacy/) so nothing was lost
   (old `live/bot.py`, `core/connection.py`, `risk/manager.py`, `strategies/*.py`, etc.).
2. **Copied the topstep5 module tree** (~6,000 lines) into [`core/`](core/), [`risk/`](risk/),
   [`strategies/`](strategies/), and [`live/`](live/). All `__init__.py` files stubbed so the
   package is importable.
3. **Replaced [`core/connection.py`](core/connection.py)** with `AlpacaCryptoConnection` — a
   drop-in adapter that exposes the same public API as `TopstepConnection` (42 methods) but
   talks to Alpaca paper via `ccxt`. Backwards-compatible alias: `TopstepConnection =
   AlpacaCryptoConnection`, so every `from core.connection import TopstepConnection` keeps
   working unchanged.
4. **Rewrote [`config/settings.py`](config/settings.py)** with BTC-specific
   `InstrumentConfig`, `AccountConfig` ($100k account, $5k hard / $2.5k soft daily loss,
   $100/day target), and a `TopstepConfig` shim that carries Alpaca creds under the
   legacy field names (`username` = Alpaca API key, `api_key` = Alpaca secret).
5. **New entry point [`run_btc.py`](run_btc.py)** — replaces `run_big_money.py`. No TopstepX
   account selection prompt; reads creds from `.env`.
6. **Updated [`requirements.txt`](requirements.txt)** (added `ccxt`, `python-dotenv`,
   `signalrcore`, `websockets`).

### Sizing convention adopted

| MES                                | BTC (this port)                       |
|------------------------------------|---------------------------------------|
| 1 contract = 1 MES                 | **1 contract = 0.01 BTC**             |
| `point_value` = $5/pt              | **`point_value` = $0.01/pt**          |
| `tick_size` = 0.25                 | **`tick_size` = 0.01 (USD)**          |
| points = price units (1 pt = 1 ES) | **points = USD price moves of BTC**   |
| `contract_id` = "CON.F.US.MES.M26" | **`contract_id` = "BTC/USD"**         |
| Native bracket orders              | **Software-tracked SL/TP**            |
| SignalR (market + user hubs)       | **Polling (no-op stub for hubs)**     |

The math `pnl_dollars = pnl_pts * point_value * contracts` works because:
- BTC at $100k, 10 contracts = 0.10 BTC = $10k notional
- A $1,000 BTC move with 10 contracts → 1000 × 0.01 × 10 = **$100 PnL** ✓

### Smoke test (passed)

```
$ python -c "from live.big_money_bot import BigMoneyBot; \
              from config.settings import Config; \
              bot = BigMoneyBot(Config.load()); \
              print(type(bot.conn).__name__)"
[VOL BASELINE] file missing — vol_z will be None.
[REGIME] thresholds file missing at data\regime_thresholds.json — using defaults
ML Signal Provider not available: No config.json in models/production
AlpacaCryptoConnection
```

The 3 warnings are **expected and benign** — those are the ES-calibrated data files
(5-year ES vol baseline, ES regime quantiles, ES-trained XGBoost models). They'll be
rebuilt for BTC in Phase B.

---

## Phase B — Wiring + Tuning (COMPLETE — ready to test)

### What landed in Phase B

1. **B2: Polling-only mode wired end-to-end.**
   - `run_btc.py` sets `bot._poll_only = True` before `bot.start()`.
   - `live/hybrid_bot.py:_main_loop` patched to honor `_poll_only`: forces the
     `_poll_bars()` path even when `is_connected=True`. Throttled to **15-second
     poll interval** (`_poll_bars_interval = 15.0`) — well under Alpaca's 200/min
     rate limit, with room for `get_open_positions`, balance checks, etc.
   - Adapter exposes the attrs the main loop reads: `is_connected` (property
     mirroring `connected`), `_user_ws_connected = True`, `_needs_reconnect`,
     `_reconnect_delays`, `_reconnect_attempt`.

2. **B4: Session-end gates disabled for 24/7 crypto.**
   - `run_btc.py` sets `bot._crypto_24_7 = True` and pre-fires
     `_bm_session_end_shutdown_fired = True`.
   - `live/big_money_bot.py:_process_bar_impl` wraps the entire 13:55 blackout +
     14:15 self-shutdown block in `if not getattr(self, "_crypto_24_7", False)`.
     Daily ET-time blackout + auto-shutdown will not fire.

3. **B5: MLL math retuned for $100k crypto account.**
   - `risk/topstep_mll.py:get_max_contracts` graduated tiers rewritten:
     `0/5/10/20/30/40/50` BTC contracts (was `0/2/5/7/9/11/13` MES).
   - `get_daily_loss_limit` cap raised from $2,000 → **$5,000** to match the
     user's 5%-of-account hard stop.
   - `AccountConfig.starting_balance=$100k`, `mll_threshold=$95k`,
     `daily_loss_hard_stop=$5k`, `daily_loss_soft_stop=$2.5k`,
     `daily_profit_target=$100`.

4. **B6: BigMoneyConfig + VolSizingConfig + LIL all BTC-scaled.**
   - `BigMoneyConfig.cushion_tiers` now BTC-sized: `5/10/20/30/40/50` contracts.
   - `BigMoneyConfig.daily_loss_limit = -$5000` (was `-$1500`).
   - `BigMoneyConfig.min_atr_to_trade = $50` (was `1.0` MES pt).
   - `BigMoneyConfig.range_mapper_max_stop_pts = $1500` (was `10.0` MES pts).
   - `VolSizingConfig.min_stop_pts = $50`, `max_stop_pts = $2000` in BigMoneyBot
     override (was `2.0` / `20.0` MES).
   - `LILConfig` switched from hardcoded `point_value=5.0, tick_size=0.25` to
     `self.instrument.point_value/tick_size` (so it picks up `0.01/0.01`).
   - LIL absolute thresholds rescaled for BTC: `graduation_min_pts=300`,
     `graduation_strong_pts=500`, `stale_mfe_threshold_pts=50`,
     `loss_to_peak_min_peak=50`, `giveback_min_peak=100`,
     `velocity_mfe_threshold=100`. ATR-relative thresholds unchanged.

### Live smoke test — 2026-05-09 21:00

```
> python run_btc.py
[STARTUP] BTC Big Money Bot | symbol=BTC/USD | contract_size=0.01 BTC | poll_only=True
Alpaca authenticated (paper) | key=PK2***JRU | symbol=BTC/USD
Account balance: $99,857.85
MLL: $95,000.00 | Cushion: $4,857.85 | TRAILING | Max contracts: 10
Loaded 59 historical bars
Status server running at http://localhost:8585
Bot is LIVE. Strategies active: VWAP_REVERT, MOMENTUM, BB_BOUNCE, ...
Instrument: BTC (BTC/USD) | $0.01/point | native ES
Risk: max 50 BTC contracts, $2429 daily loss limit (dynamic)
[poll fires every 15s — confirmed in instrumented run]
```

No errors. Polling fires on schedule. Bot is live against Alpaca paper.

### Tonight's test plan

```bash
# 1. Activate the venv (if you have one) and confirm deps
pip install -r requirements.txt

# 2. Confirm .env has Alpaca paper creds
#    ALPACA_PAPER_API_KEY=...
#    ALPACA_PAPER_SECRET_KEY=...
#    USE_PAPER_TRADING=True

# 3. Start the bot
python run_btc.py
```

Watch for, in order:
1. **Authentication** — `Alpaca authenticated (paper) | key=PK***`
2. **Balance + MLL** — `Account balance: $...` and `Max contracts: N`
3. **Historical bars** — `Loaded N historical bars` (~60 for 3-min × 3 hours)
4. **Status dashboard** — `http://localhost:8585`
5. **Bot LIVE** — strategy list prints
6. **Warmup** — needs **2 fresh 3-min candle closes** (~6 min) before any entry
7. **Bar processing** — silent unless a signal triggers; periodic MLL log every 60s

If a signal qualifies, you'll see `[ENTRY]`, `[ORDER FILLED]`, `Bracket tracked: ...`
and Alpaca paper will show a BTC/USD position.

### Known gaps to revisit (lower priority)

- **`date.today()` uses local time** in `risk/manager.py:reset_daily` and
  `risk/vol_sizing.py:reset_daily`. For BTC 24/7 should be UTC. Daily PnL will
  reset at local midnight rather than 00:00 UTC — fine for solo running, fix later.
- **`vwap_revert.py:48`** still blocks during OVERNIGHT/LONDON sessions — minor
  since the BigMoney trend logic is the primary entry path, but consider
  unblocking for crypto.
- **`Topstep5 HYBRID — Institutional Exit System`** banner still says Topstep —
  cosmetic.
- **MLL log says `Max=10 ES`** — the topstep_mll display path uses ES naming
  even when instrument is "BTC". Cosmetic.
- **`data/volume_baseline.json` and `data/regime_thresholds.json` are missing.**
  Bot gracefully degrades. Optional Phase C: rebuild on BTC history via
  `research/build_volume_baseline.py` (port required).

## Sizing Refactor 2026-05-10 — spot %-of-account sizing

Replaced the futures-prop sizing approach (fixed-$ risk + cushion-tier max contracts) with **standard quant fixed-fractional sizing** appropriate for BTC spot.

### The conceptual switch

| | Before (futures-prop) | After (spot %) |
|---|---|---|
| Per-trade risk | Fixed `$500` | `balance × risk_per_trade_pct` (default 2%) |
| Max contracts | Cushion-tier table (`$5k → 20 ct`) | `account × notional_cap_pct / price` |
| Drawdown deleverage | Smooth curve from `daily_pnl` | Discrete tiers: 0%/2%/5%/10%/15% DD → 1.00/0.75/0.50/0.25/0.00 mult |

The key insight: BTC spot is dollar-for-dollar. 0.1 BTC = $8K notional = up to $8K loss. Notional and risk are coupled, not decoupled like in futures. Cushion tiers don't make sense; % of account at risk does.

### Code changes

- **[`config/settings.py`](config/settings.py)** — added `risk_per_trade_pct=0.02`, `risk_per_trade_pct_max=0.05`, `notional_cap_pct=1.0`, and `drawdown_size_tiers` list to `TradingConfig`.
- **[`risk/vol_sizing.py`](risk/vol_sizing.py)** — new `_calc_contracts_pct()` method. `VolSizingConfig` got mirror fields + a `use_pct_sizing: bool = False` flag. The legacy fixed-$ path is preserved unchanged for futures portability.
- **[`live/big_money_bot.py`](live/big_money_bot.py)** — `VolatilitySizer` now instantiated with `use_pct_sizing=True` and passes `risk_per_trade_pct` etc. from `TradingConfig`.
- **[`live/hybrid_bot.py`](live/hybrid_bot.py)** — `start()` and `_check_balance()` now push `account_balance` and `sod_balance` into `vol_sizer` so `_calc_contracts_pct` can compute current risk dollars.

### Math sanity check

At $100k balance, 2% risk target, $80k BTC, ATR=$300, stop=$780:
- ideal contracts = `$2000 / ($780 × 0.01) = 256`
- notional cap = `$100k × 1.0 / ($80k × 0.01) = 125 ct`
- max_contracts cap = 50 (current safety ceiling)
- result: **50 ct = 0.5 BTC = $40K notional, $390 actual risk**

The hard cap binds first; realized risk ~0.4% (below 2% target). This is conservative-by-design; raise `TradingConfig.max_contracts` to ~150 once the bot has earned trust.

### DD tier verification

```
Test: 15% DD scenario
[VOL_SIZING] PCT_SIZE STOPPED: daily DD 15.0% ≥ max-DD tier — no trading
```

### Tranches stayed orthogonal

The 60/40/30 tranche split (risk-reduce / core / runner) decides **how to exit** the total position, not how to size it. Tranches still allocate from the total contracts the sizer returns. No changes there.

---

## Full Audit 2026-05-10 — ES/MES → BTC system-wide cleanup

After the morning data-feed and ATR-cap fixes, did a full audit pass to remove
**every** remaining ES/MES assumption from active code paths. Three parallel
discovery agents surfaced ~30 issues across 15 files; all addressed in this pass.

### Critical fixes (were actively blocking trades)

1. **`risk/manager.py` time-gating** — two checks were silently blocking trades
   for 60 minutes/day:
   - **Close cutoff** at 4:00–4:30 PM ET (Topstep forced-flatten window).
   - **Session-open buffer** at 6:00–6:30 PM ET (futures session start).
   Both wrapped in `if not getattr(self, "_crypto_24_7", False)`. `run_btc.py` now
   sets `bot.risk._crypto_24_7 = True` and forces `session_open_buffer_minutes=0`.

2. **`live/hybrid_bot.py:MAX_BRACKET_SL_PTS`** was `min(max(atr*1.5, 4.0), 6.0)`.
   On BTC where `atr*1.5` is in the $50–1500 range, the `6.0` ceiling clamped
   every bracket stop to $6 — guaranteed instant stop-out on every entry.
   Now `min(max(atr*1.5, 200.0), 2000.0)`.

3. **`risk/trailing_stop.py`** — entire file was ES-scale (point_value=**50.0**,
   _pts in 1–5 range). Used by HybridBot when a "single/tiny contract" position
   is taken; would have driven BTC math into the floor. Full BTC-scale rewrite:
   point_value=0.01, max_stop_pts=$800, min_stop_pts=$50,
   breakeven_trigger=$200, etc.

### Module-level defaults migrated to BTC scale

Each of these had MES/ES-scale defaults that BigMoneyBot overrides at runtime,
but any direct instantiation of the config (e.g. tests, alternate bots) would
silently misbehave. All defaults now BTC-scaled:

- `risk/vol_sizing.py:VolSizingConfig` — min/max stop pts (1.5/20 → 50/2000)
- `risk/hybrid_position_manager.py:HybridConfig` — full rewrite of all `_pts`
  defaults; `point_value` 5.0 → 0.01; `instrument` "MES" → "BTC"
- `risk/loser_intelligence.py:LILConfig` — point_value 5.0 → 0.01;
  graduation/stale/giveback/loss_to_peak/velocity `_pts` defaults bumped
  100-200× to BTC scale
- `risk/adaptive_exits.py:AdaptiveExitConfig` — point_value 5.0 → 0.01,
  instrument "MES" → "BTC" (logic is mostly ATR-relative, so most fields
  unchanged)
- `risk/topstep_mll.py:TopstepAccountConfig` — full migration: $150k Topstep →
  $100k BTC paper; instrument "MES" → "BTC"; `mes_per_es` 10 → 1; display
  branch added for `instrument == "BTC"` (now logs `Max=10 ct (0.10 BTC)`)
- `risk/position_manager.py` — added `TRANCHE_ALLOCATION_BTC` table, default
  instrument param "MES" → "BTC", BTC docstring example
- `core/adx_range_mapper.py:ADXRangeConfig` — min_range $3 → $50, max_range
  $30 → $2000, breakout_buffer $1 → $30, failed_breakout_reentry $2 → $100,
  touch_proximity $2 → $50
- `core/range_detector.py:RangeConfig` — tight_chop $10 → $300, wide_range
  $10–20 → $300–1500, edge_proximity $4 → $100, sr_cluster $3 → $100
- `core/bar_recorder.py:BarRecorder` — default instrument "MES" → "BTC"
- `core/ml_signal.py:warm_up_from_file` — accepts `symbol_glob` parameter (was
  hardcoded `"ES *.Last.txt"`); ES is still the default since this path is
  only used when an ES-trained model is present

### Display fixes (cosmetic but visible)

- `live/hybrid_bot.py:293` instrument banner — "10 MES = 1 ES" / "native ES"
  now branches on instrument name and prints "1 contract = 0.01 BTC" for BTC.
- `live/status_server.py:110` dashboard — "ES Price" → "{instrument} Price",
  with status payload exposing `instrument` field.
- `risk/topstep_mll.py:_log_mll_status` — added `instrument == "BTC"` branch:
  `Max=10 ct (0.10 BTC)` (was `Max=10 ES`).
- `live/big_money_bot.py` cushion tier log — was hardcoded "MES" suffix; now
  uses `self.instrument.instrument` so it prints "BTC".
- `live/big_money_bot.py` shadow scorer + 3 shadow variants now pass
  `symbol=self.instrument.instrument` (was hardcoded `"MES"`).
- `live/shadow_scorer.py` and `live/shadow_variant.py` module defaults
  "MES" → "BTC" (belt + suspenders).

### Verified clean (live log scan, 2026-05-10 07:58 UTC)

```
07:58:03  Cushion Tiers: ['$20,000→50BTC 2.5x/3.5x', '$15,000→40BTC ...']
07:58:03  [MLL] ... | Max=10 ct (0.10 BTC)
```

No `ATR clamp`, `close cutoff`, `session open buffer`, "ES Price", or
"native ES" messages anywhere in the startup or first-minute output.

### Files explicitly NOT modified

- `_legacy/` — original lean ccxt bot, kept as backup, NOT in import path.
- Module imports of `mes_per_es` / `to_mes` / `es_equivalent` — kept for API
  compat with the futures bot. For BTC, `mes_per_es=1` so the math is a no-op.
- Historical context comments (e.g. `# was 1.5 for MES`) left intact — they're
  educational and help future audits.

---

## Bugfix 2026-05-10 (2/2) — ES-tuned absolute thresholds were silently breaking BTC math

After fixing the data feed, the live log immediately exposed three ES-scale numeric
constants firing every 30-60 seconds and corrupting downstream calcs. These weren't
caught in Phase B because they only appear once the bot has live data flowing.

**1. `core/features.py:MAX_ATR = 15.0` → 2500.0**

Caps the ATR value so a flash-spike doesn't blow up position sizing.
- ES: ATR is 3-7 pts; >15 = NFP / FOMC spike. $15 cap is correct.
- BTC at $80k: ATR is **$50-300 in calm**, **$300-1000 in active regimes**. The
  $15 cap was firing on every bar and clamping ATR to $15, which then drove
  `vol_sizer`, `trade_quality.location_atr_mult`, exhaustion detection, LIL
  thresholds, and tranche stop math to use a fake $15 ATR.
- New cap: **$2500** — catches a true 5-sigma BTC event without false-positiving on
  normal volatility.

Live evidence (07:28-07:30 UTC, 2026-05-10):
```
ATR clamped: 53.48 → 15.00 (news spike protection)
ATR clamped: 53.73 → 15.00 (news spike protection)
ATR clamped: 57.47 → 15.00 (news spike protection)
```
A $53 ATR over a 3-min bar is **calm BTC volatility**, not a news spike.

**2. `core/adx_range_mapper.py:max_range_width_pts = 30.0` → BTC override 2000.0**

Default in the module file remains 30.0 for the futures use case. The BTC override is in
`live/big_money_bot.py` where the mapper is instantiated:
- `min_range_width_pts: 3.0 → 50.0` (BTC equivalent of "3 pts wide")
- `max_range_width_pts: 50.0 → 2000.0` (a $2000 range is still a tradeable consolidation
  on BTC — beyond that it's chop, not a range)
- `breakout_buffer_pts: 1.0 → 30.0` (need $30 above the box edge to count as a
  breakout, not noise)

**3. `core/features.py:atr()` docstring updated** — was still claiming "$15.0 cap".

### Lesson

The ES bot has ~6,000 lines and hardcoded MES scale assumptions are sprinkled
throughout: not just in the configs we already migrated (BigMoneyConfig, VolSizingConfig,
LILConfig, MLLConfig) but also in **module-level constants** and **inline overrides in
specific bot subclasses**. There's no clean grep for "ES-tuned" — they look like
ordinary numbers. The only reliable way to find them is to **let live data run through
the bot and watch for clamp-warnings, regime mis-classifications, or "trade quality
blocked" log lines** that don't make sense given the chart.

**Audit posture going forward:** each session, scan the log for `clamped`, `blocked`,
`reject`, `skipped`, or `out of range`. Each one is a candidate for an ES-scale
constant that needs BTC tuning.

---

## Bugfix 2026-05-10 (1/2) — `get_bars` returned stale data, bot was data-blind

**Symptom:** ran the bot 21:27 → 07:07 (10 hours). Zero trades. Zero entry-attempt
log lines. Zero "fresh candle" detections. Zero CSV files written by BarRecorder.
Only the 60-second MLL heartbeat appeared in the log.

**Root cause:** ccxt's Alpaca adapter has two quirks I didn't account for:

1. `fetch_ohlcv(symbol, tf, since=<ms>, limit=N)` **silently returns 0 bars** for
   small windows. Verified: `since=now-15min, limit=5` → 0 bars. Even `since=now-1h,
   limit=20` → 0 bars.
2. Without `since`, `fetch_ohlcv` pages FORWARD from a fixed anchor (today's UTC
   midnight). `limit=5` returns the **oldest 5 bars of the day**, not the newest 5.
   `limit=500` walks all the way to "now" and returns ~266 bars.

So my old adapter, which always passed `since=<computed>`, returned 0 bars on every
live poll. The bot's bar-dedup code (`is_new_candle = bar_ts != last_bar_timestamp`)
saw the same stale tuple repeatedly → never advanced `_live_bar_count` → warmup
gate never released → no signals → no trades.

**Fix** (`core/connection.py:get_bars`):
- Stop passing `since` to `fetch_ohlcv` — it's broken for small windows on this venue.
- Always fetch a generous batch (`fetch_limit = max(bars_back * 4, 500)`).
- Sort + tail-slice to `bars_back` to return the most recent bars.

Verified post-fix: `get_bars(bars_back=5)` now returns bars ending ~3 minutes behind
real time (the most-recently-closed 3-minute bar).

**Lesson:** test the adapter against the live broker for the EXACT call shapes the
inherited bot uses (small `bars_back`, repeated polls), not just the smoke-test shape
(big startup load). One-shot adapter tests can hide this kind of pagination quirk.

---

## Phase C — End-of-Day Claude Analyzer (DONE)

### What landed in Phase C

A BTC-native end-of-day review powered by **Claude Opus 4.7** with prompt caching:

1. **`core/trade_journal.py`** — append-only JSONL store for closed trades.
   Files at `data/journal/btc_trades_<YYYY-MM-DD>.jsonl` (one trade per line).
   Functions: `append_trade`, `read_trades`, `aggregate` (with by-regime / by-session
   / by-strategy breakdowns).

2. **`risk/manager.py:record_trade()` patched** — every trade `RiskManager` accepts is
   now also persisted to the journal. Best-effort, can never raise into the trading
   loop. Captures direction, contracts, btc_size, entry/exit/pnl, exit_reason,
   regime, session, strategy, duration, MFE/MAE.

3. **`analyzers/eod_summary.py`** — the analyzer:
   - Model: **`claude-opus-4-7`**.
   - **Adaptive thinking** with `display: "summarized"` (Opus 4.7 omits thinking
     content by default; we opt back in).
   - **`output_config={"effort": "high"}`** — meaningful for 4.7.
   - **Streaming** via `client.messages.stream()` + `.get_final_message()` — large
     `max_tokens` without HTTP timeout risk.
   - **Prompt caching** via `cache_control: {"type": "ephemeral"}` on the last
     stable system block. The role + discipline + report-template prompts are
     long enough to cross the 4,096-token Opus 4.7 cache floor when combined.
   - No `temperature` / `top_p` / `top_k` (those return 400 on Opus 4.7).
   - Outputs to `analyzers/reports/btc_daily_<date>.md` and `.json`.

4. **`run_eod_summary.py`** — entry point. `python run_eod_summary.py [YYYY-MM-DD]`.

5. **`requirements.txt`** — added `anthropic>=0.92.0`.

### Setup before first run

Add to your `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then:

```bash
pip install anthropic>=0.92.0
python run_eod_summary.py            # today
python run_eod_summary.py 2026-05-09 # specific date
python run_eod_summary.py --no-save  # don't persist
```

The bot already populates the trade journal automatically — no further wiring needed.

### Smoke test status (2026-05-09)

- Imports clean.
- Prompt assembly: 3 system blocks, last has `cache_control`, user payload assembled.
- No-trades path: produces a stub report without calling Claude.
- Live Claude call: request reaches Anthropic with the correct model + parameters
  (verified by a clean-shape 400 response — billing error, not API misuse).

### Why this is "minimal but compounds"

- **No TopstepX assumptions** — chart_capture / morning_report from topstep5 are
  built around the TopstepX desktop app and the equity-futures economic calendar.
  This analyzer reads only what the BTC bot itself produces (journal + MLL + bars).
- **One LLM call per day** — designed for a scheduled cron / Task Scheduler
  trigger after UTC midnight. Cost will be dominated by the prompt-cache write on
  day 1; subsequent days hit the cache for ~10× cost reduction.
- **Compounds via the journal** — every trade lands in the JSONL, so a future
  weekly / monthly report (or the LLM-agent vision from earlier) just reads more
  history. No retrofitting needed.

### Phase D candidates (future)

- Hook a scheduler (Windows Task Scheduler or `cron`) to fire `run_eod_summary.py`
  at 00:05 UTC daily.
- Weekly summary that aggregates 7 days of trade journals + 7 daily reports.
- Roll the EOD reports into a lightweight `learnings.md` that the bot reads at
  startup as additional context.
- Token-counting harness to verify cache reads on day 2 (`usage.cache_read_input_tokens`
  should be > 80% of cached tokens after the first call).

---

### Original Phase B punch-list (kept for reference)

The following items must be addressed before paper-trading goes live.

### B1. Live broker test (highest priority)
- [ ] Run `python run_btc.py` against the actual Alpaca paper account. The current smoke test
      only confirms instantiation, not real API calls. Expect issues around:
  - bar timestamp formats (`get_bars` returns ISO strings via `_ms_to_iso` — verify the
    inherited code parses them correctly; topstep5 may expect epoch ms)
  - `get_open_positions` shape (current adapter returns `netSize`/`size`/`btcSize` —
    verify the inherited reconciliation logic in [`live/hybrid_bot.py`](live/hybrid_bot.py) accepts it)
  - the Topstep `_headers()` call in [`hybrid_bot.py:1001`](live/hybrid_bot.py#L1001)
    — adapter logs a warning and returns `{}` so any path that hits it will fail loudly

### B2. Polling-only main loop
- [ ] In [`hybrid_bot.py`](live/hybrid_bot.py) the main loop drives off SignalR market-hub
      `on_quote` callbacks. Our adapter's `connect_signalr` is a no-op. The bot must
      instead poll bars on a timer. Inspect `_main_loop` and add a polling mode flag
      (`self.poll_only = True`) that bypasses the WS path. Topstep already has a
      `--poll-only` arg in `run.py`; we may need to enable it by default for crypto.

### B3. Disable ES-calibration cleanly
- [ ] [`core/volume_baseline.py`](core/volume_baseline.py) — file-missing path is fine,
      but the `vol_z = None` branch downstream needs review.
- [ ] [`core/regime_matrix.py`](core/regime_matrix.py) — same as above; defaults used.
- [ ] [`core/ml_signal.py`](core/ml_signal.py) — already disabled when `models/production/`
      is absent. Leave disabled unless we train BTC-specific models later.

### B4. Session times for 24/7 crypto
- [ ] [`core/market_data.py`](core/market_data.py) — `detect_session()` and
      `minutes_since_rth_open()` assume ES RTH (9:30-16:00 ET). For crypto, replace with
      either continuous-session ("CRYPTO") or define UTC-based bands
      (US_HOURS / EU_HOURS / ASIA_HOURS).
- [ ] [`live/big_money_bot.py`](live/big_money_bot.py) — session-end gates at
      13:55 / 14:10 / 14:15 ET (lines ~1721-1732) **must be disabled or set to never fire**
      for 24/7 crypto. Search for `_bm_session_ending` and `bm_config.session_end_*`.
- [ ] [`risk/manager.py`](risk/manager.py) — daily reset is on calendar day; verify it's
      using UTC, not ET (crypto rolls at 00:00 UTC, not 17:00 ET like futures).

### B5. Topstep MLL → crypto drawdown rule
- [ ] [`risk/topstep_mll.py`](risk/topstep_mll.py) implements Topstep's trailing MLL
      ($150K start → $145.5K floor → tracks high water mark up to $150K cap). The math
      is wrong for a $100k crypto account. Either:
      - **Simple replacement:** flat `min_balance = starting_balance * 0.95 = $95k` floor
      - **Or:** retain the MLL pattern but with crypto-friendly numbers (e.g. trailing 5%
        drawdown from peak, no cap — already wired in `AccountConfig.mll_threshold = 95_000`)

### B6. BigMoneyBot parameter tuning for BTC volume
The cushion tiers and ATR multiples in [`live/big_money_bot.py`](live/big_money_bot.py)
`BigMoneyConfig` (lines ~141-148) were tuned for MES (~$5k cushion = ~3 MES, ATR ≈ 3.75 pts):

```python
cushion_tiers = [
    (20_000, 13, 2.5, 3.5),   # $20k cushion → 13 MES
    (15_000, 11, 2.5, 3.5),
    ...
    (1_500,   2, 2.5, 3.5),
]
```

For BTC (~$500-1000 ATR on 3-min, 1 contract = 0.01 BTC, point_value = $0.01):
- $500 risk per trade ÷ ($0.01 × $1500 stop_pts) = **33 contracts ≈ 0.33 BTC ≈ $33k notional**
- So cushion tiers should map to similar contract counts

Suggested starting values:
```python
cushion_tiers = [
    (20_000, 50, 2.5, 3.5),   # 0.50 BTC = $50k notional
    (15_000, 40, 2.5, 3.5),   # 0.40 BTC
    (10_000, 30, 2.5, 3.5),   # 0.30 BTC
    ( 5_000, 20, 2.5, 3.5),   # 0.20 BTC
    ( 2_500, 10, 2.5, 3.5),   # 0.10 BTC
    ( 1_500,  5, 2.5, 3.5),   # 0.05 BTC = $5k notional (survival)
]
```
Other knobs to revisit:
- `min_atr_to_trade = 1.0 pt` → for BTC raise to ~$50 (otherwise always passes)
- `dollar_stop_after_incubation = $500` → keep as-is (account-size based)
- `daily_loss_limit = -$1500` → bump to **`-$5000`** (matches `AccountConfig.daily_loss_hard_stop`)
- `fast_confirm_max_vwap_atr = 3.0` → reasonable for BTC; keep

### B7. Parameter-tuning iteration
Once paper trading runs:
1. Capture daily summaries to `analyzers/reports/` (Topstep already writes these).
2. Compare actual ATR distribution to assumed thresholds.
3. Adjust `min_atr_to_trade`, `cushion_tiers`, and `fast_confirm_*` based on observed
   BTC 3-minute volatility.

### B8. Optional ports (lower priority)
- [ ] Copy [`analyzers/`](../topstep5/analyzers/) for daily/weekly summaries (uses Claude
      API — already configured).
- [ ] Copy [`research/`](../topstep5/research/) for backtesting + threshold calibration on
      BTC history (pull years of BTC 1-min OHLCV from Coinbase / ccxt cache).
- [ ] Copy [`tools/`](../topstep5/tools/) and [`scripts/`](../topstep5/scripts/) as needed.

---

## File-level changes summary

```
alpaca_btc/
├── _legacy/                         # backed-up original lean ccxt bot
├── config/
│   ├── __init__.py                  # NEW (empty package marker)
│   └── settings.py                  # NEW (BTC instrument, AccountConfig, Alpaca creds)
├── core/                            # 26 files copied from topstep5
│   ├── connection.py                # NEW — AlpacaCryptoConnection adapter
│   └── (all other core modules)     # copied verbatim from topstep5
├── risk/                            # all topstep5 risk modules copied
├── strategies/                      # all topstep5 strategies copied
├── live/
│   ├── hybrid_bot.py                # parent bot — copied
│   ├── alpha_bot.py                 # mid-tier bot — copied
│   ├── big_money_bot.py             # leaf bot — copied (the one we run)
│   ├── shadow_scorer.py
│   ├── shadow_variant.py
│   └── status_server.py
├── data/                            # empty (runtime artifacts will go here)
├── logs/                            # empty (rotated logs go here)
├── run_btc.py                       # NEW — entry point (no TopstepX prompts)
├── requirements.txt                 # UPDATED (ccxt, dotenv, signalrcore, websockets)
└── PORT_NOTES.md                    # this file
```

---

## How to run (after Phase B)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. .env file (project root)
cat > .env <<EOF
ALPACA_PAPER_API_KEY=PK...
ALPACA_PAPER_SECRET_KEY=...
USE_PAPER_TRADING=True
EOF

# 3. Run
python run_btc.py
```

---

## Architectural notes (for future reference)

- **Inheritance chain unchanged:** `HybridBot` → `AlphaBot` → `BigMoneyBot`. Every `self.conn.*`
  call routes through the adapter, so all of Topstep5's reconciliation, FSM, tranche state
  machine, LIL, adaptive exits, regime classifier, and trade-quality filter logic runs as-is.
- **The adapter is the only "translation" layer.** All MES-vs-BTC math is encapsulated in
  `point_value` / `contract_size_btc` / `tick_size`. The strategy and risk modules never
  see broker-specific concepts.
- **Software brackets vs native brackets:** Alpaca crypto spot doesn't support OCO orders.
  The adapter tracks SL/TP in memory and exposes `check_software_brackets(mark_price)` which
  the bot loop should call each bar to fire stop/target triggers. (Phase B item: wire this
  call into `hybrid_bot._main_loop`.)
- **WebSocket vs polling:** Topstep's market hub fires on every quote/tick. Our adapter
  stubs the hub and relies on the bot's own bar polling, which Topstep already supports
  via `--poll-only`. The trade-off is ~1-bar latency on entries; acceptable for 3-min bars.
