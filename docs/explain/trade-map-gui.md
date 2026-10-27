# Trade Map GUI

**Goal.** Show **exactly where the trading system took trades, and why**, at
TradingView-grade chart quality, for every timeframe the system supports and
flexible enough for options. It is a *viewer*: it never places an order and
never mutates a production artifact.

Read this next to `docs/TRADE_MAP_PLAN.md` (the living phase plan) and
`docs/explain/layers-and-pipeline.md` (what a "stage" is).

```
 artifacts ─┐                                      ┌─► charts.py ─► Streamlit app
 replay ────┼─► sources.py ─► SessionBundle ─► overlays.py ─┤                (scripts/trade_map_app.py)
 synthetic ─┘   (models.py)                                 └─► export.py ─► one HTML file
                                                                          (scripts/trade_map.py)
```

## 1. The invariants (every change must keep them)

1. **No look-ahead.** Every indicator, overlay and resampled candle uses only
   information available at its timestamp (§4, §5).
2. **The GUI is a viewer.** Replays are forced to `paper_sim`; the replay's
   risk store is in-memory; the app and CLI write nothing under `data/`
   (`scripts/trade_map.py` refuses `--out` inside it).
3. **One vocabulary.** `TERMINAL_STAGES`, `STAGE_LABELS` and `STAGE_STYLES`
   live in `src/diagnostics/decision_trace.py`; the runtime, the TUI, the
   diagnostics CLI and the GUI read them. `theme.STAGE_COLOURS` is *derived*.
4. **The runtime and the GUI agree.** Both call
   `position_lifecycle.process_bar` (§3): same bars in → same stage, trades
   and equity, bar for bar — by construction, not by a test comparing copies.
5. **`src/` never imports `scripts/`**, and `src/gui/__init__.py` imports no
   charting library (the Docker image installs `requirements.txt`, which has
   no plotly; the `viz` extra carries plotly + streamlit).

## 2. The data model — `src/gui/models.py`

`SessionBundle` is deliberately dumb: normalised frames and plain dicts.

| Field | Content |
|---|---|
| `bars` | OHLCV, one row per **base** bar, `timestamp` = int64 epoch seconds |
| `decisions` | one row per processed bar: regime, expert, scores, `terminal_stage`, `terminal_reason` |
| `trades` | normalised trade records (open trades have `exit_reason == "open"`) |
| `traces` | raw stage-by-stage `DecisionTrace` dicts keyed by bar timestamp |
| `zones` | zone lifecycle events (`created / touched / rejected / broke / merged / decayed`) |
| `equity` | equity per bar |
| `states` | full `SystemState` per bar (replay only) |
| `meta` | provenance: source, label, `synthetic`, `relaxed_liquidity`, `warnings` |

Column vocabularies (`BAR_COLUMNS`, `DECISION_COLUMNS`, …) are fixed here so
the three sources produce the same schema and charts never branch on origin.

* `normalize_dtypes` **never drops a row.** A missing `timestamp` /
  `entry_timestamp` raises; deciding what to do with such a row is the
  *reader's* job (`read_trade_journal`, `read_decisions_jsonl` drop and log
  them). `bar_index` is the nullable `Int64`.
* `decision_by_timestamp` and `close_by_timestamp` are built **once** per
  frame. Assigning a new `decisions` / `bars` frame invalidates them (a
  `__setattr__` hook — no `id()` cache key, which Python may reuse). Frames
  are treated as immutable once attached.

## 3. One bar, one implementation — `process_bar`

`src/execution/position_lifecycle.py::process_bar(state, *, risk_layer,
execution_engine, position, equity, fee_pct, config, …) -> BarOutcome` is the
exit → risk → order → open-position → classify → trace block. It owns:

* the exit step (`check_position_exit`), the breaker bookkeeping
  (`record_trade_result`, equity / drawdown trip), the heartbeat and
  `advance_bar`;
* `compute_position_barriers` — the **only** place `k_tp` / `k_sl` (with the
  `regime_specific_barriers` override) are applied;
* the `execution_failed` handling and the `position_open` classification;
* building the `DecisionTrace` (`terminal_stage` from the shared
  `classify_terminal_stage`).

Callers differ only in two callbacks: `on_trade_open(position) -> trade_id`
and `on_trade_close(position, exit_ts, exit_info)`. `scripts/run.py` journals
to SQLite (`journal_callbacks`); `src/gui/replay.py` appends to a list and
uses `gui-<ts>` ids. `max_concurrent_positions()` validates the config once
(raises for a value below 1).

## 4. Timeframes and indicators

* `timeframes.bucket_start_series` is the **single bucket authority** — the
  resampler and the aligner both call it. Ticks are epoch-aligned, `1D`
  UTC-midnight, `1W` Monday, `1M` calendar month.
* `resample_ohlcv` is causal: a bucket labelled `T` holds only bars in
  `[T, T+tf)`; open = the *first bar's* open, close = the *last bar's* close.
  `display_timeframes(base)` never offers a timeframe finer than the base.
* `align_to_timeframe(how="first")` keeps the value current at the bucket's
  **open** (the only causal choice for something painted across the whole
  candle — regimes); `how="last"` keeps the bucket's end value (cumulative
  quantities — equity).
* `indicators.py` is seeded to match TradingView / TA-Lib exactly. The app
  computes them on the **resampled** candles.

## 5. Overlays — `src/gui/overlays.py`

Frozen dataclasses: *what* to draw and *where*. Never a figure.

| Overlay | Derived from | Causality note |
|---|---|---|
| `EntryMarker` | trade rows | carries only what was known **at entry** (expert, regime, confidence, size, TP/SL levels) — no outcome fields exist on it |
| `ExitMarker` | trade rows | the outcome lives here |
| `TradeSpan` | trade rows | entry→exit join; open trades run to the last candle |
| `DecisionMarker` | `decisions` | one per candle: the decision that got **furthest** (rank = index in `TERMINAL_STAGES`) |
| `ZoneBand` | `zones` | a *level* from creation to the ending event (`broke/merged/decayed`) |
| `RegimeBand` | `decisions` | regime at the candle's open, runs merged |
| `EquityCurve` | `equity` | equity at each candle's close + running drawdown |

Every overlay is placed with `bucket_start(ts, timeframe)` — by **timestamp,
never array position** — so a timeframe change re-derives, and cannot misplace,
anything. Rows that cannot be placed are dropped here (no NaN / None reaches a
chart). A missing price falls back to the bar's close (`close_by_timestamp`).

## 6. Charts — `src/gui/charts.py` (the only plotly import)

* `build_price_chart(bars, overlays, timeframe, instrument, …)` — candles
  (resampled via `resample_ohlcv`), volume, indicators, optional RSI / MACD
  panes, regime bands and zone levels *below* the candles, trade lines / TP-SL
  levels / entry & exit markers / decision dots above. Raises if the overlays
  were built for a different timeframe.
* `build_decision_chart` (one row per stage), `build_stage_funnel`,
  `build_equity_chart`, `build_session_figures`, `figure_to_html`.
* **Deterministic:** explicit trace `uid`s, string timestamps, a fixed
  `div_id`, no `now()`.
* Gaps are **shapes**, never `None`-separated trace data.

`theme.py` owns the palette, the per-stage colours (derived), regime / zone /
indicator styles and the layout constants.

## 7. Sources — `src/gui/sources.py`

`artifacts → replay → synthetic`, in that order (`auto_source`).

* **artifacts** (read-only): journal, §8 decision log (honours
  `$DECISIONS_JSONL`), traces, zone log, raw OHLCV. Paths resolve from the
  **repo root** with a CWD fallback (`artifacts.resolve_artifact_path`). A
  database that exists but can't be read is logged as a warning, not shown as
  "no trades". Equity is derived from the journal and flagged.
* **replay**: the raw OHLCV through the real pipeline; liquidity gates can be
  relaxed (and are then labelled).
* **synthetic**: `src/utils/synthetic.py::synthetic_ohlcv` (the one
  generator, shared with the backtest harness and latency benchmark), replayed
  with relaxed gates and an "ILLUSTRATIVE" warning.

## 8. The app — `scripts/trade_map_app.py`

`make trade-map` → **http://127.0.0.1:8572** (`TRADE_MAP_PORT` overrides; never
8501). Sidebar: instrument, source, timeframe, overlay / stage / indicator /
pane toggles. Main: warning banners (synthetic, relaxed liquidity, stale),
metric strip, the price chart, then tabs — **Why this bar?** (terminal stage,
reason, facts, stage-by-stage trace, prev/next/next-trade navigation),
**Decisions**, **Equity**, **Trades** (click a row to jump to and zoom on it).
Clicking a candle selects the bar. Sessions are cached per parameter set.

## 9. The CLI — `scripts/trade_map.py`

`--source {auto,artifacts,replay,synthetic} --symbol --timeframe --out file.html`.
Prints trades / win rate / expectancy / drawdown and the terminal-stage funnel,
writes ONE self-contained HTML (plotly.js inlined once). Exits 0 with a
message when there is nothing to show; exits 2 with an install hint if plotly
is missing; refuses an `--out` under `data/registry.db|logs|raw|models`.

## 10. Phase 1.5 — what got faster and why

The replay was 63 % SQLite-bound (~6 connections per bar, schema DDL on every
`connect()`).

* `registry/db.py` pools **one connection per file per thread**; the schema
  runs once per connection; a deleted/replaced/closed file is detected and
  re-opened. `RiskStateStore` uses WAL + `synchronous=NORMAL`.
* The replay uses `RiskStateStore(persistent=False)` — a private in-memory
  database with **identical** breaker semantics (daily-loss accumulator,
  trade counter, trip flags).

## 11. Behaviour changes to know about

* **Timestamps are tz-aware UTC** in `scripts/run.py::load_ohlcv_for_run`;
  integer epochs are unit-detected per value (s / ms / µs / ns) by
  `src/utils/time_utils.py`, shared with the GUI.
* **Packaging / import surface.** `[tool.setuptools.packages.find]` makes
  `src` the installed package, so `import src.gui` works after
  `pip install -e .`; `from run import OpenPosition …` still works because
  `run.py` re-exports the lifecycle names.
* **`place_order` gained `fv=`** (the feature vector, for paired liquidity
  samples); `process_bar` passes it.
* **`OpenPosition` declares** `expert`, `regime`, `selection_confidence` and
  `risk_score` (they used to be bolted on with `object.__setattr__`, invisible
  to `asdict`).
* **Streamlit port move:** the demo runs on 8571 (not 8501); the Trade Map on
  8572.
* **`make latency`** passes `--bars 300` (the flag the script actually takes).
* **The bar loop moved.** `run_historical`, `run_live_loop` and the replay no
  longer contain the exit/risk/order block; `scripts/run.py` no longer calls
  `check_position_exit` itself (tests patch it in `position_lifecycle`).
  `run_live_loop` now also raises on `max_concurrent_positions < 1`, at start-up.
* **`normalize_dtypes` no longer drops rows** (it raises); `bar_index` is
  nullable.

## 12. Testing map

| Concern | Test |
|---|---|
| runtime ≡ GUI (stage, trades, equity) | `tests/e2e/test_run_observability.py` |
| pool / in-memory store / single bar loop / quality findings | `tests/unit/test_phase15_hardening.py` |
| overlay placement & causality | `tests/unit/test_gui_overlays.py` |
| figures: clean data, determinism, timeframe switch | `tests/unit/test_gui_charts.py` |
| sources, summary, CLI, "viewer can't trade" | `tests/unit/test_gui_app.py` |
| 900-bar replay budget | `tests/integration/test_gui_replay.py` |
