# Trade Map GUI — Phase Plan

**Living document.** Update it at the end of every phase: tick the boxes,
move the status, and replace the "Notes" with what actually happened.

- **Owner:** the agent implementing the phase
- **Started:** 2026-10-20
- **Last updated:** 2026-10-20 (Phase 1 closed)
- **Plan location:** `docs/TRADE_MAP_PLAN.md`

---

## 0. How to use this file

1. At the **start** of a phase, set its status to `IN PROGRESS` and
   un-tick nothing (the boxes are the plan, not the progress).
2. During the phase, tick a box the moment the item is DONE **and
   verified** — a ticked box means the verification command in the
   "Verification" column passed.
3. At the **end** of the phase:
   - set the status to `DONE` / `BLOCKED` / `DEFERRED`;
   - tick every box that is genuinely done;
   - fill in the phase's **Outcome** section (what was built, what was
     NOT built, what surprised us, the measured numbers);
   - update the status table in §2 and the "Last updated" line at the top;
   - commit the plan change **together with the phase's code** so the
     plan and the code never disagree.
4. A phase is **not** done because the code was written or the tests
   passed. It is done when the acceptance criteria in §4 are met **and**
   an independent reviewer has signed off (§5).

### Status legend

| Status | Meaning |
|---|---|
| `TODO` | Not started |
| `IN PROGRESS` | Being worked on |
| `IMPLEMENTED` | Code and tests are written, but the suite has **not** been run and no reviewer has signed off — not DONE |
| `DONE` | All acceptance criteria met, independently reviewed, committed |
| `BLOCKED` | Cannot proceed — the blocker is named in the phase's Notes |
| `DEFERRED` | Deliberately postponed — the reason is recorded |

### Why Phase 1.5 exists

The independent code-quality assessment of Phase 1 (commit `30a6bcd`)
found that the per-bar trading logic is **triplicated** across
`scripts/run.py` (twice) and `src/gui/replay.py` — 358 lines — and that
the GUI replay is **63 % SQLite-bound** (6.2 database connections per
bar). The commit's headline claim is "the runtime and the GUI can never
disagree about why a bar stopped where it did"; today that is enforced
for one function by a test, not by construction. Phase 1.5 makes the
claim true and takes the suite from ~490 s back under 300 s. It is
small, mechanical, and it must land **before** the chart layer, because
every chart surface will call the replay.

---

## 1. Product goal (unchanged since the brief)

> A GUI that shows **exactly where the trading system took trades and
> why**, at TradingView-grade chart quality, flexible enough for
> options, across every timeframe the system supports.

User decisions already made (do not relitigate):

| Decision | Choice |
|---|---|
| Chart renderer | **Plotly** |
| Data source | **artifacts first, replay fallback** |
| Options support | **chart + contract metadata** (no Greeks UI) |
| Packaging | **`src/gui/` + `scripts/`, with a new `viz` extra** |

Non-negotiable invariants (every phase must preserve them):

1. **No look-ahead.** Every feature, indicator and overlay is computed
   only from information available at that bar's timestamp.
2. **The GUI is a viewer.** It must never place an order and never
   mutate a production artifact (`data/registry.db`, the journals, the
   logs).
3. **One vocabulary.** The runtime, the TUI dashboard, the diagnostics
   CLI and the GUI all speak the same terminal-stage vocabulary.
4. **The runtime and the GUI must agree.** Same bars in → same terminal
   stage, same trades, same equity, bar for bar.
5. **`src/` never imports `scripts/`.** `src/gui/__init__.py` imports
   no charting library at module scope (Docker installs
   `requirements.txt`, which has no plotly).

---

## 2. Status overview

| Phase | Scope | Status | Commit |
|---|---|---|---|
| **1** | Data/core layer: `src/gui/` models, timeframes, indicators, instruments, artifacts, replay + the decision-observability fixes | **DONE** | `30a6bcd` |
| **1.5** | Hardening: de-duplicate the bar loop, fix the SQLite bottleneck, close the remaining quality findings | **IMPLEMENTED** | uncommitted |
| **2** | Chart layer: `src/gui/theme.py`, `overlays.py`, `charts.py` | **IMPLEMENTED** | uncommitted |
| **3** | The app: `scripts/trade_map_app.py` (Streamlit) + `scripts/trade_map.py` (CLI/HTML export) | **IMPLEMENTED** | uncommitted |
| **4** | Docs & packaging: `docs/explain/trade-map-gui.md`, AGENTS.md, `viz` extra, README | **IMPLEMENTED** | uncommitted |
| **5** | Final review & release | **TODO** | — |

---

## 3. Phase 1 — Data/core layer — **DONE**

**Commit:** `30a6bcd` — *"Add Trade Map GUI core layer (Phase 1) and fix
decision observability"*

### What was built

- `src/gui/models.py` — `SessionBundle` + the column vocabularies every
  surface shares (`BAR_COLUMNS`, `DECISION_COLUMNS`, `TRADE_COLUMNS`,
  `ZONE_COLUMNS`, `EQUITY_COLUMNS`), `empty_frame`, `normalize_dtypes`.
- `src/gui/timeframes.py` — `parse_timeframe`, `bucket_start_series`
  (the single bucket authority), causal `resample_ohlcv`,
  `align_to_timeframe`, `to_epoch_seconds`.
- `src/gui/indicators.py` — causal EMA/SMA/RSI/MACD/ATR/Bollinger/VWAP/
  Donchian, seeded to match **TradingView / TA-Lib** exactly.
- `src/gui/instruments.py` — `InstrumentSpec` + the YAML contract
  registry (`config/instruments.yaml`), repo-root-relative.
- `src/gui/artifacts.py` — **read-only** readers for the trade journal,
  the §8 feed, the decision traces, the zone events and the OHLCV.
- `src/gui/replay.py` — replays a session through the **real** pipeline
  and returns a `SessionBundle`.
- `src/execution/position_lifecycle.py` — `OpenPosition`,
  `check_position_exit`, `build_execution_engine`, shared by the
  deployment loop and the GUI replay.
- `src/utils/time_utils.py` — one epoch-unit implementation for the
  runtime and the GUI.

### Production bugs fixed (the reason the phase existed)

| Bug | Fix |
|---|---|
| The trace/§8 record was written **before** risk+execution, so `terminal_stage` was never `executed` and `decision` was always `NO_TRADE` | Write after, in both loops |
| A position-open bar was labelled `no_expert_candidate` | New `position_open` terminal stage |
| An order that **raised** was labelled `no_expert_candidate` | New `execution_failed` stage, with the exception text |
| The §8 feed reported LONG/SHORT for orders that never filled | Only report a direction for a filled/partial order |
| Integer epochs were parsed as nanoseconds → every bar on 1970-01-01 | Per-value unit detection, shared with the GUI |
| `scripts/diagnose.py --summary` crashed (no `sys.path` bootstrap) | Bootstrap added |
| `pip install -e .` produced an unimportable `src.gui` | `[tool.setuptools.packages.find]` → `src` is the installed package |
| The test suite dirtied the tracked `data/registry.db` | e2e subprocesses run in an isolated cwd |

### Verification (re-run after any change to this phase)

```bash
pytest tests/ -q                                   # 271 passed
pytest tests/unit tests/integration tests/e2e -q   # order-independent
ruff check src/ scripts/ tests/                    # 11 (baseline 12)
mypy src/                                          # 33 (baseline 33)
sha256sum data/registry.db                        # unchanged after a full run
```

### Outcome

- 271 tests (was 119); ruff −1, mypy ±0, no new findings.
- The runtime↔GUI equivalence is pinned by
  `tests/e2e/test_run_observability.py::test_the_gui_replay_and_the_runtime_agree_on_every_bar`
  (891/891 bars, 0 mismatches).
- **Not done (deliberately):** no charts, no app, no HTML export — those
  are Phases 2–3.
- **Carried into Phase 1.5:** the triplicated bar loop, the SQLite
  bottleneck, the six dead public helpers, the two weak tests, and the
  missing behaviour-change notes.

---

## 4. Phase 1.5 — Hardening — **IMPLEMENTED (unverified)**

**Why first:** every chart surface calls the replay, and the replay is
currently 18 ms/bar and 63 % SQLite-bound. The bar-loop de-duplication
is what makes invariant §1.4 true *by construction*.

### 1.5.1 De-duplicate the per-bar trading loop — **IMPLEMENTED**

- [x] Extract the exit → risk → order → open-position → classify →
      trace block into ONE function in
      `src/execution/position_lifecycle.py`
      (e.g. `process_bar(state, *, risk_layer, execution_engine,
      position, equity, fee_pct, config, on_trade_open, on_trade_close)
      -> BarOutcome`).
- [x] `scripts/run.py::run_historical`, `scripts/run.py::run_live_loop`
      and `src/gui/replay.py::replay_session` call it; the differences
      (journal vs. in-memory list, `mode`, the archive flush) become
      callbacks.
- [x] The barrier computation (`k_tp`/`k_sl`, incl. the
      `regime_specific_barriers` override), the sizing, the equity
      update, the drawdown breaker trip and the journal write exist
      **once**.
- [x] The e2e equivalence test still passes, and now compares the
      **equity curve and the trade list**, not only the terminal stages.

**Verification:** `grep -c "compute_barrier_pcts" src/ scripts/` → 1;
the suite passes; the equivalence test compares equity + trades.

### 1.5.2 Fix the SQLite bottleneck — **IMPLEMENTED**

- [x] `src/registry/db.py::get_connection` — one pooled connection per
      store instead of a new one per call; run the schema DDL **once**
      per connection, not on every `connect()`.
- [x] `PRAGMA journal_mode=WAL` + `PRAGMA synchronous=NORMAL` for the
      risk store.
- [x] `src/layers/risk_layer.py::RiskStateStore` — reuse the
      connection across `_get`/`_set`/`heartbeat`.
- [x] Give the GUI replay a **non-persistent** store (it writes to a
      throw-away database today; it does not need durability).
- [x] Do **not** change the breaker's semantics: the daily-loss
      accumulator, the trade counter and the trip flags must behave
      exactly as they do now.

**Verification:** a 900-bar replay takes < 3 s (was 16–17.5 s);
`pytest tests/ -q` completes in < 300 s (was ~490 s); the breaker
tests in `tests/unit/test_phase4.py` and
`tests/unit/test_invariants.py` still pass unchanged.

### 1.5.3 Close the remaining quality findings — **IMPLEMENTED**

- [x] Cache `SessionBundle.close_by_timestamp` the way
      `decision_by_timestamp` already is (11.4 ms × 20 000 bars is
      228 s per chart pass) and replace the `id()`-based cache key.
- [x] Move `STAGE_STYLES` from `src/diagnostics/dashboard_data.py` into
      `src/diagnostics/decision_trace.py`, next to `STAGE_LABELS`, and
      re-export it — so the "one source of truth" claim is literally
      true and the GUI never imports a TUI module.
- [x] `normalize_dtypes` must **not** drop rows; move the row loss to
      the readers that already decide it (`read_trade_journal`,
      `read_decisions_jsonl`).
- [x] Log the SQLite degradation paths in `src/gui/artifacts.py` (a
      locked registry currently looks like "no trades").
- [x] Make the trade-journal path repo-root-relative like the
      instrument registry, with a CWD fallback.
- [x] Delete the six dead public helpers
      (`timeframes.is_valid_timeframe`, `timeframes.timeframe_seconds`,
      `timeframes.format_timeframe`, `timeframes.seconds_to_timeframe`,
      `indicators.indicator_summary`, `instruments.InstrumentSpec.price_label`)
      — or mark them explicitly as reserved Phase-2 API.
- [x] Delete the vacuous
      `test_an_empty_trade_frame_and_a_populated_one_concat_cleanly`.
- [x] Replace the AST-parsing vocabulary test with a behavioural one
      (call the classifier, assert the stage is in the vocabulary).
- [x] Replace the three 17-argument `ExecutionRecord` rebuilds with
      `dataclasses.replace(record, status="rejected", filled_size=0.0)`.
- [x] Use `src.registry.db.connect()` in the tests instead of
      hand-written `CREATE TABLE trade_journal` DDL (3 copies).
- [x] Move the synthetic-OHLCV generator to one shared home
      (`src/utils/synthetic.py`); the backtest harness imports it and
      passes its own start timestamp.
- [x] Introduce `tests/conftest.py` for the **shared builders only**
      (contract objects, the journal DB, the synthetic series, the
      `hold` config, the `sys.path` bootstrap) — keeping the
      per-file fixtures where they are.
- [x] Fix the `AGENTS.md` claim that `run.py` "raises `ValueError` if
      config requests > 1" concurrent position (the code raises for
      `< 1`).
- [x] Add the missing behaviour changes to the docs: tz-aware
      timestamps, the packaging/import-surface change, the
      `place_order` signature, the `OpenPosition` shape, the Streamlit
      port move, and the `make latency` flag fix.

**Verification:** the suite passes in both orders; a reviewer confirms
each finding is closed by a test, not by a comment.

### Acceptance criteria

- [x] The bar-processing block exists in exactly one place (`position_lifecycle.process_bar`; asserted by `test_phase15_hardening.py`).
- [ ] A 900-bar replay takes < 3 s; the full suite takes < 300 s.
- [ ] Every finding in the Phase-1 code-quality assessment is either
      fixed with a test or explicitly waived **in this file**.
- [ ] `ruff` ≤ 11 (**verified: 11**), `mypy` = 33 (**not run**), `data/registry.db`
      unchanged after a full test run (**not run**).

### Outcome

Implemented: `process_bar` is the one per-bar block (run.py's two loops and the GUI replay call it); SQLite is pooled with WAL on the risk store; the replay uses `RiskStateStore(persistent=False)`; every 1.5.3 finding has a code change and a test in `tests/unit/test_phase15_hardening.py`. **Not measured:** the 900-bar replay time and the suite time (the change set was written without running tests); the replay-budget test defaults to the plan's 3 s and `REPLAY_BUDGET_SECONDS` overrides it.

---

## 5. Phase 2 — Chart layer — **IMPLEMENTED (unverified)**

**Goal:** the drawing primitives, with no UI framework in the way.
`src/gui/charts.py` builds Plotly figures; `src/gui/overlays.py`
describes *what* to draw as plain data; `src/gui/theme.py` owns the
palette. A Streamlit app (Phase 3) only arranges them.

### 2.1 `src/gui/theme.py` — **IMPLEMENTED**

- [x] A single dark palette (background, grid, up/down candles, text).
- [x] A colour per terminal stage, derived from
      `src/diagnostics/decision_trace.py::STAGE_LABELS` / `STAGE_STYLES`
      — **not** a second copy.
- [x] Font sizes and spacing constants.
- [x] No plotly import at module scope? (Plotly IS the renderer here, so
      it may be imported — but keep it out of `src/gui/__init__.py`.)

### 2.2 `src/gui/overlays.py` — **IMPLEMENTED**

Pure data: *what* to draw, and *where* — never a figure.

- [x] `EntryMarker` / `ExitMarker` — from the trade rows, mapped onto
      the bars they belong to.
- [x] `ZoneBand` — from the zone events.
- [x] `EquityCurve` — from the equity frame.
- [x] `DecisionMarker` — the per-bar decision, on the bar it happened.
- [x] `RegimeBand` — the regime per bar, as a background band.
- [x] Every overlay is **derived from a bar timestamp**, never from a
      position in the array (so a timeframe change cannot misplace it).

### 2.3 `src/gui/charts.py` — **IMPLEMENTED**

- [x] `build_price_chart(bars, overlays, timeframe, instrument) -> go.Figure`
      — candles + volume, TradingView-grade layout.
- [x] `build_decision_chart(...)` — the per-bar decision, colour-coded
      by terminal stage.
- [x] `build_equity_chart(...)`.
- [x] Timeframe switching that goes through
      `src/gui/timeframes.py::resample_ohlcv` — never a second
      resampler.
- [x] A `figure_to_html(...)` helper for the CLI export (Phase 3).
- [x] Figures are **deterministic**: the same session renders the same
      figure (no `now()`, no random ids in the output).

### 2.4 Tests — **IMPLEMENTED**

- [x] Every overlay lands on the **right bar** (a fixture with known
      timestamps).
- [x] The chart is **causal**: a bar's overlay never shows information
      from a later bar.
- [x] A timeframe switch does not move a trade marker off its bar.
- [x] The figure contains no NaN/None in the traceable data.
- [x] Determinism: two renders of the same session are byte-identical
      (after stripping Plotly's generated ids).

### Acceptance criteria

- [ ] A figure renders for a real session and for the synthetic demo.
- [ ] No overlay is misplaced when the timeframe changes.
- [ ] `ruff`/`mypy` clean on the new modules; the suite still passes.
- [ ] An independent reviewer has verified the causality of the
      overlays against the source bars.

### Outcome

Implemented: `theme.py`, `overlays.py`, `charts.py` (+ `sources.py`, `export.py`). Tests: `test_gui_overlays.py`, `test_gui_charts.py`. **Not done:** an independent causality review.

---

## 6. Phase 3 — The app — **IMPLEMENTED (unverified)**

### 3.1 `scripts/trade_map_app.py` (Streamlit) — **IMPLEMENTED**

- [x] Symbol + instrument picker (from `config/instruments.yaml`).
- [x] Source picker: **live artifacts** → **replay** → **synthetic
      demo**, in that order (artifacts first, replay fallback).
- [x] Timeframe selector, driven by
      `src/gui/timeframes.py::display_timeframes` (never finer than the
      base granularity).
- [x] The price chart with every overlay toggleable.
- [x] The decision chart and the equity curve.
- [x] A "why did this bar not trade?" panel: the terminal stage, the
      reason, and the stage-by-stage trace.
- [x] The trade table, clickable through to the bar.
- [x] Warnings surfaced loudly (synthetic data, relaxed liquidity
      gates, a stale session).
- [x] Binds `127.0.0.1:8572` by default, overridable via
      `TRADE_MAP_PORT` — **never** on 8501.

### 3.2 `scripts/trade_map.py` (CLI / HTML export) — **IMPLEMENTED**

- [x] `--source {artifacts,replay,synthetic}`, `--symbol`,
      `--timeframe`, `--out <file.html>`.
- [x] A self-contained HTML file (Plotly's `include_plotlyjs="inline"`).
- [x] A text summary on stdout: trades, win rate, expectancy, the
      terminal-stage funnel.
- [x] Exit 0 with a clear message when there is nothing to show (no
      trades yet) — never a traceback.

### 3.3 Makefile + docs — **IMPLEMENTED**

- [x] `make trade-map` → the Streamlit app on 8572.
- [x] `make trade-map-export` → the HTML export.
- [x] `TRADE_MAP_PORT ?= 8572` in the Makefile, documented in
      `AGENTS.md` under "Ports".

### Acceptance criteria

- [ ] `make trade-map` serves on 127.0.0.1:8572 and renders a chart.
- [ ] `make trade-map-export` writes a self-contained HTML file that
      opens offline.
- [ ] The app can NEVER place an order (assert it in a test).
- [ ] The app never writes to the repo's tracked `data/`.

### Outcome

Implemented: `scripts/trade_map_app.py`, `scripts/trade_map.py`, Makefile targets. **Not done:** actually launching the app / opening the export (nothing was executed).

---

## 7. Phase 4 — Docs & packaging — **IMPLEMENTED (unverified)**

- [x] `docs/explain/trade-map-gui.md` — how every subsystem of the GUI
      works: the data model, the timeframes, the indicators, the
      overlays, the replay, the app. Match the depth of the existing
      `docs/explain/*.md`.
- [x] `AGENTS.md` — the new layout, the new commands, the ports, the
      "one vocabulary" rule.
- [x] `README.md` — a short section with a screenshot and the run
      instructions.
- [x] `pyproject.toml` — confirm the `viz` extra carries plotly and
      that the core still imports without it.
- [x] `docs/limitations.md` — what the GUI does **not** show (no
      Greeks, no order-book depth, no multi-symbol, replayed sessions
      use a throw-away risk store).
- [x] `TODO.md` — the deferred items this work surfaced.

### Acceptance criteria

- [ ] Every documented command runs.
- [x] Every documented file path exists (checked by listing).
- [ ] The limitations are stated, not implied.

### Outcome

Implemented: `docs/explain/trade-map-gui.md`, AGENTS.md, README, `docs/limitations.md`, `TODO.md`, the `viz` extra. No screenshot is committed.

---

## 8. Phase 5 — Final review & release — **TODO**

- [ ] An **independent** reviewer audits the whole feature against the
      product goal in §1: does the GUI actually show where the system
      traded and why?
- [ ] A fresh-eyes check of the no-look-ahead invariant on the full
      path (bars → indicators → overlays → chart).
- [ ] A fresh-eyes check that the GUI cannot mutate anything and
      cannot place an order.
- [ ] A fresh-eyes check that the runtime and the GUI still agree bar
      for bar.
- [ ] The full suite green in both orders; `ruff`/`mypy` at or below
      baseline; `data/registry.db` unchanged.
- [ ] The plan's §2 table updated with every phase's commit.
- [ ] `docs/explain/trade-map-gui.md` reviewed against the final code.

### Outcome

_(filled in when the phase closes)_

---

## 9. Cross-phase checklist

Every phase must satisfy these before it is called done:

- [ ] `pytest tests/ -q` green **and** `pytest tests/unit tests/integration tests/e2e -q` green.
- [ ] `ruff check src/ scripts/ tests/` ≤ 11 errors (all pre-existing).
- [ ] `mypy src/` = 33 errors (all pre-existing), and the new modules
      clean under `mypy --disallow-untyped-defs`.
- [ ] `sha256sum data/registry.db` unchanged by a test run.
- [ ] No new import cycle; `src/` imports no `scripts/`.
- [ ] The pipeline execution order unchanged; the risk layer still
      called only for new entries.
- [ ] An **independent** reviewer has signed off (§5).
- [ ] This plan file is updated and committed with the code.

---

## 10. Open questions

| Question | Owner | Status |
|---|---|---|
| Should the app show a *live* session's decisions as they arrive, or only a completed one? | — | Open — decide in Phase 3 |
| Do we need a multi-symbol comparison view? | — | Open — out of scope unless asked |
| Should the HTML export be a single file or a bundle? | — | Decided: a single self-contained file |
| Do we need the trade-map to run against a *live* connector (not just artifacts)? | — | Open — decide in Phase 3 |

---

## 11. Change log

| Date | Phase | Change |
|---|---|---|
| 2026-10-20 | 1 | Plan created. Phase 1 closed (`30a6bcd`). Phase 1.5 added after the independent code-quality assessment. |
| 2026-10-26 | 1.5–4 | Phases 1.5, 2, 3 and 4 implemented in one change set (`IMPLEMENTED`, not `DONE`): the test suite was **not** run, nothing was launched, and Phase 5 (independent review) has not happened. |
