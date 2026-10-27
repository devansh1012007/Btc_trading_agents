# Btc_trading_agents — Agent Instructions

## Repo layout

```
src/
  pipeline.py              ← TradingPipeline (main orchestrator)
  state/contracts.py       ← all shared dataclasses (frozen, single source of truth)
  state/serialization.py
  layers/                  ← market_data, feature_engine, regime_detector, structure_engine,
                             expert_layer, opportunity_filter, router, risk_layer
  experts/                 ← trend, mean_reversion, breakout (+ base_expert, expert_registry)
  labeling/                ← triple_barrier labeling
  backtest/                ← replay.py, walk_forward.py
  execution/               ← execution_engine + paper_sim/paper_live/live backends
    position_lifecycle.py   ← OpenPosition, check_position_exit,
                              build_execution_engine, and process_bar():
                              THE per-bar block (exit → risk → order →
                              open → classify → trace). run_historical,
                              run_live_loop and the GUI replay all call it;
                              only the journal callbacks differ
  liquidity/               ← synthetic_liquidity_model
  registry/                ← model registry (SQLite: pending → approved → superseded)
  feedback/                ← trade_journal, zone_event_log, system_state_archive
  monitoring/              ← drift_monitor (PSI), performance_monitor
  connectors/              ← CCXTConnector
  alerting/                ← notifier
  diagnostics/             ← decision_trace.py (Phase 5: stage-by-stage bar traces)
    decision_trace.py      ← TERMINAL_STAGES (the ONE stage vocabulary),
                              classify_terminal_stage(), build_trace_stages()
    dashboard_data.py      ← loaders for the TUI console; STAGE_LABELS /
                              STAGE_STYLES live next to the vocabulary
  config/                  ← regime_adaptive.py, config_watcher.py (Phase 5)
    regime_adaptive.py     ← resolve_regime_param() — unified regime-dependent threshold resolution
    config_watcher.py      ← ConfigWatcher — hot-reload for safe config paths
  gui/                     ← Trade Map GUI core (no plotly/streamlit imports)
    models.py              ← SessionBundle + the column vocabularies every
                              surface shares (empty_frame/normalize_dtypes)
    timeframes.py          ← parse_timeframe, bucket_start_series,
                              causal resample_ohlcv, align_to_timeframe
    indicators.py          ← causal EMA/RSI/MACD/ATR/BB/VWAP/Donchian
                              (TradingView/TA-Lib seeding)
    artifacts.py           ← READ-ONLY readers for the trade journal, §8
                              log, decision traces, zone events, OHLCV
    instruments.py         ← InstrumentSpec + the YAML contract registry
    replay.py              ← replays a session through the real pipeline
                              (calls execution.position_lifecycle.process_bar)
    sources.py             ← artifacts → replay → synthetic session loaders +
                              summarize_session (no streamlit / plotly)
    theme.py               ← palette; per-stage colours DERIVED from
                              decision_trace.STAGE_STYLES (no second copy)
    overlays.py            ← WHAT to draw and WHERE, as frozen data (entries,
                              exits, spans, decisions, zones, regimes, equity)
    charts.py              ← Plotly figure builders (the only module that
                              imports plotly; never imported by gui/__init__)
    export.py              ← self-contained HTML report
config/
  settings.base.yaml       ← full config (all layers)
  settings.paper.yaml      ← env overlay (default)
  settings.live.yaml       ← env overlay
  instruments.yaml         ← instrument registry (spot/perp/option/equity)
scripts/
  run.py                   ← deployment entry (historical replay + live loop)
  full_stack_walk_forward_backtest.py ← M5.2: realized-metrics backtest vs Project_Plan.md
                                           targets, 4 baselines, calibration, reports+experiment log.
                                           Ablation toggles: --single-expert, --no-risk-layer, --no-regime-gating
  benchmark_latency.py     ← decision-latency/memory benchmark (avg/p95/RAM, synthetic)
  replay_trade.py          ← §10 trade replay: --list / --trade-id, journal+traces+OHLCV
  generate_figures.py      ← docs/images figures from a real run (matplotlib, Agg)
  demo_app.py              ← Streamlit decision demo (offline synthetic by default)
  train_*.py               ← per-component training
  retrain_all.py           ← cron retraining (never auto-promotes)
  approve_model.py         ← manual promotion in model registry
  rollback_model.py
  reset_breaker.py
  status.py
  diagnose.py              ← CLI diagnostics: --timestamp, --trade-id, --summary (Phase 5)
  validate_config_consistency.py ← pre-flight validation (Phase 5)
  preflight.py             ← runs consistency checks before session start (Phase 5)
  validate_live_backend.py ← scripted testnet validation runbook (Phase 5)
  dashboard.py             ← TUI signal console: 6 views over traces/journal/
                             reports/registry; --once for TTY-less render (M5.10)
  verify_phase1.py,
  verify_integration_phase{2,3,4}.py ← legacy manual verification scripts (not pytest)
tests/
  unit/                    ← per-component tests
    test_phase1.py         ← Market data, feature engine
    test_phase2.py         ← Structure engine + regime detection
    test_phase3.py         ← Triple-barrier, experts, opportunity filter, router
    test_phase4.py         ← Risk layer, execution, circuit breakers, registry
    test_phase5.py         ← Reporting metrics, tracing, config validation
    test_invariants.py     ← Financial invariants (position caps, breakers,
                              duplicate-order prevention) — extend when
                              adding risk controls
  unit/                    ← per-component tests
    test_decision_trace.py ← Terminal-stage vocabulary + classifier contract
    test_gui_core.py       ← Timeframes, indicators, instrument registry
    test_gui_data.py       ← Artifact readers (schema, timestamps, dtypes)
    test_gui_correctness.py← Defects the Phase-1 review found (units,
                              causality, dtypes, symbol matching)
    test_gui_regressions.py← The original review's regression set
  integration/             ← cross-layer wiring through the real pipeline
    test_integration.py    ← Full-stack integration
    test_gui_replay.py     ← GUI replay == the deployment loop, bar for bar
  e2e/
    test_backtest_smoke.py ← Backtest entrypoint as a subprocess; report artifacts
    test_run_observability.py ← The §8 feed + decision traces the GUI reads
data/                      ← gitignored: raw/, processed/, models/, logs/
  logs/decision_traces/    ← partitioned JSONL traces (Phase 5; per-stage
                              duration_ms + total_duration_ms latency fields)
  logs/decisions.jsonl     ← one structured record per bar (regime/expert/
                              opportunity/router/risk/decision — §8 feed)
  logs/system.jsonl        ← all log output as JSON lines
  faults.jsonl             ← system fault records (Phase 5)
```

## Packaging & tooling

- `pyproject.toml` — installable (`pip install -e ".[dev]"`); deps mirror
  requirements.txt (keep in sync; Docker installs from requirements.txt).
- Extras: `dev` (ruff, mypy, pytest-cov), `demo` (streamlit, matplotlib), `viz` (plotly + streamlit, for the Trade Map GUI).
- `src` is the installed top-level package (`[tool.setuptools.packages.find]`).
  Do not remove it: flat-layout auto-discovery would install `src/`'s
  CONTENTS as top-level packages and break `import src.gui`.
- CI: `.github/workflows/ci.yml` — ruff → mypy (strict modules + global
  lenient) → pytest with coverage. `make check` runs the same locally.
- mypy: `src/state` + `src/reporting` are strict (`disallow_untyped_defs`);
  everything else lenient — new modules should be fully annotated.

## Running

```bash
# Tests
pytest tests/ -v

# By tier
make unit          # tests/unit/
make integration   # tests/integration/
make e2e           # tests/e2e/

# Lint + typecheck (also run by CI)
make lint          # ruff over src/ scripts/ tests/
make typecheck     # mypy on strict modules (src/state, src/reporting)

# Single test file
pytest tests/unit/test_phase3.py -v

# Single test
pytest tests/unit/test_phase3.py::test_triple_barrier_tp_hit_first -v

# Pipeline dry-run (fetches 100 bars from Binance sandbox)
python -m src.pipeline

# Historical backtest through full stack
python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper

# Full-stack performance report vs Project_Plan.md targets (M5.2)
python scripts/full_stack_walk_forward_backtest.py --input data/raw/BTCUSDT.csv --env paper
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity  # smoke run

# Live loop (requires real credentials + confirm_real_money=true)
python scripts/run.py --mode live --env live

# Diagnostics: summary of why bars didn't trade
python scripts/diagnose.py --summary --since 2026-08-01

# Diagnostics: full trace for a specific bar
python scripts/diagnose.py --timestamp 1755900000

# Pre-flight validation before starting a session
# NOTE: on the SHIPPED config this currently FAILS the liquidity gate
# ("~98% of simulated bars would fail max_spread_pct=0.05") -- the
# synthetic liquidity model is wider than the shipped gate allows.
python scripts/preflight.py --env paper

# Config consistency validation
python scripts/validate_config_consistency.py

# TUI dashboard (live, 6 views, auto-refresh) / headless one-shot render
python scripts/dashboard.py            # keys: 1-6 views · r refresh · q quit
python scripts/dashboard.py --once

# Trade replay: list recent trades / full stage-by-stage replay of one trade
python scripts/replay_trade.py --list 10
python scripts/replay_trade.py --trade-id <id>

# Latency/memory benchmark (synthetic, deterministic)
python scripts/benchmark_latency.py --bars 300        # or: make latency

# Figures for docs/images (labeled ILLUSTRATIVE on synthetic data)
python scripts/generate_figures.py --synthetic 900 --relax-liquidity

# Streamlit decision demo (offline synthetic by default)
# NOTE: ports are NOT Streamlit's default 8501 — see "Ports" below.
streamlit run scripts/demo_app.py    # or: make demo  (http://127.0.0.1:8571)
make demo STREAMLIT_PORT=8600        # override the port

# Trade Map GUI: where the system traded, and why  (pip install -e ".[viz]")
make trade-map                       # http://127.0.0.1:8572
make trade-map TRADE_MAP_PORT=8600   # override the port
make trade-map-export                # self-contained HTML -> data/reports/trade_map.html
python scripts/trade_map.py --source synthetic --out /tmp/trade_map.html
```

# Ablation ladder (contribution of each component)
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity \
    --single-expert trend_expert     # also: --no-risk-layer, --no-regime-gating
```

## Ports (shared hosts)

Streamlit binds **127.0.0.1:8571** by default (`make demo`), not
Streamlit's 8501, because several checkouts (or several agents) can run
on the same host. `.streamlit/config.toml` holds the default; the
`STREAMLIT_PORT` Makefile variable overrides it per invocation.
The Trade Map GUI binds **127.0.0.1:8572** (`make trade-map`;
`TRADE_MAP_PORT` overrides it) so it runs next to the demo. **Nothing
here ever uses 8501.** **Never stop another project's container or dev
server to free a port — pick a different one.**

## Config system (easy to get wrong)

- **Layered YAML**: `load_layered_config(env)` merges `settings.base.yaml` + `settings.{env}.yaml` (deep merge, overlay wins).
- `scripts/run.py` and deployment scripts use `load_layered_config(args.env)` — **always pass `--env`**.
- Training/backtest scripts from Phase 2/3 are offline utilities; they read `settings.base.yaml` directly via `--config`.
- `system.data_granularity` (e.g. `"1m"`, `"30s"`) is the **only** switch for timeframe — no code change needed. All layers check it for model-granularity compatibility.
- `execution.mode` in the env overlay controls the backend: `paper_sim` | `paper_live` | `live`.
- `config/instruments.yaml` is NOT part of the layered config: it is a
  registry of contract metadata (asset class, strike, expiry, multiplier),
  resolved relative to the REPO ROOT (never the CWD) so it cannot vanish.
- **Live trading requires BOTH** `execution.mode: "live"` AND `execution.confirm_real_money: true` (guarded by `LiveTradingNotConfirmedError`).
- **Hot-reload**: `feature_engine`, `risk.circuit_breakers`, `labeling`, and `opportunity_filter.hard_rules` can be reloaded without restart via `ConfigWatcher`. `execution.mode`, `execution.confirm_real_money`, and `system.sandbox` are excluded — they require a restart.

## ML model quirks

- All models are **pickle-serialized LightGBM** (or HMM from hmmlearn). Paths live in `data/models/`.
- **Granularity mismatch → fallback mode**: if a model's saved granularity doesn't match `system.data_granularity`, the layer falls back to rule-based behavior (no crash).
- **Feature schema mismatch → fallback mode**: if a model's saved `feature_version` doesn't match `FEATURE_SCHEMA_VERSION` (currently `"v2_renamed"`), the layer falls back to rule-based behavior. This catches pre-Phase-5 models after the FeatureVector field rename.
- **Hard rules ALWAYS override ML** in both `OpportunityFilter` and `RiskLayer` — a hard-rule veto beats a model saying "tradable/approved".
- Models are **never auto-promoted**. `retrain_all.py` produces `pending_approval` versions; `approve_model.py` promotes. `rollback_model.py` restores previous.
- Walk-forward splits use **purge + embargo** (`WalkForwardSplitter`). Never use random/TimeSeriesSplit — it leaks labels.
- **Experiment log**: every backtest and training run is logged to `data/registry.db::experiments` with config hash, config snapshot, metrics, and git commit.

## Pipeline execution order (must not change)

```
MarketSnapshot → FeatureVector → RegimeState → ZoneMap →
  structural features merged back → ExpertOutputs → OpportunityScore →
  RouterDecision → RiskDecision → ExecutionRecord → SystemState
```

- Regime is computed **before** structure within a single bar (zones need `RegimeState.regime_relevance`).
- Risk layer is **only called for new entries**, never for position exits. Exit checking (TP/SL/timeout) happens before RiskLayer.evaluate() in `run.py`.
- **Decision tracing**: every bar produces a `DecisionTrace` with `StageResult` for each stage. Written to `data/logs/decision_traces/` as partitioned JSONL.

## Key constraints

- **No look-ahead**: every feature is computed only from information available at that bar's timestamp.
- **Minimum trade duration: 10 s, fill to exit** (`trade_duration.min_trade_seconds`).
  One rule in `src/labeling/hold_policy.py` + `barrier_exit_step`; the labeler, the
  live position and every model loader use it. Models carry a `training_alignment`
  contract and are refused under a different policy or when legacy. Details:
  `docs/explain/min-trade-duration.md`.
- **One concurrent position max** (`risk.circuit_breakers.max_concurrent_positions: 1`). Enforced in `process_bar` (`src/execution/position_lifecycle.py`): a bar only looks for an entry while nothing is open, so a value above 1 is accepted but still opens at most ONE position; `max_concurrent_positions()` raises `ValueError` for a value **below 1** (real multi-position support is a future item).
- **Trade rate limiting**: `max_trades_per_hour` (rolling 1-hour window) and `min_bars_between_orders` enforced in `RiskLayer.evaluate()`.
- **Circuit breakers persist** across restarts (SQLite in `data/registry.db`).
- **Kill switch**: `data/control/EMERGENCY_STOP` file halts the live loop.
- **Consistency validation**: `scripts/preflight.py` runs before session start — validates unit consistency, threshold plausibility, breaker enforcement paths, and feature schema version.
- `tests/conftest.py` holds the SHARED builders only (sys.path bootstrap, `paper_config`/`hold_config`, `make_journal_db` via the real schema, `synthetic_bars`, contract builders, and an autouse fixture that closes pooled SQLite handles). Per-file fixtures stay in their file.
- **Tests must not write to the repo's TRACKED `data/`.**
  `data/registry.db` is tracked, so a test that writes to it dirties the
  working tree; `data/raw|processed|logs|models` are gitignored. Run
  entry-point subprocesses with `cwd=` an isolated temp dir (and copy
  `config/` into it — the config and the registry are both CWD-relative).
- Lint and typecheck ARE configured: `ruff` in `pyproject.toml`
  (`[tool.ruff]`) and `mypy` (`[tool.mypy]`; `src/state` + `src/reporting`
  are strict). `make check` runs lint + typecheck + tests.

## The Trade Map GUI

- **One vocabulary.** The terminal stages (`TERMINAL_STAGES`), their labels
  (`STAGE_LABELS`) and their console styles (`STAGE_STYLES`) live in
  `src/diagnostics/decision_trace.py`. The TUI, the diagnostics CLI and the
  GUI all read them; `src/gui/theme.py` DERIVES its colours from the styles.
  Never add a stage in one place.
- **The GUI is a viewer.** It never places an order (replays are forced to
  `paper_sim`) and never writes under the tracked `data/`. Its risk-layer
  store is non-persistent (`RiskStateStore(persistent=False)`).
- **`src/` never imports `scripts/`**, and `src/gui/__init__.py` imports no
  charting library (Docker installs `requirements.txt`, which has no plotly).
- **SQLite is pooled.** `src.registry.db.connect()` yields one connection per
  file per thread (schema applied once, re-opened if the file is replaced).
  Callers must not close it; use `close_connection` / `close_all_connections`.
- Details: `docs/explain/trade-map-gui.md`.

### Living plan

`docs/TRADE_MAP_PLAN.md` is the **living plan** for the Trade Map GUI:
every phase, its tasks, its acceptance criteria and its status. Update
it **in the same commit as the phase's code** — the plan and the code
must never disagree. Phase 1 (the data/core layer) is done; Phase 1.5
(hardening) is next, then the chart layer, the app, the docs and the
final review.

## Deep-dive documentation

For implementation-level details on how every subsystem works — design decisions, failure paths, contracts, and assumptions — read the files in `docs/explain/`:

| File | Covers |
|---|---|
| `architecture.md` | Pipeline order, two-pass liquidity, granularity guard, config system |
| `states-and-contracts.md` | All 11 frozen dataclasses, serialization, `dataclasses.replace()` pattern |
| `layers-and-pipeline.md` | Every layer: regime detector, structure engine, opportunity filter, risk layer |
| `experts-and-routing.md` | Expert gate conditions, features, fallbacks, router hindsight training |
| `labeling-and-training.md` | Triple-barrier labeling, all 9 training pipelines, walk-forward CV |
| `risk-and-execution.md` | Circuit breakers, sizing formula, 3 backends, idempotent orders |
| `liquidity-and-market-logic.md` | Synthetic liquidity model, real orderbook adapter, fill simulation |
| `monitoring-and-feedback.md` | PSI drift, trade journal, zone event log, system state archive |
| `configuration-and-deployment.md` | Layered YAML, Docker, env vars, secrets handling |
| `failures-and-fallbacks.md` | Failure taxonomy, model fallback chains, known bugs |
| `alerting-and-network.md` | Telegram alerting, CCXT rate limiting, Docker retry backoff |
| `model-registry-and-feedback.md` | SQLite registry, promotion margin, feedback loop architecture |

## Retraining flow

```
retrain_all.py --input <ohlcv.csv> --config config/settings.base.yaml
  → trains each component via its train_*.py script
  → compares walk-forward metrics against champion
  → registers as pending_approval (never auto-promotes)

approve_model.py <component> <version_id>
  → promotes pending → approved_active

rollback_model.py <component> <version_id>
  → restores previous version
```

## Phase 5 additions

Phase 5 closed all Phase 4 gaps and added diagnostics, adaptive config, and performance intelligence:

- **Decision tracing**: `DecisionTrace` per bar, `StageResult` per stage, `diagnose.py` CLI
- **Cross-layer consistency**: `validate_config_consistency.py` catches unit mismatches, dead breakers, threshold disagreements
- **Regime-adaptive config**: `resolve_regime_param()` unifies all regime-dependent thresholds
- **Hot-reload**: `ConfigWatcher` reloads safe config paths without restart
- **Enforced limits**: `max_trades_per_hour`, `min_bars_between_orders`, `max_concurrent_positions` actually enforced
- **Experiment log**: `experiments` table in `data/registry.db`
- **Paper_live data collection**: paired (synthetic, real) liquidity samples logged for calibration
- **Live validation runbook**: `validate_live_backend.py` — scripted 5-step testnet procedure
- **run_live_loop()**: real implementation (replaced stub)
- **FeatureVector rename**: period-agnostic names (`momentum_short/medium/long`, `volatility_short/medium/long`)
- **FeatureEngine config-driven**: periods read from `feature_engine.*` config block

## Presentation-hardening additions (post-Phase 5)

Engineering-visibility pass per docs/PRESENTATION.md — infrastructure only,
no trading-behavior change:

- **Packaging**: `pyproject.toml` (deps mirror requirements.txt), Makefile
  (`make check` = lint+typecheck+tests), CONTRIBUTING.md, GitHub Actions CI.
- **Tests reorganized**: `tests/{unit,integration,e2e}/`; new
  `test_invariants.py` (financial invariants — extend when adding risk
  controls) and `test_no_lookahead.py` (leakage guards).
- **Structured logging**: `src/utils/logging_setup.py` — `setup_logging()`
  writes `data/logs/system.jsonl` (JSON lines); `log_trade_decision()`
  appends the §8-shaped per-bar record to `data/logs/decisions.jsonl`.
  Wired in scripts/run.py (both loops).
- **Latency instrumentation**: `TradingPipeline.last_bar_timings` per stage;
  `DecisionTrace.total_duration_ms` + `stage_timings_ms`;
  `StageResult.duration_ms`. Benchmark: `make latency`.
- **Trade replay**: `scripts/replay_trade.py --list / --trade-id` stitches
  journal + decision traces + OHLCV into a T±window stage-by-stage replay.
- **Ablation ladder**: `full_stack_walk_forward_backtest.py --single-expert
  <name> / --no-risk-layer / --no-regime-gating`; turnover metric
  (`turnover_x`) added to every finalize() block.
- **Paper-sim realism knobs** (default OFF, deterministic):
  `execution.simulated_latency_ms` (+ `latency_adverse_pct_per_100ms`) and
  `execution.order_rejection_rate`.
- **Docs**: ADRs in `docs/design-decisions/`, `docs/leakage-prevention.md`,
  `docs/limitations.md`, `docs/research-report.md` (results pending real
  data), figures via `scripts/generate_figures.py` → `docs/images/`,
  Streamlit demo `scripts/demo_app.py` (`make demo`).
- **Deferred real-data work**: tracked in root `TODO.md`.
