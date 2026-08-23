# Btc_trading_agents — Agent Instructions

## Repo layout

```
src/
  pipeline.py          ← TradingPipeline (main orchestrator)
  state/contracts.py   ← all shared dataclasses (frozen, single source of truth)
  state/serialization.py
  layers/              ← market_data, feature_engine, regime_detector, structure_engine,
                         expert_layer, opportunity_filter, router, risk_layer
  experts/             ← trend, mean_reversion, breakout (+ base_expert, expert_registry)
  labeling/            ← triple_barrier labeling
  backtest/            ← replay.py, walk_forward.py
  execution/           ← execution_engine + paper_sim/paper_live/live backends
  liquidity/           ← synthetic_liquidity_model
  registry/            ← model registry (SQLite: pending → approved → superseded)
  feedback/            ← trade_journal, zone_event_log, system_state_archive
  monitoring/          ← drift_monitor (PSI), performance_monitor
  connectors/          ← CCXTConnector
  alerting/            ← notifier
config/
  settings.base.yaml   ← full config (all layers)
  settings.paper.yaml  ← env overlay (default)
  settings.live.yaml   ← env overlay
scripts/
  run.py               ← deployment entry (historical replay + live loop)
  train_*.py           ← per-component training
  retrain_all.py       ← cron retraining (never auto-promotes)
  approve_model.py     ← manual promotion in model registry
  rollback_model.py
  reset_breaker.py
  status.py
tests/
  test_phase2.py, test_phase3.py, test_phase4.py
  Phase1_tests/        ← legacy phase 1 tests
data/                  ← gitignored: raw/, processed/, models/, logs/
```

## Running

```bash
# Tests
pytest tests/ -v

# Single test file
pytest tests/test_phase3.py -v

# Single test
pytest tests/test_phase3.py::test_triple_barrier_tp_hit_first -v

# Pipeline dry-run (fetches 100 bars from Binance sandbox)
python -m src.pipeline

# Historical backtest through full stack
python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper

# Live loop (requires real credentials + confirm_real_money=true)
python scripts/run.py --mode live --env live
```

## Config system (easy to get wrong)

- **Layered YAML**: `load_layered_config(env)` merges `settings.base.yaml` + `settings.{env}.yaml` (deep merge, overlay wins).
- `scripts/run.py` and deployment scripts use `load_layered_config(args.env)` — **always pass `--env`**.
- Training/backtest scripts from Phase 2/3 are offline utilities; they read `settings.base.yaml` directly via `--config`.
- `system.data_granularity` (e.g. `"1m"`, `"30s"`) is the **only** switch for timeframe — no code change needed. All layers check it for model-granularity compatibility.
- `execution.mode` in the env overlay controls the backend: `paper_sim` | `paper_live` | `live`.
- **Live trading requires BOTH** `execution.mode: "live"` AND `execution.confirm_real_money: true` (guarded by `LiveTradingNotConfirmedError`).

## ML model quirks

- All models are **pickle-serialized LightGBM** (or HMM from hmmlearn). Paths live in `data/models/`.
- **Granularity mismatch → fallback mode**: if a model's saved granularity doesn't match `system.data_granularity`, the layer falls back to rule-based behavior (no crash).
- **Hard rules ALWAYS override ML** in both `OpportunityFilter` and `RiskLayer` — a hard-rule veto beats a model saying "tradable/approved".
- Models are **never auto-promoted**. `retrain_all.py` produces `pending_approval` versions; `approve_model.py` promotes. `rollback_model.py` restores previous.
- Walk-forward splits use **purge + embargo** (`WalkForwardSplitter`). Never use random/TimeSeriesSplit — it leaks labels.

## Pipeline execution order (must not change)

```
MarketSnapshot → FeatureVector → RegimeState → ZoneMap →
  structural features merged back → ExpertOutputs → OpportunityScore →
  RouterDecision → RiskDecision → ExecutionRecord → SystemState
```

- Regime is computed **before** structure within a single bar (zones need `RegimeState.regime_relevance`).
- Risk layer is **only called for new entries**, never for position exits. Exit checking (TP/SL/timeout) happens before RiskLayer.evaluate() in `run.py`.

## Key constraints

- **No look-ahead**: every feature is computed only from information available at that bar's timestamp.
- **One concurrent position max** (`risk.circuit_breakers.max_concurrent_positions: 1`).
- **Circuit breakers persist** across restarts (SQLite in `data/registry.db`).
- **Kill switch**: `data/control/EMERGENCY_STOP` file halts the live loop.
- Tests have **no conftest.py** — all fixtures are inline helper functions in each test file.
- No lint/typecheck/format config exists. Just `pytest`.

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
