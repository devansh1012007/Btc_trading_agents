# Btc_trading_agents

A modular, multi-expert BTC scalping trading agent with regime-aware routing, dynamic market structure detection, and a full risk/execution/feedback pipeline.

## Overview

This project builds a layered decision engine for BTC/USDT scalping (30s–5m timeframes). Instead of a single monolithic model, the system separates perception, opportunity validation, expert decision-making, routing, risk control, and execution into independent, composable layers. Each layer can be trained, evaluated, and retrained independently.

## Why This Approach

Single-model trading systems tend to overfit to one market regime, produce confusing activity in bad conditions, and lack a safety mechanism that can override a confident but wrong prediction. This architecture addresses those limitations by:

- **Specialized experts** — each trained on the market conditions where it has edge
- **Regime detection** — the system knows whether it's trending, chopping, or panicking before deciding
- **Opportunity filtering** — hard rules reject trades where fees/slippage exceed expected edge
- **Independent risk layer** — circuit breakers and position sizing can veto any decision
- **Adaptive retraining** — walk-forward splits with purge + embargo prevent look-ahead leakage

## Architecture

```
Market Data → Feature Engine → Dynamic Structure → Regime Detection
     → Expert Layer → Opportunity Filter → Router → Risk Layer → Execution → Feedback
```

| Layer | Responsibility | Model |
|---|---|---|
| **Market Data** | Synchronized OHLCV + microstructure | Rule-based |
| **Feature Engine** | Momentum, volatility, orderflow, liquidity features | Rule-based (config-driven periods) |
| **Dynamic Structure** | Support/resistance zones with lifecycle tracking | Rule-based + LightGBM (zone validity) |
| **Regime Detection** | Trending / Chop / Panic probability vector | HMM + LightGBM |
| **Expert Layer** | Trend, Mean Reversion, Breakout signals | LightGBM (per expert) |
| **Opportunity Filter** | "Is there enough edge to trade?" | Hard rules + LightGBM |
| **Router** | Which expert to trust right now | LightGBM |
| **Risk Layer** | Position sizing, drawdown control, circuit breakers | Hard rules + LightGBM |
| **Execution** | Order placement with slipage simulation | Rule-based |
| **Feedback** | Trade journal, zone events, system state archive | — |

## Key Features

**Implemented (Phases 1–5):**

- **Triple-barrier labeling** — TP/SL/timeout resolution for training data
- **Dynamic structure engine** — pivot-based support/resistance zones with rejection/breakout classification and Beta-Bernoulli survival probability
- **Regime detection** — 3-state Gaussian HMM with panic-spike override and LightGBM distillation classifier
- **3 experts** — Trend, Mean Reversion, Breakout with regime-gated candidate selection
- **Opportunity filter** — hard-rule veto (spread, depth, volatility, regime) with ML blend
- **Router** — confidence-based expert selection with switching context tracking
- **Risk layer** — invalidation-distance sizing, 5 circuit breakers (SQLite-persisted), manual kill switch
- **3 execution backends** — `paper_sim`, `paper_live`, `live` (CCXT-based)
- **Synthetic liquidity model** — feature-conditioned spread/depth proxy for backtesting
- **Model registry** — SQLite-backed version lifecycle (pending → approved → superseded)
- **Walk-forward training** — purge + embargo splits via `WalkForwardSplitter`
- **Drift monitoring** — Population Stability Index (PSI) for feature distribution shifts
- **Layered config** — `settings.base.yaml` + env overlays (`paper`/`live`)
- **Decision tracing** — stage-by-stage bar traces with `diagnose.py` CLI
- **Cross-layer consistency validation** — catches unit mismatches, dead breakers, threshold disagreements
- **Regime-adaptive config** — unified `resolve_regime_param()` for all regime-dependent thresholds
- **Hot-reload** — safe config paths reload without restart via `ConfigWatcher`
- **Enforced limits** — `max_trades_per_hour`, `min_bars_between_orders`, `max_concurrent_positions` actually enforced
- **Experiment log** — every backtest/training run logged to registry.db
- **Live validation runbook** — scripted 5-step testnet procedure
- **FeatureVector rename** — period-agnostic names (`momentum_short/medium/long`, `volatility_short/medium/long`)
- **FeatureEngine config-driven** — periods read from `feature_engine.*` config block
- **TUI signal console** — 6-view terminal dashboard (funnel, journal, performance vs targets, calibration, risk/faults, registry) with auto-refresh and a TTY-less `--once` renderer

**Planned (not yet implemented):**

- Contextual bandit / PPO for adaptive router selection
- TransformerEncoder for sequence-based market representation
- Graph neural networks for zone lifecycle modeling
- Multi-symbol support with partitioned storage
- Real-time alerting (Telegram webhook wired, off by default)

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML** | LightGBM, hmmlearn, scikit-learn |
| **Data** | pandas, numpy, pyarrow |
| **Exchange** | CCXT (Binance sandbox/live) |
| **Backtesting** | vectorbt |
| **Config** | PyYAML (layered settings) |
| **Testing** | pytest |
| **Serialization** | Parquet (Snappy), pickle (models) |

## Local Setup

### Prerequisites

- Python 3.10 or later
- pip or venv

### Installation

```bash
# Clone the repository
git clone https://github.com/devansh1012007/Btc_trading_agents.git
cd Btc_trading_agents

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The system uses layered YAML config:

- `config/settings.base.yaml` — full configuration (all layers)
- `config/settings.paper.yaml` — paper trading overlay (sandbox=true, paper_sim backend)
- `config/settings.live.yaml` — live trading overlay (sandbox=false, live backend)

Key config to review before running:

| Setting | Location | Default | Notes |
|---|---|---|---|
| `system.data_granularity` | base.yaml | `"1m"` | Timeframe for all layers |
| `labeling.fee_pct` | base.yaml | `0.04` | Binance taker fee |
| `labeling.max_holding_bars` | base.yaml | `5` | Trade resolution window |
| `execution.mode` | paper/live.yaml | `paper_sim` | Execution backend |
| `execution.confirm_real_money` | paper/live.yaml | `false` | **Must be `true` for live orders** |
| `feature_engine.momentum_periods` | base.yaml | `{short: 1, medium: 5, long: 10}` | Config-driven (Phase 5) |
| `risk.circuit_breakers.max_trades_per_hour` | base.yaml | `10` | Enforced (Phase 5) |

### Running

```bash
# Run the full test suite
pytest tests/ -v

# Pipeline dry-run (fetches 100 bars from Binance sandbox)
python -m src.pipeline

# Historical backtest through the full stack
python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper

# Full-stack performance report vs Project_Plan.md targets
python scripts/full_stack_walk_forward_backtest.py --input data/raw/BTCUSDT.csv --env paper

# Live loop (requires real credentials + confirm_real_money=true)
python scripts/run.py --mode live --env live

# Diagnostics: summary of why bars didn't trade
python scripts/diagnose.py --summary --since 2026-08-01

# Diagnostics: full trace for a specific bar
python scripts/diagnose.py --timestamp 1755900000

# Pre-flight validation before starting a session
python scripts/preflight.py --env paper

# Config consistency validation
python scripts/validate_config_consistency.py

# TUI dashboard (live, 6 views, auto-refresh) / headless one-shot render
python scripts/dashboard.py            # keys: 1-6 views · r refresh · q quit
python scripts/dashboard.py --once
```

### Training Models

```bash
# Retrain all components (never auto-promotes)
python scripts/retrain_all.py --input data/raw/BTCUSDT.csv --config config/settings.base.yaml

# Promote a pending model after review
python scripts/approve_model.py <component> <version_id>

# Rollback to a previous version
python scripts/rollback_model.py <component> <version_id>
```

## Testing

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_phase3.py -v

# Single test
pytest tests/test_phase3.py::test_triple_barrier_tp_hit_first -v
```

No `conftest.py` — all fixtures are inline helper functions within each test file.

## Project Structure

```
src/
  pipeline.py              ← TradingPipeline orchestrator
  state/contracts.py       ← All shared dataclasses (frozen, single source of truth)
  state/serialization.py   ← Dict/Parquet serialization for every dataclass
  layers/                  ← Market data, features, regime, structure, experts,
                             opportunity filter, router, risk layer
  experts/                 ← Trend, Mean Reversion, Breakout experts
  labeling/                ← Triple-barrier labeling for training data
  backtest/                ← Full-stack replay, walk-forward splitter
  execution/               ← Execution engine + paper_sim/paper_live/live backends
  liquidity/               ← Synthetic liquidity model + real orderbook adapter
  registry/                ← SQLite model registry (pending → approved → superseded)
  feedback/                ← Trade journal, zone event log, system state archive
  monitoring/              ← PSI drift monitor, performance monitor
  connectors/              ← CCXT exchange connector
  alerting/                ← Telegram notifier (off by default)
  diagnostics/             ← Decision trace + fault log (Phase 5)
  config/                  ← Regime-adaptive + hot-reload (Phase 5)
config/
  settings.base.yaml       ← Full configuration
  settings.paper.yaml      ← Paper trading overlay
  settings.live.yaml       ← Live trading overlay
scripts/
  run.py                   ← Main deployment entry (historical replay + live loop)
  train_*.py               ← Per-component training scripts
  retrain_all.py           ← Cron retraining (never auto-promotes)
  approve_model.py         ← Manual model promotion
  rollback_model.py        ← Restore previous model version
  status.py                ← Health check
  diagnose.py              ← CLI diagnostics (Phase 5)
  validate_config_consistency.py ← Pre-flight validation (Phase 5)
  preflight.py             ← Session pre-flight (Phase 5)
  validate_live_backend.py ← Testnet validation runbook (Phase 5)
tests/
  test_phase1.py           ← Market data, feature engine
  test_phase2.py           ← Structure engine + regime detection
  test_phase3.py           ← Triple-barrier, experts, opportunity filter, router
  test_phase4.py           ← Risk layer, execution, circuit breakers, registry
  test_integration.py      ← Full-stack integration
docs/
  PHASE1_COMPLETE.md       ← Phase 1 implementation report
  PHASE2_COMPLETE.md       ← Phase 2 implementation report
  PHASE3_COMPLETE.md       ← Phase 3 implementation report
  PHASE4_COMPLETE.md       ← Phase 4 implementation report
  PHASE5_COMPLETE.md       ← Phase 5 implementation report
data/                      ← Gitignored: raw/, processed/, models/, logs/
  logs/decision_traces/    ← Partitioned JSONL traces (Phase 5)
  faults.jsonl             ← System fault records (Phase 5)
```

## Roadmap

### Phase 1 — Market Data & Features (Done)
OHLCV ingestion via CCXT, feature engineering (momentum, volatility, RSI, MACD, orderflow, liquidity).

### Phase 2 — Perception Layer (Done)
Dynamic structure engine (pivot-based zones, survival probability) + regime detection (HMM + LightGBM).

### Phase 3 — Decision Layer (Done)
Triple-barrier labeling, expert layer (3 experts), opportunity filter (hard rules + ML), router (expert selection).

### Phase 4 — Risk, Execution, Feedback (Done)
Risk layer with circuit breakers, 3 execution backends, synthetic liquidity, model registry, drift monitoring, layered config.

### Phase 5 — Diagnostics, Adaptive Config & Performance Intelligence (Done)
Decision tracing, cross-layer consistency validation, regime-adaptive config, hot-reload, enforced rate limits, experiment log, live validation runbook, real `run_live_loop()`, FeatureVector field rename, FeatureEngine config-driven.

### Future
- Contextual bandit / PPO router (adaptive expert selection)
- TransformerEncoder for sequence-based features
- Graph neural networks for zone lifecycle
- Multi-symbol support
- Real-time alerting (Telegram)
- Intigrating web jooks for live trading as soon as profitabiliy is atchived

## Disclaimer

This is a research/trading system. It does not guarantee profitability. Trading cryptocurrency involves substantial risk of loss. The models in this repository are trained on historical data and may not perform well in live markets. Use at your own risk. Never deploy capital you cannot afford to lose.
