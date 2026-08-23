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
| **Feature Engine** | Momentum, volatility, orderflow, liquidity features | Rule-based |
| **Dynamic Structure** | Support/resistance zones with lifecycle tracking | Rule-based + LightGBM (zone validity) |
| **Regime Detection** | Trending / Chop / Panic probability vector | HMM + LightGBM |
| **Expert Layer** | Trend, Mean Reversion, Breakout signals | LightGBM (per expert) |
| **Opportunity Filter** | "Is there enough edge to trade?" | Hard rules + LightGBM |
| **Router** | Which expert to trust right now | LightGBM |
| **Risk Layer** | Position sizing, drawdown control, circuit breakers | Hard rules + LightGBM |
| **Execution** | Order placement with slipage simulation | Rule-based |
| **Feedback** | Trade journal, zone events, system state archive | — |

## Key Features

**Implemented (Phases 1–4):**

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

### Running

```bash
# Run the full test suite
pytest tests/ -v

# Pipeline dry-run (fetches 100 bars from Binance sandbox)
python -m src.pipeline

# Historical backtest through the full stack
python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper

# Live loop (requires real credentials + confirm_real_money=true)
python scripts/run.py --mode live --env live
```

### Training Models

```bash
# Retrain all components (never auto-promotes)
python scripts/retrain_all.py --input data/raw/BTCUSDT.csv --config config/settings.base.yaml

# Promote a pending model after review
python scripts/approve_model.py <component> <version_id>

# Rollback to a previous version
python scripts.rollback_model.py <component> <version_id>
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
tests/
  test_phase2.py           ← Structure engine + regime detection
  test_phase3.py           ← Triple-barrier, experts, opportunity filter, router
  test_phase4.py           ← Risk layer, execution, circuit breakers, registry
docs/
  PHASE1_COMPLETE.md       ← Phase 1 implementation report
  PHASE2_COMPLETE.md       ← Phase 2 implementation report
  PHASE3_COMPLETE.md       ← Phase 3 implementation report
  PHASE4_COMPLETE.md       ← Phase 4 implementation report
data/                      ← Gitignored: raw/, processed/, models/, logs/
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

### Future
- Contextual bandit / PPO router (adaptive expert selection)
- TransformerEncoder for sequence-based features
- Graph neural networks for zone lifecycle
- Multi-symbol support
- Real-time alerting (Telegram)

## Disclaimer

This is a research/trading system. It does not guarantee profitability. Trading cryptocurrency involves substantial risk of loss. The models in this repository are trained on historical data and may not perform well in live markets. Use at your own risk. Never deploy capital you cannot afford to lose.
