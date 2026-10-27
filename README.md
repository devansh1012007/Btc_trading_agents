# Btc_trading_agents

**A modular, regime-aware, multi-expert BTC trading research platform** —
built to study adaptive strategy selection, risk-aware execution, and
robustness under changing market conditions. Profitability is one
experimental outcome this system can measure, not the project's claim.

[![CI](https://github.com/devansh1012007/Btc_trading_agents/actions/workflows/ci.yml/badge.svg)](https://github.com/devansh1012007/Btc_trading_agents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-119%20passing-brightgreen)

---

## The Problem

Single-model trading systems fail in predictable ways:

- they overfit to one market regime and fall apart when it shifts,
- they can't tell "confident" from "right",
- they have no independent mechanism to say **no**,
- their backtests quietly leak future information.

This project treats those failure modes as first-class design constraints.

## The Design

Instead of one model, a layered decision engine for BTC/USDT scalping
(30s–5m), where each layer is independently trainable, testable, and
replaceable — and a veto anywhere kills the trade:

```
Market Data → Feature Engine → Regime Detection → Dynamic Structure
     → Experts (trend / mean-reversion / breakout)
     → Opportunity Filter (hard rules ⊕ ML)
     → Router (which expert, right now?)
     → Risk Layer (sizing, caps, circuit breakers)
     → Execution (3 backends) → Feedback (journal, traces, drift)
```

| Layer | Responsibility | Model |
|---|---|---|
| **Market Data** | Synchronized OHLCV + microstructure | Rule-based |
| **Feature Engine** | Momentum, volatility, orderflow, liquidity | Rule-based (config-driven periods) |
| **Dynamic Structure** | S/R zones with lifecycle + survival probability | Rule-based + LightGBM |
| **Regime Detection** | Trending/chop/panic probability vector | Gaussian HMM + LightGBM distillation |
| **Expert Layer** | 3 specialists, regime-gated candidates | LightGBM per expert |
| **Opportunity Filter** | "Is there enough edge to trade?" | Hard rules + LightGBM |
| **Router** | Which expert to trust right now | LightGBM |
| **Risk Layer** | Sizing, drawdown control, 5 circuit breakers | Hard rules + LightGBM |
| **Execution** | `paper_sim` / `paper_live` / `live` (CCXT) | Rule-based fills |
| **Feedback** | Trade journal, decision traces, PSI drift | — |

**Non-negotiable invariants** (pinned by tests): hard rules always veto
ML; one concurrent position; persisted breakers; kill switch checked
first; no look-ahead anywhere.

## Engineering Decisions

Documented as ADRs in [`docs/design-decisions/`](docs/design-decisions/README.md):

- [LightGBM over transformers](docs/design-decisions/adr-0001-lightgbm-over-transformer.md) for tabular decisions
- [Walk-forward + purge + embargo](docs/design-decisions/adr-0002-walk-forward-validation.md) — never random splits ([leakage map](docs/leakage-prevention.md))
- [Monolith on one laptop](docs/design-decisions/adr-0003-monolith-on-one-laptop.md) — measured: ~1 ms/bar decision latency, ~960 bars/sec
- [Hard rules veto ML](docs/design-decisions/adr-0004-hard-rules-veto-ml.md)
- [Rule-based execution first](docs/design-decisions/adr-0005-rule-based-execution-first.md); supervised router → bandit → (maybe) RL
- [Frozen dataclass contracts](docs/design-decisions/adr-0006-frozen-dataclass-contracts.md) between all layers

## Experiments

The benchmark harness (`scripts/full_stack_walk_forward_backtest.py`)
pushes every bar through the **same code path as live trading** and
reports realized, net-of-cost metrics against baselines and ablations:

```bash
make backtest           # real data: data/raw/BTCUSDT.csv
make benchmark-smoke    # synthetic ablation ladder (no data needed)
```

**Baselines:** buy & hold · simple momentum · naive highest-confidence
router · no-opportunity-filter.
**Ablations:** `--single-expert <name>` · `--no-risk-layer` ·
`--no-regime-gating`.

| Metric | Baselines | Full system |
|---|---|---|
| Net return · Sharpe · Sortino | ✅ | ✅ |
| Max drawdown · profit factor · win rate | ✅ | ✅ |
| Expectancy/trade · turnover · fees+slippage | ✅ | ✅ |

> **Results pending real-data run** — the harness, baselines, ablation
> ladder, and calibration reports are complete; published tables will be
> generated from real BTCUSDT history (see [`TODO.md`](TODO.md)). Every
> number this repo shows is reproducible from a single command; nothing
> is hand-written.

**Calibration** (§5.3): expert candidates are triple-barrier-labeled
independently of routing for reliability diagrams; router calibration
compares executed-trade confidence vs realized win.

## Explainability & Debugging

Every bar produces a **decision trace** — the stage-by-stage verdict of
the full pipeline:

```bash
python scripts/replay_trade.py --list            # browse trades
python scripts/replay_trade.py --trade-id <id>   # T-30→T+30 full replay
python scripts/diagnose.py --summary             # why didn't we trade?
python scripts/dashboard.py                      # live TUI (6 views)
python scripts/dashboard.py --once               # headless render
```

Example figures (`scripts/generate_figures.py`):

| | |
|---|---|
| ![regime overlay](docs/images/regime_overlay.png) | ![router timeline](docs/images/router_timeline.png) |

## Quick Start

```bash
git clone https://github.com/devansh1012007/Btc_trading_agents.git
cd Btc_trading_agents

# Option A: Docker
docker compose up          # historical replay, paper config

# Option B: local
make install-dev
source .venv/bin/activate
make check                 # lint + typecheck + full test suite
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity
```

Key config (`config/settings.{base,paper,live}.yaml`): `system.data_granularity`
is the only timeframe switch; `execution.mode` selects the backend;
**live orders require both `mode: "live"` and `confirm_real_money: true`**.

## Testing

```
tests/unit/         per-component + financial invariants + look-ahead guards
tests/integration/  full-stack wiring through the real pipeline
tests/e2e/          backtest entrypoint as subprocess, artifact checks
```

```bash
pytest tests/ -v        # everything
make unit && make integration && make e2e
make coverage           # with coverage report
```

Financial invariants (`tests/unit/test_invariants.py`): position caps,
risk-only-shrinks, kill-switch/breaker blocking, rate limits, duplicate-
order impossibility across restarts. Leakage guards
(`tests/unit/test_no_lookahead.py`): features are a pure function of the
past; embargo gaps enforced; labels see only their window.

## Repository Layout

```
src/                       pipeline, state contracts, layers/, experts/,
                           labeling/, backtest/, execution/, liquidity/,
                           registry/, feedback/, monitoring/, diagnostics/,
                           config/, connectors/, alerting/, gui/ (Trade Map GUI)
scripts/                   run.py (deploy), full_stack_walk_forward_backtest.py,
                           benchmark_latency.py, replay_trade.py, generate_figures.py,
                           train_*.py, retrain_all.py, approve/rollback_model.py,
                           preflight.py, validate_config_consistency.py, dashboard.py
config/                    settings.base.yaml + paper/live overlays
tests/                     unit / integration / e2e
docs/                      design-decisions/ (ADRs), explain/ (deep dives),
                           leakage-prevention.md, research-report.md, limitations.md,
                           TRADE_MAP_PLAN.md (the Trade Map GUI phase plan)
data/                      gitignored: raw/, models/, logs/, reports/, registry.db
```

## Trade Map GUI

An interactive view of **exactly where the system took trades, and why** —
TradingView-style candles at any timeframe the system supports, with entries,
exits, TP/SL levels, per-bar decisions (coloured by the stage that stopped the
bar), zones, regimes and equity, plus a "why did this bar (not) trade?" panel
that walks the stage-by-stage trace.

```bash
pip install -e ".[viz]"
make trade-map            # http://127.0.0.1:8572  (TRADE_MAP_PORT overrides)
make trade-map-export     # one self-contained HTML file, opens offline
```

Data comes from live artifacts first, then a replay of the raw OHLCV, then a
synthetic demo (always labelled illustrative). It is a viewer: it never places
an order and never writes to `data/`. See
[`docs/explain/trade-map-gui.md`](docs/explain/trade-map-gui.md) and, for what
it does *not* show, [`docs/limitations.md`](docs/limitations.md).

> Screenshot: run `make trade-map` (synthetic source) and save one to
> `docs/images/trade_map.png` — none is committed yet.

## Limitations

Honest list in [`docs/limitations.md`](docs/limitations.md): non-stationarity,
regime-shift lag, synthetic-liquidity fidelity, laptop-scale constraints,
label bias in supervised routing, single-symbol/single-position scope.
The system is explicitly **not** validated for live profitability yet —
paper-trading statistics are a prerequisite (tracked in `TODO.md`).

## Roadmap

- [x] Phases 1–5: perception, decision, risk/execution, diagnostics (see `docs/PHASE*_COMPLETE.md`)
- [ ] Trade Map GUI — see the living plan in [`docs/TRADE_MAP_PLAN.md`](docs/TRADE_MAP_PLAN.md)
- [ ] Real-data benchmark + ablation publication (`TODO.md`)
- [ ] Multi-week paper-trading campaign → failure-analysis report
- [ ] Contextual bandit router (then PPO only if bandits are beaten)
- [ ] Sequence model behind the existing expert contracts
- [ ] Multi-symbol partitioned storage

## Disclaimer

Research software. No profitability guarantee. Crypto trading involves
substantial risk of loss; models are trained on historical data and may
fail in live markets. Never deploy capital you cannot afford to lose.
Live mode requires explicit double confirmation and is guarded by a
kill switch (`data/control/EMERGENCY_STOP`) plus persisted circuit breakers.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — conventional commits,
`make check` before any PR, and the hard rules (pipeline order, no
look-ahead, no auto-promotion) are enforced in review and CI.
