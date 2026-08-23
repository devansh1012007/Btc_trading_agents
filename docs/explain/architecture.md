# Architecture & Pipeline

## System Overview

A layered, multi-expert BTC/USDT scalping agent (30s–5m timeframes) that separates perception, opportunity validation, expert decision-making, routing, risk control, and execution into independent components. The system processes one bar at a time, producing a complete `SystemState` per bar.

## Pipeline Execution Order

```
MarketSnapshot → FeatureVector → RegimeState → ZoneMap
    → structural features merged back → ExpertOutputs
    → OpportunityScore → RouterDecision → RiskDecision
    → ExecutionRecord → SystemState
```

This order is **enforced** in `TradingPipeline._process_bar()` and must not change:

1. **Regime before structure** — zones need `RegimeState.regime_relevance` at the moment they're touched.
2. **Structural features merged back** into `FeatureVector` before experts see them.
3. **Risk layer only for new entries** — exits (TP/SL/timeout) are checked in `run.py` before `RiskLayer.evaluate()` is called.

## Two-Pass Liquidity Design

A chicken-and-egg problem exists: the synthetic liquidity model needs `volatility_5m` from the `FeatureVector`, but the `FeatureVector` computation needs a `MarketSnapshot` with bid/ask/depth fields.

**Resolution** (Phase 4, §4.2):
1. `MarketDataLayer.create_from_bar()` creates snapshot with crude bid/ask placeholders (mid×0.999/mid×1.001, zero depth).
2. Feature engine computes `FeatureVector` from OHLCV only.
3. `attach_synthetic_liquidity()` replaces `bid`, `ask`, `bid_volume`, `ask_volume`, `spread_pct` on the snapshot via `dataclasses.replace()`.
4. The corrected snapshot is used by the Opportunity Filter and Risk Layer.

The structure engine already consumed the pre-liquidity snapshot, which is safe — it never reads bid/ask/spread.

## Granularity Guard

Every trainable model carries a `granularity` field in its pickle payload. At load time, the layer checks it against `system.data_granularity` from config. A mismatch is treated as if the model file were absent — the layer falls back to rule-based behavior. This prevents silently using a model trained on a different timeframe.

**Affected models**: Regime HMM, regime classifier, zone survival, all 3 experts, opportunity filter, router.

## Shared State Contract

All layers read/write through frozen dataclasses defined in `src/state/contracts.py`. This is the single source of truth — no layer invents its own format. Every bar produces one `SystemState` containing all layer outputs.

## Config System

Layered YAML: `settings.base.yaml` (full config) merged with `settings.{env}.yaml` (overlay). The merge is deep — dicts merge key-by-key, lists are replaced wholesale.

- `scripts/run.py` and deployment scripts use `load_layered_config(env)` — always pass `--env`.
- Training/backtest scripts from Phase 2/3 read `settings.base.yaml` directly via `--config`.
- `execution.mode` in the env overlay controls the backend: `paper_sim` | `paper_live` | `live`.
- **Live trading requires BOTH** `execution.mode: "live"` AND `execution.confirm_real_money: true`. A single `--env live` is never enough.

## Training Scripts

All training scripts replay historical OHLCV through `replay_full_stack()` — the same code path `pipeline.py` uses at serving time. This prevents drift between training and production.

| Script | Label Source | Model |
|---|---|---|
| `train_regime_models.py` | HMM posteriors + panic override | HMM + LightGBM |
| `label_zone_events.py` → `train_zone_survival_model.py` | Zone touch events (rejection/breakout) | LightGBM |
| `train_triple_barrier.py` → `train_trend_expert.py` | Triple-barrier net return > 0 | LightGBM |
| `train_mean_reversion_expert.py` | Zone rejection events → triple-barrier | LightGBM |
| `train_breakout_expert.py` | Zone breakout events → triple-barrier | LightGBM |
| `train_opportunity_filter.py` | Best expert's triple-barrier return > 0 | LightGBM |
| `train_router.py` | Hindsight: which expert would have been best | LightGBM (multiclass) |
| `train_risk_model.py` | Stop-loss vs (take-profit/timeout) | LightGBM |

All models are pickle-serialized with `granularity` field and `.meta.json` model cards containing walk-forward fold metrics.

## Walk-Forward Splits

`WalkForwardSplitter` generates rolling train/test windows with purge + embargo gaps. This prevents label leakage: a label whose outcome window extends into the test set doesn't contaminate training data. Never use random or `TimeSeriesSplit` — it leaks.

## Model Registry

SQLite-backed lifecycle: `pending_approval` → `approved_active` → `superseded` (or `rejected`).

- `retrain_all.py` **never auto-promotes** — only creates `pending_approval`.
- `approve_model.py` is the sole promoter (human CLI).
- `rollback_model.py` restores a previous `approved_active` version.
- `get_champion()` compares walk-forward metrics against the current champion; `retrain_all.py` registers challengers that clear the `promotion_margin_pct` threshold.

## Known Limitations

1. `max_trades_per_hour` and `min_bars_between_orders` are configured in `settings.base.yaml` but **not enforced** in code.
2. `LiveBackend` and `reconciliation.py` tested against mocks only, never a real exchange.
3. `run_live_loop()` in `run.py` is a documented structural stub.
4. `volatility_30s` is always 0.0 (std of single-element array) — triple-barrier labeling uses `volatility_5m` instead via `barrier_volatility_pct()`.
5. Synthetic liquidity defaults may disagree with Opportunity Filter thresholds (needs real spread data for calibration).
