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
4. **Decision tracing** — every bar produces a `DecisionTrace` with `StageResult` per stage. Written to `data/logs/decision_traces/` as partitioned JSONL.

## Two-Pass Liquidity Design

A chicken-and-egg problem exists: the synthetic liquidity model needs `volatility_medium` from the `FeatureVector`, but the `FeatureVector` computation needs a `MarketSnapshot` with bid/ask/depth fields.

**Resolution** (Phase 4, §4.2):
1. `MarketDataLayer.create_from_bar()` creates snapshot with crude bid/ask placeholders (mid×0.999/mid×1.001, zero depth).
2. Feature engine computes `FeatureVector` from OHLCV only.
3. `attach_synthetic_liquidity()` replaces `bid`, `ask`, `bid_volume`, `ask_volume`, `spread_pct` on the snapshot via `dataclasses.replace()`.
4. The corrected snapshot is used by the Opportunity Filter and Risk Layer.

The structure engine already consumed the pre-liquidity snapshot, which is safe — it never reads bid/ask/spread.

## Granularity Guard

Every trainable model carries a `granularity` field in its pickle payload. At load time, the layer checks it against `system.data_granularity` from config. A mismatch is treated as if the model file were absent — the layer falls back to rule-based behavior. This prevents silently using a model trained on a different timeframe.

**Affected models**: Regime HMM, regime classifier, zone survival, all 3 experts, opportunity filter, router.

## Feature Schema Guard (Phase 5)

Every trainable model also carries a `feature_version` field in its pickle payload. At load time, the layer checks it against `FEATURE_SCHEMA_VERSION` (currently `"v2_renamed"`) from `contracts.py`. A mismatch is treated as if the model file were absent — the layer falls back to rule-based behavior.

This catches the case where FeatureVector field names changed (Phase 5 rename: `momentum_5m` → `momentum_medium`, etc.) but old models still reference the old names. Without this guard, `getattr(fv, "momentum_5m", 0.0)` would silently return 0.0.

**Affected models**: All models trained before Phase 5 will fall back to rule-based mode until manually retrained.

## Shared State Contract

All layers read/write through frozen dataclasses defined in `src/state/contracts.py`. This is the single source of truth — no layer invents its own format. Every bar produces one `SystemState` containing all layer outputs.

## Config System

Layered YAML: `settings.base.yaml` (full config) merged with `settings.{env}.yaml` (overlay). The merge is deep — dicts merge key-by-key, lists are replaced wholesale.

- `scripts/run.py` and deployment scripts use `load_layered_config(env)` — always pass `--env`.
- Training/backtest scripts from Phase 2/3 read `settings.base.yaml` directly via `--config`.
- `execution.mode` in the env overlay controls the backend: `paper_sim` | `paper_live` | `live`.
- **Live trading requires BOTH** `execution.mode: "live"` AND `execution.confirm_real_money: true`. A single `--env live` is never enough.
- **Hot-reload (built, not wired)**: `src/config/config_watcher.py` implements mtime-based reloading of a whitelisted safe subset (`feature_engine`, `risk.circuit_breakers`, `labeling`, `opportunity_filter.hard_rules`; `execution.*`/`system.sandbox` excluded by design), but **no loop instantiates `ConfigWatcher` today** — config changes require a restart until it's wired into `run.py`.

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

All models are pickle-serialized with `granularity`, `feature_version`, and `.meta.json` model cards containing walk-forward fold metrics.

## Walk-Forward Splits

`WalkForwardSplitter` generates rolling train/test windows with purge + embargo gaps. This prevents label leakage: a label whose outcome window extends into the test set doesn't contaminate training data. Never use random or `TimeSeriesSplit` — it leaks.

## Model Registry

SQLite-backed lifecycle: `pending_approval` → `approved_active` → `superseded` (or `rejected`).

- `retrain_all.py` **never auto-promotes** — only creates `pending_approval`.
- `approve_model.py` is the sole promoter (human CLI).
- `rollback_model.py` restores a previous `approved_active` version.
- `get_champion()` compares walk-forward metrics against the current champion; `retrain_all.py` registers challengers that clear the `promotion_margin_pct` threshold.

## Decision Tracing (Phase 5)

Every bar produces a `DecisionTrace` with `StageResult` for each stage:

```python
@dataclass(frozen=True)
class StageResult:
    stage_name: str          # "trend_expert_gate", "opportunity_filter", "router", "risk_layer"
    passed: bool
    detail: str              # human-readable: "momentum 0.03% below threshold 0.05%"
    values: Dict[str, float] # actual numbers, for reproducibility
    duration_ms: Optional[float] = None  # wall-clock stage latency (PRESENTATION §23)

@dataclass(frozen=True)
class DecisionTrace:
    timestamp: int
    symbol: str
    stages: List[StageResult]     # every stage the bar passed through, in order
    terminal_stage: str            # where the journey ended
    terminal_reason: str
    total_duration_ms: Optional[float] = None            # end-to-end decision latency
    stage_timings_ms: Optional[Dict[str, float]] = None   # pipeline-internal per-stage latencies
```

Written to `data/logs/decision_traces/` as partitioned JSONL (high volume, append-only, read with pandas). Kept separate from the trade journal (completed trades only) and the full `SystemState` Parquet archive (complete numeric record, optimized for training) — this is specifically the human-readable "why" narrative, optimized for debugging.

## Observability Streams

Three append-only JSONL streams, each serving a different consumer:

| Stream | Contents | Consumer |
|---|---|---|
| `data/logs/system.jsonl` | All log lines as JSON | `setup_logging()` in `src/utils/logging_setup.py`; wired into both loops of `scripts/run.py` |
| `data/logs/decisions.jsonl` | One §8-shaped record per bar (regime/expert/opportunity/router/risk/decision) | `log_trade_decision()`; feed for external analysis |
| `data/logs/decision_traces/` | Full stage-by-stage narrative + latency fields | `diagnose.py`, `replay_trade.py`, dashboard |

> **HEAD caveat**: `src/utils/logging_setup.py` and the pipeline/trace
> latency fields referenced above are currently present only in the stale
> `tests/src/` mirror — see "Source-Tree Regression" in
> [failures-and-fallbacks.md](failures-and-fallbacks.md).

**Latency instrumentation** (intended design): `_process_bar()` times each stage into
`TradingPipeline.last_bar_timings` (`features_ms`, `regime_ms`,
`structure_ms`, `liquidity_ms`, `decisions_ms`); the same numbers flow into
`DecisionTrace.stage_timings_ms` / `total_duration_ms`. Benchmark:
`scripts/benchmark_latency.py` (`make latency`). Per-trade stage replay:
`scripts/replay_trade.py --trade-id <id>`.

Presentation tooling that reads these artifacts: TUI dashboard
(`scripts/dashboard.py`, 6 views, `--once` for headless render), CLI
diagnostics (`scripts/diagnose.py`), trade replay
(`scripts/replay_trade.py`), figure generation
(`scripts/generate_figures.py` → `docs/images/`, labeled ILLUSTRATIVE on
synthetic data), and a Streamlit demo (`scripts/demo_app.py`).

## Backtest Harness & Ablations

`scripts/full_stack_walk_forward_backtest.py` runs the full stack (fixed models + window segmentation) against 4 baselines (`buy_and_hold`, `simple_momentum`, `naive_router`, `unfiltered`) and logs every run to `data/registry.db::experiments` (config hash + snapshot + git commit).

Ablation toggles isolate each component's contribution:

```bash
python scripts/full_stack_walk_forward_backtest.py --input data/raw/BTCUSDT.csv --env paper
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity   # smoke
  --single-expert trend_expert      # only one expert active
  --no-risk-layer                   # risk layer auto-approves at base size
  --no-regime-gating                # strips regime requirements + reject_on_regime
```

Every `finalize()` emits `turnover_x` alongside returns/drawdown metrics.

## Regime-Adaptive Config (Phase 5)

One unified mechanism for all regime-dependent thresholds:

```python
def resolve_regime_param(param_block: dict, current_regime: RegimeType) -> float:
    return param_block.get("by_regime", {}).get(current_regime.name, param_block["default"])
```

**Status: built and tested, not yet adopted.** Production code still reads the two older patterns — `labeling.regime_specific_barriers` (plain per-regime dicts in `run.py`, `replay.py`, every training script) and `opportunity_filter.regime_overrides` (dict-merged over `hard_rules` in `OpportunityFilter._effective_hard_rules()`). The `default`/`by_regime` block shape is the intended end state for risk sizing, barrier sensitivity, and filter thresholds; migrating call sites to it is deferred work.

## Known Limitations

1. **Source-tree regression at HEAD**: `scripts/run.py`, `full_stack_walk_forward_backtest.py`, and `benchmark_latency.py` depend on five pieces that exist only in the stale `tests/src/` mirror, not canonical `src/` (`logging_setup.py`, trace-latency fields, `last_bar_timings`, `compute_turnover`, paper-sim realism knobs). Those entrypoints crash until restored. Full table + recovery path: [failures-and-fallbacks.md](failures-and-fallbacks.md).
2. **`consecutive_losses_hard` breaker is dead**: Configured as `consecutive_losses_hard_halt: 6` in `settings.base.yaml`, but no production code ever calls `trip("consecutive_losses_hard")`. The consistency validator raises `ConfigConsistencyError` on it.
3. **Preflight fails on shipped defaults**: the threshold-plausibility simulation reports ~98% of bars would fail `max_spread_pct=0.05` under the shipped synthetic-liquidity coefficients — an intentional tripwire pending real-spread calibration.
4. **Pre-rename models fall back to rule-based**: All models trained before Phase 5 have `feature_version` pre-rename. They will silently fall back to rule-based mode until manually retrained.
5. **`max_concurrent_positions > 1`**: Raises a clear error if configured > 1. Real multi-position support is a named future item.
6. **Synthetic liquidity defaults disagree with Opportunity Filter thresholds**: Needs real spread data from `paper_live` mode for calibration.
7. **Monitoring/alerting/hot-reload are unwired**: drift monitor, expectancy-gap monitor, Telegram notifier, and ConfigWatcher are implemented and tested but no loop or schedule invokes them.
