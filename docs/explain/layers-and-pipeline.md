# Layers & Pipeline

## Market Data Layer (`src/layers/market_data.py`)

`MarketDataLayer` converts raw exchange data into `MarketSnapshot` objects.

### `create_from_bar()`

Takes OHLCV row + optional ticker data → `MarketSnapshot`.

When no ticker is supplied (every call from training/backtest scripts), it synthesizes bid/ask from the bar's midpoint: `bid = mid × 0.999`, `ask = mid × 1.001`, with zero depth. This crude placeholder is explicitly documented as **not trustworthy** — Phase 4's `attach_synthetic_liquidity()` replaces it after features are computed.

### `batch_create_from_dataframe()`

Bulk conversion from a pandas DataFrame. Handles both `pd.Timestamp` and integer timestamps.

## Feature Engine (`src/layers/feature_engine.py`)

Maintains a `RollingBarBuffer(maxlen=100)` of bar dicts. Computes:

| Feature | Formula |
|---|---|
| `momentum_short/medium/long` | Return over N periods |
| `volatility_short/medium/long` | Std of returns over N periods |
| `returns_5` | 5-bar return |
| `rsi_14` | Relative Strength Index (14-period) |
| `macd_signal` | MACD signal line |
| `sma_9/21` | Simple moving averages |
| `imbalance_ratio` | Aggressive buy / total volume |
| `orderflow_momentum` | Cumulative delta momentum |
| `volume_zscore` | Volume z-score over 20 bars |

**Phase 5**: Periods are now read from `feature_engine.momentum_periods.{short,medium,long}` and `feature_engine.volatility_periods.{short,medium}` in config. Previously hardcoded.

**Known bug**: `volatility_short` (periods=1) computes `np.std()` over a single-element array, which is always 0.0. Triple-barrier labeling avoids this by using `volatility_medium` via `barrier_volatility_pct()`. The bug is documented in `PHASE3_COMPLETE.md` and left unfixed in `FeatureEngine` because fixing it requires touching already-tested Phase 1/2 code.

Requires 10+ bars before producing output. Returns `None` during warmup.

## Regime Detector (`src/layers/regime_detector.py`)

Two-part design:

### Part A: HMM Layer

3-state Gaussian HMM. Served via a **self-contained numpy forward filter** (not `hmmlearn`'s internal API) — `RegimeDetector` has zero hard dependency on `hmmlearn` at serving time. The manual Gaussian-emission math must be kept in sync if `covariance_type` changes away from `"full"` in the training script.

Forward-filtered only (never smoothed) at serving time — causal.

### Part B: LightGBM Classifier

Distilled from smoothed offline HMM labels. Served causally.

### Fallback Chain

```
LightGBM classifier (granularity check, feature_version check) → HMM filter → rule-based (momentum thresholds)
```

Always available even with no trained models present. Each step checks `granularity` mismatch and falls through.

### Panic Override

Independent of the HMM. Sigmoid-based `panic_score()` on volatility/momentum/volume z-scores. When the z-score exceeds `panic_z_threshold` (default 3.0), `prob_panic_spike` is boosted. The sigmoid ensures smooth transitions (verified by `test_panic_override_smooth` — max diff < 0.1).

### Anti-Flicker

`regime_age` and `regime_persistence_score` prevent rapid regime switching. A regime change requires the new regime to maintain `persistence_transition_threshold` (default 0.6) confidence for `persistence_window_bars` (default 20) consecutive bars.

### Serving Source

Configurable via `models.regime_serving_source`: `"auto"` | `"hmm_filter"` | `"lgbm_classifier"` | `"rule_based"`. Respected as a soft preference — if you force `"hmm_filter"` but no HMM is trained, it still falls back automatically.

## Structure Engine (`src/layers/structure_engine.py`)

### Zone Generation

**Pivot-based**: Local highs/lows confirmed by `pivot_window` (default 5) bars on each side. `confirmed_at_bar` is an engine-side shadow dict (not a `Zone` field) to avoid `contracts.py` schema changes.

**Volume-based**: High-volume bars also generate zones.

### Price-Action Interaction

When a bar's high/low touches a zone, it's classified as:
- **Rejection**: Bar touches zone but doesn't close beyond it
- **Breakout**: Bar closes beyond the zone

Classification uses actual bar highs/lows/closes, not just midpoints.

### Survival Probability

Beta-Bernoulli: `(rejection_count + 1) / (rejections + breakouts + 2)` × time_decay × optional ML blend.

The ML blend is controlled by `zone_survival_blend_weight` (default 0.0 — rule-based by default). Ramps in via config once the zone validity model is trained.

### Zone Lifecycle Events

Every zone transition is logged: `created`, `touched`, `rejected`, `broke`, `merged`, `decayed`. These are flushed to `data/logs/zone_events.parquet` by `flush_zone_events()`.

### Decay & Merging

- **Decay**: Untouched zones decay after `zone_decay_bars` (default 100). Removed when `survival_probability < 0.05`.
- **Merging**: Nearby zones of the same type merge when distance < `min_zone_separation_pct × 0.5`.

### Structural Feature Adapter

`attach_structural_features()` bridges `ZoneMap` → `FeatureVector`: adds `distance_to_nearest_support/resistance`, `support_strength`, `resistance_strength` to the feature vector. Uses `dataclasses.replace()` for immutable update.

## Expert Layer (`src/layers/expert_layer.py`)

Simple orchestrator: runs all three experts, assembles `ExpertOutputs`.

## Opportunity Filter (`src/layers/opportunity_filter.py`)

### Hard Rules (Always Veto ML)

```
tradable = hard_rules_pass AND (hard_rules_score * (1 - w) + ml_prob * w > 0.5)
```

A model can never override a failed hard rule — only add confidence within a pass.

| Rule | Default | Notes |
|---|---|---|
| `edge_vs_fees_threshold` | 0.02 | Expected edge must exceed fees + slippage |
| `max_spread_pct` | 0.05 | |
| `min_depth_usdt` | 10000 | **Fixed in Phase 4**: `depth_usdt = (bid_volume + ask_volume) × close` (was comparing BTC to USDT directly) |
| `volatility_floor_pct` | 0.01 | Reject dead markets |
| `volatility_ceiling_pct` | 1.5 | Reject chaotic conditions |
| `reject_on_regime` | [PANIC_SPIKE] | Block trades in panic |
| `no_consensus_ceiling` | 0.7 | Reject violent disagreement |
| `reject_if_best_confidence_below` | 0.3 | Reject weak signals |

### Regime Overrides

Per-regime hard rule adjustments. E.g., `PANIC_SPIKE` raises `edge_vs_fees_threshold` to 0.05. Keyed by `RegimeType` member NAME (e.g., `TRENDING_UP`), matching `parse_regime_list()` convention.

### Breakdown Scores

`liquidity_score`, `volatility_score`, `expert_consensus_score`, `regime_acceptability` — additive, backward-compatible fields on `OpportunityScore` for analysis.

### Known Issue

OHLCV-only replay produces `bid_volume = ask_volume = 0.0` and ~0.2% spread. Against `min_depth_usdt: 10000` and `max_spread_pct: 0.05`, **every bar fails** both checks. Phase 4's `attach_synthetic_liquidity()` partially resolves this, but the synthetic model's mean spread (~0.09%) still exceeds the default `max_spread_pct`. This needs real spread data for calibration.

## Router (`src/layers/router.py`)

### Fallback

Highest-confidence expert above `confidence_floor_to_trade` (default 0.3). If no expert qualifies, selects `no_trade`.

### ML Model

Multiclass LightGBM (classes: `trend_expert`, `mean_reversion_expert`, `breakout_expert`, `no_trade`). Blends model probabilities with expert confidence via `blend_weight` (default 0.2 — 20% model, 80% expert confidence initially).

### Switching Context

`bars_since_switch` tracked as a **soft feature** (not a hard rule). The hard switching limit (`min_bars_between_switches`) is in the Risk Layer, independently tracked from the Router's own soft tracking — deliberate duplication per the plan's §5.2.

## Risk Layer (`src/layers/risk_layer.py`)

### Kill Switch

File existence check (`data/control/EMERGENCY_STOP`) — checked **first**, zero dependencies. If present, returns `approved=False, rejection_reason="manual_kill_switch_active"`.

### Circuit Breakers (5 Total, Persisted in SQLite)

| Breaker | Trigger | Reset |
|---|---|---|
| `consecutive_execution_errors` | 3 consecutive failures | Manual |
| `max_drawdown` | Drawdown ≥ 8% | Manual |
| `consecutive_losses_hard` | 6 consecutive losses | Manual |
| `daily_loss_limit` | Daily loss ≥ 2% | Auto (new day) |
| `data_staleness` | No heartbeat for 120s | Auto (heartbeat resumes) |

All independently tripped/reset. Verified by `test_circuit_breaker_independent_triggers`. Persisted across restarts — `RiskStateStore(db_path)` creates/opens the SQLite DB, schema is auto-created.

### Hard Rules

| Rule | Default | Notes |
|---|---|---|
| `min_invalidation_distance_pct` | 0.1 | Entry must not be too close to invalidation |
| `max_invalidation_distance_pct` | 3.0 | Entry must not be too far |
| `min_bars_between_switches` | 3 | Soft switching limit |
| `reject_on_regime` | [PANIC_SPIKE] | Defense in depth vs. Opportunity Filter |

### Sizing Formula

```
risk_amount = equity × risk_pct_per_trade (0.5%)
base_size = risk_amount / (invalidation_distance / 100)
```

Capped at `max_position_pct_of_equity` (20%). Reduced by:
- **Liquidity curve**: Smooth ramp from `thin_threshold_usdt` (20K) to `full_size_threshold_usdt` (50K)
- **Soft pause**: Size halved after `consecutive_losses_soft_pause` (3) consecutive losses

### ML Risk Blend

Optional `P(stop-out)` model. Blend weight controlled by `risk.ml_model.blend_weight` (default 0.25).

### Trade Rate Limiting (Phase 5)

Two rate limits enforced in `evaluate()`:

| Limit | Config | Enforcement |
|---|---|---|
| `max_trades_per_hour` | Rolling 1-hour window | Rejects if count ≥ limit |
| `min_bars_between_orders` | Minimum bar gap | Rejects if bars since last order < limit |

Both read from `breaker_cfg` in config. `risk_layer.record_trade(timestamp)` is called after successful execution.

### Important: Never Called for Exits

`RiskLayer.evaluate()` is only called for new-entry candidates in `run.py`. Open-position exit checking (TP/SL/timeout) never calls into the Risk Layer. This is architectural, not just a code path.
