# Labeling & Model Training

## Triple-Barrier Labeling (`src/labeling/triple_barrier.py`)

### `label_triple_barrier()`

Walks `forward_bars[:max_holding_bars]` for each bar. Checks three exit conditions:

1. **Take Profit**: Price hits `entry_price × (1 + take_profit_pct/100)` (long) or `entry_price × (1 - take_profit_pct/100)` (short)
2. **Stop Loss**: Price hits `entry_price × (1 - stop_loss_pct/100)` (long) or `entry_price × (1 + stop_loss_pct/100)` (short)
3. **Timeout**: `max_holding_bars` bars pass without TP or SL hit

**Conservative convention**: If a bar touches both TP and SL, stop-loss is assumed hit first. This prevents overestimating win rates.

### `compute_barrier_pcts()`

```
take_profit_pct = k_tp × volatility_pct
stop_loss_pct = k_sl × volatility_pct
```

### `barrier_volatility_pct()`

Uses `volatility_5m` (a genuinely multi-sample, well-defined dispersion statistic) instead of `volatility_30s` which is always 0.0 (std of single-element array). Applied at all six call sites (every training script, the backtest harness, and the shared replay helper) so there's one place this decision lives.

### `TripleBarrierOutcome`

| Field | Type | Notes |
|---|---|---|
| `exit_reason` | str | "take_profit", "stop_loss", "timeout" |
| `bars_held` | int | |
| `realized_return_pct` | float | Net of fees (fee × 2) |
| `max_adverse_excursion_pct` | float | Peak adverse move during holding |

## Training Pipelines

### Regime Models (`scripts/train_regime_models.py`)

**Step 1: HMM Fit**
- Grid search over n_states (2–7) using BIC
- Final model fitted with n_states=3 (per config)
- Gaussian emissions, full covariance

**Step 2: State Labeling**
- Offline smoothed HMM labels → map states to regime names
- Mapping: {2: 'trending_up', 0: 'trending_down', 1: 'chop_range'} (example — depends on BIC winner)
- Panic override applied post-hoc

**Step 3: LightGBM Distillation**
- Train classifier on smoothed offline labels
- Features: `observation_features` from config (momentum_5m, momentum_10m, volatility_5m, volatility_10m, volume_zscore)
- Walk-forward CV via `WalkForwardSplitter`
- Model card saved with fold metrics

**Outputs**: `data/models/regime_hmm.pkl`, `data/models/regime_classifier.pkl`, `.meta.json`

### Zone Survival Model (`scripts/label_zone_events.py` → `scripts/train_zone_survival_model.py`)

**Step 1: Label Harvesting**
- Replay bars through structure engine (no experts)
- Collect touch events from engine's event log
- Label each touch: rejection vs. breakout within `zone_label_horizon_bars` (20)

**Step 2: Training**
- Features: 12 features from `zone_survival_features.py` (strength, age, touch_count, zone_width_pct, width_to_volatility_ratio, liquidity_score, hour_utc, day_of_week, 4 regime probability features)
- Walk-forward CV
- Output: `data/models/zone_validity_model.pkl`

### Expert Training

**Trend Expert** (`scripts/train_triple_barrier.py` → `scripts/train_trend_expert.py`):
1. Replay full stack with experts
2. For each bar where trend_expert proposed BUY/SELL, apply triple-barrier labeling
3. Label: 1 if net return > 0, 0 otherwise
4. Train LightGBM on harvested features

**Mean Reversion Expert** (`scripts/train_mean_reversion_expert.py`):
1. Replay bars WITHOUT running experts
2. Use structure engine's rejection events (zone rejections)
3. Apply triple-barrier labeling forward from each rejection bar
4. Direction: rejection off support → long (+1), off resistance → short (-1)
5. Train LightGBM

**Breakout Expert** (`scripts/train_breakout_expert.py`):
1. Same as mean reversion but using breakout events
2. Direction: breakout of support → short (-1), of resistance → long (+1)
3. Train LightGBM

### Opportunity Filter (`scripts/train_opportunity_filter.py`)

**Label**: "best expert proxy" — for each bar, which expert's triple-barrier return was highest? If all ≤ 0, label is 0 (not tradable).

**Features**: `build_features()` extracts 8 features from expert_outputs, regime_state, zone_map, market, fv:
- `edge_vs_fees`, `spread_pct`, `depth_usdt`, `volatility_5m`, `best_confidence`, `expert_disagreement`, `regime_confidence`, `is_panic`

**Model**: LightGBM binary classifier. Blend weight: 0.3 (30% model, 70% hard rules).

### Router (`scripts/train_router.py`)

**Label**: Hindsight optimal expert — for each bar, which expert would have produced the best net return? If none > 0, label is "no_trade".

**Features**: `build_features()` from router.py — expert confidences, regime probs, opportunity score, `bars_since_switch`, `current_expert`.

**Model**: LightGBM multiclass (4 classes). Blend weight: 0.2 (20% model, 80% expert confidence).

### Risk Model (`scripts/train_risk_model.py`)

**Label**: 1 if stop-loss hit first, 0 if take-profit or timeout hit first.

**Features**: Expert output features + risk-relevant features (invalidation_distance, regime_confidence, volatility, spread).

**Model**: LightGBM binary classifier.

## Walk-Forward Cross-Validation

All training scripts use `WalkForwardSplitter` with parameters from `backtest:` config block:

| Parameter | Default | Meaning |
|---|---|---|
| `train_bars` | 5000 | Training window size |
| `test_bars` | 500 | Test window size |
| `step_bars` | 100 | Sliding window step |
| `embargo_bars` | 50 | Purge gap after test set |

The embargo prevents label leakage: a label whose outcome window extends into the test set doesn't contaminate training data in the next fold.

## Model Cards

Every trained model has a `.meta.json` companion:

```json
{
  "trained_at": "2025-01-01T00:00:00+00:00",
  "n_samples": 1234,
  "label_distribution": {"trend_expert": 500, "no_trade": 734},
  "feature_names": ["momentum_5m", ...],
  "walk_forward_folds": [{"fold": 0, "n_train": 5000, "n_test": 500, "accuracy": 0.72, "auc": 0.68}, ...],
  "granularity": "1m"
}
```

## Retraining Flow (`scripts/retrain_all.py`)

One-shot script (invoked by cron/systemd timer, not long-running). For each component:

1. Run its training steps (some have multi-step pipelines, e.g., zone_survival needs `label_zone_events.py` first)
2. Read `.meta.json` for walk-forward metrics
3. Compare against champion's `avg_auc`
4. If challenger clears `promotion_margin_pct` (5%), register as `pending_approval`
5. **Never touches `approved_active`** — promotion is exclusively `approve_model.py`'s job

Components: zone_survival, regime_hmm, regime_classifier, trend_expert, mean_reversion_expert, breakout_expert, opportunity_filter, router, risk_model.
