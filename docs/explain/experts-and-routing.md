# Experts & Routing

## Expert Architecture

Each expert is a specialized strategy that handles one market behavior. All experts implement `BaseExpert` ABC with two methods:

1. **`propose_candidate()`** — deterministic rule gate. Returns `None` if conditions aren't met, or an action (+1/-1) if they are.
2. **`score()`** — model inference or fallback. Only called when `propose_candidate()` returns non-None.

The rule gate is **deterministic** (no randomness). If `propose_candidate()` returns `None`, the model is **never called** — verified by `test_expert_no_candidate_skips_model` which uses an exploding mock model.

## BaseExpert

### Model Loading

Pickled model payload contains: `{"model": ..., "feature_names": [...], "granularity": "1m", "feature_version": "v2_renamed", ...}`.

At load time, checks both `granularity` and `feature_version` against running config. Mismatch → `use_fallback_mode = True`, model set to `None`.

### EV Computation

Expected value is computed from the model card's empirical stats, not a fixed guess:

```
EV = avg_favorable_return × confidence + avg_unfavorable_return × (1 - confidence)
```

### Invalidation Distance

`compute_invalidation_distance()` — distance to nearest zone that would invalidate the trade. Used by the Risk Layer for position sizing.

## TrendExpert (`src/experts/trend_expert.py`)

### Gate Conditions

All must pass:
- Regime ∈ {TRENDING_UP, TRENDING_DOWN}
- Regime confidence ≥ `regime_confidence_floor` (0.4)
- Regime persistence ≥ `persistence_requirement` (0.5)
- `momentum_medium` ≥ `momentum_threshold` (0.05)

### Features (7)

`momentum_medium`, `momentum_long`, `sma_slope`, `invalidation_distance`, `regime_confidence`, `regime_persistence_score`, `volatility_medium`

### Fallback

`fallback_confidence: 0.6`, `fallback_expected_return: 0.08`, `fallback_expected_risk: 0.12`

## MeanReversionExpert (`src/experts/mean_reversion_expert.py`)

### Gate Conditions

All must pass:
- Regime == CHOP_RANGE
- Regime confidence ≥ `regime_confidence_floor` (0.35)
- Price within `zone_proximity_pct` (0.3) of a strong zone (`zone_strength_floor: 0.3`)
- RSI outside `rsi_extremity_threshold` (20, 80)

### Features (6)

`rsi_14`, `zone_strength`, `zone_age`, `zone_proximity_pct`, `regime_confidence`, `volatility_medium`

### Training Data

Harvested from `harvest_zone_touch_candidates()` — replay bars WITHOUT running experts, then apply triple-barrier labeling forward from each rejection touch event's bar. Direction: rejection off support → long (+1), rejection off resistance → short (-1).

### Fallback

`fallback_confidence: 0.55`, `fallback_expected_return: 0.06`, `fallback_expected_risk: 0.10`

## BreakoutExpert (`src/experts/breakout_expert.py`)

### Gate Conditions

All must pass:
- Regime ∈ {TRENDING_UP, TRENDING_DOWN}
- Regime confidence ≥ `regime_confidence_floor` (0.35)
- Volatility compression: rolling `volatility_medium` buffer shows compression (implemented statefully, not from config)
- Volume surge: `volume_zscore` ≥ `volume_zscore_surge_threshold` (1.0)
- Orderflow agreement: `orderflow_momentum` ≥ `orderflow_momentum_threshold` (10)

### Features (5)

`volatility_medium`, `volatility_long`, `volume_zscore`, `orderflow_momentum`, `regime_confidence`

### Training Data

Harvested from `harvest_zone_touch_candidates()` — breakout events. Direction: breakout of support → short (-1), breakout of resistance → long (+1).

**Note**: The plan's raw volume ratio gate (`fv.volume / avg_volume_20`) was substituted for `volume_zscore` because `fv.volume` and `avg_volume_20` don't exist as fields.

### Fallback

`fallback_confidence: 0.50`, `fallback_expected_return: 0.10`, `fallback_expected_risk: 0.15`

## ExpertRegistry (`src/experts/expert_registry.py`)

Centralized construction from config. All experts receive `data_granularity` at construction time. Returns `Dict[str, BaseExpert]`.

## Router (`src/layers/router.py`)

### Decision Flow

```
1. If opportunity filter rejected → "no_trade" (reasoning="opportunity_filter_rejected")
2. If ML model available: blend model probs with normalized expert confidence
3. If no model, model inference error, or blended winner is no_trade/below floor:
   highest-confidence expert above confidence_floor_to_trade
4. Track bars_since_switch for switching context (no_trade resets nothing —
   it clears the current expert so re-entering the same expert counts as a switch)
```

### Model Training (`scripts/train_router.py`)

Hindsight-trained multiclass model. For each bar:
1. Replay all three experts (rule-only replay, no ML models needed)
2. Apply triple-barrier labeling to each candidate
3. Ask: "which expert's choice would have been best?" (max net-of-fees realized return, or no_trade if nothing beats 0)
4. **Switching context** is derived by walking the resulting HINDSIGHT label sequence itself — i.e., what switching would have looked like had the router always picked the optimal expert. This is NOT from the rule-based fallback router's own (different) live decisions.

Features include `bars_since_switch` and `current_expert` derived from the hindsight label sequence.

### Model Payload

```python
{"model": lgbm_model, "feature_names": FEATURE_NAMES, "granularity": "1m", "feature_version": "v2_renamed"}
```

### Fallback Behavior

`confidence_floor_to_trade: 0.3` — if no expert's confidence exceeds this, selects `no_trade`.

### Switching Context

`bars_since_switch` is tracked as a **soft feature** in the router's ML model. The hard switching limit (`min_bars_between_switches: 3`) is in the Risk Layer, independently tracked. This duplication is deliberate — the Router's tracking is for feature engineering, the Risk Layer's is for enforcement.
