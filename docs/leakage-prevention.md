# Leakage Prevention

Trading ML fails quietly: a single feature that peeks one bar into the
future can turn a losing strategy into a beautiful equity curve. This
document is the checklist of every place leakage could enter this system,
and the mechanism that blocks it. The corresponding regression tests live
in `tests/unit/test_no_lookahead.py`; the splitter implementation is
`src/backtest/walk_forward.py`.

## 1. Data flow and split boundaries

```text
Historical bars (chronological, never shuffled)
      ↓
Rolling walk-forward splits  (WalkForwardSplitter)
   train [i, i+T)  →  EMBARGO GAP  →  test [i+T+E, i+T+E+S)
      ↓
Train on train fold → evaluate on test fold → roll forward
```

- Splits are **strictly causal**: every test index is greater than every
  train index in the same fold (`train_idx.max() < test_idx.min()`).
- An **embargo gap** of `embargo_bars` sits between train end and test
  start, so labels whose outcome window extends past the boundary cannot
  leak their resolution into training targets.
- Never use random or `TimeSeriesSplit` splits here: `TimeSeriesSplit`
  has no purge gap, and labels with a forward-looking outcome window
  (`max_holding_bars`) straddle its boundary by construction.

**Test**: `test_train_and_test_never_overlap_and_embargo_is_respected`.

## 2. Feature timestamp rules

- A feature vector for bar *t* may only consume data with
  `timestamp ≤ t`. All rolling statistics (momentum, volatility, RSI,
  MACD, volume z-score) are computed inside `FeatureEngine`'s
  append-only rolling buffer, which is updated by `add_bar(snapshot)`
  **before** `compute_features(snapshot)` runs for that bar — the bar's
  own close is included, anything after it is not.
- Structural features (distance/strength for nearest zones) derive from
  the zone map built at bar *t* from pivots confirmed at or before *t*.
- Liquidity features in historical replay come from the synthetic model,
  which reads the same bar's OHLCV only — no future book exists.
- `FeatureVector.timestamp == snapshot.timestamp` always.

**Tests**: `test_features_are_pure_function_of_past_bars`,
`test_perturbing_future_bars_does_not_change_past_features`,
`test_feature_timestamp_matches_its_bar`.

## 3. Label generation

- Labels come from triple-barrier labeling
  (`src/labeling/triple_barrier.py`). The labeler receives an explicit
  `forward_bars` list and only ever reads
  `forward_bars[:max_holding_bars]` — it cannot see beyond the window
  the caller passes, and the caller (training harvesters) slices windows
  starting at the entry bar.
- Training harvesters draw entry features from SystemState at the event
  bar and outcomes strictly from subsequent bars.

**Tests**: `test_label_outcome_unchanged_by_extra_future_bars`,
`test_label_changes_only_when_window_truncated_before_event`.

## 4. Normalization / preprocessing fitting

- LightGBM models are tree-based: no global scaling is fitted anywhere,
  so there is no scaler whose parameters could leak test-fold statistics
  into training.
- Rolling z-scores (e.g. `volume_zscore`) normalize against the trailing
  window only — past data relative to each bar.
- Model artifacts carry `granularity` and `feature_version`
  (`FEATURE_SCHEMA_VERSION`); loaders fall back to rule-based mode on any
  mismatch rather than silently scoring misaligned features.

## 5. Transaction-cost and slippage assumptions

Leakage isn't only temporal — optimistic cost assumptions fake edge too.

| Cost | Where | Assumption |
|---|---|---|
| Trading fees | `labeling.fee_pct` (default 0.04% Binance taker), applied per side | charged on entry AND exit |
| Spread | synthetic liquidity model (`liquidity.synthetic.*`) or real order book | fills cross the spread |
| Slippage | `ExecutionRecord.slippage_pct` from the execution engine | size-dependent via depth proxy |
| Partial fills | `partial_fill_policy` in execution backends | large orders may fill partially |

All reported metrics are **realized net**: they are computed from the
closed-trade sequence after fees, spread crossing, and slippage — never
from mid-price-to-mid-price label returns. The backtest report states
this explicitly ("realized, net of fees/slippage").

## 6. Guardrails summary

| Leak vector | Blocked by |
|---|---|
| Shuffled rows | chronological ingestion; sort-by-timestamp enforced in loaders |
| Label straddling split | purge + embargo in `WalkForwardSplitter` |
| Future-looking features | rolling buffers + no-lookahead test suite |
| Scaler fitted on all data | none used (tree models); rolling z-scores only |
| Optimistic costs | fee/spread/slippage modeled in execution sim; metrics realized-net |
| Stale model/schema mismatch | granularity + feature_version guards → fallback mode |

If you add a feature, label type, or evaluation path, extend both this
document and `tests/unit/test_no_lookahead.py`.
