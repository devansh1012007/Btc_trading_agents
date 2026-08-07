# PHASE 2 IMPLEMENTATION COMPLETE ✅

## Summary

**PHASE 2: Perception Layer (Structure + Regime)** has been implemented per
the Phase 2 plan. All 10 required tests pass; the training scripts and full
pipeline were run end-to-end against a mock dataset to validate correctness
before delivery (not just reviewed for plausibility).

### What Was Built

```
✅ Shared Infra
   - src/utils/rolling_buffer.py       RollingBarBuffer (§5.1)
   - src/backtest/walk_forward.py      WalkForwardSplitter, purge + embargo (§5.2)
   - src/layers/zone_survival_features.py  Single feature definition shared by
                                             labeling, training, and serving

✅ Dynamic Structure Engine — completed (src/layers/structure_engine.py)
   - Price-action interaction detector: touches now classified as
     rejection/breakout against real bar highs/lows/closes (§6.1)
   - Pivot-confirmation look-ahead fix via confirmed_at_bar tracking (§6.2)
   - Beta-Bernoulli survival probability, replacing the old
     hardcoded-0.6-decays-with-time-only logic (§6.3)
   - Structural Feature Adapter (src/layers/structural_features.py) wires
     ZoneMap back into FeatureVector via dataclasses.replace() (§6.4)
   - Zone-survival LightGBM blending, rule-based by default, ramps in via
     models.zone_survival_blend_weight once trained (§6.5)
   - Zone.regime_relevance populated from real touch events (§6.6)

✅ Regime Detection Layer — new (src/layers/regime_detector.py)
   - 3-state Gaussian HMM, forward-filtered only (never smoothed) at
     serving time, implemented as a self-contained numpy filter with zero
     hard dependency on hmmlearn being installed (§7.2, §7.6)
   - Fast panic-spike override, independent of the HMM (§7.3)
   - Anti-flicker regime_age / regime_persistence_score (§7.4)
   - LightGBM distillation classifier trained on smoothed offline labels,
     served causally (§7.5)
   - Fallback chain: LightGBM -> HMM filter -> volatility-threshold rule,
     always available even with no trained models present

✅ Pipeline & Config (src/pipeline.py, config/settings.yaml)
   - Per-bar order: features -> regime -> structure -> structural feature
     merge -> full SystemState (§3, §8)
   - structure/regime/models/backtest config blocks added

✅ Serialization (src/state/serialization.py)
   - ZoneMap nested zones serialized as a JSON-blob column (§9 option A)
   - Extended feature/regime flattening (structural fields, age,
     persistence_score)

✅ Training Scripts (scripts/)
   - label_zone_events.py: replays bars, derives rejection/breakout
     labels from the structure engine's own touch-event log (§6.5)
   - train_zone_survival_model.py: walk-forward LightGBM + model card
   - train_regime_models.py: BIC diagnostic grid, 3-state HMM fit,
     post-hoc state->label mapping, LightGBM distillation classifier,
     walk-forward CV + model cards (§7.2, §7.5)

✅ Tests (tests/test_phase2.py) — 10/10 passing
   - test_zone_touch_classification, test_zone_breakout_classification
   - test_zone_no_lookahead, test_zone_idempotent_add_bar
   - test_structural_features_populated
   - test_regime_state_probabilities_sum_to_one
   - test_regime_filtering_is_causal
   - test_regime_fallback_without_model_file
   - test_panic_override_smooth
   - test_walk_forward_embargo

✅ Integration Test (test_integration_phase2.py)
   - 150 mock bars -> full SystemState -> Parquet round trip, verified
```

---

## Verified End-to-End (not just written — actually run)

```
$ python -m pytest tests/test_phase2.py -v
====== 10 passed in 0.38s ======

$ python test_integration_phase2.py
Computed full SystemState for 141 bars (9 require history)
Reloaded from Parquet successfully -- row count matches
Data quality checks passed (unique timestamps, no critical NaNs)
Zones present on 131/141 bars
PHASE 2 INTEGRATION TEST: ALL CHECKS PASSED

$ python scripts/label_zone_events.py --input data/raw/BTC_USDT_mock.parquet
Replayed 3000 bars, collected 6782 touch events
Labeled 6777 samples, 5 censored/dropped

$ python scripts/train_zone_survival_model.py
Saved model + model card

$ python scripts/train_regime_models.py --input data/raw/BTC_USDT_mock.parquet
BIC grid winner: n_states=5 (fitting final model with n_states=3 regardless)
State -> label mapping: {2: 'trending_up', 0: 'trending_down', 1: 'chop_range'}
Fold 0: accuracy 0.77
Saved HMM + classifier + model cards

$ python -c "RegimeDetector() loads both trained models, serves causally"
classifier loaded: True / hmm loaded: True / probs sum: 1.0
```

One real bug was caught and fixed by actually running the integration test
rather than trusting the code by inspection: `Zone.created_at` /
`last_touched_at` are sourced from a numpy timestamps array in
`structure_engine.py`, which produced `np.int64` values that
`json.dumps()` cannot serialize. `serialization.py`'s `_zone_to_dict` now
casts every field to a native Python type before encoding.

Note: the training-script output above was produced against a **synthetic
mock dataset**, not real market data — it validates that the scripts run
correctly end-to-end, not that the resulting models are good. No trained
`.pkl` files are shipped in this deliverable; `data/models/` starts empty
so nothing is mistaken for production-ready. Run the three scripts against
your real historical data to produce real models.

---

## Deliberate Design Decisions Beyond the Plan's Literal Text

The plan left a few implementation details open; here's what was chosen
and why, so you can override any of it:

1. **`confirmed_at_bar` as an engine-side shadow dict**, not a new `Zone`
   field — avoids any contracts.py schema change (§4's stated goal),
   at the cost of it only being queryable via `engine.get_confirmed_at_bar()`
   rather than being visible on the serialized `Zone` itself.
2. **HMM served via a self-contained numpy forward filter**, not
   hmmlearn's internal API — `RegimeDetector` has zero hard dependency on
   hmmlearn (only the training script needs it). Trade-off: the manual
   Gaussian-emission math must be kept in sync if you ever change
   `covariance_type` away from `"full"` in the training script.
3. **`zone_survival_features.py` as a new shared module** — not named in
   the plan's file list, but necessary to keep the labeling script,
   training script, and serving-time blend from silently drifting apart
   (the exact class of duplication risk the plan itself flags in
   finding #7).
4. **`regime_serving_source` config respected as a soft preference** — if
   you force `"hmm_filter"` but no HMM is trained yet, `RegimeDetector`
   still falls back automatically rather than raising, on the theory that
   a config mismatch shouldn't take the pipeline down.

## Still Open (per §15 of the plan — needs your sign-off, not mine)

- Phase boundary confirmation (Structure + Regime only, Experts deferred)
- Whether 1-minute bars are acceptable for zone-touch detection given the
  30-40s trade horizon, or worth revisiting before Phase 3
- `zone_label_horizon_bars=20`, `confidence_floor=0.35`,
  `persistence_transition_threshold=0.6` are still the plan's *proposed*
  starting points — untuned against real labeled data
