# PHASE 3 IMPLEMENTATION COMPLETE ✅

## Summary

**PHASE 3: Decision Layer (Experts + Opportunity Filter + Router)** is
implemented per the plan. 26/26 tests pass (10 Phase 2 + 16 Phase 3), the
full pipeline was run end-to-end against mock data, and every training
script was executed against real (if synthetic) data — not just reviewed
for plausibility. Two real defects were found and fixed by actually
running the code rather than reading it; both are documented below in
full, not glossed over.

### What Was Built

```
✅ Triple-Barrier Labeling (src/labeling/triple_barrier.py)
   - label_triple_barrier(): TP/SL/timeout resolution, conservative
     same-bar-touches-both convention, strictly bounded to
     forward_bars[:max_holding_bars] (no look-ahead by construction)
   - compute_barrier_pcts() + barrier_volatility_pct() (§5)

✅ Expert Layer (src/experts/)
   - base_expert.py: BaseExpert ABC -- rule proposes, model scores;
     fallback mode; granularity-mismatch guard (§4.7); EV computed from
     the model card's empirical avg_favorable/avg_unfavorable/
     avg_adverse_excursion stats, not a fixed guess
   - trend_expert.py, mean_reversion_expert.py, breakout_expert.py (§6.2-6.4)
   - expert_registry.py: centralized construction (§12)
   - layers/expert_layer.py: runs all three, assembles ExpertOutputs

✅ Opportunity Filter (src/layers/opportunity_filter.py, §7)
   - Hard rules with regime-specific overrides; hard rules always retain
     veto power over the ML blend (a model can't override a failed rule,
     only add confidence within a pass)
   - liquidity_score / volatility_score / expert_consensus_score /
     regime_acceptability breakdown fields (added to OpportunityScore,
     additive/backward-compatible)

✅ Router (src/layers/router.py, §8)
   - Confidence-based fallback; hindsight-trained multiclass blend;
     bars_since_switch / current_expert tracked as a soft feature only --
     no hard switching limits (deferred to Phase 4 Risk Layer per plan)

✅ Shared Infra (NEW)
   - src/backtest/replay.py: replay_full_stack() + harvest_zone_touch_candidates()
     -- the single code path every training/backtest script replays
     through, so seven scripts can't silently drift into seven different
     pipelines
   - src/utils/config_helpers.py: parse_regime_list()

✅ Granularity Guard (§4.7)
   - Every trainable model (regime HMM, regime classifier, zone-survival,
     all 3 experts, opportunity filter, router) now carries a
     `granularity` field in its pickle payload and is checked at load
     time against the live `system.data_granularity`; a mismatch is
     treated as if the file were absent, not silently used

✅ Pipeline & Config (src/pipeline.py, config/settings.yaml)
   - Full order: features -> regime -> structure -> experts -> opportunity
     filter -> router -> SystemState
   - Comprehensive settings.yaml: system/labeling/experts/opportunity_filter/
     router/models/backtest blocks, all thresholds config-driven

✅ Training & Backtest Scripts (scripts/)
   - train_triple_barrier.py, train_trend_expert.py,
     train_mean_reversion_expert.py, train_breakout_expert.py,
     train_opportunity_filter.py, train_router.py,
     backtest_phase3.py, backtest_compare_expert_choices.py

✅ Tests -- 26/26 passing
   - tests/test_phase2.py (10, carried forward)
   - tests/test_phase3.py (16): triple-barrier TP/SL/timeout/no-lookahead,
     expert gate determinism/fallback/granularity-guard/model-skip-on-None,
     opportunity filter hard-rule veto + regime gate + proxy-label feature
     shape, router no-trade fallback + switching context + granularity
     guard, YAML config loading, timeframe adaptation
   - test_integration_phase3.py: full SystemState through RouterDecision,
     Parquet round trip, Decision Layer columns present on every bar
```

---

## Two Real Findings From Actually Running This

Both were caught by running the code against data, not by reading it —
worth stating plainly rather than folding quietly into a diff.

### 1. `volatility_30s` is mathematically always 0.0 (found and fixed)

`FeatureEngine._volatility(prices, periods=1)` computes `np.std()` over a
single-element array of returns. The standard deviation of one number is
zero by definition — this isn't a tuning issue, it's a statistic that
cannot be anything but zero for `periods=1`, in any market, on any data.

Every training script scaled its take-profit/stop-loss barriers by
`k_tp/k_sl * volatility_30s`. With that always zero, every barrier
collapsed to zero width, so every simulated trade — including "take
profit" hits — resolved to almost exactly `-2 * fee_pct`: a guaranteed
loss regardless of direction or which expert proposed it. Every candidate
in the first harvest had `realized_return_pct ≈ -0.08` with zero variance.

**Fix:** added `barrier_volatility_pct()` to `triple_barrier.py`, using
`volatility_5m` (a genuinely multi-sample, well-defined dispersion
statistic) instead. Applied at all six call sites (every training script,
the backtest harness, and the shared replay helper) so there's one place
this decision lives, not six. After the fix, harvested outcomes show real
variance (std ≈ 0.02-0.03%, both wins and losses).

**Left alone on purpose:** `FeatureEngine.volatility_30s` itself is
unchanged — fixing it is a Phase 1/2 file outside this plan's stated
scope, and I'd rather flag a latent defect precisely than reach into
already-tested code under time pressure without full visibility into
what else might depend on its exact (degenerate) current behavior.
`momentum_30s` is unaffected — momentum over 1 period is a valid % change,
not a dispersion statistic, so it doesn't share this defect.

### 2. OHLCV-only replay makes the Opportunity Filter always reject (found, documented, not silently patched)

`MarketDataLayer.create_from_bar` (Phase 1, unchanged) synthesizes
`bid_volume = ask_volume = 0.0` and a synthetic ~0.2% spread whenever no
live ticker is supplied — which is every call from every training and
backtest script, since none of them fetch a ticker. Against
`opportunity_filter.hard_rules`' defaults (`min_depth_usdt: 10000`,
`max_spread_pct: 0.05`), this means **every bar fails** `insufficient_depth`
and `spread_too_wide`, unconditionally, on any OHLCV-only replay —
independent of how good the underlying signal is. The Phase 3 integration
test's `0 bars judged tradable` result is this artifact, not a signal
problem, and I want that clearly on record rather than looking like a
passing-but-empty test.

**Not fixed, deliberately:** loosening `min_depth_usdt`/`max_spread_pct`
in the shipped config to make trades "appear" would be tuning a risk gate
to produce activity rather than reflect real liquidity conditions — that's
the confusing-activity-with-effectiveness trap, and it's your call to
make with real data, not mine to quietly resolve. Two legitimate paths
forward: (a) source real bid/ask/depth history alongside OHLCV for
backtesting, or (b) use a separate, explicitly-labeled backtest config
profile with hard_rules calibrated for OHLCV-only replay. I'd lean toward
(a) since (b) risks the config silently diverging from what live trading
actually enforces.

---

## Verified End-to-End

```
$ python -m pytest tests/ -q
26 passed

$ python test_integration_phase3.py
Computed full SystemState (through RouterDecision) for 141/150 bars
Every bar has expert_outputs, opportunity_score, and router_decision populated
expert_outputs_json round-trips with all 3 experts on every bar
PHASE 3 INTEGRATION TEST: ALL CHECKS PASSED
(0 bars tradable -- finding #2 above, not a bug)

$ python scripts/train_triple_barrier.py --input <4000-bar mock data>
Harvested 1018 candidates across all 3 experts, real return variance
confirmed after finding #1's fix

$ python scripts/train_trend_expert.py
Fold accuracy ~0.98 fitting its own harvested labels (expected on
synthetic, low-diversity mock data -- not evidence of real edge)

$ python scripts/train_mean_reversion_expert.py --input <mock data>
8391 rejection-event candidates from Phase 2's touch-event stream,
36 walk-forward folds, trained successfully

$ python scripts/train_breakout_expert.py --input <mock data>
0 breakout events in this particular mock run -- correctly exits with a
clear error rather than training on nothing (mock-data limitation: pure
random walk rarely produces genuine compression->expansion breakouts)
```

Opportunity filter, router, and full backtest training were verified
functionally correct via a temporarily-relaxed diagnostic config (confirmed
385/391 bars tradable, router actually selecting `trend_expert` and
`mean_reversion_expert` in a sensible ratio once finding #2's gate is
accounted for) — then reverted; the shipped `settings.yaml` keeps the
plan's original thresholds since loosening them is your decision, per
finding #2. Full training-script runs against your real historical data
are yours to run locally, per your last message.

---

## Deliberate Design Decisions Beyond the Plan's Literal Text

1. **`OpportunityScore` gets 4 new fields** (`liquidity_score`,
   `volatility_score`, `expert_consensus_score`, `regime_acceptability`),
   all with defaults — the only `contracts.py` schema change in Phase 3,
   additive and backward-compatible. `ExpertOutputs` also gained
   `all()`/`get_by_name()`/`get_best_by_confidence()` helper methods
   (no field changes).
2. **`models.zone_model`, not `zone_survival_model`** — the plan's §9
   sketch renamed Phase 2's key; kept the original since Phase 2's code
   already reads `zone_model` everywhere, and renaming would silently
   break working config loading for no benefit.
3. **Breakout Expert's volume gate substitutes `volume_zscore`** (Phase
   2's, already computed) **for the plan's `fv.volume`/`avg_volume_20`**,
   which don't exist as fields — see `breakout_expert.py`'s docstring.
   Compression is judged from the expert's own rolling
   `volatility_5m` buffer, computed statefully, not assumed to exist
   upstream.
4. **`src/backtest/replay.py` is new, not named in the plan's file list**
   — without it, six training scripts plus the backtest harness would
   each hand-roll their own replay loop, which is exactly the
   drift-prone duplication the plan itself warns against elsewhere.
5. **Feature/period constants in `feature_engine.py` were left alone** —
   the plan's §9 sketch renames `features:` to `feature_engine:` with
   nested momentum/volatility periods, but `feature_engine.py`'s
   internals don't currently read period values from config at all
   (only `lookback_bars` is config-driven, inherited from Phase 1/2).
   Wiring that up is a real refactor of tested, working code outside
   Phase 3's stated file scope (§12 doesn't list `feature_engine.py` as
   touched) — flagging this explicitly rather than silently skipping it,
   since finding #1 above shows exactly the kind of thing hiding in that
   file's untouched corners.

## Still Open (per §15 — needs your sign-off, not mine)

- Fee tier / `slippage_buffer_pct` — confirm your actual Binance tier
- `max_holding_bars: 5` — target horizon, untuned
- `k_tp: 1.5` / `k_sl: 1.0` and the per-regime overrides — untuned against
  real volatility, and now meaningfully computable given finding #1's fix
- `regime_confidence_floor`, `no_consensus_ceiling`, `confidence_floor_to_trade`
  — reasonable starting points, not backed by real data yet
- Opportunity Filter / Router `blend_weight` (0.3 / 0.2) — ramp schedule
  as you gain confidence in each model
- **Finding #2 above** — how you want to handle OHLCV-only backtesting
  before any of the trained-model numbers mean anything
