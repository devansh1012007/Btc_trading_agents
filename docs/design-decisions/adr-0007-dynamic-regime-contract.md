# ADR-0007: Dynamic Regime Identity (`RegimeSet` and `regime_probs` Dictionary)

- **Status:** Accepted
- **Amends:** [ADR-0001](adr-0001-3state-hmm-plus-panic-override.md)

## Context
Previously, `RegimeState` stored four hard-coded scalar probability fields (`prob_trending_up`, `prob_trending_down`, `prob_chop_range`, `prob_panic_spike`), coupling 13 downstream modules to a fixed `K=4` `RegimeType` enum. Supporting data-driven regime discovery (`K in [2, max_regimes]`) across multiple backends (`gaussian_hmm`, `gmm`, `kmeans`, `ms_ar`, `garch_mixture`) requires a schema-agnostic regime contract.

## Decision
1. Introduce `DynamicRegime` and `RegimeSet` in `src/state/contracts.py`, with `LEGACY_REGIME_SET` (`legacy4`) as the default.
2. Store probabilities on `RegimeState` in `regime_probs: Dict[str, float]` accessed via `RegimeState.get_prob(regime_id)` and `regime_prob_features()`.
3. Enforce causal forward filtering (`filter_causal` / `filter_step`) in pure NumPy at serve time and Hungarian label alignment (`align_regime_labels`) across retrains.
