# Migration Guide: Dynamic Regimes, `CostModel`, `ExitPlan`, and `ModelBundle`

1. **Regime Probabilities (`RegimeState`)**:
   - Replace direct `.prob_trending_up` / `.prob_trending_down` / `.prob_chop_range` / `.prob_panic_spike` attribute reads outside `src/state/contracts.py` with `regime_state.get_prob("trending_up")` or `regime_prob_features(...)`.
2. **Consolidated `costs:` Config (`CostModel`)**:
   - Legacy `labeling.fee_pct`, `labeling.slippage_buffer_pct`, and `execution.slippage_pct` continue to work via `CostModel.from_config(config)`, or migrate to the top-level `costs:` section (`costs.fees`, `costs.spread`, `costs.slippage`, `costs.latency`, `costs.impact`, `costs.carry`).
   - Do not set both `costs.fees` and `labeling.fee_pct` without `costs._allow_legacy_coexist: true`.
3. **Database Schema (`data/registry.db`)**:
   - `src/registry/db.py` automatically backs up `registry.db` to `registry.db.bak.<user_version>` and migrates to `LATEST_SCHEMA_VERSION` (`PRAGMA user_version = 1`), adding `bundles`, `model_predictions`, and `model_outcomes`.
4. **Default-Off Guarantees**:
   - All new switches (`pipeline.order: legacy`, `execution.exit_model: barrier_level`, `positions.max_concurrent: 1`, `risk.cooldowns.enabled: false`, `risk.halts.enabled: false`, `risk.governor.enabled: false`) default to legacy behaviour so existing configurations produce bit-identical results.
