# ADR-0011: Single Stage Registry, Harness-Ordered `TrainingPlan` DAG, and Out-of-Fold Stacking

- **Status:** Accepted

## Context
Training scripts previously ran independently without enforcing upstream-to-downstream ordering or out-of-fold stacking, and the runtime pipeline ran `opportunity_filter` before `router`.

## Decision
1. Define a single stage registry in `src/pipeline_stages.py` shared by `TradingPipeline` (`staged` vs `legacy` order) and `TrainingPlan` (`src/training/plan.py`).
2. Enforce purged-and-embargoed out-of-fold predictions (`generate_oof_predictions`, `verify_oof_no_in_sample_leakage`) whenever a downstream stage (`exit_models`, `routers`, `opportunity_confirm`, `risk_sizing`) trains on upstream model outputs.
3. Verify upstream artefact hashes in `ModelBundle` (`src/registry/bundle.py`) at load and promotion time.
