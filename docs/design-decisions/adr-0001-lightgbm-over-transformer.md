# ADR-0001: LightGBM over transformer for tabular stage decisions

## Context

Every learned component in this system — regime distillation, zone
validity, the three experts, the opportunity filter, the router, risk
scoring — makes a decision from a small (~10–40) set of engineered,
heterogeneous features at a single point in time. Training runs on one
laptop with limited RAM; models must retrain weekly on a rolling window;
every prediction needs to be explainable bar-by-bar (decision traces).

## Decision

Use gradient-boosted trees (LightGBM) for every supervised component.
No neural networks in the decision path.

## Alternatives

- **Transformer encoder over recent sequences**: strongest at learning
  interactions across long raw sequences. Rejected: data-hungry (crypto
  regimes shift before we accumulate enough clean labeled data), slow to
  retrain on laptop hardware, opaque per-decision attribution.
- **Logistic regression / linear models**: fully explainable and fast,
  but cannot express the interaction structure (regime × expert
  confidence × liquidity) that the router needs.

## Trade-offs

- Give up automatic representation learning: feature engineering stays a
  manual, ongoing obligation (documented as a known cost in
  `reasoning_for_decisions.md` — "dynamic features or no features").
- Trees extrapolate poorly out of distribution. Mitigated by PSI drift
  monitoring (`monitoring/drift_monitor.py`) and walk-forward retraining
  rather than by the model class itself.

## Consequences

- Full retrain of all components is minutes, not hours (`retrain_all.py`).
- Per-decision explanations are possible via tree internals or simple
  feature attribution, which feeds the decision-trace system.
- A sequence model remains a *future* option behind the same
  `ExpertOutput`/`RouterDecision` contracts — swapping it in would not
  change the pipeline.
