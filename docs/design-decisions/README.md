# Architecture Decision Records

Decisions that shaped this system, recorded in the Napkin/ADR format:
**Context → Decision → Alternatives → Trade-offs → Consequences**.
Each ADR is immutable history — to change a decision, write a new ADR
that supersedes it.

| ADR | Decision | Status |
|---|---|---|
| [ADR-0001](adr-0001-lightgbm-over-transformer.md) | LightGBM over transformers for tabular decisions | Accepted |
| [ADR-0002](adr-0002-walk-forward-validation.md) | Walk-forward with purge + embargo, never random splits | Accepted |
| [ADR-0003](adr-0003-monolith-on-one-laptop.md) | Single-process monolith on one laptop | Accepted |
| [ADR-0004](adr-0004-hard-rules-veto-ml.md) | Hard rules always override ML in filter and risk layers | Accepted |
| [ADR-0005](adr-0005-rule-based-execution-first.md) | Rule-based execution first; supervised router before bandit/PPO | Accepted |
| [ADR-0006](adr-0006-frozen-dataclass-contracts.md) | Frozen dataclasses as cross-layer state contracts | Accepted |
