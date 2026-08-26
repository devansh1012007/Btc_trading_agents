# ADR-0005: Rule-based execution first; supervised router before bandit/PPO

## Context

Two progressions had to be chosen: (1) how orders get placed, and (2) how
the system learns *which expert to trust* over time. Both have an
ambitious end state (smart order handling; contextual-bandit or RL
routing) and a boring correct first step. The project's credibility rule
is "prove complexity buys something" — complexity is adopted only when a
measured ablation shows it pays.

## Decision

- **Execution**: deterministic, rule-based fill simulation (spread
  crossing + depth-participation cap + configurable slippage) in all
  backtests and paper modes. No learned execution model.
- **Router**: start with supervised LightGBM ranking trained on hindsight
  labels ("which expert would have won this bar?"). Contextual bandits
  are the next step; PPO-style RL only after bandit baselines are beaten.

## Alternatives

- **Learned execution / RL order placement**: needs live microstructure
  data at scale we do not have; any result would be simulation-of-
  simulation. Deferred until real fills accumulate via paper-live mode.
- **Jump straight to PPO routing**: non-stationary reward, huge variance,
  unexplainable choices — the opposite of what the decision-trace system
  promises. Rejected for now.

## Trade-offs

- Rule-based execution may misprice impact for sizes far above tested
  ones; acceptable because position caps bound participation anyway.
- Supervised routing inherits label bias from hindsight selection;
  acknowledged and planned to be replaced by online feedback (bandit)
  rather than tuned around.

## Consequences

- Every routing upgrade slot-in behind `Router.decide()`'s contract —
  the naive-confidence baseline already proves interface compatibility.
- The ablation harness (`--single-expert`, `--no-risk-layer`,
  `--no-regime-gating`, naive-router baseline) exists precisely to gate
  each future complexity upgrade on measured value.
