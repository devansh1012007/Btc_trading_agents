# ADR-0004: Hard rules always override ML in the filter and risk layers

## Context

The opportunity filter and risk layer each combine hard rules (spread
gate, depth floor, invalidation-distance bounds, regime vetoes, rate
limits) with an ML score blended into the verdict. A confident model can
be confidently wrong — precisely during panic regimes, which is when
losses compound. This system's core design bet is that a safety layer
must be able to say NO regardless of what a model says.

## Decision

In both `OpportunityFilter` and `RiskLayer`, **hard-rule vetoes are
absolute**: if any hard rule fails, the decision is rejected no matter
what the model outputs. ML can only *reduce* size or *confirm* passage;
it can never resurrect a vetoed trade. Defense in depth: the risk layer
re-checks regime rejection independently of the filter ("two code paths
have to both be wrong for a panic-regime trade to get through").

## Alternatives

- **ML as primary with rules as tiebreakers**: smoother decisions,
  but a single model failure mode (drift, bad features, schema mismatch)
  becomes a capital failure mode.
- **No ML in these layers at all**: loses calibrated sizing signals
  (e.g. P(stop-out)) that improve efficiency in normal regimes.

## Trade-offs

- Lower trade frequency than a pure-ML gate would allow.
- Two implementations of similar logic must be kept consistent —
  `scripts/validate_config_consistency.py` checks threshold agreement
  across layers at pre-flight.

## Consequences

- Model unavailability degrades gracefully to rule-based operation by
  construction; there is no mode where the system trades *because* a
  guard was missing.
- Pinned by invariant tests: `tests/unit/test_invariants.py` (kill
  switch first, breakers always block) and
  `test_risk_hard_rules_veto_ml` in `tests/unit/test_phase4.py`.
