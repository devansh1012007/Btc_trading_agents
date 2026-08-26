# ADR-0003: Single-process monolith on one laptop

## Context

The stated deployment target is a single laptop with integrated graphics
and limited RAM, running BTC scalping on 30s–5m bars: one bar per
granularity interval, a handful of LightGBM inferences per bar (measured:
~1 ms end-to-end decision latency, ~960 bars/sec throughput on the
benchmark harness). The system is research software whose credibility
depends on honest results, not on infrastructure.

## Decision

One Python process contains every layer: ingestion → features → regime →
structure → experts → filter → router → risk → execution → feedback.
SQLite for persistence (registry, breakers, journal); JSONL/Parquet for
logs and archives. Docker exists for reproducible packaging of that one
process, not for orchestration.

## Alternatives

- **Microservices** (risk service, execution service, …): adds network
  hops exactly where latency and failure modes matter most; each hop is
  a new partial-failure state to reconcile. Rejected as decoration.
- **Message-bus event architecture** (Kafka/Redis): justified at high
  event rates or multi-strategy fan-out; here it would be a second
  system to operate with no payoff.

## Trade-offs

- A crash takes down everything — mitigated by persistent circuit
  breakers (`RiskStateStore`, SQLite), the kill-switch file
  (`data/control/EMERGENCY_STOP`), order reconciliation on restart, and
  idempotent client order IDs.
- No horizontal scale. Accepted: one symbol, one position, one exchange.
  Multi-symbol partitioning is a roadmap item that would revisit this.

## Consequences

- Whole-system tests are ordinary pytest runs; no service mocks needed.
- `max_concurrent_positions: 1` is enforced in code and validated at
  startup — the monolith's assumptions are checked, not implicit.
