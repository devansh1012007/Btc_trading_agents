# ADR-0006: Frozen dataclasses as cross-layer state contracts

## Context

Ten layers exchange state every bar. Any field a layer silently adds,
renames, or mutates in place becomes an invisible coupling — the classic
failure is one layer "fixing" a value downstream of another, which then
can't be reproduced or tested. The system also needs exact
serialization (JSONL traces, Parquet archives, SQLite journal) and
per-bar immutability for look-ahead auditing.

## Decision

All shared state lives in `src/state/contracts.py` as
`@dataclass(frozen=True)` classes (`MarketSnapshot` → … → `SystemState`),
with `dataclasses.replace()` as the only mutation mechanism and explicit
serialization helpers in `state/serialization.py`. `FEATURE_SCHEMA_VERSION`
is bumped on any field rename; model loaders verify it and fall back to
rule-based mode on mismatch.

## Alternatives

- **Pydantic models**: runtime validation is attractive, but per-bar
  validation cost buys little when producers are all in-process and
  contracts are pinned by tests; frozen dataclasses are stdlib, faster,
  and already strict via mypy (`src/state/` is a strict-typing module).
- **Dicts / ad-hoc namespaces**: rejected — no schema, no immutability,
  no version guard; the historical source of silent drift bugs.

## Trade-offs

- Adding a field touches contracts + serialization + every constructor
  call site — deliberately loud, so contract changes are visible in
  review rather than accreting silently.
- Frozen containers mean "derived" states allocate copies via replace();
  at ~1 ms/bar total pipeline latency this is unmeasurable.

## Consequences

- The decision-trace/audit chain can hash and serialize any bar's full
  state deterministically.
- Downstream layers are unit-testable with plain constructed contracts —
  no mocks, matching the repo's inline-fixture test style.
