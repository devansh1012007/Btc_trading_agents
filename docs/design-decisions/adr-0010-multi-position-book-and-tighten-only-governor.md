# ADR-0010: Multi-Position `PositionBook`, Injected `Clock`, and Tighten-Only `RiskGovernor`

- **Status:** Accepted
- **Amends:** [ADR-0004](adr-0004-risk-layer-circuit-breakers-override-all.md)

## Context
`RiskStateStore` previously called `datetime.now()` and `date.today()` directly (preventing simulated UTC midnight resets and time-boxed halts in backtests), and `process_bar` could hold at most one open position.

## Decision
1. Inject `Clock` (`WallClock` in live, `BarClock` in simulation) into `RiskStateStore`, `HaltManager`, and `LiveBackend`.
2. Manage concurrent positions via `PositionBook` (`src/execution/position_book.py`), per-instance `CooldownManager` (`src/risk/cooldowns.py`), and time-boxed `HaltManager` (`src/risk/halts.py`) with the fresh-window expiry rule.
3. Bound `RiskGovernor` (`src/risk/governor.py`) strictly to tighten-only actions (`0 <= size_multiplier <= 1`, `1 <= max_concurrent_cap <= configured`, `extra_cooldown_multiplier >= 1`), preserving ADR-0004's hard-rule supremacy.
