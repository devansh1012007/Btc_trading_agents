# ADR-0008: Unified `ExitPlan` Contract and Six-Step Exit Precedence Ladder

- **Status:** Accepted

## Context
Position sizing previously sized against `invalidation_distance` while exits used regime-specific volatility barriers (`k_tp * vol`, `k_sl * vol`), and trailing stops and per-expert ML exits did not share a single deterministic evaluation rule.

## Decision
1. Define `TrailSpec` and `ExitPlan` in `src/state/contracts.py`, resolved via `resolve_exit_plan` (`expert -> barrier_model -> regime_specific -> hard_limits`) with a tighten-only `hard_max_loss_pct` cap.
2. Evaluate exits in `evaluate_exit_ladder` / `barrier_exit_step` in strict precedence order:
   - (0) Minimum-hold lockout (`min_hold_bars` and `min_hold_seconds` — never overridden)
   - (1) Tighten-only `hard_loss_cap`
   - (2) Deterministic `stop_loss`
   - (3) Deterministic `trailing_stop` (with 1-bar ratchet lag on coarse OHLCV bars to prevent same-bar high-then-low look-ahead)
   - (4) Deterministic `take_profit`
   - (5) Per-expert ML `ExitModel` (`loss_cut` then `profit_exit`)
   - (6) Deterministic `timeout`.
