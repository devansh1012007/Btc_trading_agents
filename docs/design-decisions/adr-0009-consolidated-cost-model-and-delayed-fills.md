# ADR-0009: Consolidated `CostModel`, `ExitFillModel`, and Two-Phase Delayed Fills

- **Status:** Accepted

## Context
Fee and slippage settings were previously scattered across `labeling.*` and `execution.*`, exit fills paid zero spread or slippage, and live/paper-live backends had no `close_position` method.

## Decision
1. Consolidate all fee, spread, slippage (`fixed | volatility_scaled | depth_based`), latency (`entry_delay_seconds`, `exit_delay_seconds`), market impact, carry, and rejection parameters in `CostModel` (`src/execution/cost_model.py`).
2. Model entry and exit fill prices (`entry_fill_price_with_delay`, `exit_fill_price`) over `PricePath` (`BarPath` linear interpolation or exact `TickPath` lookup) and support two-phase `PendingExit` when `exit_delay_seconds >= granularity_seconds`.
3. Add reduce-only, idempotent `ExecutionBackend.close_position` across `PaperSimBackend`, `PaperLiveBackend`, and `LiveBackend`.
