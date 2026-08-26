# ADR-0002: Walk-forward validation with purge + embargo, never random splits

## Context

Labels here are forward-looking: triple-barrier labels resolve over
`max_holding_bars` of *future* bars. Any evaluation that lets a training
row share information with an evaluation row inside that outcome window
produces inflated metrics that vanish live. Random shuffling and even
sklearn's `TimeSeriesSplit` (no purge gap) both leak by construction.

## Decision

All training and all reported backtests use `WalkForwardSplitter`
(`src/backtest/walk_forward.py`): rolling train/test windows with a
mandatory **embargo gap** between train end and test start, sized to
cover the label outcome window. The full-stack backtest additionally
runs one continuous causal replay segmented into non-overlapping
evaluation windows for per-period stability.

## Alternatives

- **Random k-fold**: leaks label outcomes across folds. Rejected outright.
- **TimeSeriesSplit**: chronological but boundary-adjacent; a label
  entered on the last train bar resolves using test-window bars.
- **Single train/test cut**: no variance estimate across market regimes;
  one lucky regime dominates the result.

## Trade-offs

- Fewer effective training rows per fold than in-sample fitting.
- Embargo discards `embargo_bars` of data at every boundary.
- The full-stack script fixes models across the run and segments windows,
  rather than re-fitting per fold — documented there as a deliberate
  simplification until per-fold retraining lands.

## Consequences

- Reported numbers are reproducible and leakage-defended; the guarantees
  are pinned by `tests/unit/test_no_lookahead.py`.
- Every experiment row in `registry.db::experiments` carries the config
  hash + git commit, so any number can be regenerated.
- See `docs/leakage-prevention.md` for the complete leak-vector map.
