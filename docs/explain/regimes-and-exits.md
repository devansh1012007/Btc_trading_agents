# Dynamic Regimes, Unified Exit Plans, Sizing Models, and Harness-Ordered Training

This guide explains how the dynamic regime detection, unified `ExitPlan` ladder, ML sizing models, multi-position book, and harness-ordered training DAG work together.

## 1. Dynamic Regime Detection (`src/regime/` & `src/layers/regime_detector.py`)
- **Pluggable Backends**: `gaussian_hmm`, `gmm`, `kmeans`, `ms_ar`, and `garch_mixture` (`src/regime/backends.py`). Every backend runs a pure-NumPy causal forward filter (`filter_causal` / `filter_step`) at serve time.
- **Data-Driven `K` Selection**: `select_k` (`src/regime/selection.py`) chooses `K in [k_min, max_regimes]` on training data using BIC subject to minimum state occupancy and minimum mean dwell bars, falling back to `K=4` (`LEGACY_REGIME_SET`) if constraints are not met.
- **Label Alignment**: `align_regime_labels` (`src/regime/alignment.py`) solves Hungarian bipartite matching across retrains and blocks silent relabeling when `lock_between_retrains` is enabled.
- **Serving Chain**: `dynamic_lgbm -> dynamic_backend -> legacy_4class_lgbm -> legacy_hmm -> rule_based`.

## 2. Unified `ExitPlan` & Six-Step Exit Precedence Ladder (`src/labeling/triple_barrier.py` & `src/exits/exit_model.py`)
Every position carries an `ExitPlan` (`stop_loss_pct`, `take_profit_pct`, `max_holding_bars`, `trailing: TrailSpec`, `hard_max_loss_pct`). On each bar, exits are evaluated in strict order:
1. **Minimum-Hold Lockout (`HoldPolicy`)**: No exit may occur before `min_hold_bars` and `min_trade_seconds` (default 10 s), net of `entry_delay_seconds`.
2. **Tighten-Only Hard Loss Cap (`hard_loss_cap`)**: `min(stop_loss_pct, exits.hard.max_loss_pct)`.
3. **Deterministic Stop-Loss (`stop_loss`)**.
4. **Deterministic Trailing Stop (`trailing_stop`)**: Ratchets with a 1-bar lag on OHLCV bars to prevent same-bar high-then-low look-ahead.
5. **Deterministic Take-Profit (`take_profit`)**.
6. **Per-Expert ML `ExitModel` (`loss_cut`, `profit_exit`)** followed by **Timeout (`timeout`)**.

## 3. Zone-Survival-Informed Barrier & Risk Sizing (`src/sizing/`)
- `BarrierSizingModel` (`src/sizing/barrier_model.py`) predicts quantile `(k_sl, k_tp)` multiples from zone-survival probabilities and K-regime probabilities, floored by the lockout adverse-excursion quantile.
- `RiskSizingModel` (`src/sizing/risk_model.py`) predicts calibrated `P(stop-out)` and maps it to a tighten-only multiplier in `[min_multiplier, 1.0]`.

## 4. Multi-Position Core, Cooldowns, Time-Boxed Halts & `RiskGovernor` (`src/execution/position_book.py` & `src/risk/`)
- `PositionBook` enforces `max_concurrent`, `max_per_expert`, `max_per_regime`, `max_per_direction`, stacking/hedging rules, and gross/net exposure caps, and assigns deterministic GUI `lane_id`s.
- `CooldownManager` and `HaltManager` (driven by injected `Clock`: `BarClock` in backtests, `WallClock` in live) enforce per-expert cool-downs and time-boxed portfolio halts with a fresh-window rule on expiry.
- `RiskGovernor` is strictly tighten-only (ADR-0004 / ADR-0010).

## 5. Harness-Ordered Training DAG (`src/training/plan.py`)
`TrainingPlan` derives its 10-stage order directly from `src/pipeline_stages.py`:
`dataset -> regime -> zone_selection -> zone_survival -> barrier_sizing -> experts -> exit_models -> routers -> opportunity_confirm -> risk_sizing`, using purged-and-embargoed out-of-fold predictions (`generate_oof_predictions`) and verifying bundle hashes via `ModelBundle`.
