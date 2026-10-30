"""
Per-expert ML Exit Model (`ExitModel`, Phase 10 T10.1-T10.4):
* Two heads:
  - `profit_head`: trained on open-position bar states in profit (`bars_held >= min_hold_bars`),
    predicting `P(exit_now_beats_holding)` -> triggers `"profit_exit"`.
  - `loss_cut_head`: trained on open-position bar states in loss (`bars_held >= min_hold_bars`),
    predicting `P(hits_stop_loss_before_recovery)` -> triggers `"loss_cut"`.
* Group-by-trade cross-validation splitter (`split_open_position_bars_by_trade`, Leakage Control L6)
  so bars from the same open trade are never split across train and validation.
* Full exit precedence ladder (`evaluate_exit_ladder`) shared between labeler and runtime:
  0. Minimum-hold lockout (`bars_held < min_hold_bars` or `held_seconds < min_hold_seconds`)
  1. Tighten-only hard loss cap (`hard_loss_cap`, C2)
  2. Deterministic `stop_loss` from `ExitPlan`
  3. Deterministic `trailing_stop` from `ExitPlan` (1-bar ratchet lag)
  4. Deterministic `take_profit` from `ExitPlan`
  5. Per-expert ML `ExitModel` (`loss_cut` then `profit_exit`)
  6. Deterministic `timeout` (`bars_held >= max_holding_bars`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lightgbm import LGBMClassifier

from ..labeling.triple_barrier import (
    TrailState,
    advance_trail,
    barrier_levels,
    raw_return_pct,
)
from ..state.contracts import ExitPlan, FeatureVector, MarketSnapshot


class TradeGroupLeakageError(ValueError):
    """Raised when bars from the same `trade_id` appear in both train and validation folds (L6)."""


EXIT_STATE_FEATURES = [
    "bars_held",
    "remaining_bars",
    "unrealized_return_pct",
    "mfe_so_far_pct",
    "mae_so_far_pct",
    "giveback_pct",
    "distance_to_tp_pct",
    "distance_to_sl_pct",
    "volatility_medium",
    "momentum_short",
]


def build_open_position_state_features(
    *,
    direction: int,
    entry_price: float,
    bar: MarketSnapshot,
    bars_held: int,
    max_holding_bars: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    mfe_so_far_pct: float,
    mae_so_far_pct: float,
    fv: Optional[FeatureVector] = None,
) -> Dict[str, float]:
    unreal = raw_return_pct(direction, entry_price, bar.close)
    giveback = max(0.0, float(mfe_so_far_pct) - max(0.0, unreal))
    return {
        "bars_held": float(bars_held),
        "remaining_bars": float(max(0, max_holding_bars - bars_held)),
        "unrealized_return_pct": float(unreal),
        "mfe_so_far_pct": float(mfe_so_far_pct),
        "mae_so_far_pct": float(mae_so_far_pct),
        "giveback_pct": float(giveback),
        "distance_to_tp_pct": float(take_profit_pct - unreal),
        "distance_to_sl_pct": float(unreal + stop_loss_pct),
        "volatility_medium": float(fv.volatility_medium) if fv is not None else 0.2,
        "momentum_short": float(fv.momentum_short) if fv is not None else 0.0,
    }


def split_open_position_bars_by_trade(
    bar_rows: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.25,
) -> Tuple[List[int], List[int]]:
    """Split open-position bar rows into `(train_indices, val_indices)` grouped strictly by `trade_id` (L6)."""
    trade_order: List[str] = []
    seen = set()
    for r in bar_rows:
        tid = str(r["trade_id"])
        if tid not in seen:
            seen.add(tid)
            trade_order.append(tid)

    n_trades = len(trade_order)
    if n_trades < 2:
        raise ValueError("Need at least 2 distinct trades to split by trade_id")

    n_val = max(1, int(round(n_trades * float(val_fraction))))
    train_tids = set(trade_order[:-n_val])
    val_tids = set(trade_order[-n_val:])
    assert not (train_tids & val_tids), "Overlap in trade_ids"

    train_idx = [i for i, r in enumerate(bar_rows) if str(r["trade_id"]) in train_tids]
    val_idx = [i for i, r in enumerate(bar_rows) if str(r["trade_id"]) in val_tids]
    return train_idx, val_idx


def verify_no_trade_group_leakage(
    bar_rows: Sequence[Mapping[str, Any]],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> None:
    """Raise `TradeGroupLeakageError` if any `trade_id` is shared between `train_idx` and `val_idx`."""
    t_set = {str(bar_rows[i]["trade_id"]) for i in train_idx}
    v_set = {str(bar_rows[i]["trade_id"]) for i in val_idx}
    overlap = t_set & v_set
    if overlap:
        raise TradeGroupLeakageError(f"Trade IDs leaked across train/val folds (L6): {sorted(overlap)}")


class ExitModel:
    """Per-expert two-head (`profit_head` + `loss_cut_head`) exit model."""

    def __init__(
        self,
        expert_id: str,
        *,
        profit_threshold: float = 0.65,
        loss_cut_threshold: float = 0.70,
        min_profit_pct: float = 0.05,
        min_loss_pct: float = 0.05,
        feature_names: Sequence[str] = EXIT_STATE_FEATURES,
    ) -> None:
        self.expert_id = str(expert_id)
        self.profit_threshold = float(profit_threshold)
        self.loss_cut_threshold = float(loss_cut_threshold)
        self.min_profit_pct = float(min_profit_pct)
        self.min_loss_pct = float(min_loss_pct)
        self.feature_names = list(feature_names)
        self.profit_head: Optional[Any] = None
        self.loss_cut_head: Optional[Any] = None

    def fit(
        self,
        bar_rows: Sequence[Mapping[str, Any]],
        *,
        random_state: int = 42,
    ) -> "ExitModel":
        profit_rows = [r for r in bar_rows if float(r.get("unrealized_return_pct", 0.0)) > 0]
        loss_rows = [r for r in bar_rows if float(r.get("unrealized_return_pct", 0.0)) <= 0]

        if len(profit_rows) >= 4 and len({int(r["y_profit_exit"]) for r in profit_rows}) >= 2:
            Xp = np.array([[float(r.get(f, 0.0)) for f in self.feature_names] for r in profit_rows], dtype=float)
            yp = np.array([int(r["y_profit_exit"]) for r in profit_rows], dtype=int)
            self.profit_head = LGBMClassifier(
                n_estimators=20, max_depth=3, random_state=random_state, verbose=-1
            ).fit(Xp, yp)

        if len(loss_rows) >= 4 and len({int(r["y_loss_cut"]) for r in loss_rows}) >= 2:
            Xl = np.array([[float(r.get(f, 0.0)) for f in self.feature_names] for r in loss_rows], dtype=float)
            yl = np.array([int(r["y_loss_cut"]) for r in loss_rows], dtype=int)
            self.loss_cut_head = LGBMClassifier(
                n_estimators=20, max_depth=3, random_state=random_state, verbose=-1
            ).fit(Xl, yl)

        return self

    def evaluate_bar(
        self,
        state_features: Mapping[str, float],
        close_price: float,
    ) -> Optional[Tuple[str, float]]:
        """Return `("loss_cut" | "profit_exit", close_price)` or `None`."""
        unreal = float(state_features.get("unrealized_return_pct", 0.0))
        row = np.array([[float(state_features.get(f, 0.0)) for f in self.feature_names]], dtype=float)

        # Loss-cut head checked first when in drawdown beyond min_loss_pct
        if unreal <= -self.min_loss_pct and self.loss_cut_head is not None:
            proba = self.loss_cut_head.predict_proba(row)[0]
            p_cut = float(proba[1]) if len(proba) > 1 else float(proba[0])
            if p_cut >= self.loss_cut_threshold:
                return ("loss_cut", float(close_price))

        # Profit-exit head checked when in profit beyond min_profit_pct
        if unreal >= self.min_profit_pct and self.profit_head is not None:
            proba = self.profit_head.predict_proba(row)[0]
            p_prof = float(proba[1]) if len(proba) > 1 else float(proba[0])
            if p_prof >= self.profit_threshold:
                return ("profit_exit", float(close_price))

        return None


def evaluate_exit_ladder(
    *,
    direction: int,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    bar: MarketSnapshot,
    bars_held: int,
    max_holding_bars: int,
    min_hold_bars: int = 1,
    exit_plan: Optional[ExitPlan] = None,
    trail_state: Optional[TrailState] = None,
    exit_model: Optional[ExitModel] = None,
    exit_mode: str = "hard_limits",  # "hard_limits" | "ml_exit" | "hybrid"
    mfe_so_far_pct: float = 0.0,
    mae_so_far_pct: float = 0.0,
    fv: Optional[FeatureVector] = None,
) -> Optional[Tuple[str, float]]:
    """Evaluate the full 6-step exit precedence ladder on `bar`."""
    eff_tp_pct = exit_plan.take_profit_pct if exit_plan is not None else take_profit_pct
    eff_sl_pct = exit_plan.stop_loss_pct if exit_plan is not None else stop_loss_pct
    eff_max_hold = exit_plan.max_holding_bars if exit_plan is not None else max_holding_bars
    hard_cap_pct = exit_plan.hard_max_loss_pct if exit_plan is not None else None
    trail_spec = exit_plan.trailing if exit_plan is not None else None

    prior_trail_level = (
        trail_state.trail_stop_level
        if (trail_state is not None and trail_state.activated)
        else None
    )

    # 0. Minimum-hold lockout (never overridden by any exit rule or ML model)
    if bars_held < min_hold_bars:
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return None

    tp_level, sl_level = barrier_levels(direction, entry_price, eff_tp_pct, eff_sl_pct)

    # 1. Tighten-only hard loss cap (C2)
    if hard_cap_pct is not None and 0 < hard_cap_pct < eff_sl_pct:
        _, hard_level = barrier_levels(direction, entry_price, eff_tp_pct, hard_cap_pct)
        hit_hard = bar.low <= hard_level if direction == 1 else bar.high >= hard_level
        if hit_hard:
            exit_px = hard_level
            if min_hold_bars > 1 and bars_held == min_hold_bars:
                exit_px = min(hard_level, bar.open) if direction == 1 else max(hard_level, bar.open)
            if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
                advance_trail(direction, entry_price, bar, trail_spec, trail_state)
            return ("hard_loss_cap", exit_px)

    if direction == 1:
        hit_tp, hit_sl = bar.high >= tp_level, bar.low <= sl_level
    else:
        hit_tp, hit_sl = bar.low <= tp_level, bar.high >= sl_level

    # 2. Deterministic stop-loss
    if hit_sl:
        exit_px = sl_level
        if min_hold_bars > 1 and bars_held == min_hold_bars:
            exit_px = min(sl_level, bar.open) if direction == 1 else max(sl_level, bar.open)
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return ("stop_loss", exit_px)

    # 3. Deterministic trailing stop (evaluated on prior_trail_level — 1-bar lag)
    if prior_trail_level is not None:
        hit_trail = bar.low <= prior_trail_level if direction == 1 else bar.high >= prior_trail_level
        if hit_trail:
            trail_px = min(prior_trail_level, bar.open) if direction == 1 else max(prior_trail_level, bar.open)
            if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
                advance_trail(direction, entry_price, bar, trail_spec, trail_state)
            return ("trailing_stop", trail_px)

    # 4. Deterministic take-profit
    if hit_tp:
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return ("take_profit", tp_level)

    # 5. Per-expert ML ExitModel (`loss_cut` / `profit_exit`) when exit_mode in ("ml_exit", "hybrid")
    if exit_model is not None and exit_mode in ("ml_exit", "hybrid"):
        state_feats = build_open_position_state_features(
            direction=direction,
            entry_price=entry_price,
            bar=bar,
            bars_held=bars_held,
            max_holding_bars=eff_max_hold,
            take_profit_pct=eff_tp_pct,
            stop_loss_pct=eff_sl_pct,
            mfe_so_far_pct=mfe_so_far_pct,
            mae_so_far_pct=mae_so_far_pct,
            fv=fv,
        )
        ml_step = exit_model.evaluate_bar(state_feats, bar.close)
        if ml_step is not None:
            if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
                advance_trail(direction, entry_price, bar, trail_spec, trail_state)
            return ml_step

    # 6. Timeout
    if bars_held >= eff_max_hold:
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return ("timeout", bar.close)

    if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
        advance_trail(direction, entry_price, bar, trail_spec, trail_state)
    return None


@dataclass(frozen=True)
class PairedRouterOutcome:
    trade_id: str
    expert_id: str
    hard_exit_reason: str
    hard_return_pct: float
    ml_exit_reason: str
    ml_return_pct: float


def simulate_with_exit_ladder(
    forward_bars: Sequence[MarketSnapshot],
    *,
    entry_price: float,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    min_hold_bars: int = 1,
    fee_pct: float = 0.04,
    exit_plan: Optional[ExitPlan] = None,
    exit_model: Optional[ExitModel] = None,
    exit_mode: str = "hard_limits",
) -> Tuple[str, int, float]:
    """Return `(exit_reason, bars_held, realized_return_pct)` under `evaluate_exit_ladder`."""
    window = list(forward_bars[: max(1, int(max_holding_bars))])
    trail_state = TrailState(best_favorable_price=entry_price)
    mfe = 0.0
    mae = 0.0
    for i, bar in enumerate(window):
        bars_held = i + 1
        fav = (bar.high - entry_price) / entry_price * 100.0 if direction == 1 else (entry_price - bar.low) / entry_price * 100.0
        adv = (entry_price - bar.low) / entry_price * 100.0 if direction == 1 else (bar.high - entry_price) / entry_price * 100.0
        mfe = max(mfe, fav)
        mae = max(mae, adv)
        step = evaluate_exit_ladder(
            direction=direction,
            entry_price=entry_price,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            bar=bar,
            bars_held=bars_held,
            max_holding_bars=max_holding_bars,
            min_hold_bars=min_hold_bars,
            exit_plan=exit_plan,
            trail_state=trail_state,
            exit_model=exit_model,
            exit_mode=exit_mode,
            mfe_so_far_pct=mfe,
            mae_so_far_pct=mae,
        )
        if step is not None:
            reason, px = step
            return reason, bars_held, raw_return_pct(direction, entry_price, px) - 2.0 * fee_pct

    last = window[-1]
    return "timeout", len(window), raw_return_pct(direction, entry_price, last.close) - 2.0 * fee_pct
