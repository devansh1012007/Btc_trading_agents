"""
The harness-aligned labeler: label a training candidate exactly the way the
harness would have traded it.

Before this module every training script labeled outcomes with its own copy
of the same ~15 lines, and those lines described a trade the harness never
takes: entry at the signal bar's CLOSE (the harness pays the simulated
FILL), no minimum hold, and every candidate on every bar (the harness holds
one position at a time). This is the one place that logic lives now.

A label here is produced by, in order:

1. **entry price** -- :func:`src.execution.execution_engine.entry_fill_price`
   with the bar's own spread and the configured slippage (the paper
   backend's own functions);
2. **barriers** -- :func:`src.labeling.triple_barrier.barriers_for_regime`
   (regime-specific ``k_tp`` / ``k_sl`` scaled by the entry bar's volatility);
3. **exit** -- :func:`src.labeling.triple_barrier.label_triple_barrier`, i.e.
   :func:`barrier_exit_step` with the minimum-hold lockout from
   :class:`~src.labeling.hold_policy.HoldPolicy`;
4. **fees** -- round-trip ``labeling.fee_pct``.

What it deliberately does NOT replay per sample: the router's choice
between experts and the risk layer's size / veto. ``select_sequential``
reproduces the one-position-at-a-time rule and the order cool-down;
``scripts/evaluate_training_alignment.py`` measures the remaining gap
against a real harness run.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..execution.execution_engine import entry_fill_price, total_slippage_pct
from ..state.contracts import MarketSnapshot, SystemState
from .hold_policy import HoldPolicy
from .triple_barrier import (
    TripleBarrierOutcome,
    barrier_volatility_pct,
    barriers_for_regime,
    label_triple_barrier,
)

SAMPLE_POLICIES = ("sequential", "all")


@dataclass(frozen=True)
class LabelContext:
    """Everything the labeler needs, derived once from the merged config."""

    policy: HoldPolicy
    config: Mapping[str, Any]
    fee_pct: float
    slippage_pct: float
    min_bars_between_orders: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LabelContext":
        exec_cfg = config.get("execution", {}) or {}
        risk_cfg = config.get("risk", {}) or {}
        breaker_cfg = risk_cfg.get("circuit_breakers", {}) or {}
        return cls(
            policy=HoldPolicy.from_config(config),
            config=config,
            fee_pct=float((config.get("labeling", {}) or {}).get("fee_pct", 0.04)),
            slippage_pct=total_slippage_pct(exec_cfg),
            min_bars_between_orders=int(
                exec_cfg.get("min_bars_between_orders", breaker_cfg.get("min_bars_between_orders", 2))
            ),
        )

    @property
    def min_hold_bars(self) -> int:
        return self.policy.min_hold_bars


def entry_price_for(ctx: LabelContext, snapshot: MarketSnapshot, direction: int) -> float:
    """The price the harness would have paid to enter on ``snapshot``."""
    side = "BUY" if direction == 1 else "SELL"
    return entry_fill_price(side, snapshot.close, snapshot.spread_pct, ctx.slippage_pct)


def outcome_row(outcome: TripleBarrierOutcome, ctx: LabelContext) -> Dict[str, Any]:
    """The label columns every training set carries (one place, one schema)."""
    return {
        "exit_reason": outcome.exit_reason,
        "bars_held": outcome.bars_held,
        "hold_seconds": ctx.policy.held_seconds(outcome.bars_held),
        "realized_return_pct": outcome.realized_return_pct,
        "max_adverse_excursion_pct": outcome.max_adverse_excursion_pct,
        "pinned_to_min_hold": outcome.pinned_to_min_hold,
        "lockout_adverse_excursion_pct": outcome.lockout_adverse_excursion_pct,
        "label": 1 if outcome.realized_return_pct > 0 else 0,
    }


def label_candidate(
    ctx: LabelContext, state: SystemState, forward_bars: Sequence[MarketSnapshot], direction: int,
    *, min_hold_bars: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Label one candidate as the harness would have traded it.

    Returns ``None`` when the candidate has no legal outcome in the data:
    no forward bars, or the data ends before the position could be
    exited -- a window shorter than the minimum hold, or an unresolved
    timeout cut off by the end of the data. (The harness would still be
    holding that position; labeling it would invent an exit.)

    ``min_hold_bars`` overrides the policy's, for the lockout-cost eval
    (``min_hold_bars=1`` labels the same candidate with no lockout).
    """
    if not forward_bars:
        return None
    hold_bars = ctx.min_hold_bars if min_hold_bars is None else min_hold_bars
    max_hold = ctx.policy.max_holding_bars

    regime_name = state.regime_state.primary_regime.name
    tp_pct, sl_pct = barriers_for_regime(
        ctx.config, regime_name, barrier_volatility_pct(state.feature_vector),
    )
    entry_price = entry_price_for(ctx, state.market_snapshot, direction)

    outcome = label_triple_barrier(
        forward_bars=list(forward_bars), entry_price=entry_price, direction=direction,
        take_profit_pct=tp_pct, stop_loss_pct=sl_pct, max_holding_bars=max_hold,
        fee_pct=ctx.fee_pct, entry_timestamp=state.timestamp, min_hold_bars=hold_bars,
    )
    if outcome.bars_held < hold_bars:
        return None                       # window ended inside the lockout
    if outcome.exit_reason == "timeout" and outcome.bars_held < max_hold:
        return None                       # window ended before the timeout: unresolved

    return {
        "entry_price": entry_price, "take_profit_pct": tp_pct, "stop_loss_pct": sl_pct,
        "regime": regime_name, **outcome_row(outcome, ctx),
    }


def select_sequential(rows: List[Mapping[str, Any]], min_bars_between_orders: int) -> List[bool]:
    """Keep only the candidates a one-position-at-a-time harness could take.

    Per expert, in bar order: a candidate is kept only if the previous KEPT
    position has exited (exits are processed before entries on a bar, so
    the exit bar itself is allowed) and the order cool-down has passed.
    Returns a keep-mask aligned with ``rows``.
    """
    keep = [False] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: (str(rows[i]["expert_name"]), int(rows[i]["bar_index"])))
    next_free: Dict[str, int] = {}
    for i in order:
        row = rows[i]
        expert = str(row["expert_name"])
        bar_index = int(row["bar_index"])
        if bar_index < next_free.get(expert, -1):
            continue
        keep[i] = True
        next_free[expert] = bar_index + max(int(row["bars_held"]), int(min_bars_between_orders))
    return keep
