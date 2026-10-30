"""
The harness-aligned labeler: label a training candidate exactly the way the
harness would have traded it, using `CostModel`, `HoldPolicy`,
`entry_fill_price_with_delay`, and `label_triple_barrier`.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..execution.cost_model import CostModel
from ..execution.execution_engine import entry_fill_price, entry_fill_price_with_delay
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
    cost_model: CostModel = CostModel()
    exit_model: str = "barrier_level"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LabelContext":
        exec_cfg = config.get("execution", {}) or {}
        risk_cfg = config.get("risk", {}) or {}
        breaker_cfg = risk_cfg.get("circuit_breakers", {}) or {}
        cost_model = CostModel.from_config(config)
        exit_model = str(exec_cfg.get("exit_model", "barrier_level"))
        return cls(
            policy=HoldPolicy.from_config(config),
            config=config,
            fee_pct=cost_model.fee_pct("entry"),
            slippage_pct=cost_model.slippage_pct("entry"),
            min_bars_between_orders=int(
                exec_cfg.get("min_bars_between_orders", breaker_cfg.get("min_bars_between_orders", 2))
            ),
            cost_model=cost_model,
            exit_model=exit_model,
        )

    @property
    def min_hold_bars(self) -> int:
        return self.policy.min_hold_bars


def entry_price_for(
    ctx: LabelContext,
    snapshot: MarketSnapshot,
    direction: int,
    *,
    next_bar: Optional[MarketSnapshot] = None,
) -> float:
    """The price the harness would have paid to enter on ``snapshot``."""
    side = "BUY" if direction == 1 else "SELL"
    if ctx.cost_model.entry_delay_seconds > 0.0 and next_bar is not None:
        px, _ = entry_fill_price_with_delay(
            side,
            snapshot,
            ctx.cost_model,
            next_bar=next_bar,
            granularity_seconds=ctx.policy.granularity_seconds,
        )
        return px
    return entry_fill_price(
        side,
        snapshot.close,
        ctx.cost_model.spread_pct(snapshot, leg="entry"),
        ctx.slippage_pct,
    )


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
    ctx: LabelContext,
    state: SystemState,
    forward_bars: Sequence[MarketSnapshot],
    direction: int,
    *,
    min_hold_bars: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Label one candidate as the harness would have traded it."""
    if not forward_bars:
        return None
    hold_bars = ctx.min_hold_bars if min_hold_bars is None else min_hold_bars
    max_hold = ctx.policy.max_holding_bars

    regime_name = state.regime_state.primary_regime.name
    tp_pct, sl_pct = barriers_for_regime(
        ctx.config, regime_name, barrier_volatility_pct(state.feature_vector),
    )
    entry_price = entry_price_for(
        ctx, state.market_snapshot, direction, next_bar=forward_bars[0] if forward_bars else None,
    )

    outcome = label_triple_barrier(
        forward_bars=list(forward_bars),
        entry_price=entry_price,
        direction=direction,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        max_holding_bars=max_hold,
        fee_pct=ctx.fee_pct,
        entry_timestamp=state.timestamp,
        min_hold_bars=hold_bars,
        cost_model=ctx.cost_model,
        exit_model=ctx.exit_model,
        granularity_seconds=ctx.policy.granularity_seconds,
    )
    if outcome.bars_held < hold_bars:
        return None
    if outcome.exit_reason == "timeout" and outcome.bars_held < max_hold:
        return None

    return {
        "entry_price": entry_price,
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
        "regime": regime_name,
        **outcome_row(outcome, ctx),
    }


def select_sequential(rows: List[Mapping[str, Any]], min_bars_between_orders: int) -> List[bool]:
    """Keep only the candidates a one-position-at-a-time harness could take."""
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


def population_rules_hash(config: Mapping[str, Any]) -> str:
    """Deterministic hash of the population-selection rules (`positions`, `cooldowns`, `min_bars_between_orders`)."""
    import hashlib
    import json

    from ..execution.position_book import PositionLimits
    from ..risk.cooldowns import CooldownConfig

    exec_cfg = config.get("execution") or {}
    risk_cfg = config.get("risk") or {}
    breaker_cfg = risk_cfg.get("circuit_breakers") or {}
    min_bars = int(exec_cfg.get("min_bars_between_orders", breaker_cfg.get("min_bars_between_orders", 2)))
    limits = PositionLimits.from_config(config)
    cd = CooldownConfig.from_config(config)
    payload = {
        "min_bars_between_orders": min_bars,
        "max_concurrent": limits.max_concurrent,
        "max_per_expert": limits.max_per_expert,
        "max_per_regime": limits.max_per_regime,
        "max_per_direction": limits.max_per_direction,
        "allow_same_expert_stacking": limits.allow_same_expert_stacking,
        "allow_hedging": limits.allow_hedging,
        "cooldowns": {
            "enabled": cd.enabled,
            "after_stop_loss_bars": cd.after_stop_loss_bars,
            "consecutive_losses": cd.consecutive_losses,
            "cooldown_bars": cd.cooldown_bars,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def select_population(
    rows: List[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
    *,
    min_bars_between_orders: Optional[int] = None,
    per_expert_independent: bool = True,
) -> List[bool]:
    """Generalize `select_sequential` to respect `PositionLimits` and `CooldownManager`.

    When `per_expert_independent=True` (the default for expert training with
    `max_concurrent == 1` and cool-downs disabled), reproduces `select_sequential`
    exactly. When `per_expert_independent=False` or concurrency/cooldowns are
    configured, simulates the active `PositionLimits` and `CooldownManager`.
    """
    from ..execution.position_book import PositionLimits
    from ..risk.cooldowns import CooldownConfig, CooldownManager

    cfg = config or {}
    exec_cfg = cfg.get("execution") or {}
    risk_cfg = cfg.get("risk") or {}
    breaker_cfg = risk_cfg.get("circuit_breakers") or {}
    min_bars = (
        int(min_bars_between_orders)
        if min_bars_between_orders is not None
        else int(exec_cfg.get("min_bars_between_orders", breaker_cfg.get("min_bars_between_orders", 2)))
    )
    limits = PositionLimits.from_config(cfg)
    cd_cfg = CooldownConfig.from_config(cfg)

    if per_expert_independent and limits.max_concurrent == 1 and not cd_cfg.enabled:
        return select_sequential(rows, min_bars)

    keep = [False] * len(rows)
    if per_expert_independent:
        # Per-expert simulation with per-expert stacking cap and cooldowns
        order = sorted(range(len(rows)), key=lambda i: (str(rows[i]["expert_name"]), int(rows[i]["bar_index"])))
        by_expert: Dict[str, List[int]] = {}
        for idx in order:
            by_expert.setdefault(str(rows[idx]["expert_name"]), []).append(idx)

        max_per_exp = limits.max_per_expert if limits.allow_same_expert_stacking else 1
        for exp_name, indices in by_expert.items():
            active_end_bars: List[int] = []
            last_order_bar = -999999
            cd_mgr = CooldownManager(cd_cfg)
            pending_outcomes: List[tuple] = []  # (end_bar, pnl_pct, exit_reason, timestamp)
            for i in indices:
                row = rows[i]
                bar_idx = int(row["bar_index"])
                ts = float(row.get("timestamp", bar_idx))
                # Resolve any trades that finished on or before bar_idx
                still_open: List[int] = []
                for end_b in active_end_bars:
                    if end_b > bar_idx:
                        still_open.append(end_b)
                active_end_bars = still_open
                remaining_outcomes = []
                for end_b, pnl, reason, end_ts in pending_outcomes:
                    if end_b <= bar_idx:
                        cd_mgr.record_outcome(exp_name, pnl, reason, end_ts, bar_index=end_b)
                    else:
                        remaining_outcomes.append((end_b, pnl, reason, end_ts))
                pending_outcomes = remaining_outcomes

                if bar_idx < last_order_bar + min_bars:
                    continue
                if len(active_end_bars) >= max_per_exp:
                    continue
                cooled, _ = cd_mgr.is_cooled_down(exp_name, ts, current_bar_index=bar_idx)
                if cooled:
                    continue
                keep[i] = True
                held = max(1, int(row["bars_held"]))
                end_bar = bar_idx + held
                active_end_bars.append(end_bar)
                last_order_bar = bar_idx
                pending_outcomes.append((
                    end_bar,
                    float(row.get("realized_return_pct", 0.0)),
                    str(row.get("exit_reason", "timeout")),
                    ts + held,
                ))
        return keep

    # Portfolio-wide simulation across all rows in bar order
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["bar_index"]), str(rows[i]["expert_name"])))
    active_trades: List[Dict[str, Any]] = []
    cd_mgr = CooldownManager(cd_cfg)
    last_order_bar = -999999
    for i in order:
        row = rows[i]
        bar_idx = int(row["bar_index"])
        ts = float(row.get("timestamp", bar_idx))
        exp_name = str(row["expert_name"])
        direction = int(row.get("direction", 1))
        regime = str(row.get("regime", "UNKNOWN"))

        still_active = []
        for tr in active_trades:
            if tr["end_bar"] <= bar_idx:
                cd_mgr.record_outcome(
                    tr["expert"], tr["pnl"], tr["reason"], tr["end_ts"], bar_index=tr["end_bar"]
                )
            else:
                still_active.append(tr)
        active_trades = still_active

        if bar_idx < last_order_bar + min_bars:
            continue
        if len(active_trades) >= limits.max_concurrent:
            continue
        same_exp = sum(1 for tr in active_trades if tr["expert"] == exp_name)
        if (not limits.allow_same_expert_stacking and same_exp >= 1) or same_exp >= limits.max_per_expert:
            continue
        same_reg = sum(1 for tr in active_trades if tr["regime"] == regime)
        if same_reg >= limits.max_per_regime:
            continue
        same_dir = sum(1 for tr in active_trades if tr["direction"] == direction)
        if same_dir >= limits.max_per_direction:
            continue
        if not limits.allow_hedging and any(tr["direction"] == -direction for tr in active_trades):
            continue
        cooled, _ = cd_mgr.is_cooled_down(exp_name, ts, current_bar_index=bar_idx)
        if cooled:
            continue

        keep[i] = True
        held = max(1, int(row["bars_held"]))
        active_trades.append({
            "expert": exp_name,
            "regime": regime,
            "direction": direction,
            "end_bar": bar_idx + held,
            "end_ts": ts + held,
            "pnl": float(row.get("realized_return_pct", 0.0)),
            "reason": str(row.get("exit_reason", "timeout")),
        })
        last_order_bar = bar_idx
    return keep
