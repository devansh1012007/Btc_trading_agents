"""
Triple-barrier labeling (§5 + Phase 3 & Phase 4): given an entry and a
forward window of bars, find which barrier gets touched first -- hard loss
cap, stop-loss, trailing-stop, take-profit, ML exit, or timeout -- and
compute the resulting net-of-costs return.

Provides:
* `EXIT_REASONS`, `STOP_EXIT_REASONS`, `PROFIT_EXIT_REASONS` (canonical vocabulary)
* `TrailState` and `advance_trail` (1-bar ratchet lag to prevent same-bar look-ahead)
* `resolve_exit_plan` (precedence ladder + tighten-only `hard_max_loss_pct` cap)
* `barrier_exit_step` and `label_triple_barrier`
"""
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

from ..state.contracts import ExitPlan, ExpertOutput, MarketSnapshot, TrailSpec

if TYPE_CHECKING:
    from ..execution.cost_model import CostModel
    from ..execution.price_path import PricePath

#: Canonical exit-reason vocabulary shared across labeler, harness, GUI, and reports.
EXIT_REASONS: Tuple[str, ...] = (
    "take_profit",
    "stop_loss",
    "trailing_stop",
    "profit_exit",
    "loss_cut",
    "hard_loss_cap",
    "timeout",
)

STOP_EXIT_REASONS: Tuple[str, ...] = (
    "stop_loss",
    "trailing_stop",
    "loss_cut",
    "hard_loss_cap",
)

PROFIT_EXIT_REASONS: Tuple[str, ...] = (
    "take_profit",
    "profit_exit",
)


def is_stop_exit_reason(reason: str) -> bool:
    return str(reason) in STOP_EXIT_REASONS


def is_valid_exit_reason(reason: str) -> bool:
    return str(reason) in EXIT_REASONS


@dataclass
class TrailState:
    """Trailing-stop ratchet state with 1-bar lag."""

    best_favorable_price: float
    activated: bool = False
    trail_stop_level: Optional[float] = None
    history: List[Optional[float]] = field(default_factory=list)


def advance_trail(
    direction: int,
    entry_price: float,
    bar: MarketSnapshot,
    trail_spec: Optional[TrailSpec],
    trail_state: Optional[TrailState] = None,
) -> TrailState:
    """Update ``trail_state`` from ``bar`` AFTER checking bar's exit conditions (1-bar lag)."""
    state = trail_state if trail_state is not None else TrailState(best_favorable_price=entry_price)
    if trail_spec is None or not trail_spec.enabled or trail_spec.distance_pct <= 0.0:
        state.history.append(None)
        return state

    if direction == 1:
        new_best = max(state.best_favorable_price, float(bar.high))
        fav_pct = (new_best - entry_price) / entry_price * 100.0
    else:
        new_best = min(state.best_favorable_price, float(bar.low))
        fav_pct = (entry_price - new_best) / entry_price * 100.0

    if fav_pct >= trail_spec.activation_pct - 1e-9:
        state.activated = True

    if state.activated:
        cand_level = (
            new_best * (1.0 - trail_spec.distance_pct / 100.0)
            if direction == 1
            else new_best * (1.0 + trail_spec.distance_pct / 100.0)
        )
        if state.trail_stop_level is None:
            state.trail_stop_level = cand_level
            state.best_favorable_price = new_best
        else:
            step_move_pct = abs(cand_level - state.trail_stop_level) / entry_price * 100.0
            better = (
                cand_level > state.trail_stop_level
                if direction == 1
                else cand_level < state.trail_stop_level
            )
            if better and step_move_pct >= max(0.0, trail_spec.step_pct) - 1e-9:
                state.trail_stop_level = cand_level
                state.best_favorable_price = new_best
    else:
        state.best_favorable_price = new_best

    state.history.append(state.trail_stop_level)
    return state


def resolve_exit_plan(
    config: Mapping[str, Any],
    *,
    expert_output: Optional[ExpertOutput] = None,
    barrier_model_plan: Optional[ExitPlan] = None,
    regime_name: str = "TRENDING_UP",
    volatility_pct: float = 0.2,
) -> ExitPlan:
    """Resolve the active `ExitPlan` according to the precedence ladder:
    1. `expert_output.exit_plan` (when `exits.plan_source in ("expert", "auto")`)
    2. `barrier_model_plan` (when `exits.plan_source in ("barrier_model", "auto")`)
    3. `barriers_for_regime(config, regime_name, volatility_pct)` (`k_tp * vol`, `k_sl * vol`)
    4. `exits.hard` fallback (`stop_loss_pct`, `take_profit_pct`, `max_holding_bars`)
    Plus tighten-only `exits.hard.max_loss_pct` cap (C2).
    """
    exits_cfg = config.get("exits", {}) or {}
    labeling_cfg = config.get("labeling", {}) or {}
    hard_cfg = exits_cfg.get("hard", {}) or {}
    trail_cfg = exits_cfg.get("trailing") or hard_cfg.get("trailing") or {}
    plan_source = str(exits_cfg.get("plan_source", "auto"))

    default_max_hold = int(hard_cfg.get("max_holding_bars", labeling_cfg.get("max_holding_bars", 5)))
    hard_max_loss: Optional[float] = None
    if hard_cfg.get("max_loss_pct") is not None:
        val = float(hard_cfg["max_loss_pct"])
        if val > 0:
            hard_max_loss = val

    default_trail = TrailSpec.from_dict(trail_cfg) if trail_cfg.get("enabled") else None

    if plan_source in ("expert", "auto") and expert_output is not None and expert_output.exit_plan is not None:
        ep = expert_output.exit_plan
        return ExitPlan(
            stop_loss_pct=ep.stop_loss_pct,
            take_profit_pct=ep.take_profit_pct,
            max_holding_bars=ep.max_holding_bars,
            trailing=ep.trailing if ep.trailing is not None else default_trail,
            hard_max_loss_pct=(
                min(ep.hard_max_loss_pct, hard_max_loss)
                if (ep.hard_max_loss_pct is not None and hard_max_loss is not None)
                else (ep.hard_max_loss_pct if ep.hard_max_loss_pct is not None else hard_max_loss)
            ),
            source="expert",
        )

    if plan_source in ("barrier_model", "auto") and barrier_model_plan is not None:
        bp = barrier_model_plan
        return ExitPlan(
            stop_loss_pct=bp.stop_loss_pct,
            take_profit_pct=bp.take_profit_pct,
            max_holding_bars=bp.max_holding_bars,
            trailing=bp.trailing if bp.trailing is not None else default_trail,
            hard_max_loss_pct=(
                min(bp.hard_max_loss_pct, hard_max_loss)
                if (bp.hard_max_loss_pct is not None and hard_max_loss is not None)
                else (bp.hard_max_loss_pct if bp.hard_max_loss_pct is not None else hard_max_loss)
            ),
            source="barrier_model",
        )

    if plan_source == "hard_limits" or (
        "stop_loss_pct" in hard_cfg and "take_profit_pct" in hard_cfg and plan_source == "hard"
    ):
        sl_pct = float(hard_cfg.get("stop_loss_pct", 1.0))
        tp_pct = float(hard_cfg.get("take_profit_pct", 1.5))
        return ExitPlan(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            max_holding_bars=default_max_hold,
            trailing=default_trail,
            hard_max_loss_pct=hard_max_loss,
            source="hard_limits",
        )

    tp_pct, sl_pct = barriers_for_regime(config, regime_name, volatility_pct)
    return ExitPlan(
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        max_holding_bars=default_max_hold,
        trailing=default_trail,
        hard_max_loss_pct=hard_max_loss,
        source="regime_specific",
    )


@dataclass(frozen=True)
class PendingExit:
    """Two-phase exit state when `exit_delay_seconds` spans one or more bars."""

    trigger_timestamp: int
    trigger_bars_held: int
    exit_reason: str
    trigger_price: float
    fill_due_timestamp: float


@dataclass(frozen=True)
class TripleBarrierOutcome:
    entry_timestamp: int
    direction: int              # +1 long, -1 short
    exit_reason: str            # one of EXIT_REASONS
    bars_held: int
    realized_return_pct: float  # net of round-trip fees and exit costs
    max_adverse_excursion_pct: float
    bars_to_resolution: int
    pinned_to_min_hold: bool = False
    lockout_adverse_excursion_pct: float = 0.0
    exit_fill_price: Optional[float] = None
    exit_trigger_price: Optional[float] = None
    trail_history: Tuple[Optional[float], ...] = ()


def barriers_for_regime(
    config: Mapping[str, Any], regime_name: str, volatility_pct: float,
) -> Tuple[float, float]:
    """Take-profit / stop-loss percentages for an entry in ``regime_name``."""
    labeling_cfg = config.get("labeling", {}) or {}
    overrides = (labeling_cfg.get("regime_specific_barriers", {}) or {}).get(regime_name, {}) or {}
    k_tp = overrides.get("k_tp", labeling_cfg.get("k_tp", 1.5))
    k_sl = overrides.get("k_sl", labeling_cfg.get("k_sl", 1.0))
    return compute_barrier_pcts(volatility_pct, k_tp, k_sl)


def barrier_levels(direction: int, entry_price: float, tp_pct: float, sl_pct: float) -> Tuple[float, float]:
    """(take-profit level, stop-loss level) for a long (+1) or short (-1)."""
    if direction == 1:
        return entry_price * (1 + tp_pct / 100), entry_price * (1 - sl_pct / 100)
    return entry_price * (1 - tp_pct / 100), entry_price * (1 + sl_pct / 100)


def barrier_exit_step(
    direction: int,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    bar: MarketSnapshot,
    bars_held: int,
    max_holding_bars: int,
    min_hold_bars: int = 1,
    *,
    exit_plan: Optional[ExitPlan] = None,
    trail_state: Optional[TrailState] = None,
) -> Optional[Tuple[str, float]]:
    """One bar of the exit rule: ``(exit_reason, trigger_price)`` or ``None``.

    Precedence on a bar after the minimum-hold lockout:
    1. `hard_loss_cap` (when `exit_plan.hard_max_loss_pct` is set and tighter than `stop_loss_pct`)
    2. `stop_loss`
    3. `trailing_stop` (evaluated against `trail_state.trail_stop_level` from PREVIOUS bars — 1-bar lag)
    4. `take_profit`
    5. `timeout`
    """
    eff_tp_pct = exit_plan.take_profit_pct if exit_plan is not None else take_profit_pct
    eff_sl_pct = exit_plan.stop_loss_pct if exit_plan is not None else stop_loss_pct
    eff_max_hold = exit_plan.max_holding_bars if exit_plan is not None else max_holding_bars
    hard_cap_pct = exit_plan.hard_max_loss_pct if exit_plan is not None else None
    trail_spec = exit_plan.trailing if exit_plan is not None else None

    # Read active trailing stop level from PRIOR bars (1-bar lag) BEFORE advancing trail on this bar
    prior_trail_level = (
        trail_state.trail_stop_level
        if (trail_state is not None and trail_state.activated)
        else None
    )

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
            exit_price = hard_level
            if min_hold_bars > 1 and bars_held == min_hold_bars:
                exit_price = min(hard_level, bar.open) if direction == 1 else max(hard_level, bar.open)
            if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
                advance_trail(direction, entry_price, bar, trail_spec, trail_state)
            return "hard_loss_cap", exit_price

    if direction == 1:
        hit_tp, hit_sl = bar.high >= tp_level, bar.low <= sl_level
    else:
        hit_tp, hit_sl = bar.low <= tp_level, bar.high >= sl_level

    # 2. Stop-loss
    if hit_sl:
        exit_price = sl_level
        if min_hold_bars > 1 and bars_held == min_hold_bars:
            exit_price = min(sl_level, bar.open) if direction == 1 else max(sl_level, bar.open)
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return "stop_loss", exit_price

    # 3. Trailing stop (checked against prior_trail_level)
    if prior_trail_level is not None:
        hit_trail = bar.low <= prior_trail_level if direction == 1 else bar.high >= prior_trail_level
        if hit_trail:
            trail_exit_px = (
                min(prior_trail_level, bar.open)
                if direction == 1
                else max(prior_trail_level, bar.open)
            )
            if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
                advance_trail(direction, entry_price, bar, trail_spec, trail_state)
            return "trailing_stop", trail_exit_px

    # 4. Take-profit
    if hit_tp:
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return "take_profit", tp_level

    # 5. Timeout
    if bars_held >= eff_max_hold:
        if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
            advance_trail(direction, entry_price, bar, trail_spec, trail_state)
        return "timeout", bar.close

    # Advance trail on this bar so next bar sees updated trail_stop_level
    if trail_state is not None or (trail_spec is not None and trail_spec.enabled):
        advance_trail(direction, entry_price, bar, trail_spec, trail_state)
    return None


def raw_return_pct(direction: int, entry_price: float, exit_price: float) -> float:
    """Gross return (%) of a long/short from entry to exit, before fees."""
    if direction == 1:
        return (exit_price - entry_price) / entry_price * 100
    return (entry_price - exit_price) / entry_price * 100


def compute_barrier_pcts(volatility_pct: float, k_tp: float, k_sl: float) -> Tuple[float, float]:
    """Volatility-scaled barrier sizing (§5, §9): TP = k_tp * vol, SL = k_sl * vol."""
    return k_tp * volatility_pct, k_sl * volatility_pct


def barrier_volatility_pct(fv: Any) -> float:
    """Volatility figure used for barrier sizing (`fv.volatility_medium`)."""
    return max(float(fv.volatility_medium), 1e-6)


def label_triple_barrier(
    forward_bars: List[MarketSnapshot],
    entry_price: float,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    fee_pct: float,
    entry_timestamp: Optional[int] = None,
    min_hold_bars: int = 1,
    *,
    cost_model: Optional["CostModel"] = None,
    exit_model: str = "barrier_level",
    granularity_seconds: float = 60.0,
    price_path: Optional["PricePath"] = None,
    size_usdt: float = 0.0,
    exit_plan: Optional[ExitPlan] = None,
) -> TripleBarrierOutcome:
    """Walk `forward_bars` applying :func:`barrier_exit_step` and :func:`exit_fill_price`."""
    from ..execution.execution_engine import exit_fill_price

    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long) or -1 (short)")
    if not forward_bars:
        raise ValueError("forward_bars must be non-empty")
    eff_max_hold = exit_plan.max_holding_bars if exit_plan is not None else max_holding_bars
    if eff_max_hold <= 0:
        raise ValueError("max_holding_bars must be positive")
    if min_hold_bars < 1:
        raise ValueError("min_hold_bars must be >= 1")
    if eff_max_hold < min_hold_bars:
        raise ValueError(
            f"max_holding_bars={eff_max_hold} is shorter than min_hold_bars={min_hold_bars}: "
            f"every timeout would be an illegal trade"
        )

    exit_delay_s = cost_model.exit_delay_seconds if cost_model is not None else 0.0
    extra_delay_bars = (
        int(math.ceil(exit_delay_s / max(granularity_seconds, 1e-6) - 1e-9))
        if exit_delay_s >= granularity_seconds > 0
        else 0
    )
    horizon_bars = eff_max_hold + extra_delay_bars
    bars_to_check = forward_bars[:horizon_bars]
    entry_ts = entry_timestamp if entry_timestamp is not None else bars_to_check[0].timestamp
    round_trip_fees = cost_model.round_trip_fee_pct() if cost_model is not None else (fee_pct * 2.0)

    max_adverse_excursion_pct = 0.0
    lockout_adverse_excursion_pct = 0.0
    pending: Optional[PendingExit] = None
    trail_state = TrailState(best_favorable_price=entry_price)

    for i, bar in enumerate(bars_to_check):
        bars_held = i + 1
        adverse = (
            (entry_price - bar.low) / entry_price * 100
            if direction == 1
            else (bar.high - entry_price) / entry_price * 100
        )
        max_adverse_excursion_pct = max(max_adverse_excursion_pct, adverse)
        if bars_held < min_hold_bars:
            lockout_adverse_excursion_pct = max(lockout_adverse_excursion_pct, adverse)

        if pending is not None:
            elapsed_s = float(bar.timestamp - pending.trigger_timestamp)
            bars_since = bars_held - pending.trigger_bars_held
            if elapsed_s >= exit_delay_s - 1e-6 or bars_since * granularity_seconds >= exit_delay_s - 1e-6:
                next_bar = forward_bars[i + 1] if (i + 1) < len(forward_bars) else None
                fill_px = exit_fill_price(
                    direction=direction,
                    trigger_price=bar.open,
                    bar=bar,
                    exit_reason=pending.exit_reason,
                    cost_model=cost_model,
                    exit_model=exit_model,
                    next_bar=next_bar,
                    granularity_seconds=granularity_seconds,
                    size_usdt=size_usdt,
                    price_path=price_path,
                )
                hold_s = max(0.0, float(bar.timestamp - entry_ts))
                carry_cost = cost_model.carry_pct(direction, hold_s) if cost_model is not None else 0.0
                return TripleBarrierOutcome(
                    entry_timestamp=entry_ts,
                    direction=direction,
                    exit_reason=pending.exit_reason,
                    bars_held=bars_held,
                    realized_return_pct=raw_return_pct(direction, entry_price, fill_px) - round_trip_fees - carry_cost,
                    max_adverse_excursion_pct=max_adverse_excursion_pct,
                    bars_to_resolution=bars_held,
                    pinned_to_min_hold=(
                        min_hold_bars > 1
                        and pending.trigger_bars_held == min_hold_bars
                        and pending.exit_reason != "timeout"
                    ),
                    lockout_adverse_excursion_pct=lockout_adverse_excursion_pct,
                    exit_fill_price=fill_px,
                    exit_trigger_price=pending.trigger_price,
                    trail_history=tuple(trail_state.history),
                )
            continue

        step = barrier_exit_step(
            direction, entry_price, take_profit_pct, stop_loss_pct, bar,
            bars_held, eff_max_hold, min_hold_bars,
            exit_plan=exit_plan, trail_state=trail_state,
        )
        if step is None:
            continue

        exit_reason, trigger_price = step
        if exit_delay_s >= granularity_seconds > 0:
            pending = PendingExit(
                trigger_timestamp=bar.timestamp,
                trigger_bars_held=bars_held,
                exit_reason=exit_reason,
                trigger_price=trigger_price,
                fill_due_timestamp=float(bar.timestamp) + exit_delay_s,
            )
            continue

        next_bar = forward_bars[i + 1] if (i + 1) < len(forward_bars) else None
        fill_px = exit_fill_price(
            direction=direction,
            trigger_price=trigger_price,
            bar=bar,
            exit_reason=exit_reason,
            cost_model=cost_model,
            exit_model=exit_model,
            next_bar=next_bar,
            granularity_seconds=granularity_seconds,
            size_usdt=size_usdt,
            price_path=price_path,
        )
        hold_s = max(0.0, float(bar.timestamp - entry_ts) + exit_delay_s)
        carry_cost = cost_model.carry_pct(direction, hold_s) if cost_model is not None else 0.0
        return TripleBarrierOutcome(
            entry_timestamp=entry_ts,
            direction=direction,
            exit_reason=exit_reason,
            bars_held=bars_held,
            realized_return_pct=raw_return_pct(direction, entry_price, fill_px) - round_trip_fees - carry_cost,
            max_adverse_excursion_pct=max_adverse_excursion_pct,
            bars_to_resolution=bars_held,
            pinned_to_min_hold=(
                min_hold_bars > 1 and bars_held == min_hold_bars and exit_reason != "timeout"
            ),
            lockout_adverse_excursion_pct=lockout_adverse_excursion_pct,
            exit_fill_price=fill_px,
            exit_trigger_price=trigger_price,
            trail_history=tuple(trail_state.history),
        )

    last_bar = bars_to_check[-1]
    final_reason = pending.exit_reason if pending is not None else "timeout"
    final_trigger = pending.trigger_price if pending is not None else last_bar.close
    fill_px = exit_fill_price(
        direction=direction,
        trigger_price=final_trigger,
        bar=last_bar,
        exit_reason=final_reason,
        cost_model=cost_model,
        exit_model=exit_model,
        granularity_seconds=granularity_seconds,
        size_usdt=size_usdt,
        price_path=price_path,
    )
    hold_s = max(0.0, float(last_bar.timestamp - entry_ts) + exit_delay_s)
    carry_cost = cost_model.carry_pct(direction, hold_s) if cost_model is not None else 0.0
    return TripleBarrierOutcome(
        entry_timestamp=entry_ts,
        direction=direction,
        exit_reason=final_reason,
        bars_held=len(bars_to_check),
        realized_return_pct=raw_return_pct(direction, entry_price, fill_px) - round_trip_fees - carry_cost,
        max_adverse_excursion_pct=max_adverse_excursion_pct,
        bars_to_resolution=len(bars_to_check),
        pinned_to_min_hold=False,
        lockout_adverse_excursion_pct=lockout_adverse_excursion_pct,
        exit_fill_price=fill_px,
        exit_trigger_price=final_trigger,
        trail_history=tuple(trail_state.history),
    )
