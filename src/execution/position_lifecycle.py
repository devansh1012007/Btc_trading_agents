"""
Shared position-lifecycle primitives: the open-position record, the
incremental triple-barrier & ML exit check, the multi-position `PositionBook`
orchestration, external agent handoff (`AgentSink`), `RiskGovernor`,
`BarrierSizingModel`, `ModelLedger`, and the execution-engine factory.

``process_bar`` is the ONE implementation of the per-bar trading block:
exit (across all open positions) -> close_position -> cooldowns/halts ->
risk (+ governor + barrier sizing + agent handoff) -> position_book.can_open ->
order -> open-position -> classify -> trace.
"""
import logging
import time
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..connectors.data_source import CCXTConnector
from ..diagnostics.decision_trace import (
    DecisionTrace,
    StageResult,
    build_trace_stages,
    classify_terminal_stage,
)
from ..exits.exit_model import ExitModel, evaluate_exit_ladder
from ..handoff.agent_sink import AgentSink, TradeSignal, build_agent_sink
from ..labeling.hold_policy import HoldPolicy
from ..labeling.triple_barrier import (
    PendingExit,
    TrailState,
    barrier_exit_step,
    barrier_volatility_pct,
    barriers_for_regime,
    raw_return_pct,
    resolve_exit_plan,
)
from ..layers.risk_layer import RiskLayer
from ..monitoring.ledger import ModelLedger, ModelOutcomeRecord, ModelPredictionRecord
from ..risk.cooldowns import CooldownManager
from ..risk.governor import RiskGovernor
from ..risk.halts import HaltManager
from ..sizing.barrier_model import BarrierSizingModel
from ..state.contracts import (
    ExecutionRecord,
    ExitPlan,
    FeatureVector,
    MarketSnapshot,
    RiskDecision,
    SystemState,
)
from .cost_model import CostModel
from .execution_engine import ExecutionBackend, ExecutionEngine, exit_fill_price
from .position_book import PositionBook
from .price_path import PricePath

logger = logging.getLogger(__name__)

__all__ = [
    "BarOutcome",
    "ClosedTrade",
    "OpenPosition",
    "build_execution_engine",
    "check_position_exit",
    "compute_position_barriers",
    "max_concurrent_positions",
    "process_bar",
]


@dataclass
class OpenPosition:
    """A position that is currently open, with its triple-barrier levels."""

    trade_id: str
    entry_timestamp: int
    entry_price: float
    direction: int
    size_usdt: float
    take_profit_pct: float
    stop_loss_pct: float
    max_holding_bars: int
    bars_held: int = 0
    fees_paid: float = 0.0
    expert: Optional[str] = None
    regime: Optional[str] = None
    selection_confidence: Optional[float] = None
    risk_score: Optional[float] = None
    min_hold_bars: int = 1
    min_hold_seconds: float = 0.0
    entry_latency_seconds: float = 0.0
    exit_delay_seconds: float = 0.0
    exit_model: str = "barrier_level"
    pending_exit: Optional[PendingExit] = None
    exit_plan: Optional[ExitPlan] = None
    trail_state: Optional[TrailState] = None
    lane_id: int = 0
    signal_id: str = ""
    mfe_so_far_pct: float = 0.0
    mae_so_far_pct: float = 0.0


def build_execution_engine(
    config: dict,
    connector: Optional[CCXTConnector],
    symbol: str,
) -> ExecutionEngine:
    """Build the execution engine for the configured ``execution.mode``."""
    exec_cfg = config.get("execution", {})
    mode = exec_cfg.get("mode", "paper_sim")
    cost_model = CostModel.from_config(config)
    entry_fee = cost_model.fee_pct("entry")

    backend: ExecutionBackend
    if mode == "paper_sim":
        from .backends.paper_sim_backend import PaperSimBackend

        backend = PaperSimBackend(exec_cfg, fee_pct=entry_fee, cost_model=cost_model)
    elif mode == "paper_live":
        from .backends.paper_live_backend import PaperLiveBackend

        backend = PaperLiveBackend(exec_cfg, connector, symbol, fee_pct=entry_fee, cost_model=cost_model)
    elif mode == "live":
        from .backends.live_backend import LiveBackend

        backend = LiveBackend(exec_cfg, connector, symbol, cost_model=cost_model)
    else:
        raise ValueError(f"Unknown execution.mode: {mode}")

    return ExecutionEngine(backend, mode)


def check_position_exit(
    position: OpenPosition,
    bar: MarketSnapshot,
    fee_pct: float,
    *,
    cost_model: Optional[CostModel] = None,
    exit_model: Optional[str] = None,
    granularity_seconds: float = 60.0,
    fv: Optional[FeatureVector] = None,
    price_path: Optional[PricePath] = None,
    next_bar: Optional[MarketSnapshot] = None,
    ml_exit_model: Optional[ExitModel] = None,
    exit_mode: str = "hard_limits",
) -> Optional[dict]:
    """One bar-step of triple-barrier / ML exit checking for a live open position."""
    position.bars_held += 1

    if position.entry_price > 0:
        fav = (
            (bar.high - position.entry_price) / position.entry_price * 100.0
            if position.direction == 1
            else (position.entry_price - bar.low) / position.entry_price * 100.0
        )
        adv = (
            (position.entry_price - bar.low) / position.entry_price * 100.0
            if position.direction == 1
            else (bar.high - position.entry_price) / position.entry_price * 100.0
        )
        position.mfe_so_far_pct = max(position.mfe_so_far_pct, fav)
        position.mae_so_far_pct = max(position.mae_so_far_pct, adv)

    held_seconds = (bar.timestamp - position.entry_timestamp) - position.entry_latency_seconds
    if position.min_hold_seconds > 0 and held_seconds < position.min_hold_seconds - 1e-3:
        if position.trail_state is None and position.exit_plan is not None and position.exit_plan.trailing is not None:
            position.trail_state = TrailState(best_favorable_price=position.entry_price)
        if position.trail_state is not None and position.exit_plan is not None:
            from ..labeling.triple_barrier import advance_trail
            advance_trail(position.direction, position.entry_price, bar, position.exit_plan.trailing, position.trail_state)
        return None

    eff_exit_model = exit_model if exit_model is not None else position.exit_model
    exit_delay_s = (
        cost_model.exit_delay_seconds
        if (cost_model is not None and cost_model.exit_delay_seconds > 0)
        else position.exit_delay_seconds
    )
    round_trip_fees = cost_model.round_trip_fee_pct() if cost_model is not None else (fee_pct * 2.0)

    if position.pending_exit is not None:
        pending = position.pending_exit
        elapsed_s = float(bar.timestamp - pending.trigger_timestamp)
        bars_since = position.bars_held - pending.trigger_bars_held
        if elapsed_s < exit_delay_s - 1e-6 and bars_since * granularity_seconds < exit_delay_s - 1e-6:
            return None
        position.pending_exit = None
        fill_px = exit_fill_price(
            direction=position.direction,
            trigger_price=bar.open,
            bar=bar,
            exit_reason=pending.exit_reason,
            cost_model=cost_model,
            exit_model=eff_exit_model,
            next_bar=next_bar,
            granularity_seconds=granularity_seconds,
            fv=fv,
            size_usdt=position.size_usdt,
            price_path=price_path,
        )
        carry_cost = cost_model.carry_pct(position.direction, max(0.0, held_seconds)) if cost_model is not None else 0.0
        realized_pnl_pct = raw_return_pct(position.direction, position.entry_price, fill_px) - round_trip_fees - carry_cost
        realized_pnl_usdt = position.size_usdt * (realized_pnl_pct / 100.0)
        return {
            "exit_price": fill_px,
            "exit_trigger_price": pending.trigger_price,
            "exit_reason": pending.exit_reason,
            "realized_pnl_pct": realized_pnl_pct,
            "realized_pnl_usdt": realized_pnl_usdt,
            "bars_held": position.bars_held,
            "hold_seconds": held_seconds,
        }

    if position.trail_state is None and position.exit_plan is not None and position.exit_plan.trailing is not None:
        position.trail_state = TrailState(best_favorable_price=position.entry_price)

    if ml_exit_model is not None and exit_mode in ("ml_exit", "hybrid"):
        step = evaluate_exit_ladder(
            direction=position.direction,
            entry_price=position.entry_price,
            take_profit_pct=position.take_profit_pct,
            stop_loss_pct=position.stop_loss_pct,
            bar=bar,
            bars_held=position.bars_held,
            max_holding_bars=position.max_holding_bars,
            min_hold_bars=position.min_hold_bars,
            exit_plan=position.exit_plan,
            trail_state=position.trail_state,
            exit_model=ml_exit_model,
            exit_mode=exit_mode,
            mfe_so_far_pct=position.mfe_so_far_pct,
            mae_so_far_pct=position.mae_so_far_pct,
            fv=fv,
        )
    else:
        step = barrier_exit_step(
            position.direction,
            position.entry_price,
            position.take_profit_pct,
            position.stop_loss_pct,
            bar,
            position.bars_held,
            position.max_holding_bars,
            position.min_hold_bars,
            exit_plan=position.exit_plan,
            trail_state=position.trail_state,
        )
    if step is None:
        return None
    exit_reason, trigger_price = step

    if exit_delay_s >= granularity_seconds > 0:
        position.pending_exit = PendingExit(
            trigger_timestamp=bar.timestamp,
            trigger_bars_held=position.bars_held,
            exit_reason=exit_reason,
            trigger_price=trigger_price,
            fill_due_timestamp=float(bar.timestamp) + exit_delay_s,
        )
        return None

    fill_px = exit_fill_price(
        direction=position.direction,
        trigger_price=trigger_price,
        bar=bar,
        exit_reason=exit_reason,
        cost_model=cost_model,
        exit_model=eff_exit_model,
        next_bar=next_bar,
        granularity_seconds=granularity_seconds,
        fv=fv,
        size_usdt=position.size_usdt,
        price_path=price_path,
    )
    carry_cost = (
        cost_model.carry_pct(position.direction, max(0.0, held_seconds + exit_delay_s))
        if cost_model is not None
        else 0.0
    )
    realized_pnl_pct = raw_return_pct(position.direction, position.entry_price, fill_px) - round_trip_fees - carry_cost
    realized_pnl_usdt = position.size_usdt * (realized_pnl_pct / 100.0)

    return {
        "exit_price": fill_px,
        "exit_trigger_price": trigger_price,
        "exit_reason": exit_reason,
        "realized_pnl_pct": realized_pnl_pct,
        "realized_pnl_usdt": realized_pnl_usdt,
        "bars_held": position.bars_held,
        "hold_seconds": held_seconds + exit_delay_s,
    }


@dataclass(frozen=True)
class ClosedTrade:
    """A position that closed on this bar, with the exit that closed it."""

    position: OpenPosition
    exit_timestamp: int
    exit_info: dict
    close_record: Optional[ExecutionRecord] = None


@dataclass
class BarOutcome:
    """Everything one bar produced, for the caller to persist or draw."""

    state: SystemState
    position: Optional[OpenPosition]
    equity: float
    position_is_open: bool
    execution_failed: bool
    execution_error: str
    stage_timings_ms: Dict[str, float]
    trace: DecisionTrace
    opened: Optional[OpenPosition] = None
    closed: Optional[ClosedTrade] = None
    stages: List[StageResult] = field(default_factory=list)
    open_positions: List[OpenPosition] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)

    @property
    def terminal_stage(self) -> str:
        return self.trace.terminal_stage

    @property
    def terminal_reason(self) -> str:
        return self.trace.terminal_reason


def max_concurrent_positions(breaker_cfg: Mapping) -> int:
    """``risk.circuit_breakers.max_concurrent_positions``, validated."""
    value = int(breaker_cfg.get("max_concurrent_positions", 1))
    if value < 1:
        raise ValueError(
            f"max_concurrent_positions={value} is invalid. Must be >= 1."
        )
    return value


def compute_position_barriers(config: Mapping, regime_state: Any, feature_vector: Any) -> tuple:
    """Take-profit / stop-loss percentages for a new position."""
    return barriers_for_regime(
        config, regime_state.primary_regime.name, barrier_volatility_pct(feature_vector),
    )


def process_bar(
    state: SystemState,
    *,
    risk_layer: RiskLayer,
    execution_engine: ExecutionEngine,
    position: Optional[OpenPosition],
    equity: float,
    fee_pct: float,
    config: Mapping,
    decision_latency_ms: Optional[float] = None,
    pipeline_timings: Optional[Mapping[str, float]] = None,
    on_trade_open: Optional[Callable[[OpenPosition], Optional[str]]] = None,
    on_trade_close: Optional[Callable[[OpenPosition, int, dict], None]] = None,
    position_book: Optional[PositionBook] = None,
    cooldown_manager: Optional[CooldownManager] = None,
    halt_manager: Optional[HaltManager] = None,
    risk_governor: Optional[RiskGovernor] = None,
    barrier_model: Optional[BarrierSizingModel] = None,
    exit_models: Optional[Mapping[str, ExitModel]] = None,
    model_ledger: Optional[ModelLedger] = None,
    agent_sink: Optional[AgentSink] = None,
    bar_index: int = 0,
) -> BarOutcome:
    """Run one bar through exit -> risk -> order -> open -> classify -> trace."""
    risk_layer.store.advance_clock(float(state.timestamp))
    risk_layer.store.heartbeat()
    hold_policy = HoldPolicy.from_config(config)
    cost_model = CostModel.from_config(config)
    exec_cfg = config.get("execution", {}) or {}
    exit_model = str(exec_cfg.get("exit_model", "barrier_level"))
    exits_cfg = config.get("exits", {}) or {}
    exit_mode = str(exits_cfg.get("mode", "hard_limits"))

    eff_agent_sink = agent_sink
    if eff_agent_sink is None:
        handoff_cfg = config.get("handoff", {}) or {}
        if str(handoff_cfg.get("mode", "off")) != "off":
            eff_agent_sink = build_agent_sink(config)

    book = position_book
    if book is None:
        book = PositionBook.from_config(
            config, initial_positions=[position] if position is not None else []
        )

    risk_decision = None
    execution_record = None
    execution_failed = False
    execution_error = ""
    stage_timings_ms: Dict[str, float] = {}
    realized_pnl_this_bar = 0.0
    opened: Optional[OpenPosition] = None
    closed: Optional[ClosedTrade] = None
    closed_trades: List[ClosedTrade] = []

    # 1. Manage all open positions first in deterministic order
    for pos in list(book.positions):
        pos_ml_exit = (
            exit_models.get(pos.expert or "")
            if (exit_models is not None and pos.expert is not None)
            else None
        )
        if cost_model.is_non_default() or exit_model != "barrier_level" or pos_ml_exit is not None:
            exit_info = check_position_exit(
                pos,
                state.market_snapshot,
                fee_pct,
                cost_model=cost_model if cost_model.is_non_default() else None,
                exit_model=exit_model,
                granularity_seconds=hold_policy.granularity_seconds,
                fv=state.feature_vector,
                ml_exit_model=pos_ml_exit,
                exit_mode=exit_mode,
            )
        else:
            exit_info = check_position_exit(pos, state.market_snapshot, fee_pct)
        if exit_info is not None:
            close_record: Optional[ExecutionRecord] = None
            close_failed = False
            try:
                close_record = execution_engine.close_position(
                    pos,
                    state.market_snapshot,
                    exit_info["exit_reason"],
                    exit_info["exit_price"],
                    fv=state.feature_vector,
                )
                risk_layer.store.record_execution_result(success=True)
            except Exception as exc:
                logger.error("Close order failed: %s", exc)
                risk_layer.store.record_execution_result(success=False)
                if risk_layer.store.consecutive_execution_errors() >= \
                        risk_layer.breaker_cfg.get("consecutive_execution_errors_halt", 3):
                    risk_layer.store.trip(
                        "consecutive_execution_errors", "3 consecutive execution failures",
                    )
                close_failed = True
                execution_failed = True
                execution_error = str(exc)

            if not close_failed:
                if (
                    close_record is not None
                    and execution_engine.mode in ("live", "paper_live")
                    and close_record.fill_price > 0
                ):
                    exit_info["exit_price"] = close_record.fill_price
                    round_trip_fees = (
                        cost_model.round_trip_fee_pct()
                        if cost_model.is_non_default()
                        else (fee_pct * 2.0)
                    )
                    exit_info["realized_pnl_pct"] = (
                        raw_return_pct(pos.direction, pos.entry_price, close_record.fill_price)
                        - round_trip_fees
                    )
                    exit_info["realized_pnl_usdt"] = pos.size_usdt * (exit_info["realized_pnl_pct"] / 100.0)

                if on_trade_close is not None:
                    on_trade_close(pos, state.timestamp, exit_info)
                if eff_agent_sink is not None and pos.signal_id:
                    eff_agent_sink.notify_close(
                        signal_id=pos.signal_id,
                        trade_id=pos.trade_id or pos.signal_id,
                        symbol=state.symbol,
                        exit_reason=str(exit_info["exit_reason"]),
                        exit_price=float(exit_info["exit_price"]),
                        realized_pnl_pct=float(exit_info["realized_pnl_pct"]),
                        timestamp=int(state.timestamp),
                    )
                if model_ledger is not None:
                    pred_id = f"router:{pos.entry_timestamp}:{pos.expert or 'unknown'}"
                    model_ledger.record_outcome(
                        ModelOutcomeRecord(
                            prediction_id=pred_id,
                            resolved_timestamp=int(state.timestamp),
                            realized_label=1 if float(exit_info["realized_pnl_pct"]) > 0 else 0,
                            realized_value=float(exit_info["realized_pnl_pct"]),
                            trade_id=pos.trade_id or None,
                        )
                    )
                risk_layer.store.record_trade_result(
                    exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
                )
                if cooldown_manager is not None and pos.expert:
                    cooldown_manager.record_outcome(
                        pos.expert,
                        exit_info["realized_pnl_pct"],
                        exit_info["exit_reason"],
                        float(state.timestamp),
                        bar_index=bar_index,
                    )
                if halt_manager is not None:
                    halt_manager.record_trade(exit_info["realized_pnl_pct"], timestamp=float(state.timestamp))
                equity += exit_info["realized_pnl_usdt"]
                drawdown = risk_layer.store.update_equity(equity)
                if drawdown >= risk_layer.breaker_cfg.get("max_drawdown_pct", 8.0):
                    risk_layer.store.trip("max_drawdown", f"drawdown={drawdown:.2f}%")
                    logger.warning(
                        "max_drawdown breaker tripped at %.2f%% -- halting new entries", drawdown
                    )
                realized_pnl_this_bar += exit_info["realized_pnl_pct"]
                logger.info(
                    "[%s] EXIT %s pnl=%.3f%% equity=%.2f",
                    state.timestamp, exit_info["exit_reason"], exit_info["realized_pnl_pct"], equity,
                )
                ct = ClosedTrade(pos, state.timestamp, exit_info, close_record=close_record)
                closed_trades.append(ct)
                if closed is None:
                    closed = ct
                book.remove(pos)

    # 2. Check if we can evaluate a new entry
    max_concurrent = max(book.limits.max_concurrent, max_concurrent_positions(risk_layer.breaker_cfg))
    position_is_open = max_concurrent == 1 and not book.is_empty()

    if not position_is_open and state.router_decision is not None:
        router_decision = state.router_decision
        expert_outputs = state.expert_outputs
        regime_state = state.regime_state
        feature_vector = state.feature_vector
        if expert_outputs is None or regime_state is None or feature_vector is None:
            raise RuntimeError(
                f"pipeline returned an incomplete SystemState (timestamp={state.timestamp})"
            )

        # Evaluate RiskGovernor if enabled
        gov_directive = None
        if risk_governor is not None and risk_governor.enabled:
            peak_eq = risk_layer.store.peak_equity()
            curr_dd = ((peak_eq - equity) / peak_eq * 100.0) if peak_eq > 0 else 0.0
            consec_losses = risk_layer.store.consecutive_losses()
            gov_directive = risk_governor.evaluate(
                drawdown_pct=max(0.0, curr_dd),
                consecutive_losses=consec_losses,
                configured_max_concurrent=max_concurrent,
            )
            if gov_directive.trigger_halt_minutes > 0 and halt_manager is not None:
                halt_manager.trigger_halt(
                    gov_directive.reason or "governor_halt",
                    float(state.timestamp),
                    duration_minutes=gov_directive.trigger_halt_minutes,
                )

        # Check time-boxed portfolio halt or per-expert cooldown before or inside risk evaluation
        pre_reject_reason: Optional[str] = None
        if halt_manager is not None:
            halted, h_reason = halt_manager.is_halted(float(state.timestamp))
            if halted:
                pre_reject_reason = h_reason
        if pre_reject_reason is None and gov_directive is not None and gov_directive.veto:
            pre_reject_reason = gov_directive.reason or "governor_veto"
        if pre_reject_reason is None and cooldown_manager is not None and router_decision.selected_expert != "no_trade":
            cooled, c_reason = cooldown_manager.is_cooled_down(
                router_decision.selected_expert, float(state.timestamp), current_bar_index=bar_index,
            )
            if cooled:
                pre_reject_reason = c_reason

        # Pre-compute barrier model exit plan when barrier_model is active so risk sizing uses unified stop (F3)
        expert_output_cand = expert_outputs.get_by_name(router_decision.selected_expert)
        cand_dir = 1 if (expert_output_cand and expert_output_cand.action.name == "BUY") else -1
        pre_barrier_plan: Optional[ExitPlan] = None
        if barrier_model is not None and barrier_model.is_ready and router_decision.selected_expert != "no_trade":
            pre_barrier_plan = barrier_model.predict_exit_plan(
                feature_vector,
                state.zone_map,
                regime_state,
                cand_dir,
                market=state.market_snapshot,
            )

        t_risk = time.perf_counter()
        if pre_reject_reason is not None:
            risk_decision = RiskDecision(
                timestamp=state.timestamp,
                symbol=state.symbol,
                approved=False,
                risk_score=1.0,
                base_size=0.0,
                adjusted_size=0.0,
                size_reduction_pct=100.0,
                rejection_reason=pre_reject_reason,
            )
        else:
            if pre_barrier_plan is not None:
                risk_decision = risk_layer.evaluate(
                    router_decision, expert_outputs, regime_state,
                    state.zone_map, feature_vector, state.market_snapshot, equity,
                    planned_stop_loss_pct=pre_barrier_plan.stop_loss_pct,
                )
            else:
                risk_decision = risk_layer.evaluate(
                    router_decision, expert_outputs, regime_state,
                    state.zone_map, feature_vector, state.market_snapshot, equity,
                )
            if risk_decision.approved and gov_directive is not None and gov_directive.size_multiplier < 1.0:
                new_adj = round(risk_decision.adjusted_size * gov_directive.size_multiplier, 4)
                red_pct = (
                    (1.0 - (new_adj / risk_decision.base_size)) * 100.0
                    if risk_decision.base_size > 0
                    else risk_decision.size_reduction_pct
                )
                risk_decision = RiskDecision(
                    timestamp=risk_decision.timestamp,
                    symbol=risk_decision.symbol,
                    approved=new_adj > 0.0,
                    risk_score=risk_decision.risk_score,
                    base_size=risk_decision.base_size,
                    adjusted_size=new_adj,
                    size_reduction_pct=round(red_pct, 2),
                    rejection_reason=None if new_adj > 0.0 else (gov_directive.reason or "governor_zero_size"),
                )
            if (
                risk_decision.approved
                and gov_directive is not None
                and len(book.positions) >= gov_directive.max_concurrent_cap
            ):
                risk_decision = RiskDecision(
                    timestamp=state.timestamp,
                    symbol=state.symbol,
                    approved=False,
                    risk_score=risk_decision.risk_score,
                    base_size=risk_decision.base_size,
                    adjusted_size=0.0,
                    size_reduction_pct=100.0,
                    rejection_reason=f"governor_max_concurrent_cap:{gov_directive.max_concurrent_cap}",
                )
            # If approved and multi-position / PositionBook limits are active, check book.can_open
            if risk_decision.approved and (max_concurrent > 1 or "positions" in config or not book.is_empty()):
                allowed, _, book_reason = book.can_open(
                    expert=router_decision.selected_expert,
                    regime=regime_state.primary_regime.value,
                    direction=cand_dir,
                    size_usdt=risk_decision.adjusted_size,
                    equity=equity,
                )
                if not allowed:
                    risk_decision = RiskDecision(
                        timestamp=state.timestamp,
                        symbol=state.symbol,
                        approved=False,
                        risk_score=risk_decision.risk_score,
                        base_size=risk_decision.base_size,
                        adjusted_size=0.0,
                        size_reduction_pct=100.0,
                        rejection_reason=book_reason,
                    )

        stage_timings_ms["risk_layer_ms"] = (time.perf_counter() - t_risk) * 1000
        state = dc_replace(state, risk_decision=risk_decision)

        if risk_decision.approved:
            expert_output = expert_outputs.get_by_name(router_decision.selected_expert)
            resolved_plan = resolve_exit_plan(
                config,
                expert_output=expert_output,
                barrier_model_plan=pre_barrier_plan,
                regime_name=regime_state.primary_regime.name,
                volatility_pct=barrier_volatility_pct(feature_vector),
            )

            # External Agent Handoff (`D9` / Phase 14)
            active_signal_id = ""
            delegated_to_external_agent = False
            external_order_id: Optional[str] = None
            if eff_agent_sink is not None and eff_agent_sink.mode != "off":
                sig = TradeSignal.build(
                    timestamp=state.timestamp,
                    symbol=state.symbol,
                    direction=cand_dir,
                    size_usdt=risk_decision.adjusted_size,
                    entry_price=state.market_snapshot.close,
                    expert_instance_id=router_decision.selected_expert,
                    regime_id=regime_state.primary_regime.value,
                    regime_set_id=regime_state.regime_set_id,
                    exit_plan=resolved_plan,
                    router_confidence=router_decision.selection_confidence,
                    risk_score=risk_decision.risk_score,
                    secret_key=eff_agent_sink.secret_key,
                )
                active_signal_id = sig.signal_id
                agent_resp = eff_agent_sink.submit(sig)
                if eff_agent_sink.mode in ("advisory", "execute"):
                    if not agent_resp.approved or agent_resp.size_multiplier <= 0.0:
                        prefix = "agent_veto" if eff_agent_sink.mode == "advisory" else "agent_reject"
                        risk_decision = RiskDecision(
                            timestamp=state.timestamp,
                            symbol=state.symbol,
                            approved=False,
                            risk_score=risk_decision.risk_score,
                            base_size=risk_decision.base_size,
                            adjusted_size=0.0,
                            size_reduction_pct=100.0,
                            rejection_reason=f"{prefix}:{agent_resp.reason}",
                        )
                        state = dc_replace(state, risk_decision=risk_decision)
                    elif agent_resp.size_multiplier < 1.0:
                        tightened = round(risk_decision.adjusted_size * agent_resp.size_multiplier, 4)
                        red_pct = (
                            (1.0 - (tightened / risk_decision.base_size)) * 100.0
                            if risk_decision.base_size > 0
                            else risk_decision.size_reduction_pct
                        )
                        risk_decision = RiskDecision(
                            timestamp=risk_decision.timestamp,
                            symbol=risk_decision.symbol,
                            approved=True,
                            risk_score=risk_decision.risk_score,
                            base_size=risk_decision.base_size,
                            adjusted_size=tightened,
                            size_reduction_pct=round(red_pct, 2),
                            rejection_reason=None,
                        )
                        state = dc_replace(state, risk_decision=risk_decision)
                    if (
                        risk_decision.approved
                        and eff_agent_sink.mode == "execute"
                        and agent_resp.executed_externally
                        and not agent_resp.used_internal_fallback
                    ):
                        delegated_to_external_agent = True
                        external_order_id = agent_resp.external_order_id or f"ext-{sig.signal_id[:10]}"

        if risk_decision.approved:
            expert_output = expert_outputs.get_by_name(router_decision.selected_expert)
            t_exec = time.perf_counter()
            if delegated_to_external_agent:
                side_str = "BUY" if cand_dir == 1 else "SELL"
                execution_record = ExecutionRecord(
                    timestamp=state.timestamp,
                    order_timestamp=state.timestamp,
                    fill_timestamp=state.timestamp,
                    symbol=state.symbol,
                    side=side_str,
                    order_price=state.market_snapshot.close,
                    fill_price=state.market_snapshot.close,
                    requested_size=risk_decision.adjusted_size,
                    filled_size=risk_decision.adjusted_size,
                    slippage_pct=0.0,
                    fees_pct=fee_pct,
                    total_cost_pct=fee_pct,
                    status="filled",
                    order_id=external_order_id or f"ext-{state.timestamp}",
                )
                risk_layer.store.record_execution_result(success=True)
            else:
                try:
                    execution_record = execution_engine.place_order(
                        risk_decision, router_decision, expert_output,
                        state.market_snapshot, fv=feature_vector,
                    )
                    risk_layer.store.record_execution_result(success=True)
                except Exception as exc:
                    logger.error("Execution failed: %s", exc)
                    risk_layer.store.record_execution_result(success=False)
                    if risk_layer.store.consecutive_execution_errors() >= \
                            risk_layer.breaker_cfg.get("consecutive_execution_errors_halt", 3):
                        risk_layer.store.trip(
                            "consecutive_execution_errors", "3 consecutive execution failures",
                        )
                    execution_record = None
                    execution_failed = True
                    execution_error = str(exc)
            stage_timings_ms["execution_ms"] = (time.perf_counter() - t_exec) * 1000

            if execution_record is not None and execution_record.status in ("filled", "partial"):
                direction = 1 if execution_record.side == "BUY" else -1
                resolved_plan = resolve_exit_plan(
                    config,
                    expert_output=expert_output,
                    barrier_model_plan=pre_barrier_plan,
                    regime_name=regime_state.primary_regime.name,
                    volatility_pct=barrier_volatility_pct(feature_vector),
                )
                tp_pct = resolved_plan.take_profit_pct
                sl_pct = resolved_plan.stop_loss_pct
                risk_layer.record_trade(state.timestamp)

                measured_latency = max(0, int(execution_record.fill_timestamp) - int(execution_record.order_timestamp))
                entry_policy = hold_policy.with_latency(max(measured_latency, hold_policy.entry_latency_seconds))

                new_position = OpenPosition(
                    trade_id="",
                    entry_timestamp=state.timestamp,
                    entry_price=execution_record.fill_price,
                    direction=direction,
                    size_usdt=execution_record.filled_size,
                    take_profit_pct=tp_pct,
                    stop_loss_pct=sl_pct,
                    max_holding_bars=resolved_plan.max_holding_bars,
                    fees_paid=execution_record.filled_size * (execution_record.fees_pct / 100),
                    expert=router_decision.selected_expert,
                    regime=regime_state.primary_regime.value,
                    selection_confidence=router_decision.selection_confidence,
                    risk_score=risk_decision.risk_score,
                    min_hold_bars=entry_policy.min_hold_bars,
                    min_hold_seconds=entry_policy.min_trade_seconds,
                    entry_latency_seconds=entry_policy.entry_latency_seconds,
                    exit_delay_seconds=entry_policy.exit_delay_seconds,
                    exit_model=exit_model,
                    exit_plan=resolved_plan
                    if (config.get("exits") or pre_barrier_plan is not None or (expert_output and expert_output.exit_plan))
                    else None,
                    trail_state=TrailState(best_favorable_price=execution_record.fill_price)
                    if resolved_plan.trailing is not None and resolved_plan.trailing.enabled
                    else None,
                    signal_id=active_signal_id,
                )
                if on_trade_open is not None:
                    trade_id = on_trade_open(new_position)
                    if trade_id:
                        new_position.trade_id = trade_id
                if model_ledger is not None:
                    pred_id = f"router:{state.timestamp}:{router_decision.selected_expert}"
                    model_ledger.record_prediction(
                        ModelPredictionRecord(
                            prediction_id=pred_id,
                            timestamp=int(state.timestamp),
                            model_name=f"router:{router_decision.selected_expert}",
                            model_version_hash="runtime_v1",
                            predicted_prob=float(router_decision.selection_confidence),
                            predicted_value=float(risk_decision.adjusted_size),
                            regime=regime_state.primary_regime.value,
                            trade_id=new_position.trade_id or None,
                        )
                    )
                book.add(new_position)
                opened = new_position
                logger.info(
                    "[%s] ENTRY %s via %s size=%.2f price=%.2f",
                    state.timestamp, execution_record.side, router_decision.selected_expert,
                    execution_record.filled_size, execution_record.fill_price,
                )

            state = dc_replace(state, execution_record=execution_record)

    stage_timings_ms.update(pipeline_timings or {})
    stages = build_trace_stages(state, stage_timings_ms)
    terminal_stage, terminal_reason = classify_terminal_stage(
        state.opportunity_score, state.router_decision,
        state.risk_decision, state.execution_record,
        position_open=position_is_open,
        execution_failed=execution_failed,
        execution_error=execution_error or None,
    )
    trace = DecisionTrace(
        timestamp=state.timestamp, symbol=state.symbol,
        stages=stages, terminal_stage=terminal_stage, terminal_reason=terminal_reason,
        total_duration_ms=decision_latency_ms,
        stage_timings_ms=dict(stage_timings_ms),
    )

    state = dc_replace(state, realized_pnl=realized_pnl_this_bar)
    risk_layer.advance_bar()

    primary_pos = book.primary_position()
    return BarOutcome(
        state=state,
        position=primary_pos,
        equity=equity,
        position_is_open=position_is_open,
        execution_failed=execution_failed,
        execution_error=execution_error,
        stage_timings_ms=stage_timings_ms,
        trace=trace,
        opened=opened,
        closed=closed,
        stages=stages,
        open_positions=book.positions,
        closed_trades=closed_trades,
    )
