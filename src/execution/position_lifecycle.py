"""
Shared position-lifecycle primitives: the open-position record, the
incremental triple-barrier exit check, and the execution-engine factory.

These three previously lived in ``scripts/run.py``. They live in ``src/``
now for two reasons:

1. ``scripts/full_stack_walk_forward_backtest.py`` already imported them
   from ``run`` (``from run import OpenPosition, build_execution_engine,
   check_position_exit``), so the scripts layer already treated them as
   shared infrastructure rather than as run-loop-private helpers.
2. The Trade Map GUI (``src/gui/replay.py``) replays positions and must
   use the SAME take-profit / stop-loss / timeout semantics as the
   deployed loop. Re-implementing the exit rule inside the GUI would let
   the replayed position lifecycle silently drift from the production
   one -- the exact class of divergence this repo's docs keep warning
   about.

``process_bar`` is the ONE implementation of the per-bar trading block:
exit -> risk -> order -> open-position -> classify -> trace. The
historical loop, the live loop and the GUI replay all call it, so "the
runtime and the GUI agree bar for bar" holds BY CONSTRUCTION rather than
by a test comparing three copies. What differs between the callers --
where a trade is journalled, what the trade id is -- is passed in as two
callbacks; nothing about the decision itself is.

``scripts/run.py`` re-exports ``OpenPosition``, ``build_execution_engine``
and ``check_position_exit``, so ``from run import OpenPosition`` keeps
working.
"""
import logging
import time
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Callable, Dict, List, Mapping, Optional

from ..connectors.data_source import CCXTConnector
from ..diagnostics.decision_trace import (
    DecisionTrace,
    StageResult,
    build_trace_stages,
    classify_terminal_stage,
)
from ..labeling.hold_policy import HoldPolicy
from ..labeling.triple_barrier import (
    barrier_exit_step,
    barrier_volatility_pct,
    barriers_for_regime,
    raw_return_pct,
)
from ..layers.risk_layer import RiskLayer
from ..state.contracts import MarketSnapshot, SystemState
from .execution_engine import ExecutionBackend, ExecutionEngine

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
    """A position that is currently open, with its triple-barrier levels.

    The provenance fields (``expert``, ``regime``, ``selection_confidence``,
    ``risk_score``) are DECLARED rather than bolted on with
    ``object.__setattr__``: an undeclared attribute is invisible to
    ``dataclasses.asdict()`` and to a type checker, so a consumer that
    reads it (the Trade Map GUI) silently got ``None`` for a trade the
    deployment loop had actually recorded.
    """
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
    # Minimum-hold lockout (see src/labeling/hold_policy.py). The defaults
    # mean "no lockout", so a position built without a policy (older
    # callers, unit tests) behaves exactly as before.
    min_hold_bars: int = 1
    min_hold_seconds: float = 0.0
    entry_latency_seconds: float = 0.0


def build_execution_engine(
    config: dict,
    connector: Optional[CCXTConnector],
    symbol: str,
) -> ExecutionEngine:
    """Build the execution engine for the configured ``execution.mode``.

    ``connector`` is only used by the paper_live / live backends; the
    paper_sim backend needs no exchange connection.

    The backends are imported inside the function: a viewer (the Trade
    Map GUI) must not drag the live exchange adapter into its import
    graph just to read a config key.
    """
    exec_cfg = config.get("execution", {})
    mode = exec_cfg.get("mode", "paper_sim")
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)

    backend: ExecutionBackend
    if mode == "paper_sim":
        from .backends.paper_sim_backend import PaperSimBackend

        backend = PaperSimBackend(exec_cfg, fee_pct=fee_pct)
    elif mode == "paper_live":
        from .backends.paper_live_backend import PaperLiveBackend

        backend = PaperLiveBackend(exec_cfg, connector, symbol, fee_pct=fee_pct)
    elif mode == "live":
        from .backends.live_backend import LiveBackend

        backend = LiveBackend(exec_cfg, connector, symbol)
    else:
        raise ValueError(f"Unknown execution.mode: {mode}")

    return ExecutionEngine(backend, mode)


def check_position_exit(
    position: OpenPosition, bar: MarketSnapshot, fee_pct: float
) -> Optional[dict]:
    """One bar-step of triple-barrier-style exit checking for a live open
    position, applied incrementally (each bar arrives one at a time).

    The exit RULE is :func:`src.labeling.triple_barrier.barrier_exit_step`
    -- the very function the training labeler calls -- so a label and a
    live trade cannot disagree about when or where a position exits.

    Minimum hold: no exit of any kind until the position has been held
    ``min_hold_bars`` bars AND ``min_hold_seconds`` of wall-clock time
    (fill to this bar's timestamp, net of the entry latency). Both are
    checked; the time check also protects against a duplicated or
    out-of-order bar. With the defaults (no policy) neither applies.
    """
    position.bars_held += 1

    held_seconds = (bar.timestamp - position.entry_timestamp) - position.entry_latency_seconds
    if position.min_hold_seconds > 0 and held_seconds < position.min_hold_seconds - 1e-3:
        return None

    step = barrier_exit_step(
        position.direction, position.entry_price, position.take_profit_pct,
        position.stop_loss_pct, bar, position.bars_held, position.max_holding_bars,
        position.min_hold_bars,
    )
    if step is None:
        return None
    exit_reason, exit_price = step

    realized_pnl_pct = raw_return_pct(position.direction, position.entry_price, exit_price) - (fee_pct * 2)
    realized_pnl_usdt = position.size_usdt * (realized_pnl_pct / 100)

    return {
        "exit_price": exit_price, "exit_reason": exit_reason,
        "realized_pnl_pct": realized_pnl_pct, "realized_pnl_usdt": realized_pnl_usdt,
        "bars_held": position.bars_held, "hold_seconds": held_seconds,
    }


# ---------------------------------------------------------------------------
# The per-bar trading block -- the single implementation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClosedTrade:
    """A position that closed on this bar, with the exit that closed it."""
    position: OpenPosition
    exit_timestamp: int
    exit_info: dict


@dataclass
class BarOutcome:
    """Everything one bar produced, for the caller to persist or draw.

    Attributes:
        state: the bar's ``SystemState`` with this bar's risk decision,
            execution record and realised PnL attached.
        position: the open position AFTER this bar (None when flat).
        equity: account equity after this bar's exit, if any.
        position_is_open: a position was already open when the bar
            looked for an entry (classifies as ``position_open``).
        execution_failed / execution_error: the order raised.
        stage_timings_ms: per-stage wall-clock latencies.
        trace: the bar's ``DecisionTrace`` (stages + terminal stage).
        opened: the position opened on this bar, if one was.
        closed: the trade that closed on this bar, if one did.
    """
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

    @property
    def terminal_stage(self) -> str:
        return self.trace.terminal_stage

    @property
    def terminal_reason(self) -> str:
        return self.trace.terminal_reason


def max_concurrent_positions(breaker_cfg: Mapping) -> int:
    """``risk.circuit_breakers.max_concurrent_positions``, validated.

    Raises ``ValueError`` when the configured value is below 1 (real
    multi-position support is a future item; values above 1 are accepted
    but still open at most one position, because ``process_bar`` only
    looks for an entry while nothing is open).
    """
    value = int(breaker_cfg.get("max_concurrent_positions", 1))
    if value < 1:
        raise ValueError(
            f"max_concurrent_positions={value} is invalid. "
            f"Must be >= 1. (Real multi-position support is a future item.)"
        )
    return value


def compute_position_barriers(config: Mapping, regime_state, feature_vector) -> tuple:
    """Take-profit / stop-loss percentages for a new position.

    A thin wrapper over :func:`src.labeling.triple_barrier.barriers_for_regime`
    -- the one barrier-sizing implementation the training scripts share.
    """
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
) -> BarOutcome:
    """Run one bar through exit -> risk -> order -> open -> classify -> trace.

    ``state`` is the pipeline's output for this bar. The pipeline returns
    a fresh ``SystemState`` every bar, so nothing here can leak the
    previous bar's risk or execution result.

    Callbacks (both optional) are the only things that differ between the
    deployment loop and the GUI replay:

    * ``on_trade_open(position)`` is called once a position has been
      built from a fill; it may return the trade id to use (the journal's
      id in the deployment loop, a synthetic id in the replay).
    * ``on_trade_close(position, exit_timestamp, exit_info)`` is called
      when a position exits (journal write / trade list append).
    """
    risk_layer.store.heartbeat()
    hold_policy = HoldPolicy.from_config(config)    # raises HoldPolicyError if unsatisfiable

    risk_decision = None
    execution_record = None
    execution_failed = False
    execution_error = ""
    stage_timings_ms: Dict[str, float] = {}
    realized_pnl_this_bar = 0.0
    opened: Optional[OpenPosition] = None
    closed: Optional[ClosedTrade] = None

    # 1. Manage any open position first -- exits are never gated by the
    # Risk Layer (see risk_layer.py's module docstring).
    if position is not None:
        exit_info = check_position_exit(position, state.market_snapshot, fee_pct)
        if exit_info is not None:
            if on_trade_close is not None:
                on_trade_close(position, state.timestamp, exit_info)
            risk_layer.store.record_trade_result(
                exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
            )
            equity += exit_info["realized_pnl_usdt"]
            drawdown = risk_layer.store.update_equity(equity)
            if drawdown >= risk_layer.breaker_cfg.get("max_drawdown_pct", 8.0):
                risk_layer.store.trip("max_drawdown", f"drawdown={drawdown:.2f}%")
                logger.warning(
                    "max_drawdown breaker tripped at %.2f%% -- halting new entries", drawdown
                )
            realized_pnl_this_bar = exit_info["realized_pnl_pct"]
            logger.info(
                "[%s] EXIT %s pnl=%.3f%% equity=%.2f",
                state.timestamp, exit_info["exit_reason"], exit_info["realized_pnl_pct"], equity,
            )
            closed = ClosedTrade(position, state.timestamp, exit_info)
            position = None

    # 2. Only look for a NEW entry if nothing is open. When a position is
    # already open the bar is still processed (traced, archived) -- it
    # simply cannot open a second one.
    max_concurrent = max_concurrent_positions(risk_layer.breaker_cfg)
    position_is_open = max_concurrent == 1 and position is not None

    if not position_is_open and position is None and state.router_decision is not None:
        router_decision = state.router_decision
        expert_outputs = state.expert_outputs
        regime_state = state.regime_state
        feature_vector = state.feature_vector
        # The pipeline always populates these on a bar it processed; a
        # None here means its contract changed under us.
        if expert_outputs is None or regime_state is None or feature_vector is None:
            raise RuntimeError(
                f"pipeline returned an incomplete SystemState (timestamp={state.timestamp})"
            )

        t_risk = time.perf_counter()
        risk_decision = risk_layer.evaluate(
            router_decision, expert_outputs, regime_state,
            state.zone_map, feature_vector, state.market_snapshot, equity,
        )
        stage_timings_ms["risk_layer_ms"] = (time.perf_counter() - t_risk) * 1000
        state = dc_replace(state, risk_decision=risk_decision)

        if risk_decision.approved:
            expert_output = expert_outputs.get_by_name(router_decision.selected_expert)
            t_exec = time.perf_counter()
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
                # Report the bar as a failed execution, not as "no expert
                # candidate": the expert and the risk layer both said yes,
                # and it was the order placement that broke.
                execution_failed = True
                execution_error = str(exc)
            stage_timings_ms["execution_ms"] = (time.perf_counter() - t_exec) * 1000

            if execution_record is not None and execution_record.status in ("filled", "partial"):
                direction = 1 if execution_record.side == "BUY" else -1
                tp_pct, sl_pct = compute_position_barriers(config, regime_state, feature_vector)
                risk_layer.record_trade(state.timestamp)

                # Entry latency: the fill's own timestamp offset, or the
                # configured assumption when that is larger (live backends
                # stamp the fill with the bar label, so they measure 0).
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
                    max_holding_bars=config.get("labeling", {}).get("max_holding_bars", 5),
                    fees_paid=execution_record.filled_size * (execution_record.fees_pct / 100),
                    expert=router_decision.selected_expert,
                    regime=regime_state.primary_regime.value,
                    selection_confidence=router_decision.selection_confidence,
                    risk_score=risk_decision.risk_score,
                    min_hold_bars=entry_policy.min_hold_bars,
                    min_hold_seconds=entry_policy.min_trade_seconds,
                    entry_latency_seconds=entry_policy.entry_latency_seconds,
                )
                if on_trade_open is not None:
                    trade_id = on_trade_open(new_position)
                    if trade_id:
                        new_position.trade_id = trade_id
                position = opened = new_position
                logger.info(
                    "[%s] ENTRY %s via %s size=%.2f price=%.2f",
                    state.timestamp, execution_record.side, router_decision.selected_expert,
                    execution_record.filled_size, execution_record.fill_price,
                )

            state = dc_replace(state, execution_record=execution_record)

    # Decision trace -- built AFTER this bar's risk and execution results
    # exist, so `executed` and `risk_layer` are classified from THIS bar.
    stage_timings_ms.update(pipeline_timings or {})
    stages = build_trace_stages(state, stage_timings_ms)
    # The classifier lives in src/diagnostics/decision_trace.py so every
    # surface shares one vocabulary.
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

    return BarOutcome(
        state=state, position=position, equity=equity,
        position_is_open=position_is_open,
        execution_failed=execution_failed, execution_error=execution_error,
        stage_timings_ms=stage_timings_ms, trace=trace,
        opened=opened, closed=closed, stages=stages,
    )
