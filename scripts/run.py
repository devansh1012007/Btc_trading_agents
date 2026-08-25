"""
Main deployment entry point (§11, M4.13's integration target). Wires the
full Phase 1-4 stack together for real: pipeline (features -> regime ->
structure -> experts -> opportunity filter -> router) -> Risk Layer ->
Execution Engine -> trade journal + zone event log + SystemState archive.

Two modes:
  --mode historical  Replays a historical OHLCV file bar-by-bar through
                      the full stack, including REAL position lifecycle
                      tracking (open position, check its own TP/SL/
                      timeout each subsequent bar, close and journal it)
                      -- this is a genuine paper_sim backtest through
                      ExecutionRecord, not the triple-barrier proxy
                      backtest_phase3.py uses for training-data
                      harvesting. Bounded and testable.
  --mode live         Polls the exchange continuously (poll interval
                      derived from system.data_granularity), attaches
                      real order-book liquidity (paper-live/live modes),
                      and runs forever until killed or the manual kill
                      switch is set. NOT exercised against a real feed in
                      this delivery -- structure is complete; validating
                      it needs your real credentials and a live connection.

One switch controls which Execution backend is live: execution.mode in
whichever settings.{env}.yaml overlay is loaded via --env. Nothing in
this loop branches on mode itself -- that's the ExecutionEngine's job,
and is exactly what keeps this script mode-blind per §6.1's design
constraint.
"""
import argparse
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.diagnostics.decision_trace import DecisionTrace, DecisionTraceWriter, StageResult
from src.execution.backends.live_backend import LiveBackend
from src.execution.backends.paper_live_backend import PaperLiveBackend
from src.execution.backends.paper_sim_backend import PaperSimBackend
from src.execution.execution_engine import ExecutionEngine
from src.execution.reconciliation import reconcile
from src.feedback.system_state_archive import write_system_state_archive
from src.feedback.trade_journal import record_trade_entry, record_trade_exit
from src.feedback.zone_event_log import flush_zone_events
from src.labeling.triple_barrier import barrier_volatility_pct, compute_barrier_pcts
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.pipeline import TradingPipeline
from src.state.contracts import ExpertAction, SystemState
from src.utils.config_helpers import load_layered_config

GRANULARITY_SECONDS = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}


@dataclass
class OpenPosition:
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


def build_execution_engine(config: dict, connector, symbol: str) -> ExecutionEngine:
    exec_cfg = config.get("execution", {})
    mode = exec_cfg.get("mode", "paper_sim")
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)

    if mode == "paper_sim":
        backend = PaperSimBackend(exec_cfg, fee_pct=fee_pct)
    elif mode == "paper_live":
        backend = PaperLiveBackend(exec_cfg, connector, symbol, fee_pct=fee_pct)
    elif mode == "live":
        backend = LiveBackend(exec_cfg, connector, symbol)
    else:
        raise ValueError(f"Unknown execution.mode: {mode}")

    return ExecutionEngine(backend, mode)


def check_position_exit(position: OpenPosition, bar, fee_pct: float) -> Optional[dict]:
    """One bar-step of triple-barrier-style exit checking for a live open
    position -- same TP/SL/timeout convention as
    src/labeling/triple_barrier.py, applied incrementally instead of over
    a pre-collected forward window (there is no "forward window" yet when
    running live; each bar arrives one at a time)."""
    entry = position.entry_price
    if position.direction == 1:
        tp_level = entry * (1 + position.take_profit_pct / 100)
        sl_level = entry * (1 - position.stop_loss_pct / 100)
        hit_tp, hit_sl = bar.high >= tp_level, bar.low <= sl_level
    else:
        tp_level = entry * (1 - position.take_profit_pct / 100)
        sl_level = entry * (1 + position.stop_loss_pct / 100)
        hit_tp, hit_sl = bar.low <= tp_level, bar.high >= sl_level

    position.bars_held += 1

    if hit_sl:
        exit_price, exit_reason = sl_level, "stop_loss"
    elif hit_tp:
        exit_price, exit_reason = tp_level, "take_profit"
    elif position.bars_held >= position.max_holding_bars:
        exit_price, exit_reason = bar.close, "timeout"
    else:
        return None

    raw_return_pct = (
        (exit_price - entry) / entry * 100 if position.direction == 1
        else (entry - exit_price) / entry * 100
    )
    realized_pnl_pct = raw_return_pct - (fee_pct * 2)
    realized_pnl_usdt = position.size_usdt * (realized_pnl_pct / 100)

    return {
        "exit_price": exit_price, "exit_reason": exit_reason,
        "realized_pnl_pct": realized_pnl_pct, "realized_pnl_usdt": realized_pnl_usdt,
    }


def run_historical(pipeline: TradingPipeline, risk_layer: RiskLayer, execution_engine: ExecutionEngine,
                    df_ohlcv: pd.DataFrame, mode: str, fee_pct: float, initial_equity: float,
                    max_bars: Optional[int] = None) -> None:
    from src.layers.market_data import MarketDataLayer

    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, pipeline.symbol)
    if max_bars:
        snapshots = snapshots[:max_bars]

    equity = initial_equity
    open_position: Optional[OpenPosition] = None
    states_buffer: List[SystemState] = []
    n_trades = 0
    trace_writer = DecisionTraceWriter()

    for snapshot in snapshots:
        # Manual kill switch + heartbeat -- checked every bar, first thing.
        state = pipeline.run_live_bar(snapshot)
        if state is None:
            continue
        risk_layer.store.heartbeat()

        # Build decision trace stages from pipeline output
        stages: List[StageResult] = []

        # Stage: expert layer
        expert_outputs = state.expert_outputs
        if expert_outputs and expert_outputs.all():
            candidates = [o for o in expert_outputs.all() if o.action != ExpertAction.NO_TRADE]
            if candidates:
                stages.append(StageResult(
                    stage_name="expert_layer", passed=True,
                    detail=f"{len(candidates)} candidate(s): {', '.join(c.expert_name for c in candidates)}",
                    values={o.expert_name: o.confidence for o in candidates},
                ))
            else:
                stages.append(StageResult(
                    stage_name="expert_layer", passed=False,
                    detail="All experts returned NO_TRADE",
                    values={},
                ))
        else:
            stages.append(StageResult(
                stage_name="expert_layer", passed=False,
                detail="No expert outputs",
                values={},
            ))

        # Stage: opportunity filter
        opp_score = state.opportunity_score
        if opp_score:
            stages.append(StageResult(
                stage_name="opportunity_filter", passed=opp_score.tradable,
                detail=" ".join(opp_score.failed_rules) if opp_score.failed_rules else "passed",
                values={"tradable_probability": opp_score.tradable_probability,
                        "environment_quality": opp_score.environment_quality},
            ))

        # Stage: router
        router_decision = state.router_decision
        if router_decision:
            stages.append(StageResult(
                stage_name="router", passed=router_decision.selected_expert != "no_trade",
                detail=router_decision.reasoning or "",
                values={"selection_confidence": router_decision.selection_confidence,
                        "no_trade_probability": router_decision.no_trade_probability},
            ))

        # Stage: risk layer
        risk_decision = state.risk_decision
        if risk_decision:
            stages.append(StageResult(
                stage_name="risk_layer", passed=risk_decision.approved,
                detail=risk_decision.rejection_reason or "",
                values={"risk_score": risk_decision.risk_score,
                        "size_reduction_pct": risk_decision.size_reduction_pct},
            ))

        # Stage: execution
        exec_record = state.execution_record
        if exec_record:
            stages.append(StageResult(
                stage_name="execution", passed=exec_record.status in ("filled", "partial"),
                detail=f"status={exec_record.status}",
                values={"fill_price": exec_record.fill_price, "slippage_pct": exec_record.slippage_pct},
            ))

        # Determine terminal stage and reason. Filter veto is classified
        # BEFORE the router: when the filter rejects, Router.decide() itself
        # returns no_trade with reasoning "opportunity_filter_rejected", so
        # checking the router first mislabels every filter veto.
        if exec_record and exec_record.status in ("filled", "partial"):
            terminal_stage = "executed"
            terminal_reason = f"filled via {state.router_decision.selected_expert if state.router_decision else 'unknown'}"
        elif risk_decision and not risk_decision.approved:
            terminal_stage = "risk_layer"
            terminal_reason = risk_decision.rejection_reason or "risk rejection"
        elif opp_score and not opp_score.tradable:
            terminal_stage = "opportunity_filter"
            terminal_reason = " | ".join(opp_score.failed_rules) if opp_score.failed_rules else "filter veto"
        elif router_decision and router_decision.selected_expert == "no_trade":
            terminal_stage = "router_no_trade"
            terminal_reason = router_decision.reasoning or "router declined"
        else:
            terminal_stage = "no_expert_candidate"
            terminal_reason = "no expert candidate"

        trace_writer.append(DecisionTrace(
            timestamp=state.timestamp, symbol=state.symbol,
            stages=stages, terminal_stage=terminal_stage, terminal_reason=terminal_reason,
        ))

        realized_pnl_this_bar = 0.0

        # 1. Manage any open position first -- exits are never gated by
        # the Risk Layer (see risk_layer.py's module docstring).
        if open_position is not None:
            exit_info = check_position_exit(open_position, state.market_snapshot, fee_pct)
            if exit_info is not None:
                record_trade_exit(
                    open_position.trade_id, state.timestamp, exit_info["exit_price"], exit_info["exit_reason"],
                    exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
                    open_position.fees_paid, 0.0,
                )
                risk_layer.store.record_trade_result(exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"])
                equity += exit_info["realized_pnl_usdt"]
                drawdown = risk_layer.store.update_equity(equity)
                if drawdown >= risk_layer.breaker_cfg.get("max_drawdown_pct", 8.0):
                    risk_layer.store.trip("max_drawdown", f"drawdown={drawdown:.2f}%")
                    logger.warning(f"max_drawdown breaker tripped at {drawdown:.2f}% -- halting new entries")
                realized_pnl_this_bar = exit_info["realized_pnl_pct"]
                logger.info(f"[{state.timestamp}] EXIT {exit_info['exit_reason']} pnl={exit_info['realized_pnl_pct']:.3f}% equity={equity:.2f}")
                open_position = None

        # 2. Only look for a NEW entry if nothing is currently open
        # (risk.circuit_breakers.max_concurrent_positions=1 in the
        # shipped config -- enforced here directly rather than as a
        # RiskLayer hard rule, since it's about STATE, not the candidate
        # itself).
        max_concurrent = risk_layer.breaker_cfg.get("max_concurrent_positions", 1)
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent_positions={max_concurrent} is invalid. "
                f"Must be >= 1. (Real multi-position support is a future item.)"
            )

        # Enforce max_concurrent_positions from config
        if max_concurrent == 1 and open_position is not None:
            # Skip new entry if we already have a position
            risk_layer.advance_bar()
            continue

        if open_position is None and state.router_decision is not None:
            risk_decision = risk_layer.evaluate(
                state.router_decision, state.expert_outputs, state.regime_state,
                state.zone_map, state.feature_vector, state.market_snapshot, equity,
            )
            state = replace(state, risk_decision=risk_decision)

            if risk_decision.approved:
                expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
                try:
                    execution_record = execution_engine.place_order(
                        risk_decision, state.router_decision, expert_output, state.market_snapshot,
                    )
                    risk_layer.store.record_execution_result(success=True)
                except Exception as e:
                    logger.error(f"Execution failed: {e}")
                    risk_layer.store.record_execution_result(success=False)
                    if risk_layer.store.consecutive_execution_errors() >= risk_layer.breaker_cfg.get("consecutive_execution_errors_halt", 3):
                        risk_layer.store.trip("consecutive_execution_errors", "3 consecutive execution failures")
                    execution_record = None

                if execution_record is not None and execution_record.status in ("filled", "partial"):
                    direction = 1 if execution_record.side == "BUY" else -1
                    labeling_cfg = pipeline.config.get("labeling", {})
                    regime_key = state.regime_state.primary_regime.name
                    overrides = labeling_cfg.get("regime_specific_barriers", {}).get(regime_key, {})
                    k_tp = overrides.get("k_tp", labeling_cfg.get("k_tp", 1.5))
                    k_sl = overrides.get("k_sl", labeling_cfg.get("k_sl", 1.0))
                    tp_pct, sl_pct = compute_barrier_pcts(barrier_volatility_pct(state.feature_vector), k_tp, k_sl)

                    # Record trade for rate limiting
                    risk_layer.record_trade(state.timestamp)

                    trade_id = record_trade_entry(
                        state.timestamp, execution_record.fill_price, state.router_decision.selected_expert,
                        state.regime_state.primary_regime.value, direction, execution_record.filled_size,
                        state.router_decision.selection_confidence, risk_decision.risk_score, mode,
                    )
                    open_position = OpenPosition(
                        trade_id=trade_id, entry_timestamp=state.timestamp, entry_price=execution_record.fill_price,
                        direction=direction, size_usdt=execution_record.filled_size,
                        take_profit_pct=tp_pct, stop_loss_pct=sl_pct,
                        max_holding_bars=labeling_cfg.get("max_holding_bars", 5),
                        fees_paid=execution_record.filled_size * (execution_record.fees_pct / 100),
                    )
                    n_trades += 1
                    logger.info(f"[{state.timestamp}] ENTRY {execution_record.side} via {state.router_decision.selected_expert} "
                                f"size={execution_record.filled_size:.2f} price={execution_record.fill_price:.2f}")

                state = replace(state, execution_record=execution_record)

        state = replace(state, realized_pnl=realized_pnl_this_bar)
        risk_layer.advance_bar()
        states_buffer.append(state)

        if len(states_buffer) >= 500:
            write_system_state_archive(states_buffer)
            states_buffer = []

    if states_buffer:
        write_system_state_archive(states_buffer)

    flush_zone_events(pipeline.structure_engine.lifecycle_events)

    logger.info(f"Historical run complete: {n_trades} trades, final equity={equity:.2f} "
                f"(started at {initial_equity:.2f}, {(equity/initial_equity - 1)*100:+.2f}%)")


def run_live_loop(pipeline: TradingPipeline, risk_layer: RiskLayer, execution_engine: ExecutionEngine,
                    config: dict, connector, symbol: str) -> None:
    """
    Continuous polling loop: fetch latest closed bar via connector, attach
    real liquidity (paper-live/live), run it through the same per-bar logic
    as run_historical, sleep until the next bar boundary, repeat until the
    kill switch is set or the process is killed.
    """
    from src.diagnostics.decision_trace import DecisionTrace, DecisionTraceWriter, FaultLogWriter, StageResult
    from src.layers.market_data import MarketDataLayer

    granularity = config.get("system", {}).get("data_granularity", "1m")
    poll_seconds = GRANULARITY_SECONDS.get(granularity, 60)
    kill_switch_path = "data/control/EMERGENCY_STOP"
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)
    initial_equity = 10000.0

    trace_writer = DecisionTraceWriter()
    fault_writer = FaultLogWriter()

    logger.info(f"Starting live loop: granularity={granularity}, poll_seconds={poll_seconds}")
    reconcile(connector, symbol)

    equity = initial_equity
    open_position: Optional[OpenPosition] = None
    n_trades = 0

    while True:
        # Kill switch check first
        if os.path.exists(kill_switch_path):
            logger.warning(f"{kill_switch_path} present -- halting live loop.")
            break

        try:
            # Fetch latest OHLCV bar
            df_ohlcv = connector.fetch_ohlcv(symbol, timeframe=granularity, limit=1)
            if df_ohlcv.empty:
                logger.warning("No data fetched, sleeping...")
                time.sleep(poll_seconds)
                continue

            # Build snapshot
            snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
            if not snapshots:
                logger.warning("No snapshots created, sleeping...")
                time.sleep(poll_seconds)
                continue

            snapshot = snapshots[0]

            # Run through pipeline
            state = pipeline.run_live_bar(snapshot)
            if state is None:
                time.sleep(poll_seconds)
                continue

            risk_layer.store.heartbeat()

            # Build decision trace stages (same as run_historical)
            stages: List[StageResult] = []

            # Stage: expert layer
            expert_outputs = state.expert_outputs
            if expert_outputs and expert_outputs.all():
                candidates = [o for o in expert_outputs.all() if o.action != ExpertAction.NO_TRADE]
                if candidates:
                    stages.append(StageResult(
                        stage_name="expert_layer", passed=True,
                        detail=f"{len(candidates)} candidate(s): {', '.join(c.expert_name for c in candidates)}",
                        values={o.expert_name: o.confidence for o in candidates},
                    ))
                else:
                    stages.append(StageResult(
                        stage_name="expert_layer", passed=False,
                        detail="All experts returned NO_TRADE",
                        values={},
                    ))
            else:
                stages.append(StageResult(
                    stage_name="expert_layer", passed=False,
                    detail="No expert outputs",
                    values={},
                ))

            # Stage: opportunity filter
            opp_score = state.opportunity_score
            if opp_score:
                stages.append(StageResult(
                    stage_name="opportunity_filter", passed=opp_score.tradable,
                    detail=" ".join(opp_score.failed_rules) if opp_score.failed_rules else "passed",
                    values={"tradable_probability": opp_score.tradable_probability,
                            "environment_quality": opp_score.environment_quality},
                ))

            # Stage: router
            router_decision = state.router_decision
            if router_decision:
                stages.append(StageResult(
                    stage_name="router", passed=router_decision.selected_expert != "no_trade",
                    detail=router_decision.reasoning or "",
                    values={"selection_confidence": router_decision.selection_confidence,
                            "no_trade_probability": router_decision.no_trade_probability},
                ))

            # Stage: risk layer
            risk_decision = state.risk_decision
            if risk_decision:
                stages.append(StageResult(
                    stage_name="risk_layer", passed=risk_decision.approved,
                    detail=risk_decision.rejection_reason or "",
                    values={"risk_score": risk_decision.risk_score,
                            "size_reduction_pct": risk_decision.size_reduction_pct},
                ))

            # Stage: execution
            exec_record = state.execution_record
            if exec_record:
                stages.append(StageResult(
                    stage_name="execution", passed=exec_record.status in ("filled", "partial"),
                    detail=f"status={exec_record.status}",
                    values={"fill_price": exec_record.fill_price, "slippage_pct": exec_record.slippage_pct},
                ))

            # Determine terminal stage and reason
            if exec_record and exec_record.status in ("filled", "partial"):
                terminal_stage = "executed"
                terminal_reason = f"filled via {state.router_decision.selected_expert if state.router_decision else 'unknown'}"
            elif risk_decision and not risk_decision.approved:
                terminal_stage = "risk_layer"
                terminal_reason = risk_decision.rejection_reason or "risk rejection"
            elif opp_score and not opp_score.tradable:
                # Filter veto must be classified BEFORE the router: when the
                # filter rejects, Router.decide() itself returns no_trade with
                # reasoning "opportunity_filter_rejected", so checking the
                # router first mislabels every filter veto as a router decline.
                terminal_stage = "opportunity_filter"
                terminal_reason = " | ".join(opp_score.failed_rules) if opp_score.failed_rules else "filter veto"
            elif router_decision and router_decision.selected_expert == "no_trade":
                terminal_stage = "router_no_trade"
                terminal_reason = router_decision.reasoning or "router declined"
            else:
                terminal_stage = "no_expert_candidate"
                terminal_reason = "no expert candidate"

            trace_writer.append(DecisionTrace(
                timestamp=state.timestamp, symbol=state.symbol,
                stages=stages, terminal_stage=terminal_stage, terminal_reason=terminal_reason,
            ))

            realized_pnl_this_bar = 0.0

            # 1. Manage any open position first
            if open_position is not None:
                exit_info = check_position_exit(open_position, state.market_snapshot, fee_pct)
                if exit_info is not None:
                    record_trade_exit(
                        open_position.trade_id, state.timestamp, exit_info["exit_price"], exit_info["exit_reason"],
                        exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
                        open_position.fees_paid, 0.0,
                    )
                    risk_layer.store.record_trade_result(exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"])
                    equity += exit_info["realized_pnl_usdt"]
                    drawdown = risk_layer.store.update_equity(equity)
                    if drawdown >= risk_layer.breaker_cfg.get("max_drawdown_pct", 8.0):
                        risk_layer.store.trip("max_drawdown", f"drawdown={drawdown:.2f}%")
                        logger.warning(f"max_drawdown breaker tripped at {drawdown:.2f}% -- halting new entries")
                    realized_pnl_this_bar = exit_info["realized_pnl_pct"]
                    logger.info(f"[{state.timestamp}] EXIT {exit_info['exit_reason']} pnl={exit_info['realized_pnl_pct']:.3f}% equity={equity:.2f}")
                    open_position = None

            # 2. Enforce max_concurrent_positions from config
            max_concurrent = risk_layer.breaker_cfg.get("max_concurrent_positions", 1)
            if max_concurrent == 1 and open_position is not None:
                risk_layer.advance_bar()
                time.sleep(poll_seconds)
                continue

            if open_position is None and state.router_decision is not None:
                risk_decision = risk_layer.evaluate(
                    state.router_decision, state.expert_outputs, state.regime_state,
                    state.zone_map, state.feature_vector, state.market_snapshot, equity,
                )
                state = replace(state, risk_decision=risk_decision)

                if risk_decision.approved:
                    expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
                    try:
                        execution_record = execution_engine.place_order(
                            risk_decision, state.router_decision, expert_output, state.market_snapshot,
                            fv=state.feature_vector,
                        )
                        risk_layer.store.record_execution_result(success=True)
                    except Exception as e:
                        logger.error(f"Execution failed: {e}")
                        risk_layer.store.record_execution_result(success=False)
                        if risk_layer.store.consecutive_execution_errors() >= risk_layer.breaker_cfg.get("consecutive_execution_errors_halt", 3):
                            risk_layer.store.trip("consecutive_execution_errors", "3 consecutive execution failures")
                        execution_record = None

                    if execution_record is not None and execution_record.status in ("filled", "partial"):
                        direction = 1 if execution_record.side == "BUY" else -1
                        labeling_cfg = pipeline.config.get("labeling", {})
                        regime_key = state.regime_state.primary_regime.name
                        overrides = labeling_cfg.get("regime_specific_barriers", {}).get(regime_key, {})
                        k_tp = overrides.get("k_tp", labeling_cfg.get("k_tp", 1.5))
                        k_sl = overrides.get("k_sl", labeling_cfg.get("k_sl", 1.0))
                        tp_pct, sl_pct = compute_barrier_pcts(barrier_volatility_pct(state.feature_vector), k_tp, k_sl)

                        risk_layer.record_trade(state.timestamp)

                        trade_id = record_trade_entry(
                            state.timestamp, execution_record.fill_price, state.router_decision.selected_expert,
                            state.regime_state.primary_regime.value, direction, execution_record.filled_size,
                            state.router_decision.selection_confidence, risk_decision.risk_score, "live",
                        )
                        open_position = OpenPosition(
                            trade_id=trade_id, entry_timestamp=state.timestamp, entry_price=execution_record.fill_price,
                            direction=direction, size_usdt=execution_record.filled_size,
                            take_profit_pct=tp_pct, stop_loss_pct=sl_pct,
                            max_holding_bars=labeling_cfg.get("max_holding_bars", 5),
                            fees_paid=execution_record.filled_size * (execution_record.fees_pct / 100),
                        )
                        n_trades += 1
                        logger.info(f"[{state.timestamp}] ENTRY {execution_record.side} via {state.router_decision.selected_expert} "
                                    f"size={execution_record.filled_size:.2f} price={execution_record.fill_price:.2f}")

                    state = replace(state, execution_record=execution_record)

            state = replace(state, realized_pnl=realized_pnl_this_bar)
            risk_layer.advance_bar()

        except Exception as e:
            fault_writer.log_fault(e, symbol, system_state_snapshot=None, timestamp=state.timestamp if 'state' in dir() else None)
            logger.error(f"Live loop error: {e}")
            logger.error(traceback.format_exc())

        time.sleep(poll_seconds)

    logger.info(f"Live loop stopped. Total trades: {n_trades}")


def main():
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    parser.add_argument("--mode", default="historical", choices=["historical", "live"])
    parser.add_argument("--input", default=None, help="Historical OHLCV Parquet/CSV (required for --mode historical)")
    parser.add_argument("--symbol", default=None, help="Overrides system.symbol from config")
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    args = parser.parse_args()

    config = load_layered_config(args.env)
    symbol = args.symbol or config.get("system", {}).get("symbol", "BTC/USDT")

    pipeline = TradingPipeline(symbol=symbol, config=config)

    risk_layer = RiskLayer(
        config=config.get("risk", {}),
        store=RiskStateStore(),
        model_path=config.get("risk", {}).get("ml_model", {}).get("model_path"),
        data_granularity=config.get("system", {}).get("data_granularity", "1m"),
    )
    execution_engine = build_execution_engine(config, pipeline.connector, symbol)
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)

    if args.mode == "historical":
        if not args.input:
            raise SystemExit("--mode historical requires --input <ohlcv file>")
        input_path = Path(args.input)
        df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
        if "timestamp" in df_ohlcv.columns:
            df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
        df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

        run_historical(pipeline, risk_layer, execution_engine, df_ohlcv, args.env, fee_pct,
                        args.initial_equity, max_bars=args.max_bars)
    else:
        run_live_loop(pipeline, risk_layer, execution_engine, config, pipeline.connector, symbol)


if __name__ == "__main__":
    main()
