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
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.diagnostics.decision_trace import (
    DecisionTrace,
    DecisionTraceWriter,
    StageResult,
    build_trace_stages,
    classify_terminal_stage,
)
from src.execution.execution_engine import ExecutionEngine
from src.execution.position_lifecycle import (
    OpenPosition,
    build_execution_engine,
    check_position_exit,
)
from src.execution.reconciliation import reconcile
from src.feedback.system_state_archive import write_system_state_archive
from src.feedback.trade_journal import record_trade_entry, record_trade_exit
from src.feedback.zone_event_log import flush_zone_events
from src.labeling.triple_barrier import barrier_volatility_pct, compute_barrier_pcts
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.pipeline import TradingPipeline
from src.state.contracts import SystemState
from src.utils.config_helpers import load_layered_config
from src.utils.logging_setup import decision_record_from_state, log_trade_decision, setup_logging
from src.utils.time_utils import describe_epoch_unit, epoch_seconds

GRANULARITY_SECONDS = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}


def load_ohlcv_for_run(input_path: Path) -> pd.DataFrame:
    """Load the historical input and normalise its timestamp column.

    A numeric timestamp column is epoch seconds / milliseconds /
    microseconds / nanoseconds -- pandas' default for a bare integer
    column treats them as nanoseconds, which silently collapses every
    bar onto 1970-01-01. The unit is detected per value by the SAME
    helper the GUI uses, so the two can never disagree about when the
    system traded, and it is logged so a mislabelled unit cannot pass
    unnoticed.
    """
    df = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            unit = describe_epoch_unit(ts) or "s"
            logger.info("Reading %s timestamps as epoch %s", input_path, unit)
        else:
            logger.info("Reading %s timestamps as datetimes", input_path)
        df["timestamp"] = pd.to_datetime(epoch_seconds(ts), unit="s", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


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
        t0 = time.perf_counter()
        state = pipeline.run_live_bar(snapshot)
        decision_latency_ms = (time.perf_counter() - t0) * 1000
        if state is None:
            continue
        risk_layer.store.heartbeat()

        # Per-bar risk / execution results. The pipeline returns a fresh
        # SystemState for every bar, so `state` can never carry the
        # previous bar's outcome; these locals hold this bar's results
        # and are reset so a bar that never reaches the risk layer
        # cannot report the previous bar's.
        risk_decision = None
        execution_record = None
        execution_failed = False
        execution_error = ""
        stage_timings_ms: Dict[str, float] = {}

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

        # Enforce max_concurrent_positions from config. When a position
        # is already open the bar is still processed (traced, archived) --
        # it simply cannot open a second one.
        position_is_open = max_concurrent == 1 and open_position is not None

        if not position_is_open:
            if open_position is None and state.router_decision is not None:
                t_risk = time.perf_counter()
                risk_decision = risk_layer.evaluate(
                    state.router_decision, state.expert_outputs, state.regime_state,
                    state.zone_map, state.feature_vector, state.market_snapshot, equity,
                )
                stage_timings_ms["risk_layer_ms"] = (time.perf_counter() - t_risk) * 1000
                state = replace(state, risk_decision=risk_decision)

                if risk_decision.approved:
                    expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
                    t_exec = time.perf_counter()
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
                        # The bar must be reported as a failed execution,
                        # not as "no expert candidate": the expert and
                        # the risk layer both said yes, and it was the
                        # order placement that broke.
                        execution_failed = True
                        execution_error = str(e)
                    stage_timings_ms["execution_ms"] = (time.perf_counter() - t_exec) * 1000

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
                            expert=state.router_decision.selected_expert,
                            regime=state.regime_state.primary_regime.value,
                            selection_confidence=state.router_decision.selection_confidence,
                            risk_score=risk_decision.risk_score,
                        )
                        n_trades += 1
                        logger.info(f"[{state.timestamp}] ENTRY {execution_record.side} via {state.router_decision.selected_expert} "
                                    f"size={execution_record.filled_size:.2f} price={execution_record.fill_price:.2f}")

                    state = replace(state, execution_record=execution_record)

        # Decision trace + §8 record -- written AFTER this bar's risk
        # and execution results exist, so `executed` and `risk_layer`
        # are classified from THIS bar instead of being absent.
        # Build decision trace stages from pipeline output
        stage_timings_ms.update(pipeline.last_bar_timings)
        stages: List[StageResult] = build_trace_stages(state, stage_timings_ms)

        # Determine terminal stage and reason. The classifier lives in
        # src/diagnostics/decision_trace.py so the runtime and the GUI
        # can never disagree about why a bar stopped where it did.
        terminal_stage, terminal_reason = classify_terminal_stage(
            state.opportunity_score, state.router_decision,
            state.risk_decision, state.execution_record,
            position_open=position_is_open,
            execution_failed=execution_failed,
            execution_error=execution_error or None,
        )

        trace_writer.append(DecisionTrace(
            timestamp=state.timestamp, symbol=state.symbol,
            stages=stages, terminal_stage=terminal_stage, terminal_reason=terminal_reason,
            total_duration_ms=decision_latency_ms,
            stage_timings_ms=stage_timings_ms,
        ))
        log_trade_decision({**decision_record_from_state(state),
                            "terminal_stage": terminal_stage, "terminal_reason": terminal_reason})

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
            t0 = time.perf_counter()
            state = pipeline.run_live_bar(snapshot)
            decision_latency_ms = (time.perf_counter() - t0) * 1000
            if state is None:
                time.sleep(poll_seconds)
                continue

            risk_layer.store.heartbeat()

            # Per-bar risk / execution results. The pipeline returns a fresh
            # SystemState for every bar, so `state` can never carry the
            # previous bar's outcome; these locals hold this bar's results
            # and are reset so a bar that never reaches the risk layer
            # cannot report the previous bar's.
            risk_decision = None
            execution_record = None
            execution_failed = False
            execution_error = ""
            stage_timings_ms: Dict[str, float] = {}

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

            # 2. Enforce max_concurrent_positions from config. When a position
            # is already open the bar is still processed (traced, archived) --
            # it simply cannot open a second one.
            max_concurrent = risk_layer.breaker_cfg.get("max_concurrent_positions", 1)
            position_is_open = max_concurrent == 1 and open_position is not None

            if not position_is_open:
                if open_position is None and state.router_decision is not None:
                    t_risk = time.perf_counter()
                    risk_decision = risk_layer.evaluate(
                        state.router_decision, state.expert_outputs, state.regime_state,
                        state.zone_map, state.feature_vector, state.market_snapshot, equity,
                    )
                    stage_timings_ms["risk_layer_ms"] = (time.perf_counter() - t_risk) * 1000
                    state = replace(state, risk_decision=risk_decision)

                    if risk_decision.approved:
                        expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
                        t_exec = time.perf_counter()
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
                            # Report the bar as a failed execution rather
                            # than as "no expert candidate": the expert
                            # and the risk layer both said yes, and it
                            # was the order placement that broke.
                            execution_failed = True
                            execution_error = str(e)
                        stage_timings_ms["execution_ms"] = (time.perf_counter() - t_exec) * 1000

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
                                expert=state.router_decision.selected_expert,
                                regime=state.regime_state.primary_regime.value,
                                selection_confidence=state.router_decision.selection_confidence,
                                risk_score=risk_decision.risk_score,
                            )
                            n_trades += 1
                            logger.info(f"[{state.timestamp}] ENTRY {execution_record.side} via {state.router_decision.selected_expert} "
                                        f"size={execution_record.filled_size:.2f} price={execution_record.fill_price:.2f}")

                        state = replace(state, execution_record=execution_record)

            # Decision trace + §8 record -- written AFTER this bar's risk
            # and execution results exist, so `executed` and `risk_layer`
            # are classified from THIS bar instead of being absent.
            # Build decision trace stages (same as run_historical)
            stage_timings_ms.update(pipeline.last_bar_timings)
            stages: List[StageResult] = build_trace_stages(state, stage_timings_ms)

            # Determine terminal stage and reason -- shared classifier, see
            # src/diagnostics/decision_trace.py.
            terminal_stage, terminal_reason = classify_terminal_stage(
                state.opportunity_score, state.router_decision,
                state.risk_decision, state.execution_record,
                position_open=position_is_open,
                execution_failed=execution_failed,
                execution_error=execution_error or None,
            )

            trace_writer.append(DecisionTrace(
                timestamp=state.timestamp, symbol=state.symbol,
                stages=stages, terminal_stage=terminal_stage, terminal_reason=terminal_reason,
                total_duration_ms=decision_latency_ms,
                stage_timings_ms=stage_timings_ms,
            ))
            log_trade_decision({**decision_record_from_state(state),
                                "terminal_stage": terminal_stage, "terminal_reason": terminal_reason})

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

    setup_logging()
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
        df_ohlcv = load_ohlcv_for_run(input_path)

        run_historical(pipeline, risk_layer, execution_engine, df_ohlcv, args.env, fee_pct,
                        args.initial_equity, max_bars=args.max_bars)
    else:
        run_live_loop(pipeline, risk_layer, execution_engine, config, pipeline.connector, symbol)


if __name__ == "__main__":
    main()
