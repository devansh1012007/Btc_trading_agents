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
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.diagnostics.decision_trace import DecisionTraceWriter, FaultLogWriter
from src.execution.execution_engine import ExecutionEngine
from src.execution.position_lifecycle import (  # noqa: F401  (re-exported for other scripts)
    BarOutcome,
    OpenPosition,
    build_execution_engine,
    check_position_exit,
    max_concurrent_positions,
    process_bar,
)
from src.execution.reconciliation import reconcile
from src.feedback.system_state_archive import write_system_state_archive
from src.feedback.trade_journal import record_trade_entry, record_trade_exit
from src.feedback.zone_event_log import flush_zone_events
from src.labeling.hold_policy import alignment_settings
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


def journal_callbacks(mode: str) -> tuple:
    """The two persistence callbacks ``process_bar`` needs.

    These are the ONLY things that differ between this loop and the GUI
    replay: here a trade is written to the SQLite trade journal (whose
    id the position then carries); the replay appends to a list.
    """
    def on_open(position: OpenPosition) -> str:
        return record_trade_entry(
            position.entry_timestamp, position.entry_price, position.expert,
            position.regime, position.direction, position.size_usdt,
            position.selection_confidence, position.risk_score, mode,
        )

    def on_close(position: OpenPosition, exit_timestamp: int, exit_info: dict) -> None:
        record_trade_exit(
            position.trade_id, exit_timestamp, exit_info["exit_price"], exit_info["exit_reason"],
            exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
            position.fees_paid, 0.0,
        )

    return on_open, on_close


def emit_bar_artifacts(outcome: BarOutcome, trace_writer: DecisionTraceWriter) -> None:
    """Write a bar's decision trace and its section-8 decision record."""
    trace_writer.append(outcome.trace)
    log_trade_decision({
        **decision_record_from_state(outcome.state),
        "terminal_stage": outcome.terminal_stage,
        "terminal_reason": outcome.terminal_reason,
    })


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
    on_open, on_close = journal_callbacks(mode)

    for snapshot in snapshots:
        t0 = time.perf_counter()
        state = pipeline.run_live_bar(snapshot)
        decision_latency_ms = (time.perf_counter() - t0) * 1000
        if state is None:
            continue

        outcome = process_bar(
            state, risk_layer=risk_layer, execution_engine=execution_engine,
            position=open_position, equity=equity, fee_pct=fee_pct, config=pipeline.config,
            decision_latency_ms=decision_latency_ms,
            pipeline_timings=pipeline.last_bar_timings,
            on_trade_open=on_open, on_trade_close=on_close,
        )
        open_position, equity = outcome.position, outcome.equity
        n_trades += outcome.opened is not None

        emit_bar_artifacts(outcome, trace_writer)
        states_buffer.append(outcome.state)

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
    as run_historical (``process_bar``), sleep until the next bar boundary,
    repeat until the kill switch is set or the process is killed.
    """
    from src.layers.market_data import MarketDataLayer

    granularity = config.get("system", {}).get("data_granularity", "1m")
    poll_seconds = GRANULARITY_SECONDS.get(granularity, 60)
    kill_switch_path = "data/control/EMERGENCY_STOP"
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)
    initial_equity = 10000.0

    trace_writer = DecisionTraceWriter()
    fault_writer = FaultLogWriter()
    on_open, on_close = journal_callbacks("live")

    # Fail fast on an invalid config rather than once per bar inside the
    # fault handler below.
    max_concurrent_positions(risk_layer.breaker_cfg)

    logger.info(f"Starting live loop: granularity={granularity}, poll_seconds={poll_seconds}")
    reconcile(connector, symbol)

    equity = initial_equity
    open_position: Optional[OpenPosition] = None
    n_trades = 0
    last_timestamp: Optional[int] = None

    while True:
        # Kill switch check first
        if os.path.exists(kill_switch_path):
            logger.warning(f"{kill_switch_path} present -- halting live loop.")
            break

        try:
            df_ohlcv = connector.fetch_ohlcv(symbol, timeframe=granularity, limit=1)
            if df_ohlcv.empty:
                logger.warning("No data fetched, sleeping...")
                time.sleep(poll_seconds)
                continue

            snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
            if not snapshots:
                logger.warning("No snapshots created, sleeping...")
                time.sleep(poll_seconds)
                continue

            t0 = time.perf_counter()
            state = pipeline.run_live_bar(snapshots[0])
            decision_latency_ms = (time.perf_counter() - t0) * 1000
            if state is None:
                time.sleep(poll_seconds)
                continue
            last_timestamp = state.timestamp

            outcome = process_bar(
                state, risk_layer=risk_layer, execution_engine=execution_engine,
                position=open_position, equity=equity, fee_pct=fee_pct, config=config,
                decision_latency_ms=decision_latency_ms,
                pipeline_timings=pipeline.last_bar_timings,
                on_trade_open=on_open, on_trade_close=on_close,
            )
            open_position, equity = outcome.position, outcome.equity
            n_trades += outcome.opened is not None

            emit_bar_artifacts(outcome, trace_writer)

        except Exception as e:
            fault_writer.log_fault(e, symbol, system_state_snapshot=None, timestamp=last_timestamp)
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

    hold_policy, require_aligned = alignment_settings(config)   # fail fast on an unsatisfiable policy
    risk_layer = RiskLayer(
        config=config.get("risk", {}),
        hold_policy=hold_policy, require_aligned_models=require_aligned,
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
