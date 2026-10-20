"""
Replay a session through the real pipeline so the GUI has something to draw.

This is the GUI's equivalent of ``scripts/run.py --mode historical``:
the same pipeline, the same risk layer, the same paper-sim execution and
the same position lifecycle -- but instead of writing artifacts and
forgetting them, it keeps every bar's ``SystemState`` in memory and
returns a :class:`~src.gui.models.SessionBundle`.

Two deliberate differences from the deployment loop:

* The risk layer's persistent store lives in a throw-away SQLite file
  (a temp directory), so replaying a session can never trip a real
  circuit breaker or write to the production registry.
* Execution is forced to ``paper_sim``. The GUI is a viewer; it must
  never be able to place an order on an exchange.
"""
import copy
import logging
import shutil
import tempfile
import time
from dataclasses import asdict
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from ..diagnostics.decision_trace import build_trace_stages, classify_terminal_stage
from ..execution.position_lifecycle import OpenPosition, build_execution_engine, check_position_exit
from ..labeling.triple_barrier import barrier_volatility_pct, compute_barrier_pcts
from ..layers.market_data import MarketDataLayer
from ..layers.risk_layer import RiskLayer, RiskStateStore
from ..pipeline import TradingPipeline
from ..state.contracts import SystemState
from .instruments import InstrumentSpec, resolve_instrument
from .models import (
    DECISION_COLUMNS,
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    ZONE_COLUMNS,
    SessionBundle,
    empty_frame,
    normalize_dtypes,
)
from .timeframes import to_epoch_seconds

logger = logging.getLogger(__name__)

def relax_liquidity(config: dict) -> dict:
    """Loosen the opportunity filter's liquidity gates.

    The shipped synthetic liquidity model produces a wider spread than
    ``opportunity_filter.hard_rules.max_spread_pct`` allows, so a
    default run rejects essentially every bar and the GUI would have
    nothing to show. This mirrors ``--relax-liquidity`` in
    ``scripts/full_stack_walk_forward_backtest.py``: it makes the run
    measure PIPELINE BEHAVIOUR, not production edge, and the GUI labels
    the session accordingly.
    """
    cfg = copy.deepcopy(config)
    hard_rules = cfg.setdefault("opportunity_filter", {}).setdefault("hard_rules", {})
    hard_rules["max_spread_pct"] = 0.5
    hard_rules["min_depth_usdt"] = 0.0
    hard_rules["edge_vs_fees_threshold"] = -10.0
    return cfg


def _terminal_stage_for(
    state: SystemState,
    position_open: bool = False,
    execution_failed: bool = False,
    execution_error: str = "",
) -> tuple:
    """Classify the bar exactly the way the deployment loop does.

    ``position_open`` and ``execution_failed`` are passed in rather than
    derived from the state, because the state cannot know them: the
    deployment loop owns the position, and a failed order leaves no
    ``ExecutionRecord`` at all. Leaving them out is how the GUI ends up
    labelling a bar "no expert candidate" when the router had in fact
    selected an expert.
    """
    return classify_terminal_stage(
        state.opportunity_score, state.router_decision,
        state.risk_decision, state.execution_record,
        position_open=position_open,
        execution_failed=execution_failed,
        execution_error=execution_error or None,
    )


def _decision_row(
    state: SystemState,
    position_open: bool = False,
    execution_failed: bool = False,
    execution_error: str = "",
) -> dict:
    """One row of the per-bar decision frame."""
    regime_state = state.regime_state
    router = state.router_decision
    opp = state.opportunity_score
    risk = state.risk_decision
    exec_record = state.execution_record

    # A trade only happened if the order FILLED. A rejected or pending
    # order is not a trade, and reporting it as one would put a phantom
    # entry marker on every chart that reads this feed.
    if exec_record is not None and exec_record.status in ("filled", "partial"):
        decision = "LONG" if exec_record.side == "BUY" else "SHORT"
    else:
        decision = "NO_TRADE"

    terminal_stage, terminal_reason = _terminal_stage_for(
        state, position_open, execution_failed, execution_error
    )

    return {
        "timestamp": state.timestamp,
        "symbol": state.symbol,
        "regime": regime_state.primary_regime.value if regime_state else "unknown",
        "regime_confidence": regime_state.regime_confidence if regime_state else None,
        "expert": router.selected_expert if router else None,
        "opportunity_score": opp.tradable_probability if opp else None,
        "router_confidence": router.selection_confidence if router else None,
        "risk_score": risk.risk_score if risk else None,
        "risk_approved": risk.approved if risk else None,
        "decision": decision,
        "terminal_stage": terminal_stage,
        "terminal_reason": terminal_reason,
    }


def _trace_dict(
    state: SystemState,
    total_duration_ms: Optional[float] = None,
    stage_timings_ms: Optional[dict] = None,
    position_open: bool = False,
    execution_failed: bool = False,
    execution_error: str = "",
) -> dict:
    """A DecisionTrace-shaped dict for one bar.

    Same shape as a trace written by the deployment loop -- including the
    latency fields -- so a consumer cannot tell a replayed trace from a
    live one.
    """
    stages = [asdict(s) for s in build_trace_stages(state, stage_timings_ms)]
    terminal_stage, terminal_reason = _terminal_stage_for(
        state, position_open, execution_failed, execution_error
    )
    return {
        "timestamp": state.timestamp,
        "symbol": state.symbol,
        "stages": stages,
        "terminal_stage": terminal_stage,
        "terminal_reason": terminal_reason,
        "total_duration_ms": total_duration_ms,
        "stage_timings_ms": dict(stage_timings_ms or {}),
    }


def replay_session(
    df_ohlcv: pd.DataFrame,
    symbol: str,
    config: dict,
    initial_equity: float = 10_000.0,
    relax: bool = False,
    instrument: Optional[InstrumentSpec] = None,
    synthetic: bool = False,
) -> SessionBundle:
    """Replay ``df_ohlcv`` through the full stack and return the session.

    Args:
        df_ohlcv: bars with ``timestamp``, ``open``, ``high``, ``low``,
            ``close`` and ``volume`` columns.
        symbol: the symbol being traded.
        config: the merged deployment config (``load_layered_config``).
        initial_equity: starting equity for the simulated account.
        relax: loosen the opportunity filter's liquidity gates.
        instrument: contract metadata; resolved from the registry when
            not supplied.
        synthetic: mark the session as generated data (the GUI labels it).
    """
    cfg = relax_liquidity(config) if relax else copy.deepcopy(config)

    # The GUI is a viewer. Never let it reach an exchange.
    cfg.setdefault("execution", {})["mode"] = "paper_sim"

    # Normalise the timestamp column before it reaches the market data
    # layer: that layer reads an INTEGER epoch, so an ISO string or a
    # millisecond column would raise (or land in the year 57000).
    df_ohlcv = df_ohlcv.copy()
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = to_epoch_seconds(df_ohlcv["timestamp"])
        df_ohlcv = df_ohlcv.dropna(subset=["timestamp"])
        df_ohlcv["timestamp"] = df_ohlcv["timestamp"].astype("int64")

    spec = instrument or resolve_instrument(symbol)

    tmp_dir = tempfile.mkdtemp(prefix="gui_replay_")

    # The pipeline creates its data directories on construction. Point
    # them at the replay's throw-away directory: a viewer must not
    # scatter directories into whatever directory it was launched from.
    pipeline = TradingPipeline(symbol=symbol, config=cfg, data_dir=f"{tmp_dir}/data")
    data_granularity = cfg.get("system", {}).get("data_granularity", "1m")
    fee_pct = cfg.get("labeling", {}).get("fee_pct", 0.04)
    max_holding_bars = cfg.get("labeling", {}).get("max_holding_bars", 5)

    try:
        risk_layer = RiskLayer(
            config=cfg.get("risk", {}),
            store=RiskStateStore(db_path=f"{tmp_dir}/registry.db"),
            model_path=cfg.get("risk", {}).get("ml_model", {}).get("model_path"),
            data_granularity=data_granularity,
            kill_switch_path=f"{tmp_dir}/EMERGENCY_STOP",
        )
        execution_engine = build_execution_engine(cfg, pipeline.connector, symbol)

        snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)

        equity = initial_equity
        open_pos: Optional[OpenPosition] = None
        trades: List[dict] = []
        equity_curve: List[tuple] = []
        decisions: List[dict] = []
        traces: Dict[int, dict] = {}
        states: Dict[int, SystemState] = {}

        for snapshot in snapshots:
            started = time.perf_counter()
            state = pipeline.run_live_bar(snapshot)
            decision_latency_ms = (time.perf_counter() - started) * 1000
            if state is None:
                continue
            risk_layer.store.heartbeat()

            risk_decision = None
            execution_record = None
            execution_failed = False
            execution_error = ""
            stage_timings_ms: Dict[str, float] = {}
            realized_pnl_this_bar = 0.0

            # 1. Exits first -- never gated by the Risk Layer.
            if open_pos is not None:
                exit_info = check_position_exit(open_pos, state.market_snapshot, fee_pct)
                if exit_info is not None:
                    risk_layer.store.record_trade_result(
                        exit_info["realized_pnl_pct"], exit_info["realized_pnl_usdt"],
                    )
                    equity += exit_info["realized_pnl_usdt"]
                    drawdown = risk_layer.store.update_equity(equity)
                    if drawdown >= risk_layer.breaker_cfg.get("max_drawdown_pct", 8.0):
                        risk_layer.store.trip("max_drawdown", f"drawdown={drawdown:.2f}%")
                    realized_pnl_this_bar = exit_info["realized_pnl_pct"]
                    trades.append(_trade_row(open_pos, state.timestamp, exit_info))
                    open_pos = None

            # 2. New entries -- only when nothing is open.
            max_concurrent = int(risk_layer.breaker_cfg.get("max_concurrent_positions", 1))
            if max_concurrent < 1:
                raise ValueError(
                    f"max_concurrent_positions={max_concurrent} is invalid. Must be >= 1."
                )
            position_is_open = max_concurrent == 1 and open_pos is not None

            # The pipeline always populates these on a bar it processed;
            # a None here means its contract changed under us.
            expert_outputs = state.expert_outputs
            regime_state = state.regime_state
            router_decision = state.router_decision
            feature_vector = state.feature_vector
            if (expert_outputs is None or regime_state is None
                    or router_decision is None or feature_vector is None):
                raise RuntimeError(
                    "pipeline returned an incomplete SystemState "
                    f"(timestamp={state.timestamp})"
                )

            if not position_is_open and open_pos is None:
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
                        # paper_sim does not raise, but paper_live and live
                        # do when the exchange is unreachable.
                        logger.warning("Execution failed: %s", exc)
                        risk_layer.store.record_execution_result(success=False)
                        if risk_layer.store.consecutive_execution_errors() >= \
                                risk_layer.breaker_cfg.get("consecutive_execution_errors_halt", 3):
                            risk_layer.store.trip(
                                "consecutive_execution_errors", "3 consecutive failures",
                            )
                        execution_record = None
                        execution_failed = True
                        execution_error = str(exc)
                    stage_timings_ms["execution_ms"] = (time.perf_counter() - t_exec) * 1000

                    if execution_record is not None and execution_record.status in ("filled", "partial"):
                        direction = 1 if execution_record.side == "BUY" else -1
                        labeling_cfg = cfg.get("labeling", {})
                        regime_key = regime_state.primary_regime.name
                        overrides = labeling_cfg.get("regime_specific_barriers", {}).get(regime_key, {})
                        k_tp = overrides.get("k_tp", labeling_cfg.get("k_tp", 1.5))
                        k_sl = overrides.get("k_sl", labeling_cfg.get("k_sl", 1.0))
                        tp_pct, sl_pct = compute_barrier_pcts(
                            barrier_volatility_pct(feature_vector), k_tp, k_sl,
                        )
                        risk_layer.record_trade(state.timestamp)
                        open_pos = OpenPosition(
                            trade_id=f"gui-{state.timestamp}",
                            entry_timestamp=state.timestamp,
                            entry_price=execution_record.fill_price,
                            direction=direction,
                            size_usdt=execution_record.filled_size,
                            take_profit_pct=tp_pct,
                            stop_loss_pct=sl_pct,
                            max_holding_bars=max_holding_bars,
                            fees_paid=execution_record.filled_size * (execution_record.fees_pct / 100),
                        )
                        open_pos.expert = router_decision.selected_expert
                        open_pos.regime = regime_state.primary_regime.value
                        open_pos.selection_confidence = router_decision.selection_confidence
                        open_pos.risk_score = risk_decision.risk_score

                state = dc_replace(state, execution_record=execution_record)

            state = dc_replace(state, realized_pnl=realized_pnl_this_bar)
            stage_timings_ms.update(pipeline.last_bar_timings)
            decisions.append(_decision_row(state, position_is_open, execution_failed,
                                       execution_error))
            traces[state.timestamp] = _trace_dict(
                state, decision_latency_ms, stage_timings_ms,
                position_is_open, execution_failed, execution_error,
            )
            states[state.timestamp] = state
            equity_curve.append((state.timestamp, equity))
            risk_layer.advance_bar()

        zone_events = _zone_frame(pipeline)
        open_trade = _open_trade_row(open_pos) if open_pos is not None else None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if open_trade is not None:
        trades.append(open_trade)

    bundle = SessionBundle(
        symbol=symbol,
        instrument=spec,
        base_timeframe=data_granularity,
        bars=_bars_frame(df_ohlcv),
        decisions=_frame(decisions, DECISION_COLUMNS),
        trades=_frame(trades, TRADE_COLUMNS),
        traces=traces,
        zones=zone_events,
        equity=_frame(
            [{"timestamp": ts, "equity": value} for ts, value in equity_curve], EQUITY_COLUMNS
        ),
        states=states,
        meta={
            "source": "replay",
            "source_label": "Replay (computed locally)",
            "synthetic": synthetic,
            "relaxed_liquidity": relax,
            "granularity": data_granularity,
            "initial_equity": initial_equity,
            "final_equity": equity,
            "n_trades": len(trades),
            "warnings": _replay_warnings(relax, synthetic),
        },
    )
    return bundle


def _trade_row(open_pos: OpenPosition, exit_timestamp: int, exit_info: dict) -> dict:
    """Normalise a closed position into the GUI's trade schema."""
    direction = open_pos.direction
    entry = open_pos.entry_price
    exit_price = exit_info["exit_price"]
    bars_held = open_pos.bars_held
    return {
        "trade_id": open_pos.trade_id,
        "entry_timestamp": open_pos.entry_timestamp,
        "exit_timestamp": exit_timestamp,
        "entry_price": entry,
        "exit_price": exit_price,
        "direction": direction,
        "expert": open_pos.expert,
        "regime": open_pos.regime,
        "size": open_pos.size_usdt,
        "pnl_pct": exit_info["realized_pnl_pct"],
        "pnl_usdt": exit_info["realized_pnl_usdt"],
        "fees_paid": open_pos.fees_paid,
        "slippage": 0.0,
        "exit_reason": exit_info["exit_reason"],
        "router_confidence": open_pos.selection_confidence,
        "risk_score": open_pos.risk_score,
        "mode": "paper_sim",
        "take_profit_pct": open_pos.take_profit_pct,
        "stop_loss_pct": open_pos.stop_loss_pct,
        "bars_held": bars_held,
    }


def _open_trade_row(open_pos: OpenPosition) -> dict:
    """A trade that is still open when the session ends.

    It has an entry but no exit: the GUI draws the entry marker and
    leaves the exit open, which is exactly what happened.
    """
    return {
        "trade_id": open_pos.trade_id,
        "entry_timestamp": open_pos.entry_timestamp,
        "exit_timestamp": None,
        "entry_price": open_pos.entry_price,
        "exit_price": None,
        "direction": open_pos.direction,
        "expert": open_pos.expert,
        "regime": open_pos.regime,
        "size": open_pos.size_usdt,
        "pnl_pct": None,
        "pnl_usdt": None,
        "fees_paid": open_pos.fees_paid,
        "slippage": 0.0,
        "exit_reason": "open",
        "router_confidence": open_pos.selection_confidence,
        "risk_score": open_pos.risk_score,
        "mode": "paper_sim",
        "take_profit_pct": open_pos.take_profit_pct,
        "stop_loss_pct": open_pos.stop_loss_pct,
        "bars_held": open_pos.bars_held,
    }


def _bars_frame(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    if df_ohlcv is None or df_ohlcv.empty:
        return empty_frame(["timestamp", "open", "high", "low", "close", "volume"])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    out = df_ohlcv.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols]
    out["timestamp"] = to_epoch_seconds(out["timestamp"])
    out = out.dropna(subset=["timestamp"])
    out["timestamp"] = out["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out.sort_values("timestamp").reset_index(drop=True)


def _frame(rows: List[dict], columns: List[str]) -> pd.DataFrame:
    """Build a frame with the given columns from a list of row dicts."""
    if not rows:
        return empty_frame(columns)
    frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[columns]
    return normalize_dtypes(frame).reset_index(drop=True)


def _zone_frame(pipeline: TradingPipeline) -> pd.DataFrame:
    """Zone lifecycle events collected during the replay."""
    events = list(getattr(pipeline.structure_engine, "lifecycle_events", []) or [])
    if not events:
        return empty_frame(ZONE_COLUMNS)
    frame = pd.DataFrame(events)
    for col in ZONE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[ZONE_COLUMNS].copy()
    if "timestamp" in out.columns:
        out["timestamp"] = to_epoch_seconds(out["timestamp"])
        out = out.dropna(subset=["timestamp"])
    return normalize_dtypes(out).reset_index(drop=True)


def _replay_warnings(relax: bool, synthetic: bool) -> List[str]:
    warnings: List[str] = []
    if synthetic:
        warnings.append(
            "ILLUSTRATIVE - synthetic data. Prices are generated, not a real market."
        )
    if relax:
        warnings.append(
            "Liquidity gates were RELAXED for this run. It measures pipeline "
            "behaviour, not production edge - do not read expectancy from it."
        )
    return warnings


#: Default epoch for the synthetic demo series (2026-01-01T00:00:00Z).
#: Fixed, not "now": a deterministic series is what lets the GUI's demo
#: mode, its tests and its screenshots all agree.
SYNTHETIC_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def synthetic_ohlcv(
    n_bars: int = 900,
    seed: int = 42,
    start: Optional[datetime] = None,
) -> pd.DataFrame:
    """Deterministic mixed-regime synthetic OHLCV for the GUI's demo mode.

    Mirrors ``scripts/full_stack_walk_forward_backtest.generate_synthetic_
    ohlcv`` so that the GUI's demo data looks like the backtest harness's.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    begin = start or SYNTHETIC_START
    timestamps = [(begin.timestamp() + i * 60) for i in range(n_bars)]
    price, rows = 50000.0, []
    segment = max(1, n_bars // 4)
    for i in range(n_bars):
        drift = 12.0 if i < segment else (-10.0 if i < 2 * segment else (2.0 if i < 3 * segment else -18.0))
        vol = 8.0 if i >= 3 * segment else 5.0
        price = max(price + rng.normal(drift, vol), 1000.0)
        open_ = price + rng.normal(0, 3)
        rows.append({
            "timestamp": int(timestamps[i]),
            "open": open_,
            "high": max(open_, price) + abs(rng.normal(0, vol * 0.8)),
            "low": min(open_, price) - abs(rng.normal(0, vol * 0.8)),
            "close": price,
            "volume": abs(rng.normal(120, 35)),
        })
    return pd.DataFrame(rows)
