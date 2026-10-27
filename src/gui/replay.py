"""
Replay a session through the real pipeline so the GUI has something to draw.

This is the GUI's equivalent of ``scripts/run.py --mode historical``:
the same pipeline, the same risk layer, the same paper-sim execution and
the same per-bar block (``src.execution.position_lifecycle.process_bar``,
which the deployment loop calls too) -- but instead of writing artifacts
and forgetting them, it keeps every bar's ``SystemState`` in memory and
returns a :class:`~src.gui.models.SessionBundle`.

Two deliberate differences from the deployment loop:

* The risk layer's store is a NON-PERSISTENT in-memory one, so replaying
  a session can never trip a real circuit breaker, write to the
  production registry, or pay for SQLite commits it does not need.
* Execution is forced to ``paper_sim``. The GUI is a viewer; it must
  never be able to place an order on an exchange.
"""
import copy
import logging
import shutil
import tempfile
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..execution.position_lifecycle import (
    BarOutcome,
    OpenPosition,
    build_execution_engine,
    process_bar,
)
from ..labeling.hold_policy import alignment_settings
from ..layers.market_data import MarketDataLayer
from ..layers.risk_layer import RiskLayer, RiskStateStore
from ..pipeline import TradingPipeline
from ..state.contracts import SystemState
from ..utils.synthetic import SYNTHETIC_START, synthetic_ohlcv
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

__all__ = [
    "SYNTHETIC_START",
    "relax_liquidity",
    "replay_session",
    "synthetic_ohlcv",
]


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


def _decision_row(outcome: BarOutcome) -> dict:
    """One row of the per-bar decision frame.

    The terminal stage comes from the bar's trace -- classified once, by
    ``process_bar`` -- so this frame can never disagree with the trace
    the same bar carries.
    """
    state = outcome.state
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
        "terminal_stage": outcome.terminal_stage,
        "terminal_reason": outcome.terminal_reason,
    }


def replay_session(
    df_ohlcv: pd.DataFrame,
    symbol: str,
    config: dict,
    initial_equity: float = 10_000.0,
    relax: bool = False,
    instrument: Optional[InstrumentSpec] = None,
    synthetic: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SessionBundle:
    """Replay ``df_ohlcv`` through the full stack and return the session.

    The per-bar trading block is ``process_bar`` -- the SAME function the
    deployment loop calls -- so a replayed bar cannot classify, size or
    exit differently from a real one.

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
        progress: optional ``callback(done, total)`` for a progress bar.
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

    try:
        hold_policy, require_aligned = alignment_settings(cfg)
        risk_layer = RiskLayer(
            config=cfg.get("risk", {}),
            hold_policy=hold_policy, require_aligned_models=require_aligned,
            # A viewer needs the breaker's SEMANTICS, not its durability:
            # an in-memory store, so a replay neither pays for SQLite
            # commits nor can ever touch a file.
            store=RiskStateStore(persistent=False),
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

        def on_open(position: OpenPosition) -> str:
            return f"gui-{position.entry_timestamp}"

        def on_close(position: OpenPosition, exit_timestamp: int, exit_info: dict) -> None:
            trades.append(_trade_row(position, exit_timestamp, exit_info))

        total = len(snapshots)
        for done, snapshot in enumerate(snapshots, start=1):
            started = time.perf_counter()
            state = pipeline.run_live_bar(snapshot)
            decision_latency_ms = (time.perf_counter() - started) * 1000
            if state is None:
                continue

            outcome = process_bar(
                state, risk_layer=risk_layer, execution_engine=execution_engine,
                position=open_pos, equity=equity, fee_pct=fee_pct, config=cfg,
                decision_latency_ms=decision_latency_ms,
                pipeline_timings=pipeline.last_bar_timings,
                on_trade_open=on_open, on_trade_close=on_close,
            )
            open_pos, equity = outcome.position, outcome.equity

            ts = outcome.state.timestamp
            decisions.append(_decision_row(outcome))
            traces[ts] = asdict(outcome.trace)
            states[ts] = outcome.state
            equity_curve.append((ts, equity))
            if progress is not None and (done % 25 == 0 or done == total):
                progress(done, total)

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
