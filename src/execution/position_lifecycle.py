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

Behaviour is unchanged: this is a pure move, not a redesign. ``scripts/
run.py`` re-exports all three names, so ``from run import OpenPosition``
keeps working.
"""
from dataclasses import dataclass
from typing import Optional

from ..connectors.data_source import CCXTConnector
from ..state.contracts import MarketSnapshot
from .execution_engine import ExecutionBackend, ExecutionEngine

__all__ = ["OpenPosition", "build_execution_engine", "check_position_exit"]


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
