"""
Session sources for the Trade Map: where a :class:`SessionBundle` comes from.

Three sources, tried in this order (``artifacts`` first, ``replay`` as
the fallback, ``synthetic`` as the always-available demo):

``artifacts``
    What a real run left on disk -- the trade journal, the section-8
    decision log, the decision traces, the zone log and the raw OHLCV.
    Read-only. Nothing is recomputed.
``replay``
    The symbol's raw OHLCV replayed through the real pipeline
    (:func:`src.gui.replay.replay_session`).
``synthetic``
    The deterministic demo series, replayed. Always labelled ILLUSTRATIVE.

This module imports neither Streamlit nor Plotly: the app and the CLI
share it, and the tests drive it headless.
"""
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .artifacts import (
    load_symbol_ohlcv,
    read_decision_traces,
    read_decisions_jsonl,
    read_trade_journal,
    read_zone_events,
)
from .instruments import resolve_instrument
from .models import BAR_COLUMNS, EQUITY_COLUMNS, SessionBundle, empty_frame, normalize_dtypes
from .timeframes import BASE_TIMEFRAMES

logger = logging.getLogger(__name__)

SOURCES = ("artifacts", "replay", "synthetic")

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: A session whose last bar is older than this is flagged as stale.
STALE_AFTER_SECONDS = 6 * 3600

DEFAULT_INITIAL_EQUITY = 10_000.0


class SourceUnavailable(Exception):
    """The requested source has nothing to show (the message says why)."""


def load_config(env: str = "paper") -> dict:
    """The layered deployment config, resolved from the repo root."""
    from ..utils.config_helpers import load_layered_config

    return load_layered_config(env, config_dir=str(_REPO_ROOT / "config"))


def infer_base_timeframe(bars: pd.DataFrame, default: str = "1m") -> str:
    """The storable granularity closest to the bars' median spacing."""
    if bars is None or len(bars) < 3:
        return default
    step = float(np.median(np.diff(bars["timestamp"].to_numpy(dtype="int64"))))
    name, seconds = min(BASE_TIMEFRAMES.items(), key=lambda kv: abs(kv[1] - step))
    return name if abs(seconds - step) <= max(1.0, seconds * 0.25) else default


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def artifacts_available(symbol: str) -> bool:
    """True when there is something to draw from disk for ``symbol``."""
    return not load_symbol_ohlcv(symbol).empty and (
        not read_trade_journal().empty or not read_decisions_jsonl().empty
    )


def _derived_equity(bars: pd.DataFrame, trades: pd.DataFrame, initial: float) -> pd.DataFrame:
    """Equity per bar from the journal: initial + realised PnL of exited trades."""
    if bars.empty:
        return empty_frame(EQUITY_COLUMNS)
    stamps = bars["timestamp"].to_numpy(dtype="int64")
    closed = trades.dropna(subset=["exit_timestamp", "pnl_usdt"]) if not trades.empty else trades
    equity = np.full(len(stamps), float(initial))
    if not closed.empty:
        order = closed.sort_values("exit_timestamp")
        exit_ts = order["exit_timestamp"].to_numpy(dtype="int64")
        cumulative = order["pnl_usdt"].to_numpy(dtype="float64").cumsum()
        idx = np.searchsorted(exit_ts, stamps, side="right")
        equity = initial + np.where(idx > 0, cumulative[np.maximum(idx - 1, 0)], 0.0)
    return normalize_dtypes(pd.DataFrame({"timestamp": stamps, "equity": equity}))


def artifacts_session(
    symbol: str, config: Optional[dict] = None, initial_equity: float = DEFAULT_INITIAL_EQUITY,
) -> SessionBundle:
    """Build a session from what a run left on disk (read-only)."""
    config = config or load_config()
    bars = load_symbol_ohlcv(symbol)
    if bars.empty:
        raise SourceUnavailable(f"no OHLCV file for {symbol} under data/raw")

    trades = read_trade_journal()
    decisions = read_decisions_jsonl()
    if not decisions.empty and decisions["symbol"].notna().any():
        decisions = decisions[decisions["symbol"] == symbol].reset_index(drop=True)
    if trades.empty and decisions.empty:
        raise SourceUnavailable("the run left no trades and no decision log yet")

    lo, hi = int(bars["timestamp"].iloc[0]), int(bars["timestamp"].iloc[-1])
    if not trades.empty:
        in_range = trades["entry_timestamp"].between(lo, hi)
        trades = trades[in_range].reset_index(drop=True)
    if not decisions.empty:
        decisions = decisions[decisions["timestamp"].between(lo, hi)].reset_index(drop=True)

    base = infer_base_timeframe(bars, config.get("system", {}).get("data_granularity", "1m"))
    warnings: List[str] = [
        "Equity is DERIVED from the trade journal (initial equity + realised PnL); "
        "the runtime does not write an equity curve.",
    ]
    age = time.time() - hi
    if age > STALE_AFTER_SECONDS:
        warnings.insert(0, f"STALE session - the last bar is {age / 3600:.1f} h old.")

    return SessionBundle(
        symbol=symbol, instrument=resolve_instrument(symbol), base_timeframe=base,
        bars=bars[BAR_COLUMNS].copy(), decisions=decisions, trades=trades,
        traces=read_decision_traces(), zones=read_zone_events(),
        equity=_derived_equity(bars, trades, initial_equity),
        meta={
            "source": "artifacts", "source_label": "Live artifacts (read-only)",
            "synthetic": False, "relaxed_liquidity": False, "granularity": base,
            "initial_equity": initial_equity, "n_trades": len(trades),
            "has_traces": True, "warnings": warnings,
        },
    )


# ---------------------------------------------------------------------------
# replay / synthetic
# ---------------------------------------------------------------------------

def replay_source_session(
    symbol: str, config: Optional[dict] = None, *, max_bars: Optional[int] = 1500,
    relax: bool = False, initial_equity: float = DEFAULT_INITIAL_EQUITY,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SessionBundle:
    """Replay the symbol's raw OHLCV through the real pipeline."""
    from .replay import replay_session

    config = config or load_config()
    bars = load_symbol_ohlcv(symbol)
    if bars.empty:
        raise SourceUnavailable(f"no OHLCV file for {symbol} under data/raw to replay")
    if max_bars:
        bars = bars.tail(int(max_bars)).reset_index(drop=True)
    return replay_session(
        bars, symbol, config, initial_equity=initial_equity, relax=relax, progress=progress,
    )


def synthetic_session(
    symbol: str = "BTC/USDT", config: Optional[dict] = None, *, n_bars: int = 900, seed: int = 42,
    relax: bool = True, initial_equity: float = DEFAULT_INITIAL_EQUITY,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SessionBundle:
    """The deterministic demo series, replayed (relaxed gates by default,
    because the shipped synthetic liquidity would reject every bar)."""
    from ..utils.synthetic import synthetic_ohlcv
    from .replay import replay_session

    config = config or load_config()
    return replay_session(
        synthetic_ohlcv(n_bars, seed=seed), symbol, config, initial_equity=initial_equity,
        relax=relax, synthetic=True, progress=progress,
    )


def load_session(source: str, symbol: str, **kwargs) -> SessionBundle:
    """Load a session from ``source`` (one of :data:`SOURCES`)."""
    if source == "artifacts":
        return artifacts_session(symbol, **{k: v for k, v in kwargs.items()
                                            if k in ("config", "initial_equity")})
    if source == "replay":
        return replay_source_session(symbol, **{k: v for k, v in kwargs.items() if k != "n_bars" and k != "seed"})
    if source == "synthetic":
        return synthetic_session(symbol, **{k: v for k, v in kwargs.items() if k != "max_bars"})
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def auto_source(symbol: str) -> str:
    """First usable source in order: artifacts -> replay -> synthetic."""
    if artifacts_available(symbol):
        return "artifacts"
    if not load_symbol_ohlcv(symbol).empty:
        return "replay"
    return "synthetic"


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize_session(bundle: SessionBundle) -> Dict[str, object]:
    """Headline numbers for the app's metric strip and the CLI's stdout."""
    trades = bundle.trades
    closed = trades[trades["exit_reason"].ne("open").fillna(True).astype(bool)] if not trades.empty else trades
    closed = closed.dropna(subset=["pnl_pct"]) if not closed.empty else closed

    n_closed = len(closed)
    wins = int((closed["pnl_usdt"] > 0).sum()) if n_closed else 0
    gross_win = float(closed.loc[closed["pnl_usdt"] > 0, "pnl_usdt"].sum()) if n_closed else 0.0
    gross_loss = float(-closed.loc[closed["pnl_usdt"] < 0, "pnl_usdt"].sum()) if n_closed else 0.0

    max_dd = 0.0
    if not bundle.equity.empty:
        eq = bundle.equity["equity"].to_numpy(dtype="float64")
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(np.where(peak > 0, (peak - eq) / peak * 100.0, 0.0)))

    funnel: List[Dict[str, object]] = []
    if not bundle.decisions.empty:
        from ..diagnostics.decision_trace import TERMINAL_STAGES, stage_label
        counts = bundle.decisions["terminal_stage"].value_counts()
        total = int(counts.sum())
        for stage in TERMINAL_STAGES:
            if stage in counts.index:
                funnel.append({"stage": stage, "label": stage_label(stage),
                               "bars": int(counts[stage]), "pct": float(counts[stage]) / total * 100.0})

    return {
        "n_bars": len(bundle.bars),
        "n_decisions": len(bundle.decisions),
        "n_trades": len(trades),
        "n_closed": n_closed,
        "n_open": len(trades) - n_closed if not trades.empty else 0,
        "win_rate_pct": (wins / n_closed * 100.0) if n_closed else None,
        "expectancy_pct": float(closed["pnl_pct"].mean()) if n_closed else None,
        "total_pnl_usdt": float(closed["pnl_usdt"].sum()) if n_closed else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_drawdown_pct": max_dd,
        "funnel": funnel,
    }
