"""
Normalized, UI-agnostic data model for the Trade Map GUI.

Every GUI surface (Streamlit app, standalone HTML export, CLI summary)
reads a :class:`SessionBundle`. A bundle is deliberately dumb: it holds
normalized frames and plain dicts, knows nothing about Plotly, Streamlit
or file layout. That keeps the interesting logic testable without a
browser and keeps the charting layer free to change shape.

Column vocabularies are fixed here so that the three data sources
(live artifacts, replay, backtest report) all produce the same frame
schema and the charts never branch on where the data came from.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from ..diagnostics.decision_trace import TERMINAL_STAGES  # noqa: F401  (re-export)
from ..state.contracts import SystemState
from .instruments import InstrumentSpec

# ``TERMINAL_STAGES`` above is re-exported from
# :mod:`src.diagnostics.decision_trace` so that the TUI dashboard, the
# diagnostics CLI and this GUI all speak ONE vocabulary -- and so that a
# stage the GUI renders can be validated against the canonical list.

BAR_COLUMNS: List[str] = ["timestamp", "open", "high", "low", "close", "volume"]

DECISION_COLUMNS: List[str] = [
    "timestamp",
    "symbol",
    "regime",
    "regime_confidence",
    "expert",
    "opportunity_score",
    "router_confidence",
    "risk_score",
    "risk_approved",
    "decision",
    "terminal_stage",
    "terminal_reason",
]

TRADE_COLUMNS: List[str] = [
    "trade_id",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "direction",
    "expert",
    "regime",
    "size",
    "pnl_pct",
    "pnl_usdt",
    "fees_paid",
    "slippage",
    "exit_reason",
    "router_confidence",
    "risk_score",
    "mode",
    "take_profit_pct",
    "stop_loss_pct",
    "bars_held",
]

ZONE_COLUMNS: List[str] = [
    "zone_id",
    "event_type",
    "timestamp",
    "bar_index",
    "price_at_event",
    "zone_strength_at_event",
    "related_zone_id",
    "regime_at_event",
]

EQUITY_COLUMNS: List[str] = ["timestamp", "equity"]

#: Columns that must be numeric for charting and aggregation to work.
#: Readers coerce these so the charting layer never has to deal with
#: ``object`` columns full of ``pd.NA``.
NUMERIC_BAR_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]

NUMERIC_TRADE_COLUMNS: List[str] = [
    "entry_timestamp", "exit_timestamp", "entry_price", "exit_price", "direction",
    "size", "pnl_pct", "pnl_usdt", "fees_paid", "slippage", "router_confidence",
    "risk_score", "take_profit_pct", "stop_loss_pct", "bars_held",
]

NUMERIC_DECISION_COLUMNS: List[str] = [
    "regime_confidence", "opportunity_score", "router_confidence", "risk_score",
]

NUMERIC_ZONE_COLUMNS: List[str] = [
    "price_at_event", "zone_strength_at_event",
]

NUMERIC_EQUITY_COLUMNS: List[str] = ["equity"]

#: Timestamp-like columns that are always present on a valid row, and
#: are therefore kept as int64 (unix seconds) rather than float64.
INT_COLUMNS: List[str] = ["timestamp", "entry_timestamp", "bar_index"]

#: Boolean columns, kept as nullable booleans so that "not reported" and
#: "reported false" stay distinguishable.
BOOL_COLUMNS: List[str] = ["risk_approved"]


def coerce_numeric(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce the given columns to float64, turning junk into NaN."""
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def coerce_booleans(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce the given columns to nullable booleans."""
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype("boolean")
    return out


def cast_int_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Cast always-present timestamp-like columns to int64.

    Rows whose value is missing are dropped rather than silently turned
    into a bogus epoch second.
    """
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.isna().any():
            out = out.loc[numeric.notna()].copy()
            numeric = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric.astype("int64")
    return out


def normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Give a frame the GUI's canonical dtypes.

    Every producer (the journal reader, the decisions reader, the replay)
    funnels through this, so a live frame and a replayed frame can be
    concatenated without pandas upcasting a price column to ``object``.

    String columns become the nullable ``string`` dtype, which is the
    same in pandas 2 and pandas 3 (pandas 3 would otherwise infer
    ``str`` for a populated frame and ``object`` for an empty one, so
    the two producers would disagree).
    """
    out = frame.copy()
    for col in (NUMERIC_BAR_COLUMNS + NUMERIC_TRADE_COLUMNS
                + NUMERIC_DECISION_COLUMNS + NUMERIC_ZONE_COLUMNS
                + NUMERIC_EQUITY_COLUMNS):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = coerce_booleans(out, BOOL_COLUMNS)
    out = cast_int_columns(out, INT_COLUMNS)
    handled = set(INT_COLUMNS) | set(BOOL_COLUMNS)
    handled |= set(NUMERIC_BAR_COLUMNS) | set(NUMERIC_TRADE_COLUMNS)
    handled |= set(NUMERIC_DECISION_COLUMNS) | set(NUMERIC_ZONE_COLUMNS)
    handled |= set(NUMERIC_EQUITY_COLUMNS)
    for col in out.columns:
        if col in handled:
            continue
        out[col] = out[col].astype("string")
    return out


def empty_frame(columns: List[str]) -> pd.DataFrame:
    """An empty frame with the given columns and the GUI's canonical dtypes.

    The dtypes match the populated frames exactly (see
    :func:`normalize_dtypes`), so concatenating an empty frame with a
    live one does not silently upcast a price column to ``object``.
    """
    df = pd.DataFrame(index=pd.RangeIndex(0))
    for col in columns:
        if col in INT_COLUMNS:
            df[col] = pd.Series(dtype="int64")
        elif col in BOOL_COLUMNS:
            df[col] = pd.Series(dtype="boolean")
        elif col in (
            NUMERIC_BAR_COLUMNS + NUMERIC_TRADE_COLUMNS + NUMERIC_DECISION_COLUMNS
            + NUMERIC_ZONE_COLUMNS + NUMERIC_EQUITY_COLUMNS
        ):
            df[col] = pd.Series(dtype="float64")
        else:
            df[col] = pd.Series(dtype="string")
    return df[list(columns)]


@dataclass
class SessionBundle:
    """Everything the GUI needs to draw one instrument's session.

    Attributes:
        symbol: the traded symbol as the pipeline knows it ("BTC/USDT").
        instrument: contract metadata (asset class, strike, expiry, ...).
        base_timeframe: granularity the bars are stored at ("1m").
        bars: OHLCV frame, one row per base bar, ``timestamp`` ascending.
        decisions: one row per bar the system processed.
        trades: normalized trade records.
        traces: raw decision traces keyed by bar timestamp.
        zones: zone lifecycle events.
        equity: equity curve (empty when the source has no equity).
        states: full ``SystemState`` per bar, when the source has them.
        meta: provenance -- source kind, labels, config hash, warnings.
    """

    symbol: str
    instrument: InstrumentSpec
    base_timeframe: str = "1m"
    bars: pd.DataFrame = field(default_factory=lambda: empty_frame(BAR_COLUMNS))
    decisions: pd.DataFrame = field(default_factory=lambda: empty_frame(DECISION_COLUMNS))
    trades: pd.DataFrame = field(default_factory=lambda: empty_frame(TRADE_COLUMNS))
    traces: Dict[int, dict] = field(default_factory=dict)
    zones: pd.DataFrame = field(default_factory=lambda: empty_frame(ZONE_COLUMNS))
    equity: pd.DataFrame = field(default_factory=lambda: empty_frame(EQUITY_COLUMNS))
    states: Dict[int, SystemState] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Cached index over ``decisions`` (see ``decision_by_timestamp``).
    _decision_index: Dict[int, dict] = field(default_factory=dict, init=False, repr=False)
    _decision_index_key: Any = field(default=None, init=False, repr=False)

    # -- derived views -------------------------------------------------

    @property
    def is_synthetic(self) -> bool:
        """True when the underlying bars are generated, not real market data."""
        return bool(self.meta.get("synthetic", False))

    @property
    def source_label(self) -> str:
        return str(self.meta.get("source_label", self.meta.get("source", "unknown")))

    @property
    def warnings(self) -> List[str]:
        return list(self.meta.get("warnings", []))

    @property
    def decision_by_timestamp(self) -> Dict[int, dict]:
        """Bar timestamp -> decision row (as a plain dict).

        Cached: the charting layer asks for this once per bar, and
        rebuilding the whole index on every access made a 20k-bar
        session quadratic.
        """
        if self.decisions.empty:
            return {}
        cache_key = (id(self.decisions), len(self.decisions))
        if self._decision_index_key != cache_key:
            out: Dict[int, dict] = {}
            for row in self.decisions.to_dict("records"):
                ts = row.get("timestamp")
                if ts is None or pd.isna(ts):
                    continue
                out[int(ts)] = row
            self._decision_index = out
            self._decision_index_key = cache_key
        return self._decision_index

    @property
    def close_by_timestamp(self) -> Dict[int, float]:
        """Bar timestamp -> close price, for filling in missing prices."""
        if self.bars.empty:
            return {}
        return {
            int(ts): float(close)
            for ts, close in zip(self.bars["timestamp"], self.bars["close"], strict=True)
            if pd.notna(close)
        }

    def has_content(self) -> bool:
        """True when there is anything at all worth drawing."""
        return not self.bars.empty or not self.trades.empty or not self.decisions.empty
