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
#: are therefore kept as int64 (unix seconds) rather than float64. A
#: missing value is a producer bug the READER must resolve (drop or
#: repair) before normalising: ``normalize_dtypes`` raises rather than
#: silently losing the row.
INT_COLUMNS: List[str] = ["timestamp", "entry_timestamp"]

#: Integer columns that may legitimately be missing. Kept as the nullable
#: ``Int64`` so an absent value stays absent instead of dropping the row
#: (a zone event without a bar index is still drawable from its time).
NULLABLE_INT_COLUMNS: List[str] = ["bar_index"]

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

    Never drops a row. A missing value cannot be represented in int64, so
    it raises: deciding what to do with a row that has no timestamp is the
    READER's job (``read_trade_journal`` and ``read_decisions_jsonl``
    already drop them), not a side effect of a dtype cast.
    """
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        missing = int(numeric.isna().sum())
        if missing:
            raise ValueError(
                f"{missing} row(s) have a missing or non-numeric {col!r}; drop or repair "
                f"them before normalising (normalize_dtypes never drops rows)"
            )
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
    for col in NULLABLE_INT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    handled = set(INT_COLUMNS) | set(NULLABLE_INT_COLUMNS) | set(BOOL_COLUMNS)
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
        elif col in NULLABLE_INT_COLUMNS:
            df[col] = pd.Series(dtype="Int64")
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

    def __setattr__(self, name: str, value: Any) -> None:
        # The lookup indexes below are derived from ``decisions`` and
        # ``bars``. Replacing either frame drops them, so a stale index
        # can never outlive the frame it was built from (no ``id()``
        # key, which Python is free to reuse). The frames are treated as
        # immutable once attached: mutate-in-place is not supported.
        object.__setattr__(self, name, value)
        if name in ("decisions", "bars"):
            self.__dict__.pop("_index_cache", None)

    def _index(self, key: str, build) -> Any:
        cache = self.__dict__.setdefault("_index_cache", {})
        if key not in cache:
            cache[key] = build()
        return cache[key]

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

        Built once per ``decisions`` frame: the charting layer asks for
        this once per bar, and rebuilding the index on every access made
        a 20k-bar session quadratic.
        """
        def build() -> Dict[int, dict]:
            out: Dict[int, dict] = {}
            for row in self.decisions.to_dict("records"):
                ts = row.get("timestamp")
                if ts is None or pd.isna(ts):
                    continue
                out[int(ts)] = row
            return out

        return self._index("decisions", build) if not self.decisions.empty else {}

    @property
    def close_by_timestamp(self) -> Dict[int, float]:
        """Bar timestamp -> close price, for filling in missing prices.

        Cached like :attr:`decision_by_timestamp` (rebuilding it per
        lookup cost 11 ms per call; at 20 000 bars that is minutes per
        chart pass).
        """
        def build() -> Dict[int, float]:
            valid = self.bars["close"].notna()
            stamps = self.bars.loc[valid, "timestamp"].to_numpy().tolist()
            closes = self.bars.loc[valid, "close"].to_numpy().tolist()
            return {int(t): float(c) for t, c in zip(stamps, closes, strict=True)}

        return self._index("closes", build) if not self.bars.empty else {}

    def has_content(self) -> bool:
        """True when there is anything at all worth drawing."""
        return not self.bars.empty or not self.trades.empty or not self.decisions.empty
