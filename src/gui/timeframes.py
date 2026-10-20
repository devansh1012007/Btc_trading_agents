"""
Timeframe parsing and OHLCV resampling.

The system stores bars at one base granularity (``system.data_granularity``
-- "10s", "30s", "1m" or "5m"). The GUI shows the same session at many
timeframes at once, so it needs to derive higher timeframes from the base
bars rather than asking the exchange for each one separately.

Two rules keep that honest:

* **Causality.** A resampled bucket labelled ``T`` aggregates only bars
  whose timestamps fall in ``[T, T + timeframe)``. No bucket ever contains
  information from a later bar, so a 15m candle drawn from 1m bars is
  exactly what the exchange would have printed.
* **Determinism.** Buckets are left-labelled and aligned to the epoch /
  UTC day, so the same bars always produce the same higher timeframe
  regardless of where the slice starts.

One authoritative bucket function (:func:`bucket_start_series`) serves both
the resampler and the aligner, so a bar can never be placed in one
bucket by the resampler and a different one by the aligner.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.time_utils import epoch_seconds

#: Timeframes the system can actually STORE as base bars.
BASE_TIMEFRAMES: Dict[str, int] = {
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "5m": 300,
}

#: Timeframes the GUI can DISPLAY, smallest first.
DISPLAY_TIMEFRAMES: Dict[str, int] = {
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1D": 86400,
    "1W": 604800,
    # Month buckets are calendar-based; 2592000 is a NOMINAL month
    # length used only for ordering and display -- never for arithmetic.
    "1M": 2592000,
}

#: pandas resample rule per display timeframe.
_PANDAS_RULES: Dict[str, str] = {
    "10s": "10s",
    "30s": "30s",
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1D": "1D",
    "1W": "W-MON",
    "1M": "MS",
}

#: Seconds per bucket for the tick-like (fixed-length) timeframes.
_TICK_SECONDS: Dict[str, int] = {
    tf: secs for tf, secs in DISPLAY_TIMEFRAMES.items() if tf in _PANDAS_RULES and tf not in ("1D", "1W", "1M")
}

#: Seconds from the Unix epoch back to Monday 1969-12-29 00:00 UTC, the
#: Monday that anchors every ``W-MON`` week. Weeks are aligned to it, not
#: to the epoch, because the epoch itself was a Thursday.
_WEEK_ANCHOR = 259_200

class UnknownTimeframeError(ValueError):
    """Raised for a timeframe string the GUI does not know how to build."""


def parse_timeframe(timeframe: str) -> int:
    """Seconds in one bar of ``timeframe``.

    Raises :class:`UnknownTimeframeError` for anything the GUI does not
    support -- better a loud error than a silently wrong resample.
    """
    key = str(timeframe).strip()
    if key in DISPLAY_TIMEFRAMES:
        return DISPLAY_TIMEFRAMES[key]
    raise UnknownTimeframeError(
        f"Unknown timeframe {timeframe!r}; expected one of {sorted(DISPLAY_TIMEFRAMES)}"
    )


def is_valid_timeframe(timeframe: str) -> bool:
    return str(timeframe).strip() in DISPLAY_TIMEFRAMES


def base_timeframe_seconds(base_timeframe: str = "1m") -> int:
    """Seconds in one bar of the base granularity.

    Falls back to one minute for a granularity the system cannot store.
    """
    if base_timeframe in BASE_TIMEFRAMES:
        return BASE_TIMEFRAMES[base_timeframe]
    try:
        return parse_timeframe(base_timeframe)
    except UnknownTimeframeError:
        return 60


def display_timeframes(base_timeframe: str = "1m") -> List[str]:
    """Display timeframes that are at or above the base granularity.

    A 30s base cannot produce a 1m candle without inventing data, so
    finer timeframes are filtered out rather than approximated.
    """
    base_seconds = base_timeframe_seconds(base_timeframe)
    return [tf for tf, secs in DISPLAY_TIMEFRAMES.items() if secs >= base_seconds]


def bucket_start_series(timestamps: "np.ndarray", timeframe: str) -> "np.ndarray":
    """Epoch second at which the bucket containing each timestamp starts.

    The single authority on bucket boundaries: the resampler and the
    aligner both call it, so a bar can never land in two different
    buckets depending on which code path placed it.

    Tick-like timeframes are epoch-aligned, ``1D`` is UTC-midnight
    aligned, ``1W`` is Monday-aligned (the exchange's weekly candle) and
    ``1M`` is calendar-month aligned.
    """
    key = str(timeframe).strip()
    if key not in _PANDAS_RULES:
        raise UnknownTimeframeError(
            f"Unknown timeframe {timeframe!r}; expected one of {sorted(DISPLAY_TIMEFRAMES)}"
        )

    ts = np.asarray(timestamps, dtype="int64")
    if ts.size == 0:
        return ts.copy()

    if key in _TICK_SECONDS:
        secs = _TICK_SECONDS[key]
        return (ts // secs) * secs
    if key == "1D":
        return (ts // 86_400) * 86_400
    if key == "1W":
        return ((ts + _WEEK_ANCHOR) // 604_800) * 604_800 - _WEEK_ANCHOR
    if key == "1M":
        naive = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_localize(None)
        month_start = naive.dt.to_period("M").dt.to_timestamp()
        return month_start.dt.as_unit("s").astype("int64").to_numpy()

    raise UnknownTimeframeError(f"Unsupported timeframe {timeframe!r}")


def bucket_start(timestamp: int, timeframe: str) -> int:
    """Unix timestamp of the bucket that contains ``timestamp``.

    Used to align a trade, a decision or a zone event to the candle it
    belongs to on a higher timeframe.
    """
    return int(bucket_start_series(np.asarray([int(timestamp)]), timeframe)[0])


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Epoch seconds for a series of timestamps.

    Accepts integer epoch seconds / milliseconds / microseconds /
    nanoseconds, numeric strings, datetime columns and ISO date strings.
    Returns a float series (NaN where the value could not be parsed) so
    the caller can drop the bad rows before casting.

    The implementation lives in :mod:`src.utils.time_utils` so that the
    deployment loop and the GUI normalise timestamps the SAME way -- two
    copies of a unit heuristic is two chances to disagree about when the
    system traded.
    """
    return epoch_seconds(series)


def resample_ohlcv(
    bars: pd.DataFrame,
    timeframe: str,
    base_timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate base bars into ``timeframe`` bars.

    The input frame must carry ``timestamp``, ``open``, ``high``, ``low``,
    ``close`` and ``volume`` columns. The output has the same columns,
    left-labelled by bucket start, with empty buckets dropped.

    A bucket's open is the open of the FIRST bar in it and its close the
    close of the LAST bar -- not the first/last *non-null* value, which
    would pull a later bar's price into the bucket's open.

    ``base_timeframe`` guards against building a candle finer than the
    data actually is (a 1m bar cannot become a 10s candle without
    inventing data).
    """
    key = str(timeframe).strip()
    if key not in _PANDAS_RULES:
        raise UnknownTimeframeError(
            f"Unknown timeframe {timeframe!r}; expected one of {sorted(DISPLAY_TIMEFRAMES)}"
        )
    if base_timeframe is not None:
        base_seconds = base_timeframe_seconds(base_timeframe)
        if parse_timeframe(key) < base_seconds:
            raise UnknownTimeframeError(
                f"Cannot build {key} candles from {base_timeframe} bars: "
                f"{key} is finer than the base granularity"
            )

    if bars.empty:
        return bars.copy()

    df = bars.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    labels = bucket_start_series(df["timestamp"].to_numpy(), key)
    df = df.assign(_bucket=labels)
    grouped = df.groupby("_bucket", sort=True)

    # Positional first/last row per bucket: GroupBy.first()/last() skip
    # nulls, which would move a bucket's open (or close) onto a later bar.
    positions = grouped.indices
    first_idx = np.asarray([positions[label][0] for label in grouped.groups], dtype="int64")
    last_idx = np.asarray([positions[label][-1] for label in grouped.groups], dtype="int64")
    bucket_labels = np.asarray(list(grouped.groups), dtype="int64")

    opens = df["open"].to_numpy()[first_idx]
    closes = df["close"].to_numpy()[last_idx]

    out = pd.DataFrame(
        {
            "timestamp": bucket_labels,
            "open": opens,
            "high": grouped["high"].max().to_numpy(),
            "low": grouped["low"].min().to_numpy(),
            "close": closes,
            "volume": grouped["volume"].sum().to_numpy(),
        }
    )
    out = out.dropna(subset=["open", "close"], how="all")
    return out[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def align_to_timeframe(
    frame: pd.DataFrame,
    timeframe: str,
    timestamp_column: str = "timestamp",
    how: str = "first",
) -> pd.DataFrame:
    """Attach each row of ``frame`` to the higher-timeframe bucket it falls in.

    ``how="first"`` (the default) keeps the value that was current at the
    bucket's OPEN -- the only causal choice when the value is painted
    across the whole bucket, because a later bar's value was not known
    at the bucket's start. ``how="last"`` keeps the value at the
    bucket's END and is only appropriate for cumulative quantities (an
    equity balance, for instance) that are drawn as a step.

    The whole row is kept, not each column independently: a per-column
    "first non-null" would mix values from different bars into one
    Frankenstein row.
    """
    if frame.empty:
        return frame.copy()

    if how not in ("first", "last"):
        raise ValueError(f"how must be 'first' or 'last', got {how!r}")
    if timestamp_column not in frame.columns:
        raise KeyError(f"{timestamp_column!r} is not a column of the frame")

    stamps = pd.to_numeric(frame[timestamp_column], errors="coerce")
    if stamps.isna().any():
        missing = int(stamps.isna().sum())
        raise ValueError(
            f"{missing} row(s) have a missing or non-numeric {timestamp_column!r}; "
            "a bar without a timestamp cannot be placed in a bucket"
        )

    out = frame.copy()
    out["_bucket"] = bucket_start_series(stamps.to_numpy(dtype="int64"), timeframe)
    out = out.sort_values(timestamp_column, kind="stable")
    out = out.drop_duplicates("_bucket", keep="first" if how == "first" else "last")

    result = out.drop(columns=["_bucket"]).copy()
    result[timestamp_column] = out["_bucket"].to_numpy()
    return result[list(frame.columns)].reset_index(drop=True)


def timeframe_seconds(timeframe: str) -> int:
    """Alias for :func:`parse_timeframe`, kept for readability at call sites."""
    return parse_timeframe(timeframe)


def format_timeframe(timeframe: str) -> str:
    """Canonical label for a timeframe string (normalising case)."""
    key = str(timeframe).strip()
    if key in DISPLAY_TIMEFRAMES:
        return key
    raise UnknownTimeframeError(f"Unknown timeframe {timeframe!r}")


def seconds_to_timeframe(seconds: int) -> Optional[str]:
    """Reverse lookup: the canonical timeframe string for a bar length."""
    for tf, secs in DISPLAY_TIMEFRAMES.items():
        if secs == int(seconds):
            return tf
    return None
