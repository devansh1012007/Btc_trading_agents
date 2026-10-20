"""
Epoch-timestamp normalisation shared by the runtime and the GUI.

The system receives timestamps in several shapes: integer epoch
seconds, integer milliseconds, integer microseconds, integer
nanoseconds, numeric strings, ISO date strings and datetime columns.
The one thing they must never become is a wrong instant.

Two traps are avoided here:

* ``astype("int64")`` on a datetime column. pandas 3 resolves that to
  MICROSECONDS, so the same expression silently means something
  different across pandas versions. ``as_unit("s")`` is explicit.
* A column-wide guess at the epoch unit. The unit is read off the
  magnitude of each value, so a column that mixes units is handled
  correctly instead of being scaled by a single guess.
"""
from typing import Optional

import numpy as np
import pandas as pd

#: Magnitude thresholds used to tell epoch seconds / millis / micros /
#: nanos apart, applied in order. Each threshold is the point past
#: which the unit is unambiguous (1e11 seconds is the year 5138, and
#: 1e14 milliseconds / 1e17 microseconds are the same instant). A
#: nanosecond value passes both thresholds and is divided by 1e6 and
#: then by 1e3; a microsecond value passes only the first.
_UNIT_THRESHOLDS = ((1e14, 1e6), (1e11, 1e3))

#: Smallest plausible unix timestamp (2001-09-09) -- below this an
#: integer column is a row counter, not a time.
_MIN_EPOCH_SECONDS = 1_000_000_000


def scale_epoch_units(values: "np.ndarray") -> "np.ndarray":
    """Normalise epoch values to seconds, detecting the unit per value."""
    out = np.asarray(values, dtype="float64").copy()
    for threshold, divisor in _UNIT_THRESHOLDS:
        mask = np.abs(out) >= threshold
        if mask.any():
            out[mask] = out[mask] / divisor
    return out


def _datetime_epoch_seconds(series: pd.Series) -> "np.ndarray":
    """Epoch seconds for a datetime-like series (NaT becomes NaN)."""
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        seconds = parsed.dt.as_unit("s").astype("int64")
    else:
        seconds = parsed.astype("datetime64[s]").astype("int64")
    values = seconds.to_numpy(dtype="float64")
    values[parsed.isna().to_numpy()] = np.nan
    return values


def epoch_seconds(series: pd.Series) -> pd.Series:
    """Epoch seconds for a series of timestamps.

    Accepts integer epoch seconds / milliseconds / microseconds /
    nanoseconds, numeric strings, datetime columns and ISO date strings.
    Returns a float series (NaN where the value could not be parsed) so
    the caller can drop the bad rows before casting.
    """
    if pd.api.types.is_bool_dtype(series):
        # A boolean is never a timestamp; refuse rather than invent 1970.
        return pd.Series(np.nan, index=series.index, dtype="float64")

    if not pd.api.types.is_datetime64_any_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            out = pd.Series(np.nan, index=series.index, dtype="float64")
            is_numeric = numeric.notna()
            out.loc[is_numeric] = scale_epoch_units(
                numeric.loc[is_numeric].to_numpy(dtype="float64")
            )
            rest = series.loc[~is_numeric]
            if len(rest) and rest.notna().any():
                out.loc[~is_numeric] = _datetime_epoch_seconds(rest)
            return out

    return pd.Series(_datetime_epoch_seconds(series), index=series.index, dtype="float64")


def epoch_unit_label(median: float) -> str:
    """The epoch unit a median magnitude implies (``s``/``ms``/``us``/``ns``)."""
    if not np.isfinite(median):
        return "unknown"
    magnitude = abs(float(median))
    if magnitude >= 1e17:
        return "ns"
    if magnitude >= 1e14:
        return "us"
    if magnitude >= 1e11:
        return "ms"
    return "s"


def describe_epoch_unit(series: pd.Series) -> Optional[str]:
    """Describe the epoch unit of a numeric timestamp column, if any."""
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    if pd.isna(median):
        return None
    return epoch_unit_label(float(median))
