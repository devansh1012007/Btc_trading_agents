"""
Technical indicators for the GUI charts.

Every indicator here is **causal**: the value at bar ``t`` uses only bars
``<= t``. That matters more than usual in this repo -- the whole point
of the trading system is that nothing looks ahead, and a chart that
showed a future-dependent indicator would make the GUI lie about the
system's behaviour.

The implementations are deliberately plain (pandas rolling/ewm) rather
than pulling in TA-Lib: the repo already depends on pandas and numpy, and
these formulas are short enough to audit by eye.

**Seeding matches TradingView / TA-Lib.** pandas' ``ewm`` seeds an
average with the FIRST value it sees; TradingView and TA-Lib seed with
the *simple average of the first N values*. The difference is visible
(a 14-point gap on a 14-period RSI), and the whole point of these
charts is that a trader can compare them against the exchange's own
screen, so the standard seeding is used.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

#: Indicator columns added by :func:`compute_indicators`, in draw order.
INDICATOR_COLUMNS = [
    "ema_fast", "ema_slow", "sma_fast", "sma_slow",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr", "bb_mid", "bb_upper", "bb_lower",
    "vwap", "donchian_upper", "donchian_mid", "donchian_lower",
]


def _seeded_ewm(
    values: pd.Series,
    period: int,
    alpha: float,
    skip: int = 0,
) -> pd.Series:
    """Exponential average seeded with the SMA of the first ``period`` values.

    ``skip`` is the index of the first value that takes part in the seed
    (Wilder's RSI, for instance, has no change to average on the very
    first bar, so its seed window starts at index 1).

    The seed is placed so that the first output lands on the last value
    of the seed window -- the same index TradingView and TA-Lib use.
    """
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    n = len(values)
    end = skip + period
    if period < 1 or n < end:
        return out

    window = values.iloc[skip:end]
    if window.isna().any():
        return out
    seed = float(window.mean())

    seeded = values.astype("float64").copy()
    seeded.iloc[skip:end] = seed
    smoothed = seeded.ewm(alpha=alpha, adjust=False).mean()

    out.iloc[end - 1] = seed
    if end < n:
        out.iloc[end:] = smoothed.iloc[end:]
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (TradingView / TA-Lib seeding)."""
    return _seeded_ewm(series, period, alpha=2.0 / (period + 1.0))


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Seeded with the simple average of the first ``period`` changes, so
    the first value lands on index ``period`` -- the same as TA-Lib.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _seeded_ewm(gain, period, alpha=1.0 / period, skip=1)
    avg_loss = _seeded_ewm(loss, period, alpha=1.0 / period, skip=1)

    out = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    # avg_loss == 0 means "no down closes": RSI is 100 unless nothing moved
    # at all, in which case 50 is the honest answer.
    out = out.mask(avg_loss == 0.0, 100.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence -> (macd, signal, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing, TA-Lib seeding).

    TA-Lib defines the true range from the SECOND bar onwards (the first
    bar has no previous close) and seeds the average with the simple
    mean of the first ``period`` true ranges, so its first value lands
    on index ``period`` -- one bar later than a naive
    ``high - low`` seed would put it. Matching that exactly matters:
    an ATR band drawn one candle to the left of the exchange's is the
    kind of thing that makes a trader stop trusting the whole chart.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # The first bar has no previous close, so its true range is
    # undefined -- TA-Lib leaves it out of the seed window. Only the
    # first ROW is masked: masking by index label would also drop a
    # duplicated first timestamp.
    true_range = true_range.copy()
    true_range.iloc[0] = np.nan
    return _seeded_ewm(true_range, period, alpha=1.0 / period, skip=1)


def bollinger(
    close: pd.Series, period: int = 20, k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands -> (middle, upper, lower)."""
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return middle, middle + k * std, middle - k * std


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Session-anchored VWAP, re-anchored on each UTC day.

    A rolling VWAP over the whole session is not a VWAP; anchoring to the
    session is what a trader means by it and what an exchange prints.
    """
    typical = (high + low + close) / 3.0
    frame = pd.DataFrame({"typical": typical, "volume": volume})
    frame["day"] = _session_key(close.index)
    frame["pv"] = frame["typical"] * frame["volume"]
    grouped = frame.groupby("day", sort=True)
    cum_pv = grouped["pv"].cumsum()
    cum_v = grouped["volume"].cumsum()
    return (cum_pv / cum_v.replace(0.0, np.nan)).rename("vwap")


#: Smallest plausible unix timestamp (2001-09-09) -- below this an index
#: of integers is a row counter, not a time, and VWAP cannot be anchored.
_MIN_EPOCH_SECONDS = 1_000_000_000


def _session_key(index: pd.Index) -> pd.Series:
    """UTC-day key for each bar, whatever the index holds.

    ``compute_indicators`` indexes by integer epoch seconds, so the
    session boundary has to be derived from those rather than assumed
    to be a ``DatetimeIndex`` -- otherwise VWAP would never re-anchor.

    An index that is neither a ``DatetimeIndex`` nor a plausible epoch
    second is rejected: silently anchoring every bar to 1970-01-01
    would produce one enormous session and a VWAP that is not a VWAP.
    """
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(index.floor("D"), index=index)
    numeric = pd.to_numeric(pd.Series(index, index=index), errors="coerce")
    if numeric.dropna().abs().max() is not None and numeric.dropna().abs().max() < _MIN_EPOCH_SECONDS:
        raise ValueError(
            "vwap needs a DatetimeIndex or an index of unix epoch seconds; "
            f"got {type(index).__name__} with values up to {numeric.dropna().abs().max()}. "
            "Index the frame by 'timestamp' before computing indicators."
        )
    days = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce").dt.floor("D")
    return pd.Series(days.to_numpy(), index=index)


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel -> (upper, middle, lower)."""
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def compute_indicators(
    bars: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_k: float = 2.0,
    donchian_period: int = 20,
    vwap_enabled: bool = True,
) -> pd.DataFrame:
    """Add every indicator column to a copy of ``bars``.

    The frame must have ``timestamp``, ``open``, ``high``, ``low``,
    ``close`` and ``volume`` columns. The result is indexed by timestamp
    so that :func:`src.gui.timeframes.align_to_timeframe` and the
    charting layer can join on it.
    """
    if bars.empty:
        out = bars.copy()
        for col in INDICATOR_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    out = bars.copy().sort_values("timestamp").set_index("timestamp", drop=False)
    close = out["close"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    volume = out["volume"].astype("float64")

    out["ema_fast"] = ema(close, ema_fast)
    out["ema_slow"] = ema(close, ema_slow)
    out["sma_fast"] = sma(close, ema_fast)
    out["sma_slow"] = sma(close, ema_slow)
    out["rsi"] = rsi(close, rsi_period)

    macd_line, macd_signal, macd_hist = macd(close, macd_fast, macd_slow, macd_signal)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    out["atr"] = atr(high, low, close, atr_period)

    bb_mid, bb_upper, bb_lower = bollinger(close, bb_period, bb_k)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    if vwap_enabled:
        out["vwap"] = vwap(high, low, close, volume)
    else:
        out["vwap"] = pd.Series(np.nan, index=out.index)

    dc_upper, dc_mid, dc_lower = donchian(high, low, donchian_period)
    out["donchian_upper"] = dc_upper
    out["donchian_mid"] = dc_mid
    out["donchian_lower"] = dc_lower

    return out


def indicator_summary(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Latest non-NaN value per indicator -- for the chart's legend strip."""
    summary: Dict[str, Optional[float]] = {}
    for col in INDICATOR_COLUMNS:
        if col not in frame.columns or frame.empty:
            summary[col] = None
            continue
        series = frame[col].dropna()
        summary[col] = float(series.iloc[-1]) if len(series) else None
    return summary
