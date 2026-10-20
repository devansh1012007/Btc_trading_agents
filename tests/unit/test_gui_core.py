"""
Trade Map GUI core: instrument metadata, timeframe resampling and
technical indicators.

The properties that matter here are causality (nothing may use a future
bar) and determinism (the same bars must always produce the same
higher timeframe), because the whole point of the GUI is to describe
what the trading system actually did.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.gui.indicators import atr, bollinger, compute_indicators, ema, macd, rsi  # noqa: E402
from src.gui.instruments import (  # noqa: E402
    InstrumentSpec,
    default_spec,
    load_registry,
    resolve_instrument,
)
from src.gui.timeframes import (  # noqa: E402
    UnknownTimeframeError,
    bucket_start,
    display_timeframes,
    parse_timeframe,
    resample_ohlcv,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def make_bars(n: int = 60, start_ts: int = 1_700_000_100, step: int = 60) -> pd.DataFrame:
    """One-minute OHLCV bars with a deterministic ramp."""
    return pd.DataFrame({
        "timestamp": [start_ts + i * step for i in range(n)],
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    })


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

def test_default_spec_is_generic_spot():
    spec = default_spec("XYZ/ABC")
    assert spec.symbol == "XYZ/ABC"
    assert spec.asset_class == "spot"
    assert spec.is_option is False
    assert spec.multiplier == 1.0


def test_option_spec_metadata_and_moneyness():
    spec = InstrumentSpec(
        symbol="BTC-261220-75000-C",
        asset_class="option",
        underlying="BTC/USDT",
        option_type="call",
        strike=75000.0,
        expiry="2026-12-20",
        multiplier=1.0,
    )
    assert spec.is_option is True
    assert spec.is_derivative is True
    assert spec.moneyness_pct(80000.0) == pytest.approx((80000.0 - 75000.0) / 80000.0 * 100)
    assert "75,000" in spec.label()
    assert "C" in spec.label()


def test_moneyness_is_none_for_non_options():
    assert default_spec("BTC/USDT").moneyness_pct(100.0) is None


def test_load_registry_reads_instruments(tmp_path):
    path = tmp_path / "instruments.yaml"
    path.write_text(
        "instruments:\n"
        "  BTC/USDT:\n"
        "    asset_class: spot\n"
        "    display_precision: 1\n"
        "  BTC-261220-75000-C:\n"
        "    asset_class: option\n"
        "    underlying: BTC/USDT\n"
        "    option_type: call\n"
        "    strike: 75000\n"
        "    expiry: '2026-12-20'\n"
        "    multiplier: 1\n",
        encoding="utf-8",
    )
    registry = load_registry(str(path))
    assert set(registry) == {"BTC/USDT", "BTC-261220-75000-C"}
    assert registry["BTC/USDT"].display_precision == 1
    assert registry["BTC-261220-75000-C"].is_option is True
    assert registry["BTC-261220-75000-C"].strike == 75000.0


def test_load_registry_missing_file_is_not_an_error(tmp_path):
    assert load_registry(str(tmp_path / "nope.yaml")) == {}


def test_load_registry_skips_a_malformed_entry(tmp_path):
    """One bad registry line must not take the GUI down."""
    path = tmp_path / "instruments.yaml"
    path.write_text(
        "instruments:\n"
        "  GOOD:\n"
        "    asset_class: spot\n"
        "  BAD:\n"
        "    asset_class: banana\n",
        encoding="utf-8",
    )
    registry = load_registry(str(path))
    assert "GOOD" in registry
    assert "BAD" not in registry
    assert resolve_instrument("BAD", registry=registry).asset_class == "spot"


def test_resolve_instrument_falls_back_for_unknown_symbol(tmp_path):
    path = tmp_path / "instruments.yaml"
    path.write_text("instruments:\n  BTC/USDT:\n    asset_class: spot\n", encoding="utf-8")
    spec = resolve_instrument("ETH/USDT", registry=load_registry(str(path)))
    assert spec.symbol == "ETH/USDT"
    assert spec.asset_class == "spot"


# ---------------------------------------------------------------------------
# timeframes
# ---------------------------------------------------------------------------

def test_parse_timeframe_known_and_unknown():
    assert parse_timeframe("1m") == 60
    assert parse_timeframe("4h") == 14400
    assert parse_timeframe("1D") == 86400
    with pytest.raises(UnknownTimeframeError):
        parse_timeframe("7q")


def test_display_timeframes_excludes_finer_than_base():
    assert display_timeframes("1m")[0] == "1m"
    assert "30s" not in display_timeframes("1m")
    assert display_timeframes("30s")[0] == "30s"


def test_resample_5m_aggregates_minute_bars_correctly():
    bars = make_bars(n=10)  # 10 consecutive 1m bars
    out = resample_ohlcv(bars, "5m")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == bars.iloc[0]["open"]
    assert first["high"] == bars.iloc[0:5]["high"].max()
    assert first["low"] == bars.iloc[0:5]["low"].min()
    assert first["close"] == bars.iloc[4]["close"]
    assert first["volume"] == bars.iloc[0:5]["volume"].sum()
    assert first["timestamp"] == bars.iloc[0]["timestamp"]


def test_resample_labels_buckets_at_their_open():
    bars = make_bars(n=10, start_ts=1_700_000_000)  # deliberately misaligned
    out = resample_ohlcv(bars, "5m")
    # A misaligned start splits the first bucket: 2 bars, then 5, then 3.
    assert len(out) == 3
    for row in out.itertuples():
        assert row.timestamp % 300 == 0
    assert out.iloc[0]["timestamp"] <= bars.iloc[0]["timestamp"]


def test_resample_is_causal_no_bucket_uses_a_future_bar():
    """Truncating the input must not change any bucket that is fully
    present in the truncated frame -- a look-ahead would show up here."""
    bars = make_bars(n=30)
    full = resample_ohlcv(bars, "5m")
    partial = resample_ohlcv(bars.iloc[:15], "5m")
    # The first two buckets are complete in both frames and must match.
    assert full.iloc[0].tolist() == partial.iloc[0].tolist()
    assert full.iloc[1].tolist() == partial.iloc[1].tolist()
    assert len(partial) < len(full)


def test_resample_drops_empty_buckets():
    bars = make_bars(n=3)
    out = resample_ohlcv(bars, "1h")
    assert len(out) == 1


def test_resample_empty_frame_is_empty():
    out = resample_ohlcv(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), "5m")
    assert out.empty


def test_bucket_start_snaps_into_the_right_bucket():
    base = 1_700_000_000
    assert bucket_start(base + 10, "5m") == (base // 300) * 300
    assert bucket_start(base, "1h") == (base // 3600) * 3600


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def test_ema_is_causal():
    """Changing a FUTURE bar must not change any earlier EMA value."""
    close = pd.Series([float(i) for i in range(30)])
    changed = close.copy()
    changed.iloc[-1] = 9999.0
    base = ema(close, 5)
    after = ema(changed, 5)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])


def test_rsi_is_bounded_and_causal():
    close = pd.Series([100.0 + (i % 7) for i in range(60)])
    out = rsi(close, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()
    changed = close.copy()
    changed.iloc[-1] = -500.0
    pd.testing.assert_series_equal(rsi(close, 14).iloc[:-1], rsi(changed, 14).iloc[:-1])


def test_rsi_all_gains_is_100():
    close = pd.Series([float(i) for i in range(30)])
    assert rsi(close, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_series_is_50():
    close = pd.Series([100.0] * 30)
    assert rsi(close, 14).iloc[-1] == pytest.approx(50.0)


def test_macd_histogram_is_line_minus_signal():
    close = pd.Series([100.0 + np.sin(i / 3) for i in range(80)])
    line, signal, hist = macd(close)
    pd.testing.assert_series_equal(hist, line - signal)


def test_atr_is_positive():
    bars = make_bars(n=40)
    out = atr(bars["high"], bars["low"], bars["close"], 14).dropna()
    assert (out > 0).all()


def test_bollinger_bands_bracket_the_middle():
    close = pd.Series([100.0 + np.sin(i / 4) for i in range(60)])
    mid, upper, lower = bollinger(close, 20, 2.0)
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_compute_indicators_adds_every_column_and_keeps_timestamps():
    bars = make_bars(n=40)
    out = compute_indicators(bars)
    for column in ("rsi", "macd", "atr", "bb_upper", "vwap", "donchian_upper"):
        assert column in out.columns
    assert list(out["timestamp"]) == list(bars["timestamp"])
