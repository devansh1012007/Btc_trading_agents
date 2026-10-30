"""
Unit tests for Phase 13 T13.4: `1s/2s/5s` timeframes and `features.windows_seconds`
time-horizon scaling across granularities.
"""
import pytest

from src.gui.timeframes import display_timeframes, parse_timeframe, resample_ohlcv
from src.layers.feature_engine import FeatureEngine
from src.utils.synthetic import synthetic_ohlcv

pytestmark = [pytest.mark.unit]


def test_sub_10s_timeframes_and_windows_seconds_scaling():
    assert parse_timeframe("1s") == 1
    assert parse_timeframe("2s") == 2
    assert parse_timeframe("5s") == 5
    assert display_timeframes("1s")[:4] == ["1s", "2s", "5s", "10s"]

    df_1s = synthetic_ohlcv(n_bars=60, bar_seconds=1)
    df_10s = resample_ohlcv(df_1s, "10s", base_timeframe="1s")
    assert len(df_10s) == 6

    # Same physical window (short=10s, medium=60s, long=120s) on 1s vs 10s bars
    fe_1s = FeatureEngine(
        config={"system": {"data_granularity": "1s"}, "features": {"windows_seconds": {"short": 10, "medium": 60, "long": 120}}}
    )
    fe_10s = FeatureEngine(
        config={"system": {"data_granularity": "10s"}, "features": {"windows_seconds": {"short": 10, "medium": 60, "long": 120}}}
    )
    assert fe_1s.momentum_medium == 60 and fe_10s.momentum_medium == 6
    assert fe_1s.momentum_long == 120 and fe_10s.momentum_long == 12
