"""
Unit tests for Phase 16 T16.5: `downsample_lod` level-of-detail downsampling.
"""
import pandas as pd
import pytest

pytest.importorskip("plotly")

from src.gui.charts import downsample_lod  # noqa: E402

pytestmark = [pytest.mark.unit]


def test_downsample_lod_caps_points_and_preserves_global_extrema():
    n = 5000
    df = pd.DataFrame({
        "timestamp": list(range(1000, 1000 + n)),
        "open": [100.0] * n,
        "high": [101.0 if i != 1234 else 250.0 for i in range(n)],
        "low": [99.0 if i != 3456 else 40.0 for i in range(n)],
        "close": [100.5] * n,
        "volume": [1.0] * n,
    })
    lod = downsample_lod(df, max_points=500)
    assert len(lod) <= 500
    assert lod["high"].max() == pytest.approx(250.0)
    assert lod["low"].min() == pytest.approx(40.0)
    assert lod["volume"].sum() == pytest.approx(float(n))
