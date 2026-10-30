"""
Unit tests for Phase 16 T16.5: `load_playback_slice` on-demand window loader.
"""
import pytest

pytest.importorskip("plotly")

from src.gui.charts import load_playback_slice  # noqa: E402
from tests.unit.gui_fixtures import START, make_bundle  # noqa: E402

pytestmark = [pytest.mark.unit]


def test_load_playback_slice_returns_bounded_window():
    bundle = make_bundle()
    sub = load_playback_slice(bundle.bars, start_ts=START + 120, window_seconds=300)
    assert len(sub) == 5
    assert sub["timestamp"].min() == START + 120
    assert sub["timestamp"].max() == START + 360
