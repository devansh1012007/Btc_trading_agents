"""
Unit tests for Phase 16 T16.6: backward compatibility with pre-Phase-16 `SessionBundle`
and overlays.
"""
import pytest

pytest.importorskip("plotly")

from src.gui import charts, overlays  # noqa: E402
from tests.unit.gui_fixtures import make_bundle  # noqa: E402

pytestmark = [pytest.mark.unit]


def test_legacy_session_bundle_renders_all_views_cleanly():
    bundle = make_bundle()
    ov_set = overlays.build_overlays(bundle, "5m")
    assert all(e.lane_id == 0 for e in ov_set.entries)
    figs = charts.build_session_figures(bundle, "5m")
    assert set(figs.keys()) == {"price", "decisions", "equity", "funnel"}
