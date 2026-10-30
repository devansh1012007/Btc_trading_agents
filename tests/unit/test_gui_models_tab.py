"""
Unit tests for Phase 16 T16.4: `build_models_scorecard_chart`.
"""
import pytest

pytest.importorskip("plotly")

from src.gui.charts import build_models_scorecard_chart  # noqa: E402
from tests.unit.gui_fixtures import make_bundle  # noqa: E402

pytestmark = [pytest.mark.unit]


def test_build_models_scorecard_chart_renders_populated_and_empty_bundles():
    bundle_empty = make_bundle()
    fig_empty = build_models_scorecard_chart(bundle_empty)
    assert fig_empty.layout.annotations

    bundle = make_bundle()
    bundle.model_scorecards = {
        "trend_expert": {"brier_score": 0.14, "realized_expectancy": 0.32},
        "router_a": {"brier_score": 0.19, "realized_expectancy": 0.21},
    }
    fig = build_models_scorecard_chart(bundle)
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == ["router_a", "trend_expert"]
