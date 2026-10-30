"""
Unit tests for Phase 16 T16.3: dynamic K regime bands and colors (`build_regime_bands`,
`theme.regime_palette`).
"""
import pytest

from src.gui import overlays as ov
from src.gui import theme
from tests.unit.gui_fixtures import make_bundle

pytestmark = [pytest.mark.unit]


def test_dynamic_regime_bands_and_palette_support_arbitrary_k():
    bundle = make_bundle()
    # Replace legacy regime names with dynamic r_0, r_1, r_2, r_3, r_4
    decs = bundle.decisions.copy()
    decs["regime"] = [f"r_{min(4, i // 24)}" for i in range(len(decs))]
    bundle.decisions = decs

    bands = ov.build_regime_bands(bundle, "5m", bundle.bars)
    assert len(bands) > 0
    palette = theme.regime_palette([b.regime for b in bands])
    assert set(palette.keys()) == {f"r_{i}" for i in range(5)}
    assert all(c.startswith("rgba(") for c in palette.values())
