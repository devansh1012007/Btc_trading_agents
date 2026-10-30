"""
Unit tests for Phase 5 T5.2: `select_k` with occupancy floor, dwell floor,
and BIC selection.
"""
import numpy as np
import pytest

from src.regime.selection import select_k

pytestmark = [pytest.mark.unit]


def test_select_k_identifies_planted_clusters_and_respects_bounds():
    rng = np.random.default_rng(123)
    # 3 well-separated contiguous blocks of 80 bars each (long dwell, balanced occupancy ~0.33)
    b0 = rng.normal(loc=[-3.0, -2.0], scale=0.2, size=(80, 2))
    b1 = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(80, 2))
    b2 = rng.normal(loc=[3.0, 2.0], scale=0.2, size=(80, 2))
    X = np.vstack([b0, b1, b2])

    res = select_k(
        X,
        backend_name="gmm",
        k_min=2,
        max_regimes=5,
        min_occupancy=0.10,
        min_mean_dwell_bars=5.0,
        min_bic_delta_pct=1.0,
    )
    assert res.selected_k == 3
    assert len(res.regime_set.regimes) == 3
    assert not res.used_fallback_k4


def test_select_k_falls_back_when_constraints_impossible():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 2))
    # Impossible occupancy constraint (> 0.90 when K >= 2) forces fallback
    res = select_k(X, backend_name="kmeans", k_min=2, max_regimes=4, min_occupancy=0.95)
    assert res.used_fallback_k4
    assert res.selected_k == 4
