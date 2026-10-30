"""
Unit tests for Phase 9 T9.2 / Leakage Control L2: `chop_candidates_by_served_regime`.
"""
import pytest

from src.experts.regime_chop import (
    NonCausalRegimeColumnError,
    chop_candidates_by_served_regime,
)

pytestmark = [pytest.mark.unit]


def test_chop_candidates_by_served_regime_and_hierarchical_fallback():
    rows = [{"regime": "r_0", "x": i} for i in range(35)] + [{"regime": "r_1", "x": i} for i in range(10)]
    res_r0 = chop_candidates_by_served_regime(rows, ["r_0"], min_samples_for_chop=30)
    assert res_r0.in_regime_count == 35
    assert not res_r0.used_hierarchical_fallback
    assert len(res_r0.rows) == 35

    # Small sample (10 < 30) triggers hierarchical shrinkage with global_prior_weight
    res_r1 = chop_candidates_by_served_regime(rows, ["r_1"], min_samples_for_chop=30, global_prior_weight=0.2)
    assert res_r1.in_regime_count == 10
    assert res_r1.used_hierarchical_fallback
    assert len(res_r1.rows) == 45
    assert sum(w == 1.0 for w in res_r1.sample_weights) == 10
    assert sum(abs(w - 0.2) < 1e-9 for w in res_r1.sample_weights) == 35


def test_chop_refuses_smoothed_non_causal_regime_column():
    with pytest.raises(NonCausalRegimeColumnError, match="L2 violation"):
        chop_candidates_by_served_regime([{"smoothed_regime": "r_0"}], ["r_0"], regime_col="smoothed_regime")
