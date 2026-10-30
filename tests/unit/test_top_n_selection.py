"""
Unit tests for Phase 9 T9.4 / Leakage Control L4: `select_top_n_instances`.
"""
import pytest

from src.experts.selection import (
    InSampleSelectionError,
    select_top_n_instances,
)

pytestmark = [pytest.mark.unit]


def test_select_top_n_instances_ranks_oof_and_rejects_in_sample():
    oof_records = (
        [{"instance_id": "trend@r_0", "regime": "r_0", "realized_return_pct": 0.6, "is_oof": True} for _ in range(5)]
        + [{"instance_id": "mr@r_0", "regime": "r_0", "realized_return_pct": 0.2, "is_oof": True} for _ in range(5)]
        + [{"instance_id": "bo@r_0", "regime": "r_0", "realized_return_pct": -0.4, "is_oof": True} for _ in range(5)]
    )
    res = select_top_n_instances(oof_records, top_n=2, per_regime=True, min_oof_trades=3)
    assert res.selected_by_regime["r_0"] == ["trend@r_0", "mr@r_0"]

    # Passing an in-sample record raises InSampleSelectionError (L4)
    bad = oof_records + [{"instance_id": "cheater@r_0", "regime": "r_0", "realized_return_pct": 9.0, "is_oof": False}]
    with pytest.raises(InSampleSelectionError, match="L4 violation"):
        select_top_n_instances(bad, top_n=1)
