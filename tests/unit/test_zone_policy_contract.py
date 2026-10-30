"""
Unit tests for Phase 6 T6.3: `refresh_policy_hash` contract enforcement in
`DynamicStructureEngine` and `model_alignment_problem`.
"""
import pickle

import pytest

from src.layers.structure_engine import DynamicStructureEngine, ZoneRefreshPolicy

pytestmark = [pytest.mark.unit]


class _DummyZoneModel:
    def predict_proba(self, X):
        return [[0.3, 0.7]]


def test_structure_engine_rejects_zone_model_with_mismatched_refresh_policy_hash(tmp_path):
    pol_a = ZoneRefreshPolicy(mode="every_n", every_n_bars=5)
    pol_b = ZoneRefreshPolicy(mode="every_n", every_n_bars=10)
    assert pol_a.policy_hash() != pol_b.policy_hash()

    model_path = tmp_path / "zone_surv.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": _DummyZoneModel(),
                "feature_names": ["zone_age"],
                "granularity": "1m",
                "refresh_policy_hash": pol_a.policy_hash(),
            },
            f,
        )

    eng_ok = DynamicStructureEngine(
        zone_survival_model_path=str(model_path),
        data_granularity="1m",
        refresh_policy=pol_a,
    )
    assert eng_ok._zone_survival_model is not None

    eng_mismatch = DynamicStructureEngine(
        zone_survival_model_path=str(model_path),
        data_granularity="1m",
        refresh_policy=pol_b,
    )
    assert eng_mismatch._zone_survival_model is None
