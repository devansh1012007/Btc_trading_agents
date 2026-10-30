"""
Unit tests for Phase 9 T9.1: per-regime `ExpertInstance` registry and
per-instance `enabled` switch.
"""
import copy

import pytest

from src.experts.expert_registry import build_experts
from src.state.contracts import DynamicRegime, ExpertAction, RegimeState
from tests.unit.test_phase3 import make_feature_vector, make_zone_map

pytestmark = [pytest.mark.unit]


def test_build_experts_supports_per_regime_instances_and_enabled_toggle(paper_config):
    cfg = copy.deepcopy(paper_config)
    cfg["experts"]["instances"] = [
        {"id": "trend@r_0", "template": "trend", "regime_scope": ["r_0"], "enabled": True},
        {"id": "trend@r_1", "template": "trend", "regime_scope": ["r_1"], "enabled": False},
    ]
    experts = build_experts(cfg, "1m")
    assert set(experts.keys()) == {"trend@r_0", "trend@r_1"}
    assert experts["trend@r_0"].enabled is True
    assert experts["trend@r_1"].enabled is False

    fv = make_feature_vector(momentum_medium=0.25)
    zm = make_zone_map()
    rs_r0 = RegimeState(
        timestamp=1000,
        symbol="BTC/USDT",
        primary_regime=DynamicRegime("r_0"),
        regime_confidence=0.85,
        regime_age=5,
        regime_persistence_score=0.9,
        regime_set_id="dyn_k2",
        regime_probs={"r_0": 0.85, "r_1": 0.15},
    )
    out0 = experts["trend@r_0"].evaluate(fv, zm, rs_r0)
    assert out0.action == ExpertAction.BUY
    assert out0.instance_id == "trend@r_0"

    # Disabled instance always outputs NO_TRADE even when its regime is active
    rs_r1 = RegimeState(
        timestamp=1000,
        symbol="BTC/USDT",
        primary_regime=DynamicRegime("r_1"),
        regime_confidence=0.85,
        regime_age=5,
        regime_persistence_score=0.9,
        regime_set_id="dyn_k2",
        regime_probs={"r_0": 0.15, "r_1": 0.85},
    )
    out1 = experts["trend@r_1"].evaluate(fv, zm, rs_r1)
    assert out1.action == ExpertAction.NO_TRADE
