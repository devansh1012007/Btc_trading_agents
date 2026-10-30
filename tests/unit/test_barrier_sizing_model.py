"""
Unit tests for Phase 7 T7.2: `BarrierSizingModel` quantile regression,
zone-survival feature consumption, and lockout adverse excursion floor.
"""
import numpy as np
import pytest

from src.sizing.barrier_model import (
    BarrierSizingModel,
    build_barrier_sizing_features,
)
from tests.unit.test_phase3 import make_feature_vector, make_regime_state, make_zone_map

pytestmark = [pytest.mark.unit]


def test_barrier_sizing_model_fits_and_enforces_lockout_floor_and_rr():
    rng = np.random.default_rng(42)
    fv = make_feature_vector(volatility_medium=0.4)
    zm = make_zone_map()
    rs = make_regime_state()
    base_feats = build_barrier_sizing_features(fv, zm, rs, direction=1)

    rows = []
    k_mae = []
    k_mfe = []
    k_lock = []
    for _ in range(40):
        r = dict(base_feats)
        r["volatility_medium"] = float(rng.uniform(0.2, 0.8))
        r["nearest_support_survival_prob"] = float(rng.uniform(0.2, 0.9))
        rows.append(r)
        k_mae.append(1.0 + 0.5 * (1.0 - r["nearest_support_survival_prob"]))
        k_mfe.append(2.2 + 0.5 * r["nearest_support_survival_prob"])
        k_lock.append(1.2)

    bm = BarrierSizingModel({"min_reward_to_risk": 1.25, "lockout_buffer_mult": 1.1}).fit(
        rows, k_mae, k_mfe, k_lockout_mae=k_lock
    )
    assert bm.is_ready
    plan = bm.predict_exit_plan(fv, zm, rs, direction=1, max_holding_bars=6, hard_max_loss_pct=0.6)
    assert plan is not None
    assert plan.source == "barrier_model"
    assert plan.take_profit_pct >= plan.stop_loss_pct * 1.25 - 1e-6
    assert plan.effective_stop_loss_pct <= 0.6
