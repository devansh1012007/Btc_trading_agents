"""
Unit tests for Phase 7 T7.3: `RiskSizingModel`, `risk_score_to_size_multiplier`,
and `RiskLayer` `apply_to_size` integration (fixing Finding F4).
"""
import pickle
import numpy as np
import pytest

from src.layers.risk_layer import FEATURE_NAMES, RiskLayer, RiskStateStore
from src.sizing.risk_model import RiskSizingModel, risk_score_to_size_multiplier
from src.state.contracts import MarketSnapshot, RouterDecision
from src.state.contracts import ExpertAction, ExpertOutputs
from tests.unit.test_phase3 import make_expert_output, make_feature_vector, make_regime_state, make_zone_map

pytestmark = [pytest.mark.unit]


class _HighStopProbModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])  # P(stop-out) = 0.8


def test_risk_score_to_size_multiplier_is_monotone_and_tighten_only():
    m_low = risk_score_to_size_multiplier(0.1, min_multiplier=0.25)
    m_high = risk_score_to_size_multiplier(0.8, min_multiplier=0.25)
    assert 0.25 <= m_high < m_low <= 1.0
    assert risk_score_to_size_multiplier(0.0, min_multiplier=0.25) == pytest.approx(1.0)
    assert risk_score_to_size_multiplier(0.99, min_multiplier=0.25) == pytest.approx(0.25)


def test_risk_layer_apply_to_size_reduces_adjusted_size_when_enabled(tmp_path):
    model_path = tmp_path / "risk_m.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": _HighStopProbModel(),
                "feature_names": list(FEATURE_NAMES),
                "granularity": "1m",
                "feature_version": "v2_renamed",
            },
            f,
        )

    snap = MarketSnapshot(
        timestamp=1000,
        symbol="BTC/USDT",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=100.0,
        bid=99.99,
        ask=100.01,
        bid_volume=500.0,
        ask_volume=500.0,
        spread_pct=0.02,
        hour_utc=12,
        day_of_week=2,
    )
    fv = make_feature_vector()
    zm = make_zone_map()
    rs = make_regime_state()
    eo = ExpertOutputs(timestamp=1000, symbol="BTC/USDT", outputs={
        "trend_expert": make_expert_output("trend_expert", action=ExpertAction.BUY, confidence=0.85, expected_return=0.5, expected_risk=0.2),
        "mean_reversion_expert": make_expert_output("mean_reversion_expert", action=ExpertAction.SELL, confidence=0.75, expected_return=0.4, expected_risk=0.2),
        "breakout_expert": make_expert_output("breakout_expert", action=ExpertAction.BUY, confidence=0.72, expected_return=0.6, expected_risk=0.2),
    })
    rd = RouterDecision(
        timestamp=1000,
        symbol="BTC/USDT",
        selected_expert="trend_expert",
        selection_confidence=0.85,
        no_trade_probability=0.15,
    )

    # 1. Default (apply_to_size=False): adjusted_size equals base_size (full liquidity)
    rl_legacy = RiskLayer(
        {"ml_model": {"blend_weight": 1.0, "apply_to_size": False}},
        store=RiskStateStore(persistent=False),
        model_path=str(model_path),
    )
    rl_legacy.store.heartbeat()
    dec_legacy = rl_legacy.evaluate(rd, eo, rs, zm, fv, snap, equity=10_000.0)

    # 2. Enabled (apply_to_size=True): P(stop-out)=0.8 -> multiplier = max(0.2, 1 - 0.8) = 0.2!
    rl_sized = RiskLayer(
        {"ml_model": {"blend_weight": 1.0, "apply_to_size": True}, "sizing": {"ml_risk_sizing": {"min_multiplier": 0.25}}},
        store=RiskStateStore(persistent=False),
        model_path=str(model_path),
    )
    rl_sized.store.heartbeat()
    dec_sized = rl_sized.evaluate(rd, eo, rs, zm, fv, snap, equity=10_000.0)

    assert dec_sized.approved
    assert dec_sized.adjusted_size == pytest.approx(dec_legacy.adjusted_size * 0.25)


def test_risk_sizing_model_fits_and_predicts():
    rng = np.random.default_rng(11)
    model = RiskSizingModel({"min_multiplier": 0.25})
    rows = [{f: float(rng.normal()) for f in model.feature_names} for _ in range(40)]
    y = [1 if i % 2 == 0 else 0 for i in range(40)]
    model.fit(rows, y)
    p_stop, mult = model.predict_stop_out_and_multiplier(rows[0])
    assert 0.0 <= p_stop <= 1.0
    assert 0.25 <= mult <= 1.0
