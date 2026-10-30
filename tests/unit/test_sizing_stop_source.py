"""
Unit tests for Phase 4 T4.5: `sizing.stop_source: invalidation_distance | exit_plan`
in `src/layers/risk_layer.py`.
"""
import pytest

from src.layers.risk_layer import compute_base_size
from src.state.contracts import ExitPlan, ExpertAction, ExpertOutput, RegimeType

pytestmark = [pytest.mark.unit]


def _eo(inv_dist: float = 0.5, plan=None) -> ExpertOutput:
    return ExpertOutput(
        expert_name="trend",
        timestamp=1000,
        symbol="BTC/USDT",
        action=ExpertAction.BUY,
        confidence=0.8,
        no_trade_probability=0.2,
        expected_return=0.8,
        expected_risk=0.3,
        reward_to_risk=2.6,
        invalidation_distance=inv_dist,
        zone_aware=True,
        model_output=0.8,
        regime_used=RegimeType.TRENDING_UP,
        exit_plan=plan,
    )


def test_stop_source_invalidation_distance_vs_exit_plan():
    equity = 10_000.0
    plan = ExitPlan(stop_loss_pct=1.0, take_profit_pct=2.0, max_holding_bars=5, hard_max_loss_pct=0.8)
    eo = _eo(inv_dist=0.4, plan=plan)

    # Default: invalidation_distance (0.4%) -> risk $50 / 0.004 = $12,500
    sz_default = compute_base_size(equity, eo, {"risk_pct_per_trade": 0.5})
    assert sz_default == pytest.approx(12_500.0)

    # exit_plan: effective_stop_loss_pct is min(1.0, 0.8) = 0.8% -> risk $50 / 0.008 = $6,250
    sz_plan = compute_base_size(equity, eo, {"risk_pct_per_trade": 0.5, "stop_source": "exit_plan"})
    assert sz_plan == pytest.approx(6_250.0)
    # At a 0.8% stop-out, the dollar loss on $6,250 is $6,250 * 0.008 = $50.00 == 0.5% of equity!
    assert sz_plan * (plan.effective_stop_loss_pct / 100.0) == pytest.approx(50.0)
