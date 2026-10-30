"""
Unit tests for Phase 11 T11.1: Router A / Router B selection, mixed exit mode
fallback (C9), and causal Bayesian `RollingSuccessTracker` (L3).
"""
import pytest

from src.layers.router import RollingSuccessTracker, Router
from src.state.contracts import OpportunityScore
from src.state.contracts import ExpertAction, ExpertOutputs
from tests.unit.test_phase3 import make_expert_output, make_regime_state, make_zone_map

pytestmark = [pytest.mark.unit]


class _FakeRouterModel:
    def __init__(self, winner: str):
        self.classes_ = ["trend_expert", "mean_reversion_expert", "breakout_expert", "no_trade"]
        self.winner = winner

    def predict_proba(self, X):
        return [[0.9 if c == self.winner else 0.033 for c in self.classes_]]


def test_rolling_success_tracker_is_strictly_causal_with_bayesian_prior():
    tracker = RollingSuccessTracker(prior_alpha=2.0, prior_beta=2.0)
    # Cold start: (0 + 2) / (0 + 4) = 0.5
    assert tracker.success_rate("trend_expert", current_timestamp=1000.0) == pytest.approx(0.5)

    # Record a winning trade that closes at t=1500
    tracker.record_closed_trade("trend_expert", exit_timestamp=1500.0, pnl_pct=0.8)
    # At t=1200 (before the trade closed!), success_rate MUST NOT see it (L3)!
    assert tracker.success_rate("trend_expert", current_timestamp=1200.0) == pytest.approx(0.5)
    # At t=1500 (once closed), success_rate updates to (1 + 2) / (1 + 4) = 0.6!
    assert tracker.success_rate("trend_expert", current_timestamp=1500.0) == pytest.approx(0.6)


def test_router_selects_router_a_vs_router_b_and_falls_back_on_mixed_exit_modes():
    eo = ExpertOutputs(timestamp=1000, symbol="BTC/USDT", outputs={
        "trend_expert": make_expert_output("trend_expert", action=ExpertAction.BUY, confidence=0.85, expected_return=0.5, expected_risk=0.2),
        "mean_reversion_expert": make_expert_output("mean_reversion_expert", action=ExpertAction.SELL, confidence=0.75, expected_return=0.4, expected_risk=0.2),
        "breakout_expert": make_expert_output("breakout_expert", action=ExpertAction.BUY, confidence=0.72, expected_return=0.6, expected_risk=0.2),
    })
    rs = make_regime_state()
    zm = make_zone_map()
    opp = OpportunityScore(
        timestamp=1000, symbol="BTC/USDT", tradable=True, tradable_probability=0.9,
        environment_quality=0.8, opportunity_strength=0.8, failed_rules=[],
    )

    router = Router({"model": {"blend_weight": 1.0}, "exit_mode": "hard_limits"})
    router._model_a = _FakeRouterModel("mean_reversion_expert")
    router._model_b = _FakeRouterModel("breakout_expert")

    # Under hard_limits -> Router A selects mean_reversion_expert
    dec_a = router.decide(eo, rs, zm, opp)
    assert dec_a.selected_expert == "mean_reversion_expert"

    # Under ml_exit -> Router B selects breakout_expert
    router.exit_mode = "ml_exit"
    dec_b = router.decide(eo, rs, zm, opp)
    assert dec_b.selected_expert == "breakout_expert"

    # Mixed exit modes across active experts (C9) -> falls back to rule/confidence router!
    router.expert_exit_modes = {"trend_expert": "ml_exit", "mean_reversion_expert": "hard_limits"}
    dec_mix = router.decide(eo, rs, zm, opp)
    assert dec_mix.reasoning == "mixed_exit_mode_fallback"
