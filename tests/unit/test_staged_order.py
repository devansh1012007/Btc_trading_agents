"""
Unit tests for Phase 11 T11.2: `pipeline.order: staged` (`opportunity_pre_filter`
before router, `opportunity_confirm` after router on the selected expert).
"""
import copy
import pytest

from src.layers.opportunity_filter import OpportunityFilter
from src.pipeline import TradingPipeline
from src.state.contracts import (
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    MarketSnapshot,
    RegimeType,
    RouterDecision,
)
from tests.unit.test_phase3 import make_feature_vector, make_regime_state, make_zone_map

pytestmark = [pytest.mark.unit]


def test_opportunity_confirm_checks_router_selected_expert_edge():
    opp = OpportunityFilter(
        config={"hard_rules": {"edge_vs_fees_threshold": 0.10, "max_spread_pct": 1.0, "min_depth_usdt": 0.0}},
        labeling_config={"fee_pct": 0.04},
    )
    snap = MarketSnapshot(
        timestamp=1000, symbol="BTC/USDT", open=100.0, high=101.0, low=99.0, close=100.0,
        volume=50.0, bid=99.99, ask=100.01, bid_volume=100.0, ask_volume=100.0, spread_pct=0.02,
        hour_utc=12, day_of_week=2,
    )
    fv = make_feature_vector()
    rs = make_regime_state()
    zm = make_zone_map()

    # Best expert by confidence ("high_conf_low_edge") has low expected_return (0.12 - 0.08 = 0.04 < 0.10),
    # whereas "high_edge" has expected_return 0.50 (edge 0.42 >= 0.10).
    outputs = ExpertOutputs(
        timestamp=1000,
        symbol="BTC/USDT",
        outputs={
            "high_edge": ExpertOutput(
                expert_name="high_edge", timestamp=1000, symbol="BTC/USDT",
                action=ExpertAction.BUY, confidence=0.80, no_trade_probability=0.20,
                expected_return=0.50, expected_risk=0.20, reward_to_risk=2.5,
                invalidation_distance=0.4, zone_aware=True, model_output=0.8, regime_used=RegimeType.TRENDING_UP,
            ),
            "low_edge": ExpertOutput(
                expert_name="low_edge", timestamp=1000, symbol="BTC/USDT",
                action=ExpertAction.BUY, confidence=0.75, no_trade_probability=0.25,
                expected_return=0.12, expected_risk=0.20, reward_to_risk=0.6,
                invalidation_distance=0.4, zone_aware=True, model_output=0.75, regime_used=RegimeType.TRENDING_UP,
            ),
        },
    )
    pre = opp.evaluate_pre_filter(rs, zm, snap, fv)
    assert pre.tradable

    # If router selects "low_edge", evaluate_confirm rejects it!
    rd_bad = RouterDecision(
        timestamp=1000, symbol="BTC/USDT", selected_expert="low_edge",
        selection_confidence=0.75, no_trade_probability=0.25,
    )
    conf_bad = opp.evaluate_confirm(outputs, rd_bad, rs, zm, snap, fv)
    assert not conf_bad.tradable
    assert any("selected_edge_below_threshold" in r for r in conf_bad.failed_rules)

    # If router selects "high_edge", evaluate_confirm approves it!
    rd_good = RouterDecision(
        timestamp=1000, symbol="BTC/USDT", selected_expert="high_edge",
        selection_confidence=0.80, no_trade_probability=0.20,
    )
    conf_good = opp.evaluate_confirm(outputs, rd_good, rs, zm, snap, fv)
    assert conf_good.tradable


def test_trading_pipeline_runs_in_staged_order(paper_config, tmp_path):
    cfg = copy.deepcopy(paper_config)
    cfg["pipeline"] = {"order": "staged"}
    pip = TradingPipeline(config=cfg, data_dir=str(tmp_path))
    assert pip.pipeline_order == "staged"
