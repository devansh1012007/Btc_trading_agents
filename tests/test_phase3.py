"""
Phase 3 test suite (§11): triple-barrier labeling, expert gates/fallback/
granularity-guard, opportunity filter hard-rule veto + regime gate, router
fallback + switching context + granularity-guard, config loading, and
timeframe adaptation.
"""
import pickle
import numpy as np
import pytest
import yaml

from src.state.contracts import (
    MarketSnapshot, FeatureVector, ZoneMap, RegimeState, RegimeType,
    ExpertOutput, ExpertOutputs, ExpertAction, OpportunityScore,
)
from src.labeling.triple_barrier import label_triple_barrier, TripleBarrierOutcome
from src.experts.trend_expert import TrendExpert
from src.experts.expert_registry import build_experts
from src.layers.opportunity_filter import OpportunityFilter, FEATURE_NAMES as OPP_FEATURE_NAMES
from src.layers.router import Router


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def make_snapshot(timestamp, open_, high, low, close, volume=100.0, symbol="BTC/USDT"):
    return MarketSnapshot(
        timestamp=timestamp, symbol=symbol, open=open_, high=high, low=low, close=close, volume=volume,
        bid=close * 0.9995, ask=close * 1.0005, bid_volume=50.0, ask_volume=50.0,
        spread_pct=0.01, hour_utc=12, day_of_week=2,
    )


def make_feature_vector(timestamp=0, momentum_5m=0.0, momentum_10m=0.0, volatility_5m=0.1, volatility_10m=0.1,
                         volatility_30s=0.05, momentum_30s=0.0, spread_pct=0.01, symbol="BTC/USDT"):
    return FeatureVector(
        timestamp=timestamp, symbol=symbol, momentum_30s=momentum_30s, momentum_5m=momentum_5m,
        volatility_30s=volatility_30s, volatility_5m=volatility_5m, returns_5=0.0, rsi_14=50.0, macd_signal=0.0,
        sma_9=101.0, sma_21=100.0, momentum_10m=momentum_10m, volatility_10m=volatility_10m,
        imbalance_ratio=0.5, orderflow_momentum=0.0, spread_pct=spread_pct, depth_10_level=100.0,
        raw_features={"volume_zscore": 0.0},
    )


def make_regime_state(primary_regime=RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8, timestamp=0):
    probs = {rt: 0.0 for rt in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.CHOP_RANGE, RegimeType.PANIC_SPIKE]}
    probs[primary_regime] = confidence
    return RegimeState(
        timestamp=timestamp, symbol="BTC/USDT",
        prob_trending_up=probs[RegimeType.TRENDING_UP], prob_trending_down=probs[RegimeType.TRENDING_DOWN],
        prob_chop_range=probs[RegimeType.CHOP_RANGE], prob_panic_spike=probs[RegimeType.PANIC_SPIKE],
        primary_regime=primary_regime, regime_confidence=confidence,
        regime_age=10, regime_persistence_score=persistence,
    )


def make_zone_map(current_price=100.0, timestamp=0):
    return ZoneMap(timestamp=timestamp, symbol="BTC/USDT", current_price=current_price)


def make_expert_output(name, action=ExpertAction.NO_TRADE, confidence=0.0, expected_return=0.0, expected_risk=0.1):
    return ExpertOutput(
        expert_name=name, timestamp=0, symbol="BTC/USDT", action=action, confidence=confidence,
        no_trade_probability=1.0 - confidence, expected_return=expected_return, expected_risk=expected_risk,
        reward_to_risk=max(expected_return / max(expected_risk, 1e-6), 0.0),
        invalidation_distance=1.0, zone_aware=False, model_output=0.0, regime_used=RegimeType.UNKNOWN,
    )


TREND_CFG = {
    "momentum_threshold": 0.05, "regime_requirement": ["TRENDING_UP", "TRENDING_DOWN"],
    "regime_confidence_floor": 0.4, "persistence_requirement": 0.5,
    "fallback_confidence": 0.6, "fallback_expected_return": 0.08, "fallback_expected_risk": 0.12,
}


# --------------------------------------------------------------------
# Triple-barrier labeling
# --------------------------------------------------------------------

def test_triple_barrier_tp_hit_first():
    bars = [
        make_snapshot(1, 100.5, 101.0, 99.5, 100.5),
        make_snapshot(2, 100.5, 102.5, 99.8, 102.0),  # TP touched (high>=102), SL not touched
        make_snapshot(3, 100.0, 100.0, 90.0, 95.0),   # would-be SL bar, must not be reached
    ]
    outcome = label_triple_barrier(
        forward_bars=bars, entry_price=100.0, direction=1,
        take_profit_pct=2.0, stop_loss_pct=2.0, max_holding_bars=3, fee_pct=0.0,
    )
    assert outcome.exit_reason == "take_profit"
    assert outcome.bars_held == 2
    assert outcome.realized_return_pct == pytest.approx(2.0, abs=0.01)


def test_triple_barrier_sl_hit_first():
    bars = [
        make_snapshot(1, 100.0, 100.5, 99.5, 100.0),
        make_snapshot(2, 100.0, 100.5, 97.5, 98.0),  # SL touched (low<=98), TP not touched
    ]
    outcome = label_triple_barrier(
        forward_bars=bars, entry_price=100.0, direction=1,
        take_profit_pct=2.0, stop_loss_pct=2.0, max_holding_bars=3, fee_pct=0.0,
    )
    assert outcome.exit_reason == "stop_loss"
    assert outcome.bars_held == 2
    assert outcome.realized_return_pct == pytest.approx(-2.0, abs=0.01)


def test_triple_barrier_timeout():
    bars = [make_snapshot(i, 100.2, 100.6, 99.6, 100.2) for i in range(1, 4)]
    outcome = label_triple_barrier(
        forward_bars=bars, entry_price=100.0, direction=1,
        take_profit_pct=5.0, stop_loss_pct=5.0, max_holding_bars=3, fee_pct=0.0,
    )
    assert outcome.exit_reason == "timeout"
    assert outcome.bars_held == 3
    assert outcome.realized_return_pct == pytest.approx(0.2, abs=0.01)


def test_triple_barrier_no_lookahead():
    # Bars 4+ contain an extreme move that WOULD trigger TP if leaked into
    # the window; max_holding_bars=3 must never see them.
    bars = [make_snapshot(i, 100.1, 100.3, 99.9, 100.1) for i in range(1, 4)]
    bars += [make_snapshot(5, 100.0, 999.0, 1.0, 500.0)]  # extreme, out of window

    outcome_short_window = label_triple_barrier(
        forward_bars=bars, entry_price=100.0, direction=1,
        take_profit_pct=5.0, stop_loss_pct=5.0, max_holding_bars=3, fee_pct=0.0,
    )
    outcome_truncated_input = label_triple_barrier(
        forward_bars=bars[:3], entry_price=100.0, direction=1,
        take_profit_pct=5.0, stop_loss_pct=5.0, max_holding_bars=3, fee_pct=0.0,
    )
    assert outcome_short_window.exit_reason == "timeout"
    assert outcome_short_window == outcome_truncated_input


# --------------------------------------------------------------------
# Expert gate / fallback / granularity guard
# --------------------------------------------------------------------

def test_expert_gate_determinism():
    expert = TrendExpert(config=TREND_CFG, model_path=None)
    fv = make_feature_vector(momentum_5m=0.2)
    regime = make_regime_state(RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8)
    zones = make_zone_map()

    d1 = expert.propose_candidate(fv, zones, regime)
    d2 = expert.propose_candidate(fv, zones, regime)
    assert d1 == d2 == 1


def test_expert_no_candidate_skips_model():
    expert = TrendExpert(config=TREND_CFG, model_path=None)

    class ExplodingModel:
        def predict_proba(self, row):
            raise AssertionError("model should never be called when propose_candidate returns None")

    expert._model = ExplodingModel()
    expert.use_fallback_mode = False
    expert._model_feature_names = ["momentum_5m", "momentum_10m", "sma_slope", "invalidation_distance",
                                    "regime_confidence", "regime_persistence_score", "volatility_5m"]

    fv = make_feature_vector(momentum_5m=0.001)  # below momentum_threshold -> gate returns None
    regime = make_regime_state(RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8)
    zones = make_zone_map()

    output = expert.evaluate(fv, zones, regime)  # must not raise
    assert output.action == ExpertAction.NO_TRADE
    assert output.confidence == 0.0


def test_expert_fallback_without_model_file():
    expert = TrendExpert(config=TREND_CFG, model_path="data/models/experts/does_not_exist.pkl")
    assert expert.use_fallback_mode is True

    fv = make_feature_vector(momentum_5m=0.2)
    regime = make_regime_state(RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8)
    zones = make_zone_map()

    output = expert.evaluate(fv, zones, regime)
    assert output.confidence == TREND_CFG["fallback_confidence"]
    assert output.expected_return == TREND_CFG["fallback_expected_return"]


def test_expert_granularity_mismatch_fallback(tmp_path):
    model_path = tmp_path / "trend_expert.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": object(), "feature_names": ["momentum_5m"], "granularity": "30s",
            "avg_favorable_return": 0.1, "avg_unfavorable_return": -0.1, "avg_adverse_excursion": 0.1,
        }, f)

    expert = TrendExpert(config=TREND_CFG, model_path=str(model_path), data_granularity="1m")
    assert expert.use_fallback_mode is True
    assert expert._model is None


# --------------------------------------------------------------------
# Opportunity Filter
# --------------------------------------------------------------------

def test_opportunity_filter_hard_rules_override():
    hard_rules = {
        "edge_vs_fees_threshold": -1.0, "max_spread_pct": 0.02, "min_depth_usdt": 0.0,
        "volatility_floor_pct": 0.0, "volatility_ceiling_pct": 10.0, "reject_on_regime": [],
        "no_consensus_ceiling": 1.0, "reject_if_best_confidence_below": 0.0,
    }
    filt = OpportunityFilter(config={"hard_rules": hard_rules, "ml_model": {"blend_weight": 0.9}},
                              labeling_config={"fee_pct": 0.0, "slippage_buffer_pct": 0.0})

    class AlwaysTradableModel:
        classes_ = [0, 1]
        def predict_proba(self, row):
            return np.array([[0.0, 1.0]])  # model says "definitely tradable"

    filt._model = AlwaysTradableModel()
    filt._model_feature_names = OPP_FEATURE_NAMES

    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(
        make_expert_output("trend_expert", ExpertAction.BUY, confidence=0.9, expected_return=1.0, expected_risk=0.1)
    )
    regime = make_regime_state(RegimeType.TRENDING_UP)
    market = make_snapshot(0, 100, 100.1, 99.9, 100)  # spread_pct=0.01 in helper -- override below
    market = MarketSnapshot(
        timestamp=0, symbol="BTC/USDT", open=100, high=100.1, low=99.9, close=100,
        volume=100, bid=99.0, ask=101.0, bid_volume=50, ask_volume=50, spread_pct=0.10,  # > max_spread_pct=0.02
        hour_utc=12, day_of_week=2,
    )
    fv = make_feature_vector()

    score = filt.evaluate(outputs, regime, make_zone_map(), market, fv)
    assert score.tradable is False  # hard rule veto beats a model that says "tradable"
    assert any("spread_too_wide" in r for r in score.failed_rules)


def test_opportunity_filter_regime_gate():
    hard_rules = {
        "edge_vs_fees_threshold": -1.0, "max_spread_pct": 1.0, "min_depth_usdt": 0.0,
        "volatility_floor_pct": 0.0, "volatility_ceiling_pct": 10.0, "reject_on_regime": ["PANIC_SPIKE"],
        "no_consensus_ceiling": 1.0, "reject_if_best_confidence_below": 0.0,
    }
    filt = OpportunityFilter(config={"hard_rules": hard_rules}, labeling_config={"fee_pct": 0.0, "slippage_buffer_pct": 0.0})

    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(
        make_expert_output("trend_expert", ExpertAction.BUY, confidence=0.9, expected_return=1.0, expected_risk=0.1)
    )
    regime = make_regime_state(RegimeType.PANIC_SPIKE, confidence=0.9)
    market = make_snapshot(0, 100, 100.1, 99.9, 100)
    fv = make_feature_vector()

    score = filt.evaluate(outputs, regime, make_zone_map(), market, fv)
    assert score.tradable is False
    assert any("regime_blocked" in r for r in score.failed_rules)


def test_opportunity_filter_proxy_labels():
    """build_features() -- the bootstrap for the best-expert-proxy training
    labels (§7.2) -- returns exactly FEATURE_NAMES, nothing more/less."""
    from src.layers.opportunity_filter import build_features

    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(
        make_expert_output("trend_expert", ExpertAction.BUY, confidence=0.7, expected_return=0.1, expected_risk=0.1)
    )
    regime = make_regime_state(RegimeType.TRENDING_UP)
    market = make_snapshot(0, 100, 100.1, 99.9, 100)
    fv = make_feature_vector()

    features = build_features(outputs, regime, make_zone_map(), market, fv, edge_vs_fees=0.05)
    assert set(features.keys()) == set(OPP_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


# --------------------------------------------------------------------
# Router
# --------------------------------------------------------------------

def test_router_no_trade_fallback():
    router = Router(config={"fallback_behavior": {"confidence_floor_to_trade": 0.3}}, model_path=None)
    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT")
    for name in ["trend_expert", "mean_reversion_expert", "breakout_expert"]:
        outputs = outputs.add_expert(make_expert_output(name, ExpertAction.NO_TRADE, confidence=0.0))

    regime = make_regime_state(RegimeType.CHOP_RANGE)
    opp_score = OpportunityScore(timestamp=0, symbol="BTC/USDT", tradable=True, tradable_probability=1.0,
                                  environment_quality=0.5, opportunity_strength=0.0)

    decision = router.decide(outputs, regime, make_zone_map(), opp_score)
    assert decision.selected_expert == "no_trade"


def test_router_switching_context_updates():
    router = Router(config={"fallback_behavior": {"confidence_floor_to_trade": 0.1}}, model_path=None)
    opp_score = OpportunityScore(timestamp=0, symbol="BTC/USDT", tradable=True, tradable_probability=1.0,
                                  environment_quality=0.5, opportunity_strength=0.0)
    regime = make_regime_state(RegimeType.TRENDING_UP)

    outputs_a = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(
        make_expert_output("trend_expert", ExpertAction.BUY, confidence=0.8)
    ).add_expert(make_expert_output("mean_reversion_expert", ExpertAction.NO_TRADE, confidence=0.0)
    ).add_expert(make_expert_output("breakout_expert", ExpertAction.NO_TRADE, confidence=0.0))

    d1 = router.decide(outputs_a, regime, make_zone_map(), opp_score)
    assert d1.selected_expert == "trend_expert"
    assert router._bars_since_switch == 0

    d2 = router.decide(outputs_a, regime, make_zone_map(), opp_score)  # same expert wins again
    assert d2.selected_expert == "trend_expert"
    assert router._bars_since_switch == 1

    outputs_b = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(
        make_expert_output("trend_expert", ExpertAction.NO_TRADE, confidence=0.0)
    ).add_expert(make_expert_output("mean_reversion_expert", ExpertAction.BUY, confidence=0.9)
    ).add_expert(make_expert_output("breakout_expert", ExpertAction.NO_TRADE, confidence=0.0))

    d3 = router.decide(outputs_b, regime, make_zone_map(), opp_score)  # switches
    assert d3.selected_expert == "mean_reversion_expert"
    assert router._bars_since_switch == 0


def test_router_granularity_mismatch_fallback(tmp_path):
    model_path = tmp_path / "router.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": object(), "feature_names": ["prob_trending_up"], "granularity": "30s"}, f)

    router = Router(config={}, model_path=str(model_path), data_granularity="1m")
    assert router._model is None  # mismatch treated as absent


# --------------------------------------------------------------------
# Config loading + timeframe adaptation
# --------------------------------------------------------------------

def test_yaml_config_loading():
    with open("config/settings.base.yaml") as f:
        config = yaml.safe_load(f)

    for block in ["system", "labeling", "experts", "opportunity_filter", "router", "models", "backtest"]:
        assert block in config, f"missing top-level config block: {block}"

    for expert_name in ["trend", "mean_reversion", "breakout"]:
        assert expert_name in config["experts"]
        assert "fallback_confidence" in config["experts"][expert_name]
        assert "model_path" in config["experts"][expert_name]


def test_timeframe_adaptation():
    """Switching system.data_granularity is a config change, not a code
    change -- build_experts() + evaluate() must run cleanly for any
    granularity string with no models trained yet (pure fallback mode)."""
    with open("config/settings.base.yaml") as f:
        config = yaml.safe_load(f)

    experts = build_experts(config, data_granularity="30s")
    assert all(e.data_granularity == "30s" for e in experts.values())
    assert all(e.use_fallback_mode for e in experts.values())  # no models trained yet

    fv = make_feature_vector(momentum_5m=0.2)
    regime = make_regime_state(RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8)
    zones = make_zone_map()

    for expert in experts.values():
        output = expert.evaluate(fv, zones, regime)  # must not raise
        assert output is not None
