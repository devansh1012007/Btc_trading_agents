"""
Structured logging tests (PRESENTATION §8): JSONL system log captures
records with extras; decision stream emits one §8-shaped record per call;
setup_logging is idempotent.
"""
import json
import logging

import pytest

from src.state.contracts import (
    ExecutionRecord,
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    OpportunityScore,
    RegimeState,
    RegimeType,
    RiskDecision,
    RouterDecision,
    SystemState,
)
from src.utils.logging_setup import (
    JsonLineFormatter,
    decision_record_from_state,
    log_trade_decision,
    setup_logging,
)

pytestmark = [pytest.mark.unit]


def test_json_formatter_inlines_extras(tmp_path):
    logger = logging.getLogger("test.jsonfmt")
    handler = logging.FileHandler(tmp_path / "system.jsonl", encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.warning("breaker tripped", extra={"breaker_type": "max_drawdown", "drawdown_pct": 9.1})

    handler.flush()
    record = json.loads((tmp_path / "system.jsonl").read_text().strip())
    assert record["level"] == "WARNING"
    assert record["message"] == "breaker tripped"
    assert record["breaker_type"] == "max_drawdown"
    assert record["drawdown_pct"] == 9.1


def test_setup_logging_writes_jsonl_file(tmp_path):
    setup_logging(json_log_file=str(tmp_path / "sys.jsonl"))
    logging.getLogger("test.setup").info("hello structured world")

    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = (tmp_path / "sys.jsonl").read_text().strip().splitlines()
    assert any('"hello structured world"' in line for line in lines)
    payload = json.loads(lines[-1])
    assert {"ts", "level", "logger", "message"} <= set(payload)


def _make_state():
    regime = RegimeState(
        timestamp=100, symbol="BTC/USDT", prob_trending_up=0.71, prob_trending_down=0.05,
        prob_chop_range=0.24, prob_panic_spike=0.0, primary_regime=RegimeType.TRENDING_UP,
        regime_confidence=0.71, regime_age=5, regime_persistence_score=0.8,
    )
    output = ExpertOutput(
        expert_name="breakout_expert", timestamp=100, symbol="BTC/USDT", action=ExpertAction.BUY,
        confidence=0.84, no_trade_probability=0.16, expected_return=0.42, expected_risk=0.18,
        reward_to_risk=2.3, invalidation_distance=0.5, zone_aware=True, model_output=0.84,
        regime_used=RegimeType.TRENDING_UP,
    )
    router = RouterDecision(timestamp=100, symbol="BTC/USDT", selected_expert="breakout_expert",
                            selection_confidence=0.76, no_trade_probability=0.1)
    opp = OpportunityScore(
        timestamp=100, symbol="BTC/USDT", tradable=True, tradable_probability=0.82,
        environment_quality=0.7, opportunity_strength=0.75, failed_rules=[],
    )
    risk = RiskDecision(
        timestamp=100, symbol="BTC/USDT", approved=True, risk_score=0.19,
        base_size=500.0, adjusted_size=480.0, size_reduction_pct=4.0, rejection_reason=None,
    )
    execution = ExecutionRecord(
        timestamp=100, order_timestamp=100, fill_timestamp=101, symbol="BTC/USDT",
        side="BUY", order_price=50_000.0, fill_price=50_010.0, requested_size=480.0,
        filled_size=480.0, slippage_pct=0.02, fees_pct=0.04, total_cost_pct=0.06,
        status="filled",
    )
    return SystemState(
        timestamp=100, symbol="BTC/USDT",
        market_snapshot=None, feature_vector=None, zone_map=None, regime_state=regime,
        expert_outputs=ExpertOutputs(timestamp=100, symbol="BTC/USDT",
                                     outputs={"breakout_expert": output}),
        opportunity_score=opp, router_decision=router, risk_decision=risk,
        execution_record=execution,
    )


def test_decision_record_matches_presentation_shape():
    record = decision_record_from_state(_make_state())

    # The exact observability fields PRESENTATION §8 asks for:
    assert record["regime"] == "trending_up"
    assert record["expert"] == "breakout_expert"
    assert record["opportunity_score"] == pytest.approx(0.82)
    assert record["router_confidence"] == pytest.approx(0.76)
    assert record["risk_score"] == pytest.approx(0.19)
    assert record["decision"] == "LONG"


def test_decision_stream_is_append_only_jsonl(tmp_path):
    path = str(tmp_path / "decisions.jsonl")
    state = _make_state()
    log_trade_decision(decision_record_from_state(state), path=path)
    log_trade_decision(decision_record_from_state(state), path=path)

    lines = [json.loads(line) for line in open(path)]
    assert len(lines) == 2
    assert all(line["symbol"] == "BTC/USDT" for line in lines)
