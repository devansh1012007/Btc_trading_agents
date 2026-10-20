"""
Decision-trace helpers: the shared terminal-stage classifier and the
shared stage builder.

Both are used by the deployment loop and by the Trade Map GUI, so a
regression here would silently change what the GUI reports about why a
bar stopped where it did.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics.decision_trace import (  # noqa: E402
    DecisionTrace,
    StageResult,
    build_trace_stages,
    classify_terminal_stage,
)
from src.state.contracts import (  # noqa: E402
    ExecutionRecord,
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    OpportunityScore,
    RiskDecision,
    RouterDecision,
    SystemState,
)

pytestmark = [pytest.mark.unit]

TS = 1_700_000_000


def make_opportunity(tradable: bool = True, failed=None) -> OpportunityScore:
    return OpportunityScore(
        timestamp=TS, symbol="BTC/USDT", tradable=tradable, tradable_probability=0.6,
        environment_quality=0.5, opportunity_strength=0.5, failed_rules=failed or [],
    )


def make_router(selected: str = "trend_expert", reasoning=None) -> RouterDecision:
    return RouterDecision(
        timestamp=TS, symbol="BTC/USDT", selected_expert=selected,
        selection_confidence=0.7, no_trade_probability=0.3,
        expert_ranking={}, reasoning=reasoning,
    )


def make_risk(approved: bool = False, reason: str = "too risky") -> RiskDecision:
    return RiskDecision(
        timestamp=TS, symbol="BTC/USDT", approved=approved, risk_score=0.9,
        base_size=100.0, adjusted_size=0.0, size_reduction_pct=100.0,
        rejection_reason=None if approved else reason,
    )


def make_execution(status: str = "filled", side: str = "BUY") -> ExecutionRecord:
    return ExecutionRecord(
        timestamp=TS, order_timestamp=TS, fill_timestamp=TS, symbol="BTC/USDT",
        side=side, order_price=100.0, fill_price=100.0, requested_size=100.0,
        filled_size=100.0, slippage_pct=0.0, fees_pct=0.04, total_cost_pct=0.04,
        status=status, order_id="x",
    )


# ---------------------------------------------------------------------------
# classify_terminal_stage
# ---------------------------------------------------------------------------

def test_filled_execution_is_the_executed_terminal_stage():
    stage, reason = classify_terminal_stage(
        make_opportunity(True), make_router("trend_expert"), make_risk(True),
        make_execution("filled"),
    )
    assert stage == "executed"
    assert "trend_expert" in reason


def test_filter_veto_beats_the_routers_no_trade():
    """The router returns no_trade with reasoning 'opportunity_filter_rejected'
    when the filter vetoes -- that must not be labelled a router decline."""
    stage, _ = classify_terminal_stage(
        make_opportunity(False, ["edge_below_threshold(-0.02)"]),
        make_router("no_trade", "opportunity_filter_rejected"),
        make_risk(False, "no_trade_selected"),
        None,
    )
    assert stage == "opportunity_filter"


def test_risk_layer_rejection_is_not_chosen_when_the_router_declined():
    """The risk layer rejects with 'no_trade_selected' on every bar the
    router already declined; that is a consequence, not a risk rejection."""
    stage, _ = classify_terminal_stage(
        make_opportunity(True),
        make_router("no_trade", "router_model_no_trade"),
        make_risk(False, "no_trade_selected"),
        None,
    )
    assert stage == "router_no_trade"


def test_no_expert_clearing_the_floor_is_not_a_router_decline():
    stage, _ = classify_terminal_stage(
        make_opportunity(True),
        make_router("no_trade", "no_expert_cleared_confidence_floor"),
        None, None,
    )
    assert stage == "no_expert_candidate"


def test_real_risk_rejection_is_reported_when_the_router_picked_an_expert():
    stage, reason = classify_terminal_stage(
        make_opportunity(True), make_router("trend_expert"),
        make_risk(False, "max_position_cap"), None,
    )
    assert stage == "risk_layer"
    assert reason == "max_position_cap"


def test_rejected_execution_is_reported_as_execution_failed():
    stage, reason = classify_terminal_stage(
        make_opportunity(True), make_router("trend_expert"), make_risk(True),
        make_execution("rejected"),
    )
    assert stage == "execution_failed"
    assert "rejected" in reason


def test_nothing_at_all_falls_back_to_no_expert_candidate():
    stage, _ = classify_terminal_stage(None, None, None, None)
    assert stage == "no_expert_candidate"


# ---------------------------------------------------------------------------
# build_trace_stages
# ---------------------------------------------------------------------------

def make_state(expert_outputs=None, opportunity=None, router=None,
               risk=None, execution=None) -> SystemState:
    return SystemState(
        timestamp=TS, symbol="BTC/USDT", market_snapshot=None,
        expert_outputs=expert_outputs, opportunity_score=opportunity,
        router_decision=router, risk_decision=risk, execution_record=execution,
    )


def test_build_trace_stages_lists_the_stages_in_pipeline_order():
    outputs = ExpertOutputs(timestamp=TS, symbol="BTC/USDT")
    outputs = outputs.add_expert(ExpertOutput(
        expert_name="trend_expert", timestamp=TS, symbol="BTC/USDT",
        action=ExpertAction.BUY, confidence=0.7, no_trade_probability=0.2,
        expected_return=0.1, expected_risk=0.1, reward_to_risk=1.0,
        invalidation_distance=0.5, zone_aware=False, model_output=0.7,
    ))
    state = make_state(
        expert_outputs=outputs,
        opportunity=make_opportunity(True),
        router=make_router("trend_expert"),
        risk=make_risk(True),
        execution=make_execution("filled"),
    )
    stages = build_trace_stages(state)
    assert [s.stage_name for s in stages] == [
        "expert_layer", "opportunity_filter", "router", "risk_layer", "execution",
    ]
    assert all(isinstance(s, StageResult) for s in stages)


def test_build_trace_stages_omits_stages_with_no_result():
    state = make_state(opportunity=make_opportunity(True))
    stages = build_trace_stages(state)
    names = [s.stage_name for s in stages]
    assert "opportunity_filter" in names
    assert "router" not in names
    assert "risk_layer" not in names
    assert "execution" not in names


def test_build_trace_stages_marks_no_trade_experts_as_failed():
    outputs = ExpertOutputs(timestamp=TS, symbol="BTC/USDT")
    outputs = outputs.add_expert(ExpertOutput(
        expert_name="trend_expert", timestamp=TS, symbol="BTC/USDT",
        action=ExpertAction.NO_TRADE, confidence=0.1, no_trade_probability=0.9,
        expected_return=0.0, expected_risk=0.0, reward_to_risk=0.0,
        invalidation_distance=0.0, zone_aware=False, model_output=0.0,
    ))
    stages = build_trace_stages(make_state(expert_outputs=outputs))
    expert_stage = next(s for s in stages if s.stage_name == "expert_layer")
    assert expert_stage.passed is False


def test_decision_trace_accepts_the_optional_latency_fields():
    trace = DecisionTrace(
        timestamp=TS, symbol="BTC/USDT", stages=[],
        terminal_stage="executed", terminal_reason="filled via trend_expert",
        total_duration_ms=1.5, stage_timings_ms={"features_ms": 0.2},
    )
    assert trace.total_duration_ms == 1.5
    assert trace.stage_timings_ms == {"features_ms": 0.2}
