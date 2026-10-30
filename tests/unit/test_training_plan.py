"""
Unit tests for Phase 12 T12.1: `TrainingPlan` DAG topological order,
content-addressed cache reuse, and downstream invalidation.
"""
import pytest

from src.training.plan import TRAINING_DAG_STAGES, TrainingPlan

pytestmark = [pytest.mark.unit]


def test_training_plan_follows_exact_harness_order_and_invalidates_downstream_cache():
    plan_a = TrainingPlan({"training_stage_configs": {"barrier_sizing": {"q_sl": 0.65}}}, data_fingerprint="df_1")
    assert plan_a.stage_names() == [name for (name, _) in TRAINING_DAG_STAGES]
    table_txt = plan_a.render_plan_table()
    assert "barrier_sizing" in table_txt and "opportunity_confirm" in table_txt

    cache = {}
    plan_a.execute(cache_store=cache)
    assert all(n.status == "executed" for n in plan_a.nodes)

    # Second run with identical config + data hits cache on every stage
    plan_a2 = TrainingPlan({"training_stage_configs": {"barrier_sizing": {"q_sl": 0.65}}}, data_fingerprint="df_1")
    plan_a2.execute(cache_store=cache)
    assert all(n.status == "cached" for n in plan_a2.nodes)

    # Changing barrier_sizing config keeps upstream stages (dataset, regime, zone_selection, zone_survival)
    # cached while invalidating barrier_sizing and all downstream stages (experts, exit_models, routers, opportunity_confirm, risk_sizing)!
    plan_b = TrainingPlan({"training_stage_configs": {"barrier_sizing": {"q_sl": 0.75}}}, data_fingerprint="df_1")
    plan_b.execute(cache_store=cache)
    status_by_name = {n.name: n.status for n in plan_b.nodes}
    assert status_by_name["dataset"] == "cached"
    assert status_by_name["regime"] == "cached"
    assert status_by_name["zone_selection"] == "cached"
    assert status_by_name["zone_survival"] == "cached"
    assert status_by_name["barrier_sizing"] == "executed"
    assert status_by_name["experts"] == "executed"
    assert status_by_name["exit_models"] == "executed"
    assert status_by_name["routers"] == "executed"
    assert status_by_name["opportunity_confirm"] == "executed"
    assert status_by_name["risk_sizing"] == "executed"
