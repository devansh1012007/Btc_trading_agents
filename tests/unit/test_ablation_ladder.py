"""
Unit tests for Phase 17 T17.1: `ABLATION_RUNGS` (`L0`..`L10`),
`holm_bonferroni_adjust`, and `FinalEvaluationGate` (`L10` holdout reuse guard).
"""
import pytest

from src.evaluation.ablation_ladder import (
    ABLATION_RUNGS,
    AblationRungResult,
    FinalEvaluationGate,
    HoldoutReuseError,
    holm_bonferroni_adjust,
)

pytestmark = [pytest.mark.unit]


def test_holm_bonferroni_and_final_evaluation_gate_prevents_holdout_reuse():
    assert len(ABLATION_RUNGS) == 11
    adj = holm_bonferroni_adjust([0.01, 0.04, 0.03])
    # Sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=max(0.06,0.04)=0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)

    rungs = [
        AblationRungResult(
            rung_id=r,
            expectancy_pct=0.10 + 0.01 * i,
            profit_factor=1.2,
            max_drawdown_pct=3.0,
            brier_score=0.18,
            min_duration_violations=0,
            p_value_vs_prev=0.01,
        )
        for i, r in enumerate(ABLATION_RUNGS)
    ]
    gate = FinalEvaluationGate()
    rep = gate.evaluate_holdout(
        "bundle_sha_1",
        rungs,
        holdout_metrics={"expectancy_pct": 0.22, "max_drawdown_pct": 3.5, "min_duration_violations": 0},
    )
    assert rep.passed

    # Evaluating the same bundle_hash a second time on the locked holdout box raises HoldoutReuseError (L10)!
    with pytest.raises(HoldoutReuseError, match="L10 violation"):
        gate.evaluate_holdout(
            "bundle_sha_1",
            rungs,
            holdout_metrics={"expectancy_pct": 0.22, "max_drawdown_pct": 3.5, "min_duration_violations": 0},
        )
