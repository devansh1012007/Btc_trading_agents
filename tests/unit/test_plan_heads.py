"""
Unit tests for Phase 9 T9.3: `PlanUtilityHead` fitting and `ExitPlan` selection.
"""
import numpy as np
import pytest

from src.experts.plan_head import DEFAULT_PLAN_GRID, PlanUtilityHead

pytestmark = [pytest.mark.unit]


def test_plan_utility_head_selects_highest_utility_template():
    rng = np.random.default_rng(7)
    n = 60
    X = rng.normal(size=(n, 2))
    # Make template 2 ("wide_runner") clearly superior when feature 0 > 0
    rets = np.column_stack([
        np.full(n, 0.1),
        np.full(n, 0.2),
        0.8 + 0.5 * X[:, 0],
    ])
    maes = np.column_stack([
        np.full(n, 0.3),
        np.full(n, 0.3),
        np.full(n, 0.2),
    ])
    head = PlanUtilityHead(["f0", "f1"], grid=DEFAULT_PLAN_GRID, risk_aversion=0.3).fit(X, rets, maes)
    plan, pred_ret, pred_mae = head.select_best_plan({"f0": 1.5, "f1": 0.0}, volatility_pct=0.5, hard_max_loss_pct=0.4)
    assert plan.source == "expert"
    assert plan.trailing is not None and plan.trailing.enabled
    assert plan.max_holding_bars == 8
    assert plan.effective_stop_loss_pct == pytest.approx(0.4)
    assert pred_ret > pred_mae
