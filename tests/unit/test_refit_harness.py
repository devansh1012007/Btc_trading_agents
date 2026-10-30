"""
Unit tests for Phase 12 T12.3: `RefitWalkForwardHarness` per-fold refit and
embargo verification.
"""
import numpy as np
import pytest

from src.training.plan import RefitWalkForwardHarness

pytestmark = [pytest.mark.unit]


def test_refit_walk_forward_harness_refits_per_fold_with_embargo():
    bars = np.arange(0, 350)
    harness = RefitWalkForwardHarness(
        train_bars=120, test_bars=50, step_bars=50, embargo_bars=6, label_horizon_bars=5
    )

    def fit_fold(fold_id: int, train_idx: np.ndarray) -> str:
        return f"bundle_fold_{fold_id}_n{len(train_idx)}"

    def eval_fold(fold_id: int, test_idx: np.ndarray, bundle_hash: str):
        return [{"fold_id": fold_id, "n_test": len(test_idx), "bundle_hash": bundle_hash}]

    results = harness.run_folds(bars, fit_fold, eval_fold)
    assert len(results) >= 3
    for rec in harness.fold_records:
        assert rec.train_end_bar + 6 <= rec.test_start_bar
