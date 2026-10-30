"""
Unit tests for Phase 12 T12.2 / Leakage Control L9: `generate_oof_predictions`
and `verify_oof_no_in_sample_leakage`.
"""
import numpy as np
import pytest

from src.training.plan import (
    OOFLeakageError,
    generate_oof_predictions,
    verify_oof_no_in_sample_leakage,
)

pytestmark = [pytest.mark.unit]


def test_generate_oof_predictions_is_purged_and_rejects_in_sample_rows():
    bars = np.arange(0, 300)
    X = np.column_stack([bars.astype(float), np.ones(300)])
    y = (bars % 2).astype(float)

    def fit_pred(X_tr, y_tr, X_te):
        mean_y = float(np.mean(y_tr))
        return np.full(len(X_te), mean_y)

    oof = generate_oof_predictions(
        bars, X, y, fit_pred,
        train_bars=100, test_bars=40, step_bars=40, embargo_bars=5, label_horizon_bars=5,
    )
    assert len(oof) > 0
    verify_oof_no_in_sample_leakage(oof)

    # Inject an in-sample row -> must raise OOFLeakageError (L9)
    bad = list(oof) + [{"bar_index": 50, "train_max_bar_index": 100, "embargo_bars": 5, "is_oof": False, "split": "train"}]
    with pytest.raises(OOFLeakageError, match="L9"):
        verify_oof_no_in_sample_leakage(bad)
