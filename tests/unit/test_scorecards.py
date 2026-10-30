"""
Unit tests for Phase 14 T14.2: `compute_model_scorecard` and `build_all_scorecards`
(Brier, ECE, log-loss, rank IC, shadow comparison, and degradation alarms).
"""
import pytest

from src.monitoring.ledger import (
    ModelLedger,
    ModelOutcomeRecord,
    ModelPredictionRecord,
)
from src.monitoring.scorecards import build_all_scorecards

pytestmark = [pytest.mark.unit]


def test_scorecards_compute_metrics_and_fire_degradation_alarm():
    ledger = ModelLedger()
    # Record 12 poorly calibrated predictions (predicting 0.90 when outcome is 0 and return is -0.8%)
    for i in range(12):
        pid = f"bad_{i}"
        ledger.record_prediction(
            ModelPredictionRecord(
                prediction_id=pid,
                timestamp=1000 + i * 10,
                model_name="trend_expert",
                model_version_hash="v1",
                predicted_prob=0.90,
                predicted_value=0.60,
            )
        )
        ledger.record_outcome(
            ModelOutcomeRecord(
                prediction_id=pid,
                resolved_timestamp=1050 + i * 10,
                realized_label=0,
                realized_value=-0.80,
            )
        )

    cards = build_all_scorecards(
        ledger,
        as_of_timestamp=2000,
        max_brier_threshold=0.25,
        min_expectancy_threshold=-0.20,
        min_samples_for_alarm=10,
    )
    sc = cards["trend_expert"]
    assert sc.n_resolved == 12
    assert sc.brier_score == pytest.approx(0.81)
    assert sc.degraded is True
    assert len(sc.alarms) == 2
