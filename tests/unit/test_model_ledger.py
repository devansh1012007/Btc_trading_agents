"""
Unit tests for Phase 14 T14.1 / Leakage Control L8: `ModelLedger` append-only
prediction/outcome storage and strict `as_of_timestamp` causal join.
"""
import pytest

from src.monitoring.ledger import (
    ModelLedger,
    ModelOutcomeRecord,
    ModelPredictionRecord,
)

pytestmark = [pytest.mark.unit]


def test_model_ledger_query_resolved_never_leaks_future_outcomes():
    ledger = ModelLedger()
    ledger.record_prediction(
        ModelPredictionRecord(
            prediction_id="p1",
            timestamp=1000,
            model_name="trend_expert",
            model_version_hash="v1",
            predicted_prob=0.75,
            predicted_value=0.40,
            regime="trending_up",
            trade_id="t1",
        )
    )
    ledger.record_Outcome = ledger.record_outcome
    ledger.record_outcome(
        ModelOutcomeRecord(
            prediction_id="p1",
            resolved_timestamp=1300,
            realized_label=1,
            realized_value=0.55,
            trade_id="t1",
        )
    )

    # At t=1200 (before p1 resolves at t=1300): query_resolved MUST return empty list (L8)!
    assert ledger.query_resolved(as_of_timestamp=1200) == []

    # At t=1300 (once resolved): p1 is returned!
    resolved = ledger.query_resolved(as_of_timestamp=1300)
    assert len(resolved) == 1
    assert resolved[0].prediction_id == "p1"
    assert resolved[0].realized_value == pytest.approx(0.55)
