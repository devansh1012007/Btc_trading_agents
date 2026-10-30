"""
Append-only ML prediction & outcome ledger (`ModelLedger`, Phase 14 T14.1).

Records every model prediction (`record_prediction`) and links it to its
realized outcome (`record_outcome`) once the forward window or trade closes.

Critical Leakage Control (L8):
`query_resolved(as_of_timestamp)` joins predictions to outcomes strictly where
`outcome.resolved_timestamp <= as_of_timestamp`. A prediction whose outcome
resolves at `t_resolve > as_of_timestamp` is NEVER returned as resolved at
`as_of_timestamp`, eliminating look-ahead in rolling scorecards and drift/degradation
alarms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..registry.db import MEMORY_PATH, connect, open_connection


@dataclass(frozen=True)
class ModelPredictionRecord:
    prediction_id: str
    timestamp: int
    model_name: str
    model_version_hash: str
    predicted_prob: float
    predicted_value: float = 0.0
    regime: str = "unknown"
    trade_id: Optional[str] = None
    is_shadow: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutcomeRecord:
    prediction_id: str
    resolved_timestamp: int
    realized_label: int
    realized_value: float
    trade_id: Optional[str] = None


@dataclass(frozen=True)
class ResolvedModelRecord:
    prediction_id: str
    prediction_timestamp: int
    resolved_timestamp: int
    model_name: str
    model_version_hash: str
    predicted_prob: float
    predicted_value: float
    realized_label: int
    realized_value: float
    regime: str
    trade_id: Optional[str]
    is_shadow: bool


class ModelLedger:
    """Append-only prediction/outcome ledger backed by SQLite (`model_predictions`, `model_outcomes`) or memory."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or MEMORY_PATH
        self._private = open_connection(MEMORY_PATH) if self.db_path == MEMORY_PATH else None
        self._predictions: Dict[str, ModelPredictionRecord] = {}
        self._outcomes: Dict[str, ModelOutcomeRecord] = {}

    def _conn(self):
        if self._private is not None:
            return self._private
        return connect(self.db_path)

    def record_prediction(self, rec: ModelPredictionRecord) -> None:
        self._predictions[rec.prediction_id] = rec
        payload_json = json.dumps({
            "predicted_prob": rec.predicted_prob,
            "predicted_value": rec.predicted_value,
            "regime": rec.regime,
            "trade_id": rec.trade_id,
            "is_shadow": rec.is_shadow,
            **rec.metadata,
        })
        kind = "shadow" if rec.is_shadow else "active"
        if self._private is not None:
            self._private.execute(
                "INSERT OR REPLACE INTO model_predictions "
                "(id, bar_ts, model_id, model_version, kind, output_json, regime_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rec.prediction_id, rec.timestamp, rec.model_name, rec.model_version_hash, kind, payload_json, rec.regime),
            )
            self._private.commit()
        else:
            with connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO model_predictions "
                    "(id, bar_ts, model_id, model_version, kind, output_json, regime_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rec.prediction_id, rec.timestamp, rec.model_name, rec.model_version_hash, kind, payload_json, rec.regime),
                )

    def record_outcome(self, out: ModelOutcomeRecord) -> None:
        self._outcomes[out.prediction_id] = out
        outcome_json = json.dumps({
            "realized_label": out.realized_label,
            "realized_value": out.realized_value,
            "trade_id": out.trade_id,
        })
        if self._private is not None:
            self._private.execute(
                "INSERT OR REPLACE INTO model_outcomes "
                "(prediction_id, outcome_ts, realized_json) VALUES (?, ?, ?)",
                (out.prediction_id, out.resolved_timestamp, outcome_json),
            )
            self._private.commit()
        else:
            with connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO model_outcomes "
                    "(prediction_id, outcome_ts, realized_json) VALUES (?, ?, ?)",
                    (out.prediction_id, out.resolved_timestamp, outcome_json),
                )

    def query_resolved(
        self,
        as_of_timestamp: float,
        *,
        model_name: Optional[str] = None,
        include_shadow: bool = False,
    ) -> List[ResolvedModelRecord]:
        """Return resolved predictions strictly where `outcome.resolved_timestamp <= as_of_timestamp` (L8)."""
        results: List[ResolvedModelRecord] = []
        for pid, pred in self._predictions.items():
            if pred.timestamp > as_of_timestamp:
                continue
            if model_name is not None and pred.model_name != model_name:
                continue
            if not include_shadow and pred.is_shadow:
                continue
            out = self._outcomes.get(pid)
            if out is None or out.resolved_timestamp > as_of_timestamp:
                continue
            results.append(
                ResolvedModelRecord(
                    prediction_id=pid,
                    prediction_timestamp=pred.timestamp,
                    resolved_timestamp=out.resolved_timestamp,
                    model_name=pred.model_name,
                    model_version_hash=pred.model_version_hash,
                    predicted_prob=pred.predicted_prob,
                    predicted_value=pred.predicted_value,
                    realized_label=out.realized_label,
                    realized_value=out.realized_value,
                    regime=pred.regime,
                    trade_id=out.trade_id or pred.trade_id,
                    is_shadow=pred.is_shadow,
                )
            )
        results.sort(key=lambda r: (r.resolved_timestamp, r.prediction_id))
        return results
