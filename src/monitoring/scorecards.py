"""
Rolling per-model scorecards, calibration/Brier/rank-IC metrics, shadow
comparison, and degradation alarms (`compute_model_scorecard`, Phase 14 T14.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ledger import ModelLedger, ResolvedModelRecord


@dataclass(frozen=True)
class ModelScorecard:
    model_name: str
    as_of_timestamp: int
    n_resolved: int
    brier_score: float
    log_loss: float
    ece: float
    rank_ic: float
    hit_rate: float
    realized_expectancy: float
    degraded: bool = False
    alarms: List[str] = field(default_factory=list)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(probs))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if np.any(mask):
            avg_p = float(np.mean(probs[mask]))
            avg_y = float(np.mean(labels[mask]))
            ece += (float(np.sum(mask)) / n) * abs(avg_p - avg_y)
    return float(ece)


def _compute_rank_ic(preds: np.ndarray, realized: np.ndarray) -> float:
    if len(preds) < 2:
        return 0.0
    # Rank correlation (Spearman via argsort ranks)
    r_x = np.argsort(np.argsort(preds)).astype(float)
    r_y = np.argsort(np.argsort(realized)).astype(float)
    sx = float(np.std(r_x))
    sy = float(np.std(r_y))
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.corrcoef(r_x, r_y)[0, 1])


def compute_model_scorecard(
    records: Sequence[ResolvedModelRecord],
    *,
    model_name: str,
    as_of_timestamp: int,
    window_size: int = 100,
    max_brier_threshold: float = 0.30,
    min_expectancy_threshold: float = -0.50,
    min_samples_for_alarm: int = 10,
) -> ModelScorecard:
    window = list(records[-max(1, int(window_size)) :])
    n = len(window)
    if n == 0:
        return ModelScorecard(
            model_name=model_name,
            as_of_timestamp=as_of_timestamp,
            n_resolved=0,
            brier_score=0.0,
            log_loss=0.0,
            ece=0.0,
            rank_ic=0.0,
            hit_rate=0.0,
            realized_expectancy=0.0,
            degraded=False,
            alarms=[],
        )

    probs = np.clip(np.array([r.predicted_prob for r in window], dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.array([float(r.realized_label) for r in window], dtype=float)
    pred_vals = np.array([float(r.predicted_value or r.predicted_prob) for r in window], dtype=float)
    real_vals = np.array([float(r.realized_value) for r in window], dtype=float)

    brier = float(np.mean((probs - labels) ** 2))
    logloss = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
    ece = _compute_ece(probs, labels)
    ic = _compute_rank_ic(pred_vals, real_vals)
    hit_rate = float(np.mean((probs >= 0.5).astype(float) == labels))
    expectancy = float(np.mean(real_vals))

    alarms: List[str] = []
    if n >= min_samples_for_alarm:
        if brier > max_brier_threshold:
            alarms.append(f"brier_exceeded({brier:.3f}>{max_brier_threshold:.3f})")
        if expectancy < min_expectancy_threshold:
            alarms.append(f"expectancy_below_floor({expectancy:.3f}<{min_expectancy_threshold:.3f})")

    return ModelScorecard(
        model_name=model_name,
        as_of_timestamp=as_of_timestamp,
        n_resolved=n,
        brier_score=brier,
        log_loss=logloss,
        ece=ece,
        rank_ic=ic,
        hit_rate=hit_rate,
        realized_expectancy=expectancy,
        degraded=len(alarms) > 0,
        alarms=alarms,
    )


def build_all_scorecards(
    ledger: ModelLedger,
    as_of_timestamp: int,
    *,
    window_size: int = 100,
    max_brier_threshold: float = 0.30,
    min_expectancy_threshold: float = -0.50,
    min_samples_for_alarm: int = 10,
) -> Dict[str, ModelScorecard]:
    all_resolved = ledger.query_resolved(as_of_timestamp, include_shadow=True)
    by_model: Dict[str, List[ResolvedModelRecord]] = {}
    for rec in all_resolved:
        key = f"{rec.model_name}:shadow" if rec.is_shadow else rec.model_name
        by_model.setdefault(key, []).append(rec)

    return {
        mname: compute_model_scorecard(
            recs,
            model_name=mname,
            as_of_timestamp=as_of_timestamp,
            window_size=window_size,
            max_brier_threshold=max_brier_threshold,
            min_expectancy_threshold=min_expectancy_threshold,
            min_samples_for_alarm=min_samples_for_alarm,
        )
        for mname, recs in sorted(by_model.items())
    }
