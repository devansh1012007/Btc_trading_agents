"""
Feature Drift Monitor (Phase 4, §9): compares a recent live feature
distribution against a model's training-time distribution via Population
Stability Index (PSI) -- the standard convention: under 0.1 stable,
0.1-0.25 moderate shift, over 0.25 significant (configurable via
monitoring.drift.psi_alert_threshold).
"""
from typing import Any, Dict, List, Optional

import numpy as np


def _psi_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from the reference distribution, so each
    reference bin has roughly equal weight."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
    over bins derived from the reference distribution's quantiles.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) < n_bins or len(current) == 0:
        return 0.0

    edges = _psi_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_feature_drift(
    reference: Dict[str, List[float]], current: Dict[str, List[float]],
    psi_alert_threshold: float = 0.25,
) -> Dict[str, Dict]:
    """
    reference/current: {feature_name: [values]}. Returns
    {feature_name: {"psi": float, "alert": bool}} for every feature
    present in both.
    """
    results = {}
    for feature_name in reference:
        if feature_name not in current:
            continue
        psi = population_stability_index(np.array(reference[feature_name]), np.array(current[feature_name]))
        results[feature_name] = {"psi": psi, "alert": psi >= psi_alert_threshold}
    return results


def check_regime_distribution(
    recent_regimes: List[str], panic_frequency_alert_threshold: float = 0.15,
    low_confidence_alert_threshold: float = 0.35, recent_confidences: Optional[List[float]] = None,
) -> Dict:
    """Production-health check: unusual PANIC_SPIKE frequency, or
    sustained low regime_confidence -- the check that's easy to build once
    and then never look at again if it isn't automated."""
    if not recent_regimes:
        return {"panic_frequency": 0.0, "panic_alert": False, "avg_confidence": None, "low_confidence_alert": False}

    panic_frequency = sum(1 for r in recent_regimes if r == "panic_spike") / len(recent_regimes)
    result: Dict[str, Any] = {"panic_frequency": panic_frequency, "panic_alert": panic_frequency > panic_frequency_alert_threshold}

    if recent_confidences:
        avg_confidence = float(np.mean(recent_confidences))
        result["avg_confidence"] = avg_confidence
        result["low_confidence_alert"] = avg_confidence < low_confidence_alert_threshold
    else:
        result["avg_confidence"] = None
        result["low_confidence_alert"] = False

    return result
