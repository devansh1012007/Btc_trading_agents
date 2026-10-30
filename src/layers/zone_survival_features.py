"""
Shared feature extraction for the zone-survival model (§6.5, Phase 2; Phase 2 T2.2).
Used by labeling, training, and serving-time blend so they can never drift
apart. Supports both the legacy 4-regime set and dynamic K-regime sets via
`regime_prob_features`.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..state.contracts import LEGACY_REGIME_SET, RegimeSet, RegimeState, Zone, regime_prob_features

BASE_ZONE_FEATURE_NAMES = [
    "strength", "age", "touch_count", "zone_width_pct",
    "width_to_volatility_ratio", "liquidity_score", "hour_utc", "day_of_week",
]

ZONE_SURVIVAL_FEATURE_NAMES = (
    BASE_ZONE_FEATURE_NAMES + regime_prob_features(LEGACY_REGIME_SET, prefix="regime_")
)


def zone_survival_feature_names(regime_set: Optional[RegimeSet] = None) -> List[str]:
    """Feature names for the zone survival model under ``regime_set``."""
    rset = regime_set or LEGACY_REGIME_SET
    return BASE_ZONE_FEATURE_NAMES + regime_prob_features(rset, prefix="regime_")


def zone_survival_features(
    zone: Zone,
    recent_volatility_pct: float,
    hour_utc: int,
    day_of_week: int,
    regime_state: Optional[RegimeState],
    regime_set: Optional[RegimeSet] = None,
) -> Dict[str, float]:
    rset = regime_set or LEGACY_REGIME_SET
    zone_width_pct = (zone.upper_bound - zone.lower_bound) / zone.mid * 100 if zone.mid else 0.0
    width_to_volatility_ratio = (
        zone_width_pct / recent_volatility_pct if recent_volatility_pct > 1e-9 else 0.0
    )

    regime_probs: Dict[str, float] = {}
    for rid in rset.regime_ids:
        regime_probs[f"regime_{rid}"] = regime_state.get_prob(rid, 0.0) if regime_state is not None else 0.0
    if regime_state is not None:
        for rid, val in regime_state.regime_probs.items():
            regime_probs.setdefault(f"regime_{rid}", float(val))

    return {
        "strength": zone.strength,
        "age": float(zone.age),
        "touch_count": float(zone.touch_count),
        "zone_width_pct": zone_width_pct,
        "width_to_volatility_ratio": width_to_volatility_ratio,
        "liquidity_score": zone.liquidity_score,
        "hour_utc": float(hour_utc),
        "day_of_week": float(day_of_week),
        **regime_probs,
    }


def features_to_row(
    features: Dict[str, float],
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    names = list(feature_names or ZONE_SURVIVAL_FEATURE_NAMES)
    return np.array([[float(features.get(name, 0.0)) for name in names]])
