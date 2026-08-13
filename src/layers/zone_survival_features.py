"""
Shared feature extraction for the zone-survival model (§6.5, Phase 2).
Used by labeling, training, and serving-time blend so they can never drift
apart.
"""
from typing import Dict, Optional

import numpy as np

from ..state.contracts import Zone, RegimeState

ZONE_SURVIVAL_FEATURE_NAMES = [
    "strength", "age", "touch_count", "zone_width_pct",
    "width_to_volatility_ratio", "liquidity_score", "hour_utc", "day_of_week",
    "regime_trending_up", "regime_trending_down", "regime_chop_range", "regime_panic_spike",
]


def zone_survival_features(
    zone: Zone,
    recent_volatility_pct: float,
    hour_utc: int,
    day_of_week: int,
    regime_state: Optional[RegimeState],
) -> Dict[str, float]:
    zone_width_pct = (zone.upper_bound - zone.lower_bound) / zone.mid * 100 if zone.mid else 0.0
    width_to_volatility_ratio = (
        zone_width_pct / recent_volatility_pct if recent_volatility_pct > 1e-9 else 0.0
    )

    if regime_state is not None:
        regime_probs = {
            "regime_trending_up": regime_state.prob_trending_up,
            "regime_trending_down": regime_state.prob_trending_down,
            "regime_chop_range": regime_state.prob_chop_range,
            "regime_panic_spike": regime_state.prob_panic_spike,
        }
    else:
        regime_probs = {
            "regime_trending_up": 0.0, "regime_trending_down": 0.0,
            "regime_chop_range": 0.0, "regime_panic_spike": 0.0,
        }

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


def features_to_row(features: Dict[str, float]) -> np.ndarray:
    return np.array([[features[name] for name in ZONE_SURVIVAL_FEATURE_NAMES]])
