"""
Structural Feature Adapter (§6.4, Phase 2): bridges ZoneMap output back
into FeatureVector's structural fields via dataclasses.replace().
"""
from dataclasses import replace
from typing import Dict, Optional

from ..state.contracts import FeatureVector, ZoneMap


def compute_structural_features(zone_map: ZoneMap) -> Dict[str, Optional[float]]:
    support = zone_map.nearest_support
    resistance = zone_map.nearest_resistance
    price = zone_map.current_price

    return {
        "distance_to_nearest_support": (
            (price - support.mid) / price * 100 if support and price else None
        ),
        "distance_to_nearest_resistance": (
            (resistance.mid - price) / price * 100 if resistance and price else None
        ),
        "support_strength": support.strength if support else None,
        "resistance_strength": resistance.strength if resistance else None,
    }


def attach_structural_features(feature_vector: FeatureVector, zone_map: Optional[ZoneMap]) -> FeatureVector:
    if zone_map is None:
        return feature_vector
    # Explicit kwargs (not **dict) so mypy validates field names/types.
    # Structural fields are Optional[float] on FeatureVector, so an absent
    # zone simply records None -- matching the freshly-computed-per-bar
    # contract where these fields start as None anyway.
    structural = compute_structural_features(zone_map)
    return replace(
        feature_vector,
        distance_to_nearest_support=structural["distance_to_nearest_support"],
        distance_to_nearest_resistance=structural["distance_to_nearest_resistance"],
        support_strength=structural["support_strength"],
        resistance_strength=structural["resistance_strength"],
    )
