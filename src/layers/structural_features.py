"""
Structural Feature Adapter (§6.4): bridges DynamicStructureEngine's
ZoneMap output back into FeatureVector's existing-but-previously-unused
structural fields (distance_to_nearest_support/resistance,
support_strength, resistance_strength).

FeatureVector is a frozen dataclass, so this module never mutates it in
place -- callers get a new FeatureVector back via dataclasses.replace()
(the "two-pass" approach from §6.4: compute base features, then merge in
the structural values once ZoneMap is available).
"""
from dataclasses import replace
from typing import Dict, Optional

from ..state.contracts import FeatureVector, ZoneMap


def compute_structural_features(zone_map: ZoneMap) -> Dict[str, Optional[float]]:
    """
    Derive the four structural fields from a ZoneMap's nearest support/
    resistance zones. Returns a plain dict so this module doesn't need to
    know about FeatureVector's other fields.
    """
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


def attach_structural_features(
    feature_vector: FeatureVector, zone_map: Optional[ZoneMap]
) -> FeatureVector:
    """
    Return a new FeatureVector with structural fields populated from
    zone_map. If zone_map is None (e.g. the structure engine doesn't have
    enough history yet), the feature_vector is returned unchanged.
    """
    if zone_map is None:
        return feature_vector
    structural = compute_structural_features(zone_map)
    return replace(feature_vector, **structural)
