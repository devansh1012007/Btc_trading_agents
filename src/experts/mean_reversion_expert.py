"""
Mean-Reversion Expert (§6.3): fades price back toward the middle of a
range when it's pressed against a strong, nearby zone with RSI at an
extreme, and the regime agrees it's a chop market.

Training data (§6.3): Phase 2's zone-touch event stream (rejection
events) -- see scripts/train_mean_reversion_expert.py.
"""
import logging
from typing import Dict, Optional

from ..labeling.hold_policy import HoldPolicy
from ..state.contracts import ExpertAction, ExpertOutput, FeatureVector, RegimeState, ZoneMap
from ..utils.config_helpers import parse_regime_list
from .base_expert import BaseExpert, compute_invalidation_distance

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["rsi_14", "zone_strength", "zone_age", "zone_proximity_pct", "regime_confidence", "volatility_medium"]


def build_features(fv: FeatureVector, zones: ZoneMap, regime: RegimeState, direction: int) -> Dict[str, float]:
    zone = zones.nearest_support if direction == 1 else zones.nearest_resistance
    price = zones.current_price
    zone_strength = zone.strength if zone else 0.0
    zone_age = float(zone.age) if zone else 0.0
    zone_proximity_pct = abs(price - zone.mid) / zone.mid * 100 if zone and price else 999.0
    return {
        "rsi_14": fv.rsi_14, "zone_strength": zone_strength, "zone_age": zone_age,
        "zone_proximity_pct": zone_proximity_pct, "regime_confidence": regime.regime_confidence,
        "volatility_medium": fv.volatility_medium,
    }


class MeanReversionExpert(BaseExpert):
    expert_name = "mean_reversion_expert"

    def __init__(
        self, config: dict, model_path: Optional[str] = None, data_granularity: str = "1m",
        hold_policy: Optional[HoldPolicy] = None, require_aligned_models: bool = False,
    ):
        super().__init__(config, model_path, data_granularity, hold_policy, require_aligned_models)
        self._regime_requirement = parse_regime_list(config["regime_requirement"])

    def propose_candidate(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState) -> Optional[int]:
        cfg = self.config

        if regime.primary_regime not in self._regime_requirement:
            return None
        if regime.regime_confidence < cfg["regime_confidence_floor"]:
            return None

        price = zones.current_price
        rsi_low, rsi_high = cfg["rsi_extremity_threshold"]

        support = zones.nearest_support
        if support and price:
            proximity = abs(price - support.mid) / support.mid * 100
            if proximity < cfg["zone_proximity_pct"] and support.strength > cfg["zone_strength_floor"]:
                if fv.rsi_14 < rsi_low:
                    return 1

        resistance = zones.nearest_resistance
        if resistance and price:
            proximity = abs(price - resistance.mid) / resistance.mid * 100
            if proximity < cfg["zone_proximity_pct"] and resistance.strength > cfg["zone_strength_floor"]:
                if fv.rsi_14 > rsi_high:
                    return -1

        return None

    def score(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState, direction: int) -> ExpertOutput:
        invalidation_distance, zone_aware = compute_invalidation_distance(fv, direction)

        if self.use_fallback_mode or self._model is None:
            confidence = self.config["fallback_confidence"]
            model_output = 0.0
        else:
            try:
                features = build_features(fv, zones, regime, direction)
                confidence = self._predict_confidence(features)
                model_output = confidence
            except Exception as e:
                logger.warning(f"[{self.expert_name}] Model inference failed, using fallback: {e}")
                confidence = self.config["fallback_confidence"]
                model_output = 0.0

        expected_return, expected_risk = self._expected_return_risk(confidence)
        expected_risk = max(expected_risk, 1e-6)
        reward_to_risk = max(expected_return / expected_risk, 0.0)

        return ExpertOutput(
            expert_name=self.expert_name, timestamp=fv.timestamp, symbol=fv.symbol,
            action=ExpertAction.BUY if direction == 1 else ExpertAction.SELL,
            confidence=confidence, no_trade_probability=1.0 - confidence,
            expected_return=expected_return, expected_risk=expected_risk,
            reward_to_risk=reward_to_risk, invalidation_distance=invalidation_distance,
            zone_aware=zone_aware, model_output=model_output, regime_used=regime.primary_regime,
        )
