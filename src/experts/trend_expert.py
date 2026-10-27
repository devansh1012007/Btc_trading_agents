"""
Trend Expert (§6.2): trades in the direction of momentum when the regime
agrees, is confident, and is persistent.
"""
import logging
from typing import Dict, Optional

from ..labeling.hold_policy import HoldPolicy
from ..state.contracts import ExpertAction, ExpertOutput, FeatureVector, RegimeState, ZoneMap
from ..utils.config_helpers import parse_regime_list
from .base_expert import BaseExpert, compute_invalidation_distance

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "momentum_medium", "momentum_long", "sma_slope", "invalidation_distance",
    "regime_confidence", "regime_persistence_score", "volatility_medium",
]


def build_features(fv: FeatureVector, zones: ZoneMap, regime: RegimeState, direction: int) -> Dict[str, float]:
    invalidation_distance, _ = compute_invalidation_distance(fv, direction)
    return {
        "momentum_medium": fv.momentum_medium,
        "momentum_long": fv.momentum_long,
        "sma_slope": fv.sma_9 - fv.sma_21,
        "invalidation_distance": invalidation_distance,
        "regime_confidence": regime.regime_confidence,
        "regime_persistence_score": regime.regime_persistence_score,
        "volatility_medium": fv.volatility_medium,
    }


class TrendExpert(BaseExpert):
    expert_name = "trend_expert"

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
        if regime.regime_persistence_score < cfg["persistence_requirement"]:
            return None

        momentum = fv.momentum_medium
        if abs(momentum) < cfg["momentum_threshold"]:
            return None

        return 1 if momentum > 0 else -1

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
