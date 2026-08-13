"""
Breakout Expert (§6.4): trades volatility expansion out of a compressed
range, confirmed by a volume surge and orderflow agreement.

Deviation from the plan's literal pseudocode: FeatureVector carries no raw
`volume` field and there's no pre-built rolling-average helper, so this
substitutes volume_zscore (already computed by FeatureEngine, §7.1) for
volume confirmation, and maintains its own rolling volatility_5m buffer
over compression_lookback_bars for the compression check, computed
statefully here.

Training data (§6.4): Phase 2's zone-touch event stream (breakout events).
"""
import logging
from typing import Dict, Optional

import numpy as np

from ..state.contracts import FeatureVector, ZoneMap, RegimeState, ExpertOutput, ExpertAction
from ..utils.config_helpers import parse_regime_list
from ..utils.rolling_buffer import RollingBarBuffer
from .base_expert import BaseExpert, compute_invalidation_distance

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["volatility_5m", "volatility_10m", "volume_zscore", "orderflow_momentum", "regime_confidence"]


def build_features(fv: FeatureVector, zones: ZoneMap, regime: RegimeState, direction: int) -> Dict[str, float]:
    return {
        "volatility_5m": fv.volatility_5m,
        "volatility_10m": fv.volatility_10m,
        "volume_zscore": (fv.raw_features or {}).get("volume_zscore", 0.0),
        "orderflow_momentum": fv.orderflow_momentum,
        "regime_confidence": regime.regime_confidence,
    }


class BreakoutExpert(BaseExpert):
    expert_name = "breakout_expert"

    def __init__(self, config: dict, model_path: Optional[str] = None, data_granularity: str = "1m"):
        super().__init__(config, model_path, data_granularity)
        self._regime_requirement = parse_regime_list(config["regime_requirement"])
        lookback = config.get("compression_lookback_bars", 20)
        self._volatility_history = RollingBarBuffer(maxlen=lookback)

    def propose_candidate(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState) -> Optional[int]:
        cfg = self.config
        self._volatility_history.append({"volatility_5m": fv.volatility_5m})

        if regime.primary_regime not in self._regime_requirement:
            return None
        if regime.regime_confidence < cfg["regime_confidence_floor"]:
            return None

        if len(self._volatility_history) < 2:
            return None

        avg_vol = float(np.mean(self._volatility_history.field_array("volatility_5m")))
        if avg_vol <= 0:
            return None
        if fv.volatility_5m > avg_vol * (1 - cfg["compression_pct_threshold"] / 100):
            return None

        volume_zscore = (fv.raw_features or {}).get("volume_zscore", 0.0)
        if volume_zscore < cfg["volume_zscore_surge_threshold"]:
            return None

        if abs(fv.orderflow_momentum) < cfg["orderflow_momentum_threshold"]:
            return None

        return 1 if fv.orderflow_momentum > 0 else -1

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
