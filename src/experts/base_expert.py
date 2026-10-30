"""
BaseExpert (§6.1): "rule proposes, model scores." propose_candidate() is a
pure, deterministic gate. score() runs in fallback mode (config defaults)
or model-enabled mode (a trained, granularity-matched model), where
expected_return/expected_risk come from the model card's empirical
avg_favorable/avg_unfavorable/avg_adverse_excursion stats (§4.7).
"""
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..labeling.hold_policy import POPULATION_EXPERT_GATE, HoldPolicy, model_alignment_problem
from ..state.contracts import (
    FEATURE_SCHEMA_VERSION,
    ExpertAction,
    ExpertOutput,
    FeatureVector,
    RegimeState,
    RegimeType,
    ZoneMap,
)

logger = logging.getLogger(__name__)


def compute_invalidation_distance(fv: FeatureVector, direction: int) -> Tuple[float, bool]:
    """Distance (%) to the nearest zone that would invalidate this trade
    direction if breached -- support below for longs, resistance above
    for shorts."""
    dist = fv.distance_to_nearest_support if direction == 1 else fv.distance_to_nearest_resistance
    if dist is None:
        return 999.0, False
    return abs(dist), True


class BaseExpert(ABC):
    expert_name: str = "base_expert"

    def __init__(
        self, config: dict, model_path: Optional[str] = None, data_granularity: str = "1m",
        hold_policy: Optional[HoldPolicy] = None, require_aligned_models: bool = False,
        *, instance_id: Optional[str] = None, enabled: bool = True, plan_head=None,
    ):
        self.config = config
        self.data_granularity = data_granularity
        self.hold_policy = hold_policy
        self.require_aligned_models = require_aligned_models
        self.instance_id = instance_id
        if instance_id:
            self.expert_name = instance_id
        self.enabled = bool(config.get("enabled", enabled))
        self.plan_head = plan_head

        self._model = None
        self._model_feature_names = None
        self._avg_favorable_return: Optional[float] = None
        self._avg_unfavorable_return: Optional[float] = None
        self._avg_adverse_excursion: Optional[float] = None
        self.use_fallback_mode = True

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        p = Path(model_path)
        if not p.exists():
            logger.info(f"[{self.expert_name}] No model found at {p}; using rule-based fallback")
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"[{self.expert_name}] Failed to load model at {p}: {e}")
            return

        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"[{self.expert_name}] Model trained on granularity={trained_granularity}, "
                f"running on {self.data_granularity}. Falling back to rule-based mode. "
                f"Retrain with scripts/train_{self.expert_name}.py."
            )
            return

        trained_feature_version = payload.get("feature_version")
        if trained_feature_version is not None and trained_feature_version != FEATURE_SCHEMA_VERSION:
            logger.warning(
                f"[{self.expert_name}] Model trained on feature_schema={trained_feature_version}, "
                f"current schema={FEATURE_SCHEMA_VERSION}. Falling back to rule-based mode."
            )
            return

        # A model is only served under the contract it was TRAINED under:
        # same minimum hold, same horizon, same population. Anything else
        # (including a legacy model with no contract) would be scoring a
        # trade the harness does not take -- fall back to the rule instead.
        problem = model_alignment_problem(
            payload, self.hold_policy, require_aligned=self.require_aligned_models,
            expected_population=POPULATION_EXPERT_GATE,
        )
        if problem is not None:
            logger.warning(
                f"[{self.expert_name}] Model at {p} is not aligned with the running harness "
                f"({problem}). Falling back to rule-based mode. "
                f"Retrain with scripts/train_{self.expert_name}.py."
            )
            return

        self._model = payload["model"]
        self._model_feature_names = payload["feature_names"]
        self._avg_favorable_return = payload.get("avg_favorable_return")
        self._avg_unfavorable_return = payload.get("avg_unfavorable_return")
        self._avg_adverse_excursion = payload.get("avg_adverse_excursion")
        self.use_fallback_mode = False
        logger.info(f"[{self.expert_name}] Loaded model from {p}")

    def _predict_confidence(self, features: Dict[str, float]) -> float:
        row = np.array([[features[name] for name in self._model_feature_names]])
        proba = self._model.predict_proba(row)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def _expected_return_risk(self, confidence: float) -> Tuple[float, float]:
        if (
            self.use_fallback_mode or self._model is None
            or self._avg_favorable_return is None or self._avg_unfavorable_return is None
        ):
            return self.config["fallback_expected_return"], self.config["fallback_expected_risk"]

        expected_return = confidence * self._avg_favorable_return + (1 - confidence) * self._avg_unfavorable_return
        expected_risk = self._avg_adverse_excursion if self._avg_adverse_excursion else self.config["fallback_expected_risk"]
        return expected_return, expected_risk

    @abstractmethod
    def propose_candidate(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def score(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState, direction: int) -> ExpertOutput:
        raise NotImplementedError

    @property
    def duration_blocked_reason(self) -> Optional[str]:
        """Why this expert must not propose ANY trade (None = it may).

        Fail closed: with an unsatisfiable hold policy (e.g. a timeout
        shorter than the minimum hold) every trade the expert proposed
        would be illegal, so it proposes none.
        """
        if self.hold_policy is None:
            return None
        problems = self.hold_policy.problems()
        return "; ".join(problems) if problems else None

    def evaluate(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState) -> ExpertOutput:
        if not self.enabled:
            return ExpertOutput(
                expert_name=self.expert_name, timestamp=fv.timestamp, symbol=fv.symbol,
                action=ExpertAction.NO_TRADE, confidence=0.0, no_trade_probability=1.0,
                expected_return=0.0, expected_risk=0.0, reward_to_risk=0.0,
                invalidation_distance=0.0, zone_aware=False, model_output=0.0,
                regime_used=regime.primary_regime if regime else RegimeType.UNKNOWN,
                instance_id=self.instance_id,
            )
        direction = None if self.duration_blocked_reason else self.propose_candidate(fv, zones, regime)
        if direction is None:
            return ExpertOutput(
                expert_name=self.expert_name, timestamp=fv.timestamp, symbol=fv.symbol,
                action=ExpertAction.NO_TRADE, confidence=0.0, no_trade_probability=1.0,
                expected_return=0.0, expected_risk=0.0, reward_to_risk=0.0,
                invalidation_distance=0.0, zone_aware=False, model_output=0.0,
                regime_used=regime.primary_regime if regime else RegimeType.UNKNOWN,
                instance_id=self.instance_id,
            )
        out = self.score(fv, zones, regime, direction)
        if self.plan_head is not None or self.instance_id is not None:
            from dataclasses import replace as dc_replace
            plan = out.exit_plan
            if plan is None and self.plan_head is not None:
                feats = {"volatility_medium": fv.volatility_medium, "regime_confidence": regime.regime_confidence}
                plan, _, _ = self.plan_head.select_best_plan(feats, max(fv.volatility_medium, 1e-4))
            out = dc_replace(out, exit_plan=plan, instance_id=self.instance_id)
        return out
