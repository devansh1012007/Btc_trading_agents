"""
Router (§8): picks which expert (if any) should control this bar's trade.
Falls back to "highest-confidence expert above a floor" when no model is
trained; blends in a hindsight-trained multiclass model once available
(§8.2). Tracks switching context (bars_since_switch, current_expert) as a
soft feature for the model -- hard switching limits are explicitly
deferred to Phase 4's Risk Layer (§8.3, §8.4).
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..state.contracts import (
    FEATURE_SCHEMA_VERSION,
    ExpertAction,
    ExpertOutputs,
    OpportunityScore,
    RegimeState,
    RouterDecision,
    ZoneMap,
)

logger = logging.getLogger(__name__)

EXPERT_NAMES = ["trend_expert", "mean_reversion_expert", "breakout_expert"]
ROUTER_CLASSES = EXPERT_NAMES + ["no_trade"]

FEATURE_NAMES = (
    ["prob_trending_up", "prob_trending_down", "prob_chop_range", "prob_panic_spike"]
    + [f"{name}_confidence" for name in EXPERT_NAMES]
    + [f"{name}_expected_return" for name in EXPERT_NAMES]
    + [f"{name}_reward_to_risk" for name in EXPERT_NAMES]
    + ["opportunity_tradable_probability", "bars_since_switch"]
)


def build_features(
    expert_outputs: ExpertOutputs, regime: RegimeState, opportunity_score: OpportunityScore, bars_since_switch: int,
) -> Dict[str, float]:
    features = {
        "prob_trending_up": regime.prob_trending_up,
        "prob_trending_down": regime.prob_trending_down,
        "prob_chop_range": regime.prob_chop_range,
        "prob_panic_spike": regime.prob_panic_spike,
        "opportunity_tradable_probability": opportunity_score.tradable_probability,
        "bars_since_switch": float(bars_since_switch),
    }
    for name in EXPERT_NAMES:
        output = expert_outputs.get_by_name(name)
        features[f"{name}_confidence"] = output.confidence if output else 0.0
        features[f"{name}_expected_return"] = output.expected_return if output else 0.0
        features[f"{name}_reward_to_risk"] = output.reward_to_risk if output else 0.0
    return features


class Router:
    def __init__(self, config: dict, model_path: Optional[str] = None, data_granularity: str = "1m"):
        self.config = config
        self.data_granularity = data_granularity
        self.fallback_cfg = config.get("fallback_behavior", {})
        ml_cfg = config.get("model", {})
        self.blend_weight = ml_cfg.get("blend_weight", 0.0)

        self._model = None
        self._model_feature_names = None
        self._current_expert: Optional[str] = None
        self._bars_since_switch = 0

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info(f"No router model at {p}; using confidence-based fallback")
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load router model at {p}: {e}")
            return

        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"Router model trained on granularity={trained_granularity}, "
                f"running on {self.data_granularity}. Falling back to confidence-based mode."
            )
            return

        trained_feature_version = payload.get("feature_version")
        if trained_feature_version is not None and trained_feature_version != FEATURE_SCHEMA_VERSION:
            logger.warning(
                f"Router model trained on feature_schema={trained_feature_version}, "
                f"current schema={FEATURE_SCHEMA_VERSION}. Falling back to confidence-based mode."
            )
            return

        self._model = payload["model"]
        self._model_feature_names = payload["feature_names"]
        logger.info(f"Loaded router model from {p}")

    def decide(
        self, expert_outputs: ExpertOutputs, regime: RegimeState, zone_map: Optional[ZoneMap],
        opportunity_score: OpportunityScore,
    ) -> RouterDecision:
        if not opportunity_score.tradable:
            decision = RouterDecision(
                timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
                selected_expert="no_trade", selection_confidence=1.0 - opportunity_score.tradable_probability,
                no_trade_probability=1.0, expert_ranking={}, reasoning="opportunity_filter_rejected",
            )
        elif self._model is not None:
            decision = self._decide_with_model(expert_outputs, regime, opportunity_score)
        else:
            decision = self._decide_with_fallback(expert_outputs, opportunity_score)

        self._update_switch_state(decision.selected_expert)
        return decision

    def _decide_with_fallback(self, expert_outputs: ExpertOutputs, opportunity_score: OpportunityScore) -> RouterDecision:
        best = expert_outputs.get_best_by_confidence()
        floor = self.fallback_cfg.get("confidence_floor_to_trade", 0.0)
        ranking = {o.expert_name: o.confidence for o in expert_outputs.all()}

        if best is None or best.action == ExpertAction.NO_TRADE or best.confidence < floor:
            return RouterDecision(
                timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
                selected_expert="no_trade", selection_confidence=1.0 - (best.confidence if best else 0.0),
                no_trade_probability=1.0, expert_ranking=ranking, reasoning="no_expert_cleared_confidence_floor",
            )

        return RouterDecision(
            timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
            selected_expert=best.expert_name, selection_confidence=best.confidence,
            no_trade_probability=1.0 - best.confidence, expert_ranking=ranking, reasoning="fallback_highest_confidence",
        )

    def _decide_with_model(self, expert_outputs: ExpertOutputs, regime: RegimeState, opportunity_score: OpportunityScore) -> RouterDecision:
        features = build_features(expert_outputs, regime, opportunity_score, self._bars_since_switch)
        try:
            row = np.array([[features[name] for name in self._model_feature_names]])
            proba = self._model.predict_proba(row)[0]
            classes = list(self._model.classes_)
        except Exception as e:
            logger.warning(f"Router model inference failed, using fallback: {e}")
            return self._decide_with_fallback(expert_outputs, opportunity_score)

        model_probs = {cls: 0.0 for cls in ROUTER_CLASSES}
        model_probs.update({cls: float(p) for cls, p in zip(classes, proba, strict=False) if cls in model_probs})

        confidence_probs = {
            name: (expert_outputs.get_by_name(name).confidence if expert_outputs.get_by_name(name) else 0.0)
            for name in EXPERT_NAMES
        }
        total_conf = sum(confidence_probs.values())
        if total_conf > 0:
            confidence_probs = {k: v / total_conf for k, v in confidence_probs.items()}
        confidence_probs["no_trade"] = max(0.0, 1.0 - sum(confidence_probs[n] for n in EXPERT_NAMES))

        w = self.blend_weight
        blended = {cls: (1 - w) * confidence_probs.get(cls, 0.0) + w * model_probs.get(cls, 0.0) for cls in ROUTER_CLASSES}
        selected = max(blended, key=blended.get)
        floor = self.fallback_cfg.get("confidence_floor_to_trade", 0.0)

        if selected == "no_trade" or blended[selected] < floor:
            return RouterDecision(
                timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
                selected_expert="no_trade", selection_confidence=blended.get("no_trade", 0.0),
                no_trade_probability=blended.get("no_trade", 0.0), expert_ranking=blended, reasoning="router_model_no_trade",
            )

        return RouterDecision(
            timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
            selected_expert=selected, selection_confidence=blended[selected],
            no_trade_probability=blended.get("no_trade", 0.0), expert_ranking=blended, reasoning="router_model",
        )

    def _update_switch_state(self, selected_expert: str) -> None:
        effective = None if selected_expert == "no_trade" else selected_expert
        if effective != self._current_expert:
            self._current_expert = effective
            self._bars_since_switch = 0
        else:
            self._bars_since_switch += 1
