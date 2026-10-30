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

from ..labeling.hold_policy import HoldPolicy, model_alignment_problem
from ..state.contracts import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_REGIME_SET,
    ExpertAction,
    ExpertOutputs,
    OpportunityScore,
    RegimeSet,
    RegimeState,
    RouterDecision,
    ZoneMap,
    regime_prob_features,
)

logger = logging.getLogger(__name__)

EXPERT_NAMES = ["trend_expert", "mean_reversion_expert", "breakout_expert"]
ROUTER_CLASSES = EXPERT_NAMES + ["no_trade"]

FEATURE_NAMES = (
    regime_prob_features(LEGACY_REGIME_SET, prefix="prob_")
    + [f"{name}_confidence" for name in EXPERT_NAMES]
    + [f"{name}_expected_return" for name in EXPERT_NAMES]
    + [f"{name}_reward_to_risk" for name in EXPERT_NAMES]
    + ["opportunity_tradable_probability", "bars_since_switch"]
)


def router_feature_names(
    regime_set: Optional[RegimeSet] = None,
    expert_names: Optional[list] = None,
) -> list:
    rset = regime_set or LEGACY_REGIME_SET
    enames = list(expert_names or EXPERT_NAMES)
    return (
        regime_prob_features(rset, prefix="prob_")
        + [f"{name}_confidence" for name in enames]
        + [f"{name}_expected_return" for name in enames]
        + [f"{name}_reward_to_risk" for name in enames]
        + ["opportunity_tradable_probability", "bars_since_switch"]
    )


def build_features(
    expert_outputs: ExpertOutputs,
    regime: RegimeState,
    opportunity_score: OpportunityScore,
    bars_since_switch: int,
    regime_set: Optional[RegimeSet] = None,
) -> Dict[str, float]:
    rset = regime_set or LEGACY_REGIME_SET
    features: Dict[str, float] = {}
    for rid in rset.regime_ids:
        features[f"prob_{rid}"] = regime.get_prob(rid, 0.0)
    for rid, pval in regime.regime_probs.items():
        features.setdefault(f"prob_{rid}", float(pval))
    features["opportunity_tradable_probability"] = opportunity_score.tradable_probability
    features["bars_since_switch"] = float(bars_since_switch)
    for name in EXPERT_NAMES:
        output = expert_outputs.get_by_name(name)
        features[f"{name}_confidence"] = output.confidence if output else 0.0
        features[f"{name}_expected_return"] = output.expected_return if output else 0.0
        features[f"{name}_reward_to_risk"] = output.reward_to_risk if output else 0.0
    return features


class Router:
    def __init__(
        self, config: dict, model_path: Optional[str] = None, data_granularity: str = "1m",
        hold_policy: Optional[HoldPolicy] = None, require_aligned_models: bool = False,
        expected_regime_set_hash: Optional[str] = None,
    ):
        self.hold_policy = hold_policy
        self.require_aligned_models = require_aligned_models
        self.expected_regime_set_hash = expected_regime_set_hash
        self.config = config
        self.data_granularity = data_granularity
        self.fallback_cfg = config.get("fallback_behavior", {})
        ml_cfg = config.get("model", {})
        self.blend_weight = ml_cfg.get("blend_weight", 0.0)

        self._model = None
        self._model_feature_names = None
        self._model_a = None
        self._model_b = None
        self.exit_mode = str(config.get("exit_mode", "hard_limits"))
        self.expert_exit_modes: Dict[str, str] = dict(config.get("expert_exit_modes") or {})
        self.success_tracker = RollingSuccessTracker()
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


        trained_rset = payload.get("regime_set_hash") or (payload.get("training_alignment") or {}).get("regime_set_hash")
        if self.expected_regime_set_hash is not None and trained_rset not in (None, self.expected_regime_set_hash):
            logger.warning(
                f"Router model trained on regime_set_hash={trained_rset}, "
                f"running on {self.expected_regime_set_hash}. Falling back to confidence-based mode."
            )
            return

        problem = model_alignment_problem(
            payload, self.hold_policy, require_aligned=self.require_aligned_models,
            expected_regime_set_hash=self.expected_regime_set_hash,
        )
        if problem is not None:
            logger.warning(
                f"Router model at {p} is not aligned with the running harness ({problem}). "
                f"Falling back to confidence-based mode. Retrain with scripts/train_router.py."
            )
            return

        self._model = payload["model"]
        self._model_feature_names = payload["feature_names"]
        logger.info(f"Loaded router model from {p}")

    def decide(
        self, expert_outputs: ExpertOutputs, regime: RegimeState, zone_map: Optional[ZoneMap],
        opportunity_score: OpportunityScore,
    ) -> RouterDecision:
        if opportunity_score is not None and not opportunity_score.tradable:
            decision = RouterDecision(
                timestamp=opportunity_score.timestamp, symbol=opportunity_score.symbol,
                selected_expert="no_trade", selection_confidence=1.0 - opportunity_score.tradable_probability,
                no_trade_probability=1.0, expert_ranking={}, reasoning="opportunity_filter_rejected",
            )
        elif self._has_mixed_exit_modes(expert_outputs):
            # Conflict C9 resolution: mixed exit modes across active experts fall back to rule/confidence router
            decision = self._decide_with_fallback(expert_outputs, opportunity_score, reasoning_override="mixed_exit_mode_fallback")
        elif self._active_router_model() is not None:
            decision = self._decide_with_model(expert_outputs, regime, opportunity_score)
        else:
            decision = self._decide_with_fallback(expert_outputs, opportunity_score)

        self._update_switch_state(decision.selected_expert)
        return decision

    def _has_mixed_exit_modes(self, expert_outputs: ExpertOutputs) -> bool:
        if not self.expert_exit_modes:
            return False
        active_modes = {
            self.expert_exit_modes.get(o.expert_name, self.exit_mode)
            for o in expert_outputs.all()
            if o.action != ExpertAction.NO_TRADE
        }
        return len(active_modes) > 1

    def _active_router_model(self):
        if self.exit_mode in ("ml_exit", "hybrid") and self._model_b is not None:
            return self._model_b
        if self.exit_mode == "hard_limits" and self._model_a is not None:
            return self._model_a
        return self._model

    def _decide_with_fallback(
        self,
        expert_outputs: ExpertOutputs,
        opportunity_score: Optional[OpportunityScore],
        reasoning_override: Optional[str] = None,
    ) -> RouterDecision:
        ts = opportunity_score.timestamp if opportunity_score is not None else expert_outputs.timestamp
        sym = opportunity_score.symbol if opportunity_score is not None else expert_outputs.symbol
        best = expert_outputs.get_best_by_confidence()
        floor = self.fallback_cfg.get("confidence_floor_to_trade", 0.0)
        ranking = {o.expert_name: o.confidence for o in expert_outputs.all()}

        if best is None or best.action == ExpertAction.NO_TRADE or best.confidence < floor:
            return RouterDecision(
                timestamp=ts, symbol=sym,
                selected_expert="no_trade", selection_confidence=1.0 - (best.confidence if best else 0.0),
                no_trade_probability=1.0, expert_ranking=ranking, reasoning="no_expert_cleared_confidence_floor",
            )

        return RouterDecision(
            timestamp=ts, symbol=sym,
            selected_expert=best.expert_name, selection_confidence=best.confidence,
            no_trade_probability=1.0 - best.confidence, expert_ranking=ranking,
            reasoning=reasoning_override or "fallback_highest_confidence",
        )

    def _decide_with_model(self, expert_outputs: ExpertOutputs, regime: RegimeState, opportunity_score: OpportunityScore) -> RouterDecision:
        features = build_features(expert_outputs, regime, opportunity_score, self._bars_since_switch)
        active_model = self._active_router_model()
        try:
            fnames = self._model_feature_names or list(features.keys())
            row = np.array([[features.get(name, 0.0) for name in fnames]])
            proba = active_model.predict_proba(row)[0]
            classes = list(active_model.classes_)
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



class RollingSuccessTracker:
    """Causal rolling expert win-rate tracker with Beta(2, 2) shrinkage prior (L3).

    Only trades whose `exit_timestamp <= current_timestamp` are included when
    computing `success_rate(expert_id, current_timestamp)`. Cold-start value is
    `(0 + 2) / (0 + 4) = 0.5`.
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0, window_trades: int = 50) -> None:
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.window_trades = max(1, int(window_trades))
        self._closed: Dict[str, list] = {}  # expert_id -> list of (exit_timestamp, is_win)

    def record_closed_trade(self, expert_id: str, exit_timestamp: float, pnl_pct: float) -> None:
        lst = self._closed.setdefault(str(expert_id), [])
        lst.append((float(exit_timestamp), 1.0 if float(pnl_pct) > 0 else 0.0))

    def success_rate(self, expert_id: str, current_timestamp: float) -> float:
        lst = self._closed.get(str(expert_id), [])
        eligible = [w for (exit_ts, w) in lst if exit_ts <= float(current_timestamp)]
        if len(eligible) > self.window_trades:
            eligible = eligible[-self.window_trades:]
        wins = float(sum(eligible))
        total = float(len(eligible))
        return (wins + self.prior_alpha) / (total + self.prior_alpha + self.prior_beta)
