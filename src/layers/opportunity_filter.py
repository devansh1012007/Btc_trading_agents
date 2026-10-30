"""
Opportunity Filter (§7): decides whether the environment is tradable at
all, before the Router picks an expert. Hard rules always apply; an
optional trained model blends in on top once available (§7.2). Regime-
specific overrides (§9's regime_overrides block) layer on top of the base
hard_rules for the bar's current regime.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..execution.cost_model import CostModel
from ..labeling.hold_policy import HoldPolicy, model_alignment_problem
from ..state.contracts import (
    FEATURE_SCHEMA_VERSION,
    ExpertOutputs,
    FeatureVector,
    MarketSnapshot,
    OpportunityScore,
    RegimeState,
    ZoneMap,
)
from ..utils.config_helpers import parse_regime_list

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "best_confidence", "mean_no_trade_probability", "edge_vs_fees",
    "spread_pct", "volatility_medium", "regime_confidence", "regime_persistence_score",
    "zone_num_active",
]


def build_features(
    expert_outputs: ExpertOutputs, regime: RegimeState, zone_map: Optional[ZoneMap],
    market: MarketSnapshot, fv: FeatureVector, edge_vs_fees: float,
) -> Dict[str, float]:
    outputs = expert_outputs.all()
    best = expert_outputs.get_best_by_confidence()
    mean_no_trade = float(np.mean([o.no_trade_probability for o in outputs])) if outputs else 1.0
    return {
        "best_confidence": best.confidence if best else 0.0,
        "mean_no_trade_probability": mean_no_trade,
        "edge_vs_fees": edge_vs_fees,
        "spread_pct": market.spread_pct,
        "volatility_medium": fv.volatility_medium,
        "regime_confidence": regime.regime_confidence,
        "regime_persistence_score": regime.regime_persistence_score,
        "zone_num_active": float(len(zone_map.zones)) if zone_map else 0.0,
    }


class OpportunityFilter:
    def __init__(
        self, config: dict, labeling_config: dict, model_path: Optional[str] = None,
        data_granularity: str = "1m", hold_policy: Optional[HoldPolicy] = None,
        require_aligned_models: bool = False, cost_model: Optional[CostModel] = None,
    ):
        self.hold_policy = hold_policy
        self.require_aligned_models = require_aligned_models
        self.config = config
        self.labeling_config = labeling_config
        self.cost_model = cost_model or CostModel.from_config({"labeling": dict(labeling_config or {})})
        self.data_granularity = data_granularity
        self.hard_rules = config.get("hard_rules", {})
        self.regime_overrides = config.get("regime_overrides", {})
        ml_cfg = config.get("ml_model", {})
        self.blend_weight = ml_cfg.get("blend_weight", 0.0)

        self._model = None
        self._model_feature_names = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info(f"No opportunity filter model at {p}; using pure hard-rules")
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load opportunity filter model at {p}: {e}")
            return

        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"Opportunity filter model trained on granularity={trained_granularity}, "
                f"running on {self.data_granularity}. Falling back to hard-rules-only mode."
            )
            return

        trained_feature_version = payload.get("feature_version")
        if trained_feature_version is not None and trained_feature_version != FEATURE_SCHEMA_VERSION:
            logger.warning(
                f"Opportunity filter model trained on feature_schema={trained_feature_version}, "
                f"current schema={FEATURE_SCHEMA_VERSION}. Falling back to hard-rules-only mode."
            )
            return


        problem = model_alignment_problem(
            payload, self.hold_policy, require_aligned=self.require_aligned_models,
        )
        if problem is not None:
            logger.warning(
                f"Opportunity filter model at {p} is not aligned with the running harness ({problem}). "
                f"Falling back to hard-rules-only mode. Retrain with scripts/train_opportunity_filter.py."
            )
            return

        self._model = payload["model"]
        self._model_feature_names = payload["feature_names"]
        logger.info(f"Loaded opportunity filter model from {p}")

    def _effective_hard_rules(self, regime: RegimeState) -> dict:
        merged = dict(self.hard_rules)
        merged.update(self.regime_overrides.get(regime.primary_regime.name, {}))
        return merged

    def evaluate(
        self, expert_outputs: ExpertOutputs, regime: RegimeState, zone_map: Optional[ZoneMap],
        market: MarketSnapshot, fv: FeatureVector,
    ) -> OpportunityScore:
        cfg = self._effective_hard_rules(regime)
        failed_rules: List[str] = []

        best = expert_outputs.get_best_by_confidence()
        round_trip_cost = self.cost_model.opportunity_filter_cost_pct(market=market, fv=fv) if self.labeling_config or self.cost_model.explicit_costs_block else 0.0
        edge_vs_fees = (best.expected_return - round_trip_cost) if best else -round_trip_cost

        if edge_vs_fees < cfg.get("edge_vs_fees_threshold", 0.0):
            failed_rules.append(f"edge_below_threshold({edge_vs_fees:.4f})")

        if market.spread_pct > cfg.get("max_spread_pct", float("inf")):
            failed_rules.append(f"spread_too_wide({market.spread_pct:.4f})")

        # bid_volume/ask_volume are base-currency units (BTC), min_depth_usdt
        # is a quote-currency (USDT) threshold -- must convert before
        # comparing. Pre-existing Phase 3 bug, masked until now: depth was
        # always 0 before Finding #2's synthetic liquidity model, so this
        # check failed for a different reason (zero depth) regardless of
        # units; fixing Finding #2 exposed the unit mismatch for the first
        # time (see PHASE4_COMPLETE.md).
        depth_usdt = (market.bid_volume + market.ask_volume) * market.close
        if depth_usdt < cfg.get("min_depth_usdt", 0.0):
            failed_rules.append(f"insufficient_depth({depth_usdt:.2f})")

        recent_vol = fv.volatility_medium
        if recent_vol < cfg.get("volatility_floor_pct", 0.0):
            failed_rules.append(f"market_too_dead({recent_vol:.4f})")
        if recent_vol > cfg.get("volatility_ceiling_pct", float("inf")):
            failed_rules.append(f"volatility_too_high({recent_vol:.4f})")

        reject_regimes = parse_regime_list(cfg.get("reject_on_regime", []))
        if regime.primary_regime in reject_regimes:
            failed_rules.append(f"regime_blocked({regime.primary_regime.value})")

        outputs = expert_outputs.all()
        mean_no_trade = float(np.mean([o.no_trade_probability for o in outputs])) if outputs else 1.0
        if mean_no_trade > cfg.get("no_consensus_ceiling", 1.0):
            failed_rules.append(f"no_consensus({mean_no_trade:.2f})")

        best_confidence = best.confidence if best else 0.0
        if best_confidence < cfg.get("reject_if_best_confidence_below", 0.0):
            failed_rules.append(f"confidence_too_low({best_confidence:.2f})")

        hard_rules_tradable = len(failed_rules) == 0

        if self._model is not None:
            features = build_features(expert_outputs, regime, zone_map, market, fv, edge_vs_fees)
            try:
                row = np.array([[features[name] for name in self._model_feature_names]])
                ml_probability = float(self._model.predict_proba(row)[0][1])
            except Exception as e:
                logger.warning(f"Opportunity filter model inference failed, using hard rules only: {e}")
                ml_probability = 1.0 if hard_rules_tradable else 0.0

            w = self.blend_weight
            hard_rules_score = 1.0 if hard_rules_tradable else 0.0
            tradable_probability = (1 - w) * hard_rules_score + w * ml_probability
            tradable = hard_rules_tradable and tradable_probability > 0.5
        else:
            tradable_probability = 1.0 if hard_rules_tradable else 0.0
            tradable = hard_rules_tradable

        liquidity_score = self._liquidity_score(market, cfg)
        volatility_score = self._volatility_score(recent_vol, cfg)
        expert_consensus_score = 1.0 - mean_no_trade
        regime_acceptability = 0.0 if regime.primary_regime in reject_regimes else regime.regime_confidence
        environment_quality = float(np.mean([liquidity_score, volatility_score, regime_acceptability]))
        opportunity_strength = best.reward_to_risk if best else 0.0

        return OpportunityScore(
            timestamp=fv.timestamp, symbol=fv.symbol, tradable=tradable,
            tradable_probability=tradable_probability, environment_quality=environment_quality,
            opportunity_strength=opportunity_strength, failed_rules=failed_rules,
            edge_vs_fees=edge_vs_fees, liquidity_score=liquidity_score,
            volatility_score=volatility_score, expert_consensus_score=expert_consensus_score,
            regime_acceptability=regime_acceptability,
        )

    @staticmethod
    def _liquidity_score(market: MarketSnapshot, cfg: dict) -> float:
        max_spread = cfg.get("max_spread_pct", 1.0) or 1.0
        spread_component = max(0.0, 1.0 - market.spread_pct / max_spread)
        min_depth = cfg.get("min_depth_usdt", 1.0) or 1.0
        depth_usdt = (market.bid_volume + market.ask_volume) * market.close
        depth_component = min(1.0, depth_usdt / min_depth) if min_depth > 0 else 1.0
        return float(np.clip((spread_component + depth_component) / 2, 0.0, 1.0))

    @staticmethod
    def _volatility_score(recent_vol: float, cfg: dict) -> float:
        floor = cfg.get("volatility_floor_pct", 0.0)
        ceiling = cfg.get("volatility_ceiling_pct", 1.0) or 1.0
        if ceiling <= floor:
            return 0.5
        midpoint = (floor + ceiling) / 2
        span = (ceiling - floor) / 2
        return float(np.clip(1.0 - abs(recent_vol - midpoint) / span, 0.0, 1.0))


    def evaluate_pre_filter(
        self,
        regime: RegimeState,
        zone_map: Optional[ZoneMap],
        market: MarketSnapshot,
        fv: FeatureVector,
    ) -> OpportunityScore:
        """Stage 1 of `staged` order (C1): environment gates (spread, depth, volatility, regime)
        evaluated BEFORE experts/router run."""
        cfg = self._effective_hard_rules(regime)
        failed_rules: List[str] = []

        if market.spread_pct > cfg.get("max_spread_pct", float("inf")):
            failed_rules.append(f"spread_too_wide({market.spread_pct:.4f})")

        depth_usdt = (market.bid_volume + market.ask_volume) * market.close
        if depth_usdt < cfg.get("min_depth_usdt", 0.0):
            failed_rules.append(f"insufficient_depth({depth_usdt:.2f})")

        recent_vol = fv.volatility_medium
        if recent_vol < cfg.get("volatility_floor_pct", 0.0):
            failed_rules.append(f"market_too_dead({recent_vol:.4f})")
        if recent_vol > cfg.get("volatility_ceiling_pct", float("inf")):
            failed_rules.append(f"volatility_too_high({recent_vol:.4f})")

        reject_regimes = parse_regime_list(cfg.get("reject_on_regime", []))
        if regime.primary_regime in reject_regimes:
            failed_rules.append(f"regime_blocked({regime.primary_regime.value})")

        tradable = len(failed_rules) == 0
        liquidity_score = self._liquidity_score(market, cfg)
        volatility_score = self._volatility_score(recent_vol, cfg)
        regime_acceptability = 0.0 if regime.primary_regime in reject_regimes else regime.regime_confidence
        environment_quality = float(np.mean([liquidity_score, volatility_score, regime_acceptability]))

        return OpportunityScore(
            timestamp=fv.timestamp,
            symbol=fv.symbol,
            tradable=tradable,
            tradable_probability=1.0 if tradable else 0.0,
            environment_quality=environment_quality,
            opportunity_strength=0.0,
            failed_rules=failed_rules,
            edge_vs_fees=0.0,
            liquidity_score=liquidity_score,
            volatility_score=volatility_score,
            expert_consensus_score=0.0,
            regime_acceptability=regime_acceptability,
        )

    def evaluate_confirm(
        self,
        expert_outputs: ExpertOutputs,
        router_decision,
        regime: RegimeState,
        zone_map: Optional[ZoneMap],
        market: MarketSnapshot,
        fv: FeatureVector,
    ) -> OpportunityScore:
        """Stage 2 of `staged` order (C1): evaluate edge-vs-fees, confidence, and ML opportunity
        model on the ROUTER-SELECTED expert after `router.decide`."""
        score = self.evaluate(expert_outputs, regime, zone_map, market, fv)
        if router_decision is None or router_decision.selected_expert == "no_trade":
            return score
        selected = expert_outputs.get_by_name(router_decision.selected_expert)
        if selected is None:
            return score
        cfg = self._effective_hard_rules(regime)
        round_trip_cost = (
            self.cost_model.opportunity_filter_cost_pct(market=market, fv=fv)
            if self.labeling_config or self.cost_model.explicit_costs_block
            else 0.0
        )
        sel_edge = selected.expected_return - round_trip_cost
        failed = list(score.failed_rules)
        if sel_edge < cfg.get("edge_vs_fees_threshold", 0.0):
            rule_str = f"selected_edge_below_threshold({sel_edge:.4f})"
            if rule_str not in failed:
                failed.append(rule_str)
        tradable = score.tradable and (len(failed) == 0)
        from dataclasses import replace as dc_replace
        return dc_replace(
            score,
            tradable=tradable,
            tradable_probability=score.tradable_probability if tradable else 0.0,
            failed_rules=failed,
            edge_vs_fees=sel_edge,
            opportunity_strength=selected.reward_to_risk,
        )
