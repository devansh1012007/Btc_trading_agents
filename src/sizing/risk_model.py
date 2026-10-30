"""
Zone-survival-informed ML risk sizing model (`RiskSizingModel`, Phase 7 T7.3).

Consumes:
* Zone-survival probabilities (`nearest_support_survival_prob`,
  `nearest_resistance_survival_prob`)
* `invalidation_distance`, `effective_stop_loss_pct`, `router_selection_confidence`,
  `regime_confidence`, `volatility_medium`, `spread_pct`, `liquidity_depth_usdt`,
  `expected_return`, `expected_risk`
* Dynamic K-regime probabilities (`regime_prob_features`)

Predicts `p_stop_out = P(stop-out)` and maps it to a deterministic,
tighten-only size multiplier `mult in [min_multiplier, 1.0]`.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..state.contracts import (
    LEGACY_REGIME_SET,
    ExitPlan,
    ExpertOutput,
    FeatureVector,
    MarketSnapshot,
    RegimeSet,
    RegimeState,
    RouterDecision,
    ZoneMap,
    regime_prob_features,
)

BASE_RISK_SIZING_FEATURES = [
    "invalidation_distance",
    "effective_stop_loss_pct",
    "router_selection_confidence",
    "regime_confidence",
    "volatility_medium",
    "spread_pct",
    "liquidity_depth_usdt",
    "expected_return",
    "expected_risk",
    "nearest_support_survival_prob",
    "nearest_resistance_survival_prob",
]


def build_risk_sizing_features(
    expert_output: ExpertOutput,
    router_decision: RouterDecision,
    regime_state: RegimeState,
    fv: FeatureVector,
    market: MarketSnapshot,
    zone_map: Optional[ZoneMap] = None,
    exit_plan: Optional[ExitPlan] = None,
    regime_set: Optional[RegimeSet] = None,
) -> Dict[str, float]:
    depth_usdt = (market.bid_volume + market.ask_volume) * market.close
    plan = exit_plan or getattr(expert_output, "exit_plan", None)
    eff_sl = float(plan.effective_stop_loss_pct) if plan is not None else float(expert_output.invalidation_distance)
    sup = zone_map.nearest_support if zone_map is not None else None
    res = zone_map.nearest_resistance if zone_map is not None else None
    rset = regime_set or LEGACY_REGIME_SET
    feats: Dict[str, float] = {
        "invalidation_distance": float(expert_output.invalidation_distance),
        "effective_stop_loss_pct": eff_sl,
        "router_selection_confidence": float(router_decision.selection_confidence),
        "regime_confidence": float(regime_state.regime_confidence),
        "volatility_medium": float(fv.volatility_medium),
        "spread_pct": float(market.spread_pct),
        "liquidity_depth_usdt": float(depth_usdt),
        "expected_return": float(expert_output.expected_return),
        "expected_risk": float(expert_output.expected_risk),
        "nearest_support_survival_prob": float(sup.survival_probability) if sup is not None else 0.5,
        "nearest_resistance_survival_prob": float(res.survival_probability) if res is not None else 0.5,
    }
    for rid in rset.regime_ids:
        feats[f'prob_{rid}'] = regime_state.get_prob(rid, 0.0)
    return feats


def risk_score_to_size_multiplier(
    p_stop_out: float,
    *,
    min_multiplier: float = 0.20,
    risk_sensitivity: float = 1.0,
) -> float:
    """Map `p_stop_out in [0, 1]` monotonically to a tighten-only multiplier in `[min_multiplier, 1.0]`."""
    floor = float(np.clip(min_multiplier, 0.0, 1.0))
    raw = 1.0 - float(risk_sensitivity) * float(np.clip(p_stop_out, 0.0, 1.0))
    return float(np.clip(raw, floor, 1.0))


class RiskSizingModel:
    """Dedicated ML risk sizing model predicting `P(stop-out)` and tighten-only size multiplier."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        regime_set: Optional[RegimeSet] = None,
    ) -> None:
        self.config = dict(config or {})
        self.regime_set = regime_set or LEGACY_REGIME_SET
        self.feature_names: List[str] = list(BASE_RISK_SIZING_FEATURES) + self.regime_set.prob_feature_names()
        self.min_multiplier = float(self.config.get("min_multiplier", 0.20))
        self.risk_sensitivity = float(self.config.get("risk_sensitivity", 1.0))
        self._clf: Optional[Any] = None

    def fit(
        self,
        X_rows: Sequence[Mapping[str, float]],
        y_stop_out: Sequence[int],
        *,
        random_state: int = 42,
    ) -> "RiskSizingModel":
        X_mat = np.array([[float(r.get(name, 0.0)) for name in self.feature_names] for r in X_rows], dtype=float)
        y_arr = np.asarray(y_stop_out, dtype=int)
        self._clf = LGBMClassifier(
            n_estimators=25, max_depth=3, random_state=random_state, verbose=-1
        ).fit(X_mat, y_arr)
        return self

    def predict_stop_out_and_multiplier(
        self,
        features: Mapping[str, float],
    ) -> Tuple[float, float]:
        if self._clf is None:
            return 0.5, risk_score_to_size_multiplier(
                0.5, min_multiplier=self.min_multiplier, risk_sensitivity=self.risk_sensitivity
            )
        row = np.array([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=float)
        proba = self._clf.predict_proba(row)[0]
        p_stop = float(proba[1]) if len(proba) > 1 else float(proba[0])
        mult = risk_score_to_size_multiplier(
            p_stop, min_multiplier=self.min_multiplier, risk_sensitivity=self.risk_sensitivity
        )
        return p_stop, mult
