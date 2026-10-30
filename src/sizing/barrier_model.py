"""
Zone-survival-informed ML barrier sizing model (`BarrierSizingModel`,
Phase 7 T7.2).

Consumes:
* Zone-survival probabilities and strengths (`nearest_support_survival_prob`,
  `nearest_resistance_survival_prob`, `support_strength`, `resistance_strength`,
  `distance_to_nearest_support`, `distance_to_nearest_resistance`)
* `volatility_medium`, `spread_pct`, `direction`
* Dynamic K-regime probabilities (`regime_prob_features`)

Predicts volatility multiples `(k_sl, k_tp)` via quantile regression plus a
lockout adverse-excursion buffer so the stop is never tighter than the
expected lockout adverse excursion, and emits an `ExitPlan(source="barrier_model")`.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from lightgbm import LGBMRegressor

from ..labeling.hold_policy import HoldPolicy, model_alignment_problem
from ..state.contracts import (
    LEGACY_REGIME_SET,
    ExitPlan,
    FeatureVector,
    MarketSnapshot,
    RegimeSet,
    RegimeState,
    TrailSpec,
    ZoneMap,
    regime_prob_features,
)

logger = logging.getLogger(__name__)

BASE_BARRIER_FEATURES = [
    "volatility_medium",
    "spread_pct",
    "direction",
    "nearest_support_survival_prob",
    "nearest_resistance_survival_prob",
    "support_strength",
    "resistance_strength",
    "distance_to_nearest_support",
    "distance_to_nearest_resistance",
    "regime_confidence",
]


def build_barrier_sizing_features(
    fv: FeatureVector,
    zone_map: Optional[ZoneMap],
    regime_state: RegimeState,
    direction: int,
    market: Optional[MarketSnapshot] = None,
    regime_set: Optional[RegimeSet] = None,
) -> Dict[str, float]:
    sup = zone_map.nearest_support if zone_map is not None else None
    res = zone_map.nearest_resistance if zone_map is not None else None
    rset = regime_set or LEGACY_REGIME_SET
    feats: Dict[str, float] = {
        "volatility_medium": float(fv.volatility_medium),
        "spread_pct": float(market.spread_pct if market is not None else fv.spread_pct),
        "direction": float(direction),
        "nearest_support_survival_prob": float(sup.survival_probability) if sup is not None else 0.5,
        "nearest_resistance_survival_prob": float(res.survival_probability) if res is not None else 0.5,
        "support_strength": float(fv.support_strength or (sup.strength if sup is not None else 0.0)),
        "resistance_strength": float(fv.resistance_strength or (res.strength if res is not None else 0.0)),
        "distance_to_nearest_support": float(fv.distance_to_nearest_support or 1.0),
        "distance_to_nearest_resistance": float(fv.distance_to_nearest_resistance or 1.0),
        "regime_confidence": float(regime_state.regime_confidence),
    }
    for rid in rset.regime_ids:
        feats[f'prob_{rid}'] = regime_state.get_prob(rid, 0.0)
    return feats


class BarrierSizingModel:
    """Dedicated ML model mapping zone survival & market features to `(k_sl, k_tp)`."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        model_path: Optional[str] = None,
        data_granularity: str = "1m",
        hold_policy: Optional[HoldPolicy] = None,
        require_aligned_models: bool = False,
        regime_set: Optional[RegimeSet] = None,
    ) -> None:
        self.config = dict(config or {})
        self.data_granularity = data_granularity
        self.hold_policy = hold_policy
        self.require_aligned_models = require_aligned_models
        self.regime_set = regime_set or LEGACY_REGIME_SET
        self.feature_names: List[str] = list(BASE_BARRIER_FEATURES) + self.regime_set.prob_feature_names()

        self.q_sl = float(self.config.get("q_sl", 0.65))
        self.q_tp = float(self.config.get("q_tp", 0.60))
        self.min_sl_pct = float(self.config.get("min_sl_pct", 0.10))
        self.max_sl_pct = float(self.config.get("max_sl_pct", 5.0))
        self.min_tp_pct = float(self.config.get("min_tp_pct", 0.15))
        self.max_tp_pct = float(self.config.get("max_tp_pct", 10.0))
        self.min_reward_to_risk = float(self.config.get("min_reward_to_risk", 1.1))
        self.lockout_buffer_mult = float(self.config.get("lockout_buffer_mult", 1.05))

        self._sl_model: Optional[Any] = None
        self._tp_model: Optional[Any] = None
        self._lockout_model: Optional[Any] = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load barrier sizing model at {p}: {e}")
            return

        if payload.get("granularity") not in (None, self.data_granularity):
            logger.warning("Barrier sizing model granularity mismatch; ignoring.")
            return

        prob = model_alignment_problem(
            payload,
            self.hold_policy,
            require_aligned=self.require_aligned_models,
            expected_regime_set_hash=self.regime_set.version_hash if self.require_aligned_models else None,
        )
        if prob is not None:
            logger.warning(f"Barrier sizing model at {p} failed alignment ({prob}); ignoring.")
            return

        self._sl_model = payload.get("sl_model")
        self._tp_model = payload.get("tp_model")
        self._lockout_model = payload.get("lockout_model")
        if payload.get("feature_names"):
            self.feature_names = list(payload["feature_names"])

    def fit(
        self,
        X_rows: Sequence[Mapping[str, float]],
        k_mae: Sequence[float],
        k_mfe: Sequence[float],
        k_lockout_mae: Optional[Sequence[float]] = None,
        *,
        random_state: int = 42,
    ) -> "BarrierSizingModel":
        X_mat = np.array([[float(r.get(name, 0.0)) for name in self.feature_names] for r in X_rows], dtype=float)
        y_sl = np.asarray(k_mae, dtype=float)
        y_tp = np.asarray(k_mfe, dtype=float)

        self._sl_model = LGBMRegressor(
            objective="quantile", alpha=self.q_sl, n_estimators=25, max_depth=3, random_state=random_state, verbose=-1
        ).fit(X_mat, y_sl)
        self._tp_model = LGBMRegressor(
            objective="quantile", alpha=self.q_tp, n_estimators=25, max_depth=3, random_state=random_state, verbose=-1
        ).fit(X_mat, y_tp)

        if k_lockout_mae is not None:
            y_lock = np.asarray(k_lockout_mae, dtype=float)
            self._lockout_model = LGBMRegressor(
                objective="quantile", alpha=0.80, n_estimators=20, max_depth=3, random_state=random_state, verbose=-1
            ).fit(X_mat, y_lock)
        return self

    @property
    def is_ready(self) -> bool:
        return self._sl_model is not None and self._tp_model is not None

    def predict_exit_plan(
        self,
        fv: FeatureVector,
        zone_map: Optional[ZoneMap],
        regime_state: RegimeState,
        direction: int,
        *,
        market: Optional[MarketSnapshot] = None,
        max_holding_bars: int = 5,
        hard_max_loss_pct: Optional[float] = None,
        trailing: Optional[TrailSpec] = None,
    ) -> Optional[ExitPlan]:
        if not self.is_ready:
            return None
        feats = build_barrier_sizing_features(fv, zone_map, regime_state, direction, market, self.regime_set)
        row = np.array([[float(feats.get(name, 0.0)) for name in self.feature_names]], dtype=float)
        vol = max(float(fv.volatility_medium), 1e-4)

        k_sl = max(0.1, float(self._sl_model.predict(row)[0]))
        k_tp = max(0.1, float(self._tp_model.predict(row)[0]))
        if self._lockout_model is not None:
            k_lock = max(0.0, float(self._lockout_model.predict(row)[0]))
            k_sl = max(k_sl, k_lock * self.lockout_buffer_mult)

        sl_pct = float(np.clip(k_sl * vol, self.min_sl_pct, self.max_sl_pct))
        tp_raw = max(k_tp * vol, sl_pct * self.min_reward_to_risk)
        tp_pct = float(np.clip(tp_raw, self.min_tp_pct, self.max_tp_pct))

        return ExitPlan(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            max_holding_bars=int(max_holding_bars),
            trailing=trailing,
            hard_max_loss_pct=hard_max_loss_pct,
            source="barrier_model",
        )
