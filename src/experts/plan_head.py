"""
Plan-conditioned utility head (`PlanUtilityHead`, Phase 9 T9.3): scores a
discrete grid of `(k_sl, k_tp, trail_activation_mult, trail_distance_mult, max_holding_bars)`
templates for an expert candidate and emits the utility-maximizing `ExitPlan`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..state.contracts import ExitPlan, TrailSpec


@dataclass(frozen=True)
class PlanGridTemplate:
    template_id: str
    k_sl: float
    k_tp: float
    max_holding_bars: int
    trail_enabled: bool = False
    trail_activation_mult: float = 1.0
    trail_distance_mult: float = 0.5
    trail_step_mult: float = 0.1

    def to_exit_plan(self, volatility_pct: float, hard_max_loss_pct: Optional[float] = None) -> ExitPlan:
        vol = max(float(volatility_pct), 1e-4)
        sl_pct = max(0.05, self.k_sl * vol)
        tp_pct = max(0.05, self.k_tp * vol)
        trailing = (
            TrailSpec(
                enabled=True,
                activation_pct=max(0.05, self.trail_activation_mult * vol),
                distance_pct=max(0.02, self.trail_distance_mult * vol),
                step_pct=max(0.01, self.trail_step_mult * vol),
            )
            if self.trail_enabled
            else None
        )
        return ExitPlan(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            max_holding_bars=self.max_holding_bars,
            trailing=trailing,
            hard_max_loss_pct=hard_max_loss_pct,
            source="expert",
        )


DEFAULT_PLAN_GRID: Tuple[PlanGridTemplate, ...] = (
    PlanGridTemplate("tight_fast", k_sl=0.8, k_tp=1.2, max_holding_bars=4, trail_enabled=False),
    PlanGridTemplate("balanced", k_sl=1.0, k_tp=1.5, max_holding_bars=5, trail_enabled=False),
    PlanGridTemplate("wide_runner", k_sl=1.2, k_tp=2.4, max_holding_bars=8, trail_enabled=True, trail_activation_mult=1.0, trail_distance_mult=0.5),
)


class PlanUtilityHead:
    """Linear / ridge utility scorer over `PlanGridTemplate` candidates."""

    def __init__(
        self,
        feature_names: Sequence[str],
        grid: Sequence[PlanGridTemplate] = DEFAULT_PLAN_GRID,
        risk_aversion: float = 0.35,
    ) -> None:
        self.feature_names = list(feature_names)
        self.grid = list(grid)
        self.risk_aversion = float(risk_aversion)
        # Per-template linear weights (D + 1 intercept) for expected return and expected MAE
        d = len(self.feature_names) + 1
        self._ret_weights: np.ndarray = np.zeros((len(self.grid), d), dtype=float)
        self._mae_weights: np.ndarray = np.zeros((len(self.grid), d), dtype=float)
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        returns_by_template: np.ndarray,
        mae_by_template: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        l2_reg: float = 1.0,
    ) -> "PlanUtilityHead":
        """Fit ridge regressors predicting `(realized_return_pct, max_adverse_excursion_pct)` per template."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X_arr.shape
        X_aug = np.hstack([X_arr, np.ones((n, 1), dtype=float)])
        w = np.ones(n, dtype=float) if sample_weights is None else np.asarray(sample_weights, dtype=float)
        W_sqrt = np.sqrt(np.maximum(w, 1e-6))[:, None]
        Xw = X_aug * W_sqrt
        reg = l2_reg * np.eye(d + 1, dtype=float)
        reg[-1, -1] = 1e-6
        gram_inv = np.linalg.pinv(Xw.T @ Xw + reg)

        m = len(self.grid)
        ret_w = np.zeros((m, d + 1), dtype=float)
        mae_w = np.zeros((m, d + 1), dtype=float)
        for j in range(m):
            yr = np.asarray(returns_by_template[:, j], dtype=float) * W_sqrt[:, 0]
            ym = np.asarray(mae_by_template[:, j], dtype=float) * W_sqrt[:, 0]
            ret_w[j] = gram_inv @ (Xw.T @ yr)
            mae_w[j] = gram_inv @ (Xw.T @ ym)

        self._ret_weights = ret_w
        self._mae_weights = mae_w
        self.is_fitted = True
        return self

    def score_templates(self, features: Dict[str, float]) -> List[Tuple[PlanGridTemplate, float, float, float]]:
        """Return `(template, pred_return, pred_mae, utility)` for each template in `self.grid`."""
        x = np.array([float(features.get(name, 0.0)) for name in self.feature_names] + [1.0], dtype=float)
        out: List[Tuple[PlanGridTemplate, float, float, float]] = []
        for j, tmpl in enumerate(self.grid):
            pred_ret = float(self._ret_weights[j] @ x) if self.is_fitted else (0.15 * tmpl.k_tp)
            pred_mae = max(0.01, float(self._mae_weights[j] @ x)) if self.is_fitted else (0.10 * tmpl.k_sl)
            utility = pred_ret - self.risk_aversion * pred_mae
            out.append((tmpl, pred_ret, pred_mae, utility))
        return out

    def select_best_plan(
        self,
        features: Dict[str, float],
        volatility_pct: float,
        hard_max_loss_pct: Optional[float] = None,
    ) -> Tuple[ExitPlan, float, float]:
        """Return `(best_exit_plan, expected_return, expected_risk)`."""
        scored = self.score_templates(features)
        best_tmpl, pred_ret, pred_mae, _ = max(scored, key=lambda item: item[3])
        return best_tmpl.to_exit_plan(volatility_pct, hard_max_loss_pct=hard_max_loss_pct), pred_ret, pred_mae
