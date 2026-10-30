"""
Out-of-fold Top-N expert instance selection (`select_top_n_instances`,
Phase 9 T9.4 / Leakage Control L4).

Ranks expert instances strictly on out-of-fold (`is_oof=True`) validation
records so in-sample training optimism can never promote an overfit expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


class InSampleSelectionError(ValueError):
    """Raised when `select_top_n_instances` is given in-sample (`is_oof=False`) records (L4)."""


@dataclass(frozen=True)
class InstanceOOFScorecard:
    instance_id: str
    regime: str
    n_trades: int
    expectancy_pct: float
    profit_factor: float
    cvar_95_loss_pct: float
    composite_score: float


@dataclass(frozen=True)
class TopNSelectionResult:
    selected_instance_ids: List[str]
    selected_by_regime: Dict[str, List[str]]
    scorecards: List[InstanceOOFScorecard] = field(default_factory=list)


def _compute_scorecard(
    instance_id: str,
    regime: str,
    returns: Sequence[float],
    *,
    metric: str,
    w_profit: float,
    w_loss: float,
) -> InstanceOOFScorecard:
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n == 0:
        return InstanceOOFScorecard(instance_id, regime, 0, 0.0, 0.0, 0.0, float("-inf"))
    expectancy = float(np.mean(arr))
    gains = float(np.sum(arr[arr > 0]))
    losses = float(np.abs(np.sum(arr[arr < 0])))
    pf = gains / max(losses, 1e-6)
    neg = arr[arr < 0]
    if len(neg) > 0:
        cutoff = float(np.percentile(neg, 5))
        tail = neg[neg <= cutoff]
        cvar_95 = float(np.abs(np.mean(tail))) if len(tail) > 0 else float(np.abs(np.mean(neg)))
    else:
        cvar_95 = 0.0

    pos_mean = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0.0
    if metric == "expectancy":
        score = expectancy
    elif metric == "profit_factor":
        score = pf
    elif metric == "least_loss":
        score = -cvar_95
    else:  # "least_loss_most_profit"
        score = w_profit * pos_mean + (1.0 - w_loss) * expectancy - w_loss * cvar_95

    return InstanceOOFScorecard(
        instance_id=instance_id,
        regime=regime,
        n_trades=n,
        expectancy_pct=expectancy,
        profit_factor=pf,
        cvar_95_loss_pct=cvar_95,
        composite_score=float(score),
    )


def select_top_n_instances(
    oof_records: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 2,
    per_regime: bool = True,
    metric: str = "least_loss_most_profit",
    w_profit: float = 0.5,
    w_loss: float = 0.5,
    min_oof_trades: int = 3,
) -> TopNSelectionResult:
    """Rank and select Top-N expert instances using out-of-fold records ONLY (L4)."""
    grouped: Dict[tuple, List[float]] = {}
    for rec in oof_records:
        if rec.get("is_oof") is False or rec.get("split") == "train":
            raise InSampleSelectionError(
                "select_top_n_instances received in-sample training record (L4 violation); "
                "pass out-of-fold validation records only."
            )
        inst_id = str(rec["instance_id"])
        reg = str(rec.get("regime", "ALL")) if per_regime else "ALL"
        ret = float(rec["realized_return_pct"])
        grouped.setdefault((reg, inst_id), []).append(ret)

    scorecards: List[InstanceOOFScorecard] = []
    by_reg: Dict[str, List[InstanceOOFScorecard]] = {}
    for (reg, inst_id), rets in sorted(grouped.items()):
        sc = _compute_scorecard(inst_id, reg, rets, metric=metric, w_profit=w_profit, w_loss=w_loss)
        scorecards.append(sc)
        if sc.n_trades >= min_oof_trades:
            by_reg.setdefault(reg, []).append(sc)

    selected_by_regime: Dict[str, List[str]] = {}
    selected_set: List[str] = []
    for reg, cards in sorted(by_reg.items()):
        ranked = sorted(cards, key=lambda c: (c.composite_score, c.expectancy_pct, c.instance_id), reverse=True)
        chosen = [c.instance_id for c in ranked[: max(1, int(top_n))]]
        selected_by_regime[reg] = chosen
        for cid in chosen:
            if cid not in selected_set:
                selected_set.append(cid)

    return TopNSelectionResult(
        selected_instance_ids=selected_set,
        selected_by_regime=selected_by_regime,
        scorecards=scorecards,
    )
