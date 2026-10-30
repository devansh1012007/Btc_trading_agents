"""
Regime label alignment across retrains (`align_regime_labels`, Phase 5 T5.3).

Solves maximum-overlap / minimum-centroid-distance bipartite matching
(Hungarian assignment) so state `r_i` after a retrain maps to the same
physical market regime as `r_i` before the retrain, and enforces
`lock_between_retrains` when `K` changes or minimum Jaccard overlap falls
below `min_match_jaccard`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


class RegimeRelabelBlockedError(RuntimeError):
    """Raised when `lock_between_retrains` blocks a regime relabeling with
    changed K or insufficient state overlap."""


@dataclass(frozen=True)
class AlignmentResult:
    new_to_old: Dict[int, int]
    per_state_jaccard: Dict[int, float]
    min_jaccard: float
    accepted: bool
    reason: str = ""


def _solve_assignment_min_cost(cost: np.ndarray) -> Tuple[int, ...]:
    """Return `perm` where `perm[new_idx] = old_idx` minimizing `sum(cost[new_idx, perm[new_idx]])`."""
    k = cost.shape[0]
    if k <= 7:
        best_perm = tuple(range(k))
        best_val = float("inf")
        for perm in itertools.permutations(range(k)):
            val = sum(float(cost[i, perm[i]]) for i in range(k))
            if val < best_val:
                best_val = val
                best_perm = perm
        return best_perm
    # Greedy fallback for K > 7
    used = set()
    out = []
    for i in range(k):
        choices = [j for j in range(k) if j not in used]
        best_j = min(choices, key=lambda j: float(cost[i, j]))
        used.add(best_j)
        out.append(best_j)
    return tuple(out)


def align_regime_labels(
    old_centroids: np.ndarray,
    new_centroids: np.ndarray,
    *,
    old_probs: Optional[np.ndarray] = None,
    new_probs: Optional[np.ndarray] = None,
    min_match_jaccard: float = 0.5,
    lock_between_retrains: bool = False,
) -> AlignmentResult:
    """Align `new_centroids` / `new_probs` to `old_centroids` / `old_probs`."""
    old_c = np.atleast_2d(np.asarray(old_centroids, dtype=float))
    new_c = np.atleast_2d(np.asarray(new_centroids, dtype=float))
    if old_c.shape[0] != new_c.shape[0]:
        msg = f"Regime count changed from K={old_c.shape[0]} to K={new_c.shape[0]}"
        if lock_between_retrains:
            raise RegimeRelabelBlockedError(msg)
        return AlignmentResult(
            new_to_old={},
            per_state_jaccard={},
            min_jaccard=0.0,
            accepted=False,
            reason=msg,
        )

    k = old_c.shape[0]
    if old_probs is not None and new_probs is not None:
        old_lbl = np.argmax(old_probs, axis=1)
        new_lbl = np.argmax(new_probs, axis=1)
        jacc_mat = np.zeros((k, k), dtype=float)
        for i in range(k):
            m_new = new_lbl == i
            for j in range(k):
                m_old = old_lbl == j
                inter = float(np.sum(m_new & m_old))
                union = float(np.sum(m_new | m_old))
                jacc_mat[i, j] = inter / union if union > 0 else 1.0
        cost = 1.0 - jacc_mat
    else:
        jacc_mat = None
        cost = np.sum((new_c[:, None, :] - old_c[None, :, :]) ** 2, axis=2)

    perm = _solve_assignment_min_cost(cost)
    new_to_old = {int(new_idx): int(perm[new_idx]) for new_idx in range(k)}

    per_jacc: Dict[int, float] = {}
    if jacc_mat is not None:
        for new_idx, old_idx in new_to_old.items():
            per_jacc[old_idx] = float(jacc_mat[new_idx, old_idx])
    else:
        max_dist = max(float(np.max(cost)), 1e-6)
        for new_idx, old_idx in new_to_old.items():
            per_jacc[old_idx] = float(max(0.0, 1.0 - cost[new_idx, old_idx] / max_dist))

    min_jacc = float(min(per_jacc.values())) if per_jacc else 0.0
    if min_jacc < min_match_jaccard:
        msg = f"min_jaccard={min_jacc:.3f} < min_match_jaccard={min_match_jaccard:.3f}"
        if lock_between_retrains:
            raise RegimeRelabelBlockedError(msg)
        return AlignmentResult(
            new_to_old=new_to_old,
            per_state_jaccard=per_jacc,
            min_jaccard=min_jacc,
            accepted=False,
            reason=msg,
        )

    return AlignmentResult(
        new_to_old=new_to_old,
        per_state_jaccard=per_jacc,
        min_jaccard=min_jacc,
        accepted=True,
    )
