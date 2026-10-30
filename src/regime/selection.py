"""
Data-driven selection of the number of regimes `K` (`select_k`, Phase 5 T5.2).

Evaluates candidate `K in [k_min, max_regimes]` on training data ONLY (L1),
enforcing:
* minimum state occupancy (`min_occupancy`)
* minimum mean dwell length in bars (`min_mean_dwell_bars`)
* penalized BIC improvement (`min_bic_delta_pct`)
with automatic fallback to `K=4` (`LEGACY_REGIME_SET`) when no candidate passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..state.contracts import LEGACY_REGIME_SET, DynamicRegime, RegimeSet
from .backends import RegimeBackend, build_regime_backend


@dataclass(frozen=True)
class CandidateKMetrics:
    k: int
    bic: float
    min_occupancy: float
    min_mean_dwell_bars: float
    passed_constraints: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class KSelectionResult:
    selected_k: int
    backend: RegimeBackend
    regime_set: RegimeSet
    candidates: List[CandidateKMetrics] = field(default_factory=list)
    used_fallback_k4: bool = False


def compute_occupancy_and_dwell(probs: np.ndarray) -> Tuple[float, float]:
    """Return `(min_state_occupancy, min_mean_dwell_bars)` for `probs` of shape `(T, K)`."""
    t_len, k = probs.shape
    if t_len == 0 or k == 0:
        return 0.0, 0.0
    labels = np.argmax(probs, axis=1)
    occupancies = np.array([np.mean(labels == j) for j in range(k)], dtype=float)
    min_occ = float(np.min(occupancies))

    dwells: Dict[int, List[int]] = {j: [] for j in range(k)}
    curr_lbl = int(labels[0])
    run_len = 1
    for idx in range(1, t_len):
        lbl = int(labels[idx])
        if lbl == curr_lbl:
            run_len += 1
        else:
            dwells[curr_lbl].append(run_len)
            curr_lbl = lbl
            run_len = 1
    dwells[curr_lbl].append(run_len)

    mean_dwells = [float(np.mean(runs)) if runs else 0.0 for runs in dwells.values()]
    return min_occ, float(np.min(mean_dwells))


def make_dynamic_regime_set(k: int, backend_name: str = "gaussian_hmm") -> RegimeSet:
    if k == 4 and backend_name == "legacy4":
        return LEGACY_REGIME_SET
    return RegimeSet(
        set_id=f"{backend_name}_k{k}",
        regime_ids=tuple(f"r_{i}" for i in range(k)),
    )


def select_k(
    X_train: np.ndarray,
    *,
    backend_name: str = "gaussian_hmm",
    k_min: int = 2,
    max_regimes: int = 6,
    min_occupancy: float = 0.03,
    min_mean_dwell_bars: float = 2.0,
    min_bic_delta_pct: float = 0.2,
    random_state: int = 42,
) -> KSelectionResult:
    """Select `K` on `X_train` without peeking at validation/test data (L1)."""
    arr = np.atleast_2d(np.asarray(X_train, dtype=float))
    if k_min < 2 or max_regimes < k_min:
        raise ValueError(f"Invalid K bounds: k_min={k_min}, max_regimes={max_regimes}")

    candidates: List[CandidateKMetrics] = []
    fitted_by_k: Dict[int, RegimeBackend] = {}

    for k in range(k_min, max_regimes + 1):
        backend = build_regime_backend(backend_name, n_components=k, random_state=random_state).fit(arr)
        probs = backend.filter_causal(arr)
        min_occ, min_dwell = compute_occupancy_and_dwell(probs)
        bic_val = backend.bic(arr)

        reasons = []
        if min_occ < min_occupancy:
            reasons.append(f"min_occupancy({min_occ:.3f}<{min_occupancy:.3f})")
        if min_dwell < min_mean_dwell_bars:
            reasons.append(f"min_dwell({min_dwell:.2f}<{min_mean_dwell_bars:.2f})")

        passed = len(reasons) == 0
        candidates.append(
            CandidateKMetrics(
                k=k,
                bic=bic_val,
                min_occupancy=min_occ,
                min_mean_dwell_bars=min_dwell,
                passed_constraints=passed,
                rejection_reason="; ".join(reasons),
            )
        )
        fitted_by_k[k] = backend

    valid = [c for c in candidates if c.passed_constraints]
    if not valid:
        fallback_k = min(max(4, k_min), max_regimes)
        fb_backend = fitted_by_k[fallback_k]
        return KSelectionResult(
            selected_k=fallback_k,
            backend=fb_backend,
            regime_set=make_dynamic_regime_set(fallback_k, backend_name),
            candidates=candidates,
            used_fallback_k4=True,
        )

    best = valid[0]
    for cand in valid[1:]:
        denom = max(abs(best.bic), 1e-6)
        improvement_pct = (best.bic - cand.bic) / denom * 100.0
        if improvement_pct >= min_bic_delta_pct:
            best = cand

    chosen_k = best.k
    return KSelectionResult(
        selected_k=chosen_k,
        backend=fitted_by_k[chosen_k],
        regime_set=make_dynamic_regime_set(chosen_k, backend_name),
        candidates=candidates,
        used_fallback_k4=False,
    )
