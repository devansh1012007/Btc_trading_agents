"""
Causal regime-chopped dataset builder (`chop_candidates_by_served_regime`,
Phase 9 T9.2 / Leakage Control L2).

Slices candidate training rows strictly by the **causal served regime**
recorded at bar `t` (`served_regime` / `regime` from `RegimeDetector.compute_regime`),
never by two-sided smoothed HMM states. When the in-regime sample count is
below `min_samples_for_chop`, applies hierarchical shrinkage by retaining
out-of-regime rows with weight `global_prior_weight`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import numpy as np


class NonCausalRegimeColumnError(ValueError):
    """Raised if a caller attempts to chop training data on smoothed / two-sided regime labels."""


@dataclass(frozen=True)
class ChoppedTrainingSet:
    rows: List[Mapping[str, Any]]
    sample_weights: np.ndarray
    in_regime_count: int
    used_hierarchical_fallback: bool


def chop_candidates_by_served_regime(
    rows: Sequence[Mapping[str, Any]],
    regime_scope: Sequence[str],
    *,
    regime_col: str = "regime",
    min_samples_for_chop: int = 30,
    global_prior_weight: float = 0.25,
) -> ChoppedTrainingSet:
    """Slice `rows` to `regime_scope` using only causal served regime labels (L2)."""
    if "smoothed" in regime_col.lower() or "two_sided" in regime_col.lower() or "viterbi_full" in regime_col.lower():
        raise NonCausalRegimeColumnError(
            f"Refusing to chop training data on non-causal regime column {regime_col!r} (L2 violation)"
        )

    scope_norm = {str(r).strip().lower() for r in regime_scope}
    in_rows: List[Mapping[str, Any]] = []
    all_rows: List[Mapping[str, Any]] = list(rows)
    is_in_mask: List[bool] = []

    for r in all_rows:
        if "smoothed_regime" in r and regime_col == "smoothed_regime":
            raise NonCausalRegimeColumnError("Row carries smoothed_regime as chop key (L2 violation)")
        val = str(r.get(regime_col, "")).strip().lower()
        matched = val in scope_norm
        is_in_mask.append(matched)
        if matched:
            in_rows.append(r)

    in_count = len(in_rows)
    if in_count >= min_samples_for_chop:
        return ChoppedTrainingSet(
            rows=in_rows,
            sample_weights=np.ones(in_count, dtype=float),
            in_regime_count=in_count,
            used_hierarchical_fallback=False,
        )

    # Hierarchical shrinkage fallback when in_count < min_samples_for_chop
    w_prior = max(1e-3, min(1.0, float(global_prior_weight)))
    weights = np.array([1.0 if m else w_prior for m in is_in_mask], dtype=float)
    return ChoppedTrainingSet(
        rows=all_rows,
        sample_weights=weights,
        in_regime_count=in_count,
        used_hierarchical_fallback=True,
    )
