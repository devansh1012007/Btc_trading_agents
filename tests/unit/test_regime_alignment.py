"""
Unit tests for Phase 5 T5.3: `align_regime_labels` Hungarian matching and
`lock_between_retrains` safety gate.
"""
import numpy as np
import pytest

from src.regime.alignment import (
    RegimeRelabelBlockedError,
    align_regime_labels,
)

pytestmark = [pytest.mark.unit]


def test_align_regime_labels_recovers_known_permutation():
    old_c = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    # Permute order: new state 0 is old 2, new 1 is old 0, new 2 is old 1
    new_c = np.array([[2.05, 0.01], [-1.95, -0.02], [0.02, 0.01]])
    res = align_regime_labels(old_c, new_c, min_match_jaccard=0.5, lock_between_retrains=True)
    assert res.accepted
    assert res.new_to_old == {0: 2, 1: 0, 2: 1}


def test_lock_between_retrains_raises_on_k_change_or_poor_jaccard():
    old_c = np.array([[-1.0, 0.0], [1.0, 0.0]])
    new_c_k3 = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(RegimeRelabelBlockedError, match="Regime count changed"):
        align_regime_labels(old_c, new_c_k3, lock_between_retrains=True)

    # Disjoint state assignments -> Jaccard = 0 < 0.6 -> raises when locked
    old_probs = np.array([[1.0, 0.0, 0.0]] * 50)
    new_probs = np.array([[0.34, 0.33, 0.33]] * 25 + [[0.0, 1.0, 0.0]] * 25)
    c3 = np.zeros((3, 2))
    with pytest.raises(RegimeRelabelBlockedError, match="min_jaccard"):
        align_regime_labels(
            c3, c3, old_probs=old_probs, new_probs=new_probs,
            min_match_jaccard=0.8, lock_between_retrains=True,
        )
