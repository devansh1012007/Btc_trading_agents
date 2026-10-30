"""
Student distillation (`distill_regime_classifier`, Phase 5 T5.4): fits a fast
calibrated classifier on causal teacher regime probabilities and records the
`RegimeSet` and `regime_set_hash`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from lightgbm import LGBMClassifier

from ..state.contracts import RegimeSet


@dataclass
class DistilledRegimeClassifier:
    model: object
    feature_names: List[str]
    regime_set: RegimeSet
    classes: List[str]

    def predict_probs_dict(self, row_2d: np.ndarray) -> Dict[str, float]:
        proba = self.model.predict_proba(row_2d)[0]
        raw = {str(cls): float(p) for cls, p in zip(self.classes, proba, strict=False)}
        ids = self.regime_set.ids()
        for rid in ids:
            raw.setdefault(rid, 0.0)
        total = sum(raw[rid] for rid in ids) or 1.0
        return {rid: raw[rid] / total for rid in ids}


def distill_regime_classifier(
    X_features: np.ndarray,
    teacher_probs: np.ndarray,
    feature_names: Sequence[str],
    regime_set: RegimeSet,
    *,
    random_state: int = 42,
) -> DistilledRegimeClassifier:
    """Train a fast student classifier on causal teacher labels `argmax(teacher_probs)`."""
    X_arr = np.atleast_2d(np.asarray(X_features, dtype=float))
    probs_arr = np.atleast_2d(np.asarray(teacher_probs, dtype=float))
    ids = regime_set.ids()
    label_idx = np.argmax(probs_arr, axis=1)
    y_str = np.array([ids[int(i)] for i in label_idx], dtype=object)

    clf = LGBMClassifier(n_estimators=30, max_depth=4, random_state=random_state, verbose=-1)
    clf.fit(X_arr, y_str)
    classes = [str(c) for c in clf.classes_]
    return DistilledRegimeClassifier(
        model=clf,
        feature_names=list(feature_names),
        regime_set=regime_set,
        classes=classes,
    )
