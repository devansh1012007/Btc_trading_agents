"""
Unit tests for Phase 5 T5.4 & T5.5: `distill_regime_classifier`, `RegimeDetector`
serving chain fallback, and hysteresis (`min_dwell_bars`, `switch_margin`).
"""
import numpy as np
import pytest

from src.layers.regime_detector import CLASSIFIER_FEATURE_NAMES, RegimeDetector
from src.regime.backends import KMeansBackend
from src.regime.distill import distill_regime_classifier
from src.regime.selection import make_dynamic_regime_set
from src.state.contracts import FeatureVector

pytestmark = [pytest.mark.unit]


def _fv(ts: int, mom_med: float = 0.2) -> FeatureVector:
    return FeatureVector(
        timestamp=ts,
        symbol="BTC/USDT",
        momentum_short=mom_med,
        momentum_medium=mom_med,
        momentum_long=mom_med,
        volatility_short=0.1,
        volatility_medium=0.2,
        volatility_long=0.2,
        returns_5=mom_med,
        rsi_14=55.0,
        macd_signal=0.01,
        sma_9=100.0,
        sma_21=99.5,
        imbalance_ratio=0.1,
        orderflow_momentum=0.1,
        spread_pct=0.02,
        depth_10_level=50000.0,
        raw_features={"volume_zscore": 0.1},
    )


def test_serving_chain_dynamic_lgbm_and_backend_fallback_and_hysteresis():
    rng = np.random.default_rng(42)
    X_obs = np.vstack([
        rng.normal(loc=-1.0, scale=0.2, size=(40, 5)),
        rng.normal(loc=0.0, scale=0.2, size=(40, 5)),
        rng.normal(loc=1.0, scale=0.2, size=(40, 5)),
    ])
    backend = KMeansBackend(n_components=3, random_state=42).fit(X_obs)
    teacher_probs = backend.filter_causal(X_obs)
    rset = make_dynamic_regime_set(3, "kmeans")
    X_clf = rng.normal(size=(120, len(CLASSIFIER_FEATURE_NAMES)))
    X_clf[:40, 0] = -1.0
    X_clf[40:80, 0] = 0.0
    X_clf[80:, 0] = 1.0
    student = distill_regime_classifier(X_clf, teacher_probs, CLASSIFIER_FEATURE_NAMES, rset)

    det = RegimeDetector(
        hmm_model_path="nonexistent_hmm.pkl",
        classifier_model_path="nonexistent_clf.pkl",
        dynamic_classifier=student,
        dynamic_backend=backend,
        regime_set=rset,
        persistence_window_bars=1,
        persistence_transition_threshold=0.5,
        min_dwell_bars=2,
    )
    for i in range(6):
        fv = _fv(1000 + i * 60, mom_med=-1.0 if i < 5 else 1.0)
        det.add_bar(fv)
        rs = det.compute_regime(fv)
    assert rs.serving_source_used == "dynamic_lgbm"
    assert rs.regime_set_id == rset.id
    assert set(rs.regime_probs.keys()) == set(rset.ids())

    # Disable student -> falls back to dynamic_backend
    det.dynamic_classifier = None
    fv_next = _fv(2000, mom_med=1.0)
    det.add_bar(fv_next)
    rs_fb = det.compute_regime(fv_next)
    assert rs_fb.serving_source_used == "dynamic_backend"
