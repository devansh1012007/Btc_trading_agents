"""
Unit tests for Phase 5 T5.1: all 5 `RegimeBackend` implementations
(`gaussian_hmm`, `gmm`, `kmeans`, `ms_ar`, `garch_mixture`), verifying:
* valid probability simplex rows (non-negative, sum to 1)
* strict causality (L1: modifying future observations `t > t0` never alters `filter_causal(X)[t0]`)
* pure-NumPy `to_serving_params()` / `from_serving_params()` round-trip parity.
"""
import numpy as np
import pytest

from src.regime.backends import BACKENDS, RegimeBackend, build_regime_backend

pytestmark = [pytest.mark.unit]


def _synthetic_regime_matrix(seed: int = 7, n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[-1.2, 0.1, 0.2], scale=0.25, size=(n // 3, 3))
    c2 = rng.normal(loc=[0.0, 0.0, 0.1], scale=0.20, size=(n // 3, 3))
    c3 = rng.normal(loc=[1.4, -0.1, 0.5], scale=0.30, size=(n - 2 * (n // 3), 3))
    return np.vstack([c1, c2, c3])


@pytest.mark.parametrize("backend_name", sorted(BACKENDS.keys()))
def test_regime_backend_fit_causal_filter_and_serving_roundtrip(backend_name: str):
    X = _synthetic_regime_matrix()
    b = build_regime_backend(backend_name, n_components=3, random_state=42).fit(X)
    probs = b.filter_causal(X)
    assert probs.shape == (X.shape[0], 3)
    assert np.all(probs >= 0.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)
    assert np.isfinite(b.bic(X))

    # Strict causality (L1): corrupting future rows t > 50 must NEVER change probs[:51]
    X_future_corrupted = X.copy()
    X_future_corrupted[51:] += 50.0
    probs_corrupted = b.filter_causal(X_future_corrupted)
    assert np.allclose(probs[:51], probs_corrupted[:51], atol=1e-12)

    # Pure-NumPy serving params round-trip
    restored = RegimeBackend.from_serving_params(b.to_serving_params())
    probs_restored = restored.filter_causal(X)
    assert np.allclose(probs, probs_restored, atol=1e-9)
