"""
Pluggable unsupervised/semi-supervised Regime Backends (Phase 5, T5.1):
* `gaussian_hmm`: Gaussian Hidden Markov Model with causal Hamilton forward filter
* `gmm`: Gaussian Mixture Model with diagonal Gaussian posterior
* `kmeans`: K-Means clustering with soft radial basis posterior
* `ms_ar`: Markov-Switching Autoregressive (`AR(p)`) model with Hamilton filter
* `garch_mixture`: Causal GARCH(1,1) conditional volatility + Gaussian mixture/HMM (A4)

Every backend implements a pure-NumPy causal `filter_causal(X)` and
single-observation `filter_step(x, prior)` so serving never runs a two-sided
smoother (e.g. Rauch-Tung-Striebel / forward-backward) and never requires
heavy training libraries at serve time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def _log_diag_gaussian(X: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """Compute log N(x | mu_k, diag(var_k)) for X of shape (N, D) or (D,)."""
    x2d = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = x2d.shape
    k = means.shape[0]
    safe_vars = np.maximum(vars_, 1e-6)
    out = np.zeros((n, k), dtype=float)
    for j in range(k):
        diff = x2d - means[j]
        logdet = float(np.sum(np.log(safe_vars[j])))
        quad = np.sum((diff * diff) / safe_vars[j], axis=1)
        out[:, j] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
    return out if X.ndim == 2 else out[0]


def _softmax_rows(log_w: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(log_w, dtype=float))
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    w = np.exp(np.clip(shifted, -60.0, 60.0))
    sums = np.sum(w, axis=1, keepdims=True)
    probs = np.where(sums > 0, w / sums, 1.0 / arr.shape[1])
    return probs if log_w.ndim == 2 else probs[0]


class RegimeBackend(ABC):
    """Uniform train-time + pure-NumPy causal serve-time regime backend."""

    backend_name: str = "base"

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.means_: np.ndarray = np.zeros((self.n_components, 1), dtype=float)
        self.vars_: np.ndarray = np.ones((self.n_components, 1), dtype=float)
        self.startprob_: np.ndarray = np.ones(self.n_components, dtype=float) / self.n_components
        self.transmat_: np.ndarray = np.eye(self.n_components, dtype=float)
        self._log_lik_: float = 0.0
        self._n_params_: int = 1

    @abstractmethod
    def fit(self, X: np.ndarray) -> "RegimeBackend":
        raise NotImplementedError

    @abstractmethod
    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Single-bar causal filter step returning `(posterior_probs, next_state_prior)`."""
        raise NotImplementedError

    def filter_causal(self, X: np.ndarray) -> np.ndarray:
        """Run `filter_step` causally from t=0..T-1; returns shape `(T, K)`."""
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((arr.shape[0], self.n_components), dtype=float)
        state: Optional[np.ndarray] = None
        for t in range(arr.shape[0]):
            probs, state = self.filter_step(arr[t], state)
            out[t] = probs
        return out

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion on `X` using causal log-likelihood."""
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        n = max(1, arr.shape[0])
        ll = self.causal_log_likelihood(arr)
        return float(-2.0 * ll + self._n_params_ * np.log(n))

    def causal_log_likelihood(self, X: np.ndarray) -> float:
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        log_em = _log_diag_gaussian(arr, self.means_, self.vars_)
        alpha = self.startprob_.copy()
        total_ll = 0.0
        for t in range(arr.shape[0]):
            pred = alpha if t == 0 else (alpha @ self.transmat_)
            pred = np.maximum(pred, 1e-12)
            pred /= np.sum(pred)
            m = float(np.max(log_em[t]))
            em = np.exp(log_em[t] - m)
            unnorm = pred * em
            c = float(np.sum(unnorm))
            if c > 0 and np.isfinite(c):
                total_ll += m + float(np.log(c))
                alpha = unnorm / c
            else:
                alpha = np.ones(self.n_components) / self.n_components
        return total_ll

    def to_serving_params(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "n_components": self.n_components,
            "means": self.means_.tolist(),
            "vars": self.vars_.tolist(),
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
        }

    @classmethod
    def from_serving_params(cls, params: Mapping[str, Any]) -> "RegimeBackend":
        bname = str(params.get("backend_name", "gaussian_hmm"))
        k = int(params.get("n_components", 4))
        inst = build_regime_backend(bname, n_components=k)
        inst.means_ = np.asarray(params["means"], dtype=float)
        inst.vars_ = np.asarray(params["vars"], dtype=float)
        inst.startprob_ = np.asarray(params["startprob"], dtype=float)
        inst.transmat_ = np.asarray(params["transmat"], dtype=float)
        if "ar_phis" in params:
            setattr(inst, "ar_phis_", np.asarray(params["ar_phis"], dtype=float))
        if "garch_params" in params:
            setattr(inst, "garch_params_", tuple(float(v) for v in params["garch_params"]))
        return inst


def _deterministic_kmeans_init(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic K-means++ initialization and Lloyd iterations in pure NumPy."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    centers = np.zeros((k, d), dtype=float)
    first = int(rng.integers(0, max(1, n)))
    centers[0] = X[first]
    for c_idx in range(1, k):
        dists = np.min(
            np.sum((X[:, None, :] - centers[None, :c_idx, :]) ** 2, axis=2),
            axis=1,
        )
        total = float(np.sum(dists))
        if total <= 1e-12:
            centers[c_idx] = X[c_idx % n] + rng.normal(0, 1e-4, size=d)
        else:
            probs = dists / total
            choice = int(rng.choice(n, p=probs))
            centers[c_idx] = X[choice]

    labels = np.zeros(n, dtype=int)
    for _ in range(25):
        dmat = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dmat, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = np.mean(X[mask], axis=0)
    # Canonical ordering by first feature mean so state indices are deterministic
    order = np.argsort(centers[:, 0])
    centers = centers[order]
    dmat = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dmat, axis=1)
    return centers, labels


class KMeansBackend(RegimeBackend):
    backend_name = "kmeans"

    def fit(self, X: np.ndarray) -> "KMeansBackend":
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = arr.shape
        k = self.n_components
        centers, labels = _deterministic_kmeans_init(arr, k, self.random_state)
        self.means_ = centers
        vars_ = np.zeros((k, d), dtype=float)
        counts = np.zeros(k, dtype=float)
        global_var = np.maximum(np.var(arr, axis=0), 1e-4)
        for j in range(k):
            mask = labels == j
            counts[j] = float(np.sum(mask))
            if counts[j] > 1:
                vars_[j] = np.maximum(np.var(arr[mask], axis=0), 1e-5)
            else:
                vars_[j] = global_var
        self.vars_ = vars_
        self.startprob_ = np.maximum(counts / max(1.0, float(n)), 1e-3)
        self.startprob_ /= np.sum(self.startprob_)
        self.transmat_ = np.tile(self.startprob_, (k, 1))
        self._n_params_ = k * d * 2 + (k - 1)
        return self

    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        _ = prior
        log_em = _log_diag_gaussian(np.asarray(x, dtype=float), self.means_, self.vars_)
        probs = _softmax_rows(log_em + np.log(np.maximum(self.startprob_, 1e-9)))
        return probs, probs


class GMMBackend(RegimeBackend):
    backend_name = "gmm"

    def fit(self, X: np.ndarray) -> "GMMBackend":
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = arr.shape
        k = self.n_components
        km = KMeansBackend(n_components=k, random_state=self.random_state).fit(arr)
        means = km.means_.copy()
        vars_ = km.vars_.copy()
        weights = km.startprob_.copy()

        # Run diagonal-covariance EM for 25 iterations
        for _ in range(25):
            log_em = _log_diag_gaussian(arr, means, vars_) + np.log(np.maximum(weights, 1e-9))[None, :]
            resp = _softmax_rows(log_em)
            nk = np.maximum(np.sum(resp, axis=0), 1e-6)
            weights = nk / float(n)
            means = (resp.T @ arr) / nk[:, None]
            for j in range(k):
                diff = arr - means[j]
                vars_[j] = np.maximum(np.sum(resp[:, j : j + 1] * (diff * diff), axis=0) / nk[j], 1e-5)

        order = np.argsort(means[:, 0])
        self.means_ = means[order]
        self.vars_ = vars_[order]
        self.startprob_ = weights[order]
        self.transmat_ = np.tile(self.startprob_, (k, 1))
        self._n_params_ = k * d * 2 + (k - 1)
        return self

    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        _ = prior
        log_em = _log_diag_gaussian(np.asarray(x, dtype=float), self.means_, self.vars_)
        probs = _softmax_rows(log_em + np.log(np.maximum(self.startprob_, 1e-9)))
        return probs, probs


class GaussianHMMBackend(RegimeBackend):
    backend_name = "gaussian_hmm"

    def fit(self, X: np.ndarray) -> "GaussianHMMBackend":
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = arr.shape
        k = self.n_components
        gmm = GMMBackend(n_components=k, random_state=self.random_state).fit(arr)
        means = gmm.means_.copy()
        vars_ = gmm.vars_.copy()
        startprob = gmm.startprob_.copy()
        # Sticky transition matrix initialization
        transmat = np.full((k, k), 0.1 / max(1, k - 1), dtype=float)
        np.fill_diagonal(transmat, 0.9 if k > 1 else 1.0)

        for _ in range(15):
            log_em = _log_diag_gaussian(arr, means, vars_)
            em = np.exp(np.clip(log_em - np.max(log_em, axis=1, keepdims=True), -60.0, 0.0))
            alpha = np.zeros((n, k), dtype=float)
            a0 = startprob * em[0]
            alpha[0] = a0 / max(float(np.sum(a0)), 1e-12)
            xi_sum = np.zeros((k, k), dtype=float)
            for t in range(1, n):
                pred = alpha[t - 1] @ transmat
                at = pred * em[t]
                s = float(np.sum(at))
                alpha[t] = at / s if s > 1e-12 else (np.ones(k) / k)
                joint = (alpha[t - 1][:, None] * transmat) * em[t][None, :]
                js = float(np.sum(joint))
                if js > 1e-12:
                    xi_sum += joint / js

            nk = np.maximum(np.sum(alpha, axis=0), 1e-6)
            means = (alpha.T @ arr) / nk[:, None]
            for j in range(k):
                diff = arr - means[j]
                vars_[j] = np.maximum(np.sum(alpha[:, j : j + 1] * (diff * diff), axis=0) / nk[j], 1e-5)
            row_sums = np.maximum(np.sum(xi_sum, axis=1, keepdims=True), 1e-6)
            transmat = np.clip(xi_sum / row_sums, 1e-4, 1.0)
            transmat /= np.sum(transmat, axis=1, keepdims=True)
            startprob = np.clip(alpha[0], 1e-3, 1.0)
            startprob /= np.sum(startprob)

        order = np.argsort(means[:, 0])
        self.means_ = means[order]
        self.vars_ = vars_[order]
        self.startprob_ = startprob[order]
        self.transmat_ = transmat[np.ix_(order, order)]
        self._n_params_ = k * d * 2 + k * (k - 1) + (k - 1)
        return self

    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        log_em = _log_diag_gaussian(np.asarray(x, dtype=float), self.means_, self.vars_)
        em = np.exp(np.clip(log_em - float(np.max(log_em)), -60.0, 0.0))
        pred = self.startprob_ if prior is None else (np.asarray(prior, dtype=float) @ self.transmat_)
        alpha = pred * em
        total = float(np.sum(alpha))
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(self.n_components, dtype=float) / self.n_components
        else:
            probs = alpha / total
        return probs, probs


class MSARBackend(RegimeBackend):
    """Markov-Switching Autoregressive (`AR(1)`) model with Hamilton forward filter."""

    backend_name = "ms_ar"

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        super().__init__(n_components=n_components, random_state=random_state)
        self.ar_phis_: np.ndarray = np.zeros(self.n_components, dtype=float)
        self._last_y: float = 0.0

    def fit(self, X: np.ndarray) -> "MSARBackend":
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        hmm = GaussianHMMBackend(n_components=self.n_components, random_state=self.random_state).fit(arr)
        self.means_ = hmm.means_.copy()
        self.vars_ = hmm.vars_.copy()
        self.startprob_ = hmm.startprob_.copy()
        self.transmat_ = hmm.transmat_.copy()

        # Estimate state-specific AR(1) coefficient on primary series X[:, 0]
        probs = hmm.filter_causal(arr)
        y = arr[:, 0]
        phis = np.zeros(self.n_components, dtype=float)
        if len(y) > 2:
            y_prev = y[:-1]
            y_curr = y[1:]
            for j in range(self.n_components):
                w = probs[1:, j]
                yc = y_curr - self.means_[j, 0]
                yp = y_prev - self.means_[j, 0]
                denom = float(np.sum(w * yp * yp))
                if denom > 1e-9:
                    phis[j] = float(np.clip(np.sum(w * yp * yc) / denom, -0.95, 0.95))
        self.ar_phis_ = phis
        d = arr.shape[1]
        k = self.n_components
        self._n_params_ = k * d * 2 + k + k * (k - 1) + (k - 1)
        return self

    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(x, dtype=float).copy()
        k = self.n_components
        d = self.means_.shape[1]
        logp = np.zeros(k, dtype=float)
        for j in range(k):
            mu = self.means_[j].copy()
            if prior is not None:
                mu[0] = self.means_[j, 0] + self.ar_phis_[j] * (self._last_y - self.means_[j, 0])
            diff = obs - mu
            safe_var = np.maximum(self.vars_[j], 1e-6)
            logp[j] = -0.5 * (
                d * np.log(2.0 * np.pi) + float(np.sum(np.log(safe_var))) + float(np.sum((diff * diff) / safe_var))
            )
        self._last_y = float(obs[0])
        em = np.exp(np.clip(logp - float(np.max(logp)), -60.0, 0.0))
        pred = self.startprob_ if prior is None else (np.asarray(prior, dtype=float) @ self.transmat_)
        alpha = pred * em
        total = float(np.sum(alpha))
        probs = alpha / total if (total > 0 and np.isfinite(total)) else (np.ones(k) / k)
        return probs, probs

    def to_serving_params(self) -> Dict[str, Any]:
        d = super().to_serving_params()
        d["ar_phis"] = self.ar_phis_.tolist()
        return d


class GARCHMixtureBackend(RegimeBackend):
    """Causal GARCH(1,1) volatility feature generator + Gaussian HMM mixture (A4)."""

    backend_name = "garch_mixture"

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        super().__init__(n_components=n_components, random_state=random_state)
        self.garch_params_: Tuple[float, float, float] = (1e-4, 0.12, 0.85)
        self._sigma2: float = 1e-3

    def _causal_garch_feature(self, r: np.ndarray) -> np.ndarray:
        omega, alpha, beta = self.garch_params_
        n = len(r)
        sig2 = np.zeros(n, dtype=float)
        curr = max(float(np.var(r[: min(20, n)])), 1e-6)
        for t in range(n):
            if t > 0:
                curr = omega + alpha * float(r[t - 1] ** 2) + beta * curr
            sig2[t] = max(curr, 1e-8)
        return 0.5 * np.log(sig2)

    def _augment(self, X: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        g_col = self._causal_garch_feature(arr[:, 0])[:, None]
        return np.hstack([arr, g_col])

    def fit(self, X: np.ndarray) -> "GARCHMixtureBackend":
        aug = self._augment(X)
        hmm = GaussianHMMBackend(n_components=self.n_components, random_state=self.random_state).fit(aug)
        self.means_ = hmm.means_.copy()
        self.vars_ = hmm.vars_.copy()
        self.startprob_ = hmm.startprob_.copy()
        self.transmat_ = hmm.transmat_.copy()
        self._n_params_ = hmm._n_params_ + 3
        self._sigma2 = float(np.exp(2.0 * self.means_[0, -1]))
        return self

    def filter_step(self, x: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(x, dtype=float)
        omega, alpha, beta = self.garch_params_
        if prior is None:
            self._sigma2 = max(float(obs[0] ** 2), 1e-6)
        else:
            self._sigma2 = max(omega + alpha * float(obs[0] ** 2) + beta * self._sigma2, 1e-8)
        aug_obs = np.append(obs, 0.5 * np.log(self._sigma2))
        log_em = _log_diag_gaussian(aug_obs, self.means_, self.vars_)
        em = np.exp(np.clip(log_em - float(np.max(log_em)), -60.0, 0.0))
        pred = self.startprob_ if prior is None else (np.asarray(prior, dtype=float) @ self.transmat_)
        post = pred * em
        total = float(np.sum(post))
        probs = post / total if (total > 0 and np.isfinite(total)) else (np.ones(self.n_components) / self.n_components)
        return probs, probs

    def filter_causal(self, X: np.ndarray) -> np.ndarray:
        aug = self._augment(X)
        arr = np.atleast_2d(aug)
        out = np.zeros((arr.shape[0], self.n_components), dtype=float)
        prior: Optional[np.ndarray] = None
        for t in range(arr.shape[0]):
            log_em = _log_diag_gaussian(arr[t], self.means_, self.vars_)
            em = np.exp(np.clip(log_em - float(np.max(log_em)), -60.0, 0.0))
            pred = self.startprob_ if prior is None else (prior @ self.transmat_)
            post = pred * em
            s = float(np.sum(post))
            prior = post / s if s > 0 else (np.ones(self.n_components) / self.n_components)
            out[t] = prior
        return out

    def bic(self, X: np.ndarray) -> float:
        aug = self._augment(X)
        return super().bic(aug)

    def to_serving_params(self) -> Dict[str, Any]:
        d = super().to_serving_params()
        d["garch_params"] = list(self.garch_params_)
        return d


BACKENDS = {
    "gaussian_hmm": GaussianHMMBackend,
    "gmm": GMMBackend,
    "kmeans": KMeansBackend,
    "ms_ar": MSARBackend,
    "garch_mixture": GARCHMixtureBackend,
}


def build_regime_backend(name: str, n_components: int = 4, random_state: int = 42) -> RegimeBackend:
    key = str(name).lower()
    if key not in BACKENDS:
        raise ValueError(f"Unknown regime backend {name!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[key](n_components=n_components, random_state=random_state)
