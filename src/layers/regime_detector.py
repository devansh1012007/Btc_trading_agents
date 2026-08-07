"""
Regime Detection Layer (Phase 2, Module B / §7).

Produces a RegimeState per bar: a probability vector over
{trending_up, trending_down, chop_range, panic_spike} plus a primary_regime
label, confidence, age, and persistence score.

Two model layers combine here:
  1. A 3-state Gaussian HMM (trending_up / trending_down / chop_range),
     filtered forward-only (never smoothed) at serving time (§7.2, §7.4).
  2. A fast, non-HMM panic-spike override rule (§7.3), since PANIC_SPIKE is
     a rare, high-variance tail event the HMM should not try to carve its
     own state out of (Baum-Welch would be unstable trying to fit a
     covariance to so few samples), and needs lower latency than HMM
     filtering can provide.

A LightGBM classifier (§7.5), trained offline against smoothed HMM+panic
labels, is the preferred serving path once trained; RegimeDetector falls
back to raw HMM filtering, then to a simple volatility-threshold rule, if
model files are absent -- the same defensive layering used elsewhere in
the codebase (e.g. MarketDataLayer's ticker fallback).

CRITICAL (§7.6): only *filtering* (forward-only, causal) is ever used in
this serving path. *Smoothing* (whole-sequence decoding) is legitimate
only when generating offline training labels -- see
scripts/train_regime_models.py.

Note: this module has NO hard dependency on hmmlearn or lightgbm. Trained
model parameters are stored as plain arrays / duck-typed objects in the
pickled payloads, so RegimeDetector can run (falling back through the
chain) even in an environment where those libraries aren't installed at
all -- only the training scripts need them.
"""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..state.contracts import FeatureVector, RegimeState, RegimeType
from ..utils.rolling_buffer import RollingBarBuffer

logger = logging.getLogger(__name__)

# Order used consistently everywhere a "3-state" HMM array is indexed.
HMM_STATE_LABELS = ["trending_up", "trending_down", "chop_range"]

# §7.1 observation feature set: small, Gaussian-friendly, HMM-only.
HMM_OBSERVATION_FEATURES = [
    "momentum_5m", "momentum_10m", "volatility_5m", "volatility_10m", "volume_zscore",
]


def panic_score(volatility_z: float, momentum_z: float, volume_z: float, z_threshold: float = 3.0) -> float:
    """
    Smooth 0-1 panic score via a sigmoid over a composite z-score, so
    prob_panic_spike doesn't discontinuously jump across the threshold
    (§7.3).
    """
    composite = max(volatility_z, abs(momentum_z), volume_z)
    return float(1.0 / (1.0 + np.exp(-(composite - z_threshold))))


@dataclass
class HMMStateMapping:
    """
    Maps a fitted HMM's raw hidden-state indices to semantic labels. This
    mapping is fit-specific (state 0 isn't always "trending_up") and must
    be re-derived every time the model is retrained -- see
    scripts/train_regime_models.py.
    """
    state_to_label: Dict[int, str]

    def label_order_indices(self) -> List[int]:
        """Return hidden-state indices ordered to match HMM_STATE_LABELS."""
        label_to_state = {v: k for k, v in self.state_to_label.items()}
        return [label_to_state[label] for label in HMM_STATE_LABELS]


class _ManualGaussianHMMFilter:
    """
    Self-contained forward-only (causal) Gaussian HMM filter, implemented
    with plain numpy from a fitted model's raw parameters (startprob_,
    transmat_, means_, covars_) rather than depending on hmmlearn's
    internal filtering API at serving time. This decouples serving from
    any particular hmmlearn version and means RegimeDetector has zero hard
    dependency on hmmlearn being installed (only train_regime_models.py
    needs it, to produce these arrays in the first place).
    """

    def __init__(self, startprob: np.ndarray, transmat: np.ndarray, means: np.ndarray, covars: np.ndarray):
        self.startprob = np.asarray(startprob, dtype=float)
        self.transmat = np.asarray(transmat, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covars = np.asarray(covars, dtype=float)
        self.n_components = len(self.startprob)
        self.k = self.means.shape[1]
        self._cov_inv = [np.linalg.inv(self.covars[i]) for i in range(self.n_components)]
        self._logdet = [np.linalg.slogdet(self.covars[i])[1] for i in range(self.n_components)]

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        logp = np.zeros(self.n_components)
        for i in range(self.n_components):
            diff = obs - self.means[i]
            logp[i] = -0.5 * (
                self.k * np.log(2 * np.pi) + self._logdet[i] + diff @ self._cov_inv[i] @ diff
            )
        return logp

    def filter_step(self, obs: np.ndarray, prev_alpha: Optional[np.ndarray]) -> np.ndarray:
        """
        One step of the forward algorithm, using only obs up to and
        including the current bar -- filtering, never smoothing (§7.6).
        """
        log_emission = self._log_emission(obs)
        emission = np.exp(log_emission - np.max(log_emission))

        if prev_alpha is None:
            alpha = self.startprob * emission
        else:
            alpha = (prev_alpha @ self.transmat) * emission

        total = alpha.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(self.n_components) / self.n_components
        return alpha / total


class RegimeDetector:
    """Per-bar regime detection with a LightGBM -> HMM -> rule-based fallback chain."""

    def __init__(
        self,
        hmm_model_path: str = "data/models/regime_hmm.pkl",
        classifier_model_path: str = "data/models/regime_classifier.pkl",
        confidence_floor: float = 0.35,
        persistence_window_bars: int = 20,
        persistence_transition_threshold: float = 0.6,
        panic_z_threshold: float = 3.0,
        observation_lookback_bars: int = 100,
        serving_source: str = "auto",
    ):
        """
        Args:
            serving_source: "auto" (prefer classifier > hmm > rule),
                "lgbm_classifier", "hmm_filter", or "rule_based" to force
                a specific path. Regardless of the requested source, this
                always falls back through the chain if the requested
                model isn't actually loaded.
        """
        self.confidence_floor = confidence_floor
        self.persistence_window_bars = persistence_window_bars
        self.persistence_transition_threshold = persistence_transition_threshold
        self.panic_z_threshold = panic_z_threshold
        self.serving_source = serving_source

        self._obs_buffer = RollingBarBuffer(maxlen=observation_lookback_bars)

        self._hmm_filter: Optional[_ManualGaussianHMMFilter] = None
        self._hmm_mapping: Optional[HMMStateMapping] = None
        self._classifier = None
        self._classifier_feature_names: Optional[List[str]] = None

        self._load_models(hmm_model_path, classifier_model_path)

        self._hmm_alpha: Optional[np.ndarray] = None
        self._current_primary: RegimeType = RegimeType.UNKNOWN
        self._regime_age = 0
        self._recent_raw_history: List[str] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self, hmm_model_path: str, classifier_model_path: str) -> None:
        hmm_path = Path(hmm_model_path)
        if hmm_path.exists():
            try:
                with open(hmm_path, "rb") as f:
                    payload = pickle.load(f)
                self._hmm_filter = _ManualGaussianHMMFilter(
                    startprob=payload["startprob"],
                    transmat=payload["transmat"],
                    means=payload["means"],
                    covars=payload["covars"],
                )
                self._hmm_mapping = HMMStateMapping(state_to_label=payload["state_to_label"])
                logger.info(f"Loaded regime HMM from {hmm_path}")
            except Exception as e:
                logger.warning(f"Failed to load HMM model at {hmm_path}: {e}")
        else:
            logger.info(
                f"No HMM model found at {hmm_path}; regime detection will use the "
                "rule-based fallback until trained"
            )

        clf_path = Path(classifier_model_path)
        if clf_path.exists():
            try:
                with open(clf_path, "rb") as f:
                    payload = pickle.load(f)
                self._classifier = payload["model"]
                self._classifier_feature_names = payload["feature_names"]
                logger.info(f"Loaded regime classifier from {clf_path}")
            except Exception as e:
                logger.warning(f"Failed to load regime classifier at {clf_path}: {e}")
        else:
            logger.info(f"No regime classifier found at {clf_path}; falling back to HMM/rule-based")

    # ------------------------------------------------------------------
    # §7.1 Observation feature set
    # ------------------------------------------------------------------

    def add_bar(self, feature_vector: FeatureVector) -> None:
        """
        Feed one bar's FeatureVector in. Builds the small HMM observation
        vector from fields FeatureEngine already computes -- reusing its
        output directly rather than a second parallel feature pipeline.
        """
        raw = feature_vector.raw_features or {}
        self._obs_buffer.append({
            "timestamp": feature_vector.timestamp,
            "momentum_5m": feature_vector.momentum_5m,
            "momentum_10m": feature_vector.momentum_10m,
            "volatility_5m": feature_vector.volatility_5m,
            "volatility_10m": feature_vector.volatility_10m,
            "volatility_30s": feature_vector.volatility_30s,
            "momentum_30s": feature_vector.momentum_30s,
            "volume_zscore": raw.get("volume_zscore", 0.0),
        })

    def _hmm_observation_vector(self) -> Optional[np.ndarray]:
        if len(self._obs_buffer) == 0:
            return None
        bar = self._obs_buffer.latest()
        return np.array([bar[name] for name in HMM_OBSERVATION_FEATURES], dtype=float)

    def _hmm_probs_in_label_order(self, hidden_state_probs: np.ndarray) -> np.ndarray:
        order = self._hmm_mapping.label_order_indices()
        return np.array([hidden_state_probs[i] for i in order])

    # ------------------------------------------------------------------
    # §7.4 Combination into RegimeState
    # ------------------------------------------------------------------

    def compute_regime(self, feature_vector: FeatureVector, symbol: str = "BTC/USDT") -> RegimeState:
        """
        Compute this bar's RegimeState. Serving preference order (unless
        serving_source forces a specific path):
          1. LightGBM classifier, if trained and loaded.
          2. Raw HMM filtering + panic override, if the HMM is trained.
          3. Simple volatility-threshold rule (always available).
        """
        if self.serving_source == "lgbm_classifier" and self._classifier is not None:
            probs = self._classify_with_lgbm(feature_vector)
        elif self.serving_source == "hmm_filter" and self._hmm_filter is not None:
            probs = self._classify_with_hmm(feature_vector)
        elif self.serving_source == "rule_based":
            probs = self._classify_with_rule(feature_vector)
        elif self._classifier is not None:
            probs = self._classify_with_lgbm(feature_vector)
        elif self._hmm_filter is not None:
            probs = self._classify_with_hmm(feature_vector)
        else:
            probs = self._classify_with_rule(feature_vector)

        return self._assemble_regime_state(probs, feature_vector.timestamp, symbol)

    def _classify_with_lgbm(self, feature_vector: FeatureVector) -> Dict[str, float]:
        try:
            row = np.array([[getattr(feature_vector, name, 0.0) or 0.0 for name in self._classifier_feature_names]])
            proba = self._classifier.predict_proba(row)[0]
            classes = list(self._classifier.classes_)
        except Exception as e:
            logger.warning(f"LightGBM regime classifier inference failed, falling back: {e}")
            if self._hmm_filter is not None:
                return self._classify_with_hmm(feature_vector)
            return self._classify_with_rule(feature_vector)

        labels = ["trending_up", "trending_down", "chop_range", "panic_spike"]
        probs = {cls: float(p) for cls, p in zip(classes, proba)}
        for label in labels:
            probs.setdefault(label, 0.0)
        total = sum(probs[l] for l in labels) or 1.0
        return {k: probs[k] / total for k in labels}

    def _classify_with_hmm(self, feature_vector: FeatureVector) -> Dict[str, float]:
        obs = self._hmm_observation_vector()
        p_panic = panic_score(
            volatility_z=self._zscore(feature_vector.volatility_30s, "volatility_30s"),
            momentum_z=self._zscore(feature_vector.momentum_30s, "momentum_30s"),
            volume_z=self._latest_volume_zscore(),
            z_threshold=self.panic_z_threshold,
        )

        if obs is None or self._hmm_filter is None:
            raw_probs = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            alpha = self._hmm_filter.filter_step(obs, self._hmm_alpha)
            self._hmm_alpha = alpha
            raw_probs = self._hmm_probs_in_label_order(alpha)

        final = raw_probs * (1 - p_panic)
        final = np.append(final, p_panic)
        total = final.sum()
        final = final / total if total > 0 else np.array([0.25, 0.25, 0.25, 0.25])

        return {
            "trending_up": float(final[0]),
            "trending_down": float(final[1]),
            "chop_range": float(final[2]),
            "panic_spike": float(final[3]),
        }

    def _classify_with_rule(self, feature_vector: FeatureVector) -> Dict[str, float]:
        """
        Simple volatility-threshold fallback, always available with zero
        trained model dependencies. Coarse, but keeps the pipeline running
        end-to-end before any model has been trained.
        """
        p_panic = panic_score(
            volatility_z=self._zscore(feature_vector.volatility_30s, "volatility_30s"),
            momentum_z=self._zscore(feature_vector.momentum_30s, "momentum_30s"),
            volume_z=self._latest_volume_zscore(),
            z_threshold=self.panic_z_threshold,
        )

        momentum = feature_vector.momentum_5m
        if momentum > 0.05:
            base = {"trending_up": 0.6, "trending_down": 0.1, "chop_range": 0.3}
        elif momentum < -0.05:
            base = {"trending_up": 0.1, "trending_down": 0.6, "chop_range": 0.3}
        else:
            base = {"trending_up": 0.2, "trending_down": 0.2, "chop_range": 0.6}

        final = {k: v * (1 - p_panic) for k, v in base.items()}
        final["panic_spike"] = p_panic
        total = sum(final.values()) or 1.0
        return {k: v / total for k, v in final.items()}

    def _zscore(self, value: float, field_name: str) -> float:
        """Rolling z-score of a scalar feature against its own recent history."""
        if len(self._obs_buffer) < 5:
            return 0.0
        history = np.array([b.get(field_name, 0.0) or 0.0 for b in self._obs_buffer])
        std = np.std(history)
        if std < 1e-9:
            return 0.0
        return float((value - np.mean(history)) / std)

    def _latest_volume_zscore(self) -> float:
        latest = self._obs_buffer.latest()
        if latest is None:
            return 0.0
        return float(latest.get("volume_zscore", 0.0) or 0.0)

    def _assemble_regime_state(self, probs: Dict[str, float], timestamp: int, symbol: str) -> RegimeState:
        primary_label = max(probs, key=probs.get)
        confidence = probs[primary_label]
        insufficient_history = len(self._obs_buffer) < 5

        if confidence < self.confidence_floor or insufficient_history:
            raw_regime = RegimeType.UNKNOWN
        else:
            raw_regime = RegimeType(primary_label)

        self._recent_raw_history.append(raw_regime.value)
        if len(self._recent_raw_history) > self.persistence_window_bars:
            self._recent_raw_history.pop(0)

        # Anti-flicker (§7.4): a display-level regime change only "counts"
        # (resets age) once persistence on the *candidate* new regime
        # clears the transition threshold -- otherwise a single noisy bar
        # could whipsaw the reported label back and forth.
        if raw_regime != self._current_primary:
            candidate_persistence = (
                self._recent_raw_history.count(raw_regime.value) / len(self._recent_raw_history)
            )
            if candidate_persistence >= self.persistence_transition_threshold:
                self._current_primary = raw_regime
                self._regime_age = 0
            else:
                self._regime_age += 1
        else:
            self._regime_age += 1

        persistence_score = (
            self._recent_raw_history.count(self._current_primary.value) / len(self._recent_raw_history)
        )

        return RegimeState(
            timestamp=timestamp,
            symbol=symbol,
            prob_trending_up=probs.get("trending_up", 0.0),
            prob_trending_down=probs.get("trending_down", 0.0),
            prob_chop_range=probs.get("chop_range", 0.0),
            prob_panic_spike=probs.get("panic_spike", 0.0),
            primary_regime=self._current_primary,
            regime_confidence=confidence,
            hmm_hidden_state=int(np.argmax(self._hmm_alpha)) if self._hmm_alpha is not None else None,
            hmm_probabilities=[float(x) for x in self._hmm_alpha] if self._hmm_alpha is not None else [],
            regime_age=self._regime_age,
            regime_persistence_score=persistence_score,
        )
