"""
Regime Detection Layer (Phase 2, §7). 3-state Gaussian HMM (forward-filtered
only), fast panic-spike override, LightGBM distillation classifier,
fallback chain LightGBM -> HMM filter -> rule-based. Phase 3 adds a
data_granularity guard (§4.7) on both model loads.
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

HMM_STATE_LABELS = ["trending_up", "trending_down", "chop_range"]
HMM_OBSERVATION_FEATURES = [
    "momentum_5m", "momentum_10m", "volatility_5m", "volatility_10m", "volume_zscore",
]


def panic_score(volatility_z: float, momentum_z: float, volume_z: float, z_threshold: float = 3.0) -> float:
    composite = max(volatility_z, abs(momentum_z), volume_z)
    return float(1.0 / (1.0 + np.exp(-(composite - z_threshold))))


@dataclass
class HMMStateMapping:
    state_to_label: Dict[int, str]

    def label_order_indices(self) -> List[int]:
        label_to_state = {v: k for k, v in self.state_to_label.items()}
        return [label_to_state[label] for label in HMM_STATE_LABELS]


class _ManualGaussianHMMFilter:
    """Self-contained forward-only Gaussian HMM filter (numpy only, no
    hmmlearn dependency at serving time)."""

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
            logp[i] = -0.5 * (self.k * np.log(2 * np.pi) + self._logdet[i] + diff @ self._cov_inv[i] @ diff)
        return logp

    def filter_step(self, obs: np.ndarray, prev_alpha: Optional[np.ndarray]) -> np.ndarray:
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
        data_granularity: str = "1m",
    ):
        self.confidence_floor = confidence_floor
        self.persistence_window_bars = persistence_window_bars
        self.persistence_transition_threshold = persistence_transition_threshold
        self.panic_z_threshold = panic_z_threshold
        self.serving_source = serving_source
        self.data_granularity = data_granularity

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

    def _granularity_mismatch(self, payload: dict, model_label: str, path: Path) -> bool:
        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"{model_label} at {path} trained on granularity={trained_granularity}, "
                f"running on {self.data_granularity}. Treating as absent (Phase 3 §4.7)."
            )
            return True
        return False

    def _load_models(self, hmm_model_path: str, classifier_model_path: str) -> None:
        hmm_path = Path(hmm_model_path)
        if hmm_path.exists():
            try:
                with open(hmm_path, "rb") as f:
                    payload = pickle.load(f)
                if not self._granularity_mismatch(payload, "Regime HMM", hmm_path):
                    self._hmm_filter = _ManualGaussianHMMFilter(
                        startprob=payload["startprob"], transmat=payload["transmat"],
                        means=payload["means"], covars=payload["covars"],
                    )
                    self._hmm_mapping = HMMStateMapping(state_to_label=payload["state_to_label"])
                    logger.info(f"Loaded regime HMM from {hmm_path}")
            except Exception as e:
                logger.warning(f"Failed to load HMM model at {hmm_path}: {e}")
        else:
            logger.info(f"No HMM model found at {hmm_path}; regime detection will use the rule-based fallback until trained")

        clf_path = Path(classifier_model_path)
        if clf_path.exists():
            try:
                with open(clf_path, "rb") as f:
                    payload = pickle.load(f)
                if not self._granularity_mismatch(payload, "Regime classifier", clf_path):
                    self._classifier = payload["model"]
                    self._classifier_feature_names = payload["feature_names"]
                    logger.info(f"Loaded regime classifier from {clf_path}")
            except Exception as e:
                logger.warning(f"Failed to load regime classifier at {clf_path}: {e}")
        else:
            logger.info(f"No regime classifier found at {clf_path}; falling back to HMM/rule-based")

    def add_bar(self, feature_vector: FeatureVector) -> None:
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

    def compute_regime(self, feature_vector: FeatureVector, symbol: str = "BTC/USDT") -> RegimeState:
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
            "trending_up": float(final[0]), "trending_down": float(final[1]),
            "chop_range": float(final[2]), "panic_spike": float(final[3]),
        }

    def _classify_with_rule(self, feature_vector: FeatureVector) -> Dict[str, float]:
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
            timestamp=timestamp, symbol=symbol,
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
