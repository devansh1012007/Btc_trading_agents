"""
Train the Regime Detection Layer (§7.2 HMM, §7.5 LightGBM distillation).

Two models are produced:
  1. A 3-state Gaussian HMM (trending_up / trending_down / chop_range),
     fit on the small, Gaussian-friendly observation set from §7.1.
     PANIC_SPIKE is deliberately excluded from the HMM (handled
     separately, see regime_detector.panic_score) -- it's a rare,
     high-variance tail event that would destabilize Baum-Welch's
     covariance fit.
  2. A LightGBM multiclass classifier, trained on (FeatureVector at T) ->
     (offline HMM+panic *smoothed*-decode label at T), so serving can run
     off one feature pipeline instead of maintaining a second HMM-only one.

Smoothing (whole-sequence decoding) is used ONLY here, to build training
labels offline. RegimeDetector's serving path uses causal, forward-only
filtering exclusively (see regime_detector.py / §7.6) -- never confuse the
two.

Usage:
    python scripts/train_regime_models.py --input data/raw/BTC_USDT_2026-01-01.parquet
"""
import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.layers.market_data import MarketDataLayer
from src.layers.feature_engine import FeatureEngine
from src.layers.regime_detector import panic_score, HMM_STATE_LABELS, HMM_OBSERVATION_FEATURES

from src.backtest.walk_forward import WalkForwardSplitter

CLASSIFIER_FEATURE_NAMES = [
    "momentum_30s", "momentum_5m", "momentum_10m",
    "volatility_30s", "volatility_5m", "volatility_10m",
    "returns_5", "rsi_14", "macd_signal", "sma_9", "sma_21",
    "imbalance_ratio", "orderflow_momentum", "spread_pct", "depth_10_level",
]


def load_config(path: str = "config/settings.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def build_feature_dataset(df_ohlcv: pd.DataFrame, symbol: str, lookback_bars: int) -> pd.DataFrame:
    """Replay bars through FeatureEngine and return a flat DataFrame of
    FeatureVector fields, one row per bar with enough history."""
    feature_engine = FeatureEngine(lookback_bars=lookback_bars)
    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)

    rows = []
    for snapshot in snapshots:
        feature_engine.add_bar(snapshot)
        fv = feature_engine.compute_features(snapshot)
        if fv is None:
            continue
        row = {name: getattr(fv, name) for name in CLASSIFIER_FEATURE_NAMES}
        row["volume_zscore"] = fv.raw_features.get("volume_zscore", 0.0)
        row["timestamp"] = fv.timestamp
        rows.append(row)

    return pd.DataFrame(rows)


def _require_hmmlearn():
    try:
        from hmmlearn import hmm
        return hmm
    except ImportError as e:
        raise ImportError(
            "hmmlearn is required to train the regime HMM but failed to import. "
            "Try `pip install hmmlearn` (a prebuilt wheel should work on most "
            "platforms without compilation). If it still fails to build, install "
            "python3-dev/build-essential first, or vendor a minimal NumPy/SciPy "
            "Gaussian HMM implementation (forward-backward + Baum-Welch for 3 "
            "states is not much code) rather than blocking on this a second time "
            "-- see PHASE1_COMPLETE.md's Known Issues."
        ) from e


def _bic(model, X: np.ndarray) -> float:
    """Bayesian Information Criterion for a fitted GaussianHMM (full covariance)."""
    n_params = (
        model.n_components * (model.n_components - 1)  # transition matrix
        + (model.n_components - 1)                       # start prob
        + model.n_components * model.n_features           # means
        + model.n_components * model.n_features * (model.n_features + 1) // 2  # full covariances
    )
    log_likelihood = model.score(X)
    return -2 * log_likelihood + n_params * np.log(len(X))


def fit_hmm_with_bic(X: np.ndarray, candidate_n_states: List[int]) -> Dict[int, float]:
    """
    Fit a GaussianHMM for each candidate state count and report BIC, purely
    as a diagnostic per §7.2 ("propose 2-5, expect 3 to win"). The
    downstream architecture (post-hoc label mapping, panic override,
    RegimeDetector's HMM_STATE_LABELS) assumes exactly 3 non-panic states,
    so the final fitted model always uses n_states=3 regardless of which
    candidate the BIC grid prefers -- this search validates that choice,
    it doesn't change the architecture automatically.
    """
    hmm = _require_hmmlearn()

    results = {}
    for n in candidate_n_states:
        try:
            model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=200, random_state=42)
            model.fit(X)
            bic = _bic(model, X)
            results[n] = bic
            logger.info(f"n_states={n}: BIC={bic:.2f}, log-likelihood={model.score(X):.2f}")
        except Exception as e:
            logger.warning(f"HMM fit failed for n_states={n}: {e}")

    if results:
        best_n = min(results, key=results.get)
        logger.info(f"BIC grid winner: n_states={best_n} (fitting final model with n_states=3 regardless)")
    return results


def fit_final_hmm(X: np.ndarray, n_states: int = 3):
    hmm = _require_hmmlearn()
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    model.fit(X)
    return model


def map_states_to_labels(model, X: np.ndarray) -> Dict[int, str]:
    """
    Post-hoc label mapping (§7.2): sort discovered states by mean signed
    return -- most negative -> trending_down, most positive -> trending_up,
    lowest volatility -> chop_range. Fit-specific; re-derived every retrain.
    """
    hidden_states = model.predict(X)  # smoothed decode, offline-only -- fine here (labels, not serving)
    momentum_col = HMM_OBSERVATION_FEATURES.index("momentum_5m")
    volatility_col = HMM_OBSERVATION_FEATURES.index("volatility_5m")

    stats = []
    for state in range(model.n_components):
        mask = hidden_states == state
        if mask.sum() == 0:
            stats.append((state, 0.0, 0.0))
            continue
        mean_momentum = float(X[mask, momentum_col].mean())
        mean_volatility = float(X[mask, volatility_col].mean())
        stats.append((state, mean_momentum, mean_volatility))

    by_momentum = sorted(stats, key=lambda s: s[1])
    trending_down_state = by_momentum[0][0]
    trending_up_state = by_momentum[-1][0]

    remaining = [s for s in stats if s[0] not in (trending_down_state, trending_up_state)]
    chop_state = min(remaining, key=lambda s: s[2])[0] if remaining else trending_down_state

    mapping = {trending_up_state: "trending_up", trending_down_state: "trending_down", chop_state: "chop_range"}
    for state, _, _ in stats:
        mapping.setdefault(state, "chop_range")
    return mapping


def _rolling_zscore(series: np.ndarray, window: int = 100) -> np.ndarray:
    """Same rolling z-score shape RegimeDetector uses live, computed
    offline over the whole array for label construction."""
    z = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        start = max(0, i - window + 1)
        hist = series[start:i + 1]
        std = np.std(hist)
        z[i] = 0.0 if std < 1e-9 else (series[i] - np.mean(hist)) / std
    return z


def build_regime_labels(model, mapping: Dict[int, str], X: np.ndarray, panic_z_threshold: float) -> np.ndarray:
    """
    Build the offline training target for the LightGBM distillation
    classifier: smoothed HMM state (mapped to a label) with the panic
    override applied per bar. Smoothing is legitimate here because these
    are training labels, not live serving output (§7.6).
    """
    posteriors = model.predict_proba(X)  # smoothed, whole-sequence
    momentum_col = HMM_OBSERVATION_FEATURES.index("momentum_5m")
    volatility_col = HMM_OBSERVATION_FEATURES.index("volatility_5m")
    volume_col = HMM_OBSERVATION_FEATURES.index("volume_zscore")

    vol_z = _rolling_zscore(X[:, volatility_col])
    mom_z = _rolling_zscore(X[:, momentum_col])
    vol_col_z = X[:, volume_col]  # already a z-score by construction

    labels = []
    for i in range(len(X)):
        state = int(np.argmax(posteriors[i]))
        label = mapping.get(state, "chop_range")
        p_panic = panic_score(vol_z[i], mom_z[i], vol_col_z[i], z_threshold=panic_z_threshold)
        if p_panic >= 0.5:
            label = "panic_spike"
        labels.append(label)
    return np.array(labels)


def train_classifier(df: pd.DataFrame, labels: np.ndarray, splitter: WalkForwardSplitter):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm is required to train the regime classifier. Try `pip install lightgbm`."
        ) from e
    from sklearn.metrics import accuracy_score

    X = df[CLASSIFIER_FEATURE_NAMES].values
    y = labels

    def make_model():
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )

    fold_metrics = []
    n_splits = splitter.n_splits(len(df))
    if n_splits == 0:
        logger.warning("Not enough samples for a walk-forward fold; fitting on all data without CV.")
    else:
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(len(df))):
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])
            preds = model.classes_[np.argmax(proba, axis=1)]

            fold_metric = {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], preds)),
            }
            fold_metrics.append(fold_metric)
            logger.info(f"Fold {fold_i}: {fold_metric}")

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the regime HMM + LightGBM distillation classifier")
    parser.add_argument("--input", required=True, help="Path to a Parquet/CSV of historical OHLCV bars")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--hmm-output", default="data/models/regime_hmm.pkl")
    parser.add_argument("--classifier-output", default="data/models/regime_classifier.pkl")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    regime_cfg = config.get("regime", {})
    backtest_cfg = config.get("backtest", {})

    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    feat_df = build_feature_dataset(df_ohlcv, args.symbol, data_cfg.get("lookback_bars", 100))
    if len(feat_df) < 50:
        raise SystemExit(f"Only {len(feat_df)} usable bars after feature computation -- need more history to train")

    X_obs = feat_df[HMM_OBSERVATION_FEATURES].values

    logger.info("Running BIC grid search (diagnostic only; final model always uses n_states=3)...")
    fit_hmm_with_bic(X_obs, candidate_n_states=[2, 3, 4, 5])

    n_states = regime_cfg.get("n_states", 3)
    hmm_model = fit_final_hmm(X_obs, n_states=n_states)
    state_to_label = map_states_to_labels(hmm_model, X_obs)
    logger.info(f"State -> label mapping: {state_to_label}")

    hmm_payload = {
        "startprob": hmm_model.startprob_,
        "transmat": hmm_model.transmat_,
        "means": hmm_model.means_,
        "covars": hmm_model.covars_,
        "state_to_label": state_to_label,
        "feature_names": HMM_OBSERVATION_FEATURES,
        "n_components": hmm_model.n_components,
    }
    hmm_output_path = Path(args.hmm_output)
    hmm_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hmm_output_path, "wb") as f:
        pickle.dump(hmm_payload, f)

    hmm_model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(feat_df)),
        "n_states": n_states,
        "state_to_label": state_to_label,
        "observation_features": HMM_OBSERVATION_FEATURES,
        "log_likelihood": float(hmm_model.score(X_obs)),
    }
    with open(hmm_output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(hmm_model_card, f, indent=2)
    logger.info(f"Saved HMM to {hmm_output_path}")

    # ---- Distillation classifier ----
    panic_z_threshold = regime_cfg.get("panic_z_threshold", 3.0)
    labels = build_regime_labels(hmm_model, state_to_label, X_obs, panic_z_threshold)
    label_counts = pd.Series(labels).value_counts().to_dict()
    logger.info(f"Offline label distribution: {label_counts}")

    splitter = WalkForwardSplitter(
        train_bars=min(backtest_cfg.get("train_bars", 5000), max(1, len(feat_df) // 2)),
        test_bars=min(backtest_cfg.get("test_bars", 1000), max(1, len(feat_df) // 4)),
        step_bars=min(backtest_cfg.get("step_bars", 1000), max(1, len(feat_df) // 4)),
        embargo_bars=backtest_cfg.get("embargo_bars", 30),
    )
    classifier, fold_metrics = train_classifier(feat_df, labels, splitter)

    clf_payload = {"model": classifier, "feature_names": CLASSIFIER_FEATURE_NAMES}
    clf_output_path = Path(args.classifier_output)
    clf_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clf_output_path, "wb") as f:
        pickle.dump(clf_payload, f)

    clf_model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(feat_df)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "feature_names": CLASSIFIER_FEATURE_NAMES,
        "walk_forward_folds": fold_metrics,
    }
    with open(clf_output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(clf_model_card, f, indent=2)

    logger.info(f"Saved classifier to {clf_output_path}")


if __name__ == "__main__":
    main()
