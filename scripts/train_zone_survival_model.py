"""
Train the zone-survival LightGBM classifier (§6.5).

Binary classifier: predicts P(zone survives / next resolution is a
rejection) from the features in zone_survival_features.py, using
walk-forward CV (no random shuffling -- touch-event order is
autocorrelated and forward-looking by construction, see §5.2).

Usage:
    python scripts/train_zone_survival_model.py --input data/processed/zone_labels.parquet
"""
import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.layers.zone_survival_features import ZONE_SURVIVAL_FEATURE_NAMES
from src.backtest.walk_forward import WalkForwardSplitter


def load_config(path: str = "config/settings.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def train_and_validate(df: pd.DataFrame, splitter: WalkForwardSplitter):
    """Fit walk-forward folds for diagnostic metrics, then fit a final
    model on all available data."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm is required to train the zone-survival model. "
            "Try `pip install lightgbm` -- a prebuilt wheel is available on most platforms."
        ) from e
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    X = df[ZONE_SURVIVAL_FEATURE_NAMES].values
    y = df["label"].values

    def make_model():
        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    fold_metrics = []
    n_splits = splitter.n_splits(len(df))
    if n_splits == 0:
        logger.warning(
            "Not enough samples for even one walk-forward fold with the configured "
            "train/test/embargo sizes -- skipping CV and fitting on all data. Treat "
            "this model as unvalidated until more labeled data is available."
        )
    else:
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(len(df))):
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            preds = (proba >= 0.5).astype(int)

            fold_metric = {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "auc": float(roc_auc_score(y[test_idx], proba)) if len(set(y[test_idx])) > 1 else None,
                "accuracy": float(accuracy_score(y[test_idx], preds)),
                "brier": float(brier_score_loss(y[test_idx], proba)),
            }
            fold_metrics.append(fold_metric)
            logger.info(f"Fold {fold_i}: {fold_metric}")

    final_model = make_model()
    final_model.fit(X, y)

    return final_model, fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the zone-survival LightGBM model")
    parser.add_argument("--input", default="data/processed/zone_labels.parquet")
    parser.add_argument("--output", default="data/models/zone_validity_model.pkl")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backtest_cfg = config.get("backtest", {})

    df = pd.read_parquet(args.input)
    if len(df) == 0:
        raise SystemExit(f"No labeled samples found in {args.input} -- run label_zone_events.py first")

    df = df.sort_values("bar_index").reset_index(drop=True)

    splitter = WalkForwardSplitter(
        train_bars=min(backtest_cfg.get("train_bars", 5000), max(1, len(df) // 2)),
        test_bars=min(backtest_cfg.get("test_bars", 1000), max(1, len(df) // 4)),
        step_bars=min(backtest_cfg.get("step_bars", 1000), max(1, len(df) // 4)),
        embargo_bars=backtest_cfg.get("embargo_bars", 30),
    )

    model, fold_metrics = train_and_validate(df, splitter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": ZONE_SURVIVAL_FEATURE_NAMES,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_rejections": int((df["label"] == 1).sum()),
        "n_breakouts": int((df["label"] == 0).sum()),
        "feature_names": ZONE_SURVIVAL_FEATURE_NAMES,
        "walk_forward_folds": fold_metrics,
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Saved model to {output_path}")
    logger.info(f"Saved model card to {meta_path}")


if __name__ == "__main__":
    main()
