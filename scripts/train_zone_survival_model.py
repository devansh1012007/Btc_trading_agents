"""
Train the zone-survival LightGBM classifier (Phase 2, §6.5), walk-forward CV.
"""
import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.backtest.walk_forward import WalkForwardSplitter
from src.layers.zone_survival_features import ZONE_SURVIVAL_FEATURE_NAMES


def load_config(path: str = "config/settings.base.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def train_and_validate(df: pd.DataFrame, splitter: WalkForwardSplitter):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    X = df[ZONE_SURVIVAL_FEATURE_NAMES].values
    y = df["label"].values

    def make_model():
        return lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)

    fold_metrics = []
    if splitter.n_splits(len(df)) == 0:
        logger.warning("Not enough samples for a walk-forward fold; fitting on all data without CV.")
    else:
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(len(df))):
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            preds = (proba >= 0.5).astype(int)
            metric = {"fold": fold_i, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                      "auc": float(roc_auc_score(y[test_idx], proba)) if len(set(y[test_idx])) > 1 else None,
                      "accuracy": float(accuracy_score(y[test_idx], preds)),
                      "brier": float(brier_score_loss(y[test_idx], proba))}
            fold_metrics.append(metric)
            logger.info(f"Fold {fold_i}: {metric}")

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the zone-survival LightGBM model")
    parser.add_argument("--input", default="data/processed/zone_labels.parquet")
    parser.add_argument("--output", default="data/models/zone_validity_model.pkl")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backtest_cfg = config.get("backtest", {})
    data_granularity = config.get("system", {}).get("data_granularity", "1m")

    df = pd.read_parquet(args.input)
    if len(df) == 0:
        raise SystemExit(f"No labeled samples found in {args.input} -- run label_zone_events.py first")

    df = df.sort_values("bar_index").reset_index(drop=True)

    splitter = WalkForwardSplitter(
        train_bars=min(backtest_cfg.get("train_bars", 5000), max(1, len(df) // 2)),
        test_bars=min(backtest_cfg.get("test_bars", 500), max(1, len(df) // 4)),
        step_bars=min(backtest_cfg.get("step_bars", 100), max(1, len(df) // 4)),
        embargo_bars=backtest_cfg.get("embargo_bars", 50),
    )

    model, fold_metrics = train_and_validate(df, splitter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_names": ZONE_SURVIVAL_FEATURE_NAMES, "granularity": data_granularity,
               "feature_version": "v2_renamed"}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "granularity": data_granularity,
        "n_samples": int(len(df)), "n_rejections": int((df["label"] == 1).sum()),
        "n_breakouts": int((df["label"] == 0).sum()), "feature_names": ZONE_SURVIVAL_FEATURE_NAMES,
        "walk_forward_folds": fold_metrics,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
