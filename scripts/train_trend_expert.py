"""
Train the Trend Expert's scoring model (§6.6, M3.4). Reads the combined
triple-barrier-labeled dataset from train_triple_barrier.py, filters to
trend_expert candidates, trains a binary LightGBM classifier via
walk-forward CV, and saves a model card recording data_granularity for
the mismatch-fallback check (§4.7).
"""
import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.experts.trend_expert import FEATURE_NAMES
from src.backtest.walk_forward import WalkForwardSplitter


def load_config(path="config/settings.base.yaml") -> dict:
    p = Path(path)
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def train_and_validate(df: pd.DataFrame, splitter: WalkForwardSplitter):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    def make_model():
        return lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
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
                      "accuracy": float(accuracy_score(y[test_idx], preds))}
            fold_metrics.append(metric)
            logger.info(f"Fold {fold_i}: {metric}")

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/triple_barrier_labels.parquet")
    parser.add_argument("--output", default="data/models/experts/trend_expert.pkl")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backtest_cfg = config.get("backtest", {})
    data_granularity = config.get("system", {}).get("data_granularity", "1m")

    df_all = pd.read_parquet(args.input)
    df = df_all[df_all["expert_name"] == "trend_expert"].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No trend_expert candidates found -- run train_triple_barrier.py first")

    df = df.sort_values("bar_index").reset_index(drop=True)

    splitter = WalkForwardSplitter(
        train_bars=min(backtest_cfg.get("train_bars", 5000), max(1, len(df) // 2)),
        test_bars=min(backtest_cfg.get("test_bars", 500), max(1, len(df) // 4)),
        step_bars=min(backtest_cfg.get("step_bars", 100), max(1, len(df) // 4)),
        embargo_bars=backtest_cfg.get("embargo_bars", 50),
    )
    model, fold_metrics = train_and_validate(df, splitter)

    favorable = df[df["label"] == 1]["realized_return_pct"]
    unfavorable = df[df["label"] == 0]["realized_return_pct"]
    payload = {
        "model": model, "feature_names": FEATURE_NAMES, "granularity": data_granularity,
        "feature_version": "v2_renamed",
        "avg_favorable_return": float(favorable.mean()) if len(favorable) else 0.0,
        "avg_unfavorable_return": float(unfavorable.mean()) if len(unfavorable) else 0.0,
        "avg_adverse_excursion": float(df["max_adverse_excursion_pct"].mean()) if len(df) else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "granularity": data_granularity,
        "n_samples": int(len(df)), "n_favorable": int((df["label"] == 1).sum()),
        "n_unfavorable": int((df["label"] == 0).sum()),
        "avg_favorable_return": payload["avg_favorable_return"],
        "avg_unfavorable_return": payload["avg_unfavorable_return"],
        "avg_adverse_excursion": payload["avg_adverse_excursion"],
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Saved trend_expert model to {output_path}")


if __name__ == "__main__":
    main()
