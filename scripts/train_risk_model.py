"""
Train the Risk Layer's ML component (§5.4, M4.2): predicts P(stop-out)
for the Router's selected candidate. Reuses Phase 3's triple-barrier
infrastructure a fourth time -- specifically max_adverse_excursion_pct,
already computed for every historical candidate and, until now, without
a consumer.

Label = 1 if the selected candidate's triple-barrier outcome resolved via
stop_loss (a bad, risk-relevant outcome), else 0 (take_profit or timeout).
"""
import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.backtest.replay import replay_full_stack
from src.labeling.triple_barrier import label_triple_barrier, compute_barrier_pcts, barrier_volatility_pct
from src.layers.risk_layer import FEATURE_NAMES, build_features
from src.state.contracts import ExpertAction
from src.backtest.walk_forward import WalkForwardSplitter


def load_config(path="config/settings.base.yaml") -> dict:
    p = Path(path)
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def harvest(df_ohlcv: pd.DataFrame, symbol: str, config: dict) -> pd.DataFrame:
    labeling_cfg = config.get("labeling", {})
    max_holding_bars = labeling_cfg.get("max_holding_bars", 5)
    fee_pct = labeling_cfg.get("fee_pct", 0.04)
    k_tp_default = labeling_cfg.get("k_tp", 1.5)
    k_sl_default = labeling_cfg.get("k_sl", 1.0)
    regime_overrides = labeling_cfg.get("regime_specific_barriers", {})

    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)

    rows: List[Dict] = []
    for i, state in enumerate(result.system_states):
        if state.router_decision is None or state.router_decision.selected_expert == "no_trade":
            continue
        expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            continue

        regime_key = state.regime_state.primary_regime.name
        overrides = regime_overrides.get(regime_key, {})
        k_tp = overrides.get("k_tp", k_tp_default)
        k_sl = overrides.get("k_sl", k_sl_default)
        volatility_pct = barrier_volatility_pct(state.feature_vector)
        take_profit_pct, stop_loss_pct = compute_barrier_pcts(volatility_pct, k_tp, k_sl)

        forward_bars = result.structure_snapshots[i + 1:]
        if not forward_bars:
            continue

        direction = 1 if expert_output.action == ExpertAction.BUY else -1
        outcome = label_triple_barrier(
            forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
            take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
        )

        features = build_features(expert_output, state.router_decision, state.regime_state,
                                   state.feature_vector, state.market_snapshot)
        rows.append({
            **features, "timestamp": state.timestamp, "bar_index": i,
            "exit_reason": outcome.exit_reason, "max_adverse_excursion_pct": outcome.max_adverse_excursion_pct,
            "label": 1 if outcome.exit_reason == "stop_loss" else 0,
        })

    return pd.DataFrame(rows)


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/models/risk_model.pkl")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.base.yaml")
    parser.add_argument("--register", action="store_true", help="Also register this version in the model registry as pending_approval")
    args = parser.parse_args()

    config = load_config(args.config)
    backtest_cfg = config.get("backtest", {})
    data_granularity = config.get("system", {}).get("data_granularity", "1m")

    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    df = harvest(df_ohlcv, args.symbol, config)
    logger.info(f"Harvested {len(df)} risk-model candidates ({(df['label']==1).sum() if len(df) else 0} stop-outs)")
    if len(df) == 0 or df["label"].nunique() < 2:
        raise SystemExit("Not enough label diversity to train the risk model -- need more/varied history")

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
    payload = {"model": model, "feature_names": FEATURE_NAMES, "granularity": data_granularity,
               "feature_version": "v2_renamed"}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "granularity": data_granularity,
        "n_samples": int(len(df)), "n_stop_outs": int((df["label"] == 1).sum()),
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2)
    logger.info(f"Saved risk model to {output_path}")

    if args.register:
        from src.registry.model_registry import register_version
        avg_auc = np.mean([m["auc"] for m in fold_metrics if m.get("auc") is not None]) if fold_metrics else None
        version_id = register_version(
            component="risk_model", file_path=str(output_path), data_granularity=data_granularity,
            walk_forward_metrics={"avg_auc": float(avg_auc) if avg_auc is not None else None, "n_samples": int(len(df))},
        )
        logger.info(f"Registered as pending_approval: {version_id}")


if __name__ == "__main__":
    main()
