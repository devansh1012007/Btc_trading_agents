"""
Train the Router's hindsight multiclass model (§8.1, §8.2, M3.6).

For each bar, reruns all three experts (rule-only replay), applies
triple-barrier labeling to each candidate, and asks "which expert's
choice would have been best?" (max net-of-fees realized return, or
no_trade if nothing beats 0). Switching context (bars_since_switch,
current_expert) is derived by walking the resulting HINDSIGHT label
sequence itself -- i.e. what switching would actually have looked like
had the router always picked the optimal expert -- not from the
rule-based fallback router's own (different) live decisions.
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
from src.layers.router import EXPERT_NAMES, ROUTER_CLASSES, FEATURE_NAMES, build_features
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

    pending: List[Dict] = []
    for i, state in enumerate(result.system_states):
        if state.expert_outputs is None or state.regime_state is None or state.opportunity_score is None:
            continue

        regime_key = state.regime_state.primary_regime.name
        overrides = regime_overrides.get(regime_key, {})
        k_tp = overrides.get("k_tp", k_tp_default)
        k_sl = overrides.get("k_sl", k_sl_default)
        volatility_pct = barrier_volatility_pct(state.feature_vector)
        take_profit_pct, stop_loss_pct = compute_barrier_pcts(volatility_pct, k_tp, k_sl)
        forward_bars = result.structure_snapshots[i + 1:]

        outcomes = {}
        if forward_bars and state.opportunity_score.tradable:
            for name in EXPERT_NAMES:
                output = state.expert_outputs.get_by_name(name)
                if output is None or output.action == ExpertAction.NO_TRADE:
                    continue
                direction = 1 if output.action == ExpertAction.BUY else -1
                outcome = label_triple_barrier(
                    forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
                    take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
                    max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
                )
                outcomes[name] = outcome.realized_return_pct

        if outcomes:
            best_name = max(outcomes, key=outcomes.get)
            optimal_expert = best_name if outcomes[best_name] > 0 else "no_trade"
        else:
            optimal_expert = "no_trade"

        pending.append({"state": state, "bar_index": i, "optimal_expert": optimal_expert})

    rows = []
    current_expert = None
    bars_since_switch = 0
    for item in pending:
        state = item["state"]
        optimal_expert = item["optimal_expert"]

        features = build_features(state.expert_outputs, state.regime_state, state.opportunity_score, bars_since_switch)
        rows.append({**features, "timestamp": state.timestamp, "bar_index": item["bar_index"], "label": optimal_expert})

        if optimal_expert != current_expert:
            current_expert = optimal_expert
            bars_since_switch = 0
        else:
            bars_since_switch += 1

    return pd.DataFrame(rows)


def train_and_validate(df, splitter):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    def make_model():
        return lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)

    fold_metrics = []
    if splitter.n_splits(len(df)) == 0:
        logger.warning("Not enough samples for a walk-forward fold; fitting on all data without CV.")
    else:
        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(len(df))):
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])
            preds = model.classes_[np.argmax(proba, axis=1)]
            metric = {"fold": fold_i, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                      "accuracy": float(accuracy_score(y[test_idx], preds))}
            fold_metrics.append(metric)
            logger.info(f"Fold {fold_i}: {metric}")

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/models/router.pkl")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backtest_cfg = config.get("backtest", {})

    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    df = harvest(df_ohlcv, args.symbol, config)
    label_dist = {str(k): int(v) for k, v in df["label"].value_counts().items()} if len(df) else {}
    logger.info(f"Harvested {len(df)} bars. Label distribution: {label_dist}")
    if len(df) == 0 or df["label"].nunique() < 2:
        raise SystemExit("Not enough label diversity to train the router -- need more/varied history")

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
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "feature_version": "v2_renamed"}, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "n_samples": int(len(df)),
        "label_distribution": label_dist, "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Saved router model to {output_path}")


if __name__ == "__main__":
    main()
