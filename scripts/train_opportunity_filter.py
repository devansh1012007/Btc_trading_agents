"""
Train the Opportunity Filter's ML tradability model (§7.2, M3.5).

Label = 1 ("this bar was worth trading") if the best proposing expert's
triple-barrier outcome at that bar was net favorable after fees; 0
otherwise (including bars where no expert proposed anything at all). This
is the "best-expert-proxy" label the plan describes in §7.2/§8.1.
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
from src.layers.opportunity_filter import FEATURE_NAMES, build_features
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
        if state.expert_outputs is None or state.regime_state is None:
            continue

        regime_key = state.regime_state.primary_regime.name
        overrides = regime_overrides.get(regime_key, {})
        k_tp = overrides.get("k_tp", k_tp_default)
        k_sl = overrides.get("k_sl", k_sl_default)
        volatility_pct = barrier_volatility_pct(state.feature_vector)
        take_profit_pct, stop_loss_pct = compute_barrier_pcts(volatility_pct, k_tp, k_sl)
        forward_bars = result.structure_snapshots[i + 1:]

        best_return = 0.0
        if forward_bars:
            for output in state.expert_outputs.all():
                if output.action == ExpertAction.NO_TRADE:
                    continue
                direction = 1 if output.action == ExpertAction.BUY else -1
                outcome = label_triple_barrier(
                    forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
                    take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
                    max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
                )
                best_return = max(best_return, outcome.realized_return_pct)

        best_output = state.expert_outputs.get_best_by_confidence()
        round_trip_cost = fee_pct * 2 + labeling_cfg.get("slippage_buffer_pct", 0.0)
        edge_vs_fees = (best_output.expected_return - round_trip_cost) if best_output else -round_trip_cost

        features = build_features(
            state.expert_outputs, state.regime_state, state.zone_map, state.market_snapshot, state.feature_vector, edge_vs_fees,
        )
        rows.append({**features, "timestamp": state.timestamp, "bar_index": i,
                     "best_realized_return_pct": best_return, "label": 1 if best_return > 0 else 0})

    return pd.DataFrame(rows)


def train_and_validate(df, splitter):
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
    parser.add_argument("--output", default="data/models/opportunity_filter.pkl")
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
    logger.info(f"Harvested {len(df)} bars ({(df['label']==1).sum()} tradable, {(df['label']==0).sum()} not)")
    if len(df) == 0:
        raise SystemExit("No bars harvested -- check input data")

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
        "n_tradable": int((df["label"] == 1).sum()), "n_not_tradable": int((df["label"] == 0).sum()),
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Saved opportunity_filter model to {output_path}")


if __name__ == "__main__":
    main()
