"""
Train the Risk Layer's ML component (§5.4, M4.2): predicts P(stop-out)
for the Router's selected candidate. Reuses the triple-barrier
infrastructure -- specifically max_adverse_excursion_pct.

Label = 1 if the selected candidate's outcome resolved via stop_loss (a
bad, risk-relevant outcome), else 0 (take_profit or timeout).

The outcome is the HARNESS's outcome (``src/labeling/harness_labels.py``):
simulated-fill entry, regime-specific barriers, and the shared exit rule
with the minimum-trade-duration lockout. A stop that the lockout defers --
or that fills through a gap on the first legal bar -- is labeled the way
the harness would have experienced it.
"""
import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.replay import replay_full_stack  # noqa: E402
from src.backtest.training_common import (  # noqa: E402
    add_alignment_args,
    assert_no_short_trades,
    bar_level_alignment,
    load_training_config,
    read_ohlcv,
    splitter_for,
    walk_forward_fit,
)
from src.labeling.harness_labels import LabelContext, label_candidate  # noqa: E402
from src.labeling.hold_policy import HoldPolicyError  # noqa: E402
from src.layers.risk_layer import FEATURE_NAMES, build_features  # noqa: E402
from src.state.contracts import ExpertAction  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POPULATION = "router_selected_bars"


def harvest(df_ohlcv: pd.DataFrame, symbol: str, config: dict, ctx: LabelContext) -> pd.DataFrame:
    max_hold = ctx.policy.max_holding_bars
    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)

    rows: List[Dict] = []
    for i, state in enumerate(result.system_states):
        if state.router_decision is None or state.router_decision.selected_expert == "no_trade":
            continue
        expert_output = state.expert_outputs.get_by_name(state.router_decision.selected_expert)
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            continue

        forward_bars = result.structure_snapshots[i + 1: i + 1 + max_hold]
        direction = 1 if expert_output.action == ExpertAction.BUY else -1
        labeled = label_candidate(ctx, state, forward_bars, direction)
        if labeled is None:
            continue

        features = build_features(expert_output, state.router_decision, state.regime_state,
                                   state.feature_vector, state.market_snapshot)
        rows.append({
            **features, "timestamp": state.timestamp, "bar_index": i,
            "exit_reason": labeled["exit_reason"],
            "max_adverse_excursion_pct": labeled["max_adverse_excursion_pct"],
            "bars_held": labeled["bars_held"], "hold_seconds": labeled["hold_seconds"],
            "label": 1 if labeled["exit_reason"] == "stop_loss" else 0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/models/risk_model.pkl")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default=None, help="explicit config file (overrides --env)")
    parser.add_argument("--register", action="store_true", help="Also register this version in the model registry as pending_approval")
    add_alignment_args(parser)
    args = parser.parse_args()

    config = load_training_config(args.config, args.env)
    data_granularity = config.get("system", {}).get("data_granularity", "1m")
    try:
        ctx = LabelContext.from_config(config)
    except HoldPolicyError as exc:
        raise SystemExit(f"Hold policy is not satisfiable: {exc}") from exc

    df = harvest(read_ohlcv(args.input), args.symbol, config, ctx)
    logger.info(f"Harvested {len(df)} risk-model candidates ({(df['label']==1).sum() if len(df) else 0} stop-outs)")
    if len(df) == 0 or df["label"].nunique() < 2:
        raise SystemExit("Not enough label diversity to train the risk model -- need more/varied history")

    df = df.sort_values("bar_index").reset_index(drop=True)
    assert_no_short_trades(df, ctx.policy, "risk_model")

    splitter = splitter_for(config, df, ctx.policy)
    model, fold_metrics = walk_forward_fit(df, FEATURE_NAMES, splitter)

    alignment = bar_level_alignment(
        ctx.policy, df, population=POPULATION, require_filter_pass=True, n_candidates=int(len(df)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_names": FEATURE_NAMES, "granularity": data_granularity,
               "feature_version": "v2_renamed", "training_alignment": alignment}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "granularity": data_granularity,
        "n_samples": int(len(df)), "n_stop_outs": int((df["label"] == 1).sum()),
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
        "walk_forward_window_unit": "bars", "training_alignment": alignment,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2, default=str)
    logger.info(f"Saved risk model to {output_path}")

    if args.register:
        from src.registry.model_registry import register_version
        avg_auc = np.mean([m["auc"] for m in fold_metrics if m.get("auc") is not None]) if fold_metrics else None
        version_id = register_version(
            component="risk_model", file_path=str(output_path), data_granularity=data_granularity,
            walk_forward_metrics={"avg_auc": float(avg_auc) if avg_auc is not None else None,
                                  "n_samples": int(len(df)), "training_alignment": alignment},
        )
        logger.info(f"Registered as pending_approval: {version_id}")


if __name__ == "__main__":
    main()
