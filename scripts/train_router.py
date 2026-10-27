"""
Train the Router's hindsight multiclass model (§8.1, §8.2, M3.6).

For each bar the Opportunity Filter passed, reruns all three experts
(rule-only replay), labels each candidate with the HARNESS's exit rule
(simulated-fill entry, regime-specific barriers, the minimum-trade-duration
lockout -- ``src/labeling/harness_labels.py``) and asks "which expert's
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb  # noqa: E402
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
from src.layers.router import EXPERT_NAMES, FEATURE_NAMES, build_features  # noqa: E402
from src.state.contracts import ExpertAction  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POPULATION = "opportunity_filter_passed_bars"


def harvest(df_ohlcv: pd.DataFrame, symbol: str, config: dict, ctx: LabelContext) -> pd.DataFrame:
    max_hold = ctx.policy.max_holding_bars
    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)

    pending: List[Dict] = []
    for i, state in enumerate(result.system_states):
        if state.expert_outputs is None or state.regime_state is None or state.opportunity_score is None:
            continue

        forward_bars = result.structure_snapshots[i + 1: i + 1 + max_hold]
        outcomes: Dict[str, float] = {}
        holds: Dict[str, float] = {}
        if forward_bars and state.opportunity_score.tradable:
            for name in EXPERT_NAMES:
                output = state.expert_outputs.get_by_name(name)
                if output is None or output.action == ExpertAction.NO_TRADE:
                    continue
                direction = 1 if output.action == ExpertAction.BUY else -1
                labeled = label_candidate(ctx, state, forward_bars, direction)
                if labeled is None:
                    continue                      # no legal outcome in the data
                outcomes[name] = labeled["realized_return_pct"]
                holds[name] = labeled["hold_seconds"]

        if outcomes:
            best_name = max(outcomes, key=outcomes.get)
            optimal_expert = best_name if outcomes[best_name] > 0 else "no_trade"
        else:
            optimal_expert = "no_trade"

        pending.append({
            "state": state, "bar_index": i, "optimal_expert": optimal_expert,
            "hold_seconds": holds.get(optimal_expert, float("nan")),
        })

    rows = []
    current_expert = None
    bars_since_switch = 0
    for item in pending:
        state = item["state"]
        optimal_expert = item["optimal_expert"]

        features = build_features(state.expert_outputs, state.regime_state, state.opportunity_score, bars_since_switch)
        rows.append({
            **features, "timestamp": state.timestamp, "bar_index": item["bar_index"],
            "label": optimal_expert, "hold_seconds": item["hold_seconds"],
            # Every router label looks up to max_holding_bars ahead (the
            # horizon the walk-forward purge must respect).
            "bars_held": max_hold,
        })

        if optimal_expert != current_expert:
            current_expert = optimal_expert
            bars_since_switch = 0
        else:
            bars_since_switch += 1

    return pd.DataFrame(rows)


def make_router_model():
    return lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/models/router.pkl")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default=None, help="explicit config file (overrides --env)")
    add_alignment_args(parser)
    args = parser.parse_args()

    config = load_training_config(args.config, args.env)
    data_granularity = config.get("system", {}).get("data_granularity", "1m")
    try:
        ctx = LabelContext.from_config(config)
    except HoldPolicyError as exc:
        raise SystemExit(f"Hold policy is not satisfiable: {exc}") from exc

    df = harvest(read_ohlcv(args.input), args.symbol, config, ctx)
    label_dist = {str(k): int(v) for k, v in df["label"].value_counts().items()} if len(df) else {}
    logger.info(f"Harvested {len(df)} bars. Label distribution: {label_dist}")
    if len(df) == 0 or df["label"].nunique() < 2:
        raise SystemExit("Not enough label diversity to train the router -- need more/varied history")

    df = df.sort_values("bar_index").reset_index(drop=True)
    assert_no_short_trades(df.dropna(subset=["hold_seconds"]), ctx.policy, "router")

    splitter = splitter_for(config, df, ctx.policy)
    model, fold_metrics = walk_forward_fit(df, FEATURE_NAMES, splitter, make_router_model, multiclass=True)

    alignment = bar_level_alignment(
        ctx.policy, df, population=POPULATION, require_filter_pass=True,
        n_candidates=int(len(df)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "model": model, "feature_names": FEATURE_NAMES, "feature_version": "v2_renamed",
            "granularity": data_granularity, "training_alignment": alignment,
        }, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "n_samples": int(len(df)),
        "granularity": data_granularity, "label_distribution": label_dist,
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
        "walk_forward_window_unit": "bars", "training_alignment": alignment,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    logger.info(f"Saved router model to {output_path}")


if __name__ == "__main__":
    main()
