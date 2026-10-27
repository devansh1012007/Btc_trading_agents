"""
Train the Opportunity Filter's ML tradability model (§7.2, M3.5).

Label = 1 ("this bar was worth trading") if the best proposing expert's
outcome at that bar was net favorable after fees; 0 otherwise (including
bars where no expert proposed anything at all). This is the
"best-expert-proxy" label the plan describes in §7.2/§8.1.

The outcome is the HARNESS's outcome: simulated-fill entry, regime-specific
barriers and the shared exit rule with the minimum-trade-duration lockout
(``src/labeling/harness_labels.py``). A bar whose only "winning" trade
would have exited inside the 10 s lockout is therefore NOT labeled
tradable on the strength of a move the harness could not have captured.
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
from src.layers.opportunity_filter import FEATURE_NAMES, build_features  # noqa: E402
from src.state.contracts import ExpertAction  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POPULATION = "all_bars_with_expert_outputs"


def harvest(df_ohlcv: pd.DataFrame, symbol: str, config: dict, ctx: LabelContext) -> pd.DataFrame:
    labeling_cfg = config.get("labeling", {})
    max_hold = ctx.policy.max_holding_bars
    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)

    rows: List[Dict] = []
    for i, state in enumerate(result.system_states):
        if state.expert_outputs is None or state.regime_state is None:
            continue

        forward_bars = result.structure_snapshots[i + 1: i + 1 + max_hold]

        best_return = 0.0
        best_hold = float("nan")
        if forward_bars:
            for output in state.expert_outputs.all():
                if output.action == ExpertAction.NO_TRADE:
                    continue
                direction = 1 if output.action == ExpertAction.BUY else -1
                labeled = label_candidate(ctx, state, forward_bars, direction)
                if labeled is None:
                    continue
                if labeled["realized_return_pct"] > best_return:
                    best_return = labeled["realized_return_pct"]
                    best_hold = labeled["hold_seconds"]

        best_output = state.expert_outputs.get_best_by_confidence()
        round_trip_cost = ctx.fee_pct * 2 + labeling_cfg.get("slippage_buffer_pct", 0.0)
        edge_vs_fees = (best_output.expected_return - round_trip_cost) if best_output else -round_trip_cost

        features = build_features(
            state.expert_outputs, state.regime_state, state.zone_map, state.market_snapshot, state.feature_vector, edge_vs_fees,
        )
        rows.append({**features, "timestamp": state.timestamp, "bar_index": i,
                     "best_realized_return_pct": best_return, "label": 1 if best_return > 0 else 0,
                     "hold_seconds": best_hold if best_return > 0 else float("nan"),
                     "bars_held": max_hold})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/models/opportunity_filter.pkl")
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
    logger.info(f"Harvested {len(df)} bars ({(df['label']==1).sum()} tradable, {(df['label']==0).sum()} not)")
    if len(df) == 0:
        raise SystemExit("No bars harvested -- check input data")

    df = df.sort_values("bar_index").reset_index(drop=True)
    assert_no_short_trades(df.dropna(subset=["hold_seconds"]), ctx.policy, "opportunity_filter")

    splitter = splitter_for(config, df, ctx.policy)
    model, fold_metrics = walk_forward_fit(df, FEATURE_NAMES, splitter)

    alignment = bar_level_alignment(
        ctx.policy, df, population=POPULATION, require_filter_pass=False, n_candidates=int(len(df)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "feature_version": "v2_renamed",
                     "granularity": data_granularity, "training_alignment": alignment}, f)

    model_card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "n_samples": int(len(df)),
        "granularity": data_granularity,
        "n_tradable": int((df["label"] == 1).sum()), "n_not_tradable": int((df["label"] == 0).sum()),
        "feature_names": FEATURE_NAMES, "walk_forward_folds": fold_metrics,
        "walk_forward_window_unit": "bars", "training_alignment": alignment,
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    logger.info(f"Saved opportunity_filter model to {output_path}")


if __name__ == "__main__":
    main()
