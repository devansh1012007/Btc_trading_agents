"""
Harvest Trend Expert candidates via live gate replay and apply
triple-barrier labeling (§5, M3.3's "Trend expert: live simulation").

Also serves as the generic triple-barrier harvesting entry point: labels
EVERY expert's candidate (not just Trend's) using its own rule-only gate
in a single replay pass, output as one combined Parquet with an
`expert_name` column. train_trend_expert.py reads this directly;
train_mean_reversion_expert.py / train_breakout_expert.py use Phase 2's
touch-event stream instead (§6.3, §6.4).

Usage:
    python scripts/train_triple_barrier.py --input data/raw/BTC_USDT_mock.parquet
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from src.backtest.replay import replay_full_stack
from src.labeling.triple_barrier import barrier_volatility_pct, compute_barrier_pcts, label_triple_barrier
from src.state.contracts import ExpertAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERT_NAMES = ["trend_expert", "mean_reversion_expert", "breakout_expert"]

BUILD_FEATURES = None  # populated lazily to avoid a hard import cycle


def _build_features_map():
    from src.experts import breakout_expert as bo_mod
    from src.experts import mean_reversion_expert as mr_mod
    from src.experts import trend_expert as trend_mod
    return {
        "trend_expert": trend_mod.build_features,
        "mean_reversion_expert": mr_mod.build_features,
        "breakout_expert": bo_mod.build_features,
    }


def load_config(path: str = "config/settings.base.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def harvest_candidates(df_ohlcv: pd.DataFrame, symbol: str, config: dict) -> pd.DataFrame:
    labeling_cfg = config.get("labeling", {})
    max_holding_bars = labeling_cfg.get("max_holding_bars", 5)
    fee_pct = labeling_cfg.get("fee_pct", 0.04)
    k_tp_default = labeling_cfg.get("k_tp", 1.5)
    k_sl_default = labeling_cfg.get("k_sl", 1.0)
    regime_overrides = labeling_cfg.get("regime_specific_barriers", {})
    build_features_map = _build_features_map()

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
        if not forward_bars:
            continue

        for expert_name in EXPERT_NAMES:
            output = state.expert_outputs.get_by_name(expert_name)
            if output is None or output.action == ExpertAction.NO_TRADE:
                continue
            direction = 1 if output.action == ExpertAction.BUY else -1

            outcome = label_triple_barrier(
                forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
                take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
                max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
            )

            features = build_features_map[expert_name](state.feature_vector, state.zone_map, state.regime_state, direction)
            rows.append({
                **features, "expert_name": expert_name, "timestamp": state.timestamp, "bar_index": i,
                "direction": direction, "exit_reason": outcome.exit_reason, "bars_held": outcome.bars_held,
                "realized_return_pct": outcome.realized_return_pct,
                "max_adverse_excursion_pct": outcome.max_adverse_excursion_pct,
                "label": 1 if outcome.realized_return_pct > 0 else 0,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Harvest + triple-barrier-label all experts' candidates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/triple_barrier_labels.parquet")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    labeled = harvest_candidates(df_ohlcv, args.symbol, config)
    logger.info(f"Harvested {len(labeled)} candidates across {EXPERT_NAMES}")
    if len(labeled):
        logger.info(labeled.groupby("expert_name")["label"].agg(["count", "mean"]).to_string())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(args.output, compression="snappy")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
