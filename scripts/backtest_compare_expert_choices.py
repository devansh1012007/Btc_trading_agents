"""
Debug/analysis script (§12): compares the Router's actual (live-serving)
choice against the hindsight-optimal expert choice, bar by bar. Useful for
spotting where the router systematically disagrees with hindsight.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.backtest.replay import replay_full_stack
from src.labeling.triple_barrier import label_triple_barrier, compute_barrier_pcts, barrier_volatility_pct
from src.layers.router import EXPERT_NAMES
from src.state.contracts import ExpertAction


def load_config(path="config/settings.base.yaml") -> dict:
    p = Path(path)
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/router_vs_hindsight.parquet")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    labeling_cfg = config.get("labeling", {})
    max_holding_bars = labeling_cfg.get("max_holding_bars", 5)
    fee_pct = labeling_cfg.get("fee_pct", 0.04)
    k_tp_default = labeling_cfg.get("k_tp", 1.5)
    k_sl_default = labeling_cfg.get("k_sl", 1.0)
    regime_overrides = labeling_cfg.get("regime_specific_barriers", {})

    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    result = replay_full_stack(df_ohlcv, args.symbol, config, run_experts=True)

    rows = []
    for i, state in enumerate(result.system_states):
        if state.router_decision is None or state.expert_outputs is None:
            continue
        regime_key = state.regime_state.primary_regime.name
        overrides = regime_overrides.get(regime_key, {})
        k_tp = overrides.get("k_tp", k_tp_default)
        k_sl = overrides.get("k_sl", k_sl_default)
        volatility_pct = barrier_volatility_pct(state.feature_vector)
        take_profit_pct, stop_loss_pct = compute_barrier_pcts(volatility_pct, k_tp, k_sl)
        forward_bars = result.structure_snapshots[i + 1:]

        outcomes = {}
        if forward_bars:
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

        hindsight_best = max(outcomes, key=outcomes.get) if outcomes else "no_trade"
        hindsight_optimal = hindsight_best if outcomes and outcomes[hindsight_best] > 0 else "no_trade"

        rows.append({
            "timestamp": state.timestamp, "bar_index": i,
            "router_choice": state.router_decision.selected_expert,
            "hindsight_optimal": hindsight_optimal,
            "agree": state.router_decision.selected_expert == hindsight_optimal,
            "trend_outcome": outcomes.get("trend_expert"),
            "mr_outcome": outcomes.get("mean_reversion_expert"),
            "breakout_outcome": outcomes.get("breakout_expert"),
        })

    df = pd.DataFrame(rows)
    agreement_rate = float(df["agree"].mean()) if len(df) else 0.0
    logger.info(f"Router/hindsight agreement rate: {agreement_rate:.2%} over {len(df)} bars")
    if len(df):
        logger.info(f"Router choice distribution:\n{df['router_choice'].value_counts().to_string()}")
        logger.info(f"Hindsight optimal distribution:\n{df['hindsight_optimal'].value_counts().to_string()}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, compression="snappy")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
