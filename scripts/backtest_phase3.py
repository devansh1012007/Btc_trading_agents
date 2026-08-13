"""
Phase 3 backtest harness (§10, §12): replays historical data through the
full trained-or-fallback pipeline, simulates each RouterDecision as an
actual triple-barrier trade, and reports aggregate metrics. Uses whatever
models are currently on disk, falling back to rule-based per-layer where
they're missing -- not a training script.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.backtest.replay import replay_full_stack
from src.labeling.triple_barrier import label_triple_barrier, compute_barrier_pcts, barrier_volatility_pct


def load_config(path="config/settings.yaml") -> dict:
    p = Path(path)
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def run_backtest(df_ohlcv: pd.DataFrame, symbol: str, config: dict) -> pd.DataFrame:
    labeling_cfg = config.get("labeling", {})
    max_holding_bars = labeling_cfg.get("max_holding_bars", 5)
    fee_pct = labeling_cfg.get("fee_pct", 0.04)
    k_tp_default = labeling_cfg.get("k_tp", 1.5)
    k_sl_default = labeling_cfg.get("k_sl", 1.0)
    regime_overrides = labeling_cfg.get("regime_specific_barriers", {})

    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)

    trades = []
    for i, state in enumerate(result.system_states):
        decision = state.router_decision
        if decision is None or decision.selected_expert == "no_trade":
            continue

        output = state.expert_outputs.get_by_name(decision.selected_expert)
        if output is None or output.action.name == "NO_TRADE":
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

        direction = 1 if output.action.name == "BUY" else -1
        outcome = label_triple_barrier(
            forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
            take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
        )

        trades.append({
            "timestamp": state.timestamp, "bar_index": i, "selected_expert": decision.selected_expert,
            "direction": direction, "exit_reason": outcome.exit_reason, "bars_held": outcome.bars_held,
            "realized_return_pct": outcome.realized_return_pct,
            "max_adverse_excursion_pct": outcome.max_adverse_excursion_pct, "regime": regime_key,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"n_trades": 0}

    returns = trades["realized_return_pct"].values
    win_rate = float((returns > 0).mean())
    total_return_pct = float(returns.sum())
    avg_return_pct = float(returns.mean())
    std_return_pct = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe_like = float(avg_return_pct / std_return_pct) if std_return_pct > 1e-9 else 0.0

    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative) if len(cumulative) else np.array([0.0])
    drawdown = cumulative - running_max
    max_drawdown_pct = float(drawdown.min()) if len(drawdown) else 0.0

    by_expert = trades.groupby("selected_expert")["realized_return_pct"].agg(["count", "mean", "sum"]).to_dict("index")
    by_regime = trades.groupby("regime")["realized_return_pct"].agg(["count", "mean", "sum"]).to_dict("index")

    return {
        "n_trades": int(len(trades)), "win_rate": win_rate, "total_return_pct": total_return_pct,
        "avg_return_pct": avg_return_pct, "std_return_pct": std_return_pct,
        "sharpe_like_ratio": sharpe_like, "max_drawdown_pct": max_drawdown_pct,
        "by_expert": by_expert, "by_regime": by_regime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/backtest_results.parquet")
    parser.add_argument("--metrics-output", default="data/processed/backtest_metrics.json")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    trades = run_backtest(df_ohlcv, args.symbol, config)
    metrics = compute_metrics(trades)

    summary = {k: v for k, v in metrics.items() if k not in ("by_expert", "by_regime")}
    logger.info(f"Backtest complete: {json.dumps(summary, indent=2, default=str)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(args.output, compression="snappy")

    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Saved trades to {args.output}, metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
