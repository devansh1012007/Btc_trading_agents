"""
Phase 3 integration test (§11, last row: "extend test_integration_phase3.py").

Generates mock OHLCV bars, runs them through the full Phase 1-3 stack
(market_data -> features -> regime -> structure -> experts -> opportunity
filter -> router -> SystemState), serializes to Parquet, reloads, and
checks the round trip -- including the new Decision Layer columns.

Run directly: python test_integration_phase3.py
"""
import logging

import numpy as np
import pandas as pd

from src.backtest.replay import replay_full_stack
from src.state.serialization import expert_outputs_from_json, system_state_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_ohlcv(n_bars: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(n_bars)]

    price = 50000.0
    rows = []
    for ts in timestamps:
        drift = rng.normal(0, 15)
        price = max(price + drift, 100.0)
        high = price + abs(rng.normal(0, 8))
        low = price - abs(rng.normal(0, 8))
        open_ = price + rng.normal(0, 4)
        close = price
        volume = abs(rng.normal(100, 30))
        rows.append({
            "timestamp": ts, "open": open_, "high": max(high, open_, close),
            "low": min(low, open_, close), "close": close, "volume": volume,
        })
    return pd.DataFrame(rows)


def main():
    n_bars = 150
    symbol = "BTC/USDT"

    import yaml
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    df_ohlcv = generate_mock_ohlcv(n_bars=n_bars)
    print(f"Generated {len(df_ohlcv)} mock OHLCV bars")

    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=True)
    print(f"Computed full SystemState (through RouterDecision) for {len(result.system_states)} bars "
          f"({n_bars - len(result.system_states)} require history)")
    assert len(result.system_states) > 0, "No SystemStates produced"
    assert all(s.expert_outputs is not None for s in result.system_states)
    assert all(s.opportunity_score is not None for s in result.system_states)
    assert all(s.router_decision is not None for s in result.system_states)
    print("Every bar has expert_outputs, opportunity_score, and router_decision populated")

    # Serialize to Parquet
    records = [system_state_to_dict(s) for s in result.system_states]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    out_path = "data/processed/test_btc_phase3_integration.parquet"
    df.to_parquet(out_path, compression="snappy")
    print(f"Serialized to Parquet ({out_path})")

    df_reloaded = pd.read_parquet(out_path)
    assert len(df_reloaded) == len(df), "Row count mismatch after round trip"
    for col in ["expert_outputs_json", "opportunity_tradable", "router_selected_expert",
                "expert_best_name", "opportunity_edge_vs_fees"]:
        assert col in df_reloaded.columns, f"{col} column missing after round trip"
    print("Reloaded from Parquet successfully -- Decision Layer columns present")

    assert df_reloaded["timestamp"].is_unique, "Duplicate timestamps found"
    for col in ["market_close", "regime_confidence", "opportunity_tradable_probability", "router_selection_confidence"]:
        assert df_reloaded[col].notna().all(), f"Unexpected NaN in {col}"
    print("Data quality checks passed (unique timestamps, no critical NaNs)")

    # expert_outputs_json round trip: every bar should have exactly 3 experts
    n_bad = 0
    for _, row in df_reloaded.iterrows():
        parsed = expert_outputs_from_json(row["expert_outputs_json"])
        if set(parsed.keys()) != {"trend_expert", "mean_reversion_expert", "breakout_expert"}:
            n_bad += 1
    assert n_bad == 0, f"{n_bad} bars had an incomplete expert_outputs_json"
    print("expert_outputs_json round-trips with all 3 experts on every bar")

    print("\nRouter selection distribution:")
    print(df_reloaded["router_selected_expert"].value_counts())
    print("\nOpportunity tradable distribution:")
    print(df_reloaded["opportunity_tradable"].value_counts())

    n_tradable_bars = int(df_reloaded["opportunity_tradable"].sum())
    n_routed_trades = int((df_reloaded["router_selected_expert"] != "no_trade").sum())
    print(f"\n{n_tradable_bars} bars judged tradable by the opportunity filter; "
          f"{n_routed_trades} bars where the router actually picked an expert")

    print("\nPHASE 3 INTEGRATION TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
