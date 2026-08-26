"""
Phase 2 integration test (§10, last row: "extend test_integration.py").

Phase 1's test_integration.py wasn't included in the uploaded files, so
this is a self-contained sibling script that exercises the same
end-to-end contract, extended through the full Phase 2 stack: generates
mock OHLCV bars, runs them through market_data -> features -> regime ->
structure -> SystemState, serializes to Parquet, reloads, and checks the
round trip.

Run directly: python test_integration_phase2.py
"""
import logging

import numpy as np
import pandas as pd

from src.layers.feature_engine import FeatureEngine
from src.layers.market_data import MarketDataLayer
from src.layers.regime_detector import RegimeDetector
from src.layers.structural_features import attach_structural_features
from src.layers.structure_engine import DynamicStructureEngine
from src.state.contracts import SystemState
from src.state.serialization import system_state_to_dict, zone_map_from_json

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

    df_ohlcv = generate_mock_ohlcv(n_bars=n_bars)
    print(f"Generated {len(df_ohlcv)} mock OHLCV bars")

    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
    print(f"Created {len(snapshots)} MarketSnapshots")

    feature_engine = FeatureEngine(lookback_bars=100)
    regime_detector = RegimeDetector(
        hmm_model_path="data/models/does_not_exist.pkl",
        classifier_model_path="data/models/does_not_exist2.pkl",
    )
    structure_engine = DynamicStructureEngine(lookback_bars=50, pivot_window=5, zone_decay_bars=100)

    system_states = []
    for snapshot in snapshots:
        feature_engine.add_bar(snapshot)
        fv = feature_engine.compute_features(snapshot)
        if fv is None:
            continue

        regime_detector.add_bar(fv)
        regime_state = regime_detector.compute_regime(fv, symbol=symbol)

        structure_engine.add_bar(snapshot)
        zone_map = structure_engine.compute_zones(
            current_price=snapshot.close, symbol=symbol, regime_state=regime_state,
        )

        fv = attach_structural_features(fv, zone_map)

        state = SystemState(
            timestamp=snapshot.timestamp, symbol=symbol,
            market_snapshot=snapshot, feature_vector=fv,
            zone_map=zone_map, regime_state=regime_state,
        )
        system_states.append(state)

    print(f"Computed full SystemState for {len(system_states)} bars ({n_bars - len(system_states)} require history)")
    assert len(system_states) > 0, "No SystemStates produced -- check history requirements"

    # Serialize to Parquet
    records = [system_state_to_dict(s) for s in system_states]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    out_path = "data/processed/test_btc_phase2_integration.parquet"
    df.to_parquet(out_path, compression="snappy")
    print(f"Serialized to Parquet ({out_path})")

    # Reload and validate round trip
    df_reloaded = pd.read_parquet(out_path)
    assert len(df_reloaded) == len(df), "Row count mismatch after round trip"
    assert "zone_map_json" in df_reloaded.columns, "zone_map_json column missing"
    assert "regime_primary" in df_reloaded.columns, "regime_primary column missing"
    assert "feature_distance_to_nearest_support" in df_reloaded.columns
    print("Reloaded from Parquet successfully -- row count matches")

    # Data quality checks
    assert df_reloaded["timestamp"].is_unique, "Duplicate timestamps found"
    critical_cols = ["market_close", "regime_confidence", "regime_prob_up"]
    for col in critical_cols:
        assert df_reloaded[col].notna().all(), f"Unexpected NaN in {col}"
    print("Data quality checks passed (unique timestamps, no critical NaNs)")

    # Zone map JSON round trip
    n_bars_with_zones = 0
    for _, row in df_reloaded.iterrows():
        zones = zone_map_from_json(row["zone_map_json"])
        if zones:
            n_bars_with_zones += 1
    print(f"Zones present on {n_bars_with_zones}/{len(df_reloaded)} bars")

    # Regime distribution summary
    print("\nRegime distribution:")
    print(df_reloaded["regime_primary"].value_counts())

    print("\nSample features (last bar):")
    last = df_reloaded.iloc[-1]
    print(f"  momentum_5m:  {last['feature_momentum_5m']:.4f}")
    print(f"  regime:       {last['regime_primary']} (confidence={last['regime_confidence']:.2f})")
    print(f"  active zones: {last['zone_num_active']}")

    print("\nPHASE 2 INTEGRATION TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
