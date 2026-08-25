"""
Build the zone-survival training set (Phase 2, §6.5). Replays historical
bars through FeatureEngine -> RegimeDetector -> DynamicStructureEngine,
collects touch events, and labels each via forward-looking horizon.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.layers.feature_engine import FeatureEngine
from src.layers.market_data import MarketDataLayer
from src.layers.regime_detector import RegimeDetector
from src.layers.structure_engine import DynamicStructureEngine
from src.layers.zone_survival_features import ZONE_SURVIVAL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/settings.base.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def replay_and_collect_touch_events(df_ohlcv: pd.DataFrame, symbol: str, config: dict) -> List[Dict]:
    struct_cfg = config.get("structure", {})
    regime_cfg = config.get("regime", {})
    models_cfg = config.get("models", {})
    data_cfg = config.get("data", {})
    data_granularity = config.get("system", {}).get("data_granularity", "1m")

    feature_engine = FeatureEngine(lookback_bars=data_cfg.get("lookback_bars", 100))
    regime_detector = RegimeDetector(
        hmm_model_path=models_cfg.get("regime_hmm", "data/models/regime_hmm.pkl"),
        classifier_model_path=models_cfg.get("regime_classifier", "data/models/regime_classifier.pkl"),
        confidence_floor=regime_cfg.get("confidence_floor", 0.35),
        persistence_window_bars=regime_cfg.get("persistence_window_bars", 20),
        persistence_transition_threshold=regime_cfg.get("persistence_transition_threshold", 0.6),
        panic_z_threshold=regime_cfg.get("panic_z_threshold", 3.0),
        data_granularity=data_granularity,
    )
    structure_engine = DynamicStructureEngine(
        lookback_bars=struct_cfg.get("lookback_bars", 50),
        pivot_window=struct_cfg.get("pivot_window", 5),
        min_zone_separation_pct=struct_cfg.get("min_zone_separation_pct", 1.0),
        zone_decay_bars=struct_cfg.get("zone_decay_bars", 100),
        breakout_survival_penalty=struct_cfg.get("breakout_survival_penalty", 0.3),
        data_granularity=data_granularity,
    )

    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)

    for snapshot in snapshots:
        feature_engine.add_bar(snapshot)
        fv = feature_engine.compute_features(snapshot)
        if fv is None:
            continue
        regime_detector.add_bar(fv)
        regime_state = regime_detector.compute_regime(fv, symbol=symbol)
        structure_engine.add_bar(snapshot)
        structure_engine.compute_zones(current_price=snapshot.close, symbol=symbol, regime_state=regime_state)

    logger.info(f"Replayed {len(snapshots)} bars, collected {len(structure_engine.touch_events)} touch events")
    return structure_engine.touch_events


def label_touch_events(touch_events: List[Dict], horizon_bars: int) -> pd.DataFrame:
    events_by_zone: Dict[str, List[int]] = {}
    for idx, ev in enumerate(touch_events):
        events_by_zone.setdefault(ev["zone_id"], []).append(idx)

    rows = []
    for idx, ev in enumerate(touch_events):
        zone_id = ev["zone_id"]
        same_zone_indices = events_by_zone[zone_id]
        pos = same_zone_indices.index(idx)

        label: Optional[int] = None
        if pos + 1 < len(same_zone_indices):
            next_idx = same_zone_indices[pos + 1]
            next_event = touch_events[next_idx]
            bars_ahead = next_event["bar_index"] - ev["bar_index"]
            if bars_ahead <= horizon_bars:
                label = 1 if next_event["event_type"] == "rejection" else 0

        if label is None:
            continue

        row = dict(ev["features"])
        row["zone_id"] = zone_id
        row["bar_index"] = ev["bar_index"]
        row["timestamp"] = ev["timestamp"]
        row["event_type"] = ev["event_type"]
        row["label"] = label
        rows.append(row)

    columns = ["zone_id", "bar_index", "timestamp", "event_type"] + ZONE_SURVIVAL_FEATURE_NAMES + ["label"]
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Label zone touch events for the zone-survival model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/zone_labels.parquet")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/settings.base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    horizon_bars = config.get("structure", {}).get("zone_label_horizon_bars", 20)

    input_path = Path(args.input)
    df_ohlcv = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    if "timestamp" in df_ohlcv.columns:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.sort_values("timestamp").reset_index(drop=True)

    touch_events = replay_and_collect_touch_events(df_ohlcv, args.symbol, config)
    labeled_df = label_touch_events(touch_events, horizon_bars)

    n_rejections = int((labeled_df['label'] == 1).sum()) if len(labeled_df) else 0
    n_breakouts = int((labeled_df['label'] == 0).sum()) if len(labeled_df) else 0
    logger.info(f"Labeled {len(labeled_df)} samples ({n_rejections} rejections, {n_breakouts} breakouts), "
                f"{len(touch_events) - len(labeled_df)} censored/dropped")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(args.output, compression="snappy")
    logger.info(f"Saved labeled dataset to {args.output}")


if __name__ == "__main__":
    main()
