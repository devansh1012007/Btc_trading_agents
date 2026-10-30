"""
Complete SystemState archive writer (Phase 4, §7.1/§7.4): the production
trade record and the entire future training archive. Partitioned by
symbol/date so adding a second trading pair later (§10) is a new
partition, not a schema change -- one Parquet file per (symbol, date).
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from ..state.contracts import SystemState
from ..state.serialization import system_state_to_dict

DEFAULT_BASE_DIR = "data/logs/system_state"
ARCHIVE_SCHEMA_VERSION = 2


def write_system_state_archive(states: List[SystemState], base_dir: str = DEFAULT_BASE_DIR) -> List[str]:
    """
    Append `states` to the archive, one file per (symbol, UTC date)
    partition. Returns the list of partition file paths touched.
    """
    if not states:
        return []

    by_partition = defaultdict(list)
    for state in states:
        dt = datetime.fromtimestamp(state.timestamp, tz=timezone.utc)
        partition_key = (state.symbol.replace("/", "_"), dt.strftime("%Y-%m-%d"))
        row = system_state_to_dict(state)
        row.setdefault("schema_version", ARCHIVE_SCHEMA_VERSION)
        by_partition[partition_key].append(row)

    touched = []
    for (symbol, date_str), records in by_partition.items():
        partition_dir = Path(base_dir) / symbol
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / f"{date_str}.parquet"

        new_df = pd.DataFrame(records)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(path, compression="snappy")
        touched.append(str(path))

    return touched


def load_system_state_archive(symbol: str, date_str: str, base_dir: str = DEFAULT_BASE_DIR) -> pd.DataFrame:
    path = Path(base_dir) / symbol.replace("/", "_") / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not df.empty:
        if "schema_version" not in df.columns:
            df["schema_version"] = 1
        if "regime_set_id" not in df.columns:
            df["regime_set_id"] = "legacy4"
        if "regime_probs_json" not in df.columns and "regime_prob_up" in df.columns:
            import json
            df["regime_probs_json"] = [
                json.dumps({
                    "trending_up": float(u or 0.0),
                    "trending_down": float(d or 0.0),
                    "chop_range": float(c or 0.0),
                    "panic_spike": float(p or 0.0),
                }, sort_keys=True)
                for u, d, c, p in zip(
                    df.get("regime_prob_up", [0.0] * len(df)),
                    df.get("regime_prob_down", [0.0] * len(df)),
                    df.get("regime_prob_chop", [0.0] * len(df)),
                    df.get("regime_prob_panic", [0.0] * len(df)),
                )
            ]
    return df
