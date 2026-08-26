"""
Zone Event Log writer (Phase 4, §7.3): flushes a DynamicStructureEngine's
lifecycle_events (created/touched/rejected/broke/merged/decayed -- see
structure_engine.py) to data/logs/zone_events.parquet. Cheap: the engine
already generates every one of these events internally as a byproduct of
running; this just persists them instead of only holding them in memory
for the current process's lifetime.

Append semantics: Parquet has no true append, so this reads the existing
file (if present), concatenates the new rows, and rewrites -- fine at the
laptop scale this system targets (per Project_Plan.md's own stated
constraint), and consistent with how Phase 2/3 handle Parquet elsewhere
in this codebase.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_PATH = "data/logs/zone_events.parquet"


def flush_zone_events(events: List[Dict], path: str = DEFAULT_PATH) -> int:
    """Append `events` to the zone event log. Returns the total row count
    after the write. No-op (returns existing count) if events is empty."""
    if not events:
        existing = pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()
        return len(existing)

    new_df = pd.DataFrame(events)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if Path(path).exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(path, compression="snappy")
    return len(combined)


def load_zone_events(path: str = DEFAULT_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=[
            "zone_id", "event_type", "timestamp", "bar_index",
            "price_at_event", "zone_strength_at_event", "related_zone_id", "regime_at_event",
        ])
    return pd.read_parquet(path)
