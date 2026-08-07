"""
Serialize/deserialize SystemState to/from Parquet, JSON.

Phase 2 addition (§9): ZoneMap.zones is a nested List[Zone] that doesn't
flatten into scalar columns the way MarketSnapshot/FeatureVector/RegimeState
do. Following the plan's recommendation, this uses option (A) -- a single
JSON-blob column (zone_map_json) on the main features Parquet file, rather
than a second normalized zones table (option B). Keeps one file per run,
matching Phase 1's stated simplicity preference; costs the ability to
query/filter zones directly in Pandas/DuckDB without a parse step (use
zone_map_from_json() for that).
"""
import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState, Zone, ZoneMap


def _zone_to_dict(zone: Zone) -> Dict[str, Any]:
    # Cast every value to a native Python type: fields derived from numpy
    # arrays upstream (e.g. created_at, sourced from a numpy timestamps
    # array in structure_engine.py) can carry numpy scalar types
    # (e.g. np.int64), which json.dumps cannot serialize directly.
    return {
        "zone_id": zone.zone_id,
        "lower_bound": float(zone.lower_bound),
        "upper_bound": float(zone.upper_bound),
        "mid": float(zone.mid),
        "strength": float(zone.strength),
        "age": int(zone.age),
        "touch_count": int(zone.touch_count),
        "rejection_count": int(zone.rejection_count),
        "breakout_count": int(zone.breakout_count),
        "rejection_score": float(zone.rejection_score),
        "breakout_score": float(zone.breakout_score),
        "liquidity_score": float(zone.liquidity_score),
        "survival_probability": float(zone.survival_probability),
        "regime_relevance": {k: float(v) for k, v in zone.regime_relevance.items()},
        "zone_type": zone.zone_type,
        "created_at": int(zone.created_at),
        "last_touched_at": int(zone.last_touched_at),
    }


def zone_map_to_json(zone_map: ZoneMap) -> str:
    """Serialize a ZoneMap's nested zone list to a JSON string (§9 option A)."""
    payload = {
        "current_price": zone_map.current_price,
        "zones": [_zone_to_dict(z) for z in zone_map.zones],
        "nearest_support_id": zone_map.nearest_support.zone_id if zone_map.nearest_support else None,
        "nearest_resistance_id": zone_map.nearest_resistance.zone_id if zone_map.nearest_resistance else None,
    }
    return json.dumps(payload)


def zone_map_from_json(zone_map_json: str) -> List[Dict[str, Any]]:
    """Parse a zone_map_json column back into a list of zone dicts (for analysis)."""
    if not zone_map_json or (isinstance(zone_map_json, float)):
        return []
    payload = json.loads(zone_map_json)
    return payload.get("zones", [])


def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }

    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })

    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
                'distance_to_nearest_support': fv.distance_to_nearest_support,
                'distance_to_nearest_resistance': fv.distance_to_nearest_resistance,
                'support_strength': fv.support_strength,
                'resistance_strength': fv.resistance_strength,
            }.items()
        })

    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
                'age': rs.regime_age,
                'persistence_score': rs.regime_persistence_score,
            }.items()
        })

    # Flatten zone map (§9: nested zones -> JSON-blob column)
    if state.zone_map:
        zm = state.zone_map
        d['zone_map_json'] = zone_map_to_json(zm)
        d['zone_num_active'] = len(zm.zones)
        d['zone_nearest_support_mid'] = zm.nearest_support.mid if zm.nearest_support else None
        d['zone_nearest_resistance_mid'] = zm.nearest_resistance.mid if zm.nearest_resistance else None

    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl

    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
