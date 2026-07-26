"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState


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
            }.items()
        })
    
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
