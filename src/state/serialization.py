"""
Serialize/deserialize SystemState to/from Parquet, JSON.

Phase 2 (§9): ZoneMap's nested zones -> one JSON-blob column
(zone_map_json), option A from the plan.
Phase 3: same JSON-blob treatment for expert_outputs (nested per-expert
dict); opportunity_score and router_decision flatten into scalar columns
since they're not nested structures.
"""
import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from .contracts import (
    SystemState, MarketSnapshot, FeatureVector, RegimeState, Zone, ZoneMap,
    ExpertOutput, ExpertOutputs, OpportunityScore, RouterDecision,
)


def _zone_to_dict(zone: Zone) -> Dict[str, Any]:
    # Cast every value to a native Python type: fields derived from numpy
    # arrays upstream (e.g. created_at) can carry numpy scalar types
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
    payload = {
        "current_price": zone_map.current_price,
        "zones": [_zone_to_dict(z) for z in zone_map.zones],
        "nearest_support_id": zone_map.nearest_support.zone_id if zone_map.nearest_support else None,
        "nearest_resistance_id": zone_map.nearest_resistance.zone_id if zone_map.nearest_resistance else None,
    }
    return json.dumps(payload)


def zone_map_from_json(zone_map_json: str) -> List[Dict[str, Any]]:
    if not zone_map_json or (isinstance(zone_map_json, float)):
        return []
    payload = json.loads(zone_map_json)
    return payload.get("zones", [])


def _expert_output_to_dict(output: ExpertOutput) -> Dict[str, Any]:
    return {
        "expert_name": output.expert_name,
        "action": output.action.name,
        "confidence": float(output.confidence),
        "no_trade_probability": float(output.no_trade_probability),
        "expected_return": float(output.expected_return),
        "expected_risk": float(output.expected_risk),
        "reward_to_risk": float(output.reward_to_risk),
        "invalidation_distance": float(output.invalidation_distance),
        "zone_aware": bool(output.zone_aware),
        "model_output": float(output.model_output),
        "regime_used": output.regime_used.value,
    }


def expert_outputs_to_json(expert_outputs: ExpertOutputs) -> str:
    return json.dumps({name: _expert_output_to_dict(o) for name, o in expert_outputs.outputs.items()})


def expert_outputs_from_json(expert_outputs_json: str) -> Dict[str, Dict[str, Any]]:
    if not expert_outputs_json or isinstance(expert_outputs_json, float):
        return {}
    return json.loads(expert_outputs_json)


def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }

    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open, 'high': ms.high, 'low': ms.low, 'close': ms.close,
                'volume': ms.volume, 'bid': ms.bid, 'ask': ms.ask, 'spread_pct': ms.spread_pct,
            }.items()
        })

    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s, 'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s, 'spread_pct': fv.spread_pct,
                'distance_to_nearest_support': fv.distance_to_nearest_support,
                'distance_to_nearest_resistance': fv.distance_to_nearest_resistance,
                'support_strength': fv.support_strength,
                'resistance_strength': fv.resistance_strength,
            }.items()
        })

    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value, 'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up, 'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range, 'prob_panic': rs.prob_panic_spike,
                'age': rs.regime_age, 'persistence_score': rs.regime_persistence_score,
            }.items()
        })

    if state.zone_map:
        zm = state.zone_map
        d['zone_map_json'] = zone_map_to_json(zm)
        d['zone_num_active'] = len(zm.zones)
        d['zone_nearest_support_mid'] = zm.nearest_support.mid if zm.nearest_support else None
        d['zone_nearest_resistance_mid'] = zm.nearest_resistance.mid if zm.nearest_resistance else None

    if state.expert_outputs:
        d['expert_outputs_json'] = expert_outputs_to_json(state.expert_outputs)
        d['expert_consensus_action'] = state.expert_outputs.get_consensus_action().name
        best = state.expert_outputs.get_best_by_confidence()
        d['expert_best_name'] = best.expert_name if best else None
        d['expert_best_confidence'] = best.confidence if best else None

    if state.opportunity_score:
        os_ = state.opportunity_score
        d.update({
            f'opportunity_{k}': v for k, v in {
                'tradable': os_.tradable, 'tradable_probability': os_.tradable_probability,
                'environment_quality': os_.environment_quality,
                'opportunity_strength': os_.opportunity_strength,
                'edge_vs_fees': os_.edge_vs_fees, 'liquidity_score': os_.liquidity_score,
                'volatility_score': os_.volatility_score,
                'expert_consensus_score': os_.expert_consensus_score,
                'regime_acceptability': os_.regime_acceptability,
                'failed_rules': json.dumps(os_.failed_rules),
            }.items()
        })

    if state.router_decision:
        rd = state.router_decision
        d.update({
            f'router_{k}': v for k, v in {
                'selected_expert': rd.selected_expert,
                'selection_confidence': rd.selection_confidence,
                'no_trade_probability': rd.no_trade_probability,
                'reasoning': rd.reasoning,
                'expert_ranking_json': json.dumps({k2: float(v2) for k2, v2 in rd.expert_ranking.items()}),
            }.items()
        })

    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl

    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
