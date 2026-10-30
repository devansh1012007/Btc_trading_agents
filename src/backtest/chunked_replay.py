"""
Bounded-memory chunked replay (`run_chunked_replay`, Phase 13 T13.5):
streams arbitrarily long OHLCV DataFrames through a single stateful
pipeline (`FeatureEngine`, `RegimeDetector`, `DynamicStructureEngine`,
`ExpertLayer`, `OpportunityFilter`, `Router`) in chunks of `chunk_size`
bars so peak memory does not grow linearly with dataset length, while
producing identical `SystemState` outputs to an unchunked full-stack replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..execution.cost_model import CostModel
from ..experts.expert_registry import build_experts
from ..labeling.hold_policy import alignment_settings
from ..layers.expert_layer import ExpertLayer
from ..layers.feature_engine import FeatureEngine
from ..layers.market_data import MarketDataLayer
from ..layers.opportunity_filter import OpportunityFilter
from ..layers.regime_detector import RegimeDetector
from ..layers.router import Router
from ..layers.structural_features import attach_structural_features
from ..layers.structure_engine import DynamicStructureEngine
from ..liquidity.synthetic_liquidity_model import attach_synthetic_liquidity
from ..state.contracts import SystemState


@dataclass
class ChunkedReplaySummary:
    total_bars_processed: int = 0
    chunks_processed: int = 0
    retained_states: List[SystemState] = field(default_factory=list)
    touch_events: List[Dict] = field(default_factory=list)


def run_chunked_replay(
    df_ohlcv: pd.DataFrame,
    symbol: str,
    config: dict,
    *,
    chunk_size: int = 5_000,
    run_experts: bool = True,
    retain_states: bool = True,
    on_chunk: Optional[Callable[[List[SystemState], int], None]] = None,
) -> ChunkedReplaySummary:
    """Process `df_ohlcv` in chunks of `chunk_size` rows through a single continuous stateful pipeline."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    data_cfg = config.get("data", {})
    struct_cfg = config.get("structure", {})
    regime_cfg = config.get("regime", {})
    models_cfg = config.get("models", {})
    opp_cfg = config.get("opportunity_filter", {})
    router_cfg = config.get("router", {})
    labeling_cfg = config.get("labeling", {})
    system_cfg = config.get("system", {})
    liquidity_cfg = config.get("liquidity", {}).get("synthetic", {})
    data_granularity = system_cfg.get("data_granularity", "1m")

    feature_engine = FeatureEngine(lookback_bars=data_cfg.get("lookback_bars", 100), config=config)
    regime_detector = RegimeDetector(
        hmm_model_path=models_cfg.get("regime_hmm", "data/models/regime_hmm.pkl"),
        classifier_model_path=models_cfg.get("regime_classifier", "data/models/regime_classifier.pkl"),
        confidence_floor=regime_cfg.get("confidence_floor", 0.35),
        persistence_window_bars=regime_cfg.get("persistence_window_bars", 20),
        persistence_transition_threshold=regime_cfg.get("persistence_transition_threshold", 0.6),
        panic_z_threshold=regime_cfg.get("panic_z_threshold", 3.0),
        serving_source=models_cfg.get("regime_serving_source", "auto"),
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

    expert_layer = opportunity_filter = router = None
    if run_experts:
        hold_policy, require_aligned = alignment_settings(config)
        experts = build_experts(config, data_granularity=data_granularity)
        expert_layer = ExpertLayer(experts)
        opportunity_filter = OpportunityFilter(
            config=opp_cfg,
            labeling_config=labeling_cfg,
            model_path=opp_cfg.get("ml_model", {}).get("model_path"),
            data_granularity=data_granularity,
            hold_policy=hold_policy,
            require_aligned_models=require_aligned,
            cost_model=CostModel.from_config(config),
        )
        router = Router(
            config=router_cfg,
            model_path=router_cfg.get("model", {}).get("model_path"),
            data_granularity=data_granularity,
            hold_policy=hold_policy,
            require_aligned_models=require_aligned,
        )

    summary = ChunkedReplaySummary()
    n_rows = len(df_ohlcv)
    for start in range(0, n_rows, chunk_size):
        chunk_df = df_ohlcv.iloc[start : start + chunk_size]
        snapshots = MarketDataLayer.batch_create_from_dataframe(chunk_df, symbol)
        chunk_states: List[SystemState] = []

        for snapshot in snapshots:
            feature_engine.add_bar(snapshot)
            fv = feature_engine.compute_features(snapshot)
            if fv is None:
                continue

            regime_detector.add_bar(fv)
            regime_state = regime_detector.compute_regime(fv, symbol=symbol)

            structure_engine.add_bar(snapshot)
            zone_map = structure_engine.compute_zones(
                current_price=snapshot.close, symbol=symbol, regime_state=regime_state
            )
            fv = attach_structural_features(fv, zone_map)
            snapshot = attach_synthetic_liquidity(snapshot, fv, liquidity_cfg)

            expert_outputs = opportunity_score = router_decision = None
            if run_experts and expert_layer is not None and opportunity_filter is not None and router is not None:
                expert_outputs = expert_layer.evaluate(fv, zone_map, regime_state)
                opportunity_score = opportunity_filter.evaluate(
                    expert_outputs, regime_state, zone_map, snapshot, fv
                )
                router_decision = router.decide(
                    expert_outputs, regime_state, zone_map, opportunity_score
                )

            st = SystemState(
                timestamp=snapshot.timestamp,
                symbol=symbol,
                market_snapshot=snapshot,
                feature_vector=fv,
                zone_map=zone_map,
                regime_state=regime_state,
                expert_outputs=expert_outputs,
                opportunity_score=opportunity_score,
                router_decision=router_decision,
            )
            chunk_states.append(st)

        summary.total_bars_processed += len(chunk_states)
        summary.chunks_processed += 1
        if on_chunk is not None:
            on_chunk(chunk_states, summary.chunks_processed)
        if retain_states:
            summary.retained_states.extend(chunk_states)

    summary.touch_events = structure_engine.touch_events
    return summary
