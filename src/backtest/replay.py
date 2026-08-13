"""
Shared full-stack replay helper (NEW, Phase 3 shared infra): runs
historical OHLCV bars through the complete Phase 1-3 pipeline (features ->
regime -> structure -> experts -> opportunity filter -> router) once, and
returns everything downstream training/backtest scripts need.

train_trend_expert.py, train_mean_reversion_expert.py,
train_breakout_expert.py, train_opportunity_filter.py, train_router.py,
backtest_phase3.py, and backtest_compare_expert_choices.py all replay
through this identical code path -- the same one pipeline.py uses at
serving time -- rather than seven slightly-different reimplementations
that could silently drift apart.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pandas as pd

from ..layers.market_data import MarketDataLayer
from ..layers.feature_engine import FeatureEngine
from ..layers.regime_detector import RegimeDetector
from ..layers.structure_engine import DynamicStructureEngine
from ..layers.structural_features import attach_structural_features
from ..layers.expert_layer import ExpertLayer
from ..layers.opportunity_filter import OpportunityFilter
from ..layers.router import Router
from ..experts.expert_registry import build_experts
from ..state.contracts import SystemState, MarketSnapshot


@dataclass
class ReplayResult:
    system_states: List[SystemState] = field(default_factory=list)
    # Snapshots actually seen by the structure engine, aligned 1:1 with
    # its internal bar_counter and with system_states (bar_index N ->
    # structure_snapshots[N-1] -> system_states[N-1]).
    structure_snapshots: List[MarketSnapshot] = field(default_factory=list)
    touch_events: List[Dict] = field(default_factory=list)


def replay_full_stack(
    df_ohlcv: pd.DataFrame, symbol: str, config: dict, run_experts: bool = True,
) -> ReplayResult:
    """
    Replay bars in causal order through the full stack. Set
    run_experts=False when a caller only needs feature/regime/structure
    output (e.g. harvesting from Phase 2's touch-event stream directly).
    """
    data_cfg = config.get("data", {})
    struct_cfg = config.get("structure", {})
    regime_cfg = config.get("regime", {})
    models_cfg = config.get("models", {})
    opp_cfg = config.get("opportunity_filter", {})
    router_cfg = config.get("router", {})
    labeling_cfg = config.get("labeling", {})
    system_cfg = config.get("system", {})
    data_granularity = system_cfg.get("data_granularity", "1m")

    feature_engine = FeatureEngine(lookback_bars=data_cfg.get("lookback_bars", 100))
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
        # No zone_survival_model here on purpose: label harvesting must
        # run against the pure rule-based engine.
    )

    expert_layer = opportunity_filter = router = None
    if run_experts:
        experts = build_experts(config, data_granularity=data_granularity)
        expert_layer = ExpertLayer(experts)
        opportunity_filter = OpportunityFilter(
            config=opp_cfg, labeling_config=labeling_cfg,
            model_path=opp_cfg.get("ml_model", {}).get("model_path"),
            data_granularity=data_granularity,
        )
        router = Router(
            config=router_cfg, model_path=router_cfg.get("model", {}).get("model_path"),
            data_granularity=data_granularity,
        )

    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
    result = ReplayResult()

    for snapshot in snapshots:
        feature_engine.add_bar(snapshot)
        fv = feature_engine.compute_features(snapshot)
        if fv is None:
            continue

        regime_detector.add_bar(fv)
        regime_state = regime_detector.compute_regime(fv, symbol=symbol)

        structure_engine.add_bar(snapshot)
        result.structure_snapshots.append(snapshot)
        zone_map = structure_engine.compute_zones(current_price=snapshot.close, symbol=symbol, regime_state=regime_state)
        fv = attach_structural_features(fv, zone_map)

        expert_outputs = opportunity_score = router_decision = None
        if run_experts:
            expert_outputs = expert_layer.evaluate(fv, zone_map, regime_state)
            opportunity_score = opportunity_filter.evaluate(expert_outputs, regime_state, zone_map, snapshot, fv)
            router_decision = router.decide(expert_outputs, regime_state, zone_map, opportunity_score)

        state = SystemState(
            timestamp=snapshot.timestamp, symbol=symbol, market_snapshot=snapshot, feature_vector=fv,
            zone_map=zone_map, regime_state=regime_state, expert_outputs=expert_outputs,
            opportunity_score=opportunity_score, router_decision=router_decision,
        )
        result.system_states.append(state)

    result.touch_events = structure_engine.touch_events
    return result


def harvest_zone_touch_candidates(
    df_ohlcv: pd.DataFrame, symbol: str, config: dict, event_type: str, expert_name: str, build_features_fn,
) -> pd.DataFrame:
    """
    Shared harvesting path for Mean-Reversion and Breakout experts (§6.3,
    §6.4): replay bars WITHOUT running experts (structure engine's own
    touch-event log already tells us when a rejection or breakout
    happened), then apply triple-barrier labeling forward from each
    matching touch event's bar.

    Direction convention:
      - rejection off support    -> price bounced UP    -> long  (+1)
      - rejection off resistance -> price bounced DOWN   -> short (-1)
      - breakout of support      -> price continued DOWN -> short (-1)
      - breakout of resistance   -> price continued UP    -> long  (+1)
    """
    from ..labeling.triple_barrier import label_triple_barrier, compute_barrier_pcts, barrier_volatility_pct

    result = replay_full_stack(df_ohlcv, symbol, config, run_experts=False)

    labeling_cfg = config.get("labeling", {})
    max_holding_bars = labeling_cfg.get("max_holding_bars", 5)
    fee_pct = labeling_cfg.get("fee_pct", 0.04)
    k_tp_default = labeling_cfg.get("k_tp", 1.5)
    k_sl_default = labeling_cfg.get("k_sl", 1.0)
    regime_overrides = labeling_cfg.get("regime_specific_barriers", {})

    state_by_bar_index = {i + 1: s for i, s in enumerate(result.system_states)}

    rows = []
    for event in result.touch_events:
        if event["event_type"] != event_type:
            continue
        bar_index = event["bar_index"]
        state = state_by_bar_index.get(bar_index)
        if state is None:
            continue

        matched_zone = next((z for z in state.zone_map.zones if z.zone_id == event["zone_id"]), None)
        if matched_zone is None:
            continue

        if event_type == "rejection":
            direction = 1 if matched_zone.zone_type == "support" else -1
        else:
            direction = -1 if matched_zone.zone_type == "support" else 1

        regime_key = state.regime_state.primary_regime.name
        overrides = regime_overrides.get(regime_key, {})
        k_tp = overrides.get("k_tp", k_tp_default)
        k_sl = overrides.get("k_sl", k_sl_default)
        volatility_pct = barrier_volatility_pct(state.feature_vector)
        take_profit_pct, stop_loss_pct = compute_barrier_pcts(volatility_pct, k_tp, k_sl)

        forward_bars = result.structure_snapshots[bar_index:]
        if not forward_bars:
            continue

        outcome = label_triple_barrier(
            forward_bars=forward_bars, entry_price=state.market_snapshot.close, direction=direction,
            take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars, fee_pct=fee_pct, entry_timestamp=state.timestamp,
        )

        features = build_features_fn(state.feature_vector, state.zone_map, state.regime_state, direction)
        rows.append({
            **features, "expert_name": expert_name, "timestamp": state.timestamp, "bar_index": bar_index,
            "direction": direction, "exit_reason": outcome.exit_reason, "bars_held": outcome.bars_held,
            "realized_return_pct": outcome.realized_return_pct,
            "max_adverse_excursion_pct": outcome.max_adverse_excursion_pct,
            "label": 1 if outcome.realized_return_pct > 0 else 0,
        })

    return pd.DataFrame(rows)
