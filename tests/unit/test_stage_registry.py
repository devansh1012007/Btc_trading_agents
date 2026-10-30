"""
Unit tests for the single Stage Registry (Phase 1, T1.1).
"""
from __future__ import annotations

import pytest

from src.diagnostics.decision_trace import build_trace_stages
from src.execution.position_lifecycle import build_execution_engine, process_bar
from src.layers.market_data import MarketDataLayer
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.pipeline import TradingPipeline
from src.pipeline_stages import (
    STAGE_BY_NAME,
    STAGES_LEGACY,
    STAGES_STAGED,
    decision_trace_stage_order,
    get_stages,
    stage_names,
)
from src.utils.config_helpers import load_layered_config
from src.utils.synthetic import synthetic_ohlcv

pytestmark = [pytest.mark.unit]


def test_stage_registry_defines_all_ten_stages_in_order() -> None:
    assert len(STAGES_STAGED) == 10
    assert len(STAGES_LEGACY) == 10
    assert stage_names("staged") == (
        "dataset", "regime", "zone_selection", "zone_survival",
        "barrier_sizing", "experts", "exit_models", "routers",
        "opportunity_confirm", "risk_sizing",
    )
    assert stage_names("legacy") == (
        "dataset", "regime", "zone_selection", "zone_survival",
        "barrier_sizing", "experts", "exit_models", "opportunity_confirm",
        "routers", "risk_sizing",
    )


def test_every_stage_declares_valid_upstream_inputs_and_fallback() -> None:
    for order in ("legacy", "staged"):
        stages = get_stages(order)
        seen = set()
        for stg in stages:
            assert stg.fallback
            assert stg.config_key
            for inp in stg.inputs:
                assert inp in seen, f"{stg.name} in {order} references unseen upstream {inp!r}"
            seen.add(stg.name)
    assert set(STAGE_BY_NAME) == set(stage_names("staged"))


def test_stage_registry_order_matches_pipeline_trace(tmp_path) -> None:
    cfg = load_layered_config("paper")
    pipeline = TradingPipeline(symbol="BTC/USDT", config=cfg, data_dir=str(tmp_path / "data"))
    assert pipeline.stages == STAGES_LEGACY

    bars = synthetic_ohlcv(25, seed=42)
    snapshots = MarketDataLayer.batch_create_from_dataframe(bars, "BTC/USDT")
    state = None
    for snap in snapshots:
        state = pipeline.run_live_bar(snap)

    assert state is not None
    risk_layer = RiskLayer(
        config=cfg.get("risk", {}),
        store=RiskStateStore(persistent=False),
        kill_switch_path=str(tmp_path / "NO_KILL"),
    )
    exec_engine = build_execution_engine(cfg, pipeline.connector, "BTC/USDT")
    outcome = process_bar(
        state, risk_layer=risk_layer, execution_engine=exec_engine,
        position=None, equity=10_000.0, fee_pct=0.04, config=cfg,
    )
    emitted = [s.stage_name for s in outcome.trace.stages]
    expected_order = list(decision_trace_stage_order("legacy"))
    indices = [expected_order.index(name) for name in emitted]
    assert indices == sorted(indices), f"Emitted trace stages {emitted} do not follow registry order {expected_order}"
    assert emitted[:3] == ["expert_layer", "opportunity_filter", "router"]


def test_unknown_pipeline_order_raises() -> None:
    with pytest.raises(ValueError, match="pipeline.order"):
        get_stages("bogus")
