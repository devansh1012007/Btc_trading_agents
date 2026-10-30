"""
Single Stage Registry (Phase 1, T1.1 / §4.1).

Declares every stage of the system in order, with its inputs, trained
artefacts, configuration switch key, and fallback description.
Both the runtime pipeline (`TradingPipeline` / `process_bar`) and the
training orchestrator (`src/training/plan.py`) derive their stage ordering
from this registry so the two can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StageSpec:
    """Metadata for a single pipeline / training stage."""
    index: int
    name: str
    runtime_role: str
    inputs: Tuple[str, ...]
    artefacts: Tuple[str, ...]
    config_key: str
    fallback: str
    trace_stage_name: str


#: Canonical stages in staged (harness-aligned training) order (§4.1).
STAGES_STAGED: Tuple[StageSpec, ...] = (
    StageSpec(
        index=0,
        name="dataset",
        runtime_role="instrument + (optional) ATM derivative",
        inputs=(),
        artefacts=("selection_manifest",),
        config_key="market.derivatives.mode",
        fallback="single configured instrument",
        trace_stage_name="market_data",
    ),
    StageSpec(
        index=1,
        name="regime",
        runtime_role="RegimeState (K regimes)",
        inputs=("dataset",),
        artefacts=("regime_backend", "regime_distiller"),
        config_key="pipeline.stages.regime_dynamic.enabled",
        fallback="legacy 4-class LightGBM -> HMM -> rules",
        trace_stage_name="regime_detector",
    ),
    StageSpec(
        index=2,
        name="zone_selection",
        runtime_role="scheduled zone (re)generation",
        inputs=("dataset", "regime"),
        artefacts=("zone_cadence_config",),
        config_key="pipeline.stages.zone_selection.enabled",
        fallback="every bar (legacy)",
        trace_stage_name="structure_engine",
    ),
    StageSpec(
        index=3,
        name="zone_survival",
        runtime_role="zone validity",
        inputs=("dataset", "regime", "zone_selection"),
        artefacts=("zone_survival_model",),
        config_key="pipeline.stages.zone_survival.enabled",
        fallback="rule-based survival (legacy)",
        trace_stage_name="zone_survival",
    ),
    StageSpec(
        index=4,
        name="barrier_sizing",
        runtime_role="default ExitPlan levels",
        inputs=("dataset", "regime", "zone_selection", "zone_survival"),
        artefacts=("barrier_sizing_sl", "barrier_sizing_tp"),
        config_key="pipeline.stages.barrier_sizing_model.enabled",
        fallback="k x volatility (legacy)",
        trace_stage_name="barrier_sizing",
    ),
    StageSpec(
        index=5,
        name="experts",
        runtime_role="ExpertOutputs per enabled instance",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing"),
        artefacts=("expert_scorers", "expert_plan_heads"),
        config_key="experts.instances",
        fallback="rule-based expert (legacy)",
        trace_stage_name="expert_layer",
    ),
    StageSpec(
        index=6,
        name="exit_models",
        runtime_role="per-bar exit decision while in trade",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing", "experts"),
        artefacts=("exit_models_profit", "exit_models_loss"),
        config_key="pipeline.stages.exit_models.enabled",
        fallback="expert plan -> hard limits",
        trace_stage_name="exit_models",
    ),
    StageSpec(
        index=7,
        name="routers",
        runtime_role="pick one instance or no_trade",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing", "experts", "exit_models"),
        artefacts=("router_a", "router_b"),
        config_key="pipeline.stages.routers.enabled",
        fallback="confidence router (legacy)",
        trace_stage_name="router",
    ),
    StageSpec(
        index=8,
        name="opportunity_confirm",
        runtime_role="ML tradability confirmation (staged)",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing", "experts", "exit_models", "routers"),
        artefacts=("opportunity_confirm_model",),
        config_key="pipeline.stages.opportunity_confirm.enabled",
        fallback="hard gate only (legacy ML blend in legacy order)",
        trace_stage_name="opportunity_filter",
    ),
    StageSpec(
        index=9,
        name="risk_sizing",
        runtime_role="final size multiplier + risk score",
        inputs=(
            "dataset", "regime", "zone_selection", "zone_survival",
            "barrier_sizing", "experts", "exit_models", "routers", "opportunity_confirm",
        ),
        artefacts=("risk_sizing_model",),
        config_key="pipeline.stages.risk_sizing_model.enabled",
        fallback="base size x liquidity x pause (legacy)",
        trace_stage_name="risk_layer",
    ),
)

#: Legacy order places opportunity_filter before router (preserving today's runtime & training order).
STAGES_LEGACY: Tuple[StageSpec, ...] = (
    STAGES_STAGED[0],  # dataset
    STAGES_STAGED[1],  # regime
    STAGES_STAGED[2],  # zone_selection
    STAGES_STAGED[3],  # zone_survival
    STAGES_STAGED[4],  # barrier_sizing
    STAGES_STAGED[5],  # experts
    STAGES_STAGED[6],  # exit_models
    StageSpec(
        index=7,
        name="opportunity_confirm",
        runtime_role="hard rules + ML tradability blend before router (legacy)",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing", "experts"),
        artefacts=("opportunity_filter_model",),
        config_key="pipeline.stages.opportunity_confirm.enabled",
        fallback="hard rules only",
        trace_stage_name="opportunity_filter",
    ),
    StageSpec(
        index=8,
        name="routers",
        runtime_role="pick one expert or no_trade (legacy)",
        inputs=("dataset", "regime", "zone_selection", "zone_survival", "barrier_sizing", "experts", "opportunity_confirm"),
        artefacts=("router_model",),
        config_key="pipeline.stages.routers.enabled",
        fallback="confidence router (legacy)",
        trace_stage_name="router",
    ),
    STAGES_STAGED[9],  # risk_sizing
)

STAGE_BY_NAME = {s.name: s for s in STAGES_STAGED}


def get_stages(order: str = "legacy") -> Tuple[StageSpec, ...]:
    """Return the ordered stage tuple for ``order`` ('legacy' or 'staged')."""
    if order == "legacy":
        return STAGES_LEGACY
    if order == "staged":
        return STAGES_STAGED
    raise ValueError(f"Unknown pipeline.order {order!r}; expected 'legacy' or 'staged'")


def stage_names(order: str = "legacy") -> Tuple[str, ...]:
    """Ordered stage names for ``order``."""
    return tuple(s.name for s in get_stages(order))


def decision_trace_stage_order(order: str = "legacy") -> Tuple[str, ...]:
    """Ordered decision-layer trace stage names emitted by ``build_trace_stages``."""
    if order == "legacy":
        return ("expert_layer", "opportunity_filter", "router", "risk_layer", "execution")
    if order == "staged":
        return ("expert_layer", "opportunity_filter_hard", "router", "opportunity_confirm", "risk_layer", "execution")
    raise ValueError(f"Unknown pipeline.order {order!r}; expected 'legacy' or 'staged'")
