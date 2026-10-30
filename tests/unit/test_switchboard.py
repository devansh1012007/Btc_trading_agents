"""
Unit tests for the typed configuration Switchboard & validator (Phase 1, T1.2).
Covers >= 25 parametrized validation cases, default legacy equivalence,
unknown-key rejection, dependency closure, and cross-layer conflict checks.
"""
from __future__ import annotations

import copy
import pytest

from src.config.consistency_validator import ConfigConsistencyError, validate_config_consistency
from src.config.switchboard import Switchboard, validate_switchboard
from src.utils.config_helpers import load_layered_config

pytestmark = [pytest.mark.unit]


def test_default_switchboard_is_legacy_for_shipped_configs() -> None:
    for env in ("paper", "live"):
        cfg = load_layered_config(env)
        sb = Switchboard.from_config(cfg)
        assert sb.is_legacy(), f"Shipped {env} config should be legacy"
        assert sb.pipeline_order == "legacy"
        assert sb.max_concurrent == 1
        assert sb.entry_delay_seconds == 0.0
        assert sb.exit_delay_seconds == 0.0
        assert sb.execution_exit_model == "barrier_level"


def test_legacy_mode_button_forces_all_switches_off() -> None:
    cfg = load_layered_config("paper")
    cfg["legacy_mode"] = True
    cfg["pipeline"] = {"order": "staged", "stages": {"regime_dynamic": {"enabled": True}}}
    cfg["positions"] = {"max_concurrent": 4}
    cfg["execution"]["entry_delay_seconds"] = 2
    sb = Switchboard.from_config(cfg)
    assert sb.is_legacy()
    assert sb.pipeline_order == "legacy"
    assert sb.max_concurrent == 1
    assert sb.entry_delay_seconds == 0.0


@pytest.mark.parametrize(
    "bad_patch,expected_match",
    [
        ({"pipeline": {"unknown_key": 1}}, "pipeline.unknown_key"),
        ({"pipeline": {"order": "random"}}, "pipeline.order"),
        ({"pipeline": {"stages": {"bogus_stage": {"enabled": True}}}}, "pipeline.stages.bogus_stage"),
        ({"pipeline": {"stages": {"zone_selection": {"mode": "hourly"}}}}, "pipeline.stages.zone_selection.mode"),
        ({"pipeline": {"stages": {"zone_selection": {"n": 0}}}}, "pipeline.stages.zone_selection.n"),
        ({"pipeline": {"order": "staged", "stages": {"opportunity_confirm": {"enabled": False}}}}, "opportunity_confirm"),
        ({"regime": {"backend": "transformer"}}, "regime.backend"),
        ({"regime": {"k": {"mode": "guess"}}}, "regime.k.mode"),
        ({"regime": {"k": {"min": 1}}}, "regime.k.min"),
        ({"regime": {"k": {"min": 5, "max": 3}}}, "regime.k.max"),
        ({"regime": {"k": {"min": 2, "max": 20, "hard_cap": 12}}}, "regime.k.max"),
        ({"regime": {"min_regime_support_pct": 0.0}}, "regime.min_regime_support_pct"),
        ({"regime": {"min_mean_dwell_bars": 0}}, "regime.min_mean_dwell_bars"),
        ({"regime": {"serving_chain": ["bad_source"]}}, "regime.serving_chain"),
        ({"experts": {"instances": [{"id": "a", "template": "bad_tmpl"}]}}, "experts.instances\\[0\\].template"),
        ({"experts": {"instances": [{"id": "a", "template": "trend"}, {"id": "a", "template": "breakout"}]}}, "Duplicate experts.instances id"),
        ({"experts": {"selection": {"metric": "magic"}}}, "experts.selection.metric"),
        ({"experts": {"selection": {"top_n": 0}}}, "experts.selection.top_n"),
        (
            {
                "pipeline": {"stages": {"routers": {"enabled": True}}},
                "experts": {"instances": [{"id": "t0", "template": "trend", "enabled": True}]},
            },
            "routers.enabled=true requires at least 2",
        ),
        ({"exits": {"default_mode": "yolo"}}, "exits.default_mode"),
        ({"exits": {"per_expert": {"trend": "invalid"}}}, "exits.per_expert.trend"),
        ({"exits": {"hard": {"max_loss_pct": -1.0}}}, "exits.hard.max_loss_pct"),
        ({"exits": {"model": {"profit_threshold": 1.5}}}, "exits.model.profit_threshold"),
        ({"sizing": {"stop_source": "random"}}, "sizing.stop_source"),
        ({"sizing": {"risk_model": {"min_multiplier": 1.5}}}, "sizing.risk_model.min_multiplier"),
        ({"execution": {"entry_delay_seconds": 2, "simulated_latency_ms": 500}}, "entry_delay_seconds"),
        ({"positions": {"max_concurrent": 0}}, "positions.max_concurrent"),
        ({"positions": {"max_concurrent": 25, "hard_cap": 20}}, "positions.max_concurrent"),
        ({"governor": {"may": ["enlarge_size"]}}, "governor.may"),
        ({"handoff": {"mode": "execute"}, "execution": {"mode": "live"}}, "handoff.mode='execute' is blocked"),
        ({"halts": {"clock": "bar"}, "execution": {"mode": "live"}}, "halts.clock must be 'wall'"),
    ],
)
def test_invalid_switchboard_configs_raise_naming_key_path(bad_patch: dict, expected_match: str) -> None:
    base = load_layered_config("paper")
    from src.utils.config_helpers import _deep_merge
    cfg = _deep_merge(base, bad_patch)
    with pytest.raises(ConfigConsistencyError, match=expected_match):
        validate_switchboard(cfg)


def test_cross_layer_conflict_detection() -> None:
    base = {"pipeline": {"order": "legacy"}, "halts": {"enabled": True, "clock": "bar"}}
    overlay = {"pipeline": {"order": "staged"}}
    with pytest.raises(ConfigConsistencyError, match="Conflicting pipeline.order"):
        validate_switchboard(base, base_config=base, overlay_config=overlay)

    overlay_live = {"execution": {"mode": "live"}}
    with pytest.raises(ConfigConsistencyError, match="halts.clock"):
        validate_switchboard(base, base_config=base, overlay_config=overlay_live)


def test_valid_rich_switchboard_parses_cleanly() -> None:
    cfg = load_layered_config("paper")
    cfg["pipeline"] = {
        "order": "staged",
        "stages": {
            "regime_dynamic": {"enabled": True},
            "zone_selection": {"enabled": True, "mode": "every_n", "n": 5},
            "routers": {"enabled": True},
            "opportunity_confirm": {"enabled": True},
        },
    }
    cfg["experts"]["instances"] = [
        {"id": "trend@r0", "template": "trend", "regime": "r0", "enabled": True},
        {"id": "mr@r1", "template": "mean_reversion", "regime": "r1", "enabled": True},
    ]
    cfg["positions"] = {"max_concurrent": 3, "max_concurrent_per_expert": 2, "max_gross_exposure_pct": 60.0}
    sb = Switchboard.from_config(cfg)
    assert not sb.is_legacy()
    assert sb.pipeline_order == "staged"
    assert sb.stages.zone_selection_mode == "every_n"
    assert sb.stages.zone_selection_n == 5
    assert sb.max_concurrent == 3
