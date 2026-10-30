"""
Unit tests for Phase 11 T11.3: `RiskGovernor` tighten-only invariants (C15 / ADR-0004).
"""
import pytest

from src.risk.governor import RiskGovernor

pytestmark = [pytest.mark.unit]


def test_governor_is_strictly_tighten_only_and_never_loosens_caps():
    gov = RiskGovernor({
        "enabled": True,
        "soft_drawdown_pct": 2.0,
        "hard_drawdown_pct": 5.0,
        "loss_streak_tighten": 2,
        "loss_streak_halt": 4,
        "halt_minutes_on_trigger": 90.0,
    })

    # Even if an external caller passes permissive requested values (size=2.5, max_concurrent=10, cooldown=0.2),
    # RiskGovernor clamps them to configured ceilings (size<=1.0, max_concurrent<=3, cooldown>=1.0)!
    d_calm = gov.evaluate(
        drawdown_pct=0.0,
        consecutive_losses=0,
        configured_max_concurrent=3,
        requested_size_multiplier=2.5,
        requested_max_concurrent=10,
        requested_cooldown_multiplier=0.2,
    )
    assert d_calm.size_multiplier == pytest.approx(1.0)
    assert d_calm.max_concurrent_cap == 3
    assert d_calm.extra_cooldown_multiplier == pytest.approx(1.0)
    assert not d_calm.veto

    # Under soft drawdown + loss streak tighten: reduces size and concurrency, increases cooldown
    d_stress = gov.evaluate(
        drawdown_pct=3.5,
        consecutive_losses=2,
        configured_max_concurrent=4,
    )
    assert d_stress.size_multiplier <= 0.5
    assert d_stress.max_concurrent_cap <= 2
    assert d_stress.extra_cooldown_multiplier >= 1.5
    assert not d_stress.veto

    # Under hard drawdown: vetoes new entries and triggers time-boxed halt
    d_halt = gov.evaluate(
        drawdown_pct=5.5,
        consecutive_losses=4,
        configured_max_concurrent=4,
    )
    assert d_halt.veto is True
    assert d_halt.size_multiplier == 0.0
    assert d_halt.trigger_halt_minutes == pytest.approx(90.0)
    assert d_halt.reason is not None and d_halt.reason.startswith("governor_veto:")
