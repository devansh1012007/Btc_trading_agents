"""
Unit tests for Phase 8 T8.5: `HaltManager` time-boxed halts and the
fresh-window rule on expiry.
"""
import pytest

from src.risk.halts import HaltConfig, HaltManager
from src.utils.clock import BarClock

pytestmark = [pytest.mark.unit]


def test_time_boxed_halt_expires_and_fresh_window_prevents_immediate_retrip():
    clock = BarClock(initial_timestamp=1_000_000.0)
    cfg = HaltConfig(
        enabled=True,
        rolling_loss_window_minutes=60.0,
        rolling_loss_max_pct=2.0,
        rolling_loss_halt_minutes=30.0,
        loss_streak_max=3,
        loss_streak_halt_minutes=20.0,
    )
    hm = HaltManager(config=cfg, clock=clock)

    # Two losses totaling -2.2% within 10 minutes trip the 30-minute rolling_loss halt
    assert hm.record_trade(-1.0, timestamp=1_000_000.0) is None
    clock.advance(1_000_300.0)
    ev = hm.record_trade(-1.2, timestamp=1_000_300.0)
    assert ev is not None
    assert ev.halt_type == "rolling_loss"
    assert ev.expires_at == pytest.approx(1_000_300.0 + 30.0 * 60.0)

    # 10 minutes into the halt: still halted
    clock.advance(1_000_900.0)
    halted, reason = hm.is_halted()
    assert halted is True and "rolling_loss" in reason

    # 30 minutes later (at expires_at = 1_002_100): halt expires!
    clock.advance(1_002_100.0)
    halted, _ = hm.is_halted()
    assert halted is False

    # Fresh-window rule: a small -0.3% loss after expiry does NOT combine with the
    # pre-halt -2.2% losses to immediately re-trip the rolling_loss halt!
    clock.advance(1_002_200.0)
    ev2 = hm.record_trade(-0.3, timestamp=1_002_200.0)
    assert ev2 is None
    assert hm.is_halted()[0] is False
