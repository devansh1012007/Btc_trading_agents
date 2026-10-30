"""
Unit tests for Phase 8 T8.4: `CooldownManager` per-instance cool-downs
and persistence across restarts.
"""
import pytest

from src.layers.risk_layer import RiskStateStore
from src.risk.cooldowns import CooldownConfig, CooldownManager

pytestmark = [pytest.mark.unit]


def test_cooldown_after_stop_loss_and_consecutive_losses_and_persistence(tmp_path):
    db_path = str(tmp_path / "cd.db")
    store1 = RiskStateStore(db_path=db_path)
    cfg = CooldownConfig(
        enabled=True,
        after_stop_loss_bars=2,
        consecutive_losses=2,
        cooldown_bars=4,
    )
    mgr1 = CooldownManager(cfg, store=store1)

    # 1. Stop loss on trend_expert at bar 10 -> cooled down for 2 bars (bars 10, 11; free at bar 12)
    reason = mgr1.record_outcome("trend_expert", pnl_pct=-0.5, exit_reason="stop_loss", timestamp=1000.0, bar_index=10)
    assert reason is not None
    cooled11, _ = mgr1.is_cooled_down("trend_expert", 1060.0, current_bar_index=11)
    cooled12, _ = mgr1.is_cooled_down("trend_expert", 1120.0, current_bar_index=12)
    assert cooled11 is True
    assert cooled12 is False
    # Other experts are unaffected
    assert mgr1.is_cooled_down("breakout_expert", 1060.0, current_bar_index=11) == (False, None)

    # 2. Second loss on trend_expert at bar 12 triggers consecutive_losses=2 -> 4 bars cooldown (until bar 16)
    mgr1.record_outcome("trend_expert", pnl_pct=-0.2, exit_reason="timeout", timestamp=1120.0, bar_index=12)
    assert mgr1.is_cooled_down("trend_expert", 1180.0, current_bar_index=15)[0] is True
    assert mgr1.is_cooled_down("trend_expert", 1240.0, current_bar_index=16)[0] is False

    # 3. Persistence across store restart
    store2 = RiskStateStore(db_path=db_path)
    mgr2 = CooldownManager(cfg, store=store2)
    assert mgr2.is_cooled_down("trend_expert", 1180.0, current_bar_index=15)[0] is True
