"""
Unit tests for Phase 16 T16.2: multi-position lane assignment (`assign_trade_lanes`)
in `src/gui/overlays.py`.
"""
import pytest

from src.gui.overlays import assign_trade_lanes

pytestmark = [pytest.mark.unit]


def test_assign_trade_lanes_gives_overlapping_trades_distinct_lanes_and_reuses_closed_lanes():
    records = [
        {"trade_id": "t1", "entry_timestamp": 1000, "exit_timestamp": 1300, "exit_reason": "take_profit"},
        {"trade_id": "t2", "entry_timestamp": 1100, "exit_timestamp": 1400, "exit_reason": "stop_loss"},
        {"trade_id": "t3", "entry_timestamp": 1300, "exit_timestamp": 1500, "exit_reason": "timeout"},
    ]
    lanes = assign_trade_lanes(records)
    # t1 (1000..1300) gets lane 0; t2 (1100..1400) overlaps t1 so gets lane 1;
    # t3 (1300..1500) starts when t1 exits at 1300, so reuses lane 0!
    assert lanes["t1"] == 0
    assert lanes["t2"] == 1
    assert lanes["t3"] == 0
