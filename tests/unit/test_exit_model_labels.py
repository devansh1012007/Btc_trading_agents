"""
Unit tests for Phase 10 T10.1 / Leakage Control L6: `split_open_position_bars_by_trade`
and `verify_no_trade_group_leakage`.
"""
import pytest

from src.exits.exit_model import (
    TradeGroupLeakageError,
    split_open_position_bars_by_trade,
    verify_no_trade_group_leakage,
)

pytestmark = [pytest.mark.unit]


def test_split_open_position_bars_by_trade_has_zero_trade_overlap():
    rows = []
    for t_idx in range(8):
        for bar_step in range(1, 5):
            rows.append({"trade_id": f"trade_{t_idx}", "bars_held": bar_step})

    train_idx, val_idx = split_open_position_bars_by_trade(rows, val_fraction=0.25)
    assert train_idx and val_idx
    verify_no_trade_group_leakage(rows, train_idx, val_idx)

    # Deliberately leak one bar of trade_0 into val_idx -> must raise TradeGroupLeakageError (L6)
    with pytest.raises(TradeGroupLeakageError, match="L6"):
        verify_no_trade_group_leakage(rows, train_idx, val_idx + [0])
