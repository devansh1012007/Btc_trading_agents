"""
Unit tests for Phase 4 T4.6: single `EXIT_REASONS` vocabulary in
`src/labeling/triple_barrier.py`.
"""
import pytest

from src.labeling.triple_barrier import (
    EXIT_REASONS,
    PROFIT_EXIT_REASONS,
    STOP_EXIT_REASONS,
    is_stop_exit_reason,
    is_valid_exit_reason,
)

pytestmark = [pytest.mark.unit]


def test_exit_reasons_vocabulary_is_complete_and_partitioned():
    expected = {
        "take_profit",
        "stop_loss",
        "trailing_stop",
        "profit_exit",
        "loss_cut",
        "hard_loss_cap",
        "timeout",
    }
    assert set(EXIT_REASONS) == expected
    assert set(STOP_EXIT_REASONS) <= set(EXIT_REASONS)
    assert set(PROFIT_EXIT_REASONS) <= set(EXIT_REASONS)
    assert not (set(STOP_EXIT_REASONS) & set(PROFIT_EXIT_REASONS))
    for r in EXIT_REASONS:
        assert is_valid_exit_reason(r)
    for r in STOP_EXIT_REASONS:
        assert is_stop_exit_reason(r)
    assert not is_stop_exit_reason("take_profit")
    assert not is_stop_exit_reason("timeout")
