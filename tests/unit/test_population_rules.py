"""
Unit tests for Phase 8 T8.6: `select_population` and `population_rules_hash`.
"""
import pytest

from src.labeling.harness_labels import (
    population_rules_hash,
    select_population,
    select_sequential,
)

pytestmark = [pytest.mark.unit]


def test_select_population_matches_select_sequential_under_default_config():
    rows = [
        {"expert_name": "trend", "bar_index": 1, "bars_held": 3, "realized_return_pct": -0.5, "exit_reason": "stop_loss"},
        {"expert_name": "trend", "bar_index": 2, "bars_held": 2, "realized_return_pct": 0.4, "exit_reason": "take_profit"},
        {"expert_name": "trend", "bar_index": 4, "bars_held": 2, "realized_return_pct": 0.3, "exit_reason": "take_profit"},
        {"expert_name": "breakout", "bar_index": 1, "bars_held": 2, "realized_return_pct": 0.5, "exit_reason": "take_profit"},
    ]
    seq = select_sequential(rows, min_bars_between_orders=2)
    pop = select_population(rows, {}, min_bars_between_orders=2)
    assert pop == seq


def test_select_population_respects_cooldowns_and_hash_changes():
    rows = [
        {"expert_name": "trend", "bar_index": 1, "bars_held": 2, "realized_return_pct": -0.5, "exit_reason": "stop_loss"},
        {"expert_name": "trend", "bar_index": 3, "bars_held": 2, "realized_return_pct": 0.4, "exit_reason": "take_profit"},
        {"expert_name": "trend", "bar_index": 6, "bars_held": 2, "realized_return_pct": 0.3, "exit_reason": "take_profit"},
    ]
    cfg_no_cd = {}
    cfg_cd = {"risk": {"cooldowns": {"enabled": True, "after_stop_loss_bars": 3}}}
    assert population_rules_hash(cfg_no_cd) != population_rules_hash(cfg_cd)

    # Without cooldown: bar 1 (ends at 3) and bar 3 (starts at 3) are both kept
    assert select_population(rows, cfg_no_cd, min_bars_between_orders=1) == [True, True, True]
    # With 3-bar stop-loss cooldown after bar 1 finishes at bar 3 (cooled until bar 6): bar 3 is skipped, bar 6 is kept!
    assert select_population(rows, cfg_cd, min_bars_between_orders=1) == [True, False, True]
