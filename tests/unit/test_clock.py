"""
Unit tests for Phase 8 T8.1: `Clock`, `WallClock`, `BarClock`, `RiskStateStore`
simulated UTC midnight reset / staleness, and AST gate forbidding direct
`datetime.now()` / `date.today()` calls in `src/layers/`, `src/execution/`, and `src/risk/`.
"""
import ast
from pathlib import Path

import pytest

from src.layers.risk_layer import RiskStateStore
from src.utils.clock import BarClock, WallClock

pytestmark = [pytest.mark.unit]


def test_bar_clock_resets_daily_loss_at_simulated_midnight():
    # 2023-11-14 23:50:00 UTC = 1_699_999_800
    clock = BarClock(initial_timestamp=1_699_999_800.0)
    store = RiskStateStore(persistent=False, clock=clock)
    store.heartbeat()
    store.record_trade_result(pnl_pct=-1.2)
    assert store.daily_loss_pct() == pytest.approx(1.2)
    assert not store.is_stale(threshold_seconds=120)

    # Advance 150 seconds -> crosses 120s staleness threshold AND crosses UTC midnight (1_700_006_400 is next day, wait:
    # 1_700_006_400 is 2023-11-15 00:00:00 UTC)
    clock.advance(1_699_999_800.0 + 150.0)
    assert store.is_stale(threshold_seconds=120)

    # Advance to 2023-11-15 00:05:00 UTC (1_700_006_700)
    clock.advance(1_700_006_700.0)
    assert store.daily_loss_pct() == pytest.approx(0.0)
    store.record_trade_result(pnl_pct=-0.4)
    assert store.daily_loss_pct() == pytest.approx(0.4)


def test_wall_clock_advance_is_noop():
    wc = WallClock()
    t1 = wc.now_epoch()
    wc.advance(1000.0)
    t2 = wc.now_epoch()
    assert t2 >= t1


def test_no_direct_datetime_now_or_date_today_in_layers_execution_or_risk():
    """AST gate: no module in src/layers/, src/execution/, or src/risk/ may call
    datetime.now() or date.today() directly."""
    violations = []
    for folder in ("src/layers", "src/execution", "src/risk"):
        for py_file in sorted(Path(folder).rglob("*.py")):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        pair = (node.func.value.id, node.func.attr)
                        if pair in (("datetime", "now"), ("date", "today")):
                            violations.append(f"{py_file}:{node.lineno} {pair[0]}.{pair[1]}()")
    assert not violations, f"Direct wall-clock calls found: {violations}"
