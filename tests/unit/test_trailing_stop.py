"""
Unit tests for Phase 4 T4.2 & T4.3: `TrailState`, `advance_trail` (1-bar ratchet
lag preventing same-bar high-then-low look-ahead), lockout enforcement, and
labeler vs runtime parity.
"""
import pytest

from src.execution.position_lifecycle import OpenPosition, check_position_exit
from src.labeling.triple_barrier import (
    TrailState,
    advance_trail,
    barrier_exit_step,
    label_triple_barrier,
)
from src.state.contracts import ExitPlan, MarketSnapshot, TrailSpec

pytestmark = [pytest.mark.unit]


def _bar(ts: int, o: float, h: float, low: float, c: float) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=o,
        high=h,
        low=low,
        close=c,
        volume=10.0,
        bid=c * 0.9999,
        ask=c * 1.0001,
        bid_volume=5.0,
        ask_volume=5.0,
        spread_pct=0.02,
        hour_utc=12,
        day_of_week=2,
    )


def test_trailing_stop_1_bar_lag_prevents_same_bar_lookahead():
    """On bar 1, price spikes to 102 (+2%, activating trail at 102*(1-0.005)=101.49)
    AND drops to 100.5 on the SAME bar. Because we do not know whether high or low
    came first inside bar 1, the trail activated on bar 1 cannot trigger until bar 2!"""
    plan = ExitPlan(
        stop_loss_pct=3.0,
        take_profit_pct=5.0,
        max_holding_bars=5,
        trailing=TrailSpec(enabled=True, activation_pct=1.0, distance_pct=0.5, step_pct=0.1),
    )
    ts = TrailState(best_favorable_price=100.0)
    bar1 = _bar(1060, 100.0, 102.0, 100.5, 101.6)
    step1 = barrier_exit_step(
        1, 100.0, 5.0, 3.0, bar1, bars_held=1, max_holding_bars=5, min_hold_bars=1,
        exit_plan=plan, trail_state=ts,
    )
    # Bar 1 must NOT trigger trailing_stop on the same bar it activated
    assert step1 is None
    assert ts.activated is True
    assert ts.trail_stop_level == pytest.approx(102.0 * 0.995)

    # Bar 2 trades down to 101.0 (below 101.49 trail stop) -> triggers trailing_stop at 101.49!
    bar2 = _bar(1120, 101.6, 101.8, 101.0, 101.2)
    step2 = barrier_exit_step(
        1, 100.0, 5.0, 3.0, bar2, bars_held=2, max_holding_bars=5, min_hold_bars=1,
        exit_plan=plan, trail_state=ts,
    )
    assert step2 is not None
    assert step2[0] == "trailing_stop"
    assert step2[1] == pytest.approx(102.0 * 0.995)


def test_trailing_stop_respects_lockout_and_matches_between_labeler_and_runtime():
    plan = ExitPlan(
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_bars=6,
        trailing=TrailSpec(enabled=True, activation_pct=1.0, distance_pct=0.5, step_pct=0.05),
    )
    bars = [
        _bar(1010, 100.0, 101.5, 99.8, 101.2),  # bar 1 (locked out, activates trail at 101.5*0.995=100.9925)
        _bar(1020, 101.2, 101.3, 100.2, 100.4),  # bar 2 (locked out even though low 100.2 < 100.9925)
        _bar(1030, 101.1, 101.2, 100.5, 100.6),  # bar 3 (first legal bar: low 100.5 <= 100.9925 -> trailing_stop!)
    ]
    out = label_triple_barrier(
        bars,
        entry_price=100.0,
        direction=1,
        take_profit_pct=4.0,
        stop_loss_pct=2.0,
        max_holding_bars=6,
        fee_pct=0.04,
        entry_timestamp=1000,
        min_hold_bars=3,
        exit_plan=plan,
    )
    assert out.exit_reason == "trailing_stop"
    assert out.bars_held == 3
    assert len(out.trail_history) == 3

    pos = OpenPosition(
        trade_id="t-trail",
        entry_timestamp=1000,
        entry_price=100.0,
        direction=1,
        size_usdt=1000.0,
        take_profit_pct=4.0,
        stop_loss_pct=2.0,
        max_holding_bars=6,
        min_hold_bars=3,
        exit_plan=plan,
    )
    assert check_position_exit(pos, bars[0], 0.04) is None
    assert check_position_exit(pos, bars[1], 0.04) is None
    info = check_position_exit(pos, bars[2], 0.04)
    assert info is not None
    assert info["exit_reason"] == "trailing_stop"
    assert info["bars_held"] == 3
    assert info["realized_pnl_pct"] == pytest.approx(out.realized_return_pct)


def test_short_trailing_stop_ratchets_downward():
    spec = TrailSpec(enabled=True, activation_pct=1.0, distance_pct=0.5, step_pct=0.2)
    ts = advance_trail(-1, 100.0, _bar(1000, 100.0, 100.2, 98.5, 98.8), spec)
    assert ts.activated is True
    assert ts.trail_stop_level == pytest.approx(98.5 * 1.005)
