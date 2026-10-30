"""
Unit tests for Phase 3 T3.1-T3.4: `exit_fill_price`, two-phase delayed exits
(`PendingExit`), and parity between `label_triple_barrier` and `check_position_exit`.
"""
import numpy as np
import pytest

from src.execution.cost_model import CostModel
from src.execution.execution_engine import exit_fill_price
from src.execution.position_lifecycle import OpenPosition, check_position_exit
from src.labeling.hold_policy import HoldPolicy
from src.labeling.triple_barrier import label_triple_barrier
from src.state.contracts import MarketSnapshot

pytestmark = [pytest.mark.unit]


def _bar(ts: int, o: float, h: float, low: float, c: float, spread: float = 0.04) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=o,
        high=h,
        low=low,
        close=c,
        volume=20.0,
        bid=c * (1 - spread / 200),
        ask=c * (1 + spread / 200),
        bid_volume=10.0,
        ask_volume=10.0,
        spread_pct=spread,
        hour_utc=12,
        day_of_week=2,
    )


def test_barrier_level_default_is_exact_trigger_price():
    b = _bar(1000, 100.0, 102.5, 99.5, 102.0)
    cm = CostModel()
    assert exit_fill_price(1, 102.0, b, "take_profit", cm, exit_model="barrier_level") == pytest.approx(102.0)
    assert exit_fill_price(-1, 98.0, b, "take_profit", cm, exit_model="barrier_level") == pytest.approx(98.0)


def test_realistic_fill_charges_spread_and_slippage_in_adverse_direction():
    b = _bar(1000, 100.0, 102.5, 99.5, 102.0, spread=0.04)
    cm = CostModel.from_config({
        "costs": {
            "spread": {"source": "snapshot"},
            "slippage": {"model": "fixed", "fixed_pct": 0.01, "stop_slippage_multiplier": 2.0},
        }
    })
    # Long TP at 100: half-spread (0.02%) + slip (0.01%) = 0.03% adverse -> 99.97
    px_long_tp = exit_fill_price(1, 100.0, b, "take_profit", cm, exit_model="realistic_fill")
    assert px_long_tp == pytest.approx(100.0 * (1 - 0.0003))

    # Long SL at 100: half-spread (0.02%) + stop slip (0.02%) = 0.04% adverse -> 99.96
    px_long_sl = exit_fill_price(1, 100.0, b, "stop_loss", cm, exit_model="realistic_fill")
    assert px_long_sl == pytest.approx(100.0 * (1 - 0.0004))

    # Short SL at 100: adverse is HIGHER fill price -> 100.04
    px_short_sl = exit_fill_price(-1, 100.0, b, "stop_loss", cm, exit_model="realistic_fill")
    assert px_short_sl == pytest.approx(100.0 * (1 + 0.0004))


def test_two_phase_pending_exit_across_1s_bars():
    """On 1s bars with 2s exit delay, a trigger on bar k fills on bar k+2."""
    cm = CostModel.from_config({
        "costs": {
            "fees": {"taker_pct": 0.04},
            "spread": {"source": "fixed", "fixed_pct": 0.0},
            "slippage": {"model": "fixed", "fixed_pct": 0.0},
            "latency": {"exit_delay_seconds": 2.0},
        }
    })
    bars = [
        _bar(1001, 100.0, 102.5, 99.9, 102.0, spread=0.0),  # bar 1: hits TP=2% (102.0), enters pending_exit
        _bar(1002, 101.8, 101.9, 101.5, 101.6, spread=0.0),  # bar 2: 1s elapsed < 2s -> still pending
        _bar(1003, 101.4, 101.5, 101.0, 101.2, spread=0.0),  # bar 3: 2s elapsed -> fills at bar 3 open (101.4)
    ]
    out = label_triple_barrier(
        bars,
        entry_price=100.0,
        direction=1,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
        fee_pct=0.04,
        entry_timestamp=1000,
        min_hold_bars=1,
        cost_model=cm,
        exit_model="realistic_fill",
        granularity_seconds=1.0,
    )
    assert out.exit_reason == "take_profit"
    assert out.bars_held == 3
    assert out.exit_fill_price == pytest.approx(101.4)

    pos = OpenPosition(
        trade_id="t1",
        entry_timestamp=1000,
        entry_price=100.0,
        direction=1,
        size_usdt=1000.0,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
        min_hold_bars=1,
        exit_delay_seconds=2.0,
        exit_model="realistic_fill",
    )
    assert check_position_exit(pos, bars[0], 0.04, cost_model=cm, granularity_seconds=1.0) is None
    assert pos.pending_exit is not None
    assert check_position_exit(pos, bars[1], 0.04, cost_model=cm, granularity_seconds=1.0) is None
    info = check_position_exit(pos, bars[2], 0.04, cost_model=cm, granularity_seconds=1.0)
    assert info is not None
    assert info["exit_reason"] == "take_profit"
    assert info["bars_held"] == 3
    assert info["exit_price"] == pytest.approx(101.4)
    assert info["realized_pnl_pct"] == pytest.approx(out.realized_return_pct)


def test_hold_policy_label_horizon_bars_includes_entry_and_exit_delays():
    pol = HoldPolicy.from_config({
        "system": {"data_granularity": "1s"},
        "labeling": {"max_holding_bars": 15},
        "costs": {"latency": {"entry_delay_seconds": 2.0, "exit_delay_seconds": 2.0}},
    })
    assert pol.min_hold_bars == 12
    assert pol.exit_delay_bars == 2
    assert pol.label_horizon_bars == 19
    assert pol.fill_to_fill_seconds(12) == pytest.approx(12.0)
