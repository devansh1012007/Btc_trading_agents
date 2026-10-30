"""
Unit tests for Phase 10 T10.2: `evaluate_exit_ladder` 6-step precedence ladder.
"""
import pytest

from src.exits.exit_model import ExitModel, evaluate_exit_ladder
from src.state.contracts import ExitPlan, MarketSnapshot

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


class _AlwaysExitHead:
    def predict_proba(self, X):
        return [[0.05, 0.95]]


def test_exit_ladder_precedence_lockout_hard_cap_stop_tp_ml_timeout():
    em = ExitModel("trend", profit_threshold=0.6, loss_cut_threshold=0.6)
    em.profit_head = _AlwaysExitHead()
    em.loss_cut_head = _AlwaysExitHead()
    plan = ExitPlan(stop_loss_pct=2.0, take_profit_pct=3.0, max_holding_bars=5, hard_max_loss_pct=1.0)

    # 0. Inside lockout (bars_held=1 < min_hold_bars=2): NOTHING can exit, even when ML head & hard cap want to!
    b_crash = _bar(1010, 100.0, 100.1, 95.0, 96.0)
    assert (
        evaluate_exit_ladder(
            direction=1, entry_price=100.0, take_profit_pct=3.0, stop_loss_pct=2.0,
            bar=b_crash, bars_held=1, max_holding_bars=5, min_hold_bars=2,
            exit_plan=plan, exit_model=em, exit_mode="ml_exit",
        )
        is None
    )

    # 1. After lockout, hard_loss_cap (1.0% -> 99.0) beats wider stop_loss (2.0%) and ML loss_cut!
    b_hard = _bar(1020, 99.5, 99.6, 98.8, 98.9)
    res_hard = evaluate_exit_ladder(
        direction=1, entry_price=100.0, take_profit_pct=3.0, stop_loss_pct=2.0,
        bar=b_hard, bars_held=2, max_holding_bars=5, min_hold_bars=2,
        exit_plan=plan, exit_model=em, exit_mode="ml_exit",
    )
    assert res_hard == ("hard_loss_cap", pytest.approx(99.0))

    # 2. When no deterministic barrier is touched and unrealized return is +0.5%, ML profit_exit fires!
    b_mild_gain = _bar(1020, 100.2, 100.6, 100.1, 100.5)
    res_ml_prof = evaluate_exit_ladder(
        direction=1, entry_price=100.0, take_profit_pct=3.0, stop_loss_pct=2.0,
        bar=b_mild_gain, bars_held=2, max_holding_bars=5, min_hold_bars=2,
        exit_plan=plan, exit_model=em, exit_mode="ml_exit",
    )
    assert res_ml_prof == ("profit_exit", pytest.approx(100.5))

    # 3. When unrealized return is -0.4% (above -1.0% hard cap), ML loss_cut fires!
    b_mild_loss = _bar(1020, 99.8, 99.9, 99.5, 99.6)
    res_ml_cut = evaluate_exit_ladder(
        direction=1, entry_price=100.0, take_profit_pct=3.0, stop_loss_pct=2.0,
        bar=b_mild_loss, bars_held=2, max_holding_bars=5, min_hold_bars=2,
        exit_plan=plan, exit_model=em, exit_mode="ml_exit",
    )
    assert res_ml_cut == ("loss_cut", pytest.approx(99.6))
