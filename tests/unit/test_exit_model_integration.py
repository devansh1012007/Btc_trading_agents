"""
Unit tests for Phase 10 T10.3 & T10.4: `ExitModel.fit` and `simulate_with_exit_ladder`
paired outcomes (`hard_limits` vs `ml_exit`).
"""
import numpy as np
import pytest

from src.exits.exit_model import (
    EXIT_STATE_FEATURES,
    ExitModel,
    simulate_with_exit_ladder,
)
from src.state.contracts import MarketSnapshot

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


def test_exit_model_fits_and_produces_paired_router_outcomes():
    rng = np.random.default_rng(19)
    rows = []
    for i in range(40):
        r = {f: float(rng.normal()) for f in EXIT_STATE_FEATURES}
        r["unrealized_return_pct"] = 0.4 if i < 20 else -0.4
        r["y_profit_exit"] = 1 if i % 2 == 0 else 0
        r["y_loss_cut"] = 1 if i % 2 == 0 else 0
        rows.append(r)

    em = ExitModel("trend", profit_threshold=0.3, loss_cut_threshold=0.3).fit(rows)
    assert em.profit_head is not None
    assert em.loss_cut_head is not None

    bars = [
        _bar(1010, 100.0, 100.6, 99.9, 100.5),
        _bar(1020, 100.5, 100.6, 99.2, 99.3),
        _bar(1030, 99.3, 99.4, 99.0, 99.1),
    ]
    r_hard, held_hard, ret_hard = simulate_with_exit_ladder(
        bars, entry_price=100.0, direction=1, take_profit_pct=2.0, stop_loss_pct=2.0,
        max_holding_bars=3, exit_model=em, exit_mode="hard_limits",
    )
    r_ml, held_ml, ret_ml = simulate_with_exit_ladder(
        bars, entry_price=100.0, direction=1, take_profit_pct=2.0, stop_loss_pct=2.0,
        max_holding_bars=3, exit_model=em, exit_mode="ml_exit",
    )
    assert r_hard == "timeout"
    assert r_ml == "profit_exit"
    assert held_ml == 1 and ret_ml > ret_hard
