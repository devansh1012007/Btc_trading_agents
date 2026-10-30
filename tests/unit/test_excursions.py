"""
Unit tests for Phase 7 T7.1: `compute_forward_excursions`.
"""
import pytest

from src.labeling.excursions import compute_forward_excursions
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


def test_compute_forward_excursions_long_and_short_and_lockout():
    bars = [
        _bar(1010, 100.0, 100.5, 98.5, 99.0),   # bar 1 (inside lockout min_hold_bars=2): adverse=1.5%, fav=0.5%
        _bar(1020, 99.0, 103.0, 98.0, 102.5),   # bar 2: fav=3.0%, adverse=2.0%
    ]
    ex_long = compute_forward_excursions(100.0, 1, bars, max_holding_bars=5, min_hold_bars=2, volatility_pct=0.5)
    assert ex_long.mfe_pct == pytest.approx(3.0)
    assert ex_long.mae_pct == pytest.approx(2.0)
    assert ex_long.lockout_mae_pct == pytest.approx(1.5)
    assert ex_long.k_mfe == pytest.approx(6.0)
    assert ex_long.k_mae == pytest.approx(4.0)
    assert ex_long.k_lockout_mae == pytest.approx(3.0)
