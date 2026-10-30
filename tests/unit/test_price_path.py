"""
Unit tests for Phase 3 T3.0b: `PricePath`, `BarPath`, and `TickPath`.
"""
import pytest

from src.execution.price_path import BarPath, PathStep, TickPath
from src.state.contracts import MarketSnapshot

pytestmark = [pytest.mark.unit]


def _snap(ts: int, o: float, h: float, low: float, c: float, spread: float = 0.02) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=o,
        high=h,
        low=low,
        close=c,
        volume=10.0,
        bid=c * (1 - spread / 200),
        ask=c * (1 + spread / 200),
        bid_volume=5.0,
        ask_volume=5.0,
        spread_pct=spread,
        hour_utc=12,
        day_of_week=2,
    )


def test_bar_path_interpolates_sub_bar_delay():
    bars = [
        _snap(1000, 100.0, 101.0, 99.0, 100.0, spread=0.02),
        _snap(1010, 105.0, 106.0, 104.0, 105.0, spread=0.04),
    ]
    bp = BarPath(bars, granularity_seconds=10.0)
    mid0, sp0, _ = bp.price_at(1000, 0.0)
    assert mid0 == pytest.approx(100.0)
    assert sp0 == pytest.approx(0.02)

    # 2s delay on 10s bar -> alpha = 0.2 -> 0.8*100 + 0.2*105 = 101.0
    mid2, sp2, _ = bp.price_at(1000, 2.0)
    assert mid2 == pytest.approx(101.0)
    assert sp2 == pytest.approx(0.024)


def test_tick_path_exact_lookup_and_first_passage():
    steps = [
        PathStep(timestamp=1000.0, open=100.0, high=100.1, low=99.9, close=100.0, bid=99.99, ask=100.01, spread_pct=0.02),
        PathStep(timestamp=1001.0, open=100.2, high=100.5, low=100.1, close=100.4, bid=100.39, ask=100.41, spread_pct=0.02),
        PathStep(timestamp=1002.0, open=100.4, high=102.2, low=100.3, close=102.0, bid=101.99, ask=102.01, spread_pct=0.02),
        PathStep(timestamp=1003.0, open=102.0, high=102.0, low=98.5, close=98.8, bid=98.79, ask=98.81, spread_pct=0.02),
    ]
    tp = TickPath(steps)
    mid, sp, step = tp.price_at(1000.0, 2.0)
    assert step.timestamp == pytest.approx(1002.0)
    assert mid == pytest.approx(102.0)
    assert sp == pytest.approx(0.02)

    # At tick resolution, TP (102.0) is crossed at t=1002 BEFORE SL (99.0) at t=1003!
    passage = tp.first_passage(1000.0, 1003.0, direction=1, tp_level=102.0, sl_level=99.0)
    assert passage == ("take_profit", 102.0, 1002.0)
