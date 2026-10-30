"""
Unit tests for Phase 6 T6.1 & T6.2: `ZoneRefreshPolicy` (`every_bar`, `every_n`,
`event`) and per-bar touch tracking in `DynamicStructureEngine`.
"""
import pytest

from src.layers.structure_engine import DynamicStructureEngine, ZoneRefreshPolicy
from src.state.contracts import MarketSnapshot, RegimeState, RegimeType

pytestmark = [pytest.mark.unit]


def _snap(ts: int, price: float, high_off: float = 0.5, low_off: float = 0.5, vol: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=price,
        high=price + high_off,
        low=price - low_off,
        close=price,
        volume=vol,
        bid=price - 0.01,
        ask=price + 0.01,
        bid_volume=10.0,
        ask_volume=10.0,
        spread_pct=0.02,
        hour_utc=12,
        day_of_week=2,
    )


def test_every_n_refresh_reduces_refresh_calls_while_keeping_touch_tracking_per_bar():
    pol_every = ZoneRefreshPolicy(mode="every_bar")
    pol_n5 = ZoneRefreshPolicy(mode="every_n", every_n_bars=5)

    eng_every = DynamicStructureEngine(pivot_window=2, refresh_policy=pol_every)
    eng_n5 = DynamicStructureEngine(pivot_window=2, refresh_policy=pol_n5)

    prices = [100.0, 102.0, 105.0, 102.0, 100.0, 98.0, 95.0, 98.0, 100.0, 105.0, 100.0, 95.0, 100.0, 105.0, 100.0]
    for i, px in enumerate(prices):
        s = _snap(1000 + i * 60, px, high_off=1.0, low_off=1.0, vol=200.0 if i % 4 == 0 else 50.0)
        eng_every.add_bar(s)
        eng_every.compute_zones(px)
        eng_n5.add_bar(s)
        eng_n5.compute_zones(px)

    assert eng_n5.refresh_count < eng_every.refresh_count
    # Touch tracking still occurred on non-refresh bars
    assert len(eng_n5.touch_events) > 0


def test_event_refresh_triggers_on_regime_change_and_displacement():
    pol_ev = ZoneRefreshPolicy(mode="event", on_regime_change=True, price_displacement_pct=2.0)
    eng = DynamicStructureEngine(pivot_window=2, refresh_policy=pol_ev)
    rs_up = RegimeState(timestamp=1000, symbol="BTC/USDT", primary_regime=RegimeType.TRENDING_UP, regime_confidence=0.8)
    rs_dn = RegimeState(timestamp=2000, symbol="BTC/USDT", primary_regime=RegimeType.TRENDING_DOWN, regime_confidence=0.8)

    for i in range(6):
        px = 100.0 + (2.0 if i == 2 else 0.0)
        s = _snap(1000 + i * 60, px, vol=300.0 if i == 2 else 50.0)
        eng.add_bar(s)
        eng.compute_zones(px, regime_state=rs_up)

    ref_before = eng.refresh_count
    # Bar with same regime and tiny move -> no refresh
    s_calm = _snap(1400, 100.1)
    eng.add_bar(s_calm)
    eng.compute_zones(100.1, regime_state=rs_up)
    assert eng.refresh_count == ref_before

    # Bar with regime change -> refreshes!
    s_flip = _snap(1460, 100.1)
    eng.add_bar(s_flip)
    eng.compute_zones(100.1, regime_state=rs_dn)
    assert eng.refresh_count == ref_before + 1
