"""
Unit tests for Phase 8 T8.2 & T8.3: `PositionBook`, concurrency/exposure/stacking/hedging
limits, deterministic lane assignment, MTM equity, and `process_bar` multi-position execution.
"""
import pytest

from src.execution.position_book import PositionBook, PositionLimits
from src.execution.position_lifecycle import OpenPosition

pytestmark = [pytest.mark.unit]


def _pos(tid: str, ts: int, expert: str = "trend", regime: str = "trending_up", direction: int = 1, size_usdt: float = 1000.0, entry: float = 100.0) -> OpenPosition:
    return OpenPosition(
        trade_id=tid,
        entry_timestamp=ts,
        entry_price=entry,
        direction=direction,
        size_usdt=size_usdt,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
        expert=expert,
        regime=regime,
    )


def test_position_book_enforces_concurrency_stacking_hedging_and_exposure():
    limits = PositionLimits(
        max_concurrent=3,
        max_per_expert=1,
        max_per_regime=2,
        max_per_direction=2,
        allow_same_expert_stacking=False,
        allow_hedging=False,
        max_gross_exposure_pct_of_equity=25.0,
        max_net_exposure_pct_of_equity=20.0,
    )
    book = PositionBook(limits=limits)
    p1 = _pos("t1", 1000, expert="trend", regime="trending_up", direction=1, size_usdt=1000.0)
    lane1 = book.add(p1)
    assert lane1 == 0

    # Same expert rejected when allow_same_expert_stacking=False
    ok, stage, reason = book.can_open(expert="trend", regime="trending_up", direction=1, size_usdt=500.0, equity=10_000.0)
    assert not ok and stage == "concurrency_limit" and "same_expert_stacking_disabled" in reason

    # Opposite direction rejected when allow_hedging=False
    ok, stage, reason = book.can_open(expert="mean_rev", regime="chop_range", direction=-1, size_usdt=500.0, equity=10_000.0)
    assert not ok and stage == "concurrency_limit" and "hedging_not_allowed" in reason

    # Exposure limit: 1000 + 1500 = 2500 -> 25% gross (allowed), but net is 25% > 20% max_net!
    ok, stage, reason = book.can_open(expert="breakout", regime="trending_up", direction=1, size_usdt=1500.0, equity=10_000.0)
    assert not ok and stage == "exposure_limit" and "net_exposure" in reason

    # 800 size -> 1800 total (18% <= 20%) -> allowed!
    ok, stage, _ = book.can_open(expert="breakout", regime="trending_up", direction=1, size_usdt=800.0, equity=10_000.0)
    assert ok and stage is None
    p2 = _pos("t2", 1010, expert="breakout", regime="trending_up", direction=1, size_usdt=800.0)
    lane2 = book.add(p2)
    assert lane2 == 1

    # Remove p1 (lane 0), next added position reuses lane 0!
    book.remove(p1)
    p3 = _pos("t3", 1020, expert="mean_rev", regime="chop_range", direction=1, size_usdt=500.0)
    assert book.add(p3) == 0


def test_mark_to_market_equity():
    book = PositionBook()
    book.add(_pos("t1", 1000, direction=1, size_usdt=1000.0, entry=100.0))
    book.add(_pos("t2", 1010, direction=-1, size_usdt=500.0, entry=100.0))
    # At close=102 (+2%): p1 gains +$20, p2 loses -$10 -> net unrealized = +$10
    assert book.unrealized_pnl_usdt(102.0) == pytest.approx(10.0)
    assert book.mark_to_market_equity(10_000.0, 102.0) == pytest.approx(10_010.0)
