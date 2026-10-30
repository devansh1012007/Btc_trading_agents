"""
Unit tests for Phase 13 T13.2: `BarBuilder` 1s aggregation, real quote spread,
carry-forward on empty seconds, and strict bucket causality.
"""
import pytest

from src.data.bar_builder import BarBuilder
from src.data.ticks import QuoteRecord, TickRecord

pytestmark = [pytest.mark.unit]


def test_bar_builder_aggregates_1s_bars_with_carry_forward_and_real_quotes():
    ticks = [
        TickRecord(timestamp_ms=1_000_100, price=100.0, size=1.0),
        TickRecord(timestamp_ms=1_000_500, price=102.0, size=2.0),
        TickRecord(timestamp_ms=1_000_900, price=101.0, size=1.5),
        # No ticks in second 1001 -> carried forward at 101.0 with volume=0.0
        TickRecord(timestamp_ms=1_002_200, price=103.0, size=0.5),
    ]
    quotes = [
        QuoteRecord(timestamp_ms=1_000_200, bid=100.9, ask=101.1, bid_size=10.0, ask_size=12.0),
    ]
    bb = BarBuilder(symbol="BTC/USDT", bar_seconds=1, carry_forward_empty=True)
    bars = bb.build_bars(ticks, quotes)
    assert len(bars) == 3

    b0, b1, b2 = bars
    assert b0.timestamp == 1000
    assert (b0.open, b0.high, b0.low, b0.close) == (100.0, 102.0, 100.0, 101.0)
    assert b0.volume == pytest.approx(4.5)
    assert b0.liquidity_source == "real_orderbook"
    assert b0.bid == pytest.approx(100.9) and b0.ask == pytest.approx(101.1)

    # Empty second 1001 carried forward from b0.close (101.0) with volume=0.0, never peeking at b2 (103.0)
    assert b1.timestamp == 1001
    assert (b1.open, b1.high, b1.low, b1.close, b1.volume) == (101.0, 101.0, 101.0, 101.0, 0.0)

    assert b2.timestamp == 1002
    assert b2.close == pytest.approx(103.0)
