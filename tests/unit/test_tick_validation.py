"""
Unit tests for Phase 13 T13.1: `validate_and_quarantine_ticks` and `TickSource`.
"""
import pytest

from src.data.ticks import QuoteRecord, TickRecord, TickSource, validate_and_quarantine_ticks

pytestmark = [pytest.mark.unit]


def test_validate_and_quarantine_ticks_quarantines_bad_records_without_crashing():
    ticks = [
        TickRecord(timestamp_ms=1000, price=100.0, size=1.5),
        TickRecord(timestamp_ms=1050, price=-5.0, size=1.0),    # bad price
        TickRecord(timestamp_ms=1100, price=100.2, size=0.0),   # bad size
        TickRecord(timestamp_ms=900, price=100.1, size=1.0),    # out-of-order timestamp
        TickRecord(timestamp_ms=1200, price=100.5, size=2.0),
    ]
    quotes = [
        QuoteRecord(timestamp_ms=1000, bid=99.9, ask=100.1, bid_size=5.0, ask_size=5.0),
        QuoteRecord(timestamp_ms=1050, bid=101.0, ask=100.0, bid_size=5.0, ask_size=5.0),  # crossed book
    ]
    res = validate_and_quarantine_ticks(ticks, quotes)
    assert len(res.valid_ticks) == 2
    assert len(res.valid_quotes) == 1
    assert len(res.quarantined) == 4

    src = TickSource(ticks, quotes, validate=True)
    tp = src.to_tick_path()
    assert len(tp.steps()) == 2
