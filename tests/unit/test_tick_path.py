"""
Unit tests for Phase 13 T13.3: sub-second fill lookup and barrier passage on `TickPath`
vs coarse `BarPath`.
"""
import pytest

from src.data.ticks import QuoteRecord, TickRecord, TickSource
from src.execution.cost_model import CostModel
from src.execution.execution_engine import exit_fill_price
from src.state.contracts import MarketSnapshot

pytestmark = [pytest.mark.unit]


def test_tick_path_exact_2s_exit_delay_fill():
    ticks = [
        TickRecord(timestamp_ms=1_000_000, price=100.0, size=1.0),
        TickRecord(timestamp_ms=1_001_000, price=100.5, size=1.0),
        TickRecord(timestamp_ms=1_002_000, price=101.2, size=1.0),
    ]
    quotes = [
        QuoteRecord(timestamp_ms=1_000_000, bid=99.99, ask=100.01, bid_size=10.0, ask_size=10.0),
        QuoteRecord(timestamp_ms=1_002_000, bid=101.18, ask=101.22, bid_size=10.0, ask_size=10.0),
    ]
    tp = TickSource(ticks, quotes).to_tick_path()
    cm = CostModel.from_config({
        "costs": {
            "spread": {"source": "snapshot"},
            "slippage": {"model": "fixed", "fixed_pct": 0.0},
            "latency": {"exit_delay_seconds": 2.0},
        }
    })
    bar = MarketSnapshot(
        timestamp=1000, symbol="BTC/USDT", open=100.0, high=101.5, low=99.8, close=100.0,
        volume=3.0, bid=99.99, ask=100.01, bid_volume=10.0, ask_volume=10.0, spread_pct=0.02,
    )
    fill_px = exit_fill_price(
        direction=1, trigger_price=100.0, bar=bar, exit_reason="take_profit",
        cost_model=cm, exit_model="realistic_fill", price_path=tp,
    )
    # At t=1000 + 2.0s = 1002.0s, mid is 101.20, spread is ~0.0395% -> long SELL fills just below 101.20
    assert 101.15 < fill_px <= 101.20
