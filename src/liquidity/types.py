"""
Shared liquidity types (Phase 4, §4). LiquidityEstimate is deliberately
NOT part of contracts.py / SystemState -- it's an intermediate object
consumed immediately to build a MarketSnapshot's bid/ask/bid_volume/
ask_volume/liquidity_source fields, not persisted state in its own right.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityEstimate:
    spread_pct: float
    bid_volume: float
    ask_volume: float
    source: str  # "synthetic" | "real_order_book"


def to_ticker_data(estimate: LiquidityEstimate, mid_price: float) -> dict:
    """
    Convert a LiquidityEstimate into the ticker_data dict shape
    MarketDataLayer.create_from_bar expects, deriving symmetric bid/ask
    around mid_price from spread_pct.
    """
    half_spread_frac = (estimate.spread_pct / 100) / 2
    return {
        "bid": mid_price * (1 - half_spread_frac),
        "ask": mid_price * (1 + half_spread_frac),
        "bid_volume": estimate.bid_volume,
        "ask_volume": estimate.ask_volume,
    }
