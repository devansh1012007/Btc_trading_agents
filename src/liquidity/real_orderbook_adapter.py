"""
Real order-book adapter (Phase 4, §4.3) -- used in paper-live-data and
live modes, where a genuine CCXT order book is available and there's no
reason to estimate what you can just ask the exchange for.
"""
from dataclasses import replace
import logging

from ..state.contracts import MarketSnapshot
from .types import LiquidityEstimate

logger = logging.getLogger(__name__)


def fetch_real_liquidity(connector, symbol: str, depth_limit: int = 10) -> LiquidityEstimate:
    """
    connector: a CCXTConnector (src/connectors/data_source.py). Uses its
    underlying ccxt exchange instance directly since fetch_order_book
    isn't wrapped there yet -- Phase 1's connector only wraps OHLCV/ticker/
    funding-rate fetches.
    """
    book = connector.exchange.fetch_order_book(symbol, limit=depth_limit)
    best_bid, _ = book["bids"][0]
    best_ask, _ = book["asks"][0]
    mid = (best_bid + best_ask) / 2
    spread_pct = ((best_ask - best_bid) / mid * 100) if mid else 0.0

    depth_bid = sum(v for _, v in book["bids"][:depth_limit])
    depth_ask = sum(v for _, v in book["asks"][:depth_limit])

    return LiquidityEstimate(spread_pct=spread_pct, bid_volume=depth_bid, ask_volume=depth_ask, source="real_order_book")


def attach_real_liquidity(bar: MarketSnapshot, connector, symbol: str, depth_limit: int = 10) -> MarketSnapshot:
    """Return a new MarketSnapshot with liquidity fields from a live order
    book. Falls back to the bar unchanged (still "synthetic"-tagged) if
    the fetch fails, rather than raising and halting the whole loop over
    a transient network blip."""
    try:
        estimate = fetch_real_liquidity(connector, symbol, depth_limit)
    except Exception as e:
        logger.warning(f"Real order-book fetch failed for {symbol}, keeping prior liquidity fields: {e}")
        return bar

    mid_price = bar.close
    half_spread_frac = (estimate.spread_pct / 100) / 2
    bid = mid_price * (1 - half_spread_frac)
    ask = mid_price * (1 + half_spread_frac)

    return replace(
        bar,
        bid=bid, ask=ask,
        bid_volume=estimate.bid_volume, ask_volume=estimate.ask_volume,
        spread_pct=estimate.spread_pct,
        liquidity_source="real_order_book",
    )
