"""
Market Data Layer: Convert raw exchange data to MarketSnapshot.
This layer is NOT predictive. It just represents current market state.
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from ..state.contracts import MarketSnapshot

logger = logging.getLogger(__name__)


class MarketDataLayer:
    """Converts raw OHLCV + ticker into MarketSnapshot objects."""

    @staticmethod
    def create_from_bar(
        symbol: str,
        timestamp_unix: int,
        ohlcv_row: Dict[str, float],
        ticker_data: Optional[Dict] = None,
    ) -> MarketSnapshot:
        """
        Create MarketSnapshot from a single bar + ticker.

        Args:
            symbol: Trading pair
            timestamp_unix: Unix timestamp (seconds)
            ohlcv_row: dict with 'open', 'high', 'low', 'close', 'volume'
            ticker_data: dict with 'bid', 'ask', 'bid_volume', 'ask_volume' (optional)

        Returns:
            MarketSnapshot object
        """
        # Unpack OHLCV
        open_price = ohlcv_row.get('open', 0.0)
        high_price = ohlcv_row.get('high', 0.0)
        low_price = ohlcv_row.get('low', 0.0)
        close_price = ohlcv_row.get('close', 0.0)
        volume = ohlcv_row.get('volume', 0.0)

        # Bid/ask (use close if no ticker)
        if ticker_data:
            bid = ticker_data.get('bid', close_price)
            ask = ticker_data.get('ask', close_price)
            bid_vol = ticker_data.get('bid_volume', 0.0)
            ask_vol = ticker_data.get('ask_volume', 0.0)
        else:
            mid = (high_price + low_price) / 2
            bid = mid * 0.999
            ask = mid * 1.001
            bid_vol = 0.0
            ask_vol = 0.0

        # Calculate spread
        mid_price = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid_price * 100) if mid_price != 0 else 0.0

        # Session info
        dt = datetime.utcfromtimestamp(timestamp_unix)
        hour_utc = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday

        snapshot = MarketSnapshot(
            timestamp=timestamp_unix,
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            bid=bid,
            ask=ask,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            spread_pct=spread_pct,
            funding_rate=ticker_data.get('funding_rate') if ticker_data else None,
            hour_utc=hour_utc,
            day_of_week=day_of_week,
        )

        return snapshot

    @staticmethod
    def batch_create_from_dataframe(
        df: pd.DataFrame,
        symbol: str,
    ) -> list:
        """
        Create MarketSnapshots from a DataFrame.

        Args:
            df: DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
            symbol: Trading pair

        Returns:
            List of MarketSnapshot objects
        """
        snapshots = []

        for _, row in df.iterrows():
            # Convert timestamp
            if isinstance(row['timestamp'], pd.Timestamp):
                ts = int(row['timestamp'].timestamp())
            else:
                ts = int(row['timestamp'])

            # Create snapshot
            ohlcv = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
            }

            snapshot = MarketDataLayer.create_from_bar(
                symbol=symbol,
                timestamp_unix=ts,
                ohlcv_row=ohlcv,
            )
            snapshots.append(snapshot)

        logger.info(f"Created {len(snapshots)} MarketSnapshots for {symbol}")
        return snapshots
