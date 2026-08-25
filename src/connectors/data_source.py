"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""

    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        self.exchange_name = exchange_name
        self.sandbox = sandbox

        try:
            exchange_class = getattr(ccxt, exchange_name)
        except AttributeError as e:
            logger.error(f"Failed to get exchange_class {e}")
            raise
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })

        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")

    def fetch_ohlcv(self,symbol: str = "BTC/USDT",timeframe: str = "1m",limit: int = 1000,) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetched {len(df)} {timeframe} bars for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise

    def fetch_ticker(self, symbol: str = "BTC/USDT") -> Dict:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'bid_volume': ticker.get('bidVolume'),
                'ask_volume': ticker.get('askVolume'),
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker {symbol}: {e}")
            raise

    def fetch_funding_rate(self, symbol: str = "BTC/USDT") -> Optional[float]:
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                fr = self.exchange.fetch_funding_rate(symbol)
                return fr.get('fundingRate')
        except Exception as e:
            logger.warning(f"Could not fetch funding rate: {e}")
        return None

    def save_ohlcv_local(self, df: pd.DataFrame, symbol: str, data_dir: str = "data/raw"):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df['date'] = df['timestamp'].dt.date
        for date, group in df.groupby('date'):
            path = f"{data_dir}/{symbol.replace('/', '_')}_{date}.parquet"
            group.to_parquet(path, compression='snappy')
            logger.info(f"Saved {len(group)} rows to {path}")

    def load_ohlcv_local(self, symbol: str, date_range: Tuple, data_dir: str = "data/raw") -> pd.DataFrame:
        symbol_str = symbol.replace('/', '_')
        files = sorted(Path(data_dir).glob(f"{symbol_str}_*.parquet"))

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")

        if not dfs:
            logger.warning(f"No local data found for {symbol}")
            return pd.DataFrame()

        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values('timestamp')
        logger.info(f"Loaded {len(df_all)} local bars for {symbol}")
        return df_all
