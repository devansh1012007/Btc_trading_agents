"""
Main data pipeline orchestrator.
Coordinates: fetch → market_data → features → save
"""
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.connectors.data_source import CCXTConnector
from src.layers.market_data import MarketDataLayer
from src.layers.feature_engine import FeatureEngine
from src.state.contracts import SystemState, MarketSnapshot, FeatureVector
from src.state.serialization import system_state_to_dict, dataframe_to_parquet

logger = logging.getLogger(__name__)


class TradingPipeline:
    """
    Main pipeline: fetch data → create market snapshots → compute features → save.
    """
    
    def __init__(
        self,
        exchange_name: str = "binance",
        sandbox: bool = True,
        symbol: str = "BTC/USDT",
        data_dir: str = "data",
    ):
        self.connector = CCXTConnector(exchange_name, sandbox)
        self.symbol = symbol
        self.data_dir = data_dir
        self.feature_engine = FeatureEngine(lookback_bars=100)
        
        # Create directories
        Path(f"{data_dir}/raw").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/processed").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/logs").mkdir(parents=True, exist_ok=True)
    
    def fetch_and_process_historical(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical data, compute features, save to Parquet.
        
        Args:
            days: Number of days to fetch
            limit: Bars per request (CCXT limit)
        
        Returns:
            DataFrame with computed features
        """
        logger.info(f"Fetching {days} days of {self.symbol} data...")
        
        # Fetch OHLCV
        df_ohlcv = self.connector.fetch_ohlcv(self.symbol, timeframe="1m", limit=limit)
        
        if df_ohlcv.empty:
            logger.error("No data fetched")
            return pd.DataFrame()
        
        # Save raw data locally
        self.connector.save_ohlcv_local(df_ohlcv, self.symbol, f"{self.data_dir}/raw")
        
        # Create MarketSnapshots
        snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, self.symbol)
        
        # Compute features
        system_states = []
        for snapshot in snapshots:
            self.feature_engine.add_bar(snapshot)
            fv = self.feature_engine.compute_features(snapshot)
            
            if fv:
                state = SystemState(
                    timestamp=snapshot.timestamp,
                    symbol=snapshot.symbol,
                    market_snapshot=snapshot,
                    feature_vector=fv,
                )
                system_states.append(state)
        
        logger.info(f"Created {len(system_states)} states with features")
        
        # Save to Parquet
        records = [system_state_to_dict(s) for s in system_states]
        df_processed = pd.DataFrame(records)
        
        output_path = f"{self.data_dir}/processed/{self.symbol.replace('/', '_')}_features.parquet"
        df_processed.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(df_processed)} rows to {output_path}")
        
        return df_processed
    
    def run_live_bar(self, snapshot: MarketSnapshot) -> Optional[SystemState]:
        """
        Process a single new bar live.
        
        Args:
            snapshot: Current market snapshot
        
        Returns:
            SystemState with computed features
        """
        self.feature_engine.add_bar(snapshot)
        fv = self.feature_engine.compute_features(snapshot)
        
        if fv:
            state = SystemState(
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                market_snapshot=snapshot,
                feature_vector=fv,
            )
            return state
        
        return None


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = TradingPipeline(sandbox=True, symbol="BTC/USDT")
    df = pipeline.fetch_and_process_historical(days=1, limit=100)
    
    print(f"\nProcessed {len(df)} bars")
    print(df.head())
