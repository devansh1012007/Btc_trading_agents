"""
Main data pipeline orchestrator.
Coordinates: fetch -> market_data -> features -> regime -> structure -> save

Phase 2 per-bar ordering (§3 of the Phase 2 plan): regime is computed
BEFORE structure within a single bar's processing, even though the
pipeline-level architecture diagram lists Dynamic Structure before Regime
Detection -- zones need to know the current regime at the moment they're
touched (Zone.regime_relevance), so regime has to be available first:

    1. MarketDataLayer.create_from_bar()      -> MarketSnapshot
    2. FeatureEngine.add_bar()/compute()      -> FeatureVector
    3. RegimeDetector.add_bar()/compute()     -> RegimeState
    4. StructureEngine.add_bar()/compute()    -> ZoneMap
    5. StructuralFeatureAdapter merges zone-derived fields back into FeatureVector
    6. Assemble SystemState(market_snapshot, feature_vector, zone_map, regime_state)
"""
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml

from src.connectors.data_source import CCXTConnector
from src.layers.market_data import MarketDataLayer
from src.layers.feature_engine import FeatureEngine
from src.layers.regime_detector import RegimeDetector
from src.layers.structure_engine import DynamicStructureEngine
from src.layers.structural_features import attach_structural_features
from src.state.contracts import SystemState, MarketSnapshot, FeatureVector
from src.state.serialization import system_state_to_dict, dataframe_to_parquet

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/settings.yaml"


class TradingPipeline:
    """
    Main pipeline: fetch data -> market snapshots -> features -> regime ->
    structure -> full SystemState -> save.
    """

    def __init__(
        self,
        exchange_name: str = "binance",
        sandbox: bool = True,
        symbol: str = "BTC/USDT",
        data_dir: str = "data",
        config_path: str = DEFAULT_CONFIG_PATH,
    ):
        self.connector = CCXTConnector(exchange_name, sandbox)
        self.symbol = symbol
        self.data_dir = data_dir
        self.config = self._load_config(config_path)

        data_cfg = self.config.get("data", {})
        struct_cfg = self.config.get("structure", {})
        regime_cfg = self.config.get("regime", {})
        models_cfg = self.config.get("models", {})

        self.feature_engine = FeatureEngine(
            lookback_bars=data_cfg.get("lookback_bars", 100),
        )

        self.regime_detector = RegimeDetector(
            hmm_model_path=models_cfg.get("regime_hmm", "data/models/regime_hmm.pkl"),
            classifier_model_path=models_cfg.get("regime_classifier", "data/models/regime_classifier.pkl"),
            confidence_floor=regime_cfg.get("confidence_floor", 0.35),
            persistence_window_bars=regime_cfg.get("persistence_window_bars", 20),
            persistence_transition_threshold=regime_cfg.get("persistence_transition_threshold", 0.6),
            panic_z_threshold=regime_cfg.get("panic_z_threshold", 3.0),
            serving_source=models_cfg.get("regime_serving_source", "auto"),
        )

        self.structure_engine = DynamicStructureEngine(
            lookback_bars=struct_cfg.get("lookback_bars", 50),
            pivot_window=struct_cfg.get("pivot_window", 5),
            min_zone_separation_pct=struct_cfg.get("min_zone_separation_pct", 1.0),
            zone_decay_bars=struct_cfg.get("zone_decay_bars", 100),
            breakout_survival_penalty=struct_cfg.get("breakout_survival_penalty", 0.3),
            zone_survival_model_path=models_cfg.get("zone_model", "data/models/zone_validity_model.pkl"),
            zone_survival_blend_weight=models_cfg.get("zone_survival_blend_weight", 0.0),
        )

        # Create directories
        Path(f"{data_dir}/raw").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/processed").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/logs").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found at {config_path}; using defaults")
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _process_bar(self, snapshot: MarketSnapshot) -> Optional[SystemState]:
        """
        Run one bar through the full Phase 2 perception stack, in the
        order decided in §3.
        """
        # 2. Feature Engine
        self.feature_engine.add_bar(snapshot)
        fv = self.feature_engine.compute_features(snapshot)
        if fv is None:
            return None

        # 3. Regime Detector (needs only FeatureVector)
        self.regime_detector.add_bar(fv)
        regime_state = self.regime_detector.compute_regime(fv, symbol=snapshot.symbol)

        # 4. Structure Engine (needs MarketSnapshot + RegimeState)
        self.structure_engine.add_bar(snapshot)
        zone_map = self.structure_engine.compute_zones(
            current_price=snapshot.close,
            symbol=snapshot.symbol,
            regime_state=regime_state,
        )

        # 5. Structural Feature Adapter: merge zone-derived fields back into fv
        fv = attach_structural_features(fv, zone_map)

        # 6. Assemble full SystemState
        state = SystemState(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            market_snapshot=snapshot,
            feature_vector=fv,
            zone_map=zone_map,
            regime_state=regime_state,
        )
        return state

    def fetch_and_process_historical(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical data, compute features/regime/structure, save to Parquet.

        Args:
            days: Number of days to fetch
            limit: Bars per request (CCXT limit)

        Returns:
            DataFrame with the fully assembled SystemState fields
        """
        logger.info(f"Fetching {days} days of {self.symbol} data...")

        df_ohlcv = self.connector.fetch_ohlcv(self.symbol, timeframe="1m", limit=limit)

        if df_ohlcv.empty:
            logger.error("No data fetched")
            return pd.DataFrame()

        self.connector.save_ohlcv_local(df_ohlcv, self.symbol, f"{self.data_dir}/raw")

        snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, self.symbol)

        system_states = []
        for snapshot in snapshots:
            state = self._process_bar(snapshot)
            if state is not None:
                system_states.append(state)

        logger.info(f"Created {len(system_states)} states with features/regime/structure")

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
            SystemState with features/regime/structure computed
        """
        return self._process_bar(snapshot)


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)

    pipeline = TradingPipeline(sandbox=True, symbol="BTC/USDT")
    df = pipeline.fetch_and_process_historical(days=1, limit=100)

    print(f"\nProcessed {len(df)} bars")
    print(df.head())
