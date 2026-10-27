"""
Main data pipeline orchestrator.
Coordinates: fetch -> market_data -> features -> regime -> structure ->
experts -> opportunity filter -> router -> save

Phase 2 per-bar ordering (§3): regime is computed BEFORE structure within
a single bar's processing -- zones need to know the current regime at the
moment they're touched (Zone.regime_relevance).

Phase 3 extends this to the Decision Layer, after structural features are
merged back into the FeatureVector:

    1. MarketDataLayer.create_from_bar()      -> MarketSnapshot
    2. FeatureEngine.add_bar()/compute()      -> FeatureVector
    3. RegimeDetector.add_bar()/compute()     -> RegimeState
    4. StructureEngine.add_bar()/compute()    -> ZoneMap
    5. StructuralFeatureAdapter merges zone-derived fields back into FeatureVector
    6. ExpertLayer.evaluate()                 -> ExpertOutputs
    7. OpportunityFilter.evaluate()           -> OpportunityScore
    8. Router.decide()                        -> RouterDecision
    9. Assemble full SystemState

Bar granularity (Phase 3 §3, §4): read from system.data_granularity in
settings.yaml and threaded through to every layer's model-loading guard
(§4.7) and the historical-fetch timeframe.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from src.connectors.data_source import CCXTConnector
from src.experts.expert_registry import build_experts
from src.labeling.hold_policy import alignment_settings
from src.layers.expert_layer import ExpertLayer
from src.layers.feature_engine import FeatureEngine
from src.layers.market_data import MarketDataLayer
from src.layers.opportunity_filter import OpportunityFilter
from src.layers.regime_detector import RegimeDetector
from src.layers.router import Router
from src.layers.structural_features import attach_structural_features
from src.layers.structure_engine import DynamicStructureEngine
from src.liquidity.synthetic_liquidity_model import attach_synthetic_liquidity
from src.state.contracts import MarketSnapshot, SystemState
from src.state.serialization import system_state_to_dict

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/settings.base.yaml"


class TradingPipeline:
    """
    Main pipeline: fetch data -> market snapshots -> features -> regime ->
    structure -> experts -> opportunity filter -> router -> full
    SystemState -> save.
    """

    def __init__(
        self,
        exchange_name: str = "binance",
        sandbox: bool = True,
        symbol: str = "BTC/USDT",
        data_dir: str = "data",
        config_path: str = DEFAULT_CONFIG_PATH,
        config: Optional[dict] = None,
    ):
        """
        Args:
            config: an already-loaded config dict (e.g. from
                src.utils.config_helpers.load_layered_config(env)) takes
                precedence over config_path when provided -- this is how
                scripts/run.py injects the env-merged (paper/live)
                config, since reassigning self.config AFTER construction
                would silently miss every layer/sub-component built
                during __init__ from the un-merged config.
        """
        self.symbol = symbol
        self.data_dir = data_dir
        self.config = config if config is not None else self._load_config(config_path)

        system_cfg = self.config.get("system", {})
        data_cfg = self.config.get("data", {})
        struct_cfg = self.config.get("structure", {})
        regime_cfg = self.config.get("regime", {})
        models_cfg = self.config.get("models", {})
        opp_cfg = self.config.get("opportunity_filter", {})
        router_cfg = self.config.get("router", {})
        labeling_cfg = self.config.get("labeling", {})
        self.liquidity_cfg = self.config.get("liquidity", {}).get("synthetic", {})

        self.data_granularity = system_cfg.get("data_granularity", "1m")
        self.connector = CCXTConnector(
            system_cfg.get("exchange", exchange_name), system_cfg.get("sandbox", sandbox)
        )

        self.feature_engine = FeatureEngine(lookback_bars=data_cfg.get("lookback_bars", 100), config=self.config)

        self.regime_detector = RegimeDetector(
            hmm_model_path=models_cfg.get("regime_hmm", "data/models/regime_hmm.pkl"),
            classifier_model_path=models_cfg.get("regime_classifier", "data/models/regime_classifier.pkl"),
            confidence_floor=regime_cfg.get("confidence_floor", 0.35),
            persistence_window_bars=regime_cfg.get("persistence_window_bars", 20),
            persistence_transition_threshold=regime_cfg.get("persistence_transition_threshold", 0.6),
            panic_z_threshold=regime_cfg.get("panic_z_threshold", 3.0),
            serving_source=models_cfg.get("regime_serving_source", "auto"),
            data_granularity=self.data_granularity,
        )

        self.structure_engine = DynamicStructureEngine(
            lookback_bars=struct_cfg.get("lookback_bars", 50),
            pivot_window=struct_cfg.get("pivot_window", 5),
            min_zone_separation_pct=struct_cfg.get("min_zone_separation_pct", 1.0),
            zone_decay_bars=struct_cfg.get("zone_decay_bars", 100),
            breakout_survival_penalty=struct_cfg.get("breakout_survival_penalty", 0.3),
            zone_survival_model_path=models_cfg.get("zone_model", "data/models/zone_validity_model.pkl"),
            zone_survival_blend_weight=models_cfg.get("zone_survival_blend_weight", 0.0),
            data_granularity=self.data_granularity,
        )

        # Fails here -- not per bar -- if the minimum-trade-duration policy
        # cannot be satisfied by this config (see src/labeling/hold_policy.py).
        hold_policy, require_aligned = alignment_settings(self.config)
        experts = build_experts(self.config, data_granularity=self.data_granularity)
        self.expert_layer = ExpertLayer(experts)

        self.opportunity_filter = OpportunityFilter(
            config=opp_cfg, labeling_config=labeling_cfg,
            model_path=opp_cfg.get("ml_model", {}).get("model_path"),
            data_granularity=self.data_granularity,
            hold_policy=hold_policy, require_aligned_models=require_aligned,
        )

        self.router = Router(
            config=router_cfg, model_path=router_cfg.get("model", {}).get("model_path"),
            data_granularity=self.data_granularity,
            hold_policy=hold_policy, require_aligned_models=require_aligned,
        )

        Path(f"{data_dir}/raw").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/processed").mkdir(parents=True, exist_ok=True)
        Path(f"{data_dir}/logs").mkdir(parents=True, exist_ok=True)

        # Per-stage latency instrumentation. Populated by every
        # _process_bar call; read via the `last_bar_timings` property.
        self._timings: Dict[str, float] = {}
        self._perf = time.perf_counter

    @property
    def last_bar_timings(self) -> Dict[str, float]:
        """Wall-clock milliseconds per pipeline stage for the most recent bar."""
        return dict(self._timings)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found at {config_path}; using defaults")
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _process_bar(self, snapshot: MarketSnapshot) -> Optional[SystemState]:
        # Per-stage wall-clock timing: written to self._timings every bar;
        # consumers (decision traces, latency benchmarks) read it after
        # run_live_bar returns. Reset per bar so a warm-up bar (which
        # returns early) cannot leave the PREVIOUS bar's numbers next to
        # this bar's.
        self._timings = {}
        t = self._perf()
        self.feature_engine.add_bar(snapshot)
        fv = self.feature_engine.compute_features(snapshot)
        self._timings["features_ms"] = (self._perf() - t) * 1000
        if fv is None:
            return None

        t = self._perf()
        self.regime_detector.add_bar(fv)
        regime_state = self.regime_detector.compute_regime(fv, symbol=snapshot.symbol)
        self._timings["regime_ms"] = (self._perf() - t) * 1000

        t = self._perf()
        self.structure_engine.add_bar(snapshot)
        zone_map = self.structure_engine.compute_zones(
            current_price=snapshot.close, symbol=snapshot.symbol, regime_state=regime_state,
        )
        self._timings["structure_ms"] = (self._perf() - t) * 1000

        t = self._perf()
        fv = attach_structural_features(fv, zone_map)

        # Phase 4 §4.2 / Finding #2 fix -- see replay.py's identical
        # comment for the full rationale and the two-pass design note.
        snapshot = attach_synthetic_liquidity(snapshot, fv, self.liquidity_cfg)
        self._timings["liquidity_ms"] = (self._perf() - t) * 1000

        t = self._perf()
        expert_outputs = self.expert_layer.evaluate(fv, zone_map, regime_state)
        self._timings["expert_layer_ms"] = (self._perf() - t) * 1000

        t = self._perf()
        opportunity_score = self.opportunity_filter.evaluate(expert_outputs, regime_state, zone_map, snapshot, fv)
        self._timings["opportunity_filter_ms"] = (self._perf() - t) * 1000

        t = self._perf()
        router_decision = self.router.decide(expert_outputs, regime_state, zone_map, opportunity_score)
        self._timings["router_ms"] = (self._perf() - t) * 1000
        self._timings["decisions_ms"] = (
            self._timings["expert_layer_ms"]
            + self._timings["opportunity_filter_ms"]
            + self._timings["router_ms"]
        )

        return SystemState(
            timestamp=snapshot.timestamp, symbol=snapshot.symbol, market_snapshot=snapshot,
            feature_vector=fv, zone_map=zone_map, regime_state=regime_state,
            expert_outputs=expert_outputs, opportunity_score=opportunity_score, router_decision=router_decision,
        )

    def fetch_and_process_historical(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        logger.info(f"Fetching {days} days of {self.symbol} data at {self.data_granularity} granularity...")

        df_ohlcv = self.connector.fetch_ohlcv(self.symbol, timeframe=self.data_granularity, limit=limit)
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

        logger.info(f"Created {len(system_states)} states with features/regime/structure/decisions")

        records = [system_state_to_dict(s) for s in system_states]
        df_processed = pd.DataFrame(records)

        output_path = f"{self.data_dir}/processed/{self.symbol.replace('/', '_')}_features.parquet"
        df_processed.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(df_processed)} rows to {output_path}")

        return df_processed

    def run_live_bar(self, snapshot: MarketSnapshot) -> Optional[SystemState]:
        return self._process_bar(snapshot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TradingPipeline(sandbox=True, symbol="BTC/USDT")
    df = pipeline.fetch_and_process_historical(days=1, limit=100)
    print(f"\nProcessed {len(df)} bars")
    print(df.head())
