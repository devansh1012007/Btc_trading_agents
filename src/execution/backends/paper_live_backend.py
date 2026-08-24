"""
Paper-Live-Data Backend (Phase 4, §6.3): uses a REAL order book for
liquidity (via the real order-book adapter), but never calls the
exchange's order-placement endpoint -- fills are still simulated, just
against the real current book instead of the synthetic proxy. This is a
meaningfully better validation step before ever touching real capital,
and costs nothing extra since the real-time connection already exists
for regime/feature computation in this mode.

Phase 5, §6.4: Logs paired (synthetic, real) liquidity samples to a
Parquet file for calibrate_synthetic_liquidity.py.
"""
import logging
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ...state.contracts import RiskDecision, RouterDecision, ExpertOutput, MarketSnapshot, ExecutionRecord, ExpertAction, FeatureVector
from ...liquidity.types import LiquidityEstimate
from ...liquidity.real_orderbook_adapter import fetch_real_liquidity
from ...liquidity.synthetic_liquidity_model import estimate_synthetic_liquidity
from ..execution_engine import ExecutionBackend, simulate_fill, make_client_order_id

logger = logging.getLogger(__name__)


class PaperLiveBackend(ExecutionBackend):
    def __init__(self, config: dict, connector, symbol: str, fee_pct: float = 0.04,
                 sample_log_path: Optional[str] = None):
        self.config = config
        self.connector = connector
        self.symbol = symbol
        self.fee_pct = fee_pct
        self.sample_log_path = sample_log_path
        self._liq_cfg = config.get("liquidity", {}).get("synthetic", {})
        self._sample_buffer: list = []

    def get_liquidity(self, symbol: str, market: MarketSnapshot, fv: Optional[FeatureVector] = None) -> LiquidityEstimate:
        try:
            return fetch_real_liquidity(self.connector, symbol)
        except Exception as e:
            logger.warning(f"Real order-book fetch failed, falling back to snapshot's existing liquidity fields: {e}")
            return LiquidityEstimate(
                spread_pct=market.spread_pct, bid_volume=market.bid_volume,
                ask_volume=market.ask_volume, source=market.liquidity_source,
            )

    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: ExpertOutput, market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"

        liquidity = self.get_liquidity(self.symbol, market, fv)
        depth_usdt = (liquidity.bid_volume + liquidity.ask_volume) * market.close

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side, mid=market.close, spread_pct=liquidity.spread_pct,
            slippage_pct=self.config.get("slippage_pct", 0.01), depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size, fee_pct=self.fee_pct,
            participation_cap=self.config.get("depth_participation_rate", 0.15),
            market_fill_remainder=self.config.get("partial_fill_policy") == "market_fill_remainder",
        )

        # Log paired liquidity sample if fv is provided
        if fv is not None and self.sample_log_path:
            synthetic = estimate_synthetic_liquidity(fv, market, self._liq_cfg)
            self._sample_buffer.append({
                "timestamp": market.timestamp,
                "synthetic_spread_pct": synthetic.spread_pct,
                "real_spread_pct": liquidity.spread_pct,
                "synthetic_depth_usdt": (synthetic.bid_volume + synthetic.ask_volume) * market.close,
                "real_depth_usdt": depth_usdt,
                "volatility_medium": fv.volatility_medium,
                "volume_zscore": (fv.raw_features or {}).get("volume_zscore", 0.0),
                "bar_volume": market.volume,
                "bar_close": market.close,
            })

        order_id = make_client_order_id(router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size)
        total_cost_pct = fees_pct + slippage_realized

        return ExecutionRecord(
            timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
            fill_timestamp=router_decision.timestamp, symbol=router_decision.symbol,
            side=side, order_price=market.close, fill_price=fill_price,
            requested_size=risk_decision.adjusted_size, filled_size=filled_size,
            slippage_pct=slippage_realized, fees_pct=fees_pct, total_cost_pct=total_cost_pct,
            status=status, order_id=order_id,
        )

    def flush_samples(self) -> int:
        """Write buffered samples to disk and return count."""
        if not self._sample_buffer or not self.sample_log_path:
            return 0

        n = len(self._sample_buffer)
        df = pd.DataFrame(self._sample_buffer)
        path = Path(self.sample_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined.to_parquet(path, compression="snappy")
        logger.info(f"Logged {n} liquidity samples to {path} (total: {len(combined)})")
        self._sample_buffer = []
        return n
