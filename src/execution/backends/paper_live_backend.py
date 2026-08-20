"""
Paper-Live-Data Backend (Phase 4, §6.3): uses a REAL order book for
liquidity (via the real order-book adapter), but never calls the
exchange's order-placement endpoint -- fills are still simulated, just
against the real current book instead of the synthetic proxy. This is a
meaningfully better validation step before ever touching real capital,
and costs nothing extra since the real-time connection already exists
for regime/feature computation in this mode.
"""
import logging

from ...state.contracts import RiskDecision, RouterDecision, ExpertOutput, MarketSnapshot, ExecutionRecord, ExpertAction
from ...liquidity.types import LiquidityEstimate
from ...liquidity.real_orderbook_adapter import fetch_real_liquidity
from ..execution_engine import ExecutionBackend, simulate_fill, make_client_order_id

logger = logging.getLogger(__name__)


class PaperLiveBackend(ExecutionBackend):
    def __init__(self, config: dict, connector, symbol: str, fee_pct: float = 0.04):
        self.config = config
        self.connector = connector
        self.symbol = symbol
        self.fee_pct = fee_pct

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
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
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"

        liquidity = self.get_liquidity(self.symbol, market)
        depth_usdt = (liquidity.bid_volume + liquidity.ask_volume) * market.close

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side, mid=market.close, spread_pct=liquidity.spread_pct,
            slippage_pct=self.config.get("slippage_pct", 0.01), depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size, fee_pct=self.fee_pct,
            participation_cap=self.config.get("depth_participation_rate", 0.15),
            market_fill_remainder=self.config.get("partial_fill_policy") == "market_fill_remainder",
        )

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
