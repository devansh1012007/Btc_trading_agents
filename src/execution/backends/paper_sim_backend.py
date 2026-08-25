"""
Paper-Sim Backend (Phase 4, §6.2): pure historical/backtest simulation.
Liquidity comes from whatever's already attached to the MarketSnapshot --
the synthetic liquidity model for historical replay (Finding #2's fix).
Fills are simulated via the shared simulate_fill() math; no network calls
at all.
"""
import logging

from ...liquidity.types import LiquidityEstimate
from ...state.contracts import ExecutionRecord, ExpertAction, ExpertOutput, MarketSnapshot, RiskDecision, RouterDecision
from ..execution_engine import ExecutionBackend, make_client_order_id, simulate_fill

logger = logging.getLogger(__name__)


class PaperSimBackend(ExecutionBackend):
    def __init__(self, config: dict, fee_pct: float = 0.04):
        self.config = config
        self.fee_pct = fee_pct

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return LiquidityEstimate(
            spread_pct=market.spread_pct, bid_volume=market.bid_volume,
            ask_volume=market.ask_volume, source=market.liquidity_source,
        )

    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: ExpertOutput, market: MarketSnapshot,
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"
        depth_usdt = (market.bid_volume + market.ask_volume) * market.close

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side, mid=market.close, spread_pct=market.spread_pct,
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
