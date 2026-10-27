"""
Paper-Sim Backend (Phase 4, §6.2): pure historical/backtest simulation.
Liquidity comes from whatever's already attached to the MarketSnapshot --
the synthetic liquidity model for historical replay (Finding #2's fix).
Fills are simulated via the shared simulate_fill() math; no network calls
at all.
"""
import logging
from typing import Optional

from ...liquidity.types import LiquidityEstimate
from ...state.contracts import (
    ExecutionRecord,
    ExpertAction,
    ExpertOutput,
    FeatureVector,
    MarketSnapshot,
    RiskDecision,
    RouterDecision,
)
from ..execution_engine import ExecutionBackend, make_client_order_id, simulate_fill
from ..execution_engine import total_slippage_pct as shared_total_slippage_pct

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
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        # --- order rejection knob ---
        rejection_rate = self.config.get("order_rejection_rate", 0.0)
        if rejection_rate >= 1.0:
            order_id = make_client_order_id(router_decision.timestamp, router_decision.symbol,
                                            "BUY" if expert_output.action == ExpertAction.BUY else "SELL",
                                            risk_decision.adjusted_size)
            return ExecutionRecord(
                timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
                fill_timestamp=router_decision.timestamp, symbol=router_decision.symbol,
                side="BUY" if expert_output.action == ExpertAction.BUY else "SELL",
                order_price=market.close, fill_price=market.close,
                requested_size=risk_decision.adjusted_size, filled_size=0.0,
                slippage_pct=0.0, fees_pct=0.0, total_cost_pct=0.0,
                status="rejected", order_id=order_id,
            )

        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"
        depth_usdt = (market.bid_volume + market.ask_volume) * market.close

        # --- simulated latency & adverse cost knobs (shared with the labeler) ---
        latency_ms = self.config.get("simulated_latency_ms", 0)
        total_slippage_pct = shared_total_slippage_pct(self.config)

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side, mid=market.close, spread_pct=market.spread_pct,
            slippage_pct=total_slippage_pct, depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size, fee_pct=self.fee_pct,
            participation_cap=self.config.get("depth_participation_rate", 0.15),
            market_fill_remainder=self.config.get("partial_fill_policy") == "market_fill_remainder",
        )

        # Round fill_timestamp down to the nearest second (latency_ms < 1000 → same second)
        fill_timestamp = int(router_decision.timestamp + latency_ms // 1000)

        order_id = make_client_order_id(router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size)
        total_cost_pct = fees_pct + slippage_realized

        return ExecutionRecord(
            timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
            fill_timestamp=fill_timestamp, symbol=router_decision.symbol,
            side=side, order_price=market.close, fill_price=fill_price,
            requested_size=risk_decision.adjusted_size, filled_size=filled_size,
            slippage_pct=slippage_realized, fees_pct=fees_pct, total_cost_pct=total_cost_pct,
            status=status, order_id=order_id,
        )
