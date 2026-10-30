"""
Paper-Sim Backend (Phase 4, §6.2 + Phase 3 Execution Realism): pure
historical/backtest simulation using shared `CostModel`, `simulate_fill`,
and `exit_fill_price`.
"""
import logging
from typing import Any, Optional

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
from ..cost_model import CostModel
from ..execution_engine import (
    ExecutionBackend,
    exit_fill_price,
    make_client_order_id,
    simulate_fill,
)

logger = logging.getLogger(__name__)


class PaperSimBackend(ExecutionBackend):
    def __init__(
        self,
        config: dict,
        fee_pct: float = 0.04,
        cost_model: Optional[CostModel] = None,
    ):
        self.config = config
        self.cost_model = cost_model or CostModel.from_exec_config(config, fee_pct=fee_pct)

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return LiquidityEstimate(
            spread_pct=market.spread_pct,
            bid_volume=market.bid_volume,
            ask_volume=market.ask_volume,
            source=market.liquidity_source,
        )

    def place_order(
        self,
        risk_decision: RiskDecision,
        router_decision: RouterDecision,
        expert_output: ExpertOutput,
        market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        rejection_rate = self.cost_model.rejects.order_rejection_rate
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"
        if rejection_rate >= 1.0:
            order_id = make_client_order_id(
                router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size,
            )
            return ExecutionRecord(
                timestamp=router_decision.timestamp,
                order_timestamp=router_decision.timestamp,
                fill_timestamp=router_decision.timestamp,
                symbol=router_decision.symbol,
                side=side,
                order_price=market.close,
                fill_price=market.close,
                requested_size=risk_decision.adjusted_size,
                filled_size=0.0,
                slippage_pct=0.0,
                fees_pct=0.0,
                total_cost_pct=0.0,
                status="rejected",
                order_id=order_id,
            )

        depth_usdt = (market.bid_volume + market.ask_volume) * market.close
        eff_spread = self.cost_model.spread_pct(market, leg="entry")
        slip_pct = self.cost_model.slippage_pct(
            "entry", fv=fv, market=market, size_usdt=risk_decision.adjusted_size,
        )
        imp_pct = self.cost_model.impact_pct(risk_decision.adjusted_size, depth_usdt)
        entry_fee_pct = self.cost_model.fee_pct("entry")

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side,
            mid=market.close,
            spread_pct=eff_spread,
            slippage_pct=slip_pct,
            depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size,
            fee_pct=entry_fee_pct,
            participation_cap=self.cost_model.impact.participation_cap,
            market_fill_remainder=self.cost_model.impact.partial_fill_policy == "market_fill_remainder",
            impact_pct=imp_pct,
        )

        fill_timestamp = int(router_decision.timestamp + int(self.cost_model.entry_delay_seconds))
        order_id = make_client_order_id(
            router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size,
        )
        total_cost_pct = fees_pct + slippage_realized

        return ExecutionRecord(
            timestamp=router_decision.timestamp,
            order_timestamp=router_decision.timestamp,
            fill_timestamp=fill_timestamp,
            symbol=router_decision.symbol,
            side=side,
            order_price=market.close,
            fill_price=fill_price,
            requested_size=risk_decision.adjusted_size,
            filled_size=filled_size,
            slippage_pct=slippage_realized,
            fees_pct=fees_pct,
            total_cost_pct=total_cost_pct,
            status=status,
            order_id=order_id,
        )

    def close_position(
        self,
        position: Any,
        market: MarketSnapshot,
        exit_reason: str,
        trigger_price: float,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        direction = int(getattr(position, "direction", 1))
        side = "SELL" if direction == 1 else "BUY"
        size_usdt = float(getattr(position, "size_usdt", 0.0))
        trade_id = getattr(position, "trade_id", None) or None
        exit_model = str(self.config.get("exit_model", "barrier_level"))

        fill_px = exit_fill_price(
            direction=direction,
            trigger_price=trigger_price,
            bar=market,
            exit_reason=exit_reason,
            cost_model=self.cost_model,
            exit_model=exit_model,
            fv=fv,
            size_usdt=size_usdt,
        )
        slip_realized = abs(fill_px - trigger_price) / trigger_price * 100.0 if trigger_price else 0.0
        exit_fee = self.cost_model.fee_pct("exit")
        fill_ts = int(market.timestamp + int(self.cost_model.exit_delay_seconds))
        order_id = make_client_order_id(
            market.timestamp, market.symbol, f"CLOSE_{side}", size_usdt, trade_id=trade_id,
        )
        return ExecutionRecord(
            timestamp=market.timestamp,
            order_timestamp=market.timestamp,
            fill_timestamp=fill_ts,
            symbol=market.symbol,
            side=side,
            order_price=trigger_price,
            fill_price=fill_px,
            requested_size=size_usdt,
            filled_size=size_usdt,
            slippage_pct=slip_realized,
            fees_pct=exit_fee,
            total_cost_pct=exit_fee + slip_realized,
            status="filled",
            order_id=order_id,
        )
