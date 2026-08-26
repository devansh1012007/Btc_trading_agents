"""
Paper-Sim Backend (Phase 4, §6.2): pure historical/backtest simulation.
Liquidity comes from whatever's already attached to the MarketSnapshot --
the synthetic liquidity model for historical replay (Finding #2's fix).
Fills are simulated via the shared simulate_fill() math; no network calls
at all.

Realism knobs (all default OFF, all deterministic — same bar -> same
outcome, so backtests stay reproducible):

  execution.simulated_latency_ms          order->fill delay. Adds adverse
        slippage of execution.latency_adverse_pct_per_100ms per 100ms and
        shifts fill_timestamp accordingly. Stress-tests latency sensitivity.
  execution.order_rejection_rate          0..1 probability the exchange
        rejects the order outright (status="rejected"). Stress-tests
        exchange outages / risk of unexecuted signals.
"""
import hashlib
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

    def _simulated_rejection(self, ts: int, symbol: str, side: str) -> bool:
        rate = float(self.config.get("order_rejection_rate", 0.0))
        if rate <= 0:
            return False
        # Deterministic pseudo-roll per (timestamp, symbol, side): rejecting
        # must not depend on wall-clock RNG state or backtests diverge.
        digest = hashlib.sha256(f"{ts}|{symbol}|{side}".encode()).digest()
        roll = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        return roll < rate

    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: ExpertOutput, market: MarketSnapshot,
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"

        if self._simulated_rejection(router_decision.timestamp, router_decision.symbol, side):
            logger.warning(f"[paper_sim] simulated exchange rejection for {side} @ {router_decision.timestamp}")
            return ExecutionRecord(
                timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
                fill_timestamp=router_decision.timestamp, symbol=router_decision.symbol,
                side=side, order_price=market.close, fill_price=market.close,
                requested_size=risk_decision.adjusted_size, filled_size=0.0,
                slippage_pct=0.0, fees_pct=0.0, total_cost_pct=0.0,
                status="rejected",
                order_id=make_client_order_id(router_decision.timestamp,
                                              router_decision.symbol, side,
                                              risk_decision.adjusted_size),
            )

        depth_usdt = (market.bid_volume + market.ask_volume) * market.close

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side, mid=market.close, spread_pct=market.spread_pct,
            slippage_pct=self.config.get("slippage_pct", 0.01), depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size, fee_pct=self.fee_pct,
            participation_cap=self.config.get("depth_participation_rate", 0.15),
            market_fill_remainder=self.config.get("partial_fill_policy") == "market_fill_remainder",
        )

        # Latency stress: adverse price drift while the order is in flight,
        # plus a fill_timestamp shift so downstream tooling can see it.
        latency_ms = float(self.config.get("simulated_latency_ms", 0.0))
        if latency_ms > 0 and status in ("filled", "partial"):
            adverse_pct = float(self.config.get("latency_adverse_pct_per_100ms", 0.0)) * (latency_ms / 100.0)
            if adverse_pct > 0:
                direction = 1 if side == "BUY" else -1  # adverse = against the taker
                fill_price *= (1 + direction * adverse_pct / 100)
                slippage_realized += adverse_pct
        fill_ts = router_decision.timestamp + int(round(latency_ms / 1000))

        order_id = make_client_order_id(router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size)
        total_cost_pct = fees_pct + slippage_realized

        return ExecutionRecord(
            timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
            fill_timestamp=fill_ts, symbol=router_decision.symbol,
            side=side, order_price=market.close, fill_price=fill_price,
            requested_size=risk_decision.adjusted_size, filled_size=filled_size,
            slippage_pct=slippage_realized, fees_pct=fees_pct, total_cost_pct=total_cost_pct,
            status=status, order_id=order_id,
        )
