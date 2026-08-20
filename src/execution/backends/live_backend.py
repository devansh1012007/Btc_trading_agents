"""
Live Backend (Phase 4, §6.4): real CCXT order placement. Three
independent, deliberately redundant safety gates -- a single mistake in
any one of them still can't risk real capital:

  1. Both execution.mode == "live" AND execution.confirm_real_money ==
     true are required at construction time -- a typo or copy-pasted
     config can't silently enable this. system.sandbox (Phase 1) still
     applies underneath: mode="live" against a sandbox/testnet account is
     meaningfully different from mode="live" against a real one, and both
     flags plus system.sandbox=false are needed to place a real order
     with real funds.
  2. API keys are never read from settings.yaml -- only the NAME of the
     environment variable holding them is configured; the connector must
     already be authenticated by the caller (this backend never reads a
     secret directly).
  3. Idempotent order placement via a deterministic clientOrderId: on
     construction-time or per-order, the orders table is checked before
     ever calling create_order -- a crash-and-restart replays the same
     decision into the same id and finds it already placed, rather than
     placing a second one.

NOT exercised against a real exchange in this delivery -- there are no
live credentials in this environment. Structure and safety gates are
complete and unit-tested (test_live_mode_requires_both_flags,
test_execution_idempotent_order_placement against a mocked exchange);
the actual create_order call path needs your real API keys to validate
end-to-end.
"""
import logging
from datetime import datetime, timezone

from ...state.contracts import RiskDecision, RouterDecision, ExpertOutput, MarketSnapshot, ExecutionRecord, ExpertAction
from ...liquidity.types import LiquidityEstimate
from ...liquidity.real_orderbook_adapter import fetch_real_liquidity
from ...registry.db import connect, DEFAULT_DB_PATH
from ..execution_engine import ExecutionBackend, make_client_order_id

logger = logging.getLogger(__name__)


class LiveTradingNotConfirmedError(RuntimeError):
    """Raised when execution.mode == 'live' is requested without the
    separate, explicit confirm_real_money flag also being true."""


class LiveBackend(ExecutionBackend):
    def __init__(self, config: dict, connector, symbol: str, db_path: str = DEFAULT_DB_PATH):
        if config.get("mode") != "live" or config.get("confirm_real_money") is not True:
            raise LiveTradingNotConfirmedError(
                "LiveBackend requires execution.mode == 'live' AND "
                "execution.confirm_real_money == true (both, explicitly) -- refusing to construct."
            )
        self.config = config
        self.connector = connector
        self.symbol = symbol
        self.db_path = db_path

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return fetch_real_liquidity(self.connector, symbol)

    def _find_existing_order(self, client_order_id: str):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
            return dict(row) if row else None

    def _record_order(self, client_order_id: str, symbol: str, side: str, requested_size: float,
                       status: str, exchange_order_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO orders (client_order_id, symbol, side, requested_size, status, "
                "exchange_order_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET status=excluded.status, "
                "exchange_order_id=excluded.exchange_order_id, updated_at=excluded.updated_at",
                (client_order_id, symbol, side, requested_size, status, exchange_order_id, now, now),
            )

    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: ExpertOutput, market: MarketSnapshot,
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"
        client_order_id = make_client_order_id(
            router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size
        )

        existing = self._find_existing_order(client_order_id)
        if existing is not None:
            logger.info(f"Order {client_order_id} already placed (status={existing['status']}) -- not re-placing.")
            return ExecutionRecord(
                timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
                fill_timestamp=router_decision.timestamp, symbol=router_decision.symbol, side=side,
                order_price=market.close, fill_price=market.close,
                requested_size=existing["requested_size"], filled_size=existing["requested_size"],
                slippage_pct=0.0, fees_pct=0.0, total_cost_pct=0.0,
                status=existing["status"], order_id=existing["exchange_order_id"] or client_order_id,
            )

        amount_base_ccy = risk_decision.adjusted_size / market.close if market.close else 0.0
        order_type = self.config.get("order_type", "limit")

        self._record_order(client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
                            status="pending", exchange_order_id="")

        try:
            order = self.connector.exchange.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side.lower(),
                amount=amount_base_ccy,
                price=market.close if order_type == "limit" else None,
                params={"clientOrderId": client_order_id},
            )
        except Exception as e:
            logger.error(f"Live order placement failed for {client_order_id}: {e}")
            self._record_order(client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
                                status="rejected", exchange_order_id="")
            raise

        exchange_order_id = str(order.get("id", ""))
        status = order.get("status", "open")
        self._record_order(client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
                            status=status, exchange_order_id=exchange_order_id)

        fill_price = order.get("average") or order.get("price") or market.close
        filled_size = (order.get("filled") or 0.0) * fill_price

        return ExecutionRecord(
            timestamp=router_decision.timestamp, order_timestamp=router_decision.timestamp,
            fill_timestamp=router_decision.timestamp, symbol=router_decision.symbol, side=side,
            order_price=market.close, fill_price=fill_price,
            requested_size=risk_decision.adjusted_size, filled_size=filled_size,
            slippage_pct=abs(fill_price - market.close) / market.close * 100 if market.close else 0.0,
            fees_pct=self.config.get("assumed_taker_fee_pct", 0.04), total_cost_pct=0.0,
            status=status, order_id=exchange_order_id or client_order_id,
        )
