"""
Live Backend (Phase 4, §6.4 + Phase 3 Execution Realism): real CCXT order
placement for both entries (`place_order`) and exits (`close_position` with
`reduceOnly=True` and deterministic idempotency via the `orders` table).
"""
import logging
from typing import Any, Optional

from ...liquidity.real_orderbook_adapter import fetch_real_liquidity
from ...liquidity.types import LiquidityEstimate
from ...registry.db import DEFAULT_DB_PATH, connect
from ...state.contracts import (
    ExecutionRecord,
    ExpertAction,
    ExpertOutput,
    FeatureVector,
    MarketSnapshot,
    RiskDecision,
    RouterDecision,
)
from ...utils.clock import Clock, WallClock
from ..cost_model import CostModel
from ..execution_engine import ExecutionBackend, make_client_order_id

logger = logging.getLogger(__name__)


class LiveTradingNotConfirmedError(RuntimeError):
    """Raised when execution.mode == 'live' is requested without
    confirm_real_money == true."""


class LiveBackend(ExecutionBackend):
    def __init__(
        self,
        config: dict,
        connector: Any,
        symbol: str,
        db_path: str = DEFAULT_DB_PATH,
        cost_model: Optional[CostModel] = None,
        clock: Optional[Clock] = None,
    ):
        if config.get("mode") != "live" or config.get("confirm_real_money") is not True:
            raise LiveTradingNotConfirmedError(
                "LiveBackend requires execution.mode == 'live' AND "
                "execution.confirm_real_money == true (both, explicitly) -- refusing to construct."
            )
        self.config = config
        self.connector = connector
        self.symbol = symbol
        self.db_path = db_path
        self.cost_model = cost_model or CostModel.from_config({"execution": config})
        self.clock = clock or WallClock()

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return fetch_real_liquidity(self.connector, symbol)

    def _find_existing_order(self, client_order_id: str) -> Optional[dict]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
            return dict(row) if row else None

    def _record_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        requested_size: float,
        status: str,
        exchange_order_id: str,
    ) -> None:
        now = self.clock.now_iso()
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO orders (client_order_id, symbol, side, requested_size, status, "
                "exchange_order_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET status=excluded.status, "
                "exchange_order_id=excluded.exchange_order_id, updated_at=excluded.updated_at",
                (client_order_id, symbol, side, requested_size, status, exchange_order_id, now, now),
            )

    def place_order(
        self,
        risk_decision: RiskDecision,
        router_decision: RouterDecision,
        expert_output: ExpertOutput,
        market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"
        client_order_id = make_client_order_id(
            router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size
        )

        existing = self._find_existing_order(client_order_id)
        if existing is not None:
            logger.info(f"Order {client_order_id} already placed (status={existing['status']}) -- not re-placing.")
            return ExecutionRecord(
                timestamp=router_decision.timestamp,
                order_timestamp=router_decision.timestamp,
                fill_timestamp=router_decision.timestamp,
                symbol=router_decision.symbol,
                side=side,
                order_price=market.close,
                fill_price=market.close,
                requested_size=existing["requested_size"],
                filled_size=existing["requested_size"],
                slippage_pct=0.0,
                fees_pct=0.0,
                total_cost_pct=0.0,
                status=existing["status"],
                order_id=existing["exchange_order_id"] or client_order_id,
            )

        amount_base_ccy = risk_decision.adjusted_size / market.close if market.close else 0.0
        order_type = self.config.get("order_type", "limit")

        self._record_order(
            client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
            status="pending", exchange_order_id="",
        )

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
            self._record_order(
                client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
                status="rejected", exchange_order_id="",
            )
            raise

        exchange_order_id = str(order.get("id", ""))
        status = order.get("status", "open")
        self._record_order(
            client_order_id, router_decision.symbol, side, risk_decision.adjusted_size,
            status=status, exchange_order_id=exchange_order_id,
        )

        fill_price = order.get("average") or order.get("price") or market.close
        filled_size = (order.get("filled") or 0.0) * fill_price

        return ExecutionRecord(
            timestamp=router_decision.timestamp,
            order_timestamp=router_decision.timestamp,
            fill_timestamp=router_decision.timestamp,
            symbol=router_decision.symbol,
            side=side,
            order_price=market.close,
            fill_price=fill_price,
            requested_size=risk_decision.adjusted_size,
            filled_size=filled_size,
            slippage_pct=abs(fill_price - market.close) / market.close * 100 if market.close else 0.0,
            fees_pct=self.cost_model.fee_pct("entry"),
            total_cost_pct=0.0,
            status=status,
            order_id=exchange_order_id or client_order_id,
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
        client_order_id = make_client_order_id(
            market.timestamp, self.symbol, f"CLOSE_{side}", size_usdt, trade_id=trade_id,
        )

        existing = self._find_existing_order(client_order_id)
        if existing is not None:
            logger.info(f"Close order {client_order_id} already placed (status={existing['status']}) -- not re-placing.")
            return ExecutionRecord(
                timestamp=market.timestamp,
                order_timestamp=market.timestamp,
                fill_timestamp=market.timestamp,
                symbol=self.symbol,
                side=side,
                order_price=trigger_price,
                fill_price=trigger_price,
                requested_size=existing["requested_size"],
                filled_size=existing["requested_size"],
                slippage_pct=0.0,
                fees_pct=self.cost_model.fee_pct("exit"),
                total_cost_pct=self.cost_model.fee_pct("exit"),
                status=existing["status"],
                order_id=existing["exchange_order_id"] or client_order_id,
            )

        entry_px = float(getattr(position, "entry_price", 0.0)) or market.close
        amount_base_ccy = size_usdt / entry_px if entry_px else 0.0
        close_order_type = self.config.get("close_order_type", "market")

        self._record_order(
            client_order_id, self.symbol, side, size_usdt,
            status="pending", exchange_order_id="",
        )

        try:
            order = self.connector.exchange.create_order(
                symbol=self.symbol,
                type=close_order_type,
                side=side.lower(),
                amount=amount_base_ccy,
                price=trigger_price if close_order_type == "limit" else None,
                params={"clientOrderId": client_order_id, "reduceOnly": True},
            )
        except Exception as e:
            logger.error(f"Live close_position failed for {client_order_id}: {e}")
            self._record_order(
                client_order_id, self.symbol, side, size_usdt,
                status="rejected", exchange_order_id="",
            )
            raise

        exchange_order_id = str(order.get("id", ""))
        status = order.get("status", "filled")
        self._record_order(
            client_order_id, self.symbol, side, size_usdt,
            status=status, exchange_order_id=exchange_order_id,
        )

        fill_price = float(order.get("average") or order.get("price") or trigger_price)
        filled_base = order.get("filled")
        filled_size = (float(filled_base) * fill_price) if filled_base is not None else size_usdt
        slip_pct = abs(fill_price - trigger_price) / trigger_price * 100.0 if trigger_price else 0.0
        exit_fee = self.cost_model.fee_pct("exit")

        return ExecutionRecord(
            timestamp=market.timestamp,
            order_timestamp=market.timestamp,
            fill_timestamp=market.timestamp,
            symbol=self.symbol,
            side=side,
            order_price=trigger_price,
            fill_price=fill_price,
            requested_size=size_usdt,
            filled_size=filled_size,
            slippage_pct=slip_pct,
            fees_pct=exit_fee,
            total_cost_pct=exit_fee + slip_pct,
            status=status,
            order_id=exchange_order_id or client_order_id,
        )
