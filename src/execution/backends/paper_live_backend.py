"""
Paper-Live-Data Backend (Phase 4, §6.3 + Phase 3 Execution Realism): uses a
REAL order book for liquidity, simulates entry and closing fills against the
real current book via `CostModel`, and logs paired (synthetic, real)
liquidity samples.
"""
import logging
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ...liquidity.real_orderbook_adapter import fetch_real_liquidity
from ...liquidity.synthetic_liquidity_model import estimate_synthetic_liquidity
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


class PaperLiveBackend(ExecutionBackend):
    def __init__(
        self,
        config: dict,
        connector: Any,
        symbol: str,
        fee_pct: float = 0.04,
        sample_log_path: Optional[str] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.config = config
        self.connector = connector
        self.symbol = symbol
        self.cost_model = cost_model or CostModel.from_exec_config(config, fee_pct=fee_pct)
        self.sample_log_path = sample_log_path
        self._liq_cfg = config.get("liquidity", {}).get("synthetic", {})
        self._sample_buffer: list = []

    def get_liquidity(
        self,
        symbol: str,
        market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> LiquidityEstimate:
        try:
            return fetch_real_liquidity(self.connector, symbol)
        except Exception as e:
            logger.warning(f"Real order-book fetch failed, falling back to snapshot's existing liquidity fields: {e}")
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
        side = "BUY" if expert_output.action == ExpertAction.BUY else "SELL"

        liquidity = self.get_liquidity(self.symbol, market, fv)
        depth_usdt = (liquidity.bid_volume + liquidity.ask_volume) * market.close
        live_snap = dc_replace(
            market,
            spread_pct=liquidity.spread_pct,
            bid_volume=liquidity.bid_volume,
            ask_volume=liquidity.ask_volume,
        )
        slip_pct = self.cost_model.slippage_pct(
            "entry", fv=fv, market=live_snap, size_usdt=risk_decision.adjusted_size,
        )
        imp_pct = self.cost_model.impact_pct(risk_decision.adjusted_size, depth_usdt)
        entry_fee = self.cost_model.fee_pct("entry")

        fill_price, filled_size, fees_pct, slippage_realized, status = simulate_fill(
            side=side,
            mid=market.close,
            spread_pct=liquidity.spread_pct,
            slippage_pct=slip_pct,
            depth_usdt=depth_usdt,
            requested_size_usdt=risk_decision.adjusted_size,
            fee_pct=entry_fee,
            participation_cap=self.cost_model.impact.participation_cap,
            market_fill_remainder=self.cost_model.impact.partial_fill_policy == "market_fill_remainder",
            impact_pct=imp_pct,
        )

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

        order_id = make_client_order_id(
            router_decision.timestamp, router_decision.symbol, side, risk_decision.adjusted_size,
        )
        total_cost_pct = fees_pct + slippage_realized
        fill_ts = int(router_decision.timestamp + int(self.cost_model.entry_delay_seconds))

        return ExecutionRecord(
            timestamp=router_decision.timestamp,
            order_timestamp=router_decision.timestamp,
            fill_timestamp=fill_ts,
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

        liquidity = self.get_liquidity(self.symbol, market, fv)
        live_snap = dc_replace(
            market,
            spread_pct=liquidity.spread_pct,
            bid_volume=liquidity.bid_volume,
            ask_volume=liquidity.ask_volume,
        )
        exit_model = str(self.config.get("exit_model", "realistic_fill"))
        fill_px = exit_fill_price(
            direction=direction,
            trigger_price=trigger_price,
            bar=live_snap,
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

    def flush_samples(self) -> int:
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
