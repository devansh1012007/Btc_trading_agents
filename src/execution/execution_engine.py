"""
Execution Engine (Phase 4, §6 + Phase 3 Execution Realism): one interface,
three swappable backends selected by `execution.mode`.

Provides:
* `entry_fill_price` and `entry_fill_price_with_delay`
* `exit_fill_price` (`ExitFillModel`)
* `simulate_fill`
* `ExecutionBackend.place_order` and `ExecutionBackend.close_position`
"""
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Tuple

from ..liquidity.types import LiquidityEstimate
from ..state.contracts import (
    ExecutionRecord,
    ExpertAction,
    ExpertOutput,
    FeatureVector,
    MarketSnapshot,
    RiskDecision,
    RouterDecision,
)
from .cost_model import CostModel
from .price_path import PricePath

logger = logging.getLogger(__name__)


def make_client_order_id(
    timestamp: int,
    symbol: str,
    side: str,
    size: float,
    trade_id: Optional[str] = None,
) -> str:
    """Deterministic clientOrderId (§6.4 point 3)."""
    suffix = f":{trade_id}" if trade_id else ""
    raw = f"{timestamp}:{symbol}:{side}:{round(size, 6)}{suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def entry_fill_price(
    side: str,
    mid: float,
    spread_pct: float,
    slippage_pct: float,
    impact_pct: float = 0.0,
) -> float:
    """Price an entry order of ``side`` fills at: mid +/- half the spread +/- slippage +/- impact."""
    spread_frac = spread_pct / 100.0
    adverse_frac = spread_frac / 2.0 + (slippage_pct + impact_pct) / 100.0
    if side.upper() == "BUY":
        return mid * (1.0 + adverse_frac)
    return mid * (1.0 - adverse_frac)


def entry_fill_price_with_delay(
    side: str,
    signal_bar: MarketSnapshot,
    cost_model: CostModel,
    *,
    next_bar: Optional[MarketSnapshot] = None,
    granularity_seconds: float = 60.0,
    fv: Optional[FeatureVector] = None,
    size_usdt: float = 0.0,
    price_path: Optional[PricePath] = None,
) -> Tuple[float, int]:
    """Compute entry fill price and fill timestamp under `cost_model.entry_delay_seconds`."""
    delay_s = cost_model.entry_delay_seconds
    fill_ts = int(signal_bar.timestamp + int(delay_s))

    if price_path is not None and delay_s > 0.0:
        ref_mid, eff_spread, step = price_path.price_at(
            float(signal_bar.timestamp), delay_s, side=side, trigger_price=signal_bar.close,
        )
        depth_usdt = step.depth_usdt
        fill_ts = int(step.timestamp)
    elif next_bar is not None and delay_s > 0.0:
        alpha = min(1.0, max(0.0, delay_s / max(granularity_seconds, 1e-6)))
        ref_mid = (1.0 - alpha) * signal_bar.close + alpha * next_bar.open
        eff_spread = cost_model.spread_pct(next_bar if alpha >= 0.5 else signal_bar, leg="entry")
        depth_usdt = (signal_bar.bid_volume + signal_bar.ask_volume) * signal_bar.close
    else:
        ref_mid = signal_bar.close
        eff_spread = cost_model.spread_pct(signal_bar, leg="entry")
        depth_usdt = (signal_bar.bid_volume + signal_bar.ask_volume) * signal_bar.close

    slip_pct = cost_model.slippage_pct("entry", fv=fv, market=signal_bar, size_usdt=size_usdt)
    imp_pct = cost_model.impact_pct(size_usdt, depth_usdt)
    fill_px = entry_fill_price(side, ref_mid, eff_spread, slip_pct, imp_pct)
    return fill_px, fill_ts


def exit_fill_price(
    direction: int,
    trigger_price: float,
    bar: MarketSnapshot,
    exit_reason: str,
    cost_model: Optional[CostModel] = None,
    *,
    exit_model: str = "realistic_fill",
    next_bar: Optional[MarketSnapshot] = None,
    granularity_seconds: float = 60.0,
    fv: Optional[FeatureVector] = None,
    size_usdt: float = 0.0,
    price_path: Optional[PricePath] = None,
) -> float:
    """Price an exit order fills at (`ExitFillModel`).

    When `exit_model == "barrier_level"` and `cost_model.exit_delay_seconds == 0`,
    returns `trigger_price` unchanged (exact legacy behaviour).
    Under `"realistic_fill"` (or when `exit_delay_seconds > 0`), applies delayed
    reference price lookup + exit half-spread + exit slippage + market impact in
    the adverse direction for the closing leg (`SELL` for longs, `BUY` for shorts).
    """
    cm = cost_model or CostModel()
    delay_s = cm.exit_delay_seconds
    if exit_model == "barrier_level" and delay_s <= 0.0 and not cm.explicit_costs_block:
        return float(trigger_price)

    closing_side = "SELL" if direction == 1 else "BUY"
    from ..labeling.triple_barrier import is_stop_exit_reason
    is_stop = is_stop_exit_reason(exit_reason)

    if price_path is not None and delay_s > 0.0:
        ref_price, eff_spread, step = price_path.price_at(
            float(bar.timestamp), delay_s, side=closing_side, trigger_price=trigger_price,
        )
        depth_usdt = step.depth_usdt
    elif next_bar is not None and delay_s > 0.0:
        alpha = min(1.0, max(0.0, delay_s / max(granularity_seconds, 1e-6)))
        ref_price = (1.0 - alpha) * float(trigger_price) + alpha * float(next_bar.open)
        eff_spread = cm.spread_pct(next_bar if alpha >= 0.5 else bar, leg="exit")
        depth_usdt = (bar.bid_volume + bar.ask_volume) * bar.close
    else:
        ref_price = float(trigger_price)
        eff_spread = cm.spread_pct(bar, leg="exit")
        depth_usdt = (bar.bid_volume + bar.ask_volume) * bar.close

    if exit_model == "barrier_level" and not cm.explicit_costs_block:
        # Only apply latency adverse drift when exit_model is barrier_level with delay_s > 0
        drift_frac = (delay_s * cm.latency.adverse_drift_pct_per_second) / 100.0
        return ref_price * (1.0 - drift_frac) if direction == 1 else ref_price * (1.0 + drift_frac)

    slip_pct = cm.slippage_pct("exit", fv=fv, market=bar, size_usdt=size_usdt, is_stop=is_stop)
    imp_pct = cm.impact_pct(size_usdt, depth_usdt)
    adverse_frac = (eff_spread / 2.0 + slip_pct + imp_pct) / 100.0
    if direction == 1:
        return ref_price * (1.0 - adverse_frac)
    return ref_price * (1.0 + adverse_frac)


def total_slippage_pct(exec_cfg: Mapping[str, Any]) -> float:
    """Slippage (%) the paper simulator charges: delegates to CostModel."""
    return CostModel.from_exec_config(exec_cfg).slippage_pct("entry")


def simulate_fill(
    side: str,
    mid: float,
    spread_pct: float,
    slippage_pct: float,
    depth_usdt: float,
    requested_size_usdt: float,
    fee_pct: float,
    participation_cap: float,
    market_fill_remainder: bool,
    impact_pct: float = 0.0,
) -> Tuple[float, float, float, float, str]:
    """Shared fill-simulation math for the two paper backends."""
    fill_price = entry_fill_price(side, mid, spread_pct, slippage_pct, impact_pct=impact_pct)

    max_fillable_usdt = depth_usdt * participation_cap
    if market_fill_remainder:
        filled_size_usdt = requested_size_usdt
    else:
        filled_size_usdt = min(requested_size_usdt, max_fillable_usdt)

    slippage_realized_pct = abs(fill_price - mid) / mid * 100 if mid else 0.0
    status = (
        "filled"
        if filled_size_usdt >= requested_size_usdt
        else ("partial" if filled_size_usdt > 0 else "rejected")
    )

    return fill_price, filled_size_usdt, fee_pct, slippage_realized_pct, status


class ExecutionBackend(ABC):
    @abstractmethod
    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self,
        risk_decision: RiskDecision,
        router_decision: RouterDecision,
        expert_output: ExpertOutput,
        market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        raise NotImplementedError

    def close_position(
        self,
        position: Any,
        market: MarketSnapshot,
        exit_reason: str,
        trigger_price: float,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        """Place a reduce-only closing order for ``position``."""
        side = "SELL" if getattr(position, "direction", 1) == 1 else "BUY"
        size_usdt = float(getattr(position, "size_usdt", 0.0))
        trade_id = getattr(position, "trade_id", None) or None
        order_id = make_client_order_id(
            market.timestamp, market.symbol, f"CLOSE_{side}", size_usdt, trade_id=trade_id,
        )
        return ExecutionRecord(
            timestamp=market.timestamp,
            order_timestamp=market.timestamp,
            fill_timestamp=market.timestamp,
            symbol=market.symbol,
            side=side,
            order_price=trigger_price,
            fill_price=trigger_price,
            requested_size=size_usdt,
            filled_size=size_usdt,
            slippage_pct=0.0,
            fees_pct=0.0,
            total_cost_pct=0.0,
            status="filled",
            order_id=order_id,
        )


class ExecutionEngine:
    """Dispatches to the configured backend (`paper_sim` | `paper_live` | `live`)."""

    def __init__(self, backend: ExecutionBackend, mode: str):
        self.backend = backend
        self.mode = mode

    def place_order(
        self,
        risk_decision: RiskDecision,
        router_decision: RouterDecision,
        expert_output: Optional[ExpertOutput],
        market: MarketSnapshot,
        fv: Optional[FeatureVector] = None,
    ) -> Optional[ExecutionRecord]:
        if not risk_decision.approved:
            return None
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            return None
        return self.backend.place_order(risk_decision, router_decision, expert_output, market, fv=fv)

    def close_position(
        self,
        position: Any,
        market: MarketSnapshot,
        exit_reason: str,
        trigger_price: float,
        fv: Optional[FeatureVector] = None,
    ) -> ExecutionRecord:
        return self.backend.close_position(position, market, exit_reason, trigger_price, fv=fv)

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return self.backend.get_liquidity(symbol, market)
