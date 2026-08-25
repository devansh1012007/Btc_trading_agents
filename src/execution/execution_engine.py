"""
Execution Engine (Phase 4, §6): one interface, three swappable backends
selected by a single config switch (execution.mode). Everything upstream
(Risk Layer, Router, Opportunity Filter, Experts) calls
execution_engine.place_order(...) and gets back an ExecutionRecord or
None -- none of them know or care which backend ran. This is deliberate:
if paper and live ran different code paths through risk/decision logic, a
paper backtest would stop meaning anything about live behavior.
"""
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..liquidity.types import LiquidityEstimate
from ..state.contracts import ExecutionRecord, ExpertAction, ExpertOutput, MarketSnapshot, RiskDecision, RouterDecision

logger = logging.getLogger(__name__)


def make_client_order_id(timestamp: int, symbol: str, side: str, size: float) -> str:
    """Deterministic clientOrderId (§6.4 point 3): the same decision
    always hashes to the same id, so a crash-and-restart before
    confirmation can check whether the order already exists rather than
    blindly re-placing it."""
    raw = f"{timestamp}:{symbol}:{side}:{round(size, 6)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def simulate_fill(
    side: str, mid: float, spread_pct: float, slippage_pct: float,
    depth_usdt: float, requested_size_usdt: float, fee_pct: float,
    participation_cap: float, market_fill_remainder: bool,
) -> Tuple[float, float, float, float, str]:
    """
    Shared fill-simulation math for the two paper backends (§6.2, §6.3):
    fills at mid +/- spread/2 +/- slippage, with a simple partial-fill
    model capping how much of a bar's synthetic/real depth we're willing
    to assume we could actually execute against.

    Returns (fill_price, filled_size_usdt, fees_pct, slippage_realized_pct, status).
    """
    spread_frac = spread_pct / 100
    if side == "BUY":
        fill_price = mid * (1 + spread_frac / 2 + slippage_pct / 100)
    else:
        fill_price = mid * (1 - spread_frac / 2 - slippage_pct / 100)

    max_fillable_usdt = depth_usdt * participation_cap
    if market_fill_remainder:
        filled_size_usdt = requested_size_usdt
    else:
        filled_size_usdt = min(requested_size_usdt, max_fillable_usdt)

    slippage_realized_pct = abs(fill_price - mid) / mid * 100 if mid else 0.0
    status = "filled" if filled_size_usdt >= requested_size_usdt else ("partial" if filled_size_usdt > 0 else "rejected")

    return fill_price, filled_size_usdt, fee_pct, slippage_realized_pct, status


class ExecutionBackend(ABC):
    @abstractmethod
    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: ExpertOutput, market: MarketSnapshot,
    ) -> ExecutionRecord:
        raise NotImplementedError


class ExecutionEngine:
    """
    Dispatches to the configured backend. `mode` is the one switch
    (execution.mode in settings.yaml): "paper_sim" | "paper_live" | "live".
    """

    def __init__(self, backend: ExecutionBackend, mode: str):
        self.backend = backend
        self.mode = mode

    def place_order(
        self, risk_decision: RiskDecision, router_decision: RouterDecision,
        expert_output: Optional[ExpertOutput], market: MarketSnapshot,
    ) -> Optional[ExecutionRecord]:
        if not risk_decision.approved:
            return None
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            return None
        return self.backend.place_order(risk_decision, router_decision, expert_output, market)

    def get_liquidity(self, symbol: str, market: MarketSnapshot) -> LiquidityEstimate:
        return self.backend.get_liquidity(symbol, market)
