"""
Unit tests for Phase 3 T3.5: `ExecutionBackend.close_position` across
`PaperSimBackend`, `PaperLiveBackend`, `LiveBackend`, and `process_bar`.
"""
import pytest

from src.execution.backends.live_backend import LiveBackend
from src.execution.backends.paper_live_backend import PaperLiveBackend
from src.execution.backends.paper_sim_backend import PaperSimBackend
from src.execution.execution_engine import ExecutionEngine
from src.execution.position_lifecycle import OpenPosition, process_bar
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.registry.db import connect
from src.state.contracts import (
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    FeatureVector,
    RegimeType,
    MarketSnapshot,
    OpportunityScore,
    RegimeState,
    RouterDecision,
    SystemState,
    ZoneMap,
)

pytestmark = [pytest.mark.unit]


def _make_state(ts: int, close: float, low: float, high: float) -> SystemState:
    snap = MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=close,
        high=high,
        low=low,
        close=close,
        volume=50.0,
        bid=close * 0.9999,
        ask=close * 1.0001,
        bid_volume=20.0,
        ask_volume=20.0,
        spread_pct=0.02,
        hour_utc=12,
        day_of_week=2,
    )
    fv = FeatureVector(
        timestamp=ts,
        symbol="BTC/USDT",
        momentum_short=0.0,
        momentum_medium=0.0,
        momentum_long=0.0,
        volatility_short=0.1,
        volatility_medium=0.2,
        volatility_long=0.2,
        returns_5=0.0,
        rsi_14=50.0,
        macd_signal=0.0,
        sma_9=close,
        sma_21=close,
        imbalance_ratio=0.0,
        orderflow_momentum=0.0,
        spread_pct=0.02,
        depth_10_level=50000.0,
    )
    rs = RegimeState(
        timestamp=ts,
        symbol="BTC/USDT",
        primary_regime=RegimeType.TRENDING_UP,
        regime_confidence=0.8,
        prob_trending_up=0.8,
        prob_trending_down=0.1,
        prob_chop_range=0.05,
        prob_panic_spike=0.05,
        regime_age=5,
        regime_persistence_score=0.8,
    )
    zm = ZoneMap(timestamp=ts, symbol="BTC/USDT", current_price=close, zones=[])
    eo = ExpertOutputs(
        timestamp=ts,
        symbol="BTC/USDT",
        outputs={
            "trend": ExpertOutput(
                timestamp=ts,
                symbol="BTC/USDT",
                expert_name="trend",
                action=ExpertAction.NO_TRADE,
                confidence=0.0,
                no_trade_probability=1.0,
                expected_return=0.0,
                expected_risk=0.1,
                reward_to_risk=0.0,
                invalidation_distance=0.5,
                zone_aware=False,
                model_output=0.0,
                regime_used=RegimeType.TRENDING_UP,
            )
        },
    )
    opp = OpportunityScore(
        timestamp=ts,
        symbol="BTC/USDT",
        tradable=False,
        tradable_probability=0.0,
        environment_quality=0.5,
        opportunity_strength=0.0,
        failed_rules=["no_consensus(1.00)"],
    )
    rd = RouterDecision(
        timestamp=ts,
        symbol="BTC/USDT",
        selected_expert=None,
        selection_confidence=0.0,
        no_trade_probability=1.0,
        expert_ranking={},
        reasoning="no_trade",
    )
    return SystemState(
        timestamp=ts,
        symbol="BTC/USDT",
        market_snapshot=snap,
        feature_vector=fv,
        zone_map=zm,
        regime_state=rs,
        expert_outputs=eo,
        opportunity_score=opp,
        router_decision=rd,
    )


def test_live_backend_close_position_is_reduce_only_and_idempotent(tmp_path):
    db_path = str(tmp_path / "reg.db")
    with connect(db_path): pass

    calls = []

    class FakeExchange:
        def create_order(self, **kwargs):
            calls.append(kwargs)
            return {"id": "exch-close-1", "status": "filled", "average": 101.9, "filled": 10.0}

    class FakeConnector:
        exchange = FakeExchange()

    backend = LiveBackend(
        {"mode": "live", "confirm_real_money": True},
        FakeConnector(),
        "BTC/USDT",
        db_path=db_path,
    )
    pos = OpenPosition(
        trade_id="trade-42",
        entry_timestamp=1000,
        entry_price=100.0,
        direction=1,
        size_usdt=1000.0,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
    )
    state = _make_state(1060, 102.0, 101.0, 102.5)
    rec1 = backend.close_position(pos, state.market_snapshot, "take_profit", 102.0)
    assert len(calls) == 1
    assert calls[0]["side"] == "sell"
    assert calls[0]["params"]["reduceOnly"] is True
    assert rec1.fill_price == pytest.approx(101.9)

    # Second call with same position + timestamp is idempotent (no second exchange call)
    rec2 = backend.close_position(pos, state.market_snapshot, "take_profit", 102.0)
    assert len(calls) == 1
    assert rec2.order_id == rec1.order_id


def test_process_bar_calls_close_position_and_keeps_position_open_on_failure(paper_config):
    state = _make_state(1060, 102.0, 101.0, 102.5)
    pos = OpenPosition(
        trade_id="trade-99",
        entry_timestamp=1000,
        entry_price=100.0,
        direction=1,
        size_usdt=1000.0,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
    )

    class FailingCloseBackend(PaperSimBackend):
        def close_position(self, position, market, exit_reason, trigger_price, fv=None):
            raise RuntimeError("exchange close rejected")

    risk_layer = RiskLayer(paper_config.get("risk", {}), store=RiskStateStore(persistent=False))
    engine_fail = ExecutionEngine(FailingCloseBackend({}), "live")
    outcome_fail = process_bar(
        state,
        risk_layer=risk_layer,
        execution_engine=engine_fail,
        position=pos,
        equity=10000.0,
        fee_pct=0.04,
        config=paper_config,
    )
    assert outcome_fail.execution_failed
    assert outcome_fail.position is not None
    assert outcome_fail.closed is None

    # Now succeed with normal PaperSimBackend
    engine_ok = ExecutionEngine(PaperSimBackend({}), "paper_sim")
    outcome_ok = process_bar(
        state,
        risk_layer=risk_layer,
        execution_engine=engine_ok,
        position=pos,
        equity=10000.0,
        fee_pct=0.04,
        config=paper_config,
    )
    assert not outcome_ok.execution_failed
    assert outcome_ok.position is None
    assert outcome_ok.closed is not None
    assert outcome_ok.closed.close_record is not None
    assert outcome_ok.closed.close_record.side == "SELL"


def test_paper_live_backend_close_position_uses_orderbook_fallback():
    class FailingConnector:
        exchange = None

    backend = PaperLiveBackend({}, FailingConnector(), "BTC/USDT")
    pos = OpenPosition(
        trade_id="trade-7",
        entry_timestamp=1000,
        entry_price=100.0,
        direction=-1,
        size_usdt=500.0,
        take_profit_pct=2.0,
        stop_loss_pct=1.0,
        max_holding_bars=5,
    )
    state = _make_state(1060, 98.0, 97.5, 98.5)
    rec = backend.close_position(pos, state.market_snapshot, "take_profit", 98.0)
    assert rec.side == "BUY"
    assert rec.status == "filled"
    assert rec.fill_price > 98.0
