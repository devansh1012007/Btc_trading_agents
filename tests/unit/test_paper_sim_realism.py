"""
Paper-sim realism knob tests (PRESENTATION §4): simulated execution
latency and deterministic order rejection — both default-off, both
deterministic per bar so backtests stay reproducible.
"""
import pytest

from src.execution.backends.paper_sim_backend import PaperSimBackend
from src.state.contracts import (
    ExpertAction,
    ExpertOutput,
    MarketSnapshot,
    RegimeType,
    RiskDecision,
    RouterDecision,
)

pytestmark = [pytest.mark.unit]


def make_snapshot(close=100.0):
    return MarketSnapshot(
        timestamp=1_700_000_000, symbol="BTC/USDT", open=close, high=close * 1.01,
        low=close * 0.99, close=close, volume=50, bid=close * 0.9999, ask=close * 1.0001,
        bid_volume=5_000, ask_volume=5_000, spread_pct=0.02,
    )


def make_expert_output(action=ExpertAction.BUY):
    return ExpertOutput(
        expert_name="trend_expert", timestamp=0, symbol="BTC/USDT", action=action,
        confidence=0.8, no_trade_probability=0.2, expected_return=0.2, expected_risk=0.1,
        reward_to_risk=2.0, invalidation_distance=0.5, zone_aware=True, model_output=0.8,
        regime_used=RegimeType.TRENDING_UP,
    )


RISK = RiskDecision(timestamp=0, symbol="BTC/USDT", approved=True, risk_score=0.1,
                    base_size=1000, adjusted_size=1000, size_reduction_pct=0.0)
ROUTER = RouterDecision(timestamp=0, symbol="BTC/USDT", selected_expert="trend_expert",
                        selection_confidence=0.8, no_trade_probability=0.1)


def test_latency_knob_off_by_default():
    backend = PaperSimBackend(config={})
    record = backend.place_order(RISK, ROUTER, make_expert_output(), make_snapshot())
    assert record.status in ("filled", "partial")
    assert record.fill_timestamp == record.order_timestamp


def test_simulated_latency_shifts_fill_and_adds_adverse_cost():
    base_backend = PaperSimBackend(config={})
    latency_backend = PaperSimBackend(config={
        "simulated_latency_ms": 250,
        "latency_adverse_pct_per_100ms": 0.01,  # +0.025% total at 250ms
    })
    market = make_snapshot()

    clean = base_backend.place_order(RISK, ROUTER, make_expert_output(), market)
    delayed = latency_backend.place_order(RISK, ROUTER, make_expert_output(), market)

    assert clean.status == "filled"
    assert delayed.fill_timestamp == delayed.order_timestamp  # 250ms < 1s rounding
    # BUY pays more the longer the order is in flight:
    assert delayed.fill_price > clean.fill_price
    assert delayed.slippage_pct > clean.slippage_pct
    assert delayed.total_cost_pct == pytest.approx(
        delayed.fees_pct + delayed.slippage_pct)


def test_rejection_knob_deterministic_and_bounded():
    config = {"order_rejection_rate": 1.0}  # always reject
    backend = PaperSimBackend(config=config)
    r1 = backend.place_order(RISK, ROUTER, make_expert_output(), make_snapshot())
    r2 = backend.place_order(RISK, ROUTER, make_expert_output(), make_snapshot())
    assert r1.status == "rejected" and r2.status == "rejected"
    assert r1.filled_size == 0.0 and r1.fees_pct == 0.0

    never = PaperSimBackend(config={"order_rejection_rate": 0.0})
    ok = never.place_order(RISK, ROUTER, make_expert_output(), make_snapshot())
    assert ok.status in ("filled", "partial")


def test_rejection_rate_zero_means_no_rejections():
    backend = PaperSimBackend(config={"order_rejection_rate": 0.0})
    for ts in range(50):
        router = RouterDecision(timestamp=ts, symbol="BTC/USDT",
                                selected_expert="trend_expert",
                                selection_confidence=0.8, no_trade_probability=0.1)
        record = backend.place_order(RISK, router, make_expert_output(), make_snapshot())
        assert record.status != "rejected"
