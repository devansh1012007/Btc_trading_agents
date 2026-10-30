"""
Comprehensive unit & loopback HTTP integration tests for the External Agent
Endpoint (`D9` / Conflict C18 / Phase 14) and runtime `process_bar` wiring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.execution.position_lifecycle import (
    OpenPosition,
    build_execution_engine,
    process_bar,
)
from src.handoff.agent_sink import (
    AgentSink,
    FileQueueAgentClient,
    HttpWebhookAgentClient,
    TradeSignal,
    build_agent_sink,
)
from src.handoff.endpoint import (
    AgentEndpointHTTPServer,
    AgentEndpointProcessor,
    AgentEndpointStore,
    ExternalAgentPolicy,
)
from src.layers.risk_layer import RiskLayer
from src.state.contracts import (
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    ExitPlan,
    FeatureVector,
    MarketSnapshot,
    OpportunityScore,
    RegimeState,
    RegimeType,
    RouterDecision,
    SystemState,
)
from src.utils.config_helpers import load_layered_config


def _sample_signal(
    *,
    size_usdt: float = 1200.0,
    confidence: float = 0.72,
    risk_score: float = 0.30,
    regime_id: str = "TRENDING_UP",
    secret_key: str = "test-d9-secret",
) -> TradeSignal:
    return TradeSignal.build(
        timestamp=1_700_000_000,
        symbol="BTC/USDT",
        direction=1,
        size_usdt=size_usdt,
        entry_price=50_000.0,
        expert_instance_id="trend_expert",
        regime_id=regime_id,
        exit_plan=ExitPlan(take_profit_pct=2.0, stop_loss_pct=1.0, max_holding_bars=10),
        router_confidence=confidence,
        risk_score=risk_score,
        secret_key=secret_key,
    )


def _sample_system_state(timestamp: int = 1_700_000_000, price: float = 50_000.0) -> SystemState:
    snap = MarketSnapshot(
        timestamp=timestamp,
        symbol="BTC/USDT",
        open=price,
        high=price * 1.02,
        low=price * 0.998,
        close=price,
        volume=100.0,
        bid=price * 0.9999,
        ask=price * 1.0001,
        bid_volume=50.0,
        ask_volume=50.0,
        spread_pct=0.01,
    )
    fv = FeatureVector(
        timestamp=timestamp,
        symbol="BTC/USDT",
        momentum_short=0.2,
        momentum_medium=0.5,
        volatility_short=0.2,
        volatility_medium=0.3,
        returns_5=0.4,
        rsi_14=55.0,
        macd_signal=0.1,
        sma_9=price * 0.999,
        sma_21=price * 0.998,
        momentum_long=0.8,
        volatility_long=0.4,
        imbalance_ratio=0.55,
        orderflow_momentum=15.0,
        spread_pct=0.01,
        depth_10_level=100_000.0,
        distance_to_nearest_support=0.5,
        distance_to_nearest_resistance=1.5,
        support_strength=0.7,
        resistance_strength=0.6,
    )
    reg = RegimeState(
        timestamp=timestamp,
        symbol="BTC/USDT",
        prob_trending_up=0.75,
        prob_trending_down=0.10,
        prob_chop_range=0.10,
        prob_panic_spike=0.05,
        primary_regime=RegimeType.TRENDING_UP,
        regime_confidence=0.75,
        regime_age=10,
        regime_persistence_score=0.80,
    )
    trend_out = ExpertOutput(
        expert_name="trend_expert",
        timestamp=timestamp,
        symbol="BTC/USDT",
        action=ExpertAction.BUY,
        confidence=0.75,
        no_trade_probability=0.15,
        expected_return=0.40,
        expected_risk=0.20,
        reward_to_risk=2.0,
        invalidation_distance=0.5,
        zone_aware=True,
        model_output=0.75,
        regime_used=RegimeType.TRENDING_UP,
        exit_plan=ExitPlan(take_profit_pct=1.0, stop_loss_pct=0.8, max_holding_bars=5),
    )
    hold_mr = ExpertOutput(
        expert_name="mean_reversion_expert",
        timestamp=timestamp,
        symbol="BTC/USDT",
        action=ExpertAction.NO_TRADE,
        confidence=0.0,
        no_trade_probability=1.0,
        expected_return=0.0,
        expected_risk=0.0,
        reward_to_risk=0.0,
        invalidation_distance=0.5,
        zone_aware=False,
        model_output=0.0,
    )
    hold_bo = ExpertOutput(
        expert_name="breakout_expert",
        timestamp=timestamp,
        symbol="BTC/USDT",
        action=ExpertAction.NO_TRADE,
        confidence=0.0,
        no_trade_probability=1.0,
        expected_return=0.0,
        expected_risk=0.0,
        reward_to_risk=0.0,
        invalidation_distance=0.5,
        zone_aware=False,
        model_output=0.0,
    )
    eos = ExpertOutputs(
        timestamp=timestamp,
        symbol="BTC/USDT",
        outputs={
            "trend_expert": trend_out,
            "mean_reversion_expert": hold_mr,
            "breakout_expert": hold_bo,
        },
    )
    opp = OpportunityScore(
        timestamp=timestamp,
        symbol="BTC/USDT",
        tradable=True,
        tradable_probability=0.80,
        environment_quality=0.80,
        opportunity_strength=0.80,
        edge_vs_fees=0.25,
    )
    rd = RouterDecision(
        timestamp=timestamp,
        symbol="BTC/USDT",
        selected_expert="trend_expert",
        selection_confidence=0.75,
        no_trade_probability=0.15,
        expert_ranking={"trend_expert": 0.75},
    )
    return SystemState(
        timestamp=timestamp,
        symbol="BTC/USDT",
        market_snapshot=snap,
        feature_vector=fv,
        regime_state=reg,
        expert_outputs=eos,
        opportunity_score=opp,
        router_decision=rd,
    )


def test_http_endpoint_loopback_hmac_idempotency_and_close() -> None:
    secret = "loopback-hmac-secret"
    store = AgentEndpointStore(":memory:")
    policy = ExternalAgentPolicy(
        min_confidence=0.60,
        min_reward_to_risk=1.5,
        max_size_usdt=800.0,
        allowed_regimes=("TRENDING_UP", "TRENDING_DOWN"),
    )
    proc = AgentEndpointProcessor(policy=policy, store=store, secret_key=secret)
    server = AgentEndpointHTTPServer(proc, host="127.0.0.1", port=0)
    base_url = server.start_in_background()
    try:
        client = HttpWebhookAgentClient(
            f"{base_url}/v1/signals",
            timeout_ms=2000,
            secret_key=secret,
            verify_response_signature=True,
        )
        sink = AgentSink(mode="advisory", secret_key=secret, handler=client, transport="http_webhook")

        sig = _sample_signal(size_usdt=1600.0, confidence=0.75, secret_key=secret)
        resp1 = sink.submit(sig)
        assert resp1.approved is True
        # Policy capped 1600 -> 800 USDT => multiplier = 0.5
        assert resp1.size_multiplier == pytest.approx(0.5)
        assert resp1.duplicate is False

        # Re-submitting identical signal_id hits idempotency cache
        resp2 = sink.submit(sig)
        assert resp2.approved is True
        assert resp2.size_multiplier == pytest.approx(0.5)
        assert resp2 == resp1

        # Direct call to server with same signal_id also returns duplicate=True from SQLite store
        raw_dup = client(sig)
        assert raw_dup["duplicate"] is True
        assert raw_dup["size_multiplier"] == pytest.approx(0.5)

        # Notify position close over HTTP
        close_ack = sink.notify_close(
            signal_id=sig.signal_id,
            trade_id="trade-42",
            symbol="BTC/USDT",
            exit_reason="take_profit",
            exit_price=51_000.0,
            realized_pnl_pct=1.92,
            timestamp=1_700_000_300,
        )
        assert close_ack is not None
        assert close_ack.get("acknowledged") is True
        counts = store.counts()
        assert counts["received_signals"] == 1
        assert counts["closed_positions"] == 1
    finally:
        server.shutdown()
        store.close()


def test_http_endpoint_rejects_wrong_hmac_with_401() -> None:
    store = AgentEndpointStore(":memory:")
    proc = AgentEndpointProcessor(store=store, secret_key="server-secret")
    server = AgentEndpointHTTPServer(proc, host="127.0.0.1", port=0)
    base_url = server.start_in_background()
    try:
        bad_client = HttpWebhookAgentClient(
            f"{base_url}/v1/signals",
            timeout_ms=1500,
            secret_key="wrong-client-secret",
        )
        sig = _sample_signal(secret_key="wrong-client-secret")
        with pytest.raises(RuntimeError, match="http_401"):
            bad_client(sig)

        # AgentSink with on_error="fail_closed" turns 401 into approved=False
        sink = AgentSink(
            mode="advisory",
            on_error="fail_closed",
            secret_key="wrong-client-secret",
            handler=bad_client,
        )
        resp = sink.submit(sig)
        assert resp.approved is False
        assert resp.size_multiplier == 0.0
        assert "http_401" in resp.reason
    finally:
        server.shutdown()
        store.close()


def test_file_queue_client_and_build_agent_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BTC_BOT_AGENT_SECRET", "env-secret-key")
    qdir = tmp_path / "queue"
    cfg: Dict[str, Any] = {
        "handoff": {
            "mode": "execute",
            "transport": "file_queue",
            "queue_dir": str(qdir),
            "secret_env_var": "BTC_BOT_AGENT_SECRET",
        }
    }
    sink = build_agent_sink(cfg)
    assert sink.mode == "execute"
    assert sink.secret_key == "env-secret-key"
    assert isinstance(sink.handler, FileQueueAgentClient)

    sig = _sample_signal(secret_key="env-secret-key")
    resp = sink.submit(sig)
    assert resp.approved is True
    assert resp.executed_externally is True
    assert (qdir / "outbox" / f"{sig.signal_id}.json").exists()


def test_process_bar_wires_external_agent_advisory_and_execute_modes() -> None:
    secret = "process-bar-secret"
    store = AgentEndpointStore(":memory:")
    # Policy caps size at 50 USDT (tighten-only)
    policy = ExternalAgentPolicy(
        min_confidence=0.60,
        max_size_usdt=50.0,
        allowed_regimes=("trending_up", "TRENDING_UP"),
    )
    proc = AgentEndpointProcessor(policy=policy, store=store, secret_key=secret)
    server = AgentEndpointHTTPServer(proc, host="127.0.0.1", port=0)
    base_url = server.start_in_background()
    try:
        cfg = load_layered_config("paper")
        cfg["handoff"] = {
            "mode": "advisory",
            "transport": "http_webhook",
            "endpoint_url": f"{base_url}/v1/signals",
            "secret_key": secret,
            "verify_response_signature": True,
        }
        risk_layer = RiskLayer(cfg)
        exec_engine = build_execution_engine(cfg, None, "BTC/USDT")
        state = _sample_system_state(1_700_000_000, 50_000.0)

        # 1. Advisory mode tightens position size to 50 USDT
        outcome = process_bar(
            state,
            risk_layer=risk_layer,
            execution_engine=exec_engine,
            position=None,
            equity=10_000.0,
            fee_pct=0.04,
            config=cfg,
        )
        assert outcome.opened is not None
        assert outcome.opened.size_usdt == pytest.approx(50.0, rel=1e-2)
        assert outcome.opened.signal_id != ""

        # 2. Execute mode delegates order to external agent and notifies close on exit
        cfg_exec = load_layered_config("paper")
        cfg_exec["handoff"] = {
            "mode": "execute",
            "transport": "http_webhook",
            "endpoint_url": f"{base_url}/v1/signals",
            "secret_key": secret,
            "verify_response_signature": True,
        }
        risk_layer_exec = RiskLayer(cfg_exec)
        state2 = _sample_system_state(1_700_000_600, 50_100.0)
        outcome_exec = process_bar(
            state2,
            risk_layer=risk_layer_exec,
            execution_engine=exec_engine,
            position=None,
            equity=10_000.0,
            fee_pct=0.04,
            config=cfg_exec,
        )
        assert outcome_exec.opened is not None
        assert outcome_exec.state.execution_record is not None
        assert outcome_exec.state.execution_record.order_id.startswith("ext-")

        # Force exit on a subsequent bar (past min_trade_seconds=10s) and verify close notification
        pos_open: OpenPosition = outcome_exec.opened
        pos_open.bars_held = 5
        state_exit = _sample_system_state(1_700_000_960, 52_000.0)
        outcome_close = process_bar(
            state_exit,
            risk_layer=risk_layer_exec,
            execution_engine=exec_engine,
            position=pos_open,
            equity=10_000.0,
            fee_pct=0.04,
            config=cfg_exec,
        )
        assert outcome_close.closed is not None
        assert store.counts()["closed_positions"] >= 1
    finally:
        server.shutdown()
        store.close()
