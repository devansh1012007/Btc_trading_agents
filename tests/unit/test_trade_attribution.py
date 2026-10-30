"""
Unit tests for Phase 14 T14.3: `TradeSignal` provenance and HMAC-SHA256 verification.
"""
from dataclasses import replace

import pytest

from src.handoff.agent_sink import TradeSignal, verify_signal_hmac
from src.state.contracts import ExitPlan

pytestmark = [pytest.mark.unit]


def test_trade_signal_hmac_signature_and_tamper_detection():
    plan = ExitPlan(stop_loss_pct=0.8, take_profit_pct=1.6, max_holding_bars=5, source="expert")
    sig = TradeSignal.build(
        timestamp=1700000000,
        symbol="BTC/USDT",
        direction=1,
        size_usdt=1250.0,
        entry_price=50000.0,
        expert_instance_id="trend_expert@r_0",
        regime_id="r_0",
        regime_set_id="hmm_k3",
        exit_plan=plan,
        router_confidence=0.82,
        risk_score=0.18,
        bundle_hash="bundle123",
        model_versions={"trend_expert@r_0": "sha_abc"},
        secret_key="super_secret_key",
    )
    assert sig.hmac_signature
    assert verify_signal_hmac(sig, "super_secret_key")
    assert not verify_signal_hmac(sig, "wrong_key")

    # Tampering with size_usdt invalidates HMAC
    tampered = replace(sig, size_usdt=99999.0)
    assert not verify_signal_hmac(tampered, "super_secret_key")
