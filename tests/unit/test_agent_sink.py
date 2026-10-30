"""
Unit tests for Phase 14 T14.4: `AgentSink` (`off | shadow | advisory | execute`),
idempotency, tighten-only advisory clamp, and error fallback (`fail_closed` vs `fallback_internal`).
"""
import pytest

from src.handoff.agent_sink import AgentSink, TradeSignal

pytestmark = [pytest.mark.unit]


def _sig(ts: int = 1000, secret: str = "k1") -> TradeSignal:
    return TradeSignal.build(
        timestamp=ts,
        symbol="BTC/USDT",
        direction=1,
        size_usdt=1000.0,
        entry_price=100.0,
        expert_instance_id="trend@r_0",
        regime_id="r_0",
        secret_key=secret,
    )


def test_agent_sink_modes_idempotency_and_tighten_only_clamp(tmp_path):
    calls = []

    def handler(s: TradeSignal):
        calls.append(s.signal_id)
        # Try to return size_multiplier=3.0 -> must be clamped to 1.0!
        return {"approved": True, "size_multiplier": 3.0, "order_id": "ext-1"}

    sink = AgentSink(
        mode="advisory",
        secret_key="k1",
        handler=handler,
        jsonl_path=tmp_path / "signals.jsonl",
    )
    s1 = _sig(1000, secret="k1")
    r1 = sink.submit(s1)
    assert r1.approved and r1.size_multiplier == pytest.approx(1.0)
    assert len(calls) == 1

    # Duplicate submission of s1 is idempotent (handler not called a second time)
    r2 = sink.submit(s1)
    assert r2 == r1
    assert len(calls) == 1


def test_agent_sink_fail_closed_vs_fallback_internal():
    def broken_handler(s: TradeSignal):
        raise RuntimeError("agent timeout")

    s = _sig(2000, secret="k1")
    sink_closed = AgentSink(mode="execute", secret_key="k1", on_error="fail_closed", handler=broken_handler)
    res_closed = sink_closed.submit(s)
    assert not res_closed.approved and res_closed.size_multiplier == 0.0

    sink_fb = AgentSink(mode="execute", secret_key="k1", on_error="fallback_internal", handler=broken_handler)
    res_fb = sink_fb.submit(s)
    assert res_fb.approved and res_fb.used_internal_fallback and not res_fb.executed_externally
