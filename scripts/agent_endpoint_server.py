#!/usr/bin/env python3
"""
CLI entrypoint for the External Agent Endpoint server (`D9` / Phase 14).

Usage:
  python scripts/agent_endpoint_server.py --host 127.0.0.1 --port 8590 --db-path data/agent_endpoint.db
  python scripts/agent_endpoint_server.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.handoff.agent_sink import AgentSink, HttpWebhookAgentClient, TradeSignal
from src.handoff.endpoint import (
    AgentEndpointHTTPServer,
    AgentEndpointProcessor,
    AgentEndpointStore,
    ExternalAgentPolicy,
)
from src.state.contracts import ExitPlan


def _run_self_test() -> int:
    secret = "self-test-hmac-secret"
    store = AgentEndpointStore(":memory:")
    policy = ExternalAgentPolicy(min_confidence=0.55, min_reward_to_risk=1.2, max_size_usdt=1000.0)
    processor = AgentEndpointProcessor(policy=policy, store=store, secret_key=secret)
    server = AgentEndpointHTTPServer(processor, host="127.0.0.1", port=0)
    base_url = server.start_in_background()
    try:
        client = HttpWebhookAgentClient(
            f"{base_url}/v1/signals",
            timeout_ms=2000,
            secret_key=secret,
            verify_response_signature=True,
        )
        sink = AgentSink(mode="advisory", secret_key=secret, handler=client, transport="http_webhook")
        sig = TradeSignal.build(
            timestamp=1700000000,
            symbol="BTC/USDT",
            direction=1,
            size_usdt=1500.0,
            entry_price=50000.0,
            expert_instance_id="trend_follow:TRENDING_UP",
            regime_id="TRENDING_UP",
            exit_plan=ExitPlan(take_profit_pct=2.4, stop_loss_pct=1.2, max_holding_bars=20),
            router_confidence=0.72,
            risk_score=0.30,
            secret_key=secret,
        )
        resp = sink.submit(sig)
        close_ack = sink.notify_close(
            signal_id=sig.signal_id,
            trade_id="t-1",
            symbol="BTC/USDT",
            exit_reason="take_profit",
            exit_price=51200.0,
            realized_pnl_pct=2.2,
            timestamp=1700000600,
        )
        summary = {
            "base_url": base_url,
            "approved": resp.approved,
            "size_multiplier": resp.size_multiplier,
            "external_order_id": resp.external_order_id,
            "reason": resp.reason,
            "close_ack": close_ack,
            "counts": store.counts(),
        }
        print(json.dumps(summary, indent=2))
        return 0 if (resp.approved and abs(resp.size_multiplier - (1000.0 / 1500.0)) < 1e-4) else 1
    finally:
        server.shutdown()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="External Agent Endpoint Server (D9)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8590, help="Bind port")
    parser.add_argument("--db-path", default=":memory:", help="SQLite store path")
    parser.add_argument("--secret-env", default="BTC_BOT_AGENT_SECRET", help="Environment variable for HMAC secret")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--min-rr", type=float, default=1.0)
    parser.add_argument("--max-size-usdt", type=float, default=10000.0)
    parser.add_argument("--self-test", action="store_true", help="Run loopback self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()

    secret = os.environ.get(args.secret_env)
    store = AgentEndpointStore(args.db_path)
    policy = ExternalAgentPolicy(
        min_confidence=args.min_confidence,
        min_reward_to_risk=args.min_rr,
        max_size_usdt=args.max_size_usdt,
    )
    processor = AgentEndpointProcessor(policy=policy, store=store, secret_key=secret)
    server = AgentEndpointHTTPServer(processor, host=args.host, port=args.port)
    print(f"External Agent Endpoint listening on {server.base_url} (HMAC={'enabled' if secret else 'disabled'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
