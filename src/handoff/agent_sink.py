"""
Trade attribution (`TradeSignal`), client transports (`HttpWebhookAgentClient`,
`SubprocessAgentClient`, `FileQueueAgentClient`), and external agent handoff
contract (`AgentSink`, Phase 14 T14.3 & T14.4 / Conflict C18 / Decision D9).

Supports four modes:
* `off`: no handoff
* `shadow`: emits signed `TradeSignal` asynchronously/append-only without altering internal execution
* `advisory`: queries external agent for `(approved, size_multiplier)` where the agent can ONLY
  veto or reduce size (`0.0 <= size_multiplier <= 1.0`, never flip direction or increase size!)
* `execute`: delegates order execution to the external agent with idempotency key `signal_id`
  and configurable error fallback (`fail_closed` | `fallback_internal`).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..state.contracts import ExitPlan


@dataclass(frozen=True)
class TradeSignal:
    signal_id: str
    timestamp: int
    symbol: str
    direction: int
    size_usdt: float
    entry_price: float
    expert_instance_id: str
    regime_id: str
    regime_set_id: str
    exit_plan: Optional[ExitPlan] = None
    router_confidence: float = 0.0
    risk_score: float = 0.0
    bundle_hash: str = ""
    model_versions: Dict[str, str] = field(default_factory=dict)
    hmac_signature: str = ""

    @classmethod
    def build(
        cls,
        *,
        timestamp: int,
        symbol: str,
        direction: int,
        size_usdt: float,
        entry_price: float,
        expert_instance_id: str,
        regime_id: str,
        regime_set_id: str = "legacy4",
        exit_plan: Optional[ExitPlan] = None,
        router_confidence: float = 0.0,
        risk_score: float = 0.0,
        bundle_hash: str = "",
        model_versions: Optional[Mapping[str, str]] = None,
        secret_key: Optional[str] = None,
    ) -> "TradeSignal":
        raw_id = f"{timestamp}:{symbol}:{direction}:{expert_instance_id}:{round(size_usdt, 4)}"
        sig_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
        sig = cls(
            signal_id=sig_id,
            timestamp=int(timestamp),
            symbol=str(symbol),
            direction=int(direction),
            size_usdt=float(size_usdt),
            entry_price=float(entry_price),
            expert_instance_id=str(expert_instance_id),
            regime_id=str(regime_id),
            regime_set_id=str(regime_set_id),
            exit_plan=exit_plan,
            router_confidence=float(router_confidence),
            risk_score=float(risk_score),
            bundle_hash=str(bundle_hash),
            model_versions=dict(model_versions or {}),
            hmac_signature="",
        )
        if secret_key:
            mac = compute_signal_hmac(sig, secret_key)
            return cls(
                signal_id=sig.signal_id,
                timestamp=sig.timestamp,
                symbol=sig.symbol,
                direction=sig.direction,
                size_usdt=sig.size_usdt,
                entry_price=sig.entry_price,
                expert_instance_id=sig.expert_instance_id,
                regime_id=sig.regime_id,
                regime_set_id=sig.regime_set_id,
                exit_plan=sig.exit_plan,
                router_confidence=sig.router_confidence,
                risk_score=sig.risk_score,
                bundle_hash=sig.bundle_hash,
                model_versions=sig.model_versions,
                hmac_signature=mac,
            )
        return sig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeSignal":
        ep_raw = payload.get("exit_plan")
        exit_plan = ExitPlan.from_dict(ep_raw) if isinstance(ep_raw, Mapping) else None
        mv_raw = payload.get("model_versions") or {}
        model_versions = {str(k): str(v) for k, v in mv_raw.items()} if isinstance(mv_raw, Mapping) else {}
        return cls(
            signal_id=str(payload["signal_id"]),
            timestamp=int(payload["timestamp"]),
            symbol=str(payload["symbol"]),
            direction=int(payload["direction"]),
            size_usdt=float(payload["size_usdt"]),
            entry_price=float(payload["entry_price"]),
            expert_instance_id=str(payload["expert_instance_id"]),
            regime_id=str(payload["regime_id"]),
            regime_set_id=str(payload.get("regime_set_id", "legacy4")),
            exit_plan=exit_plan,
            router_confidence=float(payload.get("router_confidence", 0.0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            bundle_hash=str(payload.get("bundle_hash", "")),
            model_versions=model_versions,
            hmac_signature=str(payload.get("hmac_signature", "")),
        )

    def to_canonical_dict(self, include_signature: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "size_usdt": round(self.size_usdt, 6),
            "entry_price": round(self.entry_price, 6),
            "expert_instance_id": self.expert_instance_id,
            "regime_id": self.regime_id,
            "regime_set_id": self.regime_set_id,
            "exit_plan": self.exit_plan.to_dict() if self.exit_plan is not None else None,
            "router_confidence": round(self.router_confidence, 6),
            "risk_score": round(self.risk_score, 6),
            "bundle_hash": self.bundle_hash,
            "model_versions": dict(sorted(self.model_versions.items())),
        }
        if include_signature:
            d["hmac_signature"] = self.hmac_signature
        return d


def compute_payload_hmac(payload_dict: Mapping[str, Any], secret_key: str) -> str:
    clean = {k: v for k, v in payload_dict.items() if k not in ("hmac_signature", "response_signature")}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret_key.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_signal_hmac(signal: TradeSignal, secret_key: str) -> str:
    payload = json.dumps(signal.to_canonical_dict(include_signature=False), sort_keys=True, separators=(",", ":"))
    return hmac.new(secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_signal_hmac(signal: TradeSignal, secret_key: str) -> bool:
    if not signal.hmac_signature:
        return False
    expected = compute_signal_hmac(signal, secret_key)
    return hmac.compare_digest(signal.hmac_signature, expected)


@dataclass(frozen=True)
class AgentResponse:
    approved: bool = True
    size_multiplier: float = 1.0
    executed_externally: bool = False
    external_order_id: Optional[str] = None
    reason: str = "ok"
    used_internal_fallback: bool = False
    duplicate: bool = False


class HttpWebhookAgentClient:
    """HTTP/JSON webhook transport to the External Agent Endpoint (`D9`)."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        timeout_ms: int = 1500,
        secret_key: Optional[str] = None,
        verify_response_signature: bool = False,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_seconds = max(0.05, float(timeout_ms) / 1000.0)
        self.secret_key = secret_key
        self.verify_response_signature = verify_response_signature

    def __call__(self, signal: TradeSignal) -> Mapping[str, Any]:
        sig_to_send = signal
        if self.secret_key and not sig_to_send.hmac_signature:
            mac = compute_signal_hmac(sig_to_send, self.secret_key)
            sig_to_send = TradeSignal(
                signal_id=sig_to_send.signal_id,
                timestamp=sig_to_send.timestamp,
                symbol=sig_to_send.symbol,
                direction=sig_to_send.direction,
                size_usdt=sig_to_send.size_usdt,
                entry_price=sig_to_send.entry_price,
                expert_instance_id=sig_to_send.expert_instance_id,
                regime_id=sig_to_send.regime_id,
                regime_set_id=sig_to_send.regime_set_id,
                exit_plan=sig_to_send.exit_plan,
                router_confidence=sig_to_send.router_confidence,
                risk_score=sig_to_send.risk_score,
                bundle_hash=sig_to_send.bundle_hash,
                model_versions=sig_to_send.model_versions,
                hmac_signature=mac,
            )

        body_dict = sig_to_send.to_canonical_dict(include_signature=True)
        body_bytes = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        url = (
            self.endpoint_url
            if self.endpoint_url.endswith("/v1/signals")
            else f"{self.endpoint_url}/v1/signals"
        )
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Trade-Signal-Id": sig_to_send.signal_id,
        }
        if sig_to_send.hmac_signature:
            headers["X-Trade-Signature"] = sig_to_send.hmac_signature

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_bytes = resp.read()
                parsed: Dict[str, Any] = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"http_{exc.code}:{err_body}") from exc

        if self.verify_response_signature and self.secret_key:
            resp_sig = str(parsed.get("response_signature", ""))
            expected_sig = compute_payload_hmac(parsed, self.secret_key)
            if not resp_sig or not hmac.compare_digest(resp_sig, expected_sig):
                raise RuntimeError("invalid_agent_response_signature")
        return parsed

    def notify_close(
        self,
        *,
        signal_id: str,
        trade_id: str,
        symbol: str,
        exit_reason: str,
        exit_price: float,
        realized_pnl_pct: float,
        timestamp: int,
    ) -> Mapping[str, Any]:
        base = (
            self.endpoint_url[:-len("/v1/signals")]
            if self.endpoint_url.endswith("/v1/signals")
            else self.endpoint_url
        )
        url = f"{base}/v1/positions/close"
        payload: Dict[str, Any] = {
            "signal_id": str(signal_id),
            "trade_id": str(trade_id),
            "symbol": str(symbol),
            "exit_reason": str(exit_reason),
            "exit_price": round(float(exit_price), 6),
            "realized_pnl_pct": round(float(realized_pnl_pct), 6),
            "timestamp": int(timestamp),
        }
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.secret_key:
            mac = compute_payload_hmac(payload, self.secret_key)
            payload["hmac_signature"] = mac
            headers["X-Trade-Signature"] = mac

        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))


class SubprocessAgentClient:
    """JSON-over-stdio subprocess transport for local external agents (`D9`)."""

    def __init__(self, command: List[str], *, timeout_ms: int = 1500) -> None:
        if not command:
            raise ValueError("SubprocessAgentClient requires a non-empty command list")
        self.command = list(command)
        self.timeout_seconds = max(0.05, float(timeout_ms) / 1000.0)

    def __call__(self, signal: TradeSignal) -> Mapping[str, Any]:
        payload = json.dumps(signal.to_canonical_dict(include_signature=True))
        proc = subprocess.run(
            self.command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=True,
        )
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError("subprocess_empty_stdout")
        return dict(json.loads(out.splitlines()[-1]))


class FileQueueAgentClient:
    """Filesystem outbox/inbox queue transport for external agents (`D9`)."""

    def __init__(self, queue_dir: str | Path) -> None:
        self.queue_dir = Path(queue_dir)
        self.outbox_dir = self.queue_dir / "outbox"
        self.inbox_dir = self.queue_dir / "inbox"
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, signal: TradeSignal) -> Mapping[str, Any]:
        out_file = self.outbox_dir / f"{signal.signal_id}.json"
        out_file.write_text(
            json.dumps(signal.to_canonical_dict(include_signature=True), indent=2),
            encoding="utf-8",
        )
        in_file = self.inbox_dir / f"{signal.signal_id}.json"
        if in_file.exists():
            return dict(json.loads(in_file.read_text(encoding="utf-8")))
        return {
            "approved": True,
            "size_multiplier": 1.0,
            "order_id": f"fq-{signal.signal_id[:10]}",
            "reason": "queued_to_outbox",
        }


class AgentSink:
    """Configurable trade-signal handoff dispatcher (`off | shadow | advisory | execute`)."""

    def __init__(
        self,
        *,
        mode: str = "off",
        on_error: str = "fail_closed",  # "fail_closed" | "fallback_internal"
        secret_key: Optional[str] = None,
        handler: Optional[Callable[[TradeSignal], Mapping[str, Any]]] = None,
        jsonl_path: Optional[str | Path] = None,
        transport: str = "in_memory",
    ) -> None:
        if mode not in ("off", "shadow", "advisory", "execute"):
            raise ValueError(f"Invalid AgentSink mode {mode!r}")
        self.mode = mode
        self.on_error = on_error
        self.secret_key = secret_key
        self.handler = handler
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.transport = transport
        self.emitted_signals: List[TradeSignal] = []
        self._seen_ids: Dict[str, AgentResponse] = {}
        self.closed_notifications: List[Dict[str, Any]] = []

    def submit(self, signal: TradeSignal) -> AgentResponse:
        if self.mode == "off":
            return AgentResponse(approved=True, size_multiplier=1.0, reason="mode_off")

        if self.secret_key and not signal.hmac_signature:
            signal = TradeSignal(
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                direction=signal.direction,
                size_usdt=signal.size_usdt,
                entry_price=signal.entry_price,
                expert_instance_id=signal.expert_instance_id,
                regime_id=signal.regime_id,
                regime_set_id=signal.regime_set_id,
                exit_plan=signal.exit_plan,
                router_confidence=signal.router_confidence,
                risk_score=signal.risk_score,
                bundle_hash=signal.bundle_hash,
                model_versions=signal.model_versions,
                hmac_signature=compute_signal_hmac(signal, self.secret_key),
            )

        # Idempotency: duplicate signal_id returns cached response without re-invoking handler
        if signal.signal_id in self._seen_ids:
            return self._seen_ids[signal.signal_id]

        if self.secret_key and not verify_signal_hmac(signal, self.secret_key):
            resp = AgentResponse(approved=False, size_multiplier=0.0, reason="invalid_hmac_signature")
            self._seen_ids[signal.signal_id] = resp
            return resp

        self.emitted_signals.append(signal)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal.to_canonical_dict(include_signature=True)) + "\n")

        if self.mode == "shadow":
            if self.handler is not None:
                try:
                    self.handler(signal)
                except Exception:
                    pass
            resp = AgentResponse(approved=True, size_multiplier=1.0, reason="shadow_logged")
            self._seen_ids[signal.signal_id] = resp
            return resp

        try:
            raw = self.handler(signal) if self.handler is not None else {"approved": True, "size_multiplier": 1.0}
        except Exception as exc:
            if self.on_error == "fallback_internal":
                resp = AgentResponse(
                    approved=True,
                    size_multiplier=1.0,
                    executed_externally=False,
                    reason=f"agent_error_fallback_internal:{exc}",
                    used_internal_fallback=True,
                )
            else:
                resp = AgentResponse(
                    approved=False,
                    size_multiplier=0.0,
                    executed_externally=False,
                    reason=f"agent_error_fail_closed:{exc}",
                )
            self._seen_ids[signal.signal_id] = resp
            return resp

        approved = bool(raw.get("approved", raw.get("accepted", True)))
        # Tighten-only invariant: advisory/execute agent can never increase size above 1.0
        mult = max(0.0, min(1.0, float(raw.get("size_multiplier", 1.0 if approved else 0.0))))
        if not approved:
            mult = 0.0

        ext_id = raw.get("order_id") or raw.get("external_order_id")
        resp = AgentResponse(
            approved=approved,
            size_multiplier=mult,
            executed_externally=(self.mode == "execute" and approved),
            external_order_id=str(ext_id) if ext_id else None,
            reason=str(raw.get("reason", "ok")),
            duplicate=bool(raw.get("duplicate", False)),
        )
        self._seen_ids[signal.signal_id] = resp
        return resp

    def notify_close(
        self,
        *,
        signal_id: str,
        trade_id: str,
        symbol: str,
        exit_reason: str,
        exit_price: float,
        realized_pnl_pct: float,
        timestamp: int,
    ) -> Optional[Mapping[str, Any]]:
        if self.mode == "off":
            return None
        record = {
            "signal_id": signal_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "realized_pnl_pct": realized_pnl_pct,
            "timestamp": timestamp,
        }
        self.closed_notifications.append(record)
        if hasattr(self.handler, "notify_close"):
            try:
                return self.handler.notify_close(**record)  # type: ignore[no-any-return]
            except Exception:
                return None
        return record


def build_agent_sink(
    config: Mapping[str, Any],
    *,
    handler: Optional[Callable[[TradeSignal], Mapping[str, Any]]] = None,
) -> AgentSink:
    """Build an `AgentSink` from the `handoff:` config block (`D9`)."""
    raw = config.get("handoff", {}) or {}
    mode = str(raw.get("mode", "off"))
    on_error = str(raw.get("on_error", "fail_closed"))
    transport = str(raw.get("transport", "in_memory"))
    timeout_ms = int(raw.get("timeout_ms", 1500))
    jsonl_path = raw.get("jsonl_path")

    secret_env_var = str(raw.get("secret_env_var", "BTC_BOT_AGENT_SECRET"))
    secret_key = os.environ.get(secret_env_var) or raw.get("secret_key")
    if secret_key is not None:
        secret_key = str(secret_key)

    resolved_handler = handler
    if resolved_handler is None and mode != "off":
        if transport == "http_webhook":
            endpoint_url = str(raw.get("endpoint_url", "http://127.0.0.1:8590/v1/signals"))
            resolved_handler = HttpWebhookAgentClient(
                endpoint_url,
                timeout_ms=timeout_ms,
                secret_key=secret_key,
                verify_response_signature=bool(raw.get("verify_response_signature", False)),
            )
        elif transport == "subprocess":
            cmd = list(raw.get("command", []))
            resolved_handler = SubprocessAgentClient(cmd, timeout_ms=timeout_ms)
        elif transport == "file_queue":
            qdir = str(raw.get("queue_dir", "artifacts/agent_queue"))
            resolved_handler = FileQueueAgentClient(qdir)

    return AgentSink(
        mode=mode,
        on_error=on_error,
        secret_key=secret_key,
        handler=resolved_handler,
        jsonl_path=jsonl_path,
        transport=transport,
    )
