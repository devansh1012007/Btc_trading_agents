"""
External Agent Endpoint (`D9` / Conflict C18 / Phase 14 T14.4).

Provides:
* `ExternalAgentPolicy`: configurable risk/confidence/regime/reward-to-risk and
  tighten-only sizing policy evaluated by the external agent endpoint.
* `AgentEndpointStore`: SQLite / in-memory idempotency and audit store for
  inbound `TradeSignal` and position-close events.
* `AgentEndpointProcessor`: HMAC-SHA256 verification, idempotent signal
  handling (`signal_id`), policy evaluation, and signed response creation.
* `AgentEndpointHTTPServer`: thread-safe HTTP server (`GET /health`,
  `POST /v1/signals`, `POST /v1/positions/close`, `GET /v1/signals/<id>`).
"""
from __future__ import annotations

import hmac
import json
import sqlite3
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .agent_sink import (
    TradeSignal,
    compute_payload_hmac,
    verify_signal_hmac,
)


@dataclass(frozen=True)
class ExternalAgentPolicy:
    """Policy rules enforced by the External Agent Endpoint (`D9`)."""

    min_confidence: float = 0.50
    min_reward_to_risk: float = 1.0
    max_risk_score: float = 0.85
    max_size_usdt: float = 10_000.0
    allowed_regimes: Optional[Tuple[str, ...]] = None
    blocked_experts: Tuple[str, ...] = ()
    scale_by_confidence: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ExternalAgentPolicy":
        raw = config.get("agent_policy", config) or {}
        allowed = raw.get("allowed_regimes")
        blocked = raw.get("blocked_experts", ())
        return cls(
            min_confidence=float(raw.get("min_confidence", 0.50)),
            min_reward_to_risk=float(raw.get("min_reward_to_risk", 1.0)),
            max_risk_score=float(raw.get("max_risk_score", 0.85)),
            max_size_usdt=float(raw.get("max_size_usdt", 10_000.0)),
            allowed_regimes=tuple(str(r) for r in allowed) if allowed is not None else None,
            blocked_experts=tuple(str(e) for e in blocked),
            scale_by_confidence=bool(raw.get("scale_by_confidence", False)),
        )

    def evaluate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Return `{approved, size_multiplier, reason}` respecting tighten-only sizing."""
        if signal.expert_instance_id in self.blocked_experts:
            return {
                "approved": False,
                "size_multiplier": 0.0,
                "reason": f"expert_blocked:{signal.expert_instance_id}",
            }

        if self.allowed_regimes is not None and signal.regime_id not in self.allowed_regimes:
            return {
                "approved": False,
                "size_multiplier": 0.0,
                "reason": f"regime_not_allowed:{signal.regime_id}",
            }

        if signal.router_confidence < self.min_confidence - 1e-9:
            return {
                "approved": False,
                "size_multiplier": 0.0,
                "reason": f"low_confidence:{signal.router_confidence:.3f}<{self.min_confidence:.3f}",
            }

        if signal.risk_score > self.max_risk_score + 1e-9:
            return {
                "approved": False,
                "size_multiplier": 0.0,
                "reason": f"high_risk_score:{signal.risk_score:.3f}>{self.max_risk_score:.3f}",
            }

        if signal.exit_plan is not None and signal.exit_plan.stop_loss_pct > 0:
            rr = signal.exit_plan.take_profit_pct / signal.exit_plan.stop_loss_pct
            if rr < self.min_reward_to_risk - 1e-9:
                return {
                    "approved": False,
                    "size_multiplier": 0.0,
                    "reason": f"low_reward_to_risk:{rr:.2f}<{self.min_reward_to_risk:.2f}",
                }

        mult = 1.0
        reasons: List[str] = []
        if signal.size_usdt > self.max_size_usdt > 0:
            cap_mult = self.max_size_usdt / signal.size_usdt
            mult = min(mult, cap_mult)
            reasons.append(f"capped_to_{self.max_size_usdt:.0f}usdt")

        if self.scale_by_confidence:
            conf_mult = max(0.1, min(1.0, float(signal.router_confidence)))
            if conf_mult < mult:
                mult = conf_mult
                reasons.append(f"scaled_by_confidence:{conf_mult:.3f}")

        # Never exceed 1.0 (tighten-only invariant C18)
        mult = max(0.0, min(1.0, round(mult, 6)))
        if mult <= 0.0:
            return {"approved": False, "size_multiplier": 0.0, "reason": "zero_size_after_policy"}

        return {
            "approved": True,
            "size_multiplier": mult,
            "reason": ",".join(reasons) if reasons else "policy_approved",
        }


class AgentEndpointStore:
    """Thread-safe SQLite store for idempotent signal processing and close logs."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_signals (
                    signal_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    size_usdt REAL NOT NULL,
                    expert_instance_id TEXT NOT NULL,
                    regime_id TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    size_multiplier REAL NOT NULL,
                    order_id TEXT,
                    reason TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_position_closes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    exit_price REAL NOT NULL,
                    realized_pnl_pct REAL NOT NULL,
                    timestamp INTEGER NOT NULL
                );
                """
            )
            self._conn.commit()

    def get_response(self, signal_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM agent_signals WHERE signal_id = ?",
                (signal_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(json.loads(str(row["response_json"])))

    def get_signal_record(self, signal_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT signal_json, response_json FROM agent_signals WHERE signal_id = ?",
                (signal_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "signal": json.loads(str(row["signal_json"])),
                "response": json.loads(str(row["response_json"])),
            }

    def save_signal_response(self, signal: TradeSignal, response: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO agent_signals (
                    signal_id, timestamp, symbol, direction, size_usdt,
                    expert_instance_id, regime_id, approved, size_multiplier,
                    order_id, reason, signal_json, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.timestamp,
                    signal.symbol,
                    signal.direction,
                    signal.size_usdt,
                    signal.expert_instance_id,
                    signal.regime_id,
                    1 if response.get("approved") else 0,
                    float(response.get("size_multiplier", 0.0)),
                    response.get("order_id"),
                    str(response.get("reason", "")),
                    json.dumps(signal.to_canonical_dict(include_signature=True), sort_keys=True),
                    json.dumps(dict(response), sort_keys=True),
                ),
            )
            self._conn.commit()

    def record_close(self, payload: Mapping[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO agent_position_closes (
                    signal_id, trade_id, symbol, exit_reason,
                    exit_price, realized_pnl_pct, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload.get("signal_id", "")),
                    str(payload.get("trade_id", "")),
                    str(payload.get("symbol", "")),
                    str(payload.get("exit_reason", "")),
                    float(payload.get("exit_price", 0.0)),
                    float(payload.get("realized_pnl_pct", 0.0)),
                    int(payload.get("timestamp", 0)),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            sig_c = self._conn.execute("SELECT COUNT(*) AS c FROM agent_signals").fetchone()["c"]
            cls_c = self._conn.execute("SELECT COUNT(*) AS c FROM agent_position_closes").fetchone()["c"]
            return {"received_signals": int(sig_c), "closed_positions": int(cls_c)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class AgentEndpointProcessor:
    """Core processor for the External Agent Endpoint (`D9`)."""

    def __init__(
        self,
        *,
        policy: Optional[ExternalAgentPolicy] = None,
        store: Optional[AgentEndpointStore] = None,
        secret_key: Optional[str] = None,
        execution_callback: Optional[Callable[[TradeSignal, float], str]] = None,
    ) -> None:
        self.policy = policy or ExternalAgentPolicy()
        self.store = store or AgentEndpointStore(":memory:")
        self.secret_key = secret_key
        self.execution_callback = execution_callback

    def process_signal_dict(
        self,
        payload: Mapping[str, Any],
        *,
        header_signature: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Process a raw JSON dict and return `(http_status, response_dict)`."""
        try:
            signal = TradeSignal.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return 400, {"approved": False, "size_multiplier": 0.0, "reason": f"malformed_signal:{exc}"}

        if header_signature and not signal.hmac_signature:
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
                hmac_signature=header_signature,
            )

        if self.secret_key:
            if not verify_signal_hmac(signal, self.secret_key):
                return 401, {
                    "signal_id": signal.signal_id,
                    "approved": False,
                    "size_multiplier": 0.0,
                    "reason": "unauthorized_invalid_hmac",
                }
            if header_signature and not hmac.compare_digest(header_signature, signal.hmac_signature):
                return 401, {
                    "signal_id": signal.signal_id,
                    "approved": False,
                    "size_multiplier": 0.0,
                    "reason": "header_body_hmac_mismatch",
                }

        cached = self.store.get_response(signal.signal_id)
        if cached is not None:
            dup_resp = dict(cached)
            dup_resp["duplicate"] = True
            if self.secret_key:
                dup_resp["response_signature"] = compute_payload_hmac(dup_resp, self.secret_key)
            return 200, dup_resp

        decision = self.policy.evaluate(signal)
        approved = bool(decision["approved"])
        mult = float(decision["size_multiplier"])
        order_id: Optional[str] = None
        if approved:
            if self.execution_callback is not None:
                order_id = str(self.execution_callback(signal, signal.size_usdt * mult))
            else:
                order_id = f"ext-{signal.signal_id[:12]}"

        response: Dict[str, Any] = {
            "signal_id": signal.signal_id,
            "approved": approved,
            "accepted": approved,
            "size_multiplier": mult,
            "order_id": order_id,
            "reason": str(decision["reason"]),
            "duplicate": False,
        }
        self.store.save_signal_response(signal, response)
        if self.secret_key:
            response["response_signature"] = compute_payload_hmac(response, self.secret_key)
        return 200, response

    def process_close_dict(
        self,
        payload: Mapping[str, Any],
        *,
        header_signature: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        if self.secret_key:
            sig = header_signature or str(payload.get("hmac_signature", ""))
            expected = compute_payload_hmac(payload, self.secret_key)
            if not sig or not hmac.compare_digest(sig, expected):
                return 401, {"acknowledged": False, "reason": "unauthorized_invalid_hmac"}

        close_id = self.store.record_close(payload)
        resp: Dict[str, Any] = {
            "acknowledged": True,
            "close_id": close_id,
            "signal_id": str(payload.get("signal_id", "")),
            "trade_id": str(payload.get("trade_id", "")),
        }
        if self.secret_key:
            resp["response_signature"] = compute_payload_hmac(resp, self.secret_key)
        return 200, resp


class AgentEndpointHTTPServer:
    """Threaded HTTP server exposing the External Agent Endpoint (`D9`)."""

    def __init__(
        self,
        processor: AgentEndpointProcessor,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.processor = processor
        self.host = host
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.port = int(self._httpd.server_port)
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        proc = self.processor

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return  # keep test & daemon output quiet

            def _send_json(self, status: int, data: Mapping[str, Any]) -> None:
                body = json.dumps(dict(data), sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    counts = proc.store.counts()
                    self._send_json(200, {"status": "ok", **counts})
                    return
                if self.path.startswith("/v1/signals/"):
                    sig_id = self.path.split("/v1/signals/", 1)[1].strip("/")
                    rec = proc.store.get_signal_record(sig_id)
                    if rec is None:
                        self._send_json(404, {"error": "not_found", "signal_id": sig_id})
                    else:
                        self._send_json(200, rec)
                    return
                self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_bytes = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                except Exception as exc:
                    self._send_json(400, {"approved": False, "reason": f"invalid_json:{exc}"})
                    return

                hdr_sig = self.headers.get("X-Trade-Signature")
                if self.path == "/v1/signals":
                    status, resp = proc.process_signal_dict(payload, header_signature=hdr_sig)
                    self._send_json(status, resp)
                    return
                if self.path == "/v1/positions/close":
                    status, resp = proc.process_close_dict(payload, header_signature=hdr_sig)
                    self._send_json(status, resp)
                    return
                self._send_json(404, {"error": "not_found"})

        return _Handler

    def start_in_background(self) -> str:
        if self._thread is not None and self._thread.is_alive():
            return self.base_url
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
