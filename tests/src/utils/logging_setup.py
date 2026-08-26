"""
Structured logging (PRESENTATION §8): two outputs, one call site.

1. Human console log — unchanged stdlib formatting, what developers read.
2. data/logs/system.jsonl — every log record as one JSON object
   (timestamp, level, logger, message + any extra key/values). Machine-
   readable, greppable, ingestible by any log pipeline.

Plus a dedicated decision stream: data/logs/decisions.jsonl gets exactly
one record per bar with the observability fields that matter (regime,
expert, opportunity score, router confidence, risk score, final decision)
— the "what did the system do and why" feed the dashboard and demos read.

The JSON file handler is opt-in per process: scripts call setup_logging()
at startup. Modules keep using logging.getLogger(__name__) unchanged.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SYSTEM_LOG = "data/logs/system.jsonl"
DEFAULT_DECISION_LOG = "data/logs/decisions.jsonl"


class JsonLineFormatter(logging.Formatter):
    """One JSON object per LogRecord; extra attributes are inlined."""

    RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
        "asctime", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO,
                  json_log_file: Optional[str] = DEFAULT_SYSTEM_LOG) -> None:
    """
    Idempotent root-logger configuration: human console output always,
    plus a JSONL file handler when json_log_file is given.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Replace whatever basicConfig put in place so re-calls are safe.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(console)

    if json_log_file:
        path = Path(json_log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        root.addHandler(file_handler)


def log_trade_decision(record: Dict[str, Any], path: str = DEFAULT_DECISION_LOG) -> None:
    """Append one bar's decision summary to the decisions JSONL stream."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def decision_record_from_state(state) -> Dict[str, Any]:
    """
    Build the §8-style observability record from a SystemState.
    Never raises: a partially-populated state still logs what it has.
    """
    regime = regime_confidence = None
    if state.regime_state is not None:
        regime = state.regime_state.primary_regime.value
        regime_confidence = round(state.regime_state.regime_confidence, 4)

    selected_expert = router_confidence = None
    if state.router_decision is not None:
        selected_expert = state.router_decision.selected_expert
        router_confidence = round(state.router_decision.selection_confidence, 4)

    opportunity_score = None
    if state.opportunity_score is not None:
        opportunity_score = round(state.opportunity_score.tradable_probability, 4)

    risk_score = risk_approved = None
    if state.risk_decision is not None:
        risk_score = round(state.risk_decision.risk_score, 4)
        risk_approved = state.risk_decision.approved

    decision = "NO_TRADE"
    expert_confidences: Dict[str, float] = {}
    if state.expert_outputs is not None:
        for output in state.expert_outputs.all():
            expert_confidences[output.expert_name] = round(output.confidence, 4)
    execution = state.execution_record
    if execution is not None and execution.status in ("filled", "partial"):
        decision = "LONG" if execution.side == "BUY" else "SHORT"
    elif selected_expert and selected_expert != "no_trade":
        decision = f"BLOCKED:{selected_expert}"

    return {
        "timestamp": state.timestamp,
        "symbol": state.symbol,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "expert": selected_expert,
        "router_confidence": router_confidence,
        "opportunity_score": opportunity_score,
        "risk_score": risk_score,
        "risk_approved": risk_approved,
        "decision": decision,
        "expert_confidences": expert_confidences,
    }
