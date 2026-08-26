"""
Structured logging for the trading pipeline (PRESENTATION §8).

Provides:
* JsonLineFormatter -- serializes log records to JSON, inlining ``extra``.
* setup_logging    -- idempotent JSONL handler on the root logger.
* decision_record_from_state -- flatten a SystemState into an §8-shaped dict.
* log_trade_decision -- append one decision record per bar to a JSONL file.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.state.contracts import SystemState

_DEFAULT_DECISIONS_PATH = "data/logs/decisions.jsonl"
_DEFAULT_SYSTEM_LOG_PATH = "data/logs/system.jsonl"

_LOGGER_NAME = "btc_trading_agents"
_setup_done = False


class JsonLineFormatter(logging.Formatter):
    """Emit each log record as a single JSON line, inlining ``extra`` keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Inline extra fields (everything not in the standard LogRecord attrs).
        std_attrs = set(logging.LogRecord(
            "", 0, "", 0, "", None, None).__dict__.keys()
        )
        for key, value in record.__dict__.items():
            if key in std_attrs or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    json_log_file: str = _DEFAULT_SYSTEM_LOG_PATH,
    level: int = logging.INFO,
) -> None:
    """Configure the root logger with a JSONL streaming handler.

    Idempotent: calling it multiple times won't stack duplicate handlers.
    """
    global _setup_done
    root = logging.getLogger()
    root.setLevel(level)

    Path(json_log_file).parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate handlers on repeated calls.
    for handler in list(root.handlers):
        if getattr(handler, "_is_btc_agent_jsonl", False):
            root.removeHandler(handler)

    handler = logging.FileHandler(json_log_file, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    handler.setLevel(level)
    handler._is_btc_agent_jsonl = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Also emit to stderr in a readable form so the console isn't silent.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    console._is_btc_agent_console = True  # type: ignore[attr-defined]
    root.addHandler(console)

    _setup_done = True


def decision_record_from_state(state: SystemState) -> dict[str, Any]:
    """Flatten a ``SystemState`` into the §8 observability record."""
    regime_state = state.regime_state
    router = state.router_decision
    opp = state.opportunity_score
    risk = state.risk_decision
    exec_record = state.execution_record

    regime_name = (
        regime_state.primary_regime.value.lower() if regime_state and regime_state.primary_regime else "unknown"
    )

    expert_name = router.selected_expert if router and router.selected_expert else None

    if exec_record and exec_record.side == "BUY":
        decision = "LONG"
    elif exec_record and exec_record.side == "SELL":
        decision = "SHORT"
    else:
        decision = "NO_TRADE"

    return {
        "symbol": state.symbol,
        "timestamp": state.timestamp,
        "regime": regime_name,
        "expert": expert_name,
        "opportunity_score": opp.tradable_probability if opp else None,
        "router_confidence": router.selection_confidence if router else None,
        "risk_score": risk.risk_score if risk else None,
        "decision": decision,
        "equity": None,  # filled by caller if known
    }


def log_trade_decision(record: dict[str, Any], *, path: Optional[str] = None) -> None:
    """Append a single decision record as one JSON line."""
    target = path or os.environ.get("DECISIONS_JSONL", _DEFAULT_DECISIONS_PATH)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
