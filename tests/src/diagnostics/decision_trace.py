"""
Decision Trace (Phase 5, §4): one record per bar capturing the full
stage-by-stage journey through the pipeline. Answers "who failed, and how
exactly?" for both decision-layer rejections (legitimate no-trade decisions)
and system faults (actual failures).

Two categories of failure:
1. Decision-layer rejections: no expert candidate, opportunity filter veto,
   risk rejection, circuit breaker active. Information, not bugs — but
   needs to be legible and aggregable.
2. System faults: exception, connectivity failure, stale data feed. Needs
   full context, reproducible, tied into consecutive-error circuit breaker.

Written to data/logs/decision_traces/ as partitioned JSONL (high volume,
append-only, read with pandas). Kept separate from the trade journal
(completed trades only) and the full SystemState Parquet archive (complete
numeric record, optimized for training) — this is specifically the
human-readable "why" narrative, optimized for debugging.
"""
import json
import logging
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageResult:
    """One stage's verdict in the decision journey."""
    stage_name: str          # "trend_expert_gate", "opportunity_filter", "router", "risk_layer", ...
    passed: bool
    detail: str               # human-readable: "momentum 0.03% below threshold 0.05%"
    values: Dict[str, float]  # actual numbers, for reproducibility
    duration_ms: Optional[float] = None  # wall-clock stage latency (PRESENTATION §23)


@dataclass(frozen=True)
class DecisionTrace:
    """One record per bar — the full stage-by-stage journey."""
    timestamp: int
    symbol: str
    stages: List[StageResult]     # every stage the bar passed through, in order
    terminal_stage: str            # where the journey ended
    terminal_reason: str
    total_duration_ms: Optional[float] = None           # end-to-end decision latency for this bar
    stage_timings_ms: Optional[Dict[str, float]] = None  # pipeline-internal per-stage latencies


@dataclass(frozen=True)
class SystemFault:
    """A system fault record — exception with full context."""
    timestamp: int
    symbol: str
    exception_type: str
    exception_message: str
    stack_trace: str
    system_state_snapshot: Optional[Dict[str, Any]]
    last_known_good_state_timestamp: Optional[int]


class DecisionTraceWriter:
    """Append-only JSONL writer for decision traces."""

    def __init__(self, base_dir: str = "data/logs/decision_traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _trace_path(self, timestamp: int) -> Path:
        """Partition by date for efficient reading."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        return self.base_dir / f"traces_{date_str}.jsonl"

    def append(self, trace: DecisionTrace) -> None:
        path = self._trace_path(trace.timestamp)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(trace)) + "\n")

    def append_many(self, traces: List[DecisionTrace]) -> None:
        for trace in traces:
            self.append(trace)


class FaultLogWriter:
    """Append-only JSONL writer for system faults."""

    def __init__(self, path: str = "data/logs/faults.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_fault(self, exception: Exception, symbol: str,
                  system_state_snapshot: Optional[Dict[str, Any]] = None,
                  timestamp: Optional[int] = None) -> None:
        ts = timestamp or int(datetime.now(timezone.utc).timestamp())
        fault = SystemFault(
            timestamp=ts,
            symbol=symbol,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            system_state_snapshot=system_state_snapshot,
            last_known_good_state_timestamp=None,  # set by caller if available
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(fault)) + "\n")
        logger.error(f"Fault logged: {fault.exception_type}: {fault.exception_message}")
