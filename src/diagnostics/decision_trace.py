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
from typing import Any, Dict, List, Optional, Tuple

from ..state.contracts import ExpertAction, SystemState

logger = logging.getLogger(__name__)

#: Terminal stages a bar's decision journey can end on, in pipeline order.
#: This is the single source of truth for the vocabulary -- the TUI
#: dashboard, the diagnostics CLI and the Trade Map GUI all read it.
TERMINAL_STAGES: Tuple[str, ...] = (
    "no_expert_candidate",
    "opportunity_filter",
    "router_no_trade",
    "position_open",
    "risk_layer",
    "circuit_breaker",
    "execution_failed",
    "executed",
)

#: Canonical human-readable label per terminal stage. It lives HERE, next
#: to the stage names themselves, so the two cannot drift apart: a stage
#: and its label are one fact, not two.
STAGE_LABELS: Dict[str, str] = {
    "no_expert_candidate": "No expert candidate",
    "opportunity_filter": "Failed opportunity filter",
    "router_no_trade": "Rejected by router",
    "position_open": "Position already open",
    "risk_layer": "Rejected by risk layer",
    "circuit_breaker": "Circuit breaker active",
    "execution_failed": "Execution failed",
    "executed": "Executed",
}

#: Risk-layer rejection reasons that are CONSEQUENCES of an earlier stage
#: having already declined, rather than independent risk decisions. The
#: risk layer is consulted for every bar the router produced a decision
#: for, so it answers "no_trade_selected" on bars the filter or the
#: router already turned down.
_DERIVED_RISK_REJECTIONS = ("no_trade_selected", "selected_expert_has_no_candidate")


def stage_label(stage: str) -> str:
    """Human-readable label for a terminal stage."""
    return STAGE_LABELS.get(stage, stage)


def classify_terminal_stage(
    opportunity_score: Optional[Any] = None,
    router_decision: Optional[Any] = None,
    risk_decision: Optional[Any] = None,
    execution_record: Optional[Any] = None,
    position_open: bool = False,
    execution_failed: bool = False,
    execution_error: Optional[str] = None,
) -> Tuple[str, str]:
    """Classify where a bar's decision journey ended.

    The classification follows the pipeline's own order (experts ->
    opportunity filter -> router -> risk layer -> execution) and reports
    the FIRST thing that stopped the bar. Five subtleties matter:

    * A genuine **halt** -- the manual kill switch or a tripped circuit
      breaker -- outranks everything else. The system was not allowed to
      trade at all, so no downstream stage is the interesting answer.
    * The risk layer is consulted for every bar the router produced a
      decision for, including bars where the router already declined
      ("no_trade"). Its rejection reason is then ``no_trade_selected`` --
      a *consequence* of the filter or the router, not an independent
      risk rejection. Checking the filter and the router before
      crediting the risk layer stops every filter veto from being
      mislabelled as a risk rejection.
    * ``Router.decide`` returns ``no_trade`` with an explicit
      ``reasoning`` string that distinguishes "the filter vetoed" from
      "no expert cleared the confidence floor" from "the model preferred
      no trade". Those map onto different terminal stages.
    * When a position was already open the bar could not open a second
      one. That is neither a rejection nor a trade, and labelling it
      "no expert candidate" would be plainly false.
    * ``execution_failed`` says the order placement **raised**. Without
      it a failed order leaves no ``ExecutionRecord`` at all, and the
      bar would be reported as "no expert candidate" -- naming a stage
      that never had anything to do with the failure.

    Returns a ``(stage, reason)`` pair.
    """
    if execution_failed:
        if execution_error:
            return "execution_failed", f"order placement failed: {execution_error}"
        return "execution_failed", "order placement failed"

    if execution_record is not None:
        if execution_record.status in ("filled", "partial"):
            expert = router_decision.selected_expert if router_decision else "unknown"
            return "executed", f"filled via {expert}"
        return "execution_failed", f"status={execution_record.status}"

    if risk_decision is not None and not risk_decision.approved:
        reason = risk_decision.rejection_reason or ""
        if reason.startswith("circuit_breaker_active") or reason == "manual_kill_switch_active":
            return "circuit_breaker", reason
        if reason == "selected_expert_has_no_candidate":
            return "no_expert_candidate", "no expert candidate"
        if reason not in _DERIVED_RISK_REJECTIONS:
            return "risk_layer", reason or "risk rejection"
        # "no_trade_selected": the router declined -- classify below.

    if opportunity_score is not None and not opportunity_score.tradable:
        failed = list(getattr(opportunity_score, "failed_rules", None) or [])
        return "opportunity_filter", (" | ".join(failed) if failed else "filter veto")

    if router_decision is not None and router_decision.selected_expert == "no_trade":
        reasoning = router_decision.reasoning or ""
        if reasoning == "opportunity_filter_rejected":
            return "opportunity_filter", "filter veto"
        if reasoning == "no_expert_cleared_confidence_floor":
            return "no_expert_candidate", "no expert candidate"
        return "router_no_trade", reasoning or "router declined"

    if position_open:
        return "position_open", "position already open"

    if risk_decision is not None and not risk_decision.approved:
        reason = risk_decision.rejection_reason or ""
        return "risk_layer", reason or "risk rejection"

    return "no_expert_candidate", "no expert candidate"



@dataclass(frozen=True)
class StageResult:
    """One stage's verdict in the decision journey."""
    stage_name: str          # "trend_expert_gate", "opportunity_filter", "router", "risk_layer", ...
    passed: bool
    detail: str               # human-readable: "momentum 0.03% below threshold 0.05%"
    values: Dict[str, float]  # actual numbers, for reproducibility
    duration_ms: Optional[float] = None  # wall-clock latency of this stage


@dataclass(frozen=True)
class DecisionTrace:
    """One record per bar — the full stage-by-stage journey."""
    timestamp: int
    symbol: str
    stages: List[StageResult]     # every stage the bar passed through, in order
    terminal_stage: str            # where the journey ended
    terminal_reason: str
    total_duration_ms: Optional[float] = None            # end-to-end latency for this bar
    stage_timings_ms: Optional[Dict[str, float]] = None  # pipeline-internal per-stage latencies

def build_trace_stages(
    state: SystemState,
    timings: Optional[Dict[str, float]] = None,
) -> List[StageResult]:
    """Build the stage-by-stage verdicts for one bar's ``SystemState``.

    Shared by the deployment loop (``scripts/run.py``) and the Trade Map
    GUI (``src/gui/replay.py``) so that a replayed trace is byte-for-byte
    the same shape as a live one -- otherwise the GUI would be showing
    its own opinion of the pipeline rather than the pipeline's own.

    Stages that produced no result for this bar are omitted (a bar with
    no router decision has no router stage).

    ``timings`` carries the per-stage wall-clock milliseconds measured by
    the caller (``{"risk_layer_ms": 0.4, ...}``). Each stage gets the
    duration of the work it actually represents, so a stage's
    ``duration_ms`` is never another stage's number.
    """
    stages: List[StageResult] = []
    stage_timings = timings or {}

    def duration(key: str) -> Optional[float]:
        value = stage_timings.get(key)
        return float(value) if value is not None else None

    expert_outputs = state.expert_outputs
    if expert_outputs and expert_outputs.all():
        candidates = [o for o in expert_outputs.all() if o.action != ExpertAction.NO_TRADE]
        if candidates:
            stages.append(StageResult(
                stage_name="expert_layer", passed=True,
                detail=f"{len(candidates)} candidate(s): {', '.join(c.expert_name for c in candidates)}",
                values={o.expert_name: o.confidence for o in candidates},
                duration_ms=duration("expert_layer_ms"),
            ))
        else:
            stages.append(StageResult(
                stage_name="expert_layer", passed=False,
                detail="All experts returned NO_TRADE",
                values={},
                duration_ms=duration("expert_layer_ms"),
            ))
    else:
        stages.append(StageResult(
            stage_name="expert_layer", passed=False,
            detail="No expert outputs",
            values={},
            duration_ms=duration("expert_layer_ms"),
        ))

    opp_score = state.opportunity_score
    if opp_score:
        stages.append(StageResult(
            stage_name="opportunity_filter", passed=opp_score.tradable,
            detail=" ".join(opp_score.failed_rules) if opp_score.failed_rules else "passed",
            values={"tradable_probability": opp_score.tradable_probability,
                    "environment_quality": opp_score.environment_quality},
            duration_ms=duration("opportunity_filter_ms"),
        ))

    router_decision = state.router_decision
    if router_decision:
        stages.append(StageResult(
            stage_name="router", passed=router_decision.selected_expert != "no_trade",
            detail=router_decision.reasoning or "",
            values={"selection_confidence": router_decision.selection_confidence,
                    "no_trade_probability": router_decision.no_trade_probability},
            duration_ms=duration("router_ms"),
        ))

    risk_decision = state.risk_decision
    if risk_decision:
        # A derived rejection ("no_trade_selected") means the risk layer
        # was only asked for form's sake -- the router had already
        # declined. Reporting it as a FAILED risk stage would make the
        # dashboard read "the risk layer vetoed 99% of bars", which is
        # not what happened.
        derived = risk_decision.rejection_reason in _DERIVED_RISK_REJECTIONS
        stages.append(StageResult(
            stage_name="risk_layer", passed=risk_decision.approved,
            detail=("not evaluated (router declined)"
                    if derived else (risk_decision.rejection_reason or "")),
            values={"risk_score": risk_decision.risk_score,
                    "size_reduction_pct": risk_decision.size_reduction_pct},
            duration_ms=duration("risk_layer_ms"),
        ))

    exec_record = state.execution_record
    if exec_record:
        stages.append(StageResult(
            stage_name="execution", passed=exec_record.status in ("filled", "partial"),
            detail=f"status={exec_record.status}",
            values={"fill_price": exec_record.fill_price, "slippage_pct": exec_record.slippage_pct},
            duration_ms=duration("execution_ms"),
        ))

    return stages


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
