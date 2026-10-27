"""
Dashboard data layer (Phase 5, Module G / §10 — TUI variant).

Pure-stdlib loaders/aggregators over the exact artifacts the pipeline
already writes — decision-trace JSONL, faults JSONL, registry.db, and
backtest reports. No rich/textual imports here: this module is fully
unit-testable headless, and both the live TUI (scripts/dashboard.py) and
its --once static renderer consume the same snapshot dicts.

Deliberately reads the SAME files scripts/diagnose.py reads — CLI and
dashboard are two views onto one dataset, never two systems that could
disagree (§4.4 / §10 design constraint).
"""
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

TRACE_DIR = "data/logs/decision_traces"
FAULTS_PATH = "data/logs/faults.jsonl"
REGISTRY_DB = "data/registry.db"
REPORT_DIR = "data/reports"
KILL_SWITCH_PATH = "data/control/EMERGENCY_STOP"

# Canonical stage names, their human-readable labels AND their console
# styles live in src/diagnostics/decision_trace.py -- one vocabulary, so
# the TUI dashboard, the diagnostics CLI and the Trade Map GUI can never
# disagree about what a stage is called or how it is coloured. They are
# re-exported here for the TUI's existing ``dd.STAGE_STYLES`` call sites.
#
# The drift guard between the two is a UNIT TEST (tests/unit/
# test_decision_trace.py), not an import-time assert: an assert here
# would turn a routine vocabulary addition into a boot failure of the
# whole GUI package, and it is stripped under `python -O` anyway.
from .decision_trace import STAGE_LABELS, STAGE_STYLES, TERMINAL_STAGES  # noqa: E402,F401


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def load_decision_traces(base_dir: str = TRACE_DIR) -> List[dict]:
    """All decision traces across date-partitioned JSONL files."""
    d = Path(base_dir)
    if not d.exists():
        return []
    out: List[dict] = []
    for f in sorted(d.glob("traces_*.jsonl")):
        try:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # torn tail line mid-write — skip, don't crash
        except OSError:
            continue
    return out


def funnel_breakdown(traces: List[dict]) -> List[dict]:
    """
    Terminal-stage funnel rows, ordered most-frequent first:
    [{stage, label, count, pct}] — the exact aggregation behind §10's
    Decision Funnel view and diagnose.py --summary.
    """
    total = len(traces)
    counts = Counter(t.get("terminal_stage", "?") for t in traces)
    rows = []
    for stage, n in counts.most_common():
        rows.append({
            "stage": stage,
            "label": stage_label(stage),
            "count": n,
            "pct": (n / total * 100) if total else 0.0,
        })
    return rows


def stage_pass_rates(traces: List[dict]) -> List[dict]:
    """
    Per-stage pass rates across every bar that reached that stage —
    §10's Signal Path Overview numbers.
    """
    stats: Dict[str, Counter] = defaultdict(Counter)
    for t in traces:
        for s in t.get("stages", []):
            name = s.get("stage_name", "?")
            stats[name]["n"] += 1
            if s.get("passed"):
                stats[name]["passed"] += 1
    rows = []
    for name, c in stats.items():
        n = c["n"]
        rows.append({
            "stage_name": name,
            "reached": n,
            "passed": c["passed"],
            "pass_pct": (c["passed"] / n * 100) if n else 0.0,
        })
    rows.sort(key=lambda r: -r["reached"])
    return rows


def load_faults(path: str = FAULTS_PATH, limit: int = 50) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    faults: List[dict] = []
    try:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    faults.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return faults[-limit:]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def db_snapshot(db_path: str = REGISTRY_DB) -> dict:
    """
    Guarded read of everything the dashboard shows from registry.db.
    Missing DB or missing tables degrade to empty structures — a fresh
    checkout must render an empty dashboard, never crash.
    """
    snap = {
        "db_present": Path(db_path).exists(),
        "active_breakers": [],
        "breaker_history": [],
        "risk_state": {},
        "recent_trades": [],
        "open_trades": 0,
        "trade_summary_today": {"trades_today": 0, "total_pnl_usdt_today": None,
                                "win_rate_today": None},
        "models": [],
        "experiments": [],
    }
    if not snap["db_present"]:
        return snap
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        snap["db_present"] = False
        return snap

    try:
        if _table_exists(conn, "circuit_breaker_halts"):
            snap["active_breakers"] = [dict(r) for r in conn.execute(
                "SELECT breaker_type, tripped_at, reason FROM circuit_breaker_halts "
                "WHERE active=1 ORDER BY tripped_at DESC")]
            snap["breaker_history"] = [dict(r) for r in conn.execute(
                "SELECT breaker_type, tripped_at, reason, active, reset_at "
                "FROM circuit_breaker_halts ORDER BY tripped_at DESC LIMIT 20")]

        if _table_exists(conn, "risk_state"):
            snap["risk_state"] = {r["key"]: r["value"] for r in
                                  conn.execute("SELECT key, value FROM risk_state")}

        if _table_exists(conn, "trade_journal"):
            snap["recent_trades"] = [dict(r) for r in conn.execute(
                "SELECT trade_id, entry_timestamp, exit_timestamp, entry_price, exit_price,"
                " exit_reason, expert_used, regime_at_entry, direction, realized_pnl_pct,"
                " realized_pnl_usdt, router_confidence, mode"
                " FROM trade_journal ORDER BY entry_timestamp DESC LIMIT 25")]
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM trade_journal WHERE exit_timestamp IS NULL").fetchone()
            snap["open_trades"] = int(row["n"]) if row else 0

        if _table_exists(conn, "model_versions"):
            snap["models"] = [dict(r) for r in conn.execute(
                "SELECT component, version_id, status, trained_at, data_granularity"
                " FROM model_versions ORDER BY trained_at DESC LIMIT 30")]

        if _table_exists(conn, "experiments"):
            snap["experiments"] = [dict(r) for r in conn.execute(
                "SELECT experiment_id, run_type, component, started_at, config_hash,"
                " metrics_json FROM experiments ORDER BY started_at DESC LIMIT 15")]
    finally:
        conn.close()
    return snap


def load_latest_report(report_dir: str = REPORT_DIR) -> Optional[dict]:
    """Newest backtest_*.json by mtime, or None."""
    d = Path(report_dir)
    if not d.exists():
        return None
    candidates = sorted(d.glob("backtest_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            with open(p, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def report_summary(rep: Optional[dict]) -> dict:
    """Flatten a backtest report into dashboard-friendly rows."""
    if not rep:
        return {"present": False}
    agg = rep.get("aggregate", {})
    baselines = rep.get("baselines", {})
    return {
        "present": True,
        "generated_at": rep.get("generated_at", "?"),
        "config_hash": rep.get("config_hash", "?"),
        "aggregate": agg,
        "criteria": rep.get("success_criteria", []),
        "baselines": [
            {"name": k, **{m: v.get(m) for m in
                           ("n_trades", "total_return_pct", "win_rate_pct",
                            "expectancy_per_trade_pct", "sharpe_ratio",
                            "max_drawdown_pct")}}
            for k, v in baselines.items()
        ],
        "verdicts": rep.get("baseline_verdicts", {}),
        "windows": [
            {"index": w.get("index"), "n_bars": w.get("n_bars"),
             "return_pct": (w.get("metrics") or {}).get("total_return_pct")}
            for w in rep.get("windows", [])
        ],
        "calibration_experts": {
            name: {"buckets": buckets,
                   "score": (rep.get("calibration", {}).get(
                       "expert_miscalibration_scores", {}) or {}).get(name)}
            for name, buckets in (rep.get("calibration", {}).get("experts", {}) or {}).items()
        },
        "router_calibration": {
            "buckets": rep.get("calibration", {}).get("router", []),
            "score": rep.get("calibration", {}).get("router_miscalibration_score"),
        },
    }


def system_status(db_path: str = REGISTRY_DB) -> dict:
    """Top-strip status: kill switch, active breakers, open positions."""
    return {
        "kill_switch_active": os.path.exists(KILL_SWITCH_PATH),
        "kill_switch_path": KILL_SWITCH_PATH,
    }


def build_snapshot(trace_dir: str = TRACE_DIR, db_path: str = REGISTRY_DB,
                   report_dir: str = REPORT_DIR) -> dict:
    """One call, everything the dashboard renders."""
    traces = load_decision_traces(trace_dir)
    return {
        "status": system_status(db_path),
        "funnel": funnel_breakdown(traces),
        "stage_rates": stage_pass_rates(traces),
        "n_traces": len(traces),
        "faults": load_faults(),
        "db": db_snapshot(db_path),
        "report": report_summary(load_latest_report(report_dir)),
    }
