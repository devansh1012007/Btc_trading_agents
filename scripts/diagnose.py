#!/usr/bin/env python
"""
CLI diagnostics tool (Phase 5, §4.3). Answers "what happened and why?" for
any bar, trade, or time period. Reads the same underlying decision trace
JSONL files that the dashboard reads — CLI and dashboard are two views
onto the same data, never two different systems that could disagree.

Usage:
    python scripts/diagnose.py --timestamp 1755900000
        [Full stage-by-stage trace for that bar, pretty-printed]

    python scripts/diagnose.py --trade-id T-8821
        [Entry trace, exit trace, full context for a completed trade]

    python scripts/diagnose.py --summary --since 2026-08-01
        Bars processed: 4,102
        No expert candidate:       2,890  (70.4%)
        Failed opportunity filter:   812  (19.8%)
        Rejected by router:           41  (1.0%)
        Rejected by risk layer:       97  (2.4%)
        Circuit breaker active:       12  (0.3%)
        Executed:                    250  (6.1%)
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def load_traces(base_dir: str = "data/logs/decision_traces") -> List[dict]:
    """Load all decision trace files from the base directory."""
    traces_path = Path(base_dir)
    if not traces_path.exists():
        return []
    all_traces = []
    for f in sorted(traces_path.glob("traces_*.jsonl")):
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_traces.append(json.loads(line))
    return all_traces


def load_faults(path: str = "data/logs/faults.jsonl") -> List[dict]:
    """Load all fault records."""
    p = Path(path)
    if not p.exists():
        return []
    faults = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                faults.append(json.loads(line))
    return faults


def print_trace(trace: dict) -> None:
    """Pretty-print a single decision trace."""
    print(f"\n{'='*60}")
    print(f"Decision Trace: {trace['timestamp']} | {trace['symbol']}")
    print(f"Terminal: {trace['terminal_stage']} — {trace['terminal_reason']}")
    print(f"{'='*60}")
    for i, stage in enumerate(trace['stages'], 1):
        status = "PASS" if stage['passed'] else "FAIL"
        print(f"  [{i}] {stage['stage_name']}: {status}")
        print(f"      {stage['detail']}")
        if stage['values']:
            vals = ", ".join(f"{k}={v:.4f}" for k, v in stage['values'].items())
            print(f"      Values: {vals}")
    print()


def print_summary(traces: List[dict], since: Optional[str] = None) -> None:
    """Print a summary breakdown of why bars stopped at each stage."""
    if since:
        since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        since_ts = int(since_dt.timestamp())
        traces = [t for t in traces if t['timestamp'] >= since_ts]

    if not traces:
        print("No traces found.")
        return

    total = len(traces)
    # Count by terminal stage
    terminal_counts: Dict[str, int] = {}
    for t in traces:
        stage = t['terminal_stage']
        terminal_counts[stage] = terminal_counts.get(stage, 0) + 1

    print(f"\n{'='*60}")
    print(f"Decision Summary: {total} bars processed")
    if since:
        print(f"  (since {since})")
    print(f"{'='*60}")

    # Human-readable labels for terminal stages
    stage_labels = {
        "no_expert_candidate": "No expert candidate",
        "opportunity_filter": "Failed opportunity filter",
        "router_no_trade": "Rejected by router",
        "risk_layer": "Rejected by risk layer",
        "circuit_breaker": "Circuit breaker active",
        "executed": "Executed",
    }

    for stage, count in sorted(terminal_counts.items(), key=lambda x: -x[1]):
        label = stage_labels.get(stage, stage)
        pct = count / total * 100
        print(f"  {label + ':':<30s} {count:>6d}  ({pct:>5.1f}%)")
    print()


def print_faults(faults: List[dict]) -> None:
    """Print fault records."""
    if not faults:
        print("No faults recorded.")
        return

    print(f"\n{'='*60}")
    print(f"System Faults: {len(faults)} recorded")
    print(f"{'='*60}")
    for i, fault in enumerate(faults, 1):
        print(f"\n  [{i}] {fault['timestamp']} | {fault['symbol']}")
        print(f"      {fault['exception_type']}: {fault['exception_message']}")
        if fault.get('stack_trace'):
            print(f"      Stack trace: {fault['stack_trace'][:200]}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnostics: what happened and why?")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--timestamp", type=int, help="Print full trace for a specific bar timestamp")
    group.add_argument("--trade-id", type=str, help="Print trace for a completed trade (future: joins with journal)")
    group.add_argument("--summary", action="store_true", help="Print summary breakdown of all bars")
    group.add_argument("--faults", action="store_true", help="Print all system fault records")
    parser.add_argument("--since", type=str, help="Only include traces since YYYY-MM-DD (used with --summary)")

    args = parser.parse_args()

    traces = load_traces()

    if args.timestamp:
        matching = [t for t in traces if t['timestamp'] == args.timestamp]
        if not matching:
            print(f"No trace found for timestamp {args.timestamp}")
            sys.exit(1)
        for t in matching:
            print_trace(t)

    elif args.summary:
        print_summary(traces, since=args.since)

    elif args.faults:
        faults = load_faults()
        print_faults(faults)

    elif args.trade_id:
        print(f"Trade trace for {args.trade_id}: not yet implemented (requires trade journal join)")


if __name__ == "__main__":
    main()
