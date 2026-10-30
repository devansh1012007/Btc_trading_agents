#!/usr/bin/env python
"""
Bundle evaluation & ablation report CLI (`scripts/evaluate_bundle.py`, Phase 17 T17.2).

Usage:
    python scripts/evaluate_bundle.py --bundle-hash demo_bundle --out data/reports/eval_bundle.json
    python scripts/evaluate_bundle.py --bundle-hash demo_bundle --run-replay --synthetic-bars 120
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ablation_ladder import (  # noqa: E402
    ABLATION_RUNGS,
    AblationRungResult,
    FinalEvaluationGate,
    run_real_ablation_ladder,
)
from src.training.runners import make_synthetic_training_ohlcv  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402


def build_default_synthetic_rungs() -> List[AblationRungResult]:
    return [
        AblationRungResult(
            rung_id=rung,
            expectancy_pct=0.05 + 0.01 * i,
            profit_factor=1.10 + 0.02 * i,
            max_drawdown_pct=max(1.5, 4.0 - 0.1 * i),
            brier_score=max(0.12, 0.22 - 0.005 * i),
            min_duration_violations=0,
            p_value_vs_prev=0.01,
        )
        for i, rung in enumerate(ABLATION_RUNGS)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ModelBundle across L0-L10 ablation rungs")
    parser.add_argument("--bundle-hash", default="synthetic_bundle_v1")
    parser.add_argument("--env", default="paper", help="Environment config layer")
    parser.add_argument("--run-replay", action="store_true", help="Run real pipeline replay across L0-L10")
    parser.add_argument("--synthetic-bars", type=int, default=120, help="Number of bars for replay evaluation")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    gate = FinalEvaluationGate()
    if args.run_replay:
        cfg = load_layered_config(args.env)
        df = make_synthetic_training_ohlcv(n_bars=max(110, int(args.synthetic_bars)))
        rungs = run_real_ablation_ladder(df, symbol="BTC/USDT", base_config=cfg)
        last = rungs[-1]
        holdout_metrics = {
            "expectancy_pct": last.expectancy_pct,
            "max_drawdown_pct": last.max_drawdown_pct,
            "min_duration_violations": last.min_duration_violations,
        }
    else:
        rungs = build_default_synthetic_rungs()
        holdout_metrics = {"expectancy_pct": 0.16, "max_drawdown_pct": 2.8, "min_duration_violations": 0}

    report = gate.evaluate_holdout(
        args.bundle_hash,
        rungs,
        holdout_metrics=holdout_metrics,
    )
    payload = {
        "bundle_hash": report.bundle_hash,
        "passed": report.passed,
        "reasons": report.reasons,
        "holdout_metrics": report.holdout_metrics,
        "rungs": [
            {
                "rung_id": r.rung_id,
                "expectancy_pct": r.expectancy_pct,
                "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "brier_score": r.brier_score,
                "adjusted_p_value": r.adjusted_p_value,
                "passed_gate": r.passed_gate,
                "notes": r.notes,
            }
            for r in report.rungs
        ],
    }
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
