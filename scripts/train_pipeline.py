#!/usr/bin/env python
"""
Harness-ordered training DAG orchestrator (`scripts/train_pipeline.py`, Phase 12 T12.3).

Usage:
    python scripts/train_pipeline.py --plan
    python scripts/train_pipeline.py --execute --bundle-out data/models/bundles/latest_bundle.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.plan import TrainingPlan  # noqa: E402
from src.training.runners import (  # noqa: E402
    build_bundle_from_training_context,
    build_real_stage_runners,
    make_synthetic_training_ohlcv,
)
from src.utils.config_helpers import load_layered_config  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Harness-ordered training DAG orchestrator")
    parser.add_argument("--env", default="paper", help="Environment config layer (default: paper)")
    parser.add_argument("--plan", action="store_true", help="Print the topological training plan without executing")
    parser.add_argument("--execute", action="store_true", help="Execute the training plan in topological order")
    parser.add_argument("--data-path", default=None, help="Optional CSV/Parquet OHLCV input path")
    parser.add_argument("--synthetic-bars", type=int, default=180, help="Number of synthetic bars when --data-path is omitted")
    parser.add_argument("--artefacts-dir", default="data/models/bundle_run", help="Output directory for trained stage artefacts")
    parser.add_argument("--bundle-out", default=None, help="Optional output path for the ModelBundle manifest JSON")
    args = parser.parse_args(argv)

    config = load_layered_config(args.env)
    plan = TrainingPlan(config=config)

    if args.plan or not args.execute:
        print(plan.render_plan_table())
        return 0

    if args.data_path:
        dp = Path(args.data_path)
        df_ohlcv = pd.read_parquet(dp) if dp.suffix.lower() == ".parquet" else pd.read_csv(dp)
    else:
        df_ohlcv = make_synthetic_training_ohlcv(n_bars=max(80, int(args.synthetic_bars)))

    runners = build_real_stage_runners(
        df_ohlcv,
        symbol=str((config.get("system") or {}).get("symbol", "BTC/USDT")),
        config=config,
        output_dir=args.artefacts_dir,
    )
    ctx = plan.execute(runners=runners)
    bundle = build_bundle_from_training_context(
        plan,
        ctx,
        config=config,
        df_ohlcv=df_ohlcv,
    )
    problem = bundle.verify_consistency(check_files_on_disk=True)
    if problem is not None:
        raise RuntimeError(f"Trained ModelBundle failed integrity verification: {problem}")

    if args.bundle_out:
        out_p = Path(args.bundle_out)
        bundle.save(out_p)
        print(f"Wrote verified ModelBundle ({bundle.bundle_hash[:12]}) to {out_p}")
    else:
        print(json.dumps(bundle.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
