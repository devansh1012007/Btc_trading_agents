#!/usr/bin/env python
"""
Eval: do the training labels describe what the harness actually does, and
does anything the harness does break the minimum-trade-duration rule?

One run, three questions (nothing is trained):

1. **Minimum duration.** Run the real full-stack harness and count trades
   held shorter than ``trade_duration.min_trade_seconds``. Must be 0.
2. **Label fidelity.** Label every expert candidate with the harness-aligned
   labeler (``src/labeling/harness_labels.py``) and compare, trade by trade,
   with what the harness did on the same bars: exit reason, bars held, net
   PnL. Must agree (a label that disagrees with the harness would train a
   model on a trade the harness does not take).
3. **Population & cost.** What share of the harness's trades lie inside the
   training population (opportunity-filter-passed, one position at a time),
   and what does the lockout cost in expectancy on the same candidates?

    python scripts/evaluate_training_alignment.py --synthetic 900 --relax-liquidity
    python scripts/evaluate_training_alignment.py --synthetic 3000 --bar-seconds 5 \\
        --granularity 5s --relax-liquidity          # exercises a real lockout
    python scripts/evaluate_training_alignment.py --input data/raw/BTC_USDT.parquet

Exit status: 0 = every check passed, 1 = a check failed (so it can gate a
retrain / CI), 2 = the hold policy itself is unsatisfiable.
"""
import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from src.backtest.harness_harvest import EXPERT_NAMES, apply_population, harvest_expert_samples  # noqa: E402
from src.backtest.training_common import load_training_config, read_ohlcv  # noqa: E402
from src.evaluation.trade_duration import (  # noqa: E402
    alignment_passed,
    alignment_report,
    duration_report,
    hold_stats,
    lockout_cost,
)
from src.labeling.harness_labels import LabelContext  # noqa: E402
from src.labeling.hold_policy import HoldPolicyError  # noqa: E402
from src.utils.synthetic import synthetic_ohlcv  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluate_training_alignment")


def build_config(args) -> dict:
    config = load_training_config(args.config, args.env)
    config = copy.deepcopy(config)
    if args.granularity:
        config.setdefault("system", {})["data_granularity"] = args.granularity
    if args.min_trade_seconds is not None:
        config.setdefault("trade_duration", {})["min_trade_seconds"] = args.min_trade_seconds
    if args.max_holding_bars is not None:
        config.setdefault("labeling", {})["max_holding_bars"] = args.max_holding_bars
    if args.relax_liquidity:
        from full_stack_walk_forward_backtest import maybe_relax_liquidity
        config = maybe_relax_liquidity(config, True)
    return config


def load_bars(args) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_ohlcv(args.synthetic, seed=args.seed, bar_seconds=args.bar_seconds)
    if args.input:
        df = read_ohlcv(args.input)
        if "timestamp" in df.columns and not pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        return df
    raise SystemExit("Provide --input <ohlcv file> or --synthetic <n_bars>")


def evaluate(df: pd.DataFrame, symbol: str, config: dict, initial_equity: float = 10_000.0) -> Dict[str, Any]:
    """Run the three evals; returns the structured report."""
    from full_stack_walk_forward_backtest import run_full_stack_pass

    ctx = LabelContext.from_config(config)       # raises HoldPolicyError if unsatisfiable

    # 1. the real harness
    harness = run_full_stack_pass(df, symbol, config, initial_equity)
    trades = harness["trades"]
    duration = duration_report(trades, ctx.policy)

    # 2. labels for EVERY candidate (unfiltered, overlapping) -> fidelity
    full, stats, _ = harvest_expert_samples(
        df, symbol, config, expert_names=EXPERT_NAMES, require_filter_pass=False,
        sample_policy="all", with_lockout_free=True,
    )
    fidelity = alignment_report(full, trades)

    # 3. the population a model would be trained on, and the lockout cost
    training_pop = apply_population(full, ctx, require_filter_pass=True, sample_policy="sequential")
    in_population = alignment_report(training_pop, trades)
    labeled_hold = hold_stats(training_pop["hold_seconds"] if len(training_pop) else [],
                              ctx.policy.min_trade_seconds)
    cost = (lockout_cost(training_pop["realized_return_pct"], training_pop["realized_return_free_pct"].fillna(
        training_pop["realized_return_pct"])) if len(training_pop) else lockout_cost([], []))
    pinned = float(training_pop["pinned_to_min_hold"].mean()) if len(training_pop) else None

    checks = {
        "no_trade_shorter_than_minimum": bool(duration["passed"]),
        "no_label_shorter_than_minimum": labeled_hold["violations"] == 0
                                         and hold_stats(full["hold_seconds"] if len(full) else [],
                                                        ctx.policy.min_trade_seconds)["violations"] == 0,
        "labels_reproduce_harness_trades": alignment_passed(fidelity),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "n_bars": int(len(df)),
        "policy": ctx.policy.fingerprint(),
        "harness": {"n_trades": len(trades), "final_equity": harness["final_equity"]},
        "trade_duration": duration,
        "label_fidelity": fidelity,
        "population_coverage": {
            "harness_trades": len(trades), "matched_in_training_population": in_population["matched"],
            "coverage": in_population["coverage"],
            "candidates": stats.n_candidates, "labeled": stats.n_labeled,
            "training_population": int(len(training_pop)),
        },
        "training_labels": {"hold": labeled_hold, "pinned_rate": pinned},
        "lockout_cost": cost,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render_markdown(report: Dict[str, Any]) -> str:
    pol = report["policy"]
    dur = report["trade_duration"]["overall"]
    fid = report["label_fidelity"]
    pop = report["population_coverage"]
    cost = report["lockout_cost"]
    lines = [
        "# Training / harness alignment eval", "",
        f"Generated {report['generated_at']} - {report['n_bars']} bars - **{'PASS' if report['passed'] else 'FAIL'}**", "",
        f"Policy: no trade shorter than **{pol['min_trade_seconds']:g}s** = {pol['min_hold_bars']} bar(s) of "
        f"{pol['granularity']} (entry latency {pol['entry_latency_seconds']:g}s, max hold {pol['max_holding_bars']} bars).", "",
        "## Checks", "", "| Check | Result |", "|---|---|",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "", "## Harness trades", "",
        f"{dur['n']} trades; violations **{dur['violations']}**; hold min/median/p95 = "
        f"{dur['min_s']}/{dur['median_s']}/{dur['p95_s']} s; pinned-to-lockout {report['trade_duration'].get('pinned_rate')}.", "",
        "## Label fidelity (labels vs harness, per trade)", "",
        f"- matched {fid['matched']}/{fid['n_trades']} (coverage {_pct(fid['coverage'])})",
        f"- exit reason agreement {_pct(fid['exit_reason_agreement'])}; bars-held agreement "
        f"{_pct(fid['bars_held_agreement'])}; net-PnL MAE {fid['pnl_mae_pct']}",
        f"- disagreements: {len(fid['disagreements'])}; unmatched trades: {len(fid['unmatched'])}", "",
        "## Training population", "",
        f"- {pop['candidates']} candidates -> {pop['labeled']} labeled -> {pop['training_population']} "
        f"in the training population (filter-passed, one position at a time)",
        f"- {pop['matched_in_training_population']}/{pop['harness_trades']} harness trades are inside it "
        f"(coverage {_pct(pop['coverage'])})", "",
        "## Cost of the minimum hold (same candidates, with vs without the lockout)", "",
        f"- outcomes changed by the lockout: {_pct(cost['changed_share'])}",
        f"- expectancy with lockout {cost['expectancy_locked_pct']} % vs without {cost['expectancy_free_pct']} % "
        f"(cost {cost['lockout_cost_pct']} %)",
        f"- pinned-to-lockout rate on labels: {report['training_labels']['pinned_rate']}", "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", default=None, help="OHLCV CSV/Parquet")
    parser.add_argument("--synthetic", type=int, default=None, help="generate N synthetic bars instead")
    parser.add_argument("--bar-seconds", type=int, default=60, help="synthetic: spacing between bars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--granularity", default=None, help="override system.data_granularity (e.g. 5s)")
    parser.add_argument("--min-trade-seconds", type=float, default=None)
    parser.add_argument("--max-holding-bars", type=int, default=None)
    parser.add_argument("--relax-liquidity", action="store_true")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0 (report only)")
    args = parser.parse_args(argv)

    config = build_config(args)
    df = load_bars(args)
    try:
        report = evaluate(df, args.symbol, config)
    except HoldPolicyError as exc:
        print(f"Hold policy is not satisfiable: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out_dir / f"training_alignment_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    markdown = render_markdown(report)
    (out_dir / f"training_alignment_{stamp}.md").write_text(markdown)
    print(markdown)
    return 0 if (report["passed"] or args.no_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
