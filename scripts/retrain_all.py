"""
Scheduled retraining (§8.2, M4.8). Invoked by cron/systemd timer, not a
long-running process -- one less thing that can silently die and stop
scheduling. For each configured component: retrain via its existing
Phase 2/3/4 training script (as a subprocess -- isolates each retrain and
matches exactly how a cron job would invoke this in production), compare
the resulting walk-forward metric against the current champion's, and
register the result as pending_approval in the model registry.

NEVER touches approved_active. Promotion is exclusively approve_model.py's
job (§8.3) -- this script's only side effects are training artifacts on
disk and pending_approval rows.
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.backtest.training_common import load_training_config
from src.labeling.hold_policy import (
    ALIGNED_COMPONENTS,
    POPULATION_EXPERT_GATE,
    HoldPolicy,
    HoldPolicyError,
    registry_alignment_status,
)
from src.registry.model_registry import get_champion, register_version

LABELS = "data/processed/triple_barrier_labels.parquet"

#: Scripts that label under the harness's contract and so take ``--env``
#: (the others -- zone survival, regime models -- are not trade-outcome models).
ALIGNED_SCRIPTS = {
    "scripts/train_triple_barrier.py", "scripts/train_trend_expert.py",
    "scripts/train_mean_reversion_expert.py", "scripts/train_breakout_expert.py",
    "scripts/train_opportunity_filter.py", "scripts/train_router.py", "scripts/train_risk_model.py",
}

#: The three experts train on ONE shared labeled dataset (one replay), so a
#: pass that retrains several experts labels once, not three times.
_COMPLETED_STEPS: set = set()

# component -> (list of [script, extra_args] steps to run in order, output .pkl path)
COMPONENT_STEPS = {
    "zone_survival": (
        [
            (["scripts/label_zone_events.py"], ["--output", "data/processed/zone_labels.parquet"]),
            (["scripts/train_zone_survival_model.py"], ["--input", "data/processed/zone_labels.parquet"]),
        ],
        "data/models/zone_validity_model.pkl",
    ),
    "regime_hmm": ([(["scripts/train_regime_models.py"], [])], "data/models/regime_hmm.pkl"),
    "regime_classifier": ([(["scripts/train_regime_models.py"], [])], "data/models/regime_classifier.pkl"),
    "trend_expert": (
        [
            (["scripts/train_triple_barrier.py"], ["--output", LABELS]),
            (["scripts/train_trend_expert.py"], ["--labels", LABELS]),
        ],
        "data/models/experts/trend_expert.pkl",
    ),
    "mean_reversion_expert": (
        [
            (["scripts/train_triple_barrier.py"], ["--output", LABELS]),
            (["scripts/train_mean_reversion_expert.py"], ["--labels", LABELS]),
        ],
        "data/models/experts/mean_reversion_expert.pkl",
    ),
    "breakout_expert": (
        [
            (["scripts/train_triple_barrier.py"], ["--output", LABELS]),
            (["scripts/train_breakout_expert.py"], ["--labels", LABELS]),
        ],
        "data/models/experts/breakout_expert.pkl",
    ),
    "opportunity_filter": ([(["scripts/train_opportunity_filter.py"], [])], "data/models/opportunity_filter.pkl"),
    "router": ([(["scripts/train_router.py"], [])], "data/models/router.pkl"),
    "risk_model": ([(["scripts/train_risk_model.py"], [])], "data/models/risk_model.pkl"),
}

PRIMARY_METRIC_KEYS = ["avg_auc", "auc"]  # checked in order against a fold's metrics


def _extract_primary_metric(meta: dict) -> Optional[float]:
    folds = meta.get("walk_forward_folds", [])
    aucs = [f["auc"] for f in folds if f.get("auc") is not None]
    if aucs:
        return sum(aucs) / len(aucs)
    return None


def load_config(path: Optional[str] = None, env: str = "paper") -> dict:
    """The config the retrain labels under: the harness's layered config by default."""
    return load_training_config(path, env)


def run_step(script_and_args, input_path: str, config_path: Optional[str], env: str = "paper") -> bool:
    script, extra_args = script_and_args
    cmd = [sys.executable] + script + extra_args
    # The expert scripts take their data from --labels; everything else from --input.
    if "--input" not in cmd and "--labels" not in cmd:
        cmd += ["--input", input_path]
    if config_path:
        if "--config" not in cmd:
            cmd += ["--config", config_path]
    elif script[0] in ALIGNED_SCRIPTS and "--env" not in cmd:
        cmd += ["--env", env]

    key = tuple(cmd)
    if key in _COMPLETED_STEPS:
        logger.info(f"Already done this pass, skipping: {' '.join(cmd)}")
        return True

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
        return False
    logger.info(result.stdout[-1000:])
    _COMPLETED_STEPS.add(key)
    return True


def retrain_component(component: str, input_path: str, config_path: Optional[str], config: dict,
                      env: str = "paper") -> None:
    steps, model_path = COMPONENT_STEPS[component]
    for step in steps:
        if not run_step(step, input_path, config_path, env):
            logger.error(f"[{component}] retraining aborted -- a step failed")
            return

    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.exists():
        logger.error(f"[{component}] expected model card at {meta_path} but it doesn't exist -- skipping registration")
        return

    meta = json.loads(meta_path.read_text())
    n_samples = meta.get("n_samples", 0)
    min_samples = config.get("retraining", {}).get("min_validation_samples", 500)
    if n_samples < min_samples:
        logger.warning(f"[{component}] only {n_samples} samples (< min_validation_samples={min_samples}) -- registering anyway, but flag this for review")

    challenger_metric = _extract_primary_metric(meta)
    champion = get_champion(component)
    champion_metric = None
    if champion:
        champion_metrics = json.loads(champion["walk_forward_metrics_json"] or "{}")
        champion_metric = champion_metrics.get("avg_auc")

    promotion_margin_pct = config.get("retraining", {}).get("promotion_margin_pct", 5.0)
    clears_margin = None
    if challenger_metric is not None and champion_metric:
        improvement_pct = (challenger_metric - champion_metric) / abs(champion_metric) * 100
        clears_margin = improvement_pct >= promotion_margin_pct
        logger.info(f"[{component}] challenger={challenger_metric:.4f} vs champion={champion_metric:.4f} "
                    f"({improvement_pct:+.1f}%, clears {promotion_margin_pct}% margin: {clears_margin})")
    else:
        logger.info(f"[{component}] challenger={challenger_metric} -- no champion yet to compare against")

    # Harness-alignment gate: a trade-deciding model is only promotable if it
    # was trained under the contract the running config enforces (minimum
    # trade duration, population, horizon). A failure does not block
    # registration -- it blocks promotion, and says why.
    alignment_ok, alignment_problem = True, None
    if component in ALIGNED_COMPONENTS:
        policy = HoldPolicy.from_config(config)
        expected = POPULATION_EXPERT_GATE if component.endswith("_expert") else None
        alignment_ok, alignment_problem = registry_alignment_status(
            meta, policy, expected_population=expected)
        if not alignment_ok:
            clears_margin = False
            logger.error(f"[{component}] NOT harness-aligned: {alignment_problem}. "
                         f"It will be registered, but approve_model.py will refuse it.")

    version_id = register_version(
        component=component, file_path=model_path,
        data_granularity=meta.get("granularity", "1m"),
        walk_forward_metrics={**meta, "avg_auc": challenger_metric, "clears_promotion_margin": clears_margin,
                              "alignment_ok": alignment_ok, "alignment_problem": alignment_problem},
        status="pending_approval",
    )
    logger.info(f"[{component}] registered {version_id} as pending_approval -- run scripts/approve_model.py to review")


def main():
    parser = argparse.ArgumentParser(description="Retrain all configured components; never auto-promotes")
    parser.add_argument("--input", required=True, help="Historical OHLCV Parquet/CSV used for every component")
    parser.add_argument("--config", default=None,
                        help="explicit config file (default: the layered config for --env, like the harness)")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    parser.add_argument("--components", nargs="*", default=None, help="Subset of components to retrain (default: all from config)")
    args = parser.parse_args()

    config = load_config(args.config, args.env)
    # Validate the hold policy BEFORE any training starts: an unsatisfiable
    # policy would otherwise fail inside each subprocess, one at a time.
    try:
        policy = HoldPolicy.from_config(config)
    except HoldPolicyError as exc:
        raise SystemExit(f"Refusing to retrain: the hold policy is not satisfiable: {exc}") from exc
    logger.info(f"Hold policy: no trade shorter than {policy.min_trade_seconds:g}s "
                f"({policy.min_hold_bars} bar(s) of {policy.granularity}, "
                f"entry latency {policy.entry_latency_seconds:g}s)")
    components = args.components or config.get("retraining", {}).get("components", list(COMPONENT_STEPS.keys()))

    for component in components:
        if component not in COMPONENT_STEPS:
            logger.warning(f"Unknown component '{component}', skipping")
            continue
        logger.info(f"=== Retraining {component} ===")
        retrain_component(component, args.input, args.config, config, args.env)

    logger.info("Retraining pass complete. Nothing was auto-promoted -- run scripts/approve_model.py to review.")


if __name__ == "__main__":
    main()
