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

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.registry.model_registry import get_champion, register_version

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
            (["scripts/train_triple_barrier.py"], ["--output", "data/processed/triple_barrier_labels.parquet"]),
            (["scripts/train_trend_expert.py"], ["--input", "data/processed/triple_barrier_labels.parquet"]),
        ],
        "data/models/experts/trend_expert.pkl",
    ),
    "mean_reversion_expert": ([(["scripts/train_mean_reversion_expert.py"], [])], "data/models/experts/mean_reversion_expert.pkl"),
    "breakout_expert": ([(["scripts/train_breakout_expert.py"], [])], "data/models/experts/breakout_expert.pkl"),
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


def load_config(path="config/settings.base.yaml") -> dict:
    p = Path(path)
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def run_step(script_and_args, input_path: str, config_path: str) -> bool:
    script, extra_args = script_and_args
    cmd = [sys.executable] + script + extra_args
    if "--input" not in extra_args and script[0] not in ("scripts/train_mean_reversion_expert.py", "scripts/train_breakout_expert.py"):
        pass  # some steps already specify --input explicitly above
    if "--input" not in cmd:
        cmd += ["--input", input_path]
    if "--config" not in cmd:
        cmd += ["--config", config_path]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
        return False
    logger.info(result.stdout[-1000:])
    return True


def retrain_component(component: str, input_path: str, config_path: str, config: dict) -> None:
    steps, model_path = COMPONENT_STEPS[component]
    for step in steps:
        if not run_step(step, input_path, config_path):
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

    version_id = register_version(
        component=component, file_path=model_path,
        data_granularity=meta.get("granularity", "1m"),
        walk_forward_metrics={**meta, "avg_auc": challenger_metric, "clears_promotion_margin": clears_margin},
        status="pending_approval",
    )
    logger.info(f"[{component}] registered {version_id} as pending_approval -- run scripts/approve_model.py to review")


def main():
    parser = argparse.ArgumentParser(description="Retrain all configured components; never auto-promotes")
    parser.add_argument("--input", required=True, help="Historical OHLCV Parquet/CSV used for every component")
    parser.add_argument("--config", default="config/settings.base.yaml")
    parser.add_argument("--components", nargs="*", default=None, help="Subset of components to retrain (default: all from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    components = args.components or config.get("retraining", {}).get("components", list(COMPONENT_STEPS.keys()))

    for component in components:
        if component not in COMPONENT_STEPS:
            logger.warning(f"Unknown component '{component}', skipping")
            continue
        logger.info(f"=== Retraining {component} ===")
        retrain_component(component, args.input, args.config, config)

    logger.info("Retraining pass complete. Nothing was auto-promoted -- run scripts/approve_model.py to review.")


if __name__ == "__main__":
    main()
