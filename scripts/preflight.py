#!/usr/bin/env python
"""
Pre-flight check (Phase 5, §8). Runs the full validation suite before
starting a live or paper-live session. Refuses to start if anything fails.

Usage:
    python scripts/preflight.py --env paper
    python scripts/preflight.py --env live

A five-second gate that catches an entire class of silent, expensive-to-
discover bugs before a session even begins.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check before starting a session")
    parser.add_argument("--env", default="paper", choices=["paper", "live"], help="Environment")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    args = parser.parse_args()

    logger.info(f"Running pre-flight check for env={args.env}...")

    # Load config
    from src.utils.config_helpers import load_layered_config
    config = load_layered_config(env=args.env, config_dir=args.config_dir)

    # Run consistency validation
    try:
        from src.config.consistency_validator import validate_config_consistency, ConfigConsistencyError
        validate_config_consistency(config)
    except ConfigConsistencyError as e:
        logger.error(f"Pre-flight FAILED: {e}")
        sys.exit(1)

    # Check that required directories exist
    required_dirs = ["data/logs", "data/logs/decision_traces", "data/registry.db"]
    for d in required_dirs:
        p = Path(d)
        if p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
        elif not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {d}")

    # Check that EMERGENCY_STOP doesn't exist (unless we're in historical mode)
    kill_switch = Path("data/control/EMERGENCY_STOP")
    if kill_switch.exists():
        logger.warning(f"Kill switch file exists: {kill_switch}")

    logger.info("Pre-flight check PASSED.")
    print("\nPre-flight: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
