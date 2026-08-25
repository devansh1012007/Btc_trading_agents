#!/usr/bin/env python
"""
Config consistency validator (Phase 5, §8). Run before any backtest or
live session starts. Validates:

1. Unit consistency across layers (would have caught Finding #1)
2. Threshold plausibility (would have caught Finding #2)
3. Circuit breaker enforcement paths (catches dead breakers)
4. Feature schema version (catches pre-rename models)

Usage:
    python scripts/validate_config_consistency.py
    python scripts/validate_config_consistency.py --config config/settings.base.yaml
"""
import argparse
import logging
import sys
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import yaml

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate config consistency")
    parser.add_argument("--config", default="config/settings.base.yaml", help="Config file path")
    parser.add_argument("--db-path", default="data/registry.db", help="Registry DB path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    try:
        from src.config.consistency_validator import ConfigConsistencyError, validate_config_consistency
        validate_config_consistency(config, db_path=args.db_path)
        print("\nConsistency validation PASSED.")
    except ConfigConsistencyError as e:
        print(f"\nConsistency validation FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nConsistency validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
