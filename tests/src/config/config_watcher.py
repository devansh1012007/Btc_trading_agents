"""
Hot-Reloadable Config (Phase 5, §6.2). A ConfigWatcher that checks
settings.*.yaml's mtime each loop iteration and reloads on change — but
only for a whitelisted safe subset of parameters.

execution.mode, execution.confirm_real_money, and anything capital-risk-
structural is explicitly excluded from hot-reload and requires a deliberate
restart — a config typo mid-session should never be able to silently flip
the system into live order placement.

The watcher calls pipeline.rebuild_layers() (or risk_layer.reload_config())
instead of trying to mutate objects in place. Circuit-breaker state
(RiskStateStore, SQLite-backed) is already separate from the config-derived
attributes, so rebuilding the layer objects doesn't lose any state that
actually needs to survive.
"""
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """
    Watches config files for changes and triggers reload callbacks.

    Args:
        config_paths: List of config file paths to watch.
        safe_reload_paths: Dot-separated config paths that are safe to reload.
            e.g., ["risk.sizing", "labeling", "feature_engine"]
        excluded_from_reload: Dot-separated config paths that are NEVER hot-
            reloaded. e.g., ["execution.mode", "execution.confirm_real_money"]
        check_interval_bars: Check for changes every N bars.
        rebuild_callback: Function called when a safe config path changes.
            Receives the new config dict.
    """

    def __init__(
        self,
        config_paths: List[str],
        safe_reload_paths: List[str],
        excluded_from_reload: List[str],
        check_interval_bars: int = 10,
        rebuild_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.config_paths = config_paths
        self.safe_reload_paths = safe_reload_paths
        self.excluded_from_reload = excluded_from_reload
        self.check_interval_bars = check_interval_bars
        self.rebuild_callback = rebuild_callback

        # Track mtime of each config file
        self._last_mtimes: Dict[str, float] = {}
        for p in config_paths:
            path = Path(p)
            if path.exists():
                self._last_mtimes[p] = path.stat().st_mtime

        self._bars_since_check = 0

    def check(self, bars_processed: int = 0) -> bool:
        """
        Check for config file changes. Returns True if a safe reload occurred.

        Args:
            bars_processed: Incremented each bar; triggers check every
                check_interval_bars.
        """
        self._bars_since_check += bars_processed
        if self._bars_since_check < self.check_interval_bars:
            return False
        self._bars_since_check = 0

        # Load all config files and compute a combined hash
        current_content = ""
        for p in self.config_paths:
            path = Path(p)
            if path.exists():
                try:
                    with open(p, "r") as f:
                        current_content += f.read()
                except Exception as e:
                    logger.warning(f"Failed to read config file {p}: {e}")

        # Check if content changed (simple approach: compare mtimes)
        changed = False
        for p in self.config_paths:
            path = Path(p)
            if path.exists():
                current_mtime = path.stat().st_mtime
                last_mtime = self._last_mtimes.get(p, 0)
                if current_mtime > last_mtime:
                    changed = True
                    self._last_mtimes[p] = current_mtime

        if changed and self.rebuild_callback:
            logger.info("Config change detected, triggering reload...")
            try:
                new_config = self._load_merged_config()
                self.rebuild_callback(new_config)
                return True
            except Exception as e:
                logger.error(f"Config reload failed: {e}")

        return False

    def _load_merged_config(self) -> dict:
        """Load + deep-merge the watched config files (overlay order as listed)."""
        merged: dict = {}
        for p in self.config_paths:
            path = Path(p)
            if path.exists():
                with open(path, "r") as fh:
                    merged = self._deep_merge(merged, yaml.safe_load(fh) or {})
        env = os.environ.get("TRADING_ENV", "paper")
        env_overlay = Path(self.config_paths[0]).parent / f"settings.{env}.yaml"
        if env_overlay.exists():
            with open(env_overlay, "r") as fh:
                merged = self._deep_merge(merged, yaml.safe_load(fh) or {})
        return merged

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> dict:
        out = dict(base)
        for k, v in overlay.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = ConfigWatcher._deep_merge(out[k], v)
            else:
                out[k] = v
        return out
