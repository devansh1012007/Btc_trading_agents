"""
Small shared helpers for parsing settings.yaml values into typed objects
used across experts, the opportunity filter, and the router; and (Phase
4, §11) loading the split base+environment config used by deployment
entry points.
"""
from pathlib import Path
from typing import List

import yaml

from ..state.contracts import RegimeType


def parse_regime_list(raw: List[str]) -> List:
    """
    Convert a YAML list like [TRENDING_UP, TRENDING_DOWN] or dynamic ids like
    ['r0', 'r1'] into RegimeType enum members (for the legacy 4 regimes) or
    DynamicRegime objects. Preserves KeyError when an invalid legacy-looking
    name is passed if strict_legacy=True, while supporting dynamic ids.
    """
    from ..state.contracts import DynamicRegime
    if not raw:
        return []
    out = []
    for s in raw:
        if isinstance(s, (RegimeType, DynamicRegime)):
            out.append(s)
            continue
        s_str = str(s).strip()
        up = s_str.upper()
        if up in RegimeType.__members__:
            out.append(RegimeType[up])
        elif any(s_str == m.value for m in RegimeType):
            out.append(RegimeType(s_str))
        elif up.startswith(("R", "REGIME_", "STATE_", "CLUSTER_")) and any(ch.isdigit() for ch in up):
            out.append(DynamicRegime(id=s_str.lower(), display_name=up))
        else:
            # Raise KeyError for unrecognized non-dynamic names (preserves existing unit test expectations)
            out.append(RegimeType[up])
    return out


def _read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay onto base -- overlay wins on conflicts,
    dicts merge key-by-key, everything else (including lists) is replaced
    wholesale by the overlay's value when present."""
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_layered_config(env: str = "paper", config_dir: str = "config") -> dict:
    """
    Deployment entry points (run.py, status.py, reset_breaker.py,
    approve_model.py, rollback_model.py, retrain_all.py) load config this
    way (Phase 4 §11's multi-environment split): settings.base.yaml is
    the full config, settings.{env}.yaml carries only the handful of
    env-specific deltas (system.sandbox, execution.mode,
    execution.confirm_real_money). Training/backtest scripts inherited
    from Phase 2/3 are offline utilities, not deployment-mode-aware, and
    keep their simpler single-file --config flag (defaulting to
    settings.base.yaml) -- see PHASE4_COMPLETE.md for the reasoning.
    """
    base = _read_yaml(f"{config_dir}/settings.base.yaml")
    overlay = _read_yaml(f"{config_dir}/settings.{env}.yaml")
    return _deep_merge(base, overlay)
