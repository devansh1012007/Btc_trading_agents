"""
Regime-Adaptive Configuration (Phase 5, §6.2). One unified mechanism for
all regime-dependent thresholds, replacing the two independently-written
patterns (labeling.regime_specific_barriers, opportunity_filter.regime_overrides).

Usage:
    k_tp = resolve_regime_param(param_block, current_regime)
    where param_block = {"default": 1.5, "by_regime": {"TRENDING_UP": 2.0, ...}}

This is a single-scalar-per-parameter shape. Each individual threshold
that varies by regime gets its own block with default + by_regime keys.
"""
from typing import Dict, Any, Optional
from ..state.contracts import RegimeType


def resolve_regime_param(param_block: Dict[str, Any], current_regime: RegimeType) -> float:
    """
    Resolve a regime-dependent parameter to its actual value.

    Args:
        param_block: {"default": X, "by_regime": {REGIME_NAME: X, ...}}
        current_regime: The current market regime.

    Returns:
        The value for the current regime, or the default if not specified.
    """
    by_regime = param_block.get("by_regime", {})
    if current_regime.name in by_regime:
        return float(by_regime[current_regime.name])
    return float(param_block.get("default", 0.0))


def resolve_regime_param_list(param_block: Dict[str, Any], current_regime: RegimeType,
                               default: Optional[float] = None) -> Any:
    """
    Resolve a regime-dependent parameter that may not be a simple scalar.

    Args:
        param_block: {"default": X, "by_regime": {REGIME_NAME: X, ...}}
        current_regime: The current market regime.
        default: Fallback if param_block is empty.

    Returns:
        The value for the current regime, or the default.
    """
    by_regime = param_block.get("by_regime", {})
    if current_regime.name in by_regime:
        return by_regime[current_regime.name]
    return param_block.get("default", default)
