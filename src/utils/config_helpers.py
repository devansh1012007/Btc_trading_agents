"""
Small shared helpers for parsing settings.yaml values into typed objects
used across experts, the opportunity filter, and the router.
"""
from typing import List

from ..state.contracts import RegimeType


def parse_regime_list(raw: List[str]) -> List[RegimeType]:
    """
    Convert a YAML list like [TRENDING_UP, TRENDING_DOWN] (parsed by
    PyYAML as plain strings) into RegimeType enum members. Matches on the
    enum's member NAME (TRENDING_UP), not its value ("trending_up").
    """
    if not raw:
        return []
    return [RegimeType[s.upper()] for s in raw]
