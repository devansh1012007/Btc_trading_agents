"""
Unified instrument metadata (`InstrumentSpec`, Phase 15 T15.1): re-exports and
extends `src.gui.instruments` so `src/market/`, `src/execution/`, and `src/gui/`
share one canonical instrument contract.
"""
from __future__ import annotations

from ..gui.instruments import (
    ASSET_CLASSES,
    InstrumentSpec,
    default_spec,
    list_instruments,
    load_instrument_registry,
    resolve_instrument,
)

__all__ = [
    "ASSET_CLASSES",
    "InstrumentSpec",
    "default_spec",
    "list_instruments",
    "load_instrument_registry",
    "resolve_instrument",
]
