"""
Trade Map GUI -- interactive charts of where the trading system took
trades, and why.

The package is deliberately layered so that the interesting logic is
testable without a browser:

``instruments``
    Contract metadata (spot / perp / future / option / equity) and the
    YAML registry that describes them.
``timeframes``
    Timeframe parsing and causal OHLCV resampling.
``indicators``
    Causal technical indicators.
``models``
    :class:`~src.gui.models.SessionBundle` -- the normalised data every
    surface reads.
``artifacts``
    Read-only readers for what a run leaves on disk.
``replay``
    Computes a session by running the real pipeline.

``sources``
    Where a session comes from: live artifacts, a replay, or synthetic.
``theme`` / ``overlays``
    The palette and the overlay descriptors (what to draw, where) --
    plain data, no figures.
``charts`` / ``export``
    The Plotly figure builders and the self-contained HTML report. These
    are the ONLY modules that import Plotly, and nothing in this package
    imports them eagerly -- so the core stays importable in the Docker
    image, which installs from ``requirements.txt`` without the ``viz``
    extra.

The Streamlit app lives in ``scripts/trade_map_app.py`` and the HTML
exporter CLI in ``scripts/trade_map.py``.
"""

__all__ = [
    "SessionBundle",
    "InstrumentSpec",
    "TERMINAL_STAGES",
    "default_spec",
    "load_registry",
    "resolve_instrument",
    "list_instruments",
    "parse_timeframe",
    "resample_ohlcv",
    "bucket_start",
    "bucket_start_series",
    "display_timeframes",
    "to_epoch_seconds",
    "align_to_timeframe",
    "compute_indicators",
    "read_trade_journal",
    "read_decisions_jsonl",
    "read_decision_traces",
    "read_zone_events",
    "load_ohlcv",
    "load_symbol_ohlcv",
    "replay_session",
    "synthetic_ohlcv",
    "relax_liquidity",
]

from typing import Any

from .artifacts import (
    load_ohlcv,
    load_symbol_ohlcv,
    read_decision_traces,
    read_decisions_jsonl,
    read_trade_journal,
    read_zone_events,
)
from .indicators import compute_indicators
from .instruments import (
    InstrumentSpec,
    default_spec,
    list_instruments,
    load_registry,
    resolve_instrument,
)
from .models import TERMINAL_STAGES, SessionBundle
from .timeframes import (
    align_to_timeframe,
    bucket_start,
    bucket_start_series,
    display_timeframes,
    parse_timeframe,
    resample_ohlcv,
    to_epoch_seconds,
)


def __getattr__(name: str) -> Any:
    """Import the replay engine lazily.

    ``src.gui.replay`` pulls in the whole pipeline -- and through it
    ``ccxt`` -- which costs ~1.5s. A caller that only wants to read
    artifacts or compute indicators should not pay for it, so the
    replay names are resolved on first use instead of at import time.
    """
    if name in ("replay_session", "synthetic_ohlcv", "relax_liquidity"):
        from . import replay as _replay
        return getattr(_replay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
