"""
The Trade Map's visual language: one dark palette, one colour per
terminal stage, and the layout constants every chart shares.

The per-stage colours are DERIVED from
:data:`src.diagnostics.decision_trace.STAGE_STYLES` (the console colour
names the TUI uses), so the GUI and the TUI colour a stage the same way
and a new stage cannot be added to one and forgotten in the other. This
module imports no charting library.
"""
from typing import Dict

from ..diagnostics.decision_trace import STAGE_LABELS, STAGE_STYLES, TERMINAL_STAGES

# -- palette -----------------------------------------------------------------
BACKGROUND = "#0b0f14"
PANEL = "#111823"
GRID = "#1c2735"
AXIS = "#3a4a5f"
TEXT = "#d5dde8"
TEXT_MUTED = "#8494a9"
CROSSHAIR = "#8494a9"

UP = "#26a69a"
DOWN = "#ef5350"
UP_FILL = "rgba(38,166,154,0.55)"
DOWN_FILL = "rgba(239,83,80,0.55)"
VOLUME_UP = "rgba(38,166,154,0.35)"
VOLUME_DOWN = "rgba(239,83,80,0.35)"

LONG = "#2ecc71"
SHORT = "#ff6b6b"
WIN = "#2ecc71"
LOSS = "#ff6b6b"
OPEN = "#f1c40f"
TAKE_PROFIT = "#2ecc71"
STOP_LOSS = "#ff6b6b"
EQUITY = "#4da3ff"
EQUITY_FILL = "rgba(77,163,255,0.12)"
DRAWDOWN = "rgba(239,83,80,0.25)"

#: Console colour name (``STAGE_STYLES``) -> hex.
_CONSOLE_COLOURS: Dict[str, str] = {
    "green": "#2ecc71",
    "red": "#ef5350",
    "yellow": "#f1c40f",
    "cyan": "#26c6da",
    "white": "#8494a9",
    "magenta": "#ba68c8",
}

#: Terminal stage -> hex colour, derived (never a second hand-kept copy).
STAGE_COLOURS: Dict[str, str] = {
    stage: _CONSOLE_COLOURS.get(STAGE_STYLES.get(stage, "white"), TEXT_MUTED)
    for stage in TERMINAL_STAGES
}

#: Regime -> translucent background band colour.
REGIME_COLOURS: Dict[str, str] = {
    "trending_up": "rgba(38,166,154,0.10)",
    "trending_down": "rgba(239,83,80,0.10)",
    "chop_range": "rgba(132,148,169,0.08)",
    "panic_spike": "rgba(186,104,200,0.14)",
    "unknown": "rgba(132,148,169,0.03)",
}
REGIME_FALLBACK = "rgba(132,148,169,0.06)"

#: Zone event type -> (line colour, fill colour). The types are the ones
#: ``src/layers/structure_engine.py`` emits.
ZONE_COLOURS: Dict[str, tuple] = {
    "created": ("#4da3ff", "rgba(77,163,255,0.10)"),
    "touched": ("#f1c40f", "rgba(241,196,15,0.10)"),
    "rejected": ("#2ecc71", "rgba(46,204,113,0.10)"),
    "broke": ("#ef5350", "rgba(239,83,80,0.10)"),
    "merged": ("#ba68c8", "rgba(186,104,200,0.10)"),
    "decayed": ("#8494a9", "rgba(132,148,169,0.08)"),
}
ZONE_FALLBACK = ("#8494a9", "rgba(132,148,169,0.08)")

#: Indicator column -> (colour, dash, width).
INDICATOR_STYLES: Dict[str, tuple] = {
    "ema_fast": ("#f5a623", "solid", 1.3),
    "ema_slow": ("#4da3ff", "solid", 1.3),
    "sma_fast": ("#f5a623", "dot", 1.0),
    "sma_slow": ("#4da3ff", "dot", 1.0),
    "vwap": ("#ba68c8", "solid", 1.2),
    "bb_upper": ("#7e8ea6", "dash", 1.0),
    "bb_mid": ("#7e8ea6", "dot", 1.0),
    "bb_lower": ("#7e8ea6", "dash", 1.0),
    "donchian_upper": ("#26c6da", "dash", 1.0),
    "donchian_mid": ("#26c6da", "dot", 1.0),
    "donchian_lower": ("#26c6da", "dash", 1.0),
}

# -- typography / spacing ----------------------------------------------------
FONT_FAMILY = "Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"
FONT_SIZE = 12
FONT_SIZE_TITLE = 15
FONT_SIZE_MARKER = 10
MARKER_SIZE = 13
MARGIN = {"l": 56, "r": 64, "t": 44, "b": 28}


def stage_colour(stage: str) -> str:
    """Colour for a terminal stage (muted grey for an unknown one)."""
    return STAGE_COLOURS.get(stage, TEXT_MUTED)


def stage_name(stage: str) -> str:
    """Human label for a terminal stage -- the vocabulary's own."""
    return STAGE_LABELS.get(stage, stage)


_DYNAMIC_REGIME_PALETTE = (
    "rgba(38,166,154,0.11)",
    "rgba(77,163,255,0.11)",
    "rgba(245,166,35,0.11)",
    "rgba(239,83,80,0.11)",
    "rgba(186,104,200,0.12)",
    "rgba(38,198,218,0.11)",
    "rgba(156,204,101,0.11)",
    "rgba(255,112,67,0.11)",
    "rgba(126,87,194,0.11)",
    "rgba(236,64,122,0.11)",
    "rgba(102,187,106,0.11)",
    "rgba(41,182,246,0.11)",
)


def regime_colour(regime: str) -> str:
    key = str(regime).lower()
    if key in REGIME_COLOURS:
        return REGIME_COLOURS[key]
    import hashlib
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(_DYNAMIC_REGIME_PALETTE)
    return _DYNAMIC_REGIME_PALETTE[idx]


def regime_palette(regime_ids) -> Dict[str, str]:
    """Deterministic colour map for an arbitrary sequence of regime ids."""
    return {str(rid): regime_colour(str(rid)) for rid in regime_ids}


def pnl_colour(value: float) -> str:
    return WIN if value > 0 else LOSS if value < 0 else TEXT_MUTED
