"""
Instrument metadata: what is being traded, and how to label it.

The trading system is crypto-first, but the GUI must not assume that.
A single :class:`InstrumentSpec` describes any tradable -- spot, perpetual
futures, dated futures, listed options, equities -- and the rest of the
GUI reads contract metadata off that object instead of branching on
hard-coded symbols.

Instrument metadata is loaded from a YAML registry (``config/
instruments.yaml``). Anything not in the registry falls back to a
generic spot spec, so a new symbol never breaks the GUI.

The GUI is a *viewer*: a missing, unreadable or malformed registry
degrades to "no metadata" instead of taking the page down. It is
logged loudly, because silently degrading a perpetual to a spot spec
would mislabel every number on the chart.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

ASSET_CLASSES = ("spot", "perp", "future", "option", "equity", "fx")
OPTION_TYPES = ("call", "put")

#: Repository root, derived from this file's location so the registry
#: resolves the same way no matter which directory the GUI is launched
#: from. ``src/gui/instruments.py`` -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The instrument registry, relative to the repository root.
DEFAULT_REGISTRY_PATH = str(_REPO_ROOT / "config" / "instruments.yaml")


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract metadata for one tradable instrument.

    Attributes:
        symbol: the instrument's own ticker ("BTC/USDT", "BTC-261220-75000-C").
        asset_class: one of :data:`ASSET_CLASSES`.
        underlying: ticker the instrument settles against, for derivatives.
        option_type: "call" or "put" for options.
        strike: strike price for options.
        expiry: expiry date (ISO-8601) for dated derivatives.
        multiplier: contract multiplier -- 1 for spot, 100 for US equity
            options, the contract size for futures.
        tick_size: minimum price increment.
        lot_size: minimum tradeable quantity in base units.
        currency: quote currency of the instrument.
        display_precision: decimal places for price labels.
    """

    symbol: str
    asset_class: str = "spot"
    underlying: Optional[str] = None
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    multiplier: float = 1.0
    tick_size: Optional[float] = None
    lot_size: float = 1.0
    currency: str = "USDT"
    display_precision: int = 2
    name: Optional[str] = None

    @property
    def is_option(self) -> bool:
        return self.asset_class == "option"

    @property
    def is_derivative(self) -> bool:
        return self.asset_class in ("perp", "future", "option")

    def moneyness_pct(self, spot: Optional[float]) -> Optional[float]:
        """(spot - strike) / spot * 100 for options; None when not applicable.

        Positive means the underlying is above the strike (call in the
        money, put out of the money).
        """
        if not self.is_option or self.strike is None or not spot:
            return None
        return (float(spot) - float(self.strike)) / float(spot) * 100.0

    def label(self) -> str:
        """Human-readable instrument label for chart titles and legends."""
        if self.name:
            return self.name
        if self.is_option and self.strike is not None:
            kind = (self.option_type or "?").upper()
            expiry = self.expiry or "?"
            return f"{self.underlying or self.symbol} {expiry} {self.strike:,.0f}{kind}"
        return self.symbol

    def price_label(self, price: Optional[float]) -> str:
        """Format a price with this instrument's precision."""
        if price is None or price != price:  # NaN-safe
            return "-"
        return f"{float(price):,.{self.display_precision}f}"


def default_spec(symbol: str) -> InstrumentSpec:
    """Generic spec for a symbol that is not in the registry."""
    return InstrumentSpec(symbol=symbol, asset_class="spot")


def _parse_spec(symbol: str, raw: dict) -> InstrumentSpec:
    """Build an InstrumentSpec from a registry entry, ignoring unknown keys."""
    allowed = {
        "asset_class", "underlying", "option_type", "strike", "expiry",
        "multiplier", "tick_size", "lot_size", "currency",
        "display_precision", "name",
    }
    kwargs = {k: v for k, v in (raw or {}).items() if k in allowed}

    asset_class = str(kwargs.get("asset_class", "spot"))
    if asset_class not in ASSET_CLASSES:
        raise ValueError(
            f"Unknown asset_class {asset_class!r} for {symbol!r}; "
            f"expected one of {ASSET_CLASSES}"
        )

    option_type = kwargs.get("option_type")
    if option_type is not None:
        option_type = str(option_type).lower()
        if option_type not in OPTION_TYPES:
            raise ValueError(
                f"Unknown option_type {option_type!r} for {symbol!r}; "
                f"expected one of {OPTION_TYPES}"
            )

    strike = kwargs.get("strike")
    if strike is not None:
        try:
            strike = float(strike)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"strike for {symbol!r} must be numeric, got {strike!r}") from exc
        if strike <= 0:
            raise ValueError(f"strike for {symbol!r} must be positive, got {strike}")

    expiry = kwargs.get("expiry")
    if expiry is not None:
        expiry = str(expiry)
        try:
            datetime.strptime(expiry, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"expiry for {symbol!r} must be an ISO date (YYYY-MM-DD), got {expiry!r}"
            ) from exc

    multiplier = _positive_float(symbol, "multiplier", kwargs.get("multiplier", 1.0))
    lot_size = _positive_float(symbol, "lot_size", kwargs.get("lot_size", 1.0))

    tick_size = kwargs.get("tick_size")
    if tick_size is not None:
        tick_size = _positive_float(symbol, "tick_size", tick_size)

    display_precision = kwargs.get("display_precision", 2)
    try:
        display_precision = int(display_precision)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"display_precision for {symbol!r} must be an integer, got {display_precision!r}"
        ) from exc
    if display_precision < 0:
        raise ValueError(
            f"display_precision for {symbol!r} must be >= 0, got {display_precision}"
        )

    return InstrumentSpec(
        symbol=symbol,
        asset_class=asset_class,
        underlying=kwargs.get("underlying"),
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        multiplier=multiplier,
        tick_size=tick_size,
        lot_size=lot_size,
        currency=str(kwargs.get("currency", "USDT")),
        display_precision=display_precision,
        name=kwargs.get("name"),
    )


def _positive_float(symbol: str, field: str, value: object) -> float:
    """Coerce ``value`` to a strictly positive float or raise."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} for {symbol!r} must be numeric, got {value!r}"
        ) from exc
    if number <= 0:
        raise ValueError(f"{field} for {symbol!r} must be positive, got {value!r}")
    return number


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, InstrumentSpec]:
    """Load the instrument registry from YAML.

    The file maps a symbol to its metadata. A missing file is not an
    error -- it just means every symbol falls back to a generic spec.

    A malformed entry is skipped (and logged) rather than raising, and a
    malformed *file* degrades to an empty registry: the GUI is a viewer
    and one bad registry line must not take the whole page down.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Instrument registry %s not found; every symbol falls back to a generic spec", path)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Instrument registry %s could not be read (%s); ignoring it", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.error("Instrument registry %s must be a mapping; ignoring it", path)
        return {}

    instruments_raw = raw.get("instruments", raw)
    if not isinstance(instruments_raw, dict):
        logger.error("Instrument registry %s: 'instruments' must be a mapping; ignoring it", path)
        return {}

    out: Dict[str, InstrumentSpec] = {}
    for symbol, spec_raw in instruments_raw.items():
        if not isinstance(spec_raw, dict):
            logger.error("Instrument registry %s: entry for %s must be a mapping; skipping", path, symbol)
            continue
        try:
            out[str(symbol)] = _parse_spec(str(symbol), spec_raw)
        except ValueError as exc:
            logger.error("Instrument registry %s: %s; skipping", path, exc)
    return out


def resolve_instrument(
    symbol: str,
    registry: Optional[Dict[str, InstrumentSpec]] = None,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> InstrumentSpec:
    """Resolve a symbol to its spec, falling back to a generic spot spec."""
    if registry is None:
        registry = load_registry(registry_path)
    return registry.get(symbol, default_spec(symbol))


def list_instruments(registry: Optional[Dict[str, InstrumentSpec]] = None,
                     registry_path: str = DEFAULT_REGISTRY_PATH) -> List[str]:
    """All symbols known to the registry, sorted."""
    if registry is None:
        registry = load_registry(registry_path)
    return sorted(registry)
