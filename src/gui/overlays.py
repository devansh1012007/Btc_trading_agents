"""
Overlay descriptors: WHAT to draw and WHERE -- never a figure.

Every overlay is plain, frozen data derived from a
:class:`~src.gui.models.SessionBundle`. They are the contract between the
data layer and :mod:`src.gui.charts`, and they are what the tests assert
against (a chart is hard to unit-test; a descriptor is not).

Three rules hold for every overlay here:

* **Placed by timestamp, never by array position.** An overlay's ``ts``
  is the epoch second of the CANDLE it lives on at the requested
  timeframe, computed by :func:`src.gui.timeframes.bucket_start` -- the
  same bucket authority the resampler uses. A timeframe change therefore
  re-derives every overlay and cannot misplace one.
* **Causal.** An entry marker carries only what was known at entry (the
  expert, regime, confidence, size, barrier levels); outcome lives on the
  exit marker. A bucket's decision/regime/equity is built from base bars
  inside that bucket only, using ``align_to_timeframe``'s documented
  ``first`` / ``last`` semantics.
* **No NaN, no None in anything a chart will plot.** Rows that cannot be
  placed are dropped here, once, instead of every renderer guarding.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..diagnostics.decision_trace import TERMINAL_STAGES
from .models import SessionBundle
from .timeframes import align_to_timeframe, bucket_start, bucket_start_series, parse_timeframe

#: A bucket with several decisions is summarised by the one that got
#: FURTHEST through the pipeline (``executed`` beats ``risk_layer`` beats
#: ``opportunity_filter``...). Derived from the vocabulary's own order so
#: there is no second list to keep in sync.
STAGE_RANK: Dict[str, int] = {stage: i for i, stage in enumerate(TERMINAL_STAGES)}

#: Event types that end a zone's life.
ZONE_END_EVENTS = ("broke", "merged", "decayed")


@dataclass(frozen=True)
class EntryMarker:
    ts: int                      # candle (bucket) the entry falls in
    entry_ts: int                # exact base-bar timestamp of the fill
    price: float
    direction: int               # +1 long / -1 short
    trade_id: str
    expert: str
    regime: str
    size: float
    router_confidence: Optional[float]
    risk_score: Optional[float]
    take_profit_price: Optional[float]   # known at entry
    stop_loss_price: Optional[float]     # known at entry


@dataclass(frozen=True)
class ExitMarker:
    ts: int
    exit_ts: int
    price: float
    direction: int
    trade_id: str
    reason: str                  # take_profit | stop_loss | timeout
    pnl_usdt: float
    pnl_pct: float
    bars_held: Optional[int]


@dataclass(frozen=True)
class TradeSpan:
    """Entry -> exit join, for the connecting line on a trade review.

    Open (not yet exited) trades have ``exit_ts is None`` and are drawn
    to the last visible candle.
    """
    trade_id: str
    entry_ts: int
    exit_ts: Optional[int]
    entry_price: float
    exit_price: Optional[float]
    direction: int
    pnl_usdt: Optional[float]
    is_open: bool


@dataclass(frozen=True)
class DecisionMarker:
    ts: int
    price: float                 # the candle's low: charts draw below it
    stage: str
    reason: str
    regime: str
    expert: str
    n_bars: int                  # base bars summarised by this candle


@dataclass(frozen=True)
class ZoneBand:
    zone_id: str
    start_ts: int
    end_ts: int
    price: float
    strength: Optional[float]
    end_event: Optional[str]     # None while still alive at the last bar


@dataclass(frozen=True)
class RegimeBand:
    start_ts: int
    end_ts: int
    regime: str


@dataclass(frozen=True)
class EquityCurve:
    ts: Tuple[int, ...] = ()
    equity: Tuple[float, ...] = ()
    drawdown_pct: Tuple[float, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.ts)


@dataclass
class OverlaySet:
    timeframe: str
    entries: List[EntryMarker] = field(default_factory=list)
    exits: List[ExitMarker] = field(default_factory=list)
    spans: List[TradeSpan] = field(default_factory=list)
    decisions: List[DecisionMarker] = field(default_factory=list)
    zones: List[ZoneBand] = field(default_factory=list)
    regimes: List[RegimeBand] = field(default_factory=list)
    equity: EquityCurve = field(default_factory=EquityCurve)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _text(value, default: str = "") -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default
    return str(value)


def _bucket(ts: int, timeframe: str) -> int:
    return bucket_start(int(ts), timeframe)


def _barrier(entry: float, pct: Optional[float], direction: int, favourable: bool) -> Optional[float]:
    if pct is None:
        return None
    sign = direction if favourable else -direction
    return entry * (1.0 + sign * pct / 100.0)


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------

def build_trade_overlays(
    bundle: SessionBundle, timeframe: str, bars_tf: pd.DataFrame
) -> Tuple[List[EntryMarker], List[ExitMarker], List[TradeSpan]]:
    """Entry markers, exit markers and trade spans on ``timeframe`` candles."""
    if bundle.trades.empty or bars_tf.empty:
        return [], [], []

    visible = set(bars_tf["timestamp"].astype("int64").tolist())
    closes = bundle.close_by_timestamp
    entries: List[EntryMarker] = []
    exits: List[ExitMarker] = []
    spans: List[TradeSpan] = []

    for row in bundle.trades.to_dict("records"):
        entry_ts = row.get("entry_timestamp")
        if entry_ts is None or pd.isna(entry_ts):
            continue
        entry_ts = int(entry_ts)
        direction = 1 if (_finite(row.get("direction")) or 0) >= 0 else -1
        entry_price = _finite(row.get("entry_price")) or closes.get(entry_ts)
        if entry_price is None:
            continue
        trade_id = _text(row.get("trade_id"), f"trade-{entry_ts}")
        entry_bucket = _bucket(entry_ts, timeframe)

        tp_pct = _finite(row.get("take_profit_pct"))
        sl_pct = _finite(row.get("stop_loss_pct"))

        if entry_bucket in visible:
            entries.append(EntryMarker(
                ts=entry_bucket, entry_ts=entry_ts, price=entry_price, direction=direction,
                trade_id=trade_id, expert=_text(row.get("expert"), "?"),
                regime=_text(row.get("regime"), "?"), size=_finite(row.get("size")) or 0.0,
                router_confidence=_finite(row.get("router_confidence")),
                risk_score=_finite(row.get("risk_score")),
                take_profit_price=_barrier(entry_price, tp_pct, direction, True),
                stop_loss_price=_barrier(entry_price, sl_pct, direction, False),
            ))

        exit_ts_raw = row.get("exit_timestamp")
        has_exit = exit_ts_raw is not None and not pd.isna(exit_ts_raw)
        reason = _text(row.get("exit_reason"), "")
        is_open = (not has_exit) or reason == "open"

        if has_exit and not is_open:
            exit_ts = int(exit_ts_raw)
            exit_price = _finite(row.get("exit_price")) or closes.get(exit_ts)
            exit_bucket = _bucket(exit_ts, timeframe)
            if exit_price is not None:
                if exit_bucket in visible:
                    exits.append(ExitMarker(
                        ts=exit_bucket, exit_ts=exit_ts, price=exit_price,
                        direction=direction, trade_id=trade_id, reason=reason or "unknown",
                        pnl_usdt=_finite(row.get("pnl_usdt")) or 0.0,
                        pnl_pct=_finite(row.get("pnl_pct")) or 0.0,
                        bars_held=(int(row["bars_held"]) if _finite(row.get("bars_held")) is not None else None),
                    ))
                spans.append(TradeSpan(
                    trade_id=trade_id, entry_ts=entry_bucket, exit_ts=exit_bucket,
                    entry_price=entry_price, exit_price=exit_price, direction=direction,
                    pnl_usdt=_finite(row.get("pnl_usdt")), is_open=False,
                ))
        else:
            spans.append(TradeSpan(
                trade_id=trade_id, entry_ts=entry_bucket, exit_ts=None,
                entry_price=entry_price, exit_price=None, direction=direction,
                pnl_usdt=None, is_open=True,
            ))

    return entries, exits, spans


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def build_decision_markers(
    bundle: SessionBundle, timeframe: str, bars_tf: pd.DataFrame
) -> List[DecisionMarker]:
    """One marker per candle: the decision that got furthest in that candle."""
    decisions = bundle.decisions
    if decisions.empty or bars_tf.empty:
        return []

    frame = decisions[["timestamp", "terminal_stage", "terminal_reason", "regime", "expert"]].copy()
    frame["bucket"] = bucket_start_series(frame["timestamp"].to_numpy(dtype="int64"), timeframe)
    frame["rank"] = frame["terminal_stage"].map(STAGE_RANK).fillna(-1).astype("int64")
    frame = frame.sort_values(["bucket", "rank", "timestamp"], kind="stable")
    counts = frame.groupby("bucket").size()
    best = frame.drop_duplicates("bucket", keep="last")

    lows = dict(zip(
        bars_tf["timestamp"].astype("int64").tolist(),
        bars_tf["low"].astype("float64").tolist(), strict=True,
    ))
    out: List[DecisionMarker] = []
    for row in best.to_dict("records"):
        bucket = int(row["bucket"])
        low = lows.get(bucket)
        if low is None or not np.isfinite(low) or pd.isna(row["terminal_stage"]):
            continue
        out.append(DecisionMarker(
            ts=bucket, price=float(low), stage=str(row["terminal_stage"]),
            reason=_text(row["terminal_reason"]), regime=_text(row["regime"], "unknown"),
            expert=_text(row["expert"], "-"), n_bars=int(counts.loc[bucket]),
        ))
    out.sort(key=lambda m: m.ts)
    return out


# ---------------------------------------------------------------------------
# regime bands
# ---------------------------------------------------------------------------

def build_regime_bands(
    bundle: SessionBundle, timeframe: str, bars_tf: pd.DataFrame
) -> List[RegimeBand]:
    """Contiguous runs of one regime, as background bands.

    The regime of a candle is the regime at its OPEN (``how="first"``):
    the only causal choice for a value painted across the whole candle.
    """
    decisions = bundle.decisions
    if decisions.empty or bars_tf.empty:
        return []

    aligned = align_to_timeframe(
        decisions[["timestamp", "regime"]].dropna(subset=["regime"]), timeframe, how="first",
    )
    if aligned.empty:
        return []
    width = parse_timeframe(timeframe)
    stamps = aligned["timestamp"].astype("int64").tolist()
    regimes = aligned["regime"].astype(str).tolist()

    bands: List[RegimeBand] = []
    start, current = stamps[0], regimes[0]
    previous = stamps[0]
    for ts, regime in zip(stamps[1:], regimes[1:], strict=True):
        if regime != current:
            bands.append(RegimeBand(start, ts, current))
            start, current = ts, regime
        previous = ts
    bands.append(RegimeBand(start, previous + width, current))
    return bands


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------

def build_zone_bands(
    bundle: SessionBundle, timeframe: str, bars_tf: pd.DataFrame
) -> List[ZoneBand]:
    """Each zone as a level from its creation to the event that ended it.

    The zone log records a price LEVEL (the zone's midpoint) per event,
    not its edges, so a zone is drawn as a level -- not invented bounds.
    """
    zones = bundle.zones
    if zones.empty or bars_tf.empty:
        return []

    last_end = int(bars_tf["timestamp"].iloc[-1]) + parse_timeframe(timeframe)
    out: List[ZoneBand] = []
    for zone_id, group in zones.sort_values("timestamp", kind="stable").groupby("zone_id", sort=False):
        created = group[group["event_type"].eq("created").fillna(False).astype(bool)]
        first = created.iloc[0] if not created.empty else group.iloc[0]
        price = _finite(first["price_at_event"])
        if price is None:
            continue
        start = _bucket(int(first["timestamp"]), timeframe)
        ended = group[group["event_type"].isin(ZONE_END_EVENTS).fillna(False).astype(bool)]
        if ended.empty:
            end, end_event = last_end, None
        else:
            terminal = ended.iloc[0]
            end = _bucket(int(terminal["timestamp"]), timeframe) + parse_timeframe(timeframe)
            end_event = str(terminal["event_type"])
        out.append(ZoneBand(
            zone_id=str(zone_id), start_ts=start, end_ts=max(end, start + parse_timeframe(timeframe)),
            price=price, strength=_finite(first["zone_strength_at_event"]), end_event=end_event,
        ))
    return out


# ---------------------------------------------------------------------------
# equity
# ---------------------------------------------------------------------------

def build_equity_curve(bundle: SessionBundle, timeframe: str) -> EquityCurve:
    """Equity at each candle's CLOSE (``how="last"``: a cumulative balance)."""
    if bundle.equity.empty:
        return EquityCurve()
    aligned = align_to_timeframe(bundle.equity.dropna(subset=["equity"]), timeframe, how="last")
    if aligned.empty:
        return EquityCurve()
    equity = aligned["equity"].astype("float64").to_numpy()
    peak = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (peak - equity) / peak * 100.0, 0.0)
    return EquityCurve(
        ts=tuple(int(t) for t in aligned["timestamp"]),
        equity=tuple(float(v) for v in equity),
        drawdown_pct=tuple(float(v) for v in drawdown),
    )


# ---------------------------------------------------------------------------
# the lot
# ---------------------------------------------------------------------------

def build_overlays(
    bundle: SessionBundle, timeframe: str, bars_tf: Optional[pd.DataFrame] = None
) -> OverlaySet:
    """Every overlay for ``bundle`` on ``timeframe`` candles.

    ``bars_tf`` is the resampled candle frame the overlays will sit on;
    it is computed from the bundle when not supplied.
    """
    if bars_tf is None:
        from .timeframes import resample_ohlcv
        bars_tf = resample_ohlcv(bundle.bars, timeframe, bundle.base_timeframe)

    entries, exits, spans = build_trade_overlays(bundle, timeframe, bars_tf)
    return OverlaySet(
        timeframe=timeframe,
        entries=entries, exits=exits, spans=spans,
        decisions=build_decision_markers(bundle, timeframe, bars_tf),
        zones=build_zone_bands(bundle, timeframe, bars_tf),
        regimes=build_regime_bands(bundle, timeframe, bars_tf),
        equity=build_equity_curve(bundle, timeframe),
    )
