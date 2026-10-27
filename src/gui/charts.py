"""
Plotly figure builders for the Trade Map.

Everything here turns a :class:`~src.gui.models.SessionBundle` plus the
overlay descriptors from :mod:`src.gui.overlays` into a ``go.Figure``.
There is no UI framework in the way: the Streamlit app and the HTML
exporter both just call these functions.

Guarantees:

* **One resampler.** Candles always come from
  :func:`src.gui.timeframes.resample_ohlcv`; overlays are placed with the
  same bucket authority, so a timeframe switch cannot move a marker off
  its candle.
* **Deterministic.** The same session renders the same figure: no
  ``now()``, no random trace ids (every trace gets an explicit ``uid``)
  and a fixed ``div_id`` for the HTML export.
* **Clean data.** Nothing handed to Plotly contains NaN or None -- gaps
  are drawn as shapes, never as ``None``-separated trace data.

Plotly is imported here, at module scope, on purpose: this module is the
renderer. ``src/gui/__init__.py`` never imports it.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..diagnostics.decision_trace import TERMINAL_STAGES
from . import theme
from .indicators import compute_indicators
from .instruments import InstrumentSpec
from .models import SessionBundle
from .overlays import OverlaySet, build_overlays
from .timeframes import parse_timeframe, resample_ohlcv

#: Fixed id of the exported chart's <div>, so the HTML is byte-stable.
EXPORT_DIV_ID = "trade-map-chart"

#: Plotly config the app and the export both use.
PLOTLY_CONFIG: Dict[str, object] = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "trade_map", "scale": 2},
}

#: Overlay layers the price chart can draw; the app toggles these.
PRICE_LAYERS = ("trades", "spans", "barriers", "decisions", "zones", "regimes")

#: Stages that are noise on a chart (nothing was even proposed / a
#: position was simply already open); hidden unless asked for.
QUIET_STAGES = ("no_expert_candidate", "position_open")

#: Extra panes under the price chart.
PANES = ("rsi", "macd")

#: Indicators drawn ON the price pane (the others live in a pane or are
#: not price-scaled).
PRICE_INDICATORS = tuple(theme.INDICATOR_STYLES)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def to_datetimes(epochs: Iterable[int]) -> List[str]:
    """Epoch seconds -> UTC ``YYYY-MM-DD HH:MM:SS`` strings.

    Strings (not tz-aware objects) keep the figure JSON byte-stable and
    keep the axis in UTC regardless of the viewer's locale.
    """
    arr = np.asarray(list(epochs), dtype="int64")
    if arr.size == 0:
        return []
    return pd.to_datetime(arr, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S").tolist()


def _x(epoch: int) -> str:
    return to_datetimes([epoch])[0]


def _add(fig: go.Figure, trace, uid: str, row: Optional[int] = None, col: Optional[int] = None) -> None:
    """Add a trace with a deterministic ``uid`` (Plotly would invent a random one)."""
    trace.uid = uid
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col or 1)


def _fmt(spec: InstrumentSpec, price: Optional[float]) -> str:
    return spec.price_label(price)


def _base_layout(fig: go.Figure, title: str, height: int) -> None:
    axis = dict(
        gridcolor=theme.GRID, zerolinecolor=theme.GRID, linecolor=theme.AXIS,
        tickfont=dict(color=theme.TEXT_MUTED, size=theme.FONT_SIZE),
        showspikes=True, spikecolor=theme.CROSSHAIR, spikethickness=1,
        spikedash="dot", spikemode="across",
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(size=theme.FONT_SIZE_TITLE, color=theme.TEXT)),
        paper_bgcolor=theme.BACKGROUND, plot_bgcolor=theme.BACKGROUND,
        font=dict(family=theme.FONT_FAMILY, size=theme.FONT_SIZE, color=theme.TEXT),
        height=height, margin=theme.MARGIN, dragmode="pan", hovermode="closest",
        hoverlabel=dict(bgcolor=theme.PANEL, bordercolor=theme.AXIS,
                        font=dict(family=theme.FONT_MONO, size=theme.FONT_SIZE, color=theme.TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=theme.TEXT_MUTED, size=theme.FONT_SIZE - 1)),
        uirevision="trade-map",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**{**axis, "showspikes": True, "side": "right"})


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    _base_layout(fig, "", height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=theme.TEXT_MUTED))
    return fig


def _candle_hover(frame: pd.DataFrame, spec: InstrumentSpec) -> List[str]:
    out = []
    prev_close = frame["close"].shift(1)
    for ts, o, h, low, c, v, pc in zip(
        frame["timestamp"], frame["open"], frame["high"], frame["low"],
        frame["close"], frame["volume"], prev_close, strict=True,
    ):
        change = "" if pd.isna(pc) or pc == 0 else f"  ({(c - pc) / pc * 100:+.2f}%)"
        out.append(
            f"{_x(int(ts))} UTC<br>"
            f"O {_fmt(spec, o)}  H {_fmt(spec, h)}<br>"
            f"L {_fmt(spec, low)}  C {_fmt(spec, c)}{change}<br>"
            f"Vol {v:,.2f}"
        )
    return out


# ---------------------------------------------------------------------------
# the price chart
# ---------------------------------------------------------------------------

def build_price_chart(
    bars: pd.DataFrame,
    overlays: OverlaySet,
    timeframe: str,
    instrument: InstrumentSpec,
    *,
    base_timeframe: Optional[str] = None,
    layers: Sequence[str] = PRICE_LAYERS,
    indicators: Sequence[str] = (),
    panes: Sequence[str] = (),
    decision_stages: Optional[Sequence[str]] = None,
    show_volume: bool = True,
    height: int = 720,
    title: Optional[str] = None,
) -> go.Figure:
    """Candles + volume + every requested overlay, TradingView-style.

    Args:
        bars: BASE bars (``timestamp, open, high, low, close, volume``);
            resampled here to ``timeframe``.
        overlays: from :func:`src.gui.overlays.build_overlays` for the SAME
            ``timeframe`` (a mismatch raises -- it would misplace markers).
        indicators: members of :data:`PRICE_INDICATORS` drawn on the price
            pane (``ema_fast``, ``vwap``, ``bb_upper``...). Computed on the
            RESAMPLED candles, causally.
        panes: any of :data:`PANES` -- extra RSI / MACD panes below.
        decision_stages: terminal stages to mark; default is every stage
            except :data:`QUIET_STAGES`.
    """
    if overlays.timeframe != timeframe:
        raise ValueError(
            f"overlays were built for {overlays.timeframe!r} but the chart is {timeframe!r}"
        )
    unknown = [c for c in indicators if c not in PRICE_INDICATORS]
    if unknown:
        raise ValueError(f"unknown indicator(s) {unknown}; expected {list(PRICE_INDICATORS)}")
    bad_panes = [p for p in panes if p not in PANES]
    if bad_panes:
        raise ValueError(f"unknown pane(s) {bad_panes}; expected {list(PANES)}")

    candles = resample_ohlcv(bars, timeframe, base_timeframe)
    candles = candles.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    heading = title or f"{instrument.label()} · {timeframe}"
    if candles.empty:
        return _empty_figure("No bars to draw", height)

    has_volume = show_volume
    rows = ["price"] + (["volume"] if has_volume else []) + list(panes)
    heights = {"price": 0.62, "volume": 0.14, "rsi": 0.16, "macd": 0.18}
    row_heights = [heights[r] for r in rows]
    total = sum(row_heights)
    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.015,
        row_heights=[h / total for h in row_heights],
    )
    row_of = {name: i + 1 for i, name in enumerate(rows)}

    stamps = to_datetimes(candles["timestamp"])
    first_x, last_x = stamps[0], stamps[-1]
    width = parse_timeframe(timeframe)

    # -- background layers (shapes sit BELOW the candles) -------------------
    if "regimes" in layers:
        for band in overlays.regimes:
            fig.add_vrect(
                x0=_x(band.start_ts), x1=_x(band.end_ts), fillcolor=theme.regime_colour(band.regime),
                line_width=0, layer="below", row=row_of["price"], col=1,
            )
    if "zones" in layers:
        for zone in overlays.zones:
            line, fill = theme.ZONE_COLOURS.get(zone.end_event or "created", theme.ZONE_FALLBACK)
            fig.add_shape(
                type="line", x0=_x(zone.start_ts), x1=_x(zone.end_ts), y0=zone.price, y1=zone.price,
                line=dict(color=line, width=1.4, dash="dot" if zone.end_event else "solid"),
                layer="below", row=row_of["price"], col=1,
            )

    # -- candles ------------------------------------------------------------
    _add(fig, go.Candlestick(
        x=stamps, open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"],
        name=instrument.label(), showlegend=False,
        increasing=dict(line=dict(color=theme.UP, width=1), fillcolor=theme.UP),
        decreasing=dict(line=dict(color=theme.DOWN, width=1), fillcolor=theme.DOWN),
        hovertext=_candle_hover(candles, instrument), hoverinfo="text",
    ), "candles", row_of["price"])

    # -- indicators ---------------------------------------------------------
    if indicators or panes:
        ind = compute_indicators(candles)
        for name in indicators:
            colour, dash, line_width = theme.INDICATOR_STYLES.get(name, (theme.TEXT_MUTED, "solid", 1.0))
            series = ind[name].dropna()
            if series.empty:
                continue
            _add(fig, go.Scatter(
                x=to_datetimes(series.index.to_numpy()), y=series.to_numpy(), mode="lines", name=name.replace("_", " ").upper(),
                line=dict(color=colour, dash=dash, width=line_width), hoverinfo="skip",
            ), f"ind-{name}", row_of["price"])
        if "rsi" in panes:
            _add_rsi(fig, ind, row_of["rsi"])
        if "macd" in panes:
            _add_macd(fig, ind, row_of["macd"])

    # -- volume -------------------------------------------------------------
    if has_volume:
        up = (candles["close"] >= candles["open"]).to_numpy()
        _add(fig, go.Bar(
            x=stamps, y=candles["volume"], name="Volume", showlegend=False,
            marker=dict(color=[theme.VOLUME_UP if u else theme.VOLUME_DOWN for u in up]),
            hovertemplate="Vol %{y:,.2f}<extra></extra>",
        ), "volume", row_of["volume"])

    # -- trade overlays -----------------------------------------------------
    price_row = row_of["price"]
    if "spans" in layers:
        _draw_spans(fig, overlays, last_x, width, price_row)
    if "barriers" in layers:
        _draw_barriers(fig, overlays, last_x, price_row)
    if "trades" in layers:
        _draw_trade_markers(fig, overlays, instrument, price_row)
    if "decisions" in layers:
        _draw_decisions(fig, overlays, decision_stages, candles, price_row)

    # -- axes ---------------------------------------------------------------
    _base_layout(fig, heading, height)
    fig.update_yaxes(tickformat=f",.{instrument.display_precision}f", row=price_row, col=1)
    fig.update_xaxes(range=[first_x, _x(int(candles["timestamp"].iloc[-1]) + width)])
    if len(rows) > 1:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig


def _add_rsi(fig: go.Figure, ind: pd.DataFrame, row: int) -> None:
    rsi = ind["rsi"].dropna()
    if rsi.empty:
        return
    _add(fig, go.Scatter(
        x=to_datetimes(rsi.index.to_numpy()), y=rsi.to_numpy(), mode="lines", name="RSI",
        line=dict(color="#ba68c8", width=1.2), hovertemplate="RSI %{y:.1f}<extra></extra>",
    ), "pane-rsi", row)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=theme.AXIS, width=1, dash="dot"), row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_macd(fig: go.Figure, ind: pd.DataFrame, row: int) -> None:
    macd = ind[["macd", "macd_signal", "macd_hist"]].dropna()
    if macd.empty:
        return
    stamps = to_datetimes(macd.index.to_numpy())
    _add(fig, go.Bar(
        x=stamps, y=macd["macd_hist"].to_numpy(), name="MACD hist", showlegend=False,
        marker=dict(color=[theme.VOLUME_UP if v >= 0 else theme.VOLUME_DOWN for v in macd["macd_hist"]]),
        hoverinfo="skip",
    ), "pane-macd-hist", row)
    _add(fig, go.Scatter(x=stamps, y=macd["macd"].to_numpy(), mode="lines", name="MACD",
                         line=dict(color="#4da3ff", width=1.2), hoverinfo="skip"), "pane-macd", row)
    _add(fig, go.Scatter(x=stamps, y=macd["macd_signal"].to_numpy(), mode="lines", name="Signal",
                         line=dict(color="#f5a623", width=1.2), hoverinfo="skip"), "pane-macd-signal", row)


def _draw_spans(fig: go.Figure, overlays: OverlaySet, last_x: str, width: int, row: int) -> None:
    """Entry -> exit connectors (dashed; green for a win, red for a loss)."""
    for span in overlays.spans:
        x1 = last_x if span.exit_ts is None else _x(span.exit_ts)
        y1 = span.entry_price if span.exit_price is None else span.exit_price
        colour = theme.OPEN if span.is_open else theme.pnl_colour(span.pnl_usdt or 0.0)
        fig.add_shape(
            type="line", x0=_x(span.entry_ts), x1=x1, y0=span.entry_price, y1=y1,
            line=dict(color=colour, width=1.4, dash="dash"), layer="above", row=row, col=1,
        )


def _draw_barriers(fig: go.Figure, overlays: OverlaySet, last_x: str, row: int) -> None:
    """Take-profit / stop-loss levels from entry to exit (known AT entry)."""
    ends = {s.trade_id: (s.exit_ts, s.is_open) for s in overlays.spans}
    for entry in overlays.entries:
        exit_ts, _ = ends.get(entry.trade_id, (None, True))
        x1 = last_x if exit_ts is None else _x(exit_ts)
        for level, colour in ((entry.take_profit_price, theme.TAKE_PROFIT),
                              (entry.stop_loss_price, theme.STOP_LOSS)):
            if level is None:
                continue
            fig.add_shape(
                type="line", x0=_x(entry.ts), x1=x1, y0=level, y1=level,
                line=dict(color=colour, width=1, dash="dot"), opacity=0.7,
                layer="above", row=row, col=1,
            )


def _draw_trade_markers(fig: go.Figure, overlays: OverlaySet, spec: InstrumentSpec, row: int) -> None:
    for direction, label, colour, symbol in (
        (1, "Long entry", theme.LONG, "triangle-up"),
        (-1, "Short entry", theme.SHORT, "triangle-down"),
    ):
        picked = [e for e in overlays.entries if e.direction == direction]
        if not picked:
            continue
        _add(fig, go.Scatter(
            x=to_datetimes(e.ts for e in picked), y=[e.price for e in picked], mode="markers", name=label,
            marker=dict(symbol=symbol, size=theme.MARKER_SIZE + 2, color=colour,
                        line=dict(color=theme.BACKGROUND, width=1.2)),
            text=[_entry_hover(e, spec) for e in picked], hoverinfo="text",
        ), f"entry-{direction}", row)

    exits = overlays.exits
    if exits:
        _add(fig, go.Scatter(
            x=to_datetimes(x.ts for x in exits), y=[x.price for x in exits], mode="markers+text", name="Exit",
            marker=dict(symbol="x", size=theme.MARKER_SIZE, color=[theme.pnl_colour(x.pnl_usdt) for x in exits],
                        line=dict(color=theme.BACKGROUND, width=1)),
            text=[f"{x.pnl_pct:+.2f}%" for x in exits], textposition="top center",
            textfont=dict(size=theme.FONT_SIZE_MARKER, color=theme.TEXT_MUTED),
            customdata=[_exit_hover(x, spec) for x in exits],
            hovertemplate="%{customdata}<extra></extra>",
        ), "exit", row)


def _entry_hover(e, spec: InstrumentSpec) -> str:
    side = "LONG" if e.direction == 1 else "SHORT"
    lines = [
        f"<b>{side} entry</b> · {e.trade_id}",
        f"{_x(e.entry_ts)} UTC @ {_fmt(spec, e.price)}",
        f"Expert: {e.expert}  ·  Regime: {e.regime}",
        f"Size: {e.size:,.2f}",
    ]
    if e.router_confidence is not None:
        lines.append(f"Router confidence: {e.router_confidence:.2f}")
    if e.risk_score is not None:
        lines.append(f"Risk score: {e.risk_score:.2f}")
    if e.take_profit_price is not None and e.stop_loss_price is not None:
        lines.append(f"TP {_fmt(spec, e.take_profit_price)}  ·  SL {_fmt(spec, e.stop_loss_price)}")
    return "<br>".join(lines)


def _exit_hover(x, spec: InstrumentSpec) -> str:
    held = "" if x.bars_held is None else f"<br>Held {x.bars_held} bar(s)"
    return (
        f"<b>Exit · {x.reason.replace('_', ' ')}</b> · {x.trade_id}<br>"
        f"{_x(x.exit_ts)} UTC @ {_fmt(spec, x.price)}<br>"
        f"PnL {x.pnl_usdt:+,.2f} ({x.pnl_pct:+.2f}%){held}"
    )


def _draw_decisions(
    fig: go.Figure, overlays: OverlaySet, stages: Optional[Sequence[str]],
    candles: pd.DataFrame, row: int,
) -> None:
    """Per-candle decision dots under the candle, one trace per stage."""
    wanted = [s for s in TERMINAL_STAGES if s not in QUIET_STAGES] if stages is None else list(stages)
    span = float(candles["high"].max() - candles["low"].min()) or 1.0
    offset = span * 0.012
    for stage in TERMINAL_STAGES:
        if stage not in wanted:
            continue
        picked = [d for d in overlays.decisions if d.stage == stage]
        if not picked:
            continue
        _add(fig, go.Scatter(
            x=to_datetimes(d.ts for d in picked), y=[d.price - offset for d in picked], mode="markers",
            name=theme.stage_name(stage),
            marker=dict(symbol="circle", size=7, color=theme.stage_colour(stage),
                        line=dict(color=theme.BACKGROUND, width=0.8)),
            text=[_decision_hover(d) for d in picked], hoverinfo="text",
        ), f"decision-{stage}", row)


def _decision_hover(d) -> str:
    many = f" ({d.n_bars} bars)" if d.n_bars > 1 else ""
    return (
        f"<b>{theme.stage_name(d.stage)}</b>{many}<br>{_x(d.ts)} UTC<br>"
        f"{d.reason}<br>Regime: {d.regime}  ·  Expert: {d.expert}"
    )


# ---------------------------------------------------------------------------
# the decision strip
# ---------------------------------------------------------------------------

def build_decision_chart(
    bundle: SessionBundle, overlays: OverlaySet, timeframe: str, *, height: int = 300,
    include_quiet: bool = True,
) -> go.Figure:
    """Terminal stage per candle: one row per stage, coloured by stage."""
    if overlays.timeframe != timeframe:
        raise ValueError(
            f"overlays were built for {overlays.timeframe!r} but the chart is {timeframe!r}"
        )
    markers = [d for d in overlays.decisions if include_quiet or d.stage not in QUIET_STAGES]
    if not markers:
        return _empty_figure("No decisions recorded for this session", height)

    stages = [s for s in TERMINAL_STAGES if include_quiet or s not in QUIET_STAGES]
    labels = [theme.stage_name(s) for s in stages]
    fig = go.Figure()
    for stage in stages:
        picked = [d for d in markers if d.stage == stage]
        if not picked:
            continue
        _add(fig, go.Scatter(
            x=to_datetimes(d.ts for d in picked), y=[theme.stage_name(stage)] * len(picked),
            mode="markers", name=theme.stage_name(stage), showlegend=False,
            marker=dict(symbol="square", size=8, color=theme.stage_colour(stage)),
            text=[_decision_hover(d) for d in picked], hoverinfo="text",
        ), f"strip-{stage}")
    _base_layout(fig, f"Decisions · {timeframe}", height)
    fig.update_yaxes(categoryorder="array", categoryarray=labels, side="left", showspikes=False)
    return fig


def build_stage_funnel(bundle: SessionBundle, *, height: int = 300) -> go.Figure:
    """How many bars ended on each terminal stage (every base bar)."""
    if bundle.decisions.empty:
        return _empty_figure("No decisions recorded for this session", height)
    counts = bundle.decisions["terminal_stage"].value_counts()
    stages = [s for s in TERMINAL_STAGES if s in counts.index]
    values = [int(counts[s]) for s in stages]
    total = max(1, sum(values))
    fig = go.Figure()
    _add(fig, go.Bar(
        y=[theme.stage_name(s) for s in stages], x=values, orientation="h",
        marker=dict(color=[theme.stage_colour(s) for s in stages]),
        text=[f"{v:,} · {v / total:.1%}" for v in values], textposition="auto",
        hovertemplate="%{y}: %{x:,} bars<extra></extra>", showlegend=False,
    ), "funnel")
    _base_layout(fig, "Where each bar stopped", height)
    fig.update_yaxes(autorange="reversed", side="left", showspikes=False)
    fig.update_xaxes(showspikes=False)
    fig.update_layout(dragmode=False)
    return fig


# ---------------------------------------------------------------------------
# equity
# ---------------------------------------------------------------------------

def build_equity_chart(
    overlays: OverlaySet, timeframe: str, *, initial_equity: Optional[float] = None, height: int = 300,
) -> go.Figure:
    """Equity at each candle's close, with drawdown underneath."""
    curve = overlays.equity
    if overlays.timeframe != timeframe:
        raise ValueError(
            f"overlays were built for {overlays.timeframe!r} but the chart is {timeframe!r}"
        )
    if not curve:
        return _empty_figure("No equity curve for this source", height)

    stamps = to_datetimes(curve.ts)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
    _add(fig, go.Scatter(
        x=stamps, y=curve.equity, mode="lines", name="Equity",
        line=dict(color=theme.EQUITY, width=1.8, shape="hv"),
        hovertemplate="%{x}<br>Equity %{y:,.2f}<extra></extra>",
    ), "equity", 1)
    if initial_equity is not None and np.isfinite(initial_equity):
        fig.add_hline(y=initial_equity, line=dict(color=theme.AXIS, width=1, dash="dot"), row=1, col=1)
    _add(fig, go.Scatter(
        x=stamps, y=[-d for d in curve.drawdown_pct], mode="lines", name="Drawdown", fill="tozeroy",
        line=dict(color=theme.DOWN, width=1, shape="hv"), fillcolor=theme.DRAWDOWN,
        hovertemplate="%{x}<br>Drawdown %{y:.2f}%<extra></extra>",
    ), "drawdown", 2)
    _base_layout(fig, f"Equity · {timeframe}", height)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# convenience + export
# ---------------------------------------------------------------------------

def build_session_figures(
    bundle: SessionBundle, timeframe: str, **price_kwargs
) -> Dict[str, go.Figure]:
    """All four figures for a session at one timeframe."""
    candles = resample_ohlcv(bundle.bars, timeframe, bundle.base_timeframe)
    overlays = build_overlays(bundle, timeframe, candles)
    return {
        "price": build_price_chart(
            bundle.bars, overlays, timeframe, bundle.instrument,
            base_timeframe=bundle.base_timeframe, **price_kwargs,
        ),
        "decisions": build_decision_chart(bundle, overlays, timeframe),
        "equity": build_equity_chart(
            overlays, timeframe, initial_equity=bundle.meta.get("initial_equity"),
        ),
        "funnel": build_stage_funnel(bundle),
    }


def figure_to_html(fig: go.Figure, *, title: str = "Trade Map") -> str:
    """A single-figure, fully self-contained HTML page (plotly.js inlined)."""
    return fig.to_html(
        include_plotlyjs="inline", full_html=True, div_id=EXPORT_DIV_ID,
        config=PLOTLY_CONFIG, default_width="100%", default_height="100%",
    ).replace("<head>", f"<head><title>{title}</title>", 1)
