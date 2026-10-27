"""
Self-contained HTML report: every figure, the headline numbers, the
warnings and the trade table on ONE page, with plotly.js embedded once so
the file opens offline.

Deterministic: fixed ``div_id``s, no timestamps of "now", explicit trace
uids (see :mod:`src.gui.charts`) -- the same session exports the same
bytes.
"""
import html
from typing import Dict, Mapping

import pandas as pd
import plotly.graph_objects as go

from . import theme
from .charts import PLOTLY_CONFIG, to_datetimes
from .models import SessionBundle

_FIGURE_ORDER = (("price", "Price & trades"), ("decisions", "Decisions"),
                 ("equity", "Equity"), ("funnel", "Where each bar stopped"))

_CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {theme.BACKGROUND}; color: {theme.TEXT};
  font-family: {theme.FONT_FAMILY}; font-size: 14px; line-height: 1.45; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px 28px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 15px; margin: 28px 0 8px; color: {theme.TEXT_MUTED};
  text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.sub {{ color: {theme.TEXT_MUTED}; margin-bottom: 16px; }}
.banner {{ border-left: 4px solid {theme.OPEN}; background: rgba(241,196,15,.10); padding: 10px 14px;
  margin: 8px 0; border-radius: 4px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }}
.card {{ background: {theme.PANEL}; border: 1px solid {theme.GRID}; border-radius: 8px; padding: 12px 14px; }}
.card .k {{ color: {theme.TEXT_MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
.card .v {{ font-size: 20px; font-weight: 650; margin-top: 2px; font-family: {theme.FONT_MONO}; }}
.win {{ color: {theme.WIN}; }} .loss {{ color: {theme.LOSS}; }}
.fig {{ background: {theme.BACKGROUND}; border: 1px solid {theme.GRID}; border-radius: 8px; overflow: hidden; margin: 8px 0; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-family: {theme.FONT_MONO}; font-size: 12px; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid {theme.GRID}; text-align: right; white-space: nowrap; }}
th {{ color: {theme.TEXT_MUTED}; font-weight: 600; position: sticky; top: 0; background: {theme.PANEL}; }}
td:first-child, th:first-child {{ text-align: left; }}
"""


def _card(label: str, value: str, tone: str = "") -> str:
    return (f'<div class="card"><div class="k">{html.escape(label)}</div>'
            f'<div class="v {tone}">{html.escape(value)}</div></div>')


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _tone(value) -> str:
    if value is None or value == 0:
        return ""
    return "win" if value > 0 else "loss"


def summary_cards(summary: Mapping[str, object]) -> str:
    pf = summary["profit_factor"]
    cards = [
        _card("Trades", f"{summary['n_trades']} ({summary['n_open']} open)"),
        _card("Win rate", _pct(summary["win_rate_pct"])),
        _card("Expectancy / trade", _pct(summary["expectancy_pct"]), _tone(summary["expectancy_pct"])),
        _card("Total PnL", f"{summary['total_pnl_usdt']:+,.2f}", _tone(summary["total_pnl_usdt"])),
        _card("Profit factor", "n/a" if pf is None else f"{pf:.2f}"),
        _card("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%", "loss" if summary["max_drawdown_pct"] else ""),
        _card("Bars", f"{summary['n_bars']:,}"),
    ]
    return '<div class="cards">' + "".join(cards) + "</div>"


def trades_table(bundle: SessionBundle) -> str:
    trades = bundle.trades
    if trades.empty:
        return "<p class='sub'>No trades in this session.</p>"
    spec = bundle.instrument
    head = ("Trade", "Side", "Entry (UTC)", "Entry px", "Exit (UTC)", "Exit px",
            "Reason", "PnL %", "PnL", "Expert", "Regime")
    rows = []
    for row in trades.to_dict("records"):
        exited = row["exit_timestamp"] is not None and not pd.isna(row["exit_timestamp"])
        pnl = row["pnl_usdt"]
        tone = "" if pnl is None or pd.isna(pnl) else ("win" if pnl > 0 else "loss" if pnl < 0 else "")
        cells = (
            str(row["trade_id"]), "LONG" if row["direction"] >= 0 else "SHORT",
            to_datetimes([int(row["entry_timestamp"])])[0], spec.price_label(row["entry_price"]),
            to_datetimes([int(row["exit_timestamp"])])[0] if exited else "open",
            spec.price_label(row["exit_price"]) if exited else "-",
            "-" if pd.isna(row["exit_reason"]) else str(row["exit_reason"]).replace("_", " "),
            "-" if pd.isna(row["pnl_pct"]) else f"{row['pnl_pct']:+.2f}%",
            "-" if pnl is None or pd.isna(pnl) else f"{pnl:+,.2f}",
            "-" if pd.isna(row["expert"]) else str(row["expert"]),
            "-" if pd.isna(row["regime"]) else str(row["regime"]),
        )
        tds = "".join(
            f'<td class="{tone if i in (7, 8) else ""}">{html.escape(c)}</td>' for i, c in enumerate(cells)
        )
        rows.append(f"<tr>{tds}</tr>")
    thead = "".join(f"<th>{h}</th>" for h in head)
    return f'<div class="wrap"><table><thead><tr>{thead}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_report_html(
    bundle: SessionBundle, figures: Dict[str, go.Figure], summary: Mapping[str, object], timeframe: str,
) -> str:
    """The full report page (plotly.js embedded once)."""
    blocks = []
    first = True
    for key, title in _FIGURE_ORDER:
        fig = figures.get(key)
        if fig is None:
            continue
        div = fig.to_html(
            full_html=False, include_plotlyjs="inline" if first else False,
            div_id=f"trade-map-{key}", config=PLOTLY_CONFIG,
        )
        first = False
        blocks.append(f"<h2>{html.escape(title)}</h2><div class='fig'>{div}</div>")

    banners = "".join(f"<div class='banner'>{html.escape(w)}</div>" for w in bundle.warnings)
    title = f"Trade Map - {bundle.instrument.label()}"
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='sub'>{html.escape(bundle.source_label)} &middot; {html.escape(timeframe)} candles "
        f"&middot; base bars {html.escape(bundle.base_timeframe)}</div>"
        f"{banners}{summary_cards(summary)}{''.join(blocks)}"
        f"<h2>Trades</h2>{trades_table(bundle)}"
        "</main></body></html>"
    )
