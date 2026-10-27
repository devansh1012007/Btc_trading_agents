#!/usr/bin/env python
"""
Trade Map - an interactive view of exactly where the trading system took
trades, and why.

    pip install -e ".[viz]"
    make trade-map                       # http://127.0.0.1:8572
    TRADE_MAP_PORT=8600 make trade-map   # another port
    python scripts/trade_map_app.py      # same thing, without make

A VIEWER: it never places an order (replays are forced to ``paper_sim``)
and never writes to the repo's ``data/``. Data comes from, in order:
live artifacts -> a replay of the raw OHLCV -> a synthetic demo.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_PORT = 8572   # never Streamlit's 8501: see AGENTS.md "Ports"
HOST = "127.0.0.1"


def port() -> int:
    return int(os.environ.get("TRADE_MAP_PORT", DEFAULT_PORT))


def launch() -> int:
    """Re-run this file under ``streamlit run`` on the Trade Map's port."""
    from src.gui import theme

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        f"--server.port={port()}", f"--server.address={HOST}", "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=dark", f"--theme.backgroundColor={theme.BACKGROUND}",
        f"--theme.secondaryBackgroundColor={theme.PANEL}", f"--theme.textColor={theme.TEXT}",
        f"--theme.primaryColor={theme.EQUITY}",
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__" and not _running_under_streamlit():
    sys.exit(launch())


# ---------------------------------------------------------------------------
# Everything below runs inside Streamlit.
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.diagnostics.decision_trace import TERMINAL_STAGES, stage_label  # noqa: E402
from src.gui import theme  # noqa: E402
from src.gui.charts import (  # noqa: E402
    PANES,
    PLOTLY_CONFIG,
    PRICE_INDICATORS,
    PRICE_LAYERS,
    build_decision_chart,
    build_equity_chart,
    build_price_chart,
    build_stage_funnel,
    to_datetimes,
)
from src.gui.instruments import list_instruments, resolve_instrument  # noqa: E402
from src.gui.overlays import build_overlays  # noqa: E402
from src.gui.sources import (  # noqa: E402
    SOURCES,
    SourceUnavailable,
    auto_source,
    load_config,
    load_session,
    summarize_session,
)
from src.gui.timeframes import (  # noqa: E402
    DISPLAY_TIMEFRAMES,
    bucket_start,
    display_timeframes,
    resample_ohlcv,
)

LAYER_LABELS = {
    "trades": "Entries & exits", "spans": "Trade lines", "barriers": "TP / SL levels",
    "decisions": "Decision dots", "zones": "Zones", "regimes": "Regime bands",
}
SOURCE_HELP = {
    "artifacts": "What a real run left on disk (read-only).",
    "replay": "Replay the symbol's raw OHLCV through the real pipeline.",
    "synthetic": "Deterministic demo data - illustrative only.",
}

CSS = f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1600px; }}
.tm-title {{ font-size: 1.5rem; font-weight: 700; margin: 0; }}
.tm-sub {{ color: {theme.TEXT_MUTED}; margin: 0 0 .6rem; }}
.tm-card {{ background: {theme.PANEL}; border: 1px solid {theme.GRID}; border-radius: 10px; padding: .7rem .9rem; }}
.tm-card .k {{ color: {theme.TEXT_MUTED}; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }}
.tm-card .v {{ font-size: 1.35rem; font-weight: 650; font-family: {theme.FONT_MONO}; }}
.tm-win {{ color: {theme.WIN}; }} .tm-loss {{ color: {theme.LOSS}; }}
.tm-chip {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-weight: 600; font-size: .85rem; }}
.tm-banner {{ border-left: 4px solid {theme.OPEN}; background: rgba(241,196,15,.10); padding: .55rem .85rem;
  border-radius: 6px; margin: .3rem 0; }}
.tm-banner.err {{ border-color: {theme.LOSS}; background: rgba(239,83,80,.10); }}
</style>
"""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=4)
def cached_session(source: str, symbol: str, max_bars: int, n_bars: int, seed: int,
                   relax: bool, initial_equity: float):
    """Sessions are cached per parameter set: a replay is seconds, not ms."""
    config = load_config()
    bar = st.progress(0.0, text=f"Replaying {symbol} through the pipeline...")

    def progress(done: int, total: int) -> None:
        bar.progress(min(1.0, done / max(total, 1)), text=f"Replaying bar {done:,} / {total:,}")

    try:
        return load_session(
            source, symbol, config=config, max_bars=max_bars, n_bars=n_bars, seed=seed,
            relax=relax, initial_equity=initial_equity, progress=progress,
        )
    finally:
        bar.empty()


def request_inspect(ts: int, focus: bool = False) -> None:
    """Ask to inspect bar ``ts`` on the next run.

    Streamlit forbids writing to a widget's session-state key after the
    widget exists in the current run, so every writer queues the request
    and :func:`apply_pending` applies it at the top of the next run.
    """
    st.session_state["tm_pending"] = (int(ts), focus)
    st.rerun()


def apply_pending() -> None:
    pending = st.session_state.pop("tm_pending", None)
    if pending is not None:
        ts, focus = pending
        st.session_state["tm_inspect"] = ts
        if focus:
            st.session_state["tm_focus"] = ts


def banner(text: str, error: bool = False) -> None:
    st.markdown(f"<div class='tm-banner {'err' if error else ''}'>{text}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, tone: str = "") -> str:
    return f"<div class='tm-card'><div class='k'>{label}</div><div class='v {tone}'>{value}</div></div>"


def tone(value) -> str:
    if value is None or value == 0:
        return ""
    return "tm-win" if value > 0 else "tm-loss"


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

def sidebar():
    st.sidebar.markdown("### Trade Map")

    known = list_instruments()
    default = known.index("BTC/USDT") if "BTC/USDT" in known else 0
    symbol = st.sidebar.selectbox("Instrument", known, index=default)
    custom = st.sidebar.text_input("...or another symbol", "", placeholder="ETH/USDT").strip()
    symbol = custom or symbol
    spec = resolve_instrument(symbol)
    st.sidebar.caption(f"{spec.asset_class}" + (f" · {spec.option_type} {spec.strike:,.0f} · exp {spec.expiry}"
                                                if spec.is_option and spec.strike else ""))

    recommended = auto_source(symbol)
    labels = {s: f"{s}{'  (recommended)' if s == recommended else ''}" for s in SOURCES}
    source = st.sidebar.radio("Data source", SOURCES, index=SOURCES.index(recommended),
                              format_func=labels.get, help="Artifacts first, replay as the fallback.")
    st.sidebar.caption(SOURCE_HELP[source])

    max_bars, n_bars, seed, relax = 1500, 900, 42, False
    if source == "replay":
        max_bars = st.sidebar.slider("Most recent bars", 200, 5000, 1500, step=100)
        relax = st.sidebar.checkbox("Relax liquidity gates", value=False,
                                    help="Measures pipeline behaviour, not production edge.")
    elif source == "synthetic":
        n_bars = st.sidebar.slider("Bars", 300, 3000, 900, step=100)
        seed = st.sidebar.number_input("Seed", 0, 99999, 42)
        relax = st.sidebar.checkbox("Relax liquidity gates", value=True,
                                    help="Required for the shipped synthetic liquidity to trade at all.")
    initial_equity = st.sidebar.number_input("Initial equity", 100.0, 10_000_000.0, 10_000.0, step=1000.0)
    st.sidebar.divider()
    return symbol, source, max_bars, n_bars, int(seed), relax, float(initial_equity)


def chart_controls(bundle):
    st.sidebar.markdown("### Chart")
    allowed = display_timeframes(bundle.base_timeframe)
    span = int(bundle.bars["timestamp"].iloc[-1] - bundle.bars["timestamp"].iloc[0]) if len(bundle.bars) > 1 else 0
    default_tf = next((tf for tf in allowed if span / DISPLAY_TIMEFRAMES[tf] <= 1500), allowed[-1])
    if st.session_state.get("tm_tf") not in allowed:
        st.session_state["tm_tf"] = default_tf
    timeframe = st.sidebar.select_slider("Timeframe", allowed, key="tm_tf",
                                         help="Candles are resampled causally from the base bars.")
    layers = st.sidebar.multiselect("Overlays", PRICE_LAYERS, default=list(PRICE_LAYERS),
                                    format_func=LAYER_LABELS.get)
    stages = st.sidebar.multiselect(
        "Decision stages", TERMINAL_STAGES, default=[s for s in TERMINAL_STAGES
                                                      if s not in ("no_expert_candidate", "position_open")],
        format_func=stage_label, disabled="decisions" not in layers)
    indicators = st.sidebar.multiselect("Indicators", PRICE_INDICATORS, default=["ema_fast", "ema_slow"],
                                        format_func=lambda c: c.replace("_", " ").upper())
    panes = st.sidebar.multiselect("Panes", PANES, default=[], format_func=str.upper)
    volume = st.sidebar.checkbox("Volume", value=True)
    height = st.sidebar.slider("Chart height", 480, 1000, 720, step=20)
    return timeframe, layers, stages, indicators, panes, volume, height


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def header(bundle, summary) -> None:
    st.markdown(f"<p class='tm-title'>{bundle.instrument.label()}</p>"
                f"<p class='tm-sub'>{bundle.source_label} · base bars {bundle.base_timeframe} · "
                f"{summary['n_bars']:,} bars</p>", unsafe_allow_html=True)
    if bundle.is_synthetic:
        banner("<b>Synthetic data.</b> Prices are generated, not a real market - illustrative only.", error=True)
    if bundle.meta.get("relaxed_liquidity"):
        banner("<b>Liquidity gates relaxed.</b> This run measures pipeline behaviour, not production edge. "
               "Do not read expectancy from it.")
    for warning in bundle.warnings:
        if not (bundle.is_synthetic and warning.startswith("ILLUSTRATIVE")) \
                and not (bundle.meta.get("relaxed_liquidity") and warning.startswith("Liquidity")):
            banner(warning, error=warning.startswith("STALE"))

    pf = summary["profit_factor"]
    cards = [
        ("Trades", f"{summary['n_trades']} <small>({summary['n_open']} open)</small>", ""),
        ("Win rate", "n/a" if summary["win_rate_pct"] is None else f"{summary['win_rate_pct']:.1f}%", ""),
        ("Expectancy", "n/a" if summary["expectancy_pct"] is None else f"{summary['expectancy_pct']:+.2f}%",
         tone(summary["expectancy_pct"])),
        ("Total PnL", f"{summary['total_pnl_usdt']:+,.2f}", tone(summary["total_pnl_usdt"])),
        ("Profit factor", "n/a" if pf is None else f"{pf:.2f}", ""),
        ("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%", "tm-loss" if summary["max_drawdown_pct"] else ""),
    ]
    for col, (label, value, t) in zip(st.columns(len(cards)), cards, strict=True):
        col.markdown(metric_card(label, value, t), unsafe_allow_html=True)


def inspected_bar(bundle):
    """The base-bar timestamp under inspection (clicked, picked or jumped to)."""
    stamps = bundle.decisions["timestamp"].astype("int64").tolist() if not bundle.decisions.empty \
        else bundle.bars["timestamp"].astype("int64").tolist()
    if not stamps:
        return None
    current = st.session_state.get("tm_inspect")
    if current not in stamps:
        current = next((int(t) for t in bundle.trades["entry_timestamp"] if int(t) in stamps), stamps[0]) \
            if not bundle.trades.empty else stamps[0]
        st.session_state["tm_inspect"] = current
    return stamps


def why_panel(bundle) -> None:
    """Why did (or didn't) this bar trade?"""
    st.markdown("#### Why did this bar (not) trade?")
    stamps = inspected_bar(bundle)
    if stamps is None:
        st.info("This session has no bars to inspect.")
        return

    executed = [int(t) for t in bundle.decisions.loc[bundle.decisions["terminal_stage"] == "executed", "timestamp"]] \
        if not bundle.decisions.empty else []
    nav = st.columns([1, 1, 1, 1, 3])
    idx = stamps.index(st.session_state["tm_inspect"])
    if nav[0].button("◀ Prev", use_container_width=True, disabled=idx == 0):
        request_inspect(stamps[idx - 1])
    if nav[1].button("Next ▶", use_container_width=True, disabled=idx == len(stamps) - 1):
        request_inspect(stamps[idx + 1])
    later = [t for t in executed if t > st.session_state["tm_inspect"]]
    if nav[2].button("Next trade ⏭", use_container_width=True, disabled=not later):
        request_inspect(later[0], focus=True)
    earlier = [t for t in executed if t < st.session_state["tm_inspect"]]
    if nav[3].button("⏮ Prev trade", use_container_width=True, disabled=not earlier):
        request_inspect(earlier[-1], focus=True)
    nav[4].select_slider("Bar", stamps, key="tm_inspect", label_visibility="collapsed",
                         format_func=lambda t: to_datetimes([t])[0] + " UTC")

    ts = st.session_state["tm_inspect"]
    decision = bundle.decision_by_timestamp.get(ts)
    trace = bundle.traces.get(ts)
    if decision is None:
        st.info("No decision was recorded for this bar.")
        return

    stage = str(decision["terminal_stage"])
    colour = theme.stage_colour(stage)
    st.markdown(
        f"<span class='tm-chip' style='background:{colour}22;color:{colour};border:1px solid {colour}'>"
        f"{stage_label(stage)}</span> &nbsp; <span style='color:{theme.TEXT_MUTED}'>"
        f"{decision['terminal_reason']}</span>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        facts = {
            "Bar (UTC)": to_datetimes([ts])[0], "Regime": decision["regime"],
            "Expert": decision["expert"], "Decision": decision["decision"],
            "Opportunity": decision["opportunity_score"], "Router confidence": decision["router_confidence"],
            "Risk score": decision["risk_score"], "Risk approved": decision["risk_approved"],
        }
        st.dataframe(pd.DataFrame({"": list(facts), "value": [("-" if pd.isna(v) else v) if not isinstance(v, str) else v
                                                              for v in facts.values()]}).astype(str),
                     hide_index=True, use_container_width=True)
    with right:
        if trace and trace.get("stages"):
            rows = [{
                "Stage": s["stage_name"], "Passed": "✅" if s["passed"] else "⛔", "Detail": s["detail"],
                "ms": None if s.get("duration_ms") is None else round(s["duration_ms"], 2),
            } for s in trace["stages"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No stage trace is available for this bar.")


def trades_panel(bundle) -> None:
    st.markdown("#### Trades")
    trades = bundle.trades
    if trades.empty:
        st.info("No trades in this session yet.")
        return
    spec = bundle.instrument
    view = pd.DataFrame({
        "Trade": trades["trade_id"],
        "Side": ["LONG" if d >= 0 else "SHORT" for d in trades["direction"].fillna(1)],
        "Entry (UTC)": to_datetimes(trades["entry_timestamp"].astype("int64")),
        "Entry px": [spec.price_label(p) for p in trades["entry_price"]],
        "Exit px": [spec.price_label(p) if not pd.isna(p) else "open" for p in trades["exit_price"]],
        "Reason": trades["exit_reason"].fillna("-").astype(str).str.replace("_", " "),
        "PnL %": trades["pnl_pct"], "PnL": trades["pnl_usdt"], "Held": trades["bars_held"],
        "Expert": trades["expert"].fillna("-").astype(str), "Regime": trades["regime"].fillna("-").astype(str),
    })
    styled = view.style.format({"PnL %": "{:+.2f}%", "PnL": "{:+,.2f}", "Held": "{:.0f}"}, na_rep="-").map(
        lambda v: f"color: {theme.pnl_colour(v)}" if isinstance(v, (int, float)) and not pd.isna(v) else "",
        subset=["PnL %", "PnL"])
    try:
        event = st.dataframe(styled, hide_index=True, use_container_width=True, on_select="rerun",
                             selection_mode="single-row", key="tm_trades")
        picked = event.selection.rows if event is not None else []
    except TypeError:                                  # Streamlit < 1.35: no row selection
        st.dataframe(styled, hide_index=True, use_container_width=True)
        picked = []
    if picked:
        entry_ts = int(trades["entry_timestamp"].iloc[picked[0]])
        if st.session_state.get("tm_inspect") != entry_ts:
            request_inspect(entry_ts, focus=True)


def focus_range(fig, bundle, timeframe: str) -> None:
    """Zoom the chart around the inspected / selected bar."""
    focus = st.session_state.get("tm_focus")
    if focus is None:
        return
    width = DISPLAY_TIMEFRAMES[timeframe]
    bucket = bucket_start(int(focus), timeframe)
    fig.update_xaxes(range=[to_datetimes([bucket - 60 * width])[0], to_datetimes([bucket + 60 * width])[0]])
    fig.update_layout(uirevision=f"focus-{focus}-{timeframe}")


def main() -> None:
    st.set_page_config(page_title="Trade Map", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    apply_pending()

    symbol, source, max_bars, n_bars, seed, relax, initial_equity = sidebar()
    try:
        bundle = cached_session(source, symbol, max_bars, n_bars, seed, relax, initial_equity)
    except SourceUnavailable as exc:
        st.warning(f"**{source}** has nothing to show for {symbol}: {exc}.")
        if source != "synthetic":
            st.caption("Pick another source in the sidebar, or run the system first (`make pipeline`).")
        return
    except Exception as exc:                           # noqa: BLE001 - a viewer must not die on a bad file
        st.error(f"Could not build the session: {exc}")
        st.exception(exc)
        return

    if not bundle.has_content() or bundle.bars.empty:
        st.info(f"The {source} session for {symbol} is empty - nothing to draw yet.")
        return

    summary = summarize_session(bundle)
    header(bundle, summary)

    timeframe, layers, stages, indicators, panes, volume, height = chart_controls(bundle)
    candles = resample_ohlcv(bundle.bars, timeframe, bundle.base_timeframe)
    overlays = build_overlays(bundle, timeframe, candles)

    price = build_price_chart(
        bundle.bars, overlays, timeframe, bundle.instrument, base_timeframe=bundle.base_timeframe,
        layers=layers, indicators=indicators, panes=panes, decision_stages=stages,
        show_volume=volume, height=height,
    )
    focus_range(price, bundle, timeframe)
    clicked = st.plotly_chart(price, use_container_width=True, config=PLOTLY_CONFIG,
                              on_select="rerun", selection_mode="points", key="tm_price")
    try:
        points = clicked.selection.points if clicked is not None else []
    except AttributeError:
        points = []
    # A chart selection persists across reruns; act on a click only ONCE,
    # or a stale click would undo the user's later Prev / Next navigation.
    click_id = None if not points else (points[0].get("x"), points[0].get("curve_number"))
    if points and points[0].get("x") is not None and st.session_state.get("tm_last_click") != click_id:
        st.session_state["tm_last_click"] = click_id
        base = pd.to_datetime(points[0]["x"], utc=True)
        picked = int(base.timestamp())
        stamps = bundle.decisions["timestamp"].astype("int64").to_numpy() if not bundle.decisions.empty else None
        if stamps is not None and len(stamps):
            span = DISPLAY_TIMEFRAMES[timeframe]
            inside = stamps[(stamps >= picked) & (stamps < picked + span)]
            target = int(inside[-1]) if len(inside) else int(stamps[(abs(stamps - picked)).argmin()])
            if st.session_state.get("tm_inspect") != target:
                request_inspect(target)

    tab_why, tab_decisions, tab_equity, tab_trades = st.tabs(
        ["Why this bar?", "Decisions", "Equity", "Trades"])
    with tab_why:
        why_panel(bundle)
    with tab_decisions:
        left, right = st.columns([3, 2])
        left.plotly_chart(build_decision_chart(bundle, overlays, timeframe), use_container_width=True,
                          config=PLOTLY_CONFIG, key="tm_decisions")
        right.plotly_chart(build_stage_funnel(bundle), use_container_width=True, config=PLOTLY_CONFIG,
                           key="tm_funnel")
    with tab_equity:
        st.plotly_chart(build_equity_chart(overlays, timeframe, initial_equity=bundle.meta.get("initial_equity")),
                        use_container_width=True, config=PLOTLY_CONFIG, key="tm_equity")
    with tab_trades:
        trades_panel(bundle)


if _running_under_streamlit() or os.environ.get("TRADE_MAP_FORCE_RENDER"):
    main()
