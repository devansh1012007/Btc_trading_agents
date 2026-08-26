#!/usr/bin/env python
"""
Interactive demo (PRESENTATION §25): a Streamlit app that lets a
recruiter walk the full decision pipeline — market → regime → experts →
opportunity → router → risk → trades — over a historical window, without
installing an exchange API.

    pip install -e ".[demo]"
    streamlit run scripts/demo_app.py

Runs fully offline on synthetic data by default (--input accepts a real
OHLCV CSV/Parquet). All computation goes through the same replay/execution
functions the backtest harness uses — the demo shows the real system.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import streamlit as st  # noqa: E402
from full_stack_walk_forward_backtest import (  # noqa: E402
    generate_synthetic_ohlcv,
    maybe_relax_liquidity,
    run_full_stack_pass,
)

from src.backtest.replay import replay_full_stack  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402

REGIME_COLORS = {"trending_up": "#2ecc71", "trending_down": "#e74c3c",
                 "chop_range": "#95a5a6", "panic_spike": "#8e44ad"}

st.set_page_config(page_title="BTC Trading Agents — decision demo",
                   layout="wide")


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running the full pipeline over the window…")
def compute(bars: int, seed: int, input_path: str | None):
    config = maybe_relax_liquidity(load_layered_config("paper"), True)
    symbol = config.get("system", {}).get("symbol", "BTC/USDT")
    if input_path:
        df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
        if not np.issubdtype(df["timestamp"].dtype, np.integer):
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        df = df.sort_values("timestamp").reset_index(drop=True).head(bars)
        tag = f"{symbol} — REAL DATA"
    else:
        df = generate_synthetic_ohlcv(bars, seed=seed)
        tag = "ILLUSTRATIVE — synthetic data"

    exec_pass = run_full_stack_pass(df, symbol, config, initial_equity=10_000.0)
    replayed = replay_full_stack(df, symbol, config, run_experts=True)
    return {
        "config": config, "symbol": symbol, "tag": tag,
        "exec": exec_pass,
        "states": replayed.system_states,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_price_regime(states):
    ts = [datetime.fromtimestamp(s.timestamp, tz=timezone.utc) for s in states]
    close = [s.market_snapshot.close for s in states]
    regimes = [s.regime_state.primary_regime.value if s.regime_state else "unknown"
               for s in states]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(ts, close, color="#2c3e50", lw=1.0, zorder=2)
    for regime, color in REGIME_COLORS.items():
        mask = [r == regime for r in regimes]
        if any(mask):
            ax.fill_between(ts, min(close), max(close), where=mask,
                            color=color, alpha=0.18, step="mid", label=regime.replace("_", " "))
    ax.set_ylabel("Close (USDT)")
    ax.legend(loc="upper left", ncols=4, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_router(states):
    decisions = [(s.timestamp, s.router_decision.selected_expert)
                 for s in states if s.router_decision is not None]
    experts = sorted({e for _, e in decisions})
    y_of = {e: i for i, e in enumerate(experts)}
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.scatter([datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in decisions],
               [y_of[e] for _, e in decisions], s=6, marker="s", color="#2980b9")
    ax.set_yticks(range(len(experts)))
    ax.set_yticklabels(experts, fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return fig


def fig_equity(exec_pass, initial_equity):
    curve = np.asarray(exec_pass["equity_curve"], dtype=float)
    ts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in exec_pass["bar_ts"]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(ts, curve, color="#2980b9", lw=1.3, label="Full system (realized, net)")
    ax.axhline(initial_equity, color="#bdc3c7", lw=0.8)
    ax.set_ylabel("Equity (USDT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def funnel_counts(states):
    """Perception-stage counts from the causal replay (no risk/execution)."""
    total = len(states)
    with_candidates = sum(1 for s in states if s.expert_outputs is not None
                          and any(o.action != 0 for o in s.expert_outputs.all()))
    tradable = sum(1 for s in states if s.opportunity_score is not None and s.opportunity_score.tradable)
    routed = sum(1 for s in states if s.router_decision is not None
                 and s.router_decision.selected_expert != "no_trade")
    return [("bars processed", total), ("expert candidates", with_candidates),
            ("opportunity filter passed", tradable), ("router selected expert", routed)]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.title("BTC Trading Agents — decision pipeline demo")
    st.caption("Same replay/execution code paths as the backtest harness. "
               "Synthetic data by default (labeled ILLUSTRATIVE); no exchange needed.")

    with st.sidebar:
        st.header("Window")
        source = st.radio("Data source", ["Synthetic", "Real CSV/Parquet"])
        input_path = None
        if source == "Synthetic":
            bars = st.slider("Bars", 600, 3000, 1200, step=100)
            seed = st.number_input("Seed", 0, 9999, 42)
        else:
            input_path = st.text_input("Path to OHLCV file", "data/raw/BTCUSDT.csv")
            bars = st.slider("Bars to use", 600, 5000, 1500, step=100)
            seed = 42

    result = compute(bars, int(seed), input_path if source == "Real CSV/Parquet" else None)
    states, exec_pass = result["states"], result["exec"]
    st.subheader(f"{result['symbol']} · {result['tag']} · {len(states)} bars")

    tab_market, tab_funnel, tab_trades, tab_equity = st.tabs(
        ["Market & Regime", "Decision Funnel", "Trades", "Equity"])

    with tab_market:
        st.pyplot(fig_price_regime(states))
        st.pyplot(fig_router(states))

    with tab_funnel:
        rows = funnel_counts(states)
        rows.append(("orders executed (exec pass)", len(exec_pass["trades"])))
        cols = st.columns(len(rows))
        for col, (label, n) in zip(cols, rows, strict=True):
            col.metric(label, n)
        st.bar_chart({label: n for label, n in rows})
        st.caption("Perception stages from the causal replay; executions from the "
                   "full-stack pass (risk layer + paper_sim fills).")

    with tab_trades:
        trades = exec_pass["trades"]
        if not trades:
            st.info("No executed trades in this window.")
        else:
            df = pd.DataFrame(trades)[
                ["entry_timestamp", "expert", "regime", "direction", "size_usdt",
                 "pnl_pct", "pnl_usdt", "exit_reason"]].copy()
            df["entry"] = df["entry_timestamp"].map(
                lambda t: datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%m-%d %H:%M"))
            df["side"] = df["direction"].map({1: "LONG", -1: "SHORT"})
            st.dataframe(df[["entry", "expert", "regime", "side", "size_usdt",
                             "pnl_pct", "pnl_usdt", "exit_reason"]], use_container_width=True)
            st.caption("Full per-bar stage replay for any journaled trade: "
                       "`python scripts/replay_trade.py --trade-id <id>`")

    with tab_equity:
        st.pyplot(fig_equity(exec_pass, 10_000.0))
        metrics = exec_pass["trades"] and _aggregate(exec_pass["trades"])
        if metrics:
            c = st.columns(4)
            c[0].metric("Trades", metrics["n_trades"])
            c[1].metric("Win rate", f"{metrics['win_rate']:.1f}%")
            c[2].metric("Expectancy/trade", f"{metrics['expectancy']:+.3f}%")
            c[3].metric("Total PnL", f"{metrics['total_usdt']:+.2f} USDT")


def _aggregate(trades):
    n = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    return {
        "n_trades": n,
        "win_rate": len(wins) / n * 100 if n else 0.0,
        "expectancy": sum(t["pnl_pct"] for t in trades) / n if n else 0.0,
        "total_usdt": sum(t["pnl_usdt"] for t in trades),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--bars", type=int, default=1200)
    # Streamlit ignores argv beyond `streamlit run`; parse only for direct use.
    known, _ = parser.parse_known_args()
    main()
