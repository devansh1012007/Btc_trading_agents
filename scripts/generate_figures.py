#!/usr/bin/env python
"""
Figure generation (PRESENTATION §17): renders the visual evidence set —
equity curve + drawdown, regime overlay, router timeline, trade-decision
chart — from an actual measured run (never invented numbers).

    python scripts/generate_figures.py --synthetic 900 --relax-liquidity
    python scripts/generate_figures.py --input data/raw/BTCUSDT.csv

PNGs land in docs/images/. Synthetic runs are labeled ILLUSTRATIVE in the
figure titles; real-data runs are labeled REAL DATA.

Requires matplotlib (pip install matplotlib).
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from full_stack_walk_forward_backtest import (  # noqa: E402
    generate_synthetic_ohlcv,
    maybe_relax_liquidity,
    run_full_stack_pass,
)

from src.backtest.replay import replay_full_stack  # noqa: E402
from src.labeling.triple_barrier import barrier_volatility_pct  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402

IMAGES_DIR = ROOT / "docs" / "images"

REGIME_COLORS = {"trending_up": "#2ecc71", "trending_down": "#e74c3c",
                 "chop_range": "#95a5a6", "panic_spike": "#8e44ad"}
EXPERT_COLORS = {"trend_expert": "#2980b9", "mean_reversion_expert": "#27ae60",
                 "breakout_expert": "#d35400", "no_trade": "#bdc3c7"}


def ts_axis(values):
    return [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in values]


def figure_equity_and_drawdown(pass_result, initial_equity, tag):
    curve = np.asarray(pass_result["equity_curve"], dtype=float)
    ts = ts_axis(pass_result["bar_ts"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(ts, curve, lw=1.4, color="#2980b9", label="Full system (realized, net of costs)")
    closes = np.array([s.close for s in pass_result["snapshots"]])
    bh = initial_equity * closes[:len(ts)] / closes[0]
    ax1.plot(ts, bh, lw=1.0, ls="--", color="#7f8c8d", label="Buy & hold")
    ax1.axhline(initial_equity, color="#bdc3c7", lw=0.8)
    ax1.set_ylabel("Equity (USDT)")
    ax1.set_title(f"Equity Curve — {tag}", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    running_max = np.maximum.accumulate(curve)
    dd = np.where(running_max > 0, (running_max - curve) / running_max * 100, 0.0)
    ax2.fill_between(ts, dd, color="#c0392b", alpha=0.55)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = IMAGES_DIR / "equity_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def figure_regime_overlay(states, tag):
    ts = ts_axis([s.timestamp for s in states])
    close = [s.market_snapshot.close for s in states]
    regimes = [s.regime_state.primary_regime.value if s.regime_state else "unknown"
               for s in states]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts, close, color="#2c3e50", lw=1.0, zorder=2)
    for regime, color in REGIME_COLORS.items():
        mask = [r == regime for r in regimes]
        if any(mask):
            ax.fill_between(ts, min(close), max(close), where=mask,
                            color=color, alpha=0.18, step="mid",
                            label=regime.replace("_", " "), zorder=1)
    ax.set_title(f"Price + Regime Labels — {tag}", fontsize=12)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close (USDT)")
    ax.legend(loc="upper left", ncols=4, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = IMAGES_DIR / "regime_overlay.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def figure_router_timeline(states, tag):
    decisions = [(s.timestamp, s.router_decision.selected_expert)
                 for s in states if s.router_decision is not None]
    experts = sorted({e for _, e in decisions})
    y_of = {e: i for i, e in enumerate(experts)}

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(ts_axis([t for t, _ in decisions]),
               [y_of[e] for _, e in decisions],
               c=[EXPERT_COLORS.get(e, "#bdc3c7") for _, e in decisions],
               s=6, marker="s")
    ax.set_yticks(range(len(experts)))
    ax.set_yticklabels([e.replace("_", "\n") for e in experts], fontsize=8)
    ax.set_title(f"Router Decisions Over Time — {tag}", fontsize=12)
    ax.set_xlabel("Time (UTC)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    out = IMAGES_DIR / "router_timeline.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def figure_trade_decision(trade, state_by_ts, labeling_cfg, tag):
    """Price + zones + entry + TP/SL around one executed trade."""
    entry_ts = int(trade["entry_timestamp"])
    entry_state = state_by_ts.get(entry_ts)
    if entry_state is None or entry_state.zone_map is None:
        return None

    timestamps = sorted(state_by_ts)
    lo_i = max(0, timestamps.index(entry_ts) - 30)
    hi_i = min(len(timestamps), timestamps.index(entry_ts) + 31)
    window = [state_by_ts[t] for t in timestamps[lo_i:hi_i]]
    ts = ts_axis([s.timestamp for s in window])
    closes = [s.market_snapshot.close for s in window]

    zones = entry_state.zone_map
    direction = int(trade["direction"])
    vol_pct = barrier_volatility_pct(entry_state.feature_vector)
    k_tp = float(labeling_cfg.get("k_tp", 1.5))
    k_sl = float(labeling_cfg.get("k_sl", 1.0))
    tp_price = entry_state.market_snapshot.close * (1 + direction * k_tp * vol_pct / 100)
    sl_price = entry_state.market_snapshot.close * (1 - direction * k_sl * vol_pct / 100)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(ts, closes, color="#2c3e50", lw=1.1, label="Close")
    for zone in zones.zones[:6]:
        color = "#27ae60" if zone.zone_type == "support" else "#c0392b"
        ax.axhspan(zone.lower_bound, zone.upper_bound, color=color, alpha=0.10)
        ax.axhline((zone.lower_bound + zone.upper_bound) / 2, color=color,
                   lw=0.7, ls=":", alpha=0.6)
    ax.axhline(tp_price, color="#2ecc71", lw=1.0, ls="--", label="take profit")
    ax.axhline(sl_price, color="#e74c3c", lw=1.0, ls="--", label="stop loss")

    marker = "^" if direction == 1 else "v"
    ax.scatter([ts[timestamps.index(entry_ts) - lo_i]],
               [entry_state.market_snapshot.close],
               marker=marker, s=120, color="#f39c12", zorder=5,
               label=f"{trade['expert']} entry ({'LONG' if direction == 1 else 'SHORT'})")

    ax.set_title(f"Trade Decision @ {ts_axis([entry_ts])[0]:%Y-%m-%d %H:%M} UTC "
                 f"[{trade['exit_reason']}] — {tag}", fontsize=11)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = IMAGES_DIR / "trade_decision.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate docs/images figures from a real run")
    parser.add_argument("--input", default=None, help="OHLCV CSV/Parquet")
    parser.add_argument("--synthetic", type=int, default=None)
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    parser.add_argument("--relax-liquidity", action="store_true")
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    config = load_layered_config(args.env)
    if args.relax_liquidity:
        config = maybe_relax_liquidity(config, True)
    symbol = config.get("system", {}).get("symbol", "BTC/USDT")

    if args.synthetic:
        df = generate_synthetic_ohlcv(args.synthetic)
        tag = f"ILLUSTRATIVE — synthetic data ({args.synthetic} bars)"
    elif args.input:
        path = Path(args.input)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        if not np.issubdtype(df["timestamp"].dtype, np.integer):
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        df = df.sort_values("timestamp").reset_index(drop=True)
        tag = f"{symbol} — REAL DATA"
    else:
        raise SystemExit("Provide --input <ohlcv file> or --synthetic <n_bars>")

    print("Running full-stack execution pass (equity/drawdown/trades)...")
    exec_pass = run_full_stack_pass(df, symbol, config, initial_equity=10_000.0)

    print("Running perception replay (regime/router panels)...")
    replayed = replay_full_stack(df, symbol, config, run_experts=True)
    states = replayed.system_states
    state_by_ts = {s.timestamp: s for s in states}

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        figure_equity_and_drawdown(exec_pass, 10_000.0, tag),
        figure_regime_overlay(states, tag),
        figure_router_timeline(states, tag),
    ]
    if exec_pass["trades"]:
        written.append(figure_trade_decision(
            exec_pass["trades"][0], state_by_ts,
            config.get("labeling", {}), tag))
    else:
        print("No executed trades — skipping trade-decision figure.")

    print("\nFigures written:")
    for p in written:
        if p:
            print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
