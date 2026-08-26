#!/usr/bin/env python
"""
Trade Replay (PRESENTATION §10): reconstruct the full decision journey of
a historical trade, bar by bar, stage by stage.

    python scripts/replay_trade.py --list                 # browse recent trades
    python scripts/replay_trade.py --trade-id <uuid>      # full replay

Sources stitched together:
  - data/registry.db::trade_journal   (entry/exit, expert, regime, PnL)
  - data/logs/decision_traces/*.jsonl (per-bar pipeline stage verdicts)
  - optional --input OHLCV CSV/Parquet (market context around the trade)

For each bar in the replay window it prints the market state and where the
pipeline terminated that bar; for the entry bar it prints every stage's
verdict, values, and latency. This is the debugging tool AND the demo:
"Why did the system take this trade?" answerable end to end.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.registry.db import DEFAULT_DB_PATH, connect  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402

TRACE_DIR = ROOT / "data" / "logs" / "decision_traces"

STAGE_ORDER = ["expert_layer", "opportunity_filter", "router", "risk_layer", "execution"]


def fmt_ts(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Journal access
# ---------------------------------------------------------------------------

def list_trades(limit: int, db_path: str) -> pd.DataFrame:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT trade_id, entry_timestamp, exit_timestamp, expert_used, regime_at_entry, "
            "direction, realized_pnl_pct, exit_reason FROM trade_journal "
            "ORDER BY entry_timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "trade_id", "entry_timestamp", "exit_timestamp", "expert_used",
        "regime_at_entry", "direction", "realized_pnl_pct", "exit_reason"])


def load_trade(trade_id: str, db_path: str) -> dict:
    with connect(db_path) as conn:
        conn.row_factory = __import__("sqlite3").Row
        row = conn.execute(
            "SELECT * FROM trade_journal WHERE trade_id = ?", (trade_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Trace access
# ---------------------------------------------------------------------------

def load_traces_in_window(start_ts: int, end_ts: int) -> list:
    """Scan partitioned trace files covering [start_ts, end_ts]."""
    traces = []
    if not TRACE_DIR.exists():
        return traces
    days = sorted({datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                   for ts in (start_ts, end_ts)})
    for day in days:
        path = TRACE_DIR / f"traces_{day}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if start_ts <= int(t.get("timestamp", 0)) <= end_ts:
                traces.append(t)
    traces.sort(key=lambda t: int(t["timestamp"]))
    return traces


def render_trace_stages(trace: dict) -> None:
    print(f"    {'stage':<20}{'result':<10}{'detail'}")
    for stage in trace.get("stages", []):
        verdict = "PASS" if stage.get("passed") else "FAIL"
        dur = stage.get("duration_ms")
        suffix = f" ({dur:.2f}ms)" if dur is not None else ""
        print(f"    {stage['stage_name']:<20}{verdict:<10}{stage.get('detail', '')}{suffix}")
        values = stage.get("values") or {}
        if values:
            joined = ", ".join(f"{k}={v}" for k, v in list(values.items())[:6])
            print(f"    {'':<20}{'':<10}{joined}")


# ---------------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------------

def load_market_window(input_path, start_ts: int, end_ts: int):
    if not input_path:
        return None
    df = pd.read_csv(input_path) if str(input_path).endswith(".csv") else pd.read_parquet(input_path)
    if isinstance(df["timestamp"].iloc[0], (int, float)) or str(df["timestamp"].dtype).startswith("float"):
        df["ts_int"] = df["timestamp"].astype("int64")
    else:
        df["ts_int"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
    window = df[(df["ts_int"] >= start_ts) & (df["ts_int"] <= end_ts)]
    return window.set_index("ts_int")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def replay(trade: dict, window_bars: int, granularity_seconds: int, input_path) -> None:
    entry_ts = int(trade["entry_timestamp"])
    exit_ts = int(trade["exit_timestamp"]) if trade["exit_timestamp"] else entry_ts
    start = min(entry_ts - window_bars * granularity_seconds, entry_ts)
    end = max(exit_ts + window_bars * granularity_seconds, exit_ts)

    print("=" * 74)
    print(f"TRADE REPLAY  {trade['trade_id']}")
    print("=" * 74)
    print(f"  Expert      : {trade['expert_used']}   (regime at entry: {trade['regime_at_entry']})")
    side = "LONG" if trade["direction"] == 1 else "SHORT"
    print(f"  Direction   : {side}   size: {trade['size']:.2f} USDT")
    print(f"  Entry       : {fmt_ts(entry_ts)} @ {trade['entry_price']:.2f}   "
          f"router_conf={trade['router_confidence']}  risk={trade['risk_score_at_entry']}")
    if trade["exit_timestamp"]:
        print(f"  Exit        : {fmt_ts(exit_ts)} @ {trade['exit_price']:.2f} "
              f"[{trade['exit_reason']}]")
        print(f"  Result      : pnl {trade['realized_pnl_pct']:+.3f}%  "
              f"({trade['realized_pnl_usdt']:+.2f} USDT, fees {trade['fees_paid']:.2f})")
    else:
        print("  Exit        : position still open")

    market = load_market_window(input_path, start, end)
    traces = {int(t["timestamp"]): t for t in load_traces_in_window(start, end)}

    print("-" * 74)
    print(f"TIMELINE  ({fmt_ts(start)} -> {fmt_ts(end)}, window ±{window_bars} bars)")
    print("-" * 74)

    all_ts = sorted(set((list(market.index) if market is not None else [])
                        + list(traces.keys())
                        + [entry_ts] + ([exit_ts] if trade["exit_timestamp"] else [])))

    for ts in all_ts:
        marker = ""
        if ts == entry_ts:
            marker = "  <<< ENTRY"
        elif ts == exit_ts and trade["exit_timestamp"]:
            marker = "  <<< EXIT"
        price_s = ""
        if market is not None and ts in market.index:
            row = market.loc[ts]
            price_s = f"close={row['close']:.2f}"
        trace = traces.get(ts)
        summary = f"terminal={trace['terminal_stage']} ({trace['terminal_reason']})" if trace else "no trace recorded"
        print(f"\n  T{ts - entry_ts:+5d}s  {fmt_ts(ts)}  {price_s}{marker}")
        print(f"      {summary}")
        if ts == entry_ts and trace:
            print("      --- full pipeline stages at entry ---")
            render_trace_stages(trace)
            timings = trace.get("stage_timings_ms") or {}
            if timings:
                joined = ", ".join(f"{k}={v:.2f}ms" for k, v in sorted(timings.items()))
                print(f"      stage latencies: {joined}")

    if not traces:
        print("\n  (No decision traces found for this window — traces may have been"
              "\n   written before tracing existed, or cleaned up.)")


def main():
    parser = argparse.ArgumentParser(description="Replay a historical trade's decision journey")
    parser.add_argument("--list", type=int, nargs="?", const=15, metavar="N",
                        help="list N recent trades (default 15)")
    parser.add_argument("--trade-id", default=None)
    parser.add_argument("--window-bars", type=int, default=6)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--input", default=None, help="OHLCV CSV/Parquet for market context")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    args = parser.parse_args()

    config = load_layered_config(args.env)
    granularity = config.get("system", {}).get("data_granularity", "1m")
    granularity_seconds = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}.get(granularity, 60)

    if args.list is not None:
        trades = list_trades(args.list, args.db)
        if trades.empty:
            print("No trades journaled yet.")
            return
        show = trades.copy()
        show["entry"] = show["entry_timestamp"].map(fmt_ts)
        show["pnl%"] = show["realized_pnl_pct"].map(lambda p: f"{p:+.3f}" if p is not None else "open")
        print(show[["trade_id", "entry", "expert_used", "regime_at_entry",
                    "direction", "pnl%", "exit_reason"]].to_string(index=False))
        return

    if not args.trade_id:
        parser.error("either --list or --trade-id is required")

    trade = load_trade(args.trade_id, args.db)
    if trade is None:
        print(f"No trade found with id {args.trade_id}")
        sys.exit(1)

    replay(trade, args.window_bars, granularity_seconds, args.input)


if __name__ == "__main__":
    main()
