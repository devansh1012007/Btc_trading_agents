#!/usr/bin/env python
"""
Trade Map CLI: export a session as ONE self-contained HTML file and print
a text summary.

    python scripts/trade_map.py --source synthetic --out trade_map.html
    python scripts/trade_map.py --source artifacts --symbol BTC/USDT --timeframe 5m \\
        --out data/reports/trade_map.html
    make trade-map-export

The HTML embeds plotly.js, so it opens offline and can be mailed around.
This tool is a viewer: it never places an order and writes only the file
named by ``--out`` (never a tracked ``data/`` artifact; ``--out`` refuses
to overwrite the registry or the logs).

Exit status is 0 -- with a clear message, never a traceback -- when there
is simply nothing to show yet.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROTECTED = ("data/registry.db", "data/logs", "data/raw", "data/models")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Trade Map as a self-contained HTML file")
    parser.add_argument("--source", default="auto", choices=("auto", "artifacts", "replay", "synthetic"),
                        help="auto = artifacts -> replay -> synthetic (default)")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default=None, help="display timeframe (default: sensible for the data)")
    parser.add_argument("--out", default=None, help="HTML file to write (default: no HTML, summary only)")
    parser.add_argument("--max-bars", type=int, default=1500, help="replay: most recent N bars")
    parser.add_argument("--bars", type=int, default=900, help="synthetic: number of bars")
    parser.add_argument("--seed", type=int, default=42, help="synthetic: RNG seed")
    parser.add_argument("--relax-liquidity", action="store_true", help="replay: loosen the liquidity gates")
    parser.add_argument("--indicators", default="ema_fast,ema_slow",
                        help="comma-separated price-pane indicators ('' for none)")
    parser.add_argument("--panes", default="", help="comma-separated extra panes: rsi,macd")
    parser.add_argument("--env", default="paper", choices=("paper", "live"), help="config overlay to read")
    return parser.parse_args(argv)


def _check_output_path(out: Path) -> None:
    resolved = out.resolve()
    for protected in PROTECTED:
        target = (ROOT / protected).resolve()
        if resolved == target or target in resolved.parents:
            raise SystemExit(f"refusing to write {out}: {protected} is a production artifact")
    if resolved.suffix.lower() not in (".html", ".htm"):
        raise SystemExit(f"--out must be an .html file, got {out.name}")


def default_timeframe(bundle) -> str:
    """Coarsest-sensible default: keep roughly <= 1500 candles on screen."""
    from src.gui.timeframes import DISPLAY_TIMEFRAMES, display_timeframes

    allowed = display_timeframes(bundle.base_timeframe)
    span = int(bundle.bars["timestamp"].iloc[-1] - bundle.bars["timestamp"].iloc[0]) if len(bundle.bars) > 1 else 0
    for tf in allowed:
        if span / DISPLAY_TIMEFRAMES[tf] <= 1500:
            return tf
    return allowed[-1]


def format_summary(bundle, summary) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value:.2f}%"

    lines = [
        f"Trade Map - {bundle.instrument.label()}  [{bundle.source_label}]",
        f"  bars: {summary['n_bars']:,}   decisions: {summary['n_decisions']:,}   "
        f"trades: {summary['n_trades']} ({summary['n_closed']} closed, {summary['n_open']} open)",
        f"  win rate: {pct(summary['win_rate_pct'])}   expectancy/trade: {pct(summary['expectancy_pct'])}   "
        f"total PnL: {summary['total_pnl_usdt']:+,.2f}   max drawdown: {summary['max_drawdown_pct']:.2f}%",
    ]
    if summary["funnel"]:
        lines.append("  where each bar stopped:")
        for row in summary["funnel"]:
            lines.append(f"    {row['label']:<28} {row['bars']:>7,}  {row['pct']:>5.1f}%")
    for warning in bundle.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else None
    if out is not None:
        _check_output_path(out)

    from src.gui.sources import (
        SourceUnavailable,
        auto_source,
        load_config,
        load_session,
        summarize_session,
    )

    config = load_config(args.env)
    source = auto_source(args.symbol) if args.source == "auto" else args.source
    try:
        bundle = load_session(
            source, args.symbol, config=config, max_bars=args.max_bars,
            n_bars=args.bars, seed=args.seed, relax=args.relax_liquidity or source == "synthetic",
        )
    except SourceUnavailable as exc:
        print(f"Nothing to show from the '{source}' source: {exc}.")
        print("Run the system first (make pipeline / scripts/run.py), or try --source synthetic.")
        return 0

    if not bundle.has_content():
        print(f"The '{source}' session for {args.symbol} is empty - nothing to show yet.")
        return 0

    summary = summarize_session(bundle)
    print(format_summary(bundle, summary))
    if out is None:
        return 0

    try:
        from src.gui.charts import build_session_figures
    except ImportError:
        print("\nHTML export needs plotly:  pip install -e '.[viz]'", file=sys.stderr)
        return 2

    timeframe = args.timeframe or default_timeframe(bundle)
    indicators = [i for i in args.indicators.split(",") if i]
    panes = [p for p in args.panes.split(",") if p]
    figures = build_session_figures(bundle, timeframe, indicators=indicators, panes=panes)

    from src.gui.export import render_report_html

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report_html(bundle, figures, summary, timeframe), encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB) - open it in any browser, offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
