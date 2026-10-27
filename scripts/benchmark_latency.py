#!/usr/bin/env python
"""
Software performance benchmark (PRESENTATION §23/§24): measures the
decision pipeline's own latency and resource footprint on synthetic data —
no network, no market data required.

Measures, per bar (post-warmup):
  - end-to-end decision latency (run_live_bar wall clock)
  - per-stage pipeline latency (features / regime / structure / liquidity /
    decisions), from TradingPipeline.last_bar_timings

Reports mean / p50 / p95 / max, throughput in bars/sec, and peak RSS.
Output: <report-dir>/latency_<timestamp>.{json,md} + stdout table.

Usage:
    python scripts/benchmark_latency.py --bars 500
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.layers.market_data import MarketDataLayer  # noqa: E402
from src.pipeline import TradingPipeline  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402
from src.utils.synthetic import synthetic_ohlcv  # noqa: E402

WARMUP_BARS = 110  # > default feature lookback (100)


def generate_synthetic_ohlcv(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Same series as the walk-forward harness (``src/utils/synthetic.py``)."""
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return synthetic_ohlcv(n_bars, seed=seed, start=start)


def pct(values, q):
    return float(np.percentile(values, q)) if values else None


def hardware_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": None,
        "ram_total_gb": None,
    }
    try:
        info["cpu_count"] = __import__("os").cpu_count()
    except Exception:
        pass
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return info


def run_benchmark(bars: int) -> dict:
    config = load_layered_config("paper")
    symbol = config.get("system", {}).get("symbol", "BTC/USDT")
    df = generate_synthetic_ohlcv(WARMUP_BARS + bars)
    snapshots = MarketDataLayer.batch_create_from_dataframe(df, symbol)

    pipeline = TradingPipeline(symbol=symbol, config=config)

    # Warmup: fill rolling buffers so measured bars have full history.
    for snapshot in snapshots[:WARMUP_BARS]:
        pipeline.run_live_bar(snapshot)

    total_ms, stage_ms = [], {}
    processed = 0
    for snapshot in snapshots[WARMUP_BARS:]:
        t0 = time.perf_counter()
        state = pipeline.run_live_bar(snapshot)
        dt_ms = (time.perf_counter() - t0) * 1000
        if state is None:
            continue
        total_ms.append(dt_ms)
        processed += 1
        for stage, value in pipeline.last_bar_timings.items():
            stage_ms.setdefault(stage, []).append(value)

    import resource
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB -> GB

    def stats(values):
        return {
            "mean_ms": round(float(np.mean(values)), 3),
            "p50_ms": round(pct(values, 50), 3),
            "p95_ms": round(pct(values, 95), 3),
            "max_ms": round(float(np.max(values)), 3),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bars_processed": processed,
        "warmup_bars": WARMUP_BARS,
        "granularity": config.get("system", {}).get("data_granularity", "1m"),
        "hardware": hardware_info(),
        "peak_rss_gb": round(peak_rss_gb, 3),
        "end_to_end_decision_latency": stats(total_ms),
        "throughput_bars_per_sec": round(len(total_ms) / (sum(total_ms) / 1000.0), 1),
        "stage_latencies": {stage: stats(vals) for stage, vals in sorted(stage_ms.items())},
    }
    return report


def render_markdown(report: dict) -> str:
    e2e = report["end_to_end_decision_latency"]
    lines = [
        "# Decision-Latency Benchmark",
        "",
        f"Generated: {report['generated_at']}  |  bars: {report['bars_processed']} "
        f"(+{report['warmup_bars']} warmup)  |  granularity: {report['granularity']}",
        "",
        "| Metric | Mean | P50 | P95 | Max |",
        "|---|---|---|---|---|",
        (f"| **End-to-end decision (ms)** | {e2e['mean_ms']} | {e2e['p50_ms']} | "
         f"{e2e['p95_ms']} | {e2e['max_ms']} |"),
    ]
    for stage, s in report["stage_latencies"].items():
        lines.append(f"| {stage} | {s['mean_ms']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |")
    lines += [
        "",
        f"Throughput: **{report['throughput_bars_per_sec']} bars/sec** · "
        f"Peak RSS: **{report['peak_rss_gb']} GB**",
        "",
        f"Hardware: {report['hardware'].get('machine', '?')} · "
        f"{report['hardware'].get('cpu_count') or '?'} cores · "
        f"{report['hardware'].get('ram_total_gb') or '?'} GB RAM · Python {report['hardware']['python']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Pipeline latency benchmark")
    parser.add_argument("--bars", type=int, default=300, help="measured bars after warmup")
    parser.add_argument("--report-dir", default="data/reports")
    args = parser.parse_args()

    report = run_benchmark(args.bars)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"latency_{stamp}.json"
    md_path = out_dir / f"latency_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))

    print("\n" + "=" * 62)
    print("DECISION LATENCY BENCHMARK")
    print("=" * 62)
    e2e = report["end_to_end_decision_latency"]
    print(f"Bars processed : {report['bars_processed']}")
    print(f"Decision (ms)  : mean={e2e['mean_ms']}  p50={e2e['p50_ms']}  "
          f"p95={e2e['p95_ms']}  max={e2e['max_ms']}")
    for stage, s in report["stage_latencies"].items():
        print(f"  {stage:<14} mean={s['mean_ms']:<8} p95={s['p95_ms']}")
    print(f"Throughput     : {report['throughput_bars_per_sec']} bars/sec")
    print(f"Peak RSS       : {report['peak_rss_gb']} GB")
    print(f"\nReports saved to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
