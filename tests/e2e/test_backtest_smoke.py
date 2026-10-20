"""
E2E test: full-stack backtest entrypoint, run exactly as a user would.

Executes scripts/full_stack_walk_forward_backtest.py as a subprocess over
synthetic OHLCV (no network, no data files) and verifies the complete
artifact chain: process exits 0, JSON + markdown reports land in the
report dir, and the report payload carries the full metric suite.

This is the same command `make benchmark-smoke` runs; here it doubles as
the regression net for the deployment entrypoint itself.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "full_stack_walk_forward_backtest.py"

pytestmark = [pytest.mark.e2e]


def _run_synthetic_backtest(report_dir: Path, n_bars: int, workdir: Path) -> subprocess.CompletedProcess:
    """Run the backtest entrypoint in an isolated working directory.

    The entrypoint resolves both the config and the experiment log
    relative to the process CWD, so running it from the repository root
    would write into the TRACKED ``data/registry.db`` and dirty the
    working tree on every test run. The config is copied in so the run
    still sees the shipped settings.
    """
    shutil.copytree(ROOT / "config", workdir / "config", dirs_exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--synthetic", str(n_bars),
            "--relax-liquidity",
            "--window-bars", "100",
            "--report-dir", str(report_dir),
        ],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_full_stack_backtest_produces_reports(tmp_path):
    result = _run_synthetic_backtest(tmp_path, n_bars=300, workdir=tmp_path / "run")
    assert result.returncode == 0, f"backtest failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    json_files = list(tmp_path.glob("backtest_*.json"))
    md_files = list(tmp_path.glob("backtest_*.md"))
    assert len(json_files) == 1, f"expected one JSON report, got {json_files}"
    assert len(md_files) == 1, f"expected one markdown report, got {md_files}"

    report = json.loads(json_files[0].read_text())

    # Report provenance
    assert report["config_hash"], "config hash missing from report"
    assert report["generated_at"]

    # Full realized-metric suite present and non-null (PRESENTATION §2 table)
    agg = report["aggregate"]
    for key in (
        "n_trades", "total_return_pct", "max_drawdown_pct", "win_rate_pct",
        "profit_factor", "expectancy_per_trade_pct",
        "sharpe_ratio", "sortino_ratio",
    ):
        assert key in agg, f"metric '{key}' missing from aggregate"
        assert agg[key] is not None, f"metric '{key}' was null"

    # Baselines ran alongside (buy&hold / momentum / naive router / unfiltered)
    baselines = report.get("baselines", {})
    expected_baselines = {
        "buy_and_hold", "simple_momentum",
        "naive_router", "unfiltered",
    }
    assert expected_baselines.issubset(baselines.keys()), (
        f"missing baselines: {expected_baselines - set(baselines.keys())}"
    )

    # Markdown report renders headline numbers, not an empty shell
    md_text = md_files[0].read_text()
    assert "Success criteria" in md_text


def test_full_stack_backtest_is_deterministic_for_fixed_seed(tmp_path):
    """
    Same synthetic seed → identical trade count. Guards against accidental
    nondeterminism sneaking into the pipeline (dict ordering, RNG reuse).
    """
    first = _run_synthetic_backtest(tmp_path / "a", n_bars=300, workdir=tmp_path / "run_a")
    second = _run_synthetic_backtest(tmp_path / "b", n_bars=300, workdir=tmp_path / "run_b")
    assert first.returncode == 0 and second.returncode == 0

    r1 = json.loads((tmp_path / "a" / sorted(p.name for p in (tmp_path / "a").glob("*.json"))[0]).read_text())
    r2 = json.loads((tmp_path / "b" / sorted(p.name for p in (tmp_path / "b").glob("*.json"))[0]).read_text())
    assert r1["aggregate"]["n_trades"] == r2["aggregate"]["n_trades"]


def test_the_backtest_does_not_touch_the_production_registry(tmp_path):
    """A test run must never dirty the tracked data/registry.db."""
    import hashlib

    registry = ROOT / "data" / "registry.db"
    before = hashlib.sha256(registry.read_bytes()).hexdigest() if registry.exists() else None

    result = _run_synthetic_backtest(tmp_path / "r", n_bars=200, workdir=tmp_path / "run")
    assert result.returncode == 0, f"backtest failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    after = hashlib.sha256(registry.read_bytes()).hexdigest() if registry.exists() else None
    assert before == after, "the backtest entrypoint wrote to the production registry"
