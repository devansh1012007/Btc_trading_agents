"""
E2E test: the deployment entrypoint's observability feed.

``scripts/run.py --mode historical`` is what produces the artifacts the
Trade Map GUI reads: the §8 decision log, the decision traces and the
trade journal. This test runs it exactly as a user would and checks the
three properties the GUI depends on:

* the run completes and writes every artifact;
* bar timestamps are real wall-clock times (an integer epoch column must
  not be read as nanoseconds);
* the decision log classifies executed bars as ``executed`` -- the trace
  has to be written AFTER the bar's risk and execution results exist,
  otherwise every trade is invisible to the GUI.
"""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "run.py"

pytestmark = [pytest.mark.e2e]


def write_synthetic_csv(path: Path, n_bars: int = 900, seed: int = 42) -> None:
    """The same mixed-regime synthetic series the backtest harness uses."""
    rng = np.random.default_rng(seed)
    start = 1_767_225_600  # 2026-01-01T00:00:00Z, whole minutes
    price, rows = 50000.0, []
    segment = n_bars // 4
    for i in range(n_bars):
        drift = 12.0 if i < segment else (-10.0 if i < 2 * segment else (2.0 if i < 3 * segment else -18.0))
        vol = 8.0 if i >= 3 * segment else 5.0
        price = max(price + rng.normal(drift, vol), 1000.0)
        open_ = price + rng.normal(0, 3)
        rows.append({
            "timestamp": start + i * 60,
            "open": open_,
            "high": max(open_, price) + abs(rng.normal(0, vol * 0.8)),
            "low": min(open_, price) - abs(rng.normal(0, vol * 0.8)),
            "close": price,
            "volume": abs(rng.normal(120, 35)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    """Run the deployment entrypoint in an isolated working directory.

    The copied paper overlay relaxes the opportunity filter's liquidity
    gates: the shipped synthetic liquidity model produces a wider spread
    than the gate allows, so an unrelaxed run rejects essentially every
    bar and the journal stays empty. Relaxing them in the ISOLATED copy
    (never in the repo's config) keeps the test able to observe a real
    fill without changing what the deployment entrypoint ships.
    """
    work = tmp_path_factory.mktemp("run_obs")
    shutil.copytree(ROOT / "config", work / "config")
    with open(work / "config" / "settings.paper.yaml", "a", encoding="utf-8") as fh:
        fh.write(
            "\nopportunity_filter:\n"
            "  hard_rules:\n"
            "    max_spread_pct: 0.5\n"
            "    min_depth_usdt: 0.0\n"
            "    edge_vs_fees_threshold: -10.0\n"
        )
    csv_path = work / "bars.csv"
    write_synthetic_csv(csv_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "historical",
         "--input", str(csv_path), "--env", "paper"],
        cwd=str(work), capture_output=True, text=True, timeout=600,
    )
    return result, work


def test_historical_run_succeeds(run_result):
    result, _ = run_result
    assert result.returncode == 0, f"run failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"


def test_run_writes_every_artifact_the_gui_reads(run_result):
    _, work = run_result
    assert (work / "data" / "logs" / "decisions.jsonl").exists()
    assert list((work / "data" / "logs" / "decision_traces").glob("traces_*.jsonl"))
    assert (work / "data" / "registry.db").exists()
    assert (work / "data" / "logs" / "zone_events.parquet").exists()


def _read_decisions(work: Path) -> list:
    return [
        json.loads(line)
        for line in (work / "data" / "logs" / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _read_traces(work: Path) -> list:
    traces = []
    for path in sorted((work / "data" / "logs" / "decision_traces").glob("traces_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                traces.append(json.loads(line))
    return traces


def test_bar_timestamps_are_real_times_not_nanoseconds(run_result):
    _, work = run_result
    records = _read_decisions(work)
    assert records, "no decision records written"
    stamps = [r["timestamp"] for r in records]
    assert min(stamps) > 1_600_000_000, f"timestamps collapsed to the epoch: {stamps[:5]}"
    # One record per bar: strictly increasing AND without duplicates.
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps)), "duplicate decision records"


def test_the_decision_trace_files_also_report_executed_bars(run_result):
    """The trace FILE (not just the §8 feed) must carry the terminal
    stage. The GUI reads the trace files, so a correct §8 record with a
    wrong trace file would still show the wrong thing."""
    _, work = run_result
    traces = _read_traces(work)
    assert traces, "no decision traces written"
    stages = {t.get("terminal_stage") for t in traces}
    assert "executed" in stages, f"no executed bars in the trace files: {stages}"

    executed = [t for t in traces if t.get("terminal_stage") == "executed"]
    assert executed
    for trace in executed:
        assert trace.get("total_duration_ms") is not None
        assert trace.get("stage_timings_ms"), "trace carries no per-stage timings"
        stage_names = {s.get("stage_name") for s in trace.get("stages", [])}
        assert "execution" in stage_names, "an executed bar has no execution stage"


def test_the_decision_feed_carries_the_fields_the_gui_schema_promises(run_result):
    _, work = run_result
    records = _read_decisions(work)
    for record in records[:50]:
        assert "regime_confidence" in record
        assert "risk_approved" in record


def test_executed_bars_are_classified_as_executed(run_result):
    """The decision record is written after the bar's risk/execution
    results, so a filled bar must show terminal_stage == 'executed'."""
    _, work = run_result
    records = _read_decisions(work)
    stages = {r.get("terminal_stage") for r in records}
    assert "opportunity_filter" in stages, f"no filter vetoes recorded: {stages}"

    executed = [r for r in records if r.get("terminal_stage") == "executed"]
    assert executed, f"no executed bars recorded; stages seen: {stages}"
    assert all(r.get("decision") in ("LONG", "SHORT") for r in executed)


def test_trade_journal_agrees_with_the_executed_bars(run_result):
    _, work = run_result
    records = _read_decisions(work)
    executed_ts = {r["timestamp"] for r in records if r.get("terminal_stage") == "executed"}

    conn = sqlite3.connect(work / "data" / "registry.db")
    conn.row_factory = sqlite3.Row
    trades = [dict(r) for r in conn.execute("SELECT * FROM trade_journal")]
    conn.close()

    assert trades, "the trade journal is empty"
    assert {t["entry_timestamp"] for t in trades} == executed_ts


def _replayed_session(work: Path):
    from src.gui.artifacts import load_ohlcv
    from src.gui.replay import relax_liquidity, replay_session
    from src.utils.config_helpers import load_layered_config

    bars = load_ohlcv(str(work / "bars.csv"))
    assert not bars.empty
    config = relax_liquidity(load_layered_config("paper"))
    return replay_session(bars, "BTC/USDT", config, relax=False)


def _runtime_trades(work: Path) -> list:
    conn = sqlite3.connect(work / "data" / "registry.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM trade_journal ORDER BY entry_timestamp")]
    conn.close()
    return rows


def test_the_gui_replay_and_the_runtime_agree_on_every_bar(run_result):
    """The change set's central claim: the runtime and the GUI can never
    disagree about why a bar stopped where it did.

    Both paths now run the SAME ``process_bar``, so this is a regression
    guard on the wiring (callbacks, config, store), not the only thing
    holding the two together. It compares the terminal stage of every bar.
    """
    _, work = run_result
    runtime = {r["timestamp"]: r.get("terminal_stage") for r in _read_decisions(work)}
    session = _replayed_session(work)

    gui = {
        int(ts): stage
        for ts, stage in zip(
            session.decisions["timestamp"], session.decisions["terminal_stage"], strict=True
        )
    }

    common = set(runtime) & set(gui)
    assert common, "the runtime and the GUI produced no bars in common"
    assert set(runtime) == set(gui), "the two paths processed different bars"
    mismatches = {
        ts: (runtime[ts], gui[ts]) for ts in common if runtime[ts] != gui[ts]
    }
    assert not mismatches, f"the runtime and the GUI disagree on {len(mismatches)} bar(s): " \
        f"{dict(list(mismatches.items())[:5])}"


def test_the_gui_replay_and_the_runtime_agree_on_the_trade_list(run_result):
    """Same bars in -> the same trades out: entry, exit, price, reason, PnL."""
    _, work = run_result
    session = _replayed_session(work)
    runtime = _runtime_trades(work)

    gui = session.trades.sort_values("entry_timestamp").reset_index(drop=True)
    assert len(runtime) == len(gui), (
        f"the runtime journalled {len(runtime)} trades, the GUI replayed {len(gui)}"
    )

    for journal, replayed in zip(runtime, gui.to_dict("records"), strict=True):
        assert journal["entry_timestamp"] == replayed["entry_timestamp"]
        assert journal["direction"] == replayed["direction"]
        assert journal["entry_price"] == pytest.approx(replayed["entry_price"])
        assert journal["size"] == pytest.approx(replayed["size"])
        assert journal["expert_used"] == replayed["expert"]
        assert journal["regime_at_entry"] == replayed["regime"]
        if replayed["exit_reason"] == "open":
            assert journal["exit_timestamp"] is None
            continue
        assert journal["exit_timestamp"] == replayed["exit_timestamp"]
        assert journal["exit_reason"] == replayed["exit_reason"]
        assert journal["exit_price"] == pytest.approx(replayed["exit_price"])
        assert journal["realized_pnl_usdt"] == pytest.approx(replayed["pnl_usdt"])
        assert journal["realized_pnl_pct"] == pytest.approx(replayed["pnl_pct"])


def test_the_gui_replay_and_the_runtime_agree_on_the_equity_curve(run_result):
    """The equity at every bar -- not just the final number -- must match.

    The runtime writes no equity curve, so it is rebuilt from the trade
    journal: initial equity plus the realised PnL of every trade that had
    exited by that bar.
    """
    _, work = run_result
    session = _replayed_session(work)
    initial = float(session.meta["initial_equity"])

    exits = sorted(
        (t["exit_timestamp"], t["realized_pnl_usdt"])
        for t in _runtime_trades(work)
        if t["exit_timestamp"] is not None
    )

    equity = initial
    cursor = 0
    mismatches = []
    for ts, gui_equity in zip(session.equity["timestamp"], session.equity["equity"], strict=True):
        while cursor < len(exits) and exits[cursor][0] <= ts:
            equity += exits[cursor][1]
            cursor += 1
        if abs(equity - gui_equity) > 1e-6 * max(1.0, abs(equity)):
            mismatches.append((int(ts), equity, float(gui_equity)))
    assert not mismatches, f"equity curves differ on {len(mismatches)} bar(s): {mismatches[:3]}"
    assert session.meta["final_equity"] == pytest.approx(equity)
