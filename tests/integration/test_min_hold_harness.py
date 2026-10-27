"""
End-to-end guarantees of the minimum-trade-duration change, through the REAL
full-stack harness (pipeline, risk layer, paper-sim execution, the shared
per-bar block). No model is trained: the harness runs with the experts'
rule-based fallback, which is the path a fresh or legacy checkout takes.

Marked ``integration``; it runs a few seconds per granularity.
"""
import copy
import importlib
import sys
from pathlib import Path

import pytest

from src.utils.synthetic import synthetic_ohlcv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = [pytest.mark.integration]

HARNESS = importlib.import_module("full_stack_walk_forward_backtest")


def relaxed(config, granularity, max_bars, latency_ms=0):
    cfg = copy.deepcopy(config)
    cfg["system"]["data_granularity"] = granularity
    cfg["labeling"]["max_holding_bars"] = max_bars
    cfg["execution"]["simulated_latency_ms"] = latency_ms
    cfg["execution"]["mode"] = "paper_sim"
    return HARNESS.maybe_relax_liquidity(cfg, True)


@pytest.fixture(scope="module")
def bars_10s():
    return synthetic_ohlcv(900, seed=7, bar_seconds=10)


@pytest.fixture(scope="module")
def bars_5s():
    return synthetic_ohlcv(1800, seed=7, bar_seconds=5)


def run(df, config, symbol="BTC/USDT"):
    return HARNESS.run_full_stack_pass(df, symbol, config, 10_000.0)


# ---------------------------------------------------------------------------
# the rule, end to end
# ---------------------------------------------------------------------------

def test_no_harness_trade_is_shorter_than_ten_seconds_at_10s_bars(paper_config, bars_10s):
    result = run(bars_10s, relaxed(paper_config, "10s", 5))
    trades = result["trades"]
    assert trades, "the synthetic series should produce trades under relaxed gates"
    assert min(t["hold_seconds"] for t in trades) >= 10.0


def test_no_harness_trade_is_shorter_than_ten_seconds_at_5s_bars(paper_config, bars_5s):
    """The case the rule actually bites: 5 s bars, so a one-bar trade is 5 s."""
    result = run(bars_5s, relaxed(paper_config, "5s", 10))
    trades = result["trades"]
    assert trades
    assert min(t["hold_seconds"] for t in trades) >= 10.0
    assert all(t["min_hold_bars"] == 2 for t in trades)


def test_latency_is_part_of_the_hold(paper_config, bars_10s):
    """With 2 s of simulated latency a 10 s bar's first exit is only 8 s after
    the fill, so every trade must now wait a second bar."""
    result = run(bars_10s, relaxed(paper_config, "10s", 6, latency_ms=2000))
    trades = result["trades"]
    assert trades
    assert min(t["hold_seconds"] for t in trades) >= 10.0
    assert min(t["bars_held"] for t in trades) >= 2


def test_the_harness_and_the_training_labeler_agree_on_every_trade(paper_config, bars_5s):
    """The alignment eval as a regression test: the labels a model would be
    trained on describe the trades the harness actually took."""
    from src.backtest.harness_harvest import harvest_expert_samples
    from src.evaluation.trade_duration import alignment_passed, alignment_report

    cfg = relaxed(paper_config, "5s", 10)
    result = run(bars_5s, cfg)
    labels, _, _ = harvest_expert_samples(
        bars_5s, "BTC/USDT", cfg, require_filter_pass=False, sample_policy="all")
    report = alignment_report(labels, result["trades"])
    assert report["n_trades"] > 0
    assert report["coverage"] == pytest.approx(1.0), report["unmatched"][:3]
    assert report["exit_reason_agreement"] == pytest.approx(1.0), report["disagreements"][:3]
    assert report["bars_held_agreement"] == pytest.approx(1.0)
    assert report["pnl_max_abs_error_pct"] < 1e-6
    assert alignment_passed(report)


def test_the_duration_report_passes_on_a_real_run(paper_config, bars_5s):
    from src.evaluation.trade_duration import duration_report
    from src.labeling.hold_policy import HoldPolicy

    cfg = relaxed(paper_config, "5s", 10)
    report = duration_report(run(bars_5s, cfg)["trades"], HoldPolicy.from_config(cfg))
    assert report["passed"] and report["overall"]["violations"] == 0
    assert report["overall"]["min_s"] >= 10.0


# ---------------------------------------------------------------------------
# the harness is the same per-bar block as the runtime
# ---------------------------------------------------------------------------

def test_the_harness_no_longer_carries_its_own_exit_rule():
    text = (ROOT / "scripts/full_stack_walk_forward_backtest.py").read_text()
    assert "process_bar(" in text
    assert "check_position_exit(" not in text
    assert "compute_barrier_pcts(" not in text


def test_the_harness_return_contract_includes_snapshots():
    """Regression: a rewrite once dropped this key and crashed run_backtest."""
    text = (ROOT / "scripts/full_stack_walk_forward_backtest.py").read_text()
    body = text.split("def run_full_stack_pass", 1)[1].split("def label_expert_candidates", 1)[0]
    assert body.count('"snapshots": snapshots') == 1


def test_a_policy_the_config_cannot_satisfy_stops_the_harness_before_any_trade(paper_config, bars_5s):
    from src.labeling.hold_policy import HoldPolicyError

    cfg = relaxed(paper_config, "1s", 5)                     # 5 x 1 s < 10 s
    with pytest.raises(HoldPolicyError):
        run(bars_5s, cfg)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_the_harness_report_carries_a_trade_duration_section():
    """``run_backtest`` -- the function that writes the harness report --
    must include the trade_duration section. Checked on the source, because a
    full run_backtest (three passes) is the e2e smoke test's job."""
    text = (ROOT / "scripts/full_stack_walk_forward_backtest.py").read_text()
    body = text.split("def run_backtest", 1)[1].split("\ndef config_hash", 1)[0]
    assert '"trade_duration": duration_report(' in body
    assert "HoldPolicy.from_config(config)" in body


def test_the_eval_script_runs_end_to_end_and_writes_its_report(tmp_path, paper_config, monkeypatch):
    eval_mod = importlib.import_module("evaluate_training_alignment")
    code = eval_mod.main([
        "--synthetic", "600", "--bar-seconds", "5", "--granularity", "5s", "--max-holding-bars", "10",
        "--relax-liquidity", "--report-dir", str(tmp_path),
    ])
    assert code == 0
    written = list(tmp_path.glob("training_alignment_*.json"))
    assert written, "the eval should write a JSON report"
    import json
    data = json.loads(written[0].read_text())
    assert data["passed"] is True
    assert data["checks"]["no_trade_shorter_than_minimum"] is True
    assert data["trade_duration"]["overall"]["violations"] == 0
