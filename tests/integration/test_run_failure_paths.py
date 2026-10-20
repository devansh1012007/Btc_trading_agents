"""
Integration test: the deployment loop's failure paths.

``scripts/run.py`` is the entrypoint the Trade Map GUI's data comes
from. Two of its branches decide what the GUI shows for a bar that
failed, and neither is reachable through the normal paper-sim path:

* an order placement that RAISES (the exchange is unreachable) must be
  reported as ``execution_failed``, not as "no expert candidate";
* an order that is REJECTED must not be reported as a trade.

The tests drive the real ``run_historical`` loop with a stubbed
execution engine, in an isolated working directory, so nothing touches
the repository's ``data/``.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.execution.execution_engine import ExecutionEngine  # noqa: E402
from src.gui.replay import relax_liquidity, synthetic_ohlcv  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402

pytestmark = [pytest.mark.integration]


def _load_run_module():
    """Import ``scripts/run.py`` as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("trade_run", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trade_run"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_module():
    return _load_run_module()


@pytest.fixture(scope="module")
def config():
    return load_layered_config("paper")


def _build_stack(run_module, config, bars, execution_engine):
    """The same objects ``main()`` builds for a historical run."""
    from src.execution.position_lifecycle import build_execution_engine
    from src.layers.risk_layer import RiskLayer, RiskStateStore
    from src.pipeline import TradingPipeline

    # The shipped liquidity gates reject every bar of the synthetic
    # series (its spread is wider than the gate allows), so the run
    # would never attempt an order and the test would prove nothing.
    cfg = relax_liquidity(config)
    cfg = {**cfg, "execution": {"mode": "paper_sim"}}
    pipeline = TradingPipeline(symbol="BTC/USDT", config=cfg)
    risk_layer = RiskLayer(
        config=cfg.get("risk", {}),
        store=RiskStateStore(db_path="data/registry.db"),
        model_path=cfg.get("risk", {}).get("ml_model", {}).get("model_path"),
        data_granularity=cfg.get("system", {}).get("data_granularity", "1m"),
        kill_switch_path="data/control/EMERGENCY_STOP",
    )
    if execution_engine is None:
        execution_engine = build_execution_engine(cfg, pipeline.connector, "BTC/USDT")
    return pipeline, risk_layer, execution_engine


def _load_through_the_runtime(run_module, bars, workdir):
    """The runtime's OWN loader, so its timestamp handling is the code
    under test (not a copy of it in the test)."""
    csv_path = workdir / "bars.csv"
    bars.to_csv(csv_path, index=False)
    return run_module.load_ohlcv_for_run(csv_path)


def _run(run_module, config, bars, execution_engine, workdir):
    """Run the historical loop in an isolated working directory."""
    import os

    previous = Path.cwd()
    os.chdir(workdir)
    try:
        pipeline, risk_layer, engine = _build_stack(
            run_module, config, bars, execution_engine
        )
        run_module.run_historical(
            pipeline, risk_layer, engine, bars, "paper", 0.04, 10_000.0
        )
    finally:
        os.chdir(previous)
    path = workdir / "data" / "logs" / "decisions.jsonl"
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_a_raised_order_is_reported_as_execution_failed(run_module, config, tmp_path):
    """An order that blew up must not be reported as "no expert
    candidate" -- that names a stage which had nothing to do with the
    failure."""
    from src.execution.backends.paper_sim_backend import PaperSimBackend

    class _ExplodingBackend(PaperSimBackend):
        def place_order(self, *args, **kwargs):
            raise RuntimeError("exchange unreachable")

    engine = ExecutionEngine(_ExplodingBackend({}, fee_pct=0.04), "paper_sim")

    bars = synthetic_ohlcv(900, seed=42)
    records = _run(run_module, config, bars, engine, tmp_path)
    assert records, "no decision records were written"

    failed = [r for r in records if r.get("terminal_stage") == "execution_failed"]
    assert failed, (
        "no bar was reported as a failed execution; stages seen: "
        f"{ {r.get('terminal_stage') for r in records} }"
    )
    for record in failed:
        assert record.get("decision") == "NO_TRADE"
        assert "exchange unreachable" in str(record.get("terminal_reason"))
        assert record.get("risk_approved") is True, (
            "the risk layer had approved the trade -- the order is what failed"
        )


def test_a_rejected_order_is_not_reported_as_a_trade(run_module, config, tmp_path):
    """A rejected order never filled: it is not a trade."""
    from src.execution.backends.paper_sim_backend import PaperSimBackend

    class _RejectingBackend(PaperSimBackend):
        def place_order(self, risk_decision, router_decision, expert_output,
                        market_snapshot, **kwargs):
            record = super().place_order(
                risk_decision, router_decision, expert_output, market_snapshot, **kwargs
            )
            return type(record)(
                timestamp=record.timestamp, order_timestamp=record.order_timestamp,
                fill_timestamp=record.fill_timestamp, symbol=record.symbol,
                side=record.side, order_price=record.order_price,
                fill_price=record.fill_price, requested_size=record.requested_size,
                filled_size=0.0, slippage_pct=record.slippage_pct,
                fees_pct=record.fees_pct, total_cost_pct=record.total_cost_pct,
                status="rejected", order_id=record.order_id,
            )

    engine = ExecutionEngine(_RejectingBackend({}, fee_pct=0.04), "paper_sim")

    bars = synthetic_ohlcv(900, seed=42)
    records = _run(run_module, config, bars, engine, tmp_path)
    assert records

    rejected = [r for r in records if r.get("terminal_stage") == "execution_failed"]
    assert rejected, "the test never produced a rejected order"
    assert all(r.get("decision") == "NO_TRADE" for r in records), (
        "a rejected order was reported as a trade"
    )
    assert not [r for r in records if r.get("decision") in ("LONG", "SHORT")]


def test_a_position_open_bar_is_reported_as_such(run_module, config, tmp_path):
    """The bar that could not open a second position says so."""
    bars = synthetic_ohlcv(900, seed=42)
    records = _run(run_module, config, bars, None, tmp_path)
    stages = {r.get("terminal_stage") for r in records}
    assert "position_open" in stages, (
        f"no bar was reported as blocked by an open position; stages: {stages}"
    )
    open_rows = [r for r in records if r.get("terminal_stage") == "position_open"]
    for record in open_rows:
        assert record.get("decision") == "NO_TRADE"
        assert "position" in str(record.get("terminal_reason"))


def test_a_rejected_order_does_not_reach_the_trade_journal(run_module, config, tmp_path):
    """The §8 feed is not the only record: a rejected order must not
    produce a journal entry either, or the GUI would draw a trade with
    no size."""
    from src.execution.backends.paper_sim_backend import PaperSimBackend

    class _RejectingBackend(PaperSimBackend):
        def place_order(self, risk_decision, router_decision, expert_output,
                        market_snapshot, **kwargs):
            record = super().place_order(
                risk_decision, router_decision, expert_output, market_snapshot, **kwargs
            )
            return type(record)(
                timestamp=record.timestamp, order_timestamp=record.order_timestamp,
                fill_timestamp=record.fill_timestamp, symbol=record.symbol,
                side=record.side, order_price=record.order_price,
                fill_price=record.fill_price, requested_size=record.requested_size,
                filled_size=0.0, slippage_pct=record.slippage_pct,
                fees_pct=record.fees_pct, total_cost_pct=record.total_cost_pct,
                status="rejected", order_id=record.order_id,
            )

    engine = ExecutionEngine(_RejectingBackend({}, fee_pct=0.04), "paper_sim")
    bars = synthetic_ohlcv(900, seed=42)
    _run(run_module, config, bars, engine, tmp_path)

    import sqlite3

    journal = tmp_path / "data" / "registry.db"
    if not journal.exists():
        return  # nothing was journalled at all -- nothing to assert
    conn = sqlite3.connect(journal)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM trade_journal")]
    conn.close()
    # Every order was rejected, so NO trade may be journalled at all.
    for row in rows:
        assert row["size"] and float(row["size"]) > 0, (
            f"a zero-size trade was journalled: {row}"
        )
    assert not rows, (
        f"{len(rows)} trade(s) were journalled for orders that never filled: {rows[:2]}"
    )


def test_the_runtime_records_the_trade_provenance(run_module, config, tmp_path):
    """The open position must carry the expert, regime, confidence and
    risk score -- the GUI's trade table reads them."""
    import sqlite3

    bars = synthetic_ohlcv(900, seed=42)
    _run(run_module, config, bars, None, tmp_path)

    journal = tmp_path / "data" / "registry.db"
    assert journal.exists()
    conn = sqlite3.connect(journal)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM trade_journal")]
    conn.close()
    assert rows, "the run journalled no trades"
    for row in rows:
        assert row["expert_used"], f"the trade lost its expert: {row}"
        assert row["regime_at_entry"], f"the trade lost its regime: {row}"
        assert row["router_confidence"] is not None


def test_the_open_position_carries_its_provenance(run_module, config, tmp_path, monkeypatch):
    """The open position must carry the expert, regime, confidence and
    risk score.

    The assertion is on the POSITION, not on the trade journal: the
    journal is filled from a separate call, so a journal-only assertion
    would not notice the position losing its provenance.
    """
    captured = []
    original = run_module.check_position_exit

    def capture(position, bar, fee_pct):
        captured.append(position)
        return original(position, bar, fee_pct)

    monkeypatch.setattr(run_module, "check_position_exit", capture)

    bars = synthetic_ohlcv(900, seed=42)
    _run(run_module, config, bars, None, tmp_path)

    assert captured, "no position was ever opened"
    position = captured[0]
    assert position.expert, f"the position lost its expert: {position}"
    assert position.regime, f"the position lost its regime: {position}"
    assert position.selection_confidence is not None
    assert position.risk_score is not None


def test_the_runtime_accepts_a_string_epoch_timestamp_column(run_module, config, tmp_path):
    """A CSV whose timestamp column holds epoch seconds as TEXT must
    load: the runtime and the GUI must agree on what a timestamp is."""
    bars = synthetic_ohlcv(300, seed=12)
    bars = bars.assign(timestamp=bars["timestamp"].astype(str))
    reloaded = _load_through_the_runtime(run_module, bars, tmp_path)

    records = _run(run_module, config, reloaded, None, tmp_path)
    assert records, "no decision records were written"
    stamps = [r["timestamp"] for r in records]
    assert min(stamps) > 1_600_000_000, f"the timestamps collapsed: {stamps[:5]}"
