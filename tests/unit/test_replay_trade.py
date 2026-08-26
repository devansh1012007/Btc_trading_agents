"""
Trade-replay tooling tests (PRESENTATION §10): the replay CLI's data
stitching — journal row + decision-trace window -> rendered timeline.
Runs against a temp SQLite journal and temp trace partition, no repo state.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_trade as rt  # noqa: E402

from src.feedback.trade_journal import record_trade_entry, record_trade_exit  # noqa: E402

pytestmark = [pytest.mark.unit]

ENTRY_TS = 1_900_000_000
EXIT_TS = ENTRY_TS + 4 * 60


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    """Journal with one closed trade + matching trace partition."""
    db_path = str(tmp_path / "registry.db")
    trade_id = record_trade_entry(
        entry_timestamp=ENTRY_TS, entry_price=50_000.0, expert_used="breakout_expert",
        regime_at_entry="trending_up", direction=1, size=480.0,
        router_confidence=0.76, risk_score_at_entry=0.19,
        mode="paper_sim", db_path=db_path,
    )
    record_trade_exit(
        trade_id=trade_id, exit_timestamp=EXIT_TS, exit_price=50_300.0,
        exit_reason="take_profit", realized_pnl_pct=0.55, realized_pnl_usdt=2.64,
        fees_paid=0.38, slippage_realized=0.02, db_path=db_path,
    )

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    day = datetime.fromtimestamp(ENTRY_TS, tz=timezone.utc).strftime("%Y-%m-%d")
    stages = [
        {"stage_name": "expert_layer", "passed": True, "detail": "1 candidate(s): breakout_expert",
         "values": {"breakout_expert": 0.84}},
        {"stage_name": "opportunity_filter", "passed": True, "detail": "passed",
         "values": {"tradable_probability": 0.82}},
        {"stage_name": "router", "passed": True, "detail": "",
         "values": {"selection_confidence": 0.76}},
        {"stage_name": "risk_layer", "passed": True, "detail": "",
         "values": {"risk_score": 0.19}},
        {"stage_name": "execution", "passed": True, "detail": "status=filled",
         "values": {"fill_price": 50_010.0}},
    ]
    trace = {
        "timestamp": ENTRY_TS, "symbol": "BTC/USDT", "stages": stages,
        "terminal_stage": "executed", "terminal_reason": "filled via breakout_expert",
        "total_duration_ms": 1.2, "stage_timings_ms": {"features_ms": 0.4, "regime_ms": 0.1},
    }
    (trace_dir / f"traces_{day}.jsonl").write_text(json.dumps(trace) + "\n")

    # Pre-entry bar trace (no-trade bar) so the timeline has context.
    pre = {**trace, "timestamp": ENTRY_TS - 60, "terminal_stage": "opportunity_filter",
           "terminal_reason": "filter veto"}
    with open(trace_dir / f"traces_{day}.jsonl", "a") as f:
        f.write(json.dumps(pre) + "\n")

    monkeypatch.setattr(rt, "TRACE_DIR", trace_dir)
    return rt, db_path, trade_id


def test_list_and_load(seeded):
    rt_mod, db_path, trade_id = seeded
    trades = rt_mod.list_trades(10, db_path)
    assert len(trades) == 1
    loaded = rt_mod.load_trade(trade_id, db_path)
    assert loaded["expert_used"] == "breakout_expert"
    assert float(loaded["realized_pnl_pct"]) == pytest.approx(0.55)


def test_replay_renders_entry_stages_and_timeline(seeded, capsys):
    rt_mod, db_path, trade_id = seeded
    trade = rt_mod.load_trade(trade_id, db_path)

    rt_mod.replay(trade, window_bars=6, granularity_seconds=60, input_path=None)
    out = capsys.readouterr().out

    assert "TRADE REPLAY" in out
    assert "take_profit" in out            # exit outcome surfaced
    assert "+0.550%" in out                # realized pnl
    assert "<<< ENTRY" in out
    assert "<<< EXIT" in out
    assert "--- full pipeline stages at entry ---" in out
    assert "risk_layer" in out and "PASS" in out
    assert "stage latencies" in out
    assert "terminal=opportunity_filter" in out   # pre-entry veto bar visible


def test_trace_window_scanning_respects_bounds(seeded):
    rt_mod, _db, _tid = seeded
    inside = rt_mod.load_traces_in_window(ENTRY_TS - 120, ENTRY_TS + 120)
    outside = rt_mod.load_traces_in_window(ENTRY_TS + 10_000, ENTRY_TS + 20_000)
    assert len(inside) == 2
    assert outside == []
