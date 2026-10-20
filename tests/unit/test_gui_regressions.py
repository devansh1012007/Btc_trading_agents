"""
Regression tests for the bugs the Phase-1 review surfaced.

Each test here pins one specific failure mode that was live at the time
of the review, so a future change cannot silently reintroduce it.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import src.gui  # noqa: E402
from src.diagnostics.decision_trace import classify_terminal_stage  # noqa: E402
from src.gui.artifacts import load_ohlcv, read_trade_journal  # noqa: E402
from src.gui.indicators import compute_indicators  # noqa: E402
from src.gui.timeframes import align_to_timeframe, resample_ohlcv, to_epoch_seconds  # noqa: E402
from src.state.contracts import (  # noqa: E402
    ExecutionRecord,
    OpportunityScore,
    RiskDecision,
    RouterDecision,
    SystemState,
)
from src.utils.logging_setup import decision_record_from_state  # noqa: E402

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# T3: the package's public API must actually exist
# ---------------------------------------------------------------------------

def test_every_exported_name_is_importable():
    missing = [name for name in src.gui.__all__ if not hasattr(src.gui, name)]
    assert not missing, f"src.gui.__all__ advertises names that do not exist: {missing}"


def test_the_package_does_not_import_plotly_or_streamlit(monkeypatch):
    """The Docker image installs from requirements.txt, which has neither."""
    import importlib
    import sys as _sys

    for blocked in ("plotly", "streamlit"):
        monkeypatch.setitem(_sys.modules, blocked, None)
    for name in [n for n in list(_sys.modules) if n.startswith("src.gui")]:
        monkeypatch.delitem(_sys.modules, name, raising=False)
    importlib.import_module("src.gui")
    assert "plotly" not in _sys.modules or _sys.modules["plotly"] is None


# ---------------------------------------------------------------------------
# T1: datetime-stamped OHLCV must not be corrupted by the pandas-3
#     microsecond/nanosecond resolution change
# ---------------------------------------------------------------------------

def test_datetime_string_ohlcv_keeps_real_timestamps(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2023-11-14 00:00:00,100,101,99,100.5,10\n"
        "2023-11-14 00:01:00,100.5,102,100,101.5,11\n"
        "2023-11-14 00:02:00,101.5,103,101,102.5,12\n",
        encoding="utf-8",
    )
    frame = load_ohlcv(str(path))
    assert len(frame) == 3, "bars were collapsed by a unit-resolution bug"
    assert frame.iloc[0]["timestamp"] == 1699920000
    assert frame.iloc[1]["timestamp"] == 1699920060
    assert frame.iloc[0]["close"] == 100.5


def test_to_epoch_seconds_handles_all_timestamp_shapes():
    assert to_epoch_seconds(pd.Series([1_700_000_000])).iloc[0] == 1_700_000_000
    assert to_epoch_seconds(pd.Series([1_700_000_000_000])).iloc[0] == 1_700_000_000
    assert to_epoch_seconds(pd.Series(["2023-11-14 00:00:00"])).iloc[0] == 1_699_920_000


# ---------------------------------------------------------------------------
# T2: align_to_timeframe must not produce two columns named "timestamp"
# ---------------------------------------------------------------------------

def test_align_to_timeframe_does_not_duplicate_the_timestamp_column():
    frame = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060, 1_700_000_120],
        "regime": ["a", "b", "c"],
    })
    out = align_to_timeframe(frame, "5m")
    assert list(out.columns).count("timestamp") == 1
    assert isinstance(out["timestamp"], pd.Series)
    assert len(out) >= 1


def test_align_to_timeframe_first_is_causal():
    """The value painted on a bucket must be the one known at its open."""
    frame = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060, 1_700_000_120],
        "regime": ["first", "second", "third"],
    })
    out = align_to_timeframe(frame, "5m", how="first")
    first_bucket = out[out["timestamp"] <= 1_700_000_060]
    assert list(first_bucket["regime"]) == ["first"]


# ---------------------------------------------------------------------------
# T4: VWAP must re-anchor per session
# ---------------------------------------------------------------------------

def test_vwap_reanchors_at_the_session_boundary():
    """Two sessions of hourly bars: the second session's VWAP must be its
    own cumulative VWAP, not a continuation of the first day's."""
    start = 1_700_000_000  # 2023-11-14 22:13:20 UTC
    stamps = [start + i * 3600 for i in range(48)]
    close = [100.0 + i for i in range(48)]
    bars = pd.DataFrame({
        "timestamp": stamps,
        "open": close, "high": close, "low": close, "close": close,
        "volume": [10.0] * 48,
    })
    out = compute_indicators(bars)
    days = pd.to_datetime(out["timestamp"], unit="s", utc=True).dt.floor("D")
    second_day = out[days == days.iloc[-1]]
    typical = second_day["close"].iloc[0]
    expected = typical  # a single bar in the session -> VWAP == its own price
    assert second_day["vwap"].iloc[0] == pytest.approx(expected)
    # And it must NOT equal the all-history VWAP.
    all_history_vwap = out["vwap"].iloc[-1]
    assert second_day["vwap"].iloc[0] != pytest.approx(all_history_vwap)


# ---------------------------------------------------------------------------
# T5 / T6: the terminal-stage classifier must tell the truth
# ---------------------------------------------------------------------------

def _opportunity(tradable=True, failed=None) -> OpportunityScore:
    return OpportunityScore(
        timestamp=1, symbol="BTC/USDT", tradable=tradable, tradable_probability=0.5,
        environment_quality=0.5, opportunity_strength=0.5, failed_rules=failed or [],
    )


def _router(selected="trend_expert", reasoning=None) -> RouterDecision:
    return RouterDecision(
        timestamp=1, symbol="BTC/USDT", selected_expert=selected,
        selection_confidence=0.6, no_trade_probability=0.4,
        expert_ranking={}, reasoning=reasoning,
    )


def _risk(approved=False, reason="no_trade_selected") -> RiskDecision:
    return RiskDecision(
        timestamp=1, symbol="BTC/USDT", approved=approved, risk_score=0.0,
        base_size=0.0, adjusted_size=0.0, size_reduction_pct=100.0,
        rejection_reason=None if approved else reason,
    )


def test_a_bar_during_an_open_position_is_not_reported_as_no_expert_candidate():
    """The router picked an expert; the bar simply could not open a second
    position. Calling that 'no expert candidate' is plainly false."""
    stage, _ = classify_terminal_stage(
        _opportunity(True), _router("trend_expert"), None, None, position_open=True,
    )
    assert stage == "position_open"


def test_a_tripped_circuit_breaker_is_visible_even_when_the_router_declined():
    stage, reason = classify_terminal_stage(
        _opportunity(True), _router("no_trade", "router_model_no_trade"),
        _risk(False, "circuit_breaker_active:max_drawdown"), None,
    )
    assert stage == "circuit_breaker"
    assert "max_drawdown" in reason


def test_the_manual_kill_switch_is_visible():
    stage, reason = classify_terminal_stage(
        _opportunity(True), _router("no_trade", "router_model_no_trade"),
        _risk(False, "manual_kill_switch_active"), None,
    )
    assert stage == "circuit_breaker"
    assert "kill_switch" in reason


def test_an_expert_with_no_candidate_is_not_reported_as_a_risk_rejection():
    stage, _ = classify_terminal_stage(
        _opportunity(True), _router("trend_expert"),
        _risk(False, "selected_expert_has_no_candidate"), None,
    )
    assert stage == "no_expert_candidate"


# ---------------------------------------------------------------------------
# T7 / T11: a trade is only a trade if the order filled
# ---------------------------------------------------------------------------

def _state_with_execution(status: str) -> SystemState:
    execution = ExecutionRecord(
        timestamp=1, order_timestamp=1, fill_timestamp=1, symbol="BTC/USDT",
        side="BUY", order_price=100.0, fill_price=100.0, requested_size=100.0,
        filled_size=100.0, slippage_pct=0.0, fees_pct=0.04, total_cost_pct=0.04,
        status=status, order_id="x",
    )
    return SystemState(
        timestamp=1, symbol="BTC/USDT", market_snapshot=None,
        router_decision=_router(), execution_record=execution,
    )


def test_a_rejected_order_is_not_reported_as_a_long():
    record = decision_record_from_state(_state_with_execution("rejected"))
    assert record["decision"] == "NO_TRADE"


def test_a_pending_order_is_not_reported_as_a_long():
    record = decision_record_from_state(_state_with_execution("pending"))
    assert record["decision"] == "NO_TRADE"


def test_a_filled_order_is_reported_as_a_long():
    record = decision_record_from_state(_state_with_execution("filled"))
    assert record["decision"] == "LONG"


def test_a_partial_fill_counts_as_a_trade():
    record = decision_record_from_state(_state_with_execution("partial"))
    assert record["decision"] == "LONG"


def test_a_failed_execution_is_classified_as_execution_failed():
    stage, reason = classify_terminal_stage(
        _opportunity(True), _router(), _risk(True),
        _state_with_execution("rejected").execution_record,
    )
    assert stage == "execution_failed"
    assert "rejected" in reason


# ---------------------------------------------------------------------------
# T12 / T13 / T14: the readers are read-only and correctly typed
# ---------------------------------------------------------------------------

def test_reading_the_journal_does_not_modify_the_database(tmp_path):
    db = tmp_path / "registry.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trade_journal (trade_id TEXT PRIMARY KEY, entry_timestamp INTEGER, "
        "entry_price REAL, exit_timestamp INTEGER, exit_price REAL, exit_reason TEXT, "
        "expert_used TEXT, regime_at_entry TEXT, direction INTEGER, size REAL, "
        "realized_pnl_pct REAL, realized_pnl_usdt REAL, fees_paid REAL, "
        "slippage_realized REAL, router_confidence REAL, risk_score_at_entry REAL, mode TEXT)"
    )
    conn.commit()
    conn.close()
    before = db.read_bytes()
    read_trade_journal(str(db))
    assert db.read_bytes() == before


def test_trade_columns_are_numeric_not_object(tmp_path):
    db = tmp_path / "registry.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trade_journal (trade_id TEXT PRIMARY KEY, entry_timestamp INTEGER, "
        "entry_price REAL, exit_timestamp INTEGER, exit_price REAL, exit_reason TEXT, "
        "expert_used TEXT, regime_at_entry TEXT, direction INTEGER, size REAL, "
        "realized_pnl_pct REAL, realized_pnl_usdt REAL, fees_paid REAL, "
        "slippage_realized REAL, router_confidence REAL, risk_score_at_entry REAL, mode TEXT)"
    )
    conn.execute("INSERT INTO trade_journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("t1", 100, 1.0, 200, 2.0, "take_profit", "e", "r", 1, 10.0, 1.0, 1.0,
                  0.1, 0.0, 0.5, 0.2, "paper"))
    conn.commit()
    conn.close()
    frame = read_trade_journal(str(db))
    assert frame["pnl_pct"].dtype.kind == "f"
    assert frame["exit_timestamp"].dtype.kind == "f"
    assert frame["entry_timestamp"].dtype.kind == "i"


def test_resample_of_an_empty_frame_keeps_the_schema():
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = resample_ohlcv(empty, "5m")
    assert out.empty
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_decision_log_reader_keeps_the_terminal_stage(tmp_path):
    """The READER must keep the field.

    The previous version of this test built the frame with pandas and
    asserted on it, so it passed even when the reader dropped the column
    entirely.
    """
    from src.gui.artifacts import read_decisions_jsonl

    path = tmp_path / "decisions.jsonl"
    path.write_text(json.dumps({
        "symbol": "BTC/USDT", "timestamp": 1700000000, "regime": "chop_range",
        "expert": "no_trade", "decision": "NO_TRADE",
        "terminal_stage": "position_open", "terminal_reason": "position already open",
    }) + "\n", encoding="utf-8")
    frame = read_decisions_jsonl(str(path))
    assert len(frame) == 1
    assert frame.iloc[0]["terminal_stage"] == "position_open"
    assert frame.iloc[0]["terminal_reason"] == "position already open"
