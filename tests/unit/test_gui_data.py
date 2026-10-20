"""
Trade Map GUI data readers: the trade journal, the §8 decision log, the
decision traces and OHLCV files.

Every reader must degrade to an empty (but correctly shaped) result
rather than raise -- the GUI is pointed at whatever a run happened to
leave on disk, and a missing file is the normal case on a fresh
checkout.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.gui.artifacts import (  # noqa: E402
    find_ohlcv_files,
    load_ohlcv,
    load_symbol_ohlcv,
    read_decision_traces,
    read_decisions_jsonl,
    read_trade_journal,
    read_zone_events,
)
from src.gui.models import DECISION_COLUMNS, TRADE_COLUMNS  # noqa: E402

pytestmark = [pytest.mark.unit]


def make_journal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trade_journal ("
        "trade_id TEXT PRIMARY KEY, entry_timestamp INTEGER, entry_price REAL, "
        "exit_timestamp INTEGER, exit_price REAL, exit_reason TEXT, expert_used TEXT, "
        "regime_at_entry TEXT, direction INTEGER, size REAL, realized_pnl_pct REAL, "
        "realized_pnl_usdt REAL, fees_paid REAL, slippage_realized REAL, "
        "router_confidence REAL, risk_score_at_entry REAL, mode TEXT)"
    )
    conn.execute(
        "INSERT INTO trade_journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("t1", 1_700_000_000, 100.0, 1_700_000_060, 101.5, "take_profit",
         "trend_expert", "trending_up", 1, 250.0, 1.35, 3.4, 0.12, 0.01, 0.7, 0.2, "paper"),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# trade journal
# ---------------------------------------------------------------------------

def test_read_trade_journal_normalises_columns(tmp_path):
    db = tmp_path / "registry.db"
    make_journal_db(db)
    frame = read_trade_journal(str(db))
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["trade_id"] == "t1"
    assert row["expert"] == "trend_expert"
    assert row["regime"] == "trending_up"
    assert row["pnl_pct"] == 1.35
    assert row["pnl_usdt"] == 3.4
    assert row["slippage"] == 0.01
    assert row["risk_score"] == 0.2
    assert list(frame.columns) == TRADE_COLUMNS


def test_read_trade_journal_on_missing_database_is_empty(tmp_path):
    frame = read_trade_journal(str(tmp_path / "absent.db"))
    assert frame.empty
    assert list(frame.columns) == TRADE_COLUMNS


def test_read_trade_journal_on_database_without_the_table(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    frame = read_trade_journal(str(db))
    assert frame.empty


def test_read_trade_journal_does_not_create_the_database(tmp_path):
    db = tmp_path / "must_not_appear.db"
    read_trade_journal(str(db))
    assert not db.exists()


# ---------------------------------------------------------------------------
# decisions.jsonl
# ---------------------------------------------------------------------------

def test_read_decisions_jsonl_normalises(tmp_path):
    path = tmp_path / "decisions.jsonl"
    records = [
        {"symbol": "BTC/USDT", "timestamp": 2, "regime": "chop_range", "expert": "no_trade",
         "opportunity_score": 0.2, "router_confidence": 0.8, "risk_score": 0.1,
         "decision": "NO_TRADE", "equity": None, "terminal_stage": "opportunity_filter",
         "terminal_reason": "edge_below_threshold"},
        {"symbol": "BTC/USDT", "timestamp": 1, "regime": "trending_up",
         "expert": "trend_expert", "decision": "LONG"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    frame = read_decisions_jsonl(str(path))
    assert len(frame) == 2
    assert list(frame["timestamp"]) == [1, 2]  # sorted
    assert list(frame.columns) == DECISION_COLUMNS
    assert frame.iloc[1]["terminal_stage"] == "opportunity_filter"


def test_read_decisions_jsonl_skips_torn_tail_line(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"timestamp": 1, "symbol": "X"}\n{"timestamp": 2, "sym', encoding="utf-8")
    frame = read_decisions_jsonl(str(path))
    assert len(frame) == 1


def test_read_decisions_jsonl_missing_file_is_empty(tmp_path):
    frame = read_decisions_jsonl(str(tmp_path / "nope.jsonl"))
    assert frame.empty
    assert list(frame.columns) == DECISION_COLUMNS


# ---------------------------------------------------------------------------
# traces / zones
# ---------------------------------------------------------------------------

def test_read_decision_traces_keys_by_timestamp(tmp_path):
    d = tmp_path / "decision_traces"
    d.mkdir()
    (d / "traces_2026-01-01.jsonl").write_text(
        json.dumps({"timestamp": 42, "symbol": "BTC/USDT", "stages": [],
                    "terminal_stage": "executed", "terminal_reason": "x"}) + "\n",
        encoding="utf-8",
    )
    traces = read_decision_traces(str(d))
    assert 42 in traces
    assert traces[42]["terminal_stage"] == "executed"


def test_read_zone_events_missing_file_is_empty(tmp_path):
    frame = read_zone_events(str(tmp_path / "zone_events.parquet"))
    assert frame.empty


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def test_load_ohlcv_from_csv_with_epoch_seconds(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700000000,100,101,99,100.5,10\n"
        "1700000060,100.5,102,100,101.5,11\n",
        encoding="utf-8",
    )
    frame = load_ohlcv(str(path))
    assert list(frame["timestamp"]) == [1_700_000_000, 1_700_000_060]
    assert frame.iloc[0]["close"] == 100.5


def test_load_ohlcv_detects_millisecond_timestamps(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700000000000,100,101,99,100.5,10\n",
        encoding="utf-8",
    )
    frame = load_ohlcv(str(path))
    assert frame.iloc[0]["timestamp"] == 1_700_000_000


def test_load_ohlcv_accepts_alias_columns(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("ts,o,h,l,c,v\n1700000000,100,101,99,100.5,10\n", encoding="utf-8")
    frame = load_ohlcv(str(path))
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == 100.5


def test_load_ohlcv_missing_file_is_empty(tmp_path):
    frame = load_ohlcv(str(tmp_path / "nope.csv"))
    assert frame.empty


def test_find_ohlcv_files_and_merge(tmp_path):
    (tmp_path / "BTC_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "ETH_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    found = find_ohlcv_files("BTC/USDT", str(tmp_path))
    assert len(found) == 1
    assert "BTC_USDT" in found[0].name
    assert load_symbol_ohlcv("BTC/USDT", str(tmp_path)).empty  # unreadable parquet -> empty
