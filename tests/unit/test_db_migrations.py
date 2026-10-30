"""
Unit tests for versioned SQLite migrations (Phase 1, T1.4).
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.registry.db import (
    LATEST_SCHEMA_VERSION,
    SCHEMA,
    close_connection,
    connect,
    open_connection,
)

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parent.parent.parent


def test_migration_upgrades_legacy_schema_preserving_data_and_creating_backup(tmp_path) -> None:
    db_file = tmp_path / "legacy_registry.db"
    # Create a legacy v0 DB using only the original SCHEMA (user_version = 0)
    raw = sqlite3.connect(str(db_file))
    raw.executescript(SCHEMA)
    raw.execute(
        "INSERT INTO trade_journal (trade_id, entry_timestamp, entry_price, expert_used, mode) "
        "VALUES ('t-legacy', 1700000000, 50000.0, 'trend_expert', 'paper')"
    )
    raw.commit()
    raw.close()

    with connect(str(db_file)) as conn:
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver == LATEST_SCHEMA_VERSION
        row = conn.execute("SELECT * FROM trade_journal WHERE trade_id='t-legacy'").fetchone()
        assert row is not None
        assert row["entry_price"] == 50000.0
        # New columns exist and default to None
        assert "exit_source" in row.keys()
        assert "hold_seconds" in row.keys()
        assert "regime_id" in row.keys()
        assert "expert_instance" in row.keys()
        assert "signal_id" in row.keys()

    close_connection(str(db_file))
    assert (tmp_path / "legacy_registry.db.bak.0").exists()


def test_migrated_head_db_equals_fresh_db_schema(tmp_path) -> None:
    head_copy = tmp_path / "head_registry.db"
    shutil.copy2(ROOT / "data" / "registry.db", head_copy)

    fresh_db = tmp_path / "fresh_registry.db"

    with connect(str(head_copy)) as c_head, connect(str(fresh_db)) as c_fresh:
        for tbl in ("trade_journal", "model_versions", "bundles", "model_predictions", "model_outcomes"):
            cols_head = [r[1] for r in c_head.execute(f"PRAGMA table_info({tbl})").fetchall()]
            cols_fresh = [r[1] for r in c_fresh.execute(f"PRAGMA table_info({tbl})").fetchall()]
            assert cols_head == cols_fresh, f"Column mismatch on {tbl}: {cols_head} != {cols_fresh}"
        # Idempotent re-run
        c_head_2 = open_connection(str(head_copy))
        assert c_head_2.execute("PRAGMA user_version").fetchone()[0] == LATEST_SCHEMA_VERSION
        c_head_2.close()
