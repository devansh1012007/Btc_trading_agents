"""
Phase 1.5 hardening: the pooled SQLite layer, the in-memory risk store,
the single per-bar implementation and the remaining quality findings.

Each test pins one finding from the independent Phase-1 assessment so
that it cannot quietly come back.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.gui.artifacts import read_trade_journal, resolve_artifact_path
from src.gui.models import (
    BAR_COLUMNS,
    ZONE_COLUMNS,
    SessionBundle,
    cast_int_columns,
    empty_frame,
    normalize_dtypes,
)
from src.layers.risk_layer import RiskStateStore
from src.registry import db as registry_db

ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# 1.5.2 -- the pooled registry connection
# ---------------------------------------------------------------------------

def test_connect_reuses_one_connection_per_file(tmp_path):
    path = str(tmp_path / "registry.db")
    with registry_db.connect(path) as first:
        pass
    with registry_db.connect(path) as second:
        pass
    assert first is second


def test_the_schema_runs_once_per_connection_not_once_per_call(tmp_path, monkeypatch):
    path = str(tmp_path / "registry.db")
    calls = []
    real = registry_db.open_connection

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(registry_db, "open_connection", counting)
    for _ in range(25):
        with registry_db.connect(path):
            pass
    assert len(calls) == 1


def test_a_deleted_and_recreated_database_is_not_served_from_a_stale_handle(tmp_path):
    path = tmp_path / "registry.db"
    with registry_db.connect(str(path)) as conn:
        conn.execute("INSERT INTO risk_state (key, value, updated_at) VALUES ('k', '1', 'now')")
    path.unlink()
    with registry_db.connect(str(path)) as conn:
        rows = conn.execute("SELECT * FROM risk_state").fetchall()
    assert rows == []


def test_a_closed_pooled_connection_is_reopened(tmp_path):
    path = str(tmp_path / "registry.db")
    conn = registry_db.get_connection(path)
    conn.close()
    again = registry_db.get_connection(path)
    assert again is not conn
    again.execute("SELECT 1")


def test_connect_rolls_back_on_an_exception(tmp_path):
    path = str(tmp_path / "registry.db")
    with pytest.raises(RuntimeError):
        with registry_db.connect(path) as conn:
            conn.execute("INSERT INTO risk_state (key, value, updated_at) VALUES ('k', '1', 'now')")
            raise RuntimeError("boom")
    with registry_db.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM risk_state").fetchone()[0] == 0


def test_memory_connections_are_never_shared():
    a = registry_db.get_connection(registry_db.MEMORY_PATH)
    b = registry_db.get_connection(registry_db.MEMORY_PATH)
    assert a is not b
    a.close()
    b.close()


def test_the_risk_store_uses_wal(tmp_path):
    path = str(tmp_path / "registry.db")
    store = RiskStateStore(db_path=path)
    store.heartbeat()
    with registry_db.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_the_risk_store_persists_across_instances(tmp_path):
    """The breaker must survive a process restart -- semantics unchanged."""
    path = str(tmp_path / "registry.db")
    RiskStateStore(db_path=path).trip("max_drawdown", "test")
    registry_db.close_all_connections()          # a restart
    assert RiskStateStore(db_path=path).is_tripped("max_drawdown")


# ---------------------------------------------------------------------------
# 1.5.2 -- the non-persistent store behaves exactly like the persistent one
# ---------------------------------------------------------------------------

def _drive(store: RiskStateStore) -> dict:
    store.heartbeat()
    store.record_trade_result(-1.0, -10.0)
    store.record_trade_result(-0.5, -5.0)
    store.record_trade_result(0.4, 4.0)
    store.record_execution_result(success=False)
    store.record_execution_result(success=False)
    drawdown = store.update_equity(10_000.0)
    drawdown = store.update_equity(9_000.0)
    store.trip("max_drawdown", "dd")
    return {
        "consecutive_losses": store.consecutive_losses(),
        "daily_loss_pct": store.daily_loss_pct(),
        "execution_errors": store.consecutive_execution_errors(),
        "drawdown": drawdown,
        "current_drawdown": store.current_drawdown_pct(),
        "tripped": store.is_tripped("max_drawdown"),
        "halts": [h["breaker_type"] for h in store.active_halts()],
        "stale": store.is_stale(3600),
    }


def test_the_in_memory_store_matches_the_persistent_store(tmp_path):
    persistent = _drive(RiskStateStore(db_path=str(tmp_path / "registry.db")))
    in_memory = _drive(RiskStateStore(persistent=False))
    assert in_memory == persistent


def test_the_in_memory_store_touches_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _drive(RiskStateStore(persistent=False))
    assert list(tmp_path.iterdir()) == []


def test_two_in_memory_stores_do_not_share_state():
    a = RiskStateStore(persistent=False)
    b = RiskStateStore(persistent=False)
    a.trip("max_drawdown", "x")
    assert a.is_tripped("max_drawdown")
    assert not b.is_tripped("max_drawdown")


# ---------------------------------------------------------------------------
# 1.5.1 -- the per-bar block exists exactly once
# ---------------------------------------------------------------------------

def _strip_comments(source: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


def test_the_barrier_computation_exists_once_in_the_runtime_path():
    """Barrier sizing lives in ``labeling.triple_barrier.barriers_for_regime``
    (shared with the training scripts); the lifecycle calls it exactly once."""
    lifecycle = (ROOT / "src/execution/position_lifecycle.py").read_text()
    assert len(re.findall(r"barriers_for_regime\(", _strip_comments(lifecycle))) == 1
    assert "compute_barrier_pcts" not in _strip_comments(lifecycle)
    for other in ("scripts/run.py", "src/gui/replay.py"):
        text = _strip_comments((ROOT / other).read_text())
        assert "compute_barrier_pcts" not in text, f"{other} carries its own barrier maths"
        assert "check_position_exit(" not in text, f"{other} carries its own exit step"
        assert "classify_terminal_stage(" not in text, f"{other} classifies bars itself"
        assert "record_trade_result(" not in text, f"{other} updates the breaker itself"


def test_every_loop_calls_process_bar():
    for path in ("scripts/run.py", "src/gui/replay.py"):
        text = (ROOT / path).read_text()
        assert "process_bar(" in text


def test_max_concurrent_positions_is_validated_once():
    from src.execution.position_lifecycle import max_concurrent_positions

    assert max_concurrent_positions({}) == 1
    assert max_concurrent_positions({"max_concurrent_positions": 3}) == 3
    with pytest.raises(ValueError):
        max_concurrent_positions({"max_concurrent_positions": 0})


# ---------------------------------------------------------------------------
# 1.5.3 -- the remaining findings
# ---------------------------------------------------------------------------

def _bundle(n=50):
    from src.gui.instruments import default_spec

    bars = pd.DataFrame({
        "timestamp": np.arange(n, dtype="int64") * 60 + 1_700_000_000,
        "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(n) + 1.0, "volume": 1.0,
    })
    decisions = pd.DataFrame({
        "timestamp": bars["timestamp"], "terminal_stage": "executed",
    })
    return SessionBundle(
        symbol="X", instrument=default_spec("X"), bars=bars, decisions=decisions,
    )


def test_the_close_index_is_built_once_and_cached():
    bundle = _bundle()
    first = bundle.close_by_timestamp
    assert bundle.close_by_timestamp is first
    assert first[1_700_000_000] == 1.0


def test_the_decision_index_is_built_once_and_cached():
    bundle = _bundle()
    assert bundle.decision_by_timestamp is bundle.decision_by_timestamp


def test_replacing_a_frame_invalidates_its_index():
    bundle = _bundle()
    stale = bundle.close_by_timestamp
    bundle.bars = bundle.bars.assign(close=99.0)
    fresh = bundle.close_by_timestamp
    assert fresh is not stale
    assert set(fresh.values()) == {99.0}


def test_an_empty_bundle_has_empty_indexes():
    from src.gui.instruments import default_spec

    bundle = SessionBundle(symbol="X", instrument=default_spec("X"))
    assert bundle.close_by_timestamp == {}
    assert bundle.decision_by_timestamp == {}


def test_normalize_dtypes_never_drops_a_row():
    frame = pd.DataFrame({"timestamp": [1, None, 3], "close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing"):
        normalize_dtypes(frame)
    with pytest.raises(ValueError):
        cast_int_columns(frame, ["timestamp"])


def test_a_zone_event_without_a_bar_index_survives_normalisation():
    frame = pd.DataFrame({
        "zone_id": ["z1", "z2"], "event_type": ["created", "touched"],
        "timestamp": [100, 200], "bar_index": [5, None],
    })
    out = normalize_dtypes(frame)
    assert len(out) == 2
    assert pd.isna(out["bar_index"].iloc[1])
    assert empty_frame(ZONE_COLUMNS)["bar_index"].dtype.kind == "i"


def test_the_readers_decide_the_row_loss(tmp_path, make_journal_db):
    db = make_journal_db(tmp_path / "registry.db", rows=[{"trade_id": "ok"}])
    with registry_db.connect(str(db)) as conn:
        conn.execute("UPDATE trade_journal SET entry_timestamp = NULL WHERE trade_id = 'ok'")
        conn.execute(
            "INSERT INTO trade_journal (trade_id, entry_timestamp) VALUES ('good', 1700000500)"
        )
    registry_db.close_all_connections()
    frame = read_trade_journal(str(db))
    assert list(frame["trade_id"]) == ["good"]


def test_a_locked_or_corrupt_registry_is_logged_not_silent(tmp_path, caplog):
    bad = tmp_path / "registry.db"
    bad.write_bytes(b"this is not a sqlite database" * 20)
    with caplog.at_level("WARNING", logger="src.gui.artifacts"):
        frame = read_trade_journal(str(bad))
    assert frame.empty
    assert any("trade journal" in r.message.lower() or "database" in r.message.lower()
               for r in caplog.records)


def test_artifact_paths_resolve_from_the_repo_root_with_a_cwd_fallback(tmp_path, monkeypatch):
    # Repo root wins, wherever the process was started.
    monkeypatch.chdir(tmp_path)
    assert resolve_artifact_path("config/instruments.yaml") == str(ROOT / "config" / "instruments.yaml")
    # Neither exists -> a stable repo-root path for the 'missing' message.
    assert resolve_artifact_path("data/nope.db") == str(ROOT / "data" / "nope.db")
    # Only the CWD has it -> fall back to the CWD.
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "only_here.db").write_bytes(b"")
    assert resolve_artifact_path("scratch/only_here.db") == str(tmp_path / "scratch" / "only_here.db")
    # Absolute paths are untouched.
    assert resolve_artifact_path(str(tmp_path / "x.db")) == str(tmp_path / "x.db")


def test_the_style_map_is_defined_next_to_the_vocabulary():
    from src.diagnostics import dashboard_data, decision_trace

    assert decision_trace.STAGE_STYLES is dashboard_data.STAGE_STYLES
    assert set(decision_trace.STAGE_STYLES) == set(decision_trace.TERMINAL_STAGES)


def test_the_gui_never_imports_a_tui_module():
    for path in (ROOT / "src" / "gui").glob("*.py"):
        text = path.read_text()
        assert "dashboard_data import STAGE" not in text, path.name


def test_the_synthetic_series_has_one_home():
    from src.gui import replay
    from src.utils import synthetic

    assert replay.synthetic_ohlcv is synthetic.synthetic_ohlcv
    a = synthetic.synthetic_ohlcv(30, seed=1)
    assert list(a.columns) == BAR_COLUMNS
    assert a.equals(synthetic.synthetic_ohlcv(30, seed=1))


def test_the_dead_helpers_are_gone():
    from src.gui import timeframes

    for name in ("is_valid_timeframe", "timeframe_seconds", "format_timeframe", "seconds_to_timeframe"):
        assert not hasattr(timeframes, name), name


def test_sqlite_is_not_imported_by_the_pure_models():
    assert "sqlite3" not in (ROOT / "src/gui/models.py").read_text()
    assert sqlite3  # the stdlib module is what the pool wraps
