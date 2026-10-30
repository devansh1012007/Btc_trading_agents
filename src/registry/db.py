"""
Shared SQLite connection + schema + versioned migrations for everything that
needs transactional read-modify-write state (Phase 4, §7.4; Phase 1, T1.4):
circuit breaker counters, the trade journal, the model registry, model bundles,
the model prediction ledger, and idempotent order tracking.

Migrations use SQLite's `PRAGMA user_version` inside a single transaction,
backing up existing file-backed databases to `<path>.bak.<from_version>`
before applying schema upgrades, so both a fresh database and an existing
HEAD `registry.db` converge to the exact same schema without data loss.
"""
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/registry.db"

#: SQLite's own spelling for a private in-memory database. A connection to
#: it is never pooled: two ``:memory:`` connections are two different
#: databases, so sharing one by path would leak state between callers.
MEMORY_PATH = ":memory:"

SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS circuit_breaker_halts (
    breaker_type TEXT PRIMARY KEY,
    tripped_at TEXT,
    reason TEXT,
    active INTEGER NOT NULL DEFAULT 0,
    reset_at TEXT
);

CREATE TABLE IF NOT EXISTS trade_journal (
    trade_id TEXT PRIMARY KEY,
    entry_timestamp INTEGER,
    entry_price REAL,
    exit_timestamp INTEGER,
    exit_price REAL,
    exit_reason TEXT,
    expert_used TEXT,
    regime_at_entry TEXT,
    direction INTEGER,
    size REAL,
    realized_pnl_pct REAL,
    realized_pnl_usdt REAL,
    fees_paid REAL,
    slippage_realized REAL,
    router_confidence REAL,
    risk_score_at_entry REAL,
    mode TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    data_granularity TEXT,
    file_path TEXT NOT NULL,
    walk_forward_metrics_json TEXT,
    status TEXT NOT NULL,
    promoted_at TEXT,
    promoted_by TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    requested_size REAL,
    status TEXT,
    exchange_order_id TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,          -- "backtest" | "training"
    started_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,       -- content hash of the config used
    config_snapshot TEXT NOT NULL,   -- the actual YAML config (truncated)
    component TEXT,                  -- component name (e.g. "router", "full_stack")
    metrics_json TEXT,               -- JSON of resulting metrics
    git_commit TEXT                  -- git commit if available
);
"""

#: Versioned migrations applied on top of migration 0 (SCHEMA).
#: Each entry is (target_version, list_of_sql_statements).
MIGRATIONS: List[Tuple[int, List[str]]] = [
    (
        1,
        [
            "ALTER TABLE trade_journal ADD COLUMN exit_source TEXT",
            "ALTER TABLE trade_journal ADD COLUMN hold_seconds REAL",
            "ALTER TABLE trade_journal ADD COLUMN regime_id TEXT",
            "ALTER TABLE trade_journal ADD COLUMN expert_instance TEXT",
            "ALTER TABLE trade_journal ADD COLUMN signal_id TEXT",
            "ALTER TABLE trade_journal ADD COLUMN plan_source TEXT",
            "ALTER TABLE trade_journal ADD COLUMN model_versions TEXT",
            "ALTER TABLE trade_journal ADD COLUMN bundle_hash TEXT",
            "ALTER TABLE trade_journal ADD COLUMN router_variant TEXT",
            "ALTER TABLE trade_journal ADD COLUMN governor_flags TEXT",
            "ALTER TABLE trade_journal ADD COLUMN agent_verdict TEXT",
            "ALTER TABLE trade_journal ADD COLUMN order_ts_ms INTEGER",
            "ALTER TABLE trade_journal ADD COLUMN fill_ts_ms INTEGER",
            "ALTER TABLE trade_journal ADD COLUMN exit_decided_ts_ms INTEGER",
            "ALTER TABLE model_versions ADD COLUMN bundle_id TEXT",
            """CREATE TABLE IF NOT EXISTS bundles (
                bundle_id TEXT PRIMARY KEY,
                bundle_hash TEXT NOT NULL,
                regime_set_hash TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                status TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                promoted_at TEXT,
                promoted_by TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS model_predictions (
                id TEXT PRIMARY KEY,
                bar_ts INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                bundle_hash TEXT,
                kind TEXT NOT NULL DEFAULT 'active',
                features_hash TEXT,
                output_json TEXT NOT NULL,
                used_in_decision INTEGER NOT NULL DEFAULT 1,
                signal_id TEXT,
                regime_id TEXT,
                instance TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS model_outcomes (
                prediction_id TEXT PRIMARY KEY,
                outcome_ts INTEGER NOT NULL,
                realized_json TEXT NOT NULL
            )""",
        ],
    ),
]

LATEST_SCHEMA_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else 0


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def apply_migrations(conn: sqlite3.Connection, db_path: str = MEMORY_PATH) -> int:
    """Apply base SCHEMA and any pending versioned migrations inside a transaction."""
    conn.executescript(SCHEMA)
    cur_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if cur_version >= LATEST_SCHEMA_VERSION:
        return cur_version

    # Take a backup before migrating an existing file-backed DB that has tables/rows
    if db_path != MEMORY_PATH and os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        bak_path = f"{db_path}.bak.{cur_version}"
        if not os.path.exists(bak_path):
            try:
                shutil.copy2(db_path, bak_path)
            except OSError as exc:
                logger.warning("Could not write DB backup %s: %s", bak_path, exc)

    try:
        with conn:
            for target_ver, statements in MIGRATIONS:
                if target_ver <= cur_version:
                    continue
                for stmt in statements:
                    if stmt.strip().upper().startswith("ALTER TABLE"):
                        parts = stmt.split()
                        tbl = parts[2]
                        col = parts[5]
                        if col in _table_columns(conn, tbl):
                            continue
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {target_ver}")
                cur_version = target_ver
    except Exception:
        conn.rollback()
        raise
    return cur_version


class _Pool(threading.local):
    """Per-thread connection pool, keyed by resolved database path."""

    def __init__(self) -> None:
        self.entries: dict = {}


_POOL = _Pool()


def _file_identity(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def open_connection(db_path: str = DEFAULT_DB_PATH, wal: bool = False) -> sqlite3.Connection:
    """Open a NEW, un-pooled connection with the schema and migrations applied."""
    if db_path != MEMORY_PATH:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if wal and db_path != MEMORY_PATH:
        _enable_wal(conn, db_path)
    apply_migrations(conn, db_path=db_path)
    return conn


def _enable_wal(conn: sqlite3.Connection, db_path: str) -> None:
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error as exc:  # e.g. a read-only filesystem
        logger.warning("Could not enable WAL on %s: %s", db_path, exc)


def _is_alive(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False


def get_connection(db_path: str = DEFAULT_DB_PATH, wal: bool = False) -> sqlite3.Connection:
    """The pooled connection for ``db_path`` (one per thread per file)."""
    if db_path == MEMORY_PATH:
        return open_connection(db_path)

    key = str(Path(db_path).resolve())
    identity = _file_identity(key)
    entry = _POOL.entries.get(key)
    if entry is not None:
        conn, opened_identity, has_wal = entry
        if identity == opened_identity and identity is not None and _is_alive(conn):
            if wal and not has_wal:
                _enable_wal(conn, key)
                _POOL.entries[key] = (conn, opened_identity, True)
            return conn
        try:
            conn.close()
        except sqlite3.Error:
            pass
        del _POOL.entries[key]

    conn = open_connection(key, wal=wal)
    _POOL.entries[key] = (conn, _file_identity(key), wal)
    return conn


def close_connection(db_path: str = DEFAULT_DB_PATH) -> None:
    """Close this thread's pooled connection for ``db_path`` (if any)."""
    key = str(Path(db_path).resolve())
    entry = _POOL.entries.pop(key, None)
    if entry is not None:
        try:
            entry[0].close()
        except sqlite3.Error:
            pass


def close_all_connections() -> None:
    """Close every pooled connection owned by the calling thread."""
    for key in list(_POOL.entries):
        conn = _POOL.entries.pop(key)[0]
        try:
            conn.close()
        except sqlite3.Error:
            pass


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH, wal: bool = False):
    """Context-managed pooled connection."""
    conn = get_connection(db_path, wal=wal)
    try:
        yield conn
        conn.commit()
    except BaseException:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        if db_path == MEMORY_PATH:
            conn.close()


def log_experiment(
    run_type: str,
    config: dict,
    metrics: Optional[dict] = None,
    component: Optional[str] = None,
    git_commit: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Log a backtest or training run to the experiments table."""
    config_str = yaml.dump(config, default_flow_style=False, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    config_snapshot = config_str[:4000]
    now = datetime.now(timezone.utc)
    experiment_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(config_hash.encode()).hexdigest()[:8]}"
    metrics_json = json.dumps(metrics) if metrics else None

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, run_type, started_at, config_hash, "
            "config_snapshot, component, metrics_json, git_commit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, run_type, now.isoformat(),
             config_hash, config_snapshot, component, metrics_json, git_commit),
        )

    return experiment_id
