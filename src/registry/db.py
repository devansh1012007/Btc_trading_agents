"""
Shared SQLite connection + schema for everything that needs transactional
read-modify-write state (Phase 4, §7.4): circuit breaker counters, the
trade journal, the model registry, and idempotent order tracking. One
file (data/registry.db), one schema, one place every other Phase 4 module
gets a connection from -- avoids four modules each hand-rolling their own
sqlite3.connect() and CREATE TABLE IF NOT EXISTS.
"""
import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


class _Pool(threading.local):
    """Per-thread connection pool, keyed by resolved database path.

    A connection is bound to the thread that opened it (``sqlite3``'s own
    default), so the pool is thread-local rather than guarded by a lock.
    Each entry remembers the file identity (device, inode) it was opened
    against: a database file that was deleted and re-created -- a test's
    temp directory, a manual reset -- is detected and re-opened instead
    of serving a handle to a dead file.
    """

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
    """Open a NEW, un-pooled connection with the schema applied.

    The schema DDL runs once per connection, here -- not on every
    ``connect()`` call, which used to be the replay's dominant cost.
    ``wal=True`` switches the database to WAL + ``synchronous=NORMAL``:
    a read-modify-write per bar then costs a page append instead of a
    journal create / fsync / delete.
    """
    if db_path != MEMORY_PATH:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if wal and db_path != MEMORY_PATH:
        _enable_wal(conn, db_path)
    conn.executescript(SCHEMA)
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
    """The pooled connection for ``db_path`` (one per thread per file).

    Callers must NOT close it -- use :func:`close_connection` /
    :func:`close_all_connections`. A closed or replaced file is
    re-opened transparently.
    """
    if db_path == MEMORY_PATH:
        return open_connection(db_path)

    key = str(Path(db_path).resolve())
    identity = _file_identity(key)
    entry = _POOL.entries.get(key)
    if entry is not None:
        conn, opened_identity, has_wal = entry
        if identity == opened_identity and identity is not None and _is_alive(conn):
            if wal and not has_wal:
                # Another store opened the file first; upgrade it once.
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
    """Context-managed pooled connection.

    Commits on a clean exit and rolls back on an exception. The
    connection stays open for the next caller; a ``:memory:`` path gets a
    private connection that is closed on exit.
    """
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
