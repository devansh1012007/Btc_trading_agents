"""
Shared SQLite connection + schema for everything that needs transactional
read-modify-write state (Phase 4, §7.4): circuit breaker counters, the
trade journal, the model registry, and idempotent order tracking. One
file (data/registry.db), one schema, one place every other Phase 4 module
gets a connection from -- avoids four modules each hand-rolling their own
sqlite3.connect() and CREATE TABLE IF NOT EXISTS.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = "data/registry.db"

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
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH):
    """Context-managed connection: commits on clean exit, always closes."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
