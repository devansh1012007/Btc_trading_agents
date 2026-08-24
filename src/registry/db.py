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
from typing import Optional

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


def log_experiment(
    run_type: str,
    config: dict,
    metrics: Optional[dict] = None,
    component: Optional[str] = None,
    git_commit: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Log a backtest or training run to the experiments table."""
    import hashlib
    from datetime import datetime, timezone

    # Compute config hash
    config_str = yaml.dump(config, default_flow_style=False)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # Truncate config snapshot (store first 4000 chars)
    config_snapshot = config_str[:4000]

    experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(config_hash.encode()).hexdigest()[:8]}"
    metrics_json = json.dumps(metrics) if metrics else None

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, run_type, started_at, config_hash, "
            "config_snapshot, component, metrics_json, git_commit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, run_type, datetime.now(timezone.utc).isoformat(),
             config_hash, config_snapshot, component, metrics_json, git_commit),
        )

    return experiment_id
