"""
Trade Journal (Phase 4, §7.2): a separate, human-readable artifact from
the dense per-bar SystemState archive -- this is what you'd actually
scroll through to sanity-check the bot, not the raw Parquet firehose.
Backed by SQLite (data/registry.db), same shared connection as circuit
breakers and the model registry.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict

from ..registry.db import connect, DEFAULT_DB_PATH


def record_trade_entry(
    entry_timestamp: int, entry_price: float, expert_used: str, regime_at_entry: str,
    direction: int, size: float, router_confidence: float, risk_score_at_entry: float,
    mode: str, db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Insert a new open trade, return its trade_id for the later exit update."""
    trade_id = str(uuid.uuid4())
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO trade_journal (trade_id, entry_timestamp, entry_price, expert_used, "
            "regime_at_entry, direction, size, router_confidence, risk_score_at_entry, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_id, entry_timestamp, entry_price, expert_used, regime_at_entry, direction,
             size, router_confidence, risk_score_at_entry, mode),
        )
    return trade_id


def record_trade_exit(
    trade_id: str, exit_timestamp: int, exit_price: float, exit_reason: str,
    realized_pnl_pct: float, realized_pnl_usdt: float, fees_paid: float, slippage_realized: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE trade_journal SET exit_timestamp=?, exit_price=?, exit_reason=?, "
            "realized_pnl_pct=?, realized_pnl_usdt=?, fees_paid=?, slippage_realized=? WHERE trade_id=?",
            (exit_timestamp, exit_price, exit_reason, realized_pnl_pct, realized_pnl_usdt,
             fees_paid, slippage_realized, trade_id),
        )


def get_recent_trades(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM trade_journal ORDER BY entry_timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_open_trades(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM trade_journal WHERE exit_timestamp IS NULL"
        ).fetchall()
        return [dict(r) for r in rows]


def summarize(db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Today's trade count and PnL -- what scripts/status.py surfaces."""
    today_start = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM trade_journal WHERE entry_timestamp >= ? AND exit_timestamp IS NOT NULL",
            (today_start,),
        ).fetchall()
    trades = [dict(r) for r in rows]
    total_pnl_usdt = sum(t["realized_pnl_usdt"] or 0.0 for t in trades)
    wins = sum(1 for t in trades if (t["realized_pnl_pct"] or 0) > 0)
    return {
        "trades_today": len(trades),
        "total_pnl_usdt_today": total_pnl_usdt,
        "win_rate_today": wins / len(trades) if trades else None,
    }
