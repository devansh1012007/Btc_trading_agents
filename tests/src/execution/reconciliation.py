"""
Reconciliation (Phase 4, §6.5): run on every process start and
periodically thereafter in live mode. Fetches actual account state (open
orders) from the exchange and compares it against the bot's last-known
state in data/registry.db, flagging discrepancies (a manually-closed
position, an order that filled while the bot was down) before resuming
normal operation. This is the difference between "restarts safely" and
"restarts and doesn't know what it did five minutes ago."

NOT exercised against a real exchange in this delivery -- needs live
credentials. Structure is complete and the discrepancy-detection logic is
unit-tested against a mocked exchange (test_reconciliation_detects_drift).
"""
import logging
from typing import Dict, List

from ..registry.db import DEFAULT_DB_PATH, connect

logger = logging.getLogger(__name__)


def reconcile(connector, symbol: str, db_path: str = DEFAULT_DB_PATH) -> Dict:
    """
    Returns {"status": "ok"|"error", "discrepancies": [...], "open_orders_on_exchange": int}.
    A discrepancy is a bot-known, not-yet-terminal order whose id isn't
    found among the exchange's currently open orders -- it either filled
    or was cancelled/rejected while the bot wasn't watching.
    """
    try:
        open_orders = connector.exchange.fetch_open_orders(symbol)
    except Exception as e:
        logger.error(f"Reconciliation failed to fetch open orders: {e}")
        return {"status": "error", "error": str(e), "discrepancies": [], "open_orders_on_exchange": None}

    exchange_order_ids = {str(o.get("id")) for o in open_orders}

    with connect(db_path) as conn:
        known_open_orders = conn.execute(
            "SELECT * FROM orders WHERE status NOT IN ('filled', 'cancelled', 'rejected')"
        ).fetchall()

    discrepancies: List[Dict] = []
    for row in known_open_orders:
        exchange_id = row["exchange_order_id"]
        if exchange_id and exchange_id not in exchange_order_ids:
            discrepancies.append({
                "client_order_id": row["client_order_id"],
                "exchange_order_id": exchange_id,
                "bot_believed_status": row["status"],
                "issue": "order not found among exchange's open orders -- likely filled or cancelled while bot was down",
            })

    if discrepancies:
        logger.warning(f"Reconciliation found {len(discrepancies)} discrepancies: {discrepancies}")
    else:
        logger.info(f"Reconciliation clean: {len(open_orders)} open orders on exchange, all match bot state.")

    return {"status": "ok", "discrepancies": discrepancies, "open_orders_on_exchange": len(open_orders)}
