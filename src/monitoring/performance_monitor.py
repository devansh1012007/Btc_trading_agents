"""
Live-vs-Backtest Expectancy Gap Monitor (Phase 4, §9): the single
strongest drift signal in this system -- compares an expert's/the
router's actual rolling realized performance (from the trade journal)
against what its walk-forward validation predicted. A meaningful,
sustained gap here is worth more than any single feature's PSI score.
"""
from typing import Dict, List, Optional

import numpy as np


def compute_expectancy_gap(
    validation_expectancy_pct: float, live_trades_pnl_pct: List[float],
    gap_alert_threshold_pct: float = 30.0,
) -> Dict:
    """
    live_trades_pnl_pct: realized_pnl_pct for the most recent rolling
    window of trades (monitoring.performance.rolling_window_trades).
    gap_alert_threshold_pct: alert when live expectancy is this much
    WORSE (relatively) than what validation predicted.
    """
    if not live_trades_pnl_pct:
        return {"live_expectancy_pct": None, "gap_pct": None, "alert": False, "n_trades": 0}

    live_expectancy = float(np.mean(live_trades_pnl_pct))

    if validation_expectancy_pct == 0:
        gap_pct = None
        alert = live_expectancy < 0
    else:
        gap_pct = (validation_expectancy_pct - live_expectancy) / abs(validation_expectancy_pct) * 100
        alert = gap_pct >= gap_alert_threshold_pct

    return {
        "live_expectancy_pct": live_expectancy,
        "validation_expectancy_pct": validation_expectancy_pct,
        "gap_pct": gap_pct,
        "alert": alert,
        "n_trades": len(live_trades_pnl_pct),
    }


def expectancy_gap_by_component(
    trades: List[Dict], validation_expectancies: Dict[str, float],
    rolling_window_trades: int = 50, gap_alert_threshold_pct: float = 30.0,
) -> Dict[str, Dict]:
    """
    trades: trade_journal rows (dicts with 'expert_used', 'realized_pnl_pct').
    validation_expectancies: {component: walk-forward-predicted expectancy_pct}.
    Returns {component: compute_expectancy_gap(...) result}.
    """
    by_component: Dict[str, List[float]] = {}
    for t in trades:
        if t.get("realized_pnl_pct") is None:
            continue
        by_component.setdefault(t["expert_used"], []).append(t["realized_pnl_pct"])

    results = {}
    for component, validation_expectancy in validation_expectancies.items():
        recent = by_component.get(component, [])[-rolling_window_trades:]
        results[component] = compute_expectancy_gap(validation_expectancy, recent, gap_alert_threshold_pct)
    return results
