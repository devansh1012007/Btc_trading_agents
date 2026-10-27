"""A hand-built SessionBundle with KNOWN timestamps, for overlay / chart tests.

No pipeline, no replay: every timestamp is chosen here so a test can say
exactly which candle a marker must land on at every timeframe.
"""
import numpy as np
import pandas as pd

from src.gui.instruments import default_spec
from src.gui.models import (
    DECISION_COLUMNS,
    EQUITY_COLUMNS,
    NUMERIC_TRADE_COLUMNS,
    TRADE_COLUMNS,
    ZONE_COLUMNS,
    SessionBundle,
    coerce_numeric,
    normalize_dtypes,
)

#: A whole hour, so 5m / 15m / 1h buckets all start on a bar boundary.
START = 1_700_006_400
N_BARS = 120


def ts(i: int) -> int:
    """Timestamp of base bar ``i`` (one-minute bars)."""
    return START + i * 60


def make_bundle() -> SessionBundle:
    idx = np.arange(N_BARS)
    close = 100.0 + np.sin(idx / 9.0) * 3 + idx * 0.02
    bars = pd.DataFrame({
        "timestamp": [ts(i) for i in idx],
        "open": close - 0.1, "high": close + 0.6, "low": close - 0.6, "close": close,
        "volume": 10.0 + (idx % 7),
    })

    def trade(i_entry, i_exit, direction, reason, pnl_pct, **extra):
        entry = float(close[i_entry])
        row = {
            "trade_id": f"t{i_entry}", "entry_timestamp": ts(i_entry),
            "exit_timestamp": None if i_exit is None else ts(i_exit),
            "entry_price": entry,
            "exit_price": None if i_exit is None else entry * (1 + direction * pnl_pct / 100),
            "direction": direction, "expert": "trend_expert", "regime": "trending_up", "size": 500.0,
            "pnl_pct": None if i_exit is None else pnl_pct,
            "pnl_usdt": None if i_exit is None else 500.0 * pnl_pct / 100,
            "fees_paid": 0.2, "slippage": 0.0, "exit_reason": reason,
            "router_confidence": 0.7, "risk_score": 0.2, "mode": "paper_sim",
            "take_profit_pct": 1.5, "stop_loss_pct": 1.0,
            "bars_held": None if i_exit is None else i_exit - i_entry,
        }
        row.update(extra)
        return row

    trades = pd.DataFrame([
        trade(7, 22, 1, "take_profit", 1.5),
        trade(40, 50, -1, "stop_loss", -1.0),
        trade(100, None, 1, "open", 0.0),
    ])

    entry_bars = {7, 40, 100}
    decisions = pd.DataFrame([{
        "timestamp": ts(i), "symbol": "BTC/USDT",
        "regime": "trending_up" if i < 60 else "chop_range", "regime_confidence": 0.8,
        "expert": "trend_expert", "opportunity_score": 0.6, "router_confidence": 0.7,
        "risk_score": 0.2, "risk_approved": i in entry_bars,
        "decision": "LONG" if i in entry_bars else "NO_TRADE",
        "terminal_stage": "executed" if i in entry_bars
        else ("position_open" if 7 < i <= 22 else "opportunity_filter"),
        "terminal_reason": "filled via trend_expert" if i in entry_bars else "spread",
    } for i in idx])

    zones = pd.DataFrame([
        {"zone_id": "z1", "event_type": "created", "timestamp": ts(3), "bar_index": 3,
         "price_at_event": 101.0, "zone_strength_at_event": 0.6, "related_zone_id": None,
         "regime_at_event": "trending_up"},
        {"zone_id": "z1", "event_type": "broke", "timestamp": ts(30), "bar_index": 30,
         "price_at_event": 101.0, "zone_strength_at_event": 0.4, "related_zone_id": None,
         "regime_at_event": "trending_up"},
        {"zone_id": "z2", "event_type": "created", "timestamp": ts(10), "bar_index": 10,
         "price_at_event": 99.0, "zone_strength_at_event": 0.7, "related_zone_id": None,
         "regime_at_event": "trending_up"},
    ])

    equity_values = np.full(N_BARS, 10_000.0)
    equity_values[22:] += 7.5
    equity_values[50:] -= 5.0
    equity = pd.DataFrame({"timestamp": [ts(i) for i in idx], "equity": equity_values})

    traces = {
        ts(i): {"timestamp": ts(i), "symbol": "BTC/USDT", "terminal_stage": row.terminal_stage,
                "terminal_reason": row.terminal_reason, "stages": [
                    {"stage_name": "expert_layer", "passed": True, "detail": "1 candidate(s)",
                     "values": {}, "duration_ms": 0.1}],
                "total_duration_ms": 1.0, "stage_timings_ms": {}}
        for i, row in zip(idx, decisions.itertuples(), strict=True)
    }

    return SessionBundle(
        symbol="BTC/USDT", instrument=default_spec("BTC/USDT"), base_timeframe="1m",
        bars=bars,
        decisions=normalize_dtypes(decisions[DECISION_COLUMNS]),
        trades=normalize_dtypes(coerce_numeric(trades[TRADE_COLUMNS], NUMERIC_TRADE_COLUMNS)),
        zones=normalize_dtypes(zones[ZONE_COLUMNS]),
        equity=normalize_dtypes(equity[EQUITY_COLUMNS]),
        traces=traces,
        meta={"source": "test", "source_label": "Hand-built fixture", "synthetic": True,
              "initial_equity": 10_000.0, "warnings": ["ILLUSTRATIVE - fixture"]},
    )
