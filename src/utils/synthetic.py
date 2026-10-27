"""
Deterministic mixed-regime synthetic OHLCV (trend / chop / trend / panic).

One generator for every consumer: the walk-forward backtest harness, the
latency benchmark, the Trade Map GUI's demo mode and the tests. They used
to carry three near-identical copies; the only thing they legitimately
differ in is the START TIMESTAMP, so that is the one parameter the caller
supplies. A fixed start gives a reproducible series (the GUI demo, its
tests and its screenshots all agree); ``datetime.now()`` gives the
harness's historical "bars ending around now" behaviour.
"""
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

#: Default epoch for the synthetic demo series (2026-01-01T00:00:00Z).
#: Fixed, not "now": a deterministic series is what lets the GUI's demo
#: mode, its tests and its screenshots all agree.
SYNTHETIC_START = datetime(2026, 1, 1, tzinfo=timezone.utc)

BAR_SECONDS = 60


def synthetic_ohlcv(
    n_bars: int = 900,
    seed: int = 42,
    start: Optional[datetime] = None,
    bar_seconds: int = BAR_SECONDS,
) -> pd.DataFrame:
    """Mixed-regime synthetic BTC/USDT bars (one-minute by default).

    Args:
        n_bars: number of one-minute bars.
        seed: RNG seed; the same ``(n_bars, seed, start)`` always yields
            the same frame.
        start: timestamp of the first bar (defaults to
            :data:`SYNTHETIC_START`).
        bar_seconds: spacing between bars (default 60 s). The minimum-trade-
            duration tests generate 5 s / 10 s bars with it.

    ``timestamp`` is integer epoch seconds.
    """
    rng = np.random.default_rng(seed)
    begin = (start or SYNTHETIC_START).timestamp()
    price, rows = 50000.0, []
    segment = max(1, n_bars // 4)
    for i in range(n_bars):
        drift = 12.0 if i < segment else (-10.0 if i < 2 * segment else (2.0 if i < 3 * segment else -18.0))
        vol = 8.0 if i >= 3 * segment else 5.0
        price = max(price + rng.normal(drift, vol), 1000.0)
        open_ = price + rng.normal(0, 3)
        rows.append({
            "timestamp": int(begin + i * bar_seconds),
            "open": open_,
            "high": max(open_, price) + abs(rng.normal(0, vol * 0.8)),
            "low": min(open_, price) - abs(rng.normal(0, vol * 0.8)),
            "close": price,
            "volume": abs(rng.normal(120, 35)),
        })
    return pd.DataFrame(rows)
