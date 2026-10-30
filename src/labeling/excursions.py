"""
Forward excursion labeler (`compute_forward_excursions`, Phase 7 T7.1):
computes maximum favorable excursion (`mfe_pct`), maximum adverse excursion
(`mae_pct`), and lockout adverse excursion (`lockout_mae_pct`, inside
`bars_held < min_hold_bars`), plus volatility-normalized multiples
(`k_mfe`, `k_mae`, `k_lockout_mae`) for training `BarrierSizingModel`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..state.contracts import MarketSnapshot


@dataclass(frozen=True)
class ExcursionSample:
    direction: int
    mfe_pct: float
    mae_pct: float
    lockout_mae_pct: float
    volatility_pct: float
    k_mfe: float
    k_mae: float
    k_lockout_mae: float
    bars_examined: int


def compute_forward_excursions(
    entry_price: float,
    direction: int,
    forward_bars: Sequence[MarketSnapshot],
    *,
    max_holding_bars: int = 5,
    min_hold_bars: int = 1,
    volatility_pct: float = 0.2,
) -> ExcursionSample:
    """Compute MFE, MAE, and lockout MAE over `forward_bars[:max_holding_bars]`."""
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    window = list(forward_bars[: max(1, int(max_holding_bars))])
    if not window:
        raise ValueError("forward_bars must be non-empty")

    mfe = 0.0
    mae = 0.0
    lockout_mae = 0.0
    for i, bar in enumerate(window):
        bars_held = i + 1
        if direction == 1:
            fav = max(0.0, (bar.high - entry_price) / entry_price * 100.0)
            adv = max(0.0, (entry_price - bar.low) / entry_price * 100.0)
        else:
            fav = max(0.0, (entry_price - bar.low) / entry_price * 100.0)
            adv = max(0.0, (bar.high - entry_price) / entry_price * 100.0)
        mfe = max(mfe, fav)
        mae = max(mae, adv)
        if bars_held < min_hold_bars:
            lockout_mae = max(lockout_mae, adv)

    vol = max(float(volatility_pct), 1e-6)
    return ExcursionSample(
        direction=direction,
        mfe_pct=float(mfe),
        mae_pct=float(mae),
        lockout_mae_pct=float(lockout_mae),
        volatility_pct=vol,
        k_mfe=float(mfe / vol),
        k_mae=float(mae / vol),
        k_lockout_mae=float(lockout_mae / vol),
        bars_examined=len(window),
    )
