"""
Triple-barrier labeling (§5): given an entry and a forward window of bars,
find which barrier gets touched first -- take-profit, stop-loss, or
timeout -- and compute the resulting net-of-fees return.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..state.contracts import MarketSnapshot


@dataclass(frozen=True)
class TripleBarrierOutcome:
    entry_timestamp: int
    direction: int              # +1 long, -1 short
    exit_reason: str            # "take_profit" | "stop_loss" | "timeout"
    bars_held: int
    realized_return_pct: float  # net of round-trip fees
    max_adverse_excursion_pct: float
    bars_to_resolution: int


def compute_barrier_pcts(volatility_pct: float, k_tp: float, k_sl: float) -> Tuple[float, float]:
    """Volatility-scaled barrier sizing (§5, §9): TP = k_tp * vol, SL = k_sl * vol."""
    return k_tp * volatility_pct, k_sl * volatility_pct


def barrier_volatility_pct(fv) -> float:
    """
    The volatility figure every Phase 3 barrier-sizing call site should
    use. Deliberately volatility_5m, NOT volatility_30s: FeatureEngine
    computes volatility_30s as the std of a single 1-period return
    (periods=1), and np.std() of a one-element array is 0.0 by
    mathematical necessity -- not a bug you can tune around, a
    denominator that is always exactly zero. Every barrier collapsed to
    zero width and every simulated trade lost exactly the round-trip fee
    until this was traced down (see PHASE3_COMPLETE.md). volatility_5m
    (periods=5) is a well-defined dispersion statistic and the closest
    genuinely-computed alternative.
    """
    return max(fv.volatility_5m, 1e-6)


def label_triple_barrier(
    forward_bars: List[MarketSnapshot],
    entry_price: float,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    fee_pct: float,
    entry_timestamp: Optional[int] = None,
) -> TripleBarrierOutcome:
    """
    Walk forward_bars looking for the first touch of TP, SL, or timeout.
    Only ever reads forward_bars[:max_holding_bars] -- the caller controls
    look-ahead by what it passes in.

    Barrier-touch convention: if a single bar's high/low touches BOTH the
    TP and SL level, this assumes the adverse barrier (stop-loss) was hit
    first -- the same conservative, can't-see-intrabar-ordering caveat
    Phase 2's §6.7 documents for zone touches at this granularity.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long) or -1 (short)")
    if not forward_bars:
        raise ValueError("forward_bars must be non-empty")
    if max_holding_bars <= 0:
        raise ValueError("max_holding_bars must be positive")

    if direction == 1:
        tp_level = entry_price * (1 + take_profit_pct / 100)
        sl_level = entry_price * (1 - stop_loss_pct / 100)
    else:
        tp_level = entry_price * (1 - take_profit_pct / 100)
        sl_level = entry_price * (1 + stop_loss_pct / 100)

    bars_to_check = forward_bars[:max_holding_bars]
    entry_ts = entry_timestamp if entry_timestamp is not None else bars_to_check[0].timestamp

    max_adverse_excursion_pct = 0.0

    for i, bar in enumerate(bars_to_check):
        adverse = (
            (entry_price - bar.low) / entry_price * 100 if direction == 1
            else (bar.high - entry_price) / entry_price * 100
        )
        max_adverse_excursion_pct = max(max_adverse_excursion_pct, adverse)

        if direction == 1:
            hit_tp = bar.high >= tp_level
            hit_sl = bar.low <= sl_level
        else:
            hit_tp = bar.low <= tp_level
            hit_sl = bar.high >= sl_level

        if hit_sl:
            exit_price, exit_reason = sl_level, "stop_loss"
        elif hit_tp:
            exit_price, exit_reason = tp_level, "take_profit"
        else:
            continue

        raw_return_pct = (
            (exit_price - entry_price) / entry_price * 100 if direction == 1
            else (entry_price - exit_price) / entry_price * 100
        )
        return TripleBarrierOutcome(
            entry_timestamp=entry_ts, direction=direction, exit_reason=exit_reason,
            bars_held=i + 1, realized_return_pct=raw_return_pct - (fee_pct * 2),
            max_adverse_excursion_pct=max_adverse_excursion_pct, bars_to_resolution=i + 1,
        )

    last_bar = bars_to_check[-1]
    raw_return_pct = (
        (last_bar.close - entry_price) / entry_price * 100 if direction == 1
        else (entry_price - last_bar.close) / entry_price * 100
    )
    return TripleBarrierOutcome(
        entry_timestamp=entry_ts, direction=direction, exit_reason="timeout",
        bars_held=len(bars_to_check), realized_return_pct=raw_return_pct - (fee_pct * 2),
        max_adverse_excursion_pct=max_adverse_excursion_pct, bars_to_resolution=len(bars_to_check),
    )
