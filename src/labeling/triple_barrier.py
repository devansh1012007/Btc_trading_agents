"""
Triple-barrier labeling (§5): given an entry and a forward window of bars,
find which barrier gets touched first -- take-profit, stop-loss, or
timeout -- and compute the resulting net-of-fees return.

THE exit rule lives here, once: :func:`barrier_exit_step`. The labeler
(:func:`label_triple_barrier`, used by every training script) and the
position lifecycle (``check_position_exit``, used by the harness, the live
loop and the GUI replay) both call it, so a label and a live trade cannot
disagree about when -- or at what price -- a position exits. That includes
the MINIMUM-HOLD LOCKOUT (``min_hold_bars``): no exit of any kind before
the position has been held that long (see ``hold_policy.py``).
"""
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

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
    #: the exit happened on the FIRST bar the lockout allowed (the
    #: barrier was, or may have been, reachable earlier). A high share of
    #: these means the barriers are tight relative to the minimum hold.
    pinned_to_min_hold: bool = False
    #: worst adverse move (%) seen while the lockout still forbade an exit
    lockout_adverse_excursion_pct: float = 0.0


def barriers_for_regime(
    config: Mapping[str, Any], regime_name: str, volatility_pct: float,
) -> Tuple[float, float]:
    """Take-profit / stop-loss percentages for an entry in ``regime_name``.

    The ONE implementation of barrier sizing: ``labeling.k_tp`` / ``k_sl``,
    overridden per regime by ``labeling.regime_specific_barriers`` (keyed by
    the ``RegimeType`` member NAME), scaled by the entry bar's volatility.
    The training scripts, the harness and ``process_bar`` all call it.
    """
    labeling_cfg = config.get("labeling", {}) or {}
    overrides = (labeling_cfg.get("regime_specific_barriers", {}) or {}).get(regime_name, {}) or {}
    k_tp = overrides.get("k_tp", labeling_cfg.get("k_tp", 1.5))
    k_sl = overrides.get("k_sl", labeling_cfg.get("k_sl", 1.0))
    return compute_barrier_pcts(volatility_pct, k_tp, k_sl)


def barrier_levels(direction: int, entry_price: float, tp_pct: float, sl_pct: float) -> Tuple[float, float]:
    """(take-profit level, stop-loss level) for a long (+1) or short (-1)."""
    if direction == 1:
        return entry_price * (1 + tp_pct / 100), entry_price * (1 - sl_pct / 100)
    return entry_price * (1 - tp_pct / 100), entry_price * (1 + sl_pct / 100)


def barrier_exit_step(
    direction: int, entry_price: float, take_profit_pct: float, stop_loss_pct: float,
    bar: MarketSnapshot, bars_held: int, max_holding_bars: int, min_hold_bars: int = 1,
) -> Optional[Tuple[str, float]]:
    """One bar of the exit rule: ``(exit_reason, exit_price)`` or ``None``.

    ``bars_held`` counts the bar being examined (the first bar after the
    entry is ``1``). Order of precedence on a bar that touches several
    barriers: stop-loss, then take-profit, then timeout -- the conservative
    "cannot see intrabar ordering" convention.

    Minimum-hold lockout: while ``bars_held < min_hold_bars`` NOTHING can
    exit -- not a stop, not a target, not the timeout. On the first bar the
    lockout allows (when it lasted more than one bar) a stop-loss fills at
    the WORSE of the stop level and that bar's open: the position could not
    act on the move that carried price through its stop, so it does not get
    the stop price for free.
    """
    if bars_held < min_hold_bars:
        return None

    tp_level, sl_level = barrier_levels(direction, entry_price, take_profit_pct, stop_loss_pct)
    if direction == 1:
        hit_tp, hit_sl = bar.high >= tp_level, bar.low <= sl_level
    else:
        hit_tp, hit_sl = bar.low <= tp_level, bar.high >= sl_level

    if hit_sl:
        exit_price = sl_level
        if min_hold_bars > 1 and bars_held == min_hold_bars:
            exit_price = min(sl_level, bar.open) if direction == 1 else max(sl_level, bar.open)
        return "stop_loss", exit_price
    if hit_tp:
        return "take_profit", tp_level
    if bars_held >= max_holding_bars:
        return "timeout", bar.close
    return None


def raw_return_pct(direction: int, entry_price: float, exit_price: float) -> float:
    """Gross return (%) of a long/short from entry to exit, before fees."""
    if direction == 1:
        return (exit_price - entry_price) / entry_price * 100
    return (entry_price - exit_price) / entry_price * 100


def compute_barrier_pcts(volatility_pct: float, k_tp: float, k_sl: float) -> Tuple[float, float]:
    """Volatility-scaled barrier sizing (§5, §9): TP = k_tp * vol, SL = k_sl * vol."""
    return k_tp * volatility_pct, k_sl * volatility_pct


def barrier_volatility_pct(fv) -> float:
    """
    The volatility figure every Phase 3 barrier-sizing call site should
    use. Deliberately volatility_medium, NOT volatility_short: FeatureEngine
    computes volatility_short as the std of a single 1-period return
    (periods=1), and np.std() of a one-element array is 0.0 by
    mathematical necessity -- not a bug you can tune around, a
    denominator that is always exactly zero. Every barrier collapsed to
    zero width and every simulated trade lost exactly the round-trip fee
    until this was traced down (see PHASE3_COMPLETE.md). volatility_medium
    (periods=5) is a well-defined dispersion statistic and the closest
    genuinely-computed alternative.
    """
    return max(fv.volatility_medium, 1e-6)


def label_triple_barrier(
    forward_bars: List[MarketSnapshot],
    entry_price: float,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    fee_pct: float,
    entry_timestamp: Optional[int] = None,
    min_hold_bars: int = 1,
) -> TripleBarrierOutcome:
    """
    Walk forward_bars applying :func:`barrier_exit_step` until the position
    exits. Only ever reads forward_bars[:max_holding_bars] -- the caller
    controls look-ahead by what it passes in.

    ``min_hold_bars`` is the minimum-hold lockout (``HoldPolicy.min_hold_bars``);
    the default of 1 is "no lockout" and reproduces the original labels.

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
    if min_hold_bars < 1:
        raise ValueError("min_hold_bars must be >= 1")
    if max_holding_bars < min_hold_bars:
        raise ValueError(
            f"max_holding_bars={max_holding_bars} is shorter than min_hold_bars={min_hold_bars}: "
            f"every timeout would be an illegal trade"
        )

    bars_to_check = forward_bars[:max_holding_bars]
    entry_ts = entry_timestamp if entry_timestamp is not None else bars_to_check[0].timestamp

    max_adverse_excursion_pct = 0.0
    lockout_adverse_excursion_pct = 0.0

    for i, bar in enumerate(bars_to_check):
        bars_held = i + 1
        adverse = (
            (entry_price - bar.low) / entry_price * 100 if direction == 1
            else (bar.high - entry_price) / entry_price * 100
        )
        max_adverse_excursion_pct = max(max_adverse_excursion_pct, adverse)
        if bars_held < min_hold_bars:
            lockout_adverse_excursion_pct = max(lockout_adverse_excursion_pct, adverse)

        step = barrier_exit_step(
            direction, entry_price, take_profit_pct, stop_loss_pct, bar,
            bars_held, max_holding_bars, min_hold_bars,
        )
        if step is None:
            continue

        exit_reason, exit_price = step
        return TripleBarrierOutcome(
            entry_timestamp=entry_ts, direction=direction, exit_reason=exit_reason,
            bars_held=bars_held,
            realized_return_pct=raw_return_pct(direction, entry_price, exit_price) - (fee_pct * 2),
            max_adverse_excursion_pct=max_adverse_excursion_pct, bars_to_resolution=bars_held,
            pinned_to_min_hold=(min_hold_bars > 1 and bars_held == min_hold_bars
                                and exit_reason != "timeout"),
            lockout_adverse_excursion_pct=lockout_adverse_excursion_pct,
        )

    # Ran out of forward bars before any barrier / the timeout: the window
    # ended, so resolve at the last available close (end-of-data), but never
    # inside the lockout -- a truncated window shorter than the minimum hold
    # is not a legal trade and is reported as such by the caller.
    last_bar = bars_to_check[-1]
    return TripleBarrierOutcome(
        entry_timestamp=entry_ts, direction=direction, exit_reason="timeout",
        bars_held=len(bars_to_check),
        realized_return_pct=raw_return_pct(direction, entry_price, last_bar.close) - (fee_pct * 2),
        max_adverse_excursion_pct=max_adverse_excursion_pct, bars_to_resolution=len(bars_to_check),
        pinned_to_min_hold=False, lockout_adverse_excursion_pct=lockout_adverse_excursion_pct,
    )
