"""
Intrabar / sub-second price-path abstraction (`PricePath`, `BarPath`, `TickPath`)
used by `entry_fill_price`, `exit_fill_price`, `barrier_exit_step`, and
`label_triple_barrier`.

* `BarPath` wraps a sequence of OHLCV `MarketSnapshot` bars and resolves
  delayed fills (`t + delta_s`) via linear interpolation between the trigger
  bar and the next bar (with optional worst-case adverse excursion bound).
* `TickPath` wraps sub-second or 1-second `PathStep` records (trades + top of
  book) and resolves delayed fills and barrier crossings at the exact first
  step with `timestamp >= t + delta_s`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..state.contracts import MarketSnapshot


@dataclass(frozen=True)
class PathStep:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float
    spread_pct: float
    volume: float = 0.0
    depth_usdt: float = 0.0

    @classmethod
    def from_snapshot(cls, snap: MarketSnapshot) -> "PathStep":
        depth = (snap.bid_volume + snap.ask_volume) * snap.close
        return cls(
            timestamp=float(snap.timestamp),
            open=float(snap.open),
            high=float(snap.high),
            low=float(snap.low),
            close=float(snap.close),
            bid=float(snap.bid),
            ask=float(snap.ask),
            spread_pct=float(snap.spread_pct),
            volume=float(snap.volume),
            depth_usdt=float(depth),
        )


class PricePath(ABC):
    """Uniform price-path lookup for entry/exit fill simulation and barrier passage."""

    @abstractmethod
    def steps(self) -> Sequence[PathStep]:
        raise NotImplementedError

    @abstractmethod
    def price_at(
        self,
        t_start: float,
        delay_seconds: float,
        side: str = "BUY",
        trigger_price: Optional[float] = None,
    ) -> Tuple[float, float, PathStep]:
        """Return `(reference_mid_price, spread_pct, step)` at `t_start + delay_seconds`."""
        raise NotImplementedError

    @abstractmethod
    def first_passage(
        self,
        t_start: float,
        t_end: float,
        direction: int,
        tp_level: float,
        sl_level: float,
    ) -> Optional[Tuple[str, float, float]]:
        """Return `(reason, trigger_price, timestamp)` for the first barrier crossed in `[t_start, t_end]`."""
        raise NotImplementedError


class BarPath(PricePath):
    """Bar-resolution `PricePath` with linear sub-bar delay interpolation."""

    def __init__(
        self,
        bars: Sequence[MarketSnapshot],
        granularity_seconds: float = 60.0,
        worst_case_intrabar: bool = False,
    ) -> None:
        if not bars:
            raise ValueError("BarPath requires at least one MarketSnapshot")
        self._bars = list(bars)
        self._steps = [PathStep.from_snapshot(b) for b in self._bars]
        self.granularity_seconds = max(1e-6, float(granularity_seconds))
        self.worst_case_intrabar = bool(worst_case_intrabar)

    def steps(self) -> Sequence[PathStep]:
        return self._steps

    def price_at(
        self,
        t_start: float,
        delay_seconds: float,
        side: str = "BUY",
        trigger_price: Optional[float] = None,
    ) -> Tuple[float, float, PathStep]:
        target_ts = float(t_start) + max(0.0, float(delay_seconds))
        # Locate the step covering t_start
        idx = 0
        for i, step in enumerate(self._steps):
            if step.timestamp <= t_start:
                idx = i
            else:
                break

        base_step = self._steps[idx]
        ref_start = float(trigger_price) if trigger_price is not None else base_step.close
        if delay_seconds <= 0.0:
            return ref_start, base_step.spread_pct, base_step

        # Walk forward across bars if delay spans multiple bars
        bars_offset = int(delay_seconds // self.granularity_seconds)
        rem_seconds = delay_seconds - bars_offset * self.granularity_seconds
        target_idx = min(len(self._steps) - 1, idx + bars_offset)
        curr_step = self._steps[target_idx]
        next_idx = min(len(self._steps) - 1, target_idx + 1)
        next_step = self._steps[next_idx]

        if bars_offset > 0 and target_idx > idx:
            ref_start = curr_step.open

        alpha = min(1.0, max(0.0, rem_seconds / self.granularity_seconds))
        if next_idx > target_idx:
            end_price = next_step.open
            interp_spread = (1.0 - alpha) * curr_step.spread_pct + alpha * next_step.spread_pct
        else:
            end_price = curr_step.close
            interp_spread = curr_step.spread_pct

        mid_price = (1.0 - alpha) * ref_start + alpha * end_price
        if self.worst_case_intrabar:
            if side.upper() == "BUY":
                mid_price = max(mid_price, curr_step.high if alpha > 0 else mid_price)
            else:
                mid_price = min(mid_price, curr_step.low if alpha > 0 else mid_price)

        chosen_step = next_step if (bars_offset > 0 or alpha >= 1.0) else curr_step
        _ = target_ts
        return float(mid_price), float(interp_spread), chosen_step

    def first_passage(
        self,
        t_start: float,
        t_end: float,
        direction: int,
        tp_level: float,
        sl_level: float,
    ) -> Optional[Tuple[str, float, float]]:
        for step in self._steps:
            if step.timestamp < t_start:
                continue
            if step.timestamp > t_end:
                break
            if direction == 1:
                hit_sl = step.low <= sl_level
                hit_tp = step.high >= tp_level
            else:
                hit_sl = step.high >= sl_level
                hit_tp = step.low <= tp_level
            # Conservative tie-break on coarse bars: stop_loss wins
            if hit_sl:
                return ("stop_loss", sl_level, step.timestamp)
            if hit_tp:
                return ("take_profit", tp_level, step.timestamp)
        return None


class TickPath(PricePath):
    """Exact tick / 1s-bar resolution `PricePath`."""

    def __init__(self, steps: Sequence[PathStep]) -> None:
        if not steps:
            raise ValueError("TickPath requires at least one PathStep")
        self._steps = sorted(steps, key=lambda s: s.timestamp)

    @classmethod
    def from_snapshots(cls, snapshots: Sequence[MarketSnapshot]) -> "TickPath":
        return cls([PathStep.from_snapshot(s) for s in snapshots])

    def steps(self) -> Sequence[PathStep]:
        return self._steps

    def price_at(
        self,
        t_start: float,
        delay_seconds: float,
        side: str = "BUY",
        trigger_price: Optional[float] = None,
    ) -> Tuple[float, float, PathStep]:
        target_ts = float(t_start) + max(0.0, float(delay_seconds))
        for step in self._steps:
            if step.timestamp >= target_ts - 1e-9:
                mid = (step.bid + step.ask) / 2.0 if (step.bid > 0 and step.ask > 0) else step.close
                return float(mid), float(step.spread_pct), step
        last = self._steps[-1]
        mid = (last.bid + last.ask) / 2.0 if (last.bid > 0 and last.ask > 0) else last.close
        return float(mid), float(last.spread_pct), last

    def first_passage(
        self,
        t_start: float,
        t_end: float,
        direction: int,
        tp_level: float,
        sl_level: float,
    ) -> Optional[Tuple[str, float, float]]:
        for step in self._steps:
            if step.timestamp < t_start - 1e-9:
                continue
            if step.timestamp > t_end + 1e-9:
                break
            if direction == 1:
                hit_sl = step.low <= sl_level or (step.bid > 0 and step.bid <= sl_level)
                hit_tp = step.high >= tp_level or (step.bid > 0 and step.bid >= tp_level)
            else:
                hit_sl = step.high >= sl_level or (step.ask > 0 and step.ask >= sl_level)
                hit_tp = step.low <= tp_level or (step.ask > 0 and step.ask <= tp_level)
            if hit_sl:
                return ("stop_loss", sl_level, step.timestamp)
            if hit_tp:
                return ("take_profit", tp_level, step.timestamp)
        return None
