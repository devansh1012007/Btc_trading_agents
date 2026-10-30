"""
Clock abstraction (`Clock`, `WallClock`, `BarClock`) injected into
`RiskStateStore`, `CooldownManager`, `HaltManager`, and `LiveBackend`.

No module in `src/layers/`, `src/execution/`, or `src/risk/` may call
`datetime.now()` or `date.today()` directly; all time queries go through an
injected `Clock`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class Clock(ABC):
    """Uniform clock interface for wall-clock live operation and bar-clock simulation."""

    @abstractmethod
    def now_epoch(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def now_utc(self) -> datetime:
        raise NotImplementedError

    def now_iso(self) -> str:
        return self.now_utc().isoformat()

    def today_iso(self) -> str:
        return self.now_utc().date().isoformat()

    @abstractmethod
    def advance(self, timestamp: float) -> None:
        raise NotImplementedError


class WallClock(Clock):
    """Production wall-clock using UTC system time (`advance` is a no-op)."""

    def now_epoch(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def advance(self, timestamp: float) -> None:
        _ = timestamp


class BarClock(Clock):
    """Simulated bar-time clock driven by bar timestamps."""

    def __init__(self, initial_timestamp: float = 1_700_000_000.0) -> None:
        self._epoch = float(initial_timestamp)

    def now_epoch(self) -> float:
        return self._epoch

    def now_utc(self) -> datetime:
        return datetime.fromtimestamp(self._epoch, tz=timezone.utc)

    def advance(self, timestamp: float) -> None:
        self._epoch = float(timestamp)

    def advance_seconds(self, delta_seconds: float) -> None:
        self._epoch += float(delta_seconds)
