"""
Time-boxed portfolio halt manager (`HaltManager`) driven by injected `Clock`.

Implements:
* `rolling_loss`: cumulative loss % inside `window_minutes` >= `max_loss_pct` -> halt for `halt_minutes`
* `rolling_expectancy`: mean PnL % over `window_trades` < `min_expectancy_pct` -> halt for `halt_minutes`
* `loss_streak`: consecutive portfolio losses >= `max_consecutive_losses` -> halt for `halt_minutes`
* Fresh-window rule on resume: when a time-boxed halt expires at `expires_at`,
  only trades closing strictly AFTER `expires_at` count toward the next rolling
  window / loss streak, so the same losses that tripped the halt cannot
  immediately re-trip it on the bar it expires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..utils.clock import Clock, WallClock


@dataclass(frozen=True)
class HaltConfig:
    enabled: bool = False
    rolling_loss_window_minutes: float = 120.0
    rolling_loss_max_pct: float = 0.0
    rolling_loss_halt_minutes: float = 180.0
    rolling_expectancy_window_trades: int = 0
    rolling_expectancy_min_pct: float = -999.0
    rolling_expectancy_halt_minutes: float = 120.0
    loss_streak_max: int = 0
    loss_streak_halt_minutes: float = 120.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HaltConfig":
        risk_cfg = config.get("risk") or {}
        halts_cfg = risk_cfg.get("halts") or config.get("halts") or {}
        if not halts_cfg:
            return cls(enabled=False)
        rl = halts_cfg.get("rolling_loss") or {}
        re_ = halts_cfg.get("rolling_expectancy") or {}
        ls = halts_cfg.get("loss_streak") or {}
        return cls(
            enabled=bool(halts_cfg.get("enabled", True)),
            rolling_loss_window_minutes=float(rl.get("window_minutes", 120.0)),
            rolling_loss_max_pct=float(rl.get("max_loss_pct", 0.0)),
            rolling_loss_halt_minutes=float(rl.get("halt_minutes", 180.0)),
            rolling_expectancy_window_trades=int(re_.get("window_trades", 0)),
            rolling_expectancy_min_pct=float(re_.get("min_expectancy_pct", -999.0)),
            rolling_expectancy_halt_minutes=float(re_.get("halt_minutes", 120.0)),
            loss_streak_max=int(ls.get("max_consecutive_losses", 0)),
            loss_streak_halt_minutes=float(ls.get("halt_minutes", 120.0)),
        )


@dataclass
class HaltEvent:
    halt_type: str
    reason: str
    tripped_at: float
    expires_at: float
    active: bool = True


class HaltManager:
    """Time-boxed portfolio halt manager."""

    def __init__(
        self,
        config: Optional[HaltConfig] = None,
        clock: Optional[Clock] = None,
        store: Optional[Any] = None,
    ) -> None:
        self.config = config or HaltConfig()
        self.clock = clock or WallClock()
        self.store = store
        self._trades: List[Tuple[float, float]] = []  # (timestamp, pnl_pct)
        self._loss_streak: int = 0
        self._window_floor_ts: float = 0.0
        self._active_halt: Optional[HaltEvent] = None
        self.history: List[HaltEvent] = []

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        clock: Optional[Clock] = None,
        store: Optional[Any] = None,
    ) -> "HaltManager":
        return cls(config=HaltConfig.from_config(config), clock=clock, store=store)

    def _expire_if_due(self, now_ts: float) -> None:
        if self._active_halt is not None and self._active_halt.active:
            if now_ts >= self._active_halt.expires_at - 1e-6:
                expired = self._active_halt
                expired.active = False
                # Fresh-window rule: only trades strictly after expires_at count for new halts
                self._window_floor_ts = max(self._window_floor_ts, float(expired.expires_at))
                self._loss_streak = 0
                self._trades = [(t, p) for (t, p) in self._trades if t > self._window_floor_ts]
                self._active_halt = None
                if self.store is not None:
                    self.store.reset(f"time_boxed:{expired.halt_type}")

    def trigger_manual_or_governor_halt(
        self,
        halt_type: str,
        reason: str,
        halt_minutes: float,
        timestamp: Optional[float] = None,
    ) -> HaltEvent:
        now_ts = float(timestamp) if timestamp is not None else self.clock.now_epoch()
        expires_at = now_ts + max(0.0, float(halt_minutes)) * 60.0
        ev = HaltEvent(
            halt_type=halt_type,
            reason=reason,
            tripped_at=now_ts,
            expires_at=expires_at,
            active=True,
        )
        self._active_halt = ev
        self.history.append(ev)
        if self.store is not None:
            self.store.trip(f"time_boxed:{halt_type}", f"{reason} (until {expires_at:.0f})")
        return ev

    def record_trade(self, pnl_pct: float, timestamp: Optional[float] = None) -> Optional[HaltEvent]:
        now_ts = float(timestamp) if timestamp is not None else self.clock.now_epoch()
        self._expire_if_due(now_ts)
        if not self.config.enabled:
            return None

        if now_ts <= self._window_floor_ts:
            return None

        self._trades.append((now_ts, float(pnl_pct)))
        if pnl_pct <= 0:
            self._loss_streak += 1
        else:
            self._loss_streak = 0

        # 1. Loss streak rule
        if (
            self.config.loss_streak_max > 0
            and self._loss_streak >= self.config.loss_streak_max
            and self.config.loss_streak_halt_minutes > 0
        ):
            return self.trigger_manual_or_governor_halt(
                "loss_streak",
                f"loss_streak={self._loss_streak}",
                self.config.loss_streak_halt_minutes,
                timestamp=now_ts,
            )

        # 2. Rolling loss over window_minutes
        if self.config.rolling_loss_max_pct > 0 and self.config.rolling_loss_halt_minutes > 0:
            cutoff = max(self._window_floor_ts, now_ts - self.config.rolling_loss_window_minutes * 60.0)
            window_losses = sum(abs(p) for (t, p) in self._trades if t >= cutoff and p < 0)
            if window_losses >= self.config.rolling_loss_max_pct - 1e-9:
                return self.trigger_manual_or_governor_halt(
                    "rolling_loss",
                    f"rolling_loss={window_losses:.3f}%>={self.config.rolling_loss_max_pct:.3f}%",
                    self.config.rolling_loss_halt_minutes,
                    timestamp=now_ts,
                )

        # 3. Rolling expectancy over window_trades
        w_trades = self.config.rolling_expectancy_window_trades
        if w_trades > 0 and self.config.rolling_expectancy_halt_minutes > 0:
            eligible = [p for (t, p) in self._trades if t > self._window_floor_ts]
            if len(eligible) >= w_trades:
                recent = eligible[-w_trades:]
                exp = sum(recent) / len(recent)
                if exp < self.config.rolling_expectancy_min_pct:
                    return self.trigger_manual_or_governor_halt(
                        "rolling_expectancy",
                        f"rolling_expectancy={exp:.3f}%<{self.config.rolling_expectancy_min_pct:.3f}%",
                        self.config.rolling_expectancy_halt_minutes,
                        timestamp=now_ts,
                    )

        return None

    def is_halted(self, timestamp: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        now_ts = float(timestamp) if timestamp is not None else self.clock.now_epoch()
        self._expire_if_due(now_ts)
        if self._active_halt is not None and self._active_halt.active:
            return (
                True,
                f"portfolio_halt:{self._active_halt.halt_type}:{self._active_halt.reason}",
            )
        return (False, None)

    @property
    def active_halt(self) -> Optional[HaltEvent]:
        return self._active_halt
