"""
Per-expert / per-instance cool-down manager (`CooldownManager`).

Supports:
* Cool-down after any stop exit (`after_stop_loss_bars` / `after_stop_loss_seconds`)
* Consecutive-loss streak cool-down (`consecutive_losses` -> `cooldown_bars` / `cooldown_seconds`)
* Rolling window expectancy / win-rate degradation cool-down
* Persistence into `RiskStateStore` so active cool-downs survive restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..labeling.triple_barrier import is_stop_exit_reason


@dataclass(frozen=True)
class CooldownConfig:
    enabled: bool = False
    after_stop_loss_bars: int = 0
    after_stop_loss_seconds: float = 0.0
    consecutive_losses: int = 3
    cooldown_bars: int = 0
    cooldown_seconds: float = 0.0
    rolling_window_trades: int = 10
    min_win_rate: float = 0.0
    min_expectancy_pct: float = -999.0
    rolling_cooldown_bars: int = 0
    rolling_cooldown_seconds: float = 0.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CooldownConfig":
        risk_cfg = config.get("risk") or {}
        cd_cfg = risk_cfg.get("cooldowns") or config.get("cooldowns") or {}
        if not cd_cfg:
            return cls(enabled=False)
        return cls(
            enabled=bool(cd_cfg.get("enabled", True)),
            after_stop_loss_bars=int(cd_cfg.get("after_stop_loss_bars", 0)),
            after_stop_loss_seconds=float(cd_cfg.get("after_stop_loss_seconds", 0.0)),
            consecutive_losses=int(cd_cfg.get("consecutive_losses", 3)),
            cooldown_bars=int(cd_cfg.get("cooldown_bars", 0)),
            cooldown_seconds=float(cd_cfg.get("cooldown_seconds", 0.0)),
            rolling_window_trades=int(cd_cfg.get("rolling_window_trades", 10)),
            min_win_rate=float(cd_cfg.get("min_win_rate", 0.0)),
            min_expectancy_pct=float(cd_cfg.get("min_expectancy_pct", -999.0)),
            rolling_cooldown_bars=int(cd_cfg.get("rolling_cooldown_bars", 0)),
            rolling_cooldown_seconds=float(cd_cfg.get("rolling_cooldown_seconds", 0.0)),
        )


@dataclass
class InstanceCooldownState:
    consecutive_losses: int = 0
    recent_pnls: List[float] = field(default_factory=list)
    until_bar_index: int = -1
    until_timestamp: float = -1.0
    reason: Optional[str] = None


class CooldownManager:
    """Per-expert-instance cool-down tracker."""

    def __init__(
        self,
        config: Optional[CooldownConfig] = None,
        store: Optional[Any] = None,
    ) -> None:
        self.config = config or CooldownConfig()
        self.store = store
        self._states: Dict[str, InstanceCooldownState] = {}
        self._load_from_store()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        store: Optional[Any] = None,
    ) -> "CooldownManager":
        return cls(config=CooldownConfig.from_config(config), store=store)

    def _load_from_store(self) -> None:
        if self.store is None:
            return
        raw = self.store._get("expert_cooldowns", {})
        if isinstance(raw, dict):
            for inst_id, data in raw.items():
                self._states[inst_id] = InstanceCooldownState(
                    consecutive_losses=int(data.get("consecutive_losses", 0)),
                    recent_pnls=[float(x) for x in data.get("recent_pnls", [])],
                    until_bar_index=int(data.get("until_bar_index", -1)),
                    until_timestamp=float(data.get("until_timestamp", -1.0)),
                    reason=data.get("reason"),
                )

    def _save_to_store(self) -> None:
        if self.store is None:
            return
        payload = {
            inst_id: {
                "consecutive_losses": st.consecutive_losses,
                "recent_pnls": list(st.recent_pnls),
                "until_bar_index": st.until_bar_index,
                "until_timestamp": st.until_timestamp,
                "reason": st.reason,
            }
            for inst_id, st in self._states.items()
        }
        self.store._set("expert_cooldowns", payload)

    def record_outcome(
        self,
        instance_id: str,
        pnl_pct: float,
        exit_reason: str,
        timestamp: float,
        bar_index: int = 0,
        extra_bars_multiplier: float = 1.0,
    ) -> Optional[str]:
        """Update per-instance outcome history and activate cool-down if a trigger fires."""
        if not self.config.enabled or not instance_id:
            return None

        st = self._states.setdefault(instance_id, InstanceCooldownState())
        if pnl_pct <= 0:
            st.consecutive_losses += 1
        else:
            st.consecutive_losses = 0

        st.recent_pnls.append(float(pnl_pct))
        win_cap = max(1, self.config.rolling_window_trades)
        if len(st.recent_pnls) > win_cap:
            st.recent_pnls = st.recent_pnls[-win_cap:]

        mult = max(1.0, float(extra_bars_multiplier))
        triggered_reason: Optional[str] = None
        add_bars = 0
        add_seconds = 0.0

        # 1. Stop-loss exit trigger
        if is_stop_exit_reason(exit_reason) and (
            self.config.after_stop_loss_bars > 0 or self.config.after_stop_loss_seconds > 0
        ):
            add_bars = max(add_bars, int(round(self.config.after_stop_loss_bars * mult)))
            add_seconds = max(add_seconds, self.config.after_stop_loss_seconds * mult)
            triggered_reason = f"stop_exit({exit_reason})"

        # 2. Consecutive losses trigger
        if (
            self.config.consecutive_losses > 0
            and st.consecutive_losses >= self.config.consecutive_losses
            and (self.config.cooldown_bars > 0 or self.config.cooldown_seconds > 0)
        ):
            add_bars = max(add_bars, int(round(self.config.cooldown_bars * mult)))
            add_seconds = max(add_seconds, self.config.cooldown_seconds * mult)
            triggered_reason = f"consecutive_losses({st.consecutive_losses})"
            # Reset consecutive counter so on resume it starts a fresh streak
            st.consecutive_losses = 0

        # 3. Rolling window win-rate / expectancy trigger
        if len(st.recent_pnls) >= win_cap and (
            self.config.rolling_cooldown_bars > 0 or self.config.rolling_cooldown_seconds > 0
        ):
            wins = sum(1 for x in st.recent_pnls if x > 0)
            win_rate = wins / len(st.recent_pnls)
            expectancy = sum(st.recent_pnls) / len(st.recent_pnls)
            if win_rate < self.config.min_win_rate or expectancy < self.config.min_expectancy_pct:
                add_bars = max(add_bars, int(round(self.config.rolling_cooldown_bars * mult)))
                add_seconds = max(add_seconds, self.config.rolling_cooldown_seconds * mult)
                triggered_reason = f"rolling_degradation(wr={win_rate:.2f},exp={expectancy:.3f}%)"
                st.recent_pnls.clear()

        if triggered_reason is not None and (add_bars > 0 or add_seconds > 0):
            if add_bars > 0:
                st.until_bar_index = max(st.until_bar_index, bar_index + add_bars)
            if add_seconds > 0:
                st.until_timestamp = max(st.until_timestamp, float(timestamp) + add_seconds)
            st.reason = triggered_reason
            self._save_to_store()
            return triggered_reason

        self._save_to_store()
        return None

    def is_cooled_down(
        self,
        instance_id: str,
        current_timestamp: float,
        current_bar_index: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        if not self.config.enabled or not instance_id:
            return (False, None)
        st = self._states.get(instance_id)
        if st is None:
            return (False, None)
        by_bar = st.until_bar_index > 0 and current_bar_index < st.until_bar_index
        by_time = st.until_timestamp > 0 and float(current_timestamp) < st.until_timestamp - 1e-6
        if by_bar or by_time:
            return (True, f"expert_cooldown:{instance_id}:{st.reason or 'active'}")
        return (False, None)

    def active_cooldowns(self, current_timestamp: float, current_bar_index: int = 0) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for inst_id in list(self._states.keys()):
            active, reason = self.is_cooled_down(inst_id, current_timestamp, current_bar_index)
            if active and reason:
                out[inst_id] = reason
        return out
