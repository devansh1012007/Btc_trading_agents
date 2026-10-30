"""
Multi-position book (`PositionBook`) enforcing portfolio concurrency,
per-expert/per-regime/per-direction caps, same-expert stacking, hedging rules,
gross/net exposure limits, deterministic lane assignment, and mark-to-market
equity/drawdown calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..labeling.triple_barrier import raw_return_pct


@dataclass(frozen=True)
class PositionLimits:
    max_concurrent: int = 1
    max_per_expert: int = 1
    max_per_regime: int = 1
    max_per_direction: int = 1
    allow_same_expert_stacking: bool = False
    allow_hedging: bool = False
    max_gross_exposure_pct_of_equity: float = 100.0
    max_net_exposure_pct_of_equity: float = 100.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PositionLimits":
        pos_cfg = config.get("positions") or {}
        risk_cfg = config.get("risk") or {}
        breaker_cfg = risk_cfg.get("circuit_breakers") or {}
        legacy_max = int(breaker_cfg.get("max_concurrent_positions", 1))
        max_conc = int(pos_cfg.get("max_concurrent", legacy_max))
        if max_conc < 1:
            raise ValueError(f"max_concurrent={max_conc} must be >= 1")
        default_sub_cap = max_conc if "positions" in config else max(1, max_conc)
        return cls(
            max_concurrent=max_conc,
            max_per_expert=int(pos_cfg.get("max_per_expert", default_sub_cap)),
            max_per_regime=int(pos_cfg.get("max_per_regime", default_sub_cap)),
            max_per_direction=int(pos_cfg.get("max_per_direction", default_sub_cap)),
            allow_same_expert_stacking=bool(pos_cfg.get("allow_same_expert_stacking", False)),
            allow_hedging=bool(pos_cfg.get("allow_hedging", False)),
            max_gross_exposure_pct_of_equity=float(pos_cfg.get("max_gross_exposure_pct_of_equity", 100.0)),
            max_net_exposure_pct_of_equity=float(pos_cfg.get("max_net_exposure_pct_of_equity", 100.0)),
        )


class PositionBook:
    """Deterministic multi-position container."""

    def __init__(
        self,
        limits: Optional[PositionLimits] = None,
        initial_positions: Optional[List[Any]] = None,
    ) -> None:
        self.limits = limits or PositionLimits()
        self._positions: List[Any] = []
        if initial_positions:
            for p in initial_positions:
                self.add(p)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        initial_positions: Optional[List[Any]] = None,
    ) -> "PositionBook":
        return cls(limits=PositionLimits.from_config(config), initial_positions=initial_positions)

    @property
    def positions(self) -> List[Any]:
        return list(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def is_empty(self) -> bool:
        return len(self._positions) == 0

    def primary_position(self) -> Optional[Any]:
        return self._positions[0] if self._positions else None

    def gross_exposure_usdt(self) -> float:
        return float(sum(float(getattr(p, "size_usdt", 0.0)) for p in self._positions))

    def net_exposure_usdt(self) -> float:
        return float(
            sum(
                int(getattr(p, "direction", 1)) * float(getattr(p, "size_usdt", 0.0))
                for p in self._positions
            )
        )

    def _next_free_lane(self) -> int:
        used = {int(getattr(p, "lane_id", 0)) for p in self._positions if getattr(p, "lane_id", None) is not None}
        lane = 0
        while lane in used:
            lane += 1
        return lane

    def add(self, position: Any) -> int:
        used = {int(getattr(p, "lane_id", 0)) for p in self._positions if getattr(p, "lane_id", None) is not None}
        lane = getattr(position, "lane_id", None)
        if lane is None or int(lane) in used:
            lane = self._next_free_lane()
            try:
                setattr(position, "lane_id", lane)
            except Exception:
                pass
        self._positions.append(position)
        self._positions.sort(key=lambda p: (int(getattr(p, "entry_timestamp", 0)), str(getattr(p, "trade_id", ""))))
        return int(lane)

    def remove(self, position: Any) -> None:
        self._positions = [p for p in self._positions if p is not position]

    def can_open(
        self,
        *,
        expert: Optional[str],
        regime: Optional[str],
        direction: int,
        size_usdt: float,
        equity: float,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check whether a candidate position can be opened under `self.limits`.

        Returns `(allowed, terminal_stage, rejection_reason)`.
        """
        if len(self._positions) >= self.limits.max_concurrent:
            return (
                False,
                "concurrency_limit",
                f"concurrency_limit:max_concurrent({self.limits.max_concurrent}) reached",
            )

        if expert is not None:
            same_expert_count = sum(1 for p in self._positions if getattr(p, "expert", None) == expert)
            if not self.limits.allow_same_expert_stacking and same_expert_count >= 1:
                return (
                    False,
                    "concurrency_limit",
                    f"concurrency_limit:same_expert_stacking_disabled({expert})",
                )
            if same_expert_count >= self.limits.max_per_expert:
                return (
                    False,
                    "concurrency_limit",
                    f"concurrency_limit:max_per_expert({self.limits.max_per_expert}) reached for {expert}",
                )

        if regime is not None:
            same_regime_count = sum(1 for p in self._positions if getattr(p, "regime", None) == regime)
            if same_regime_count >= self.limits.max_per_regime:
                return (
                    False,
                    "concurrency_limit",
                    f"concurrency_limit:max_per_regime({self.limits.max_per_regime}) reached for {regime}",
                )

        same_dir_count = sum(1 for p in self._positions if int(getattr(p, "direction", 1)) == direction)
        if same_dir_count >= self.limits.max_per_direction:
            return (
                False,
                "concurrency_limit",
                f"concurrency_limit:max_per_direction({self.limits.max_per_direction}) reached",
            )

        if not self.limits.allow_hedging:
            opp_dir_count = sum(1 for p in self._positions if int(getattr(p, "direction", 1)) == -direction)
            if opp_dir_count > 0:
                return (
                    False,
                    "concurrency_limit",
                    "concurrency_limit:hedging_not_allowed",
                )

        if equity > 0 and size_usdt > 0:
            new_gross_pct = (self.gross_exposure_usdt() + size_usdt) / equity * 100.0
            if new_gross_pct > self.limits.max_gross_exposure_pct_of_equity + 1e-9:
                return (
                    False,
                    "exposure_limit",
                    f"exposure_limit:gross_exposure({new_gross_pct:.2f}%>{self.limits.max_gross_exposure_pct_of_equity:.2f}%)",
                )

            new_net_pct = abs(self.net_exposure_usdt() + direction * size_usdt) / equity * 100.0
            if new_net_pct > self.limits.max_net_exposure_pct_of_equity + 1e-9:
                return (
                    False,
                    "exposure_limit",
                    f"exposure_limit:net_exposure({new_net_pct:.2f}%>{self.limits.max_net_exposure_pct_of_equity:.2f}%)",
                )

        return (True, None, None)

    def unrealized_pnl_usdt(self, close_price: float) -> float:
        if close_price <= 0 or not self._positions:
            return 0.0
        total = 0.0
        for p in self._positions:
            ret_pct = raw_return_pct(int(p.direction), float(p.entry_price), float(close_price))
            total += float(p.size_usdt) * (ret_pct / 100.0)
        return float(total)

    def mark_to_market_equity(self, realized_equity: float, close_price: float) -> float:
        return float(realized_equity) + self.unrealized_pnl_usdt(close_price)

    def to_summary_dict(self, realized_equity: float = 0.0, close_price: float = 0.0) -> Dict[str, Any]:
        return {
            "open_count": len(self._positions),
            "max_concurrent": self.limits.max_concurrent,
            "gross_exposure_usdt": self.gross_exposure_usdt(),
            "net_exposure_usdt": self.net_exposure_usdt(),
            "unrealized_pnl_usdt": self.unrealized_pnl_usdt(close_price),
            "mtm_equity": self.mark_to_market_equity(realized_equity, close_price),
        }
