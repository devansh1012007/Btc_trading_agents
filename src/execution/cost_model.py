"""
Single consolidated cost model (`CostModel`) built from `costs:` config with
backward-compatible fallback to legacy keys (`labeling.fee_pct`,
`labeling.slippage_buffer_pct`, `execution.slippage_pct`,
`execution.simulated_latency_ms`, `execution.latency_adverse_pct_per_100ms`,
`execution.depth_participation_rate`, `execution.partial_fill_policy`,
`execution.order_rejection_rate`, `execution.assumed_taker_fee_pct`).

Every cost consumer (`label_triple_barrier`, `check_position_exit`,
`entry_fill_price`, `exit_fill_price`, `simulate_fill`, `OpportunityFilter`,
`PaperSimBackend`, `PaperLiveBackend`, `LiveBackend`, and `HoldPolicy`) reads
through `CostModel` so entry/exit fees, spread, slippage, latency drift,
market impact, and carry cannot drift apart.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..state.contracts import FeatureVector, MarketSnapshot


def has_legacy_cost_conflict(config: Mapping[str, Any]) -> bool:
    """True when both `costs.fees` and legacy `labeling.fee_pct` are set
    without `_allow_legacy_coexist`."""
    costs_cfg = config.get("costs")
    if not isinstance(costs_cfg, Mapping):
        return False
    labeling_cfg = config.get("labeling") or {}
    return (
        "fees" in costs_cfg
        and "fee_pct" in labeling_cfg
        and costs_cfg.get("_allow_legacy_coexist") is not True
    )


@dataclass(frozen=True)
class FeeSchedule:
    maker_pct: float = 0.04
    taker_pct: float = 0.04
    entry_role: str = "taker"
    exit_role: str = "taker"
    regulatory_pct: float = 0.0
    per_contract_fee: float = 0.0


@dataclass(frozen=True)
class SpreadSchedule:
    source: str = "snapshot"  # "snapshot" | "fixed" | "synthetic"
    fixed_pct: float = 0.02
    floor_pct: float = 0.0
    ceiling_pct: float = float("inf")
    exit_spread_multiplier: float = 1.0


@dataclass(frozen=True)
class SlippageSchedule:
    model: str = "fixed"  # "fixed" | "volatility_scaled" | "depth_based"
    fixed_pct: float = 0.01
    vol_base_pct: float = 0.005
    vol_k: float = 0.05
    depth_base_pct: float = 0.005
    depth_k_participation: float = 0.10
    max_slippage_pct: float = float("inf")
    exit_slippage_multiplier: float = 1.0
    stop_slippage_multiplier: float = 1.0


@dataclass(frozen=True)
class LatencySchedule:
    entry_delay_seconds: float = 0.0
    exit_delay_seconds: float = 0.0
    adverse_drift_pct_per_second: float = 0.0


@dataclass(frozen=True)
class ImpactSchedule:
    enabled: bool = False
    k_sqrt: float = 0.0
    participation_cap: float = 0.15
    partial_fill_policy: str = "partial"


@dataclass(frozen=True)
class CarrySchedule:
    enabled: bool = False
    funding_interval_hours: float = 8.0
    default_funding_rate_pct: float = 0.0
    borrow_apr_pct: float = 0.0


@dataclass(frozen=True)
class RejectSchedule:
    order_rejection_rate: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class CostModel:
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    spread: SpreadSchedule = field(default_factory=SpreadSchedule)
    slippage: SlippageSchedule = field(default_factory=SlippageSchedule)
    latency: LatencySchedule = field(default_factory=LatencySchedule)
    impact: ImpactSchedule = field(default_factory=ImpactSchedule)
    carry: CarrySchedule = field(default_factory=CarrySchedule)
    rejects: RejectSchedule = field(default_factory=RejectSchedule)
    slippage_buffer_pct: float = 0.0
    explicit_costs_block: bool = False

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "CostModel":
        if not config:
            return cls()
        costs_cfg = config.get("costs") or {}
        has_explicit = bool(costs_cfg)
        labeling_cfg = config.get("labeling") or {}
        exec_cfg = config.get("execution") or {}
        duration_cfg = config.get("trade_duration") or {}

        # Legacy fee fallback
        legacy_fee = labeling_cfg.get("fee_pct")
        if legacy_fee is None:
            legacy_fee = exec_cfg.get("assumed_taker_fee_pct", 0.04)
        legacy_fee_val = float(legacy_fee)

        fees_blk = costs_cfg.get("fees") or {}
        taker_pct = float(fees_blk.get("taker_pct", legacy_fee_val))
        maker_pct = float(fees_blk.get("maker_pct", taker_pct if "fees" not in costs_cfg else 0.02))
        fees = FeeSchedule(
            maker_pct=maker_pct,
            taker_pct=taker_pct,
            entry_role=str(fees_blk.get("entry_role", "taker")),
            exit_role=str(fees_blk.get("exit_role", "taker")),
            regulatory_pct=float(fees_blk.get("regulatory_pct", 0.0)),
            per_contract_fee=float(fees_blk.get("per_contract_fee", 0.0)),
        )

        spread_blk = costs_cfg.get("spread") or {}
        spread = SpreadSchedule(
            source=str(spread_blk.get("source", "snapshot")),
            fixed_pct=float(spread_blk.get("fixed_pct", 0.02)),
            floor_pct=float(spread_blk.get("floor_pct", 0.0)),
            ceiling_pct=float(spread_blk.get("ceiling_pct", float("inf"))),
            exit_spread_multiplier=float(spread_blk.get("exit_spread_multiplier", 1.0)),
        )

        slip_blk = costs_cfg.get("slippage") or {}
        legacy_slip = float(exec_cfg.get("slippage_pct", 0.01))
        vol_blk = slip_blk.get("vol_scaled") or {}
        depth_blk = slip_blk.get("depth_based") or {}
        slippage = SlippageSchedule(
            model=str(slip_blk.get("model", "fixed")),
            fixed_pct=float(slip_blk.get("fixed_pct", legacy_slip)),
            vol_base_pct=float(vol_blk.get("base_pct", 0.005)),
            vol_k=float(vol_blk.get("k_vol", 0.05)),
            depth_base_pct=float(depth_blk.get("base_pct", 0.005)),
            depth_k_participation=float(depth_blk.get("k_participation", 0.10)),
            max_slippage_pct=float(slip_blk.get("max_slippage_pct", float("inf"))),
            exit_slippage_multiplier=float(slip_blk.get("exit_slippage_multiplier", 1.0)),
            stop_slippage_multiplier=float(slip_blk.get("stop_slippage_multiplier", 1.0)),
        )

        lat_blk = costs_cfg.get("latency") or {}
        legacy_lat_s = float(exec_cfg.get("simulated_latency_ms", 0.0)) / 1000.0
        if duration_cfg.get("assumed_entry_latency_seconds") is not None and "entry_delay_seconds" not in lat_blk:
            legacy_lat_s = float(duration_cfg["assumed_entry_latency_seconds"])
        legacy_adverse_per_s = float(exec_cfg.get("latency_adverse_pct_per_100ms", 0.0)) * 10.0
        latency = LatencySchedule(
            entry_delay_seconds=max(0.0, float(lat_blk.get("entry_delay_seconds", legacy_lat_s))),
            exit_delay_seconds=max(
                0.0,
                float(lat_blk.get("exit_delay_seconds", exec_cfg.get("exit_delay_seconds", 0.0))),
            ),
            adverse_drift_pct_per_second=max(
                0.0,
                float(lat_blk.get("adverse_drift_pct_per_second", legacy_adverse_per_s)),
            ),
        )

        imp_blk = costs_cfg.get("impact") or {}
        impact = ImpactSchedule(
            enabled=bool(imp_blk.get("enabled", False)),
            k_sqrt=float(imp_blk.get("k_sqrt", 0.0)),
            participation_cap=float(
                imp_blk.get("participation_cap", exec_cfg.get("depth_participation_rate", 0.15))
            ),
            partial_fill_policy=str(
                imp_blk.get("partial_fill_policy", exec_cfg.get("partial_fill_policy", "partial"))
            ),
        )

        carry_blk = costs_cfg.get("carry") or {}
        carry = CarrySchedule(
            enabled=bool(carry_blk.get("enabled", False)),
            funding_interval_hours=float(carry_blk.get("funding_interval_hours", 8.0)),
            default_funding_rate_pct=float(carry_blk.get("default_funding_rate_pct", 0.0)),
            borrow_apr_pct=float(carry_blk.get("borrow_apr_pct", 0.0)),
        )

        rej_blk = costs_cfg.get("rejects") or {}
        rejects = RejectSchedule(
            order_rejection_rate=float(
                rej_blk.get("order_rejection_rate", exec_cfg.get("order_rejection_rate", 0.0))
            ),
            seed=int(rej_blk.get("seed", 42)),
        )

        slippage_buffer_pct = float(labeling_cfg.get("slippage_buffer_pct", 0.0))
        return cls(
            fees=fees,
            spread=spread,
            slippage=slippage,
            latency=latency,
            impact=impact,
            carry=carry,
            rejects=rejects,
            slippage_buffer_pct=slippage_buffer_pct,
            explicit_costs_block=has_explicit,
        )

    @classmethod
    def from_exec_config(cls, exec_cfg: Mapping[str, Any], fee_pct: float = 0.04) -> "CostModel":
        """Build a CostModel when a caller only holds the `execution:` sub-dict and `fee_pct`."""
        if "costs" in exec_cfg:
            return cls.from_config(exec_cfg)
        return cls.from_config({"execution": dict(exec_cfg), "labeling": {"fee_pct": fee_pct}})

    @property
    def entry_delay_seconds(self) -> float:
        return self.latency.entry_delay_seconds

    @property
    def exit_delay_seconds(self) -> float:
        return self.latency.exit_delay_seconds

    def fee_pct(self, leg: str = "entry") -> float:
        role = self.fees.entry_role if leg == "entry" else self.fees.exit_role
        base = self.fees.maker_pct if role == "maker" else self.fees.taker_pct
        return base + self.fees.regulatory_pct

    def round_trip_fee_pct(self) -> float:
        return self.fee_pct("entry") + self.fee_pct("exit")

    def spread_pct(self, market: Optional[MarketSnapshot] = None, leg: str = "entry") -> float:
        if self.spread.source == "fixed" or market is None:
            raw = self.spread.fixed_pct
        else:
            raw = float(market.spread_pct)
        clamped = min(max(raw, self.spread.floor_pct), self.spread.ceiling_pct)
        if leg == "exit":
            clamped *= self.spread.exit_spread_multiplier
        return max(0.0, clamped)

    def slippage_pct(
        self,
        leg: str = "entry",
        *,
        fv: Optional[FeatureVector] = None,
        market: Optional[MarketSnapshot] = None,
        size_usdt: float = 0.0,
        is_stop: bool = False,
    ) -> float:
        model = self.slippage.model
        if model == "volatility_scaled":
            vol = float(fv.volatility_medium) if fv is not None else 0.0
            base = self.slippage.vol_base_pct + self.slippage.vol_k * max(0.0, vol)
        elif model == "depth_based":
            depth_usdt = (
                (market.bid_volume + market.ask_volume) * market.close
                if market is not None and market.close > 0
                else 0.0
            )
            ratio = size_usdt / max(depth_usdt, 1e-6) if depth_usdt > 0 else 1.0
            base = self.slippage.depth_base_pct + self.slippage.depth_k_participation * max(0.0, ratio)
        else:
            base = self.slippage.fixed_pct

        delay_s = self.latency.entry_delay_seconds if leg == "entry" else self.latency.exit_delay_seconds
        latency_drift = delay_s * self.latency.adverse_drift_pct_per_second if delay_s > 0 else 0.0
        total = base + latency_drift

        if leg == "exit":
            total *= self.slippage.exit_slippage_multiplier
        if is_stop:
            total *= self.slippage.stop_slippage_multiplier

        return min(max(0.0, total), self.slippage.max_slippage_pct)

    def impact_pct(self, size_usdt: float, depth_usdt: float) -> float:
        if not self.impact.enabled or size_usdt <= 0:
            return 0.0
        if depth_usdt <= 0:
            return 0.0
        return max(0.0, self.impact.k_sqrt * math.sqrt(size_usdt / depth_usdt))

    def carry_pct(
        self,
        direction: int,
        hold_seconds: float,
        funding_rate_pct: Optional[float] = None,
    ) -> float:
        if not self.carry.enabled or hold_seconds <= 0:
            return 0.0
        hours = hold_seconds / 3600.0
        interval = max(self.carry.funding_interval_hours, 1e-6)
        rate = self.carry.default_funding_rate_pct if funding_rate_pct is None else float(funding_rate_pct)
        # Positive funding rate is paid by longs (+1) and received by shorts (-1)
        funding_cost = (hours / interval) * rate * (1.0 if direction == 1 else -1.0)
        borrow_cost = (hours / (24.0 * 365.0)) * self.carry.borrow_apr_pct
        return funding_cost + borrow_cost

    def opportunity_filter_cost_pct(
        self,
        market: Optional[MarketSnapshot] = None,
        fv: Optional[FeatureVector] = None,
    ) -> float:
        """Round-trip cost estimate used by `OpportunityFilter`."""
        if not self.explicit_costs_block:
            return self.round_trip_fee_pct() + self.slippage_buffer_pct
        if self.slippage_buffer_pct > 0:
            return self.round_trip_fee_pct() + self.slippage_buffer_pct
        return (
            self.round_trip_fee_pct()
            + self.slippage_pct("entry", fv=fv, market=market)
            + self.slippage_pct("exit", fv=fv, market=market)
        )

    def is_non_default(self) -> bool:
        """True when explicit `costs:` settings or exit delays are active."""
        return (
            self.explicit_costs_block
            or self.latency.exit_delay_seconds > 0
            or self.impact.enabled
            or self.carry.enabled
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fees": {
                "maker_pct": round(self.fees.maker_pct, 8),
                "taker_pct": round(self.fees.taker_pct, 8),
                "entry_role": self.fees.entry_role,
                "exit_role": self.fees.exit_role,
                "regulatory_pct": round(self.fees.regulatory_pct, 8),
                "per_contract_fee": round(self.fees.per_contract_fee, 8),
            },
            "spread": {
                "source": self.spread.source,
                "fixed_pct": round(self.spread.fixed_pct, 8),
                "floor_pct": round(self.spread.floor_pct, 8),
                "ceiling_pct": None if math.isinf(self.spread.ceiling_pct) else round(self.spread.ceiling_pct, 8),
                "exit_spread_multiplier": round(self.spread.exit_spread_multiplier, 8),
            },
            "slippage": {
                "model": self.slippage.model,
                "fixed_pct": round(self.slippage.fixed_pct, 8),
                "vol_base_pct": round(self.slippage.vol_base_pct, 8),
                "vol_k": round(self.slippage.vol_k, 8),
                "depth_base_pct": round(self.slippage.depth_base_pct, 8),
                "depth_k_participation": round(self.slippage.depth_k_participation, 8),
                "max_slippage_pct": (
                    None if math.isinf(self.slippage.max_slippage_pct) else round(self.slippage.max_slippage_pct, 8)
                ),
                "exit_slippage_multiplier": round(self.slippage.exit_slippage_multiplier, 8),
                "stop_slippage_multiplier": round(self.slippage.stop_slippage_multiplier, 8),
            },
            "latency": {
                "entry_delay_seconds": round(self.latency.entry_delay_seconds, 6),
                "exit_delay_seconds": round(self.latency.exit_delay_seconds, 6),
                "adverse_drift_pct_per_second": round(self.latency.adverse_drift_pct_per_second, 8),
            },
            "impact": {
                "enabled": self.impact.enabled,
                "k_sqrt": round(self.impact.k_sqrt, 8),
                "participation_cap": round(self.impact.participation_cap, 8),
                "partial_fill_policy": self.impact.partial_fill_policy,
            },
            "carry": {
                "enabled": self.carry.enabled,
                "funding_interval_hours": round(self.carry.funding_interval_hours, 6),
                "default_funding_rate_pct": round(self.carry.default_funding_rate_pct, 8),
                "borrow_apr_pct": round(self.carry.borrow_apr_pct, 8),
            },
        }

    def fingerprint_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
