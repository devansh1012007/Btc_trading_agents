"""
Tighten-Only Autonomous Risk Governor (`RiskGovernor`, Phase 11 T11.3 / C15 / ADR-0004).

Inspects portfolio drawdown, loss streak, volatility spike, and model
degradation alarms, and emits a `GovernorDirective`.

Hard Invariant (C15 / ADR-0004):
* `size_multiplier` is clamped to `[0.0, 1.0]` (never > 1.0)
* `max_concurrent_cap` is clamped to `[1, configured_max_concurrent]` (never > configured)
* `extra_cooldown_multiplier` is clamped to `>= 1.0` (never shortens a cooldown)
* Can trigger a time-boxed halt on `HaltManager`, but can NEVER clear or shorten an active halt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GovernorDirective:
    size_multiplier: float = 1.0
    max_concurrent_cap: int = 1
    extra_cooldown_multiplier: float = 1.0
    trigger_halt_minutes: float = 0.0
    veto: bool = False
    reason: Optional[str] = None


class RiskGovernor:
    """Autonomous tighten-only risk governor."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        gov_cfg = (cfg.get("risk") or {}).get("governor") or cfg.get("governor") or cfg
        self.enabled = bool(gov_cfg.get("enabled", False))
        self.soft_drawdown_pct = float(gov_cfg.get("soft_drawdown_pct", 3.0))
        self.hard_drawdown_pct = float(gov_cfg.get("hard_drawdown_pct", 6.0))
        self.loss_streak_tighten = int(gov_cfg.get("loss_streak_tighten", 2))
        self.loss_streak_halt = int(gov_cfg.get("loss_streak_halt", 4))
        self.halt_minutes_on_trigger = float(gov_cfg.get("halt_minutes_on_trigger", 120.0))
        self.min_size_multiplier = float(gov_cfg.get("min_size_multiplier", 0.25))

    def evaluate(
        self,
        *,
        drawdown_pct: float,
        consecutive_losses: int,
        configured_max_concurrent: int = 1,
        volatility_zscore: float = 0.0,
        model_degraded: bool = False,
        requested_size_multiplier: Optional[float] = None,
        requested_max_concurrent: Optional[int] = None,
        requested_cooldown_multiplier: Optional[float] = None,
    ) -> GovernorDirective:
        base_conc = max(1, int(configured_max_concurrent))
        if not self.enabled:
            return GovernorDirective(
                size_multiplier=1.0,
                max_concurrent_cap=base_conc,
                extra_cooldown_multiplier=1.0,
                trigger_halt_minutes=0.0,
                veto=False,
                reason=None,
            )

        size_mult = 1.0
        conc_cap = base_conc
        cd_mult = 1.0
        halt_mins = 0.0
        veto = False
        reasons = []

        # Tighten on drawdown
        if drawdown_pct >= self.hard_drawdown_pct:
            veto = True
            size_mult = 0.0
            conc_cap = 1
            cd_mult = max(cd_mult, 2.0)
            halt_mins = max(halt_mins, self.halt_minutes_on_trigger)
            reasons.append(f"hard_drawdown({drawdown_pct:.2f}%)")
        elif drawdown_pct >= self.soft_drawdown_pct:
            span = max(self.hard_drawdown_pct - self.soft_drawdown_pct, 1e-6)
            frac = min(1.0, (drawdown_pct - self.soft_drawdown_pct) / span)
            size_mult = min(size_mult, max(self.min_size_multiplier, 1.0 - 0.5 * frac))
            conc_cap = min(conc_cap, max(1, base_conc - 1))
            cd_mult = max(cd_mult, 1.5)
            reasons.append(f"soft_drawdown({drawdown_pct:.2f}%)")

        # Tighten on consecutive losses
        if self.loss_streak_halt > 0 and consecutive_losses >= self.loss_streak_halt:
            veto = True
            size_mult = 0.0
            conc_cap = 1
            cd_mult = max(cd_mult, 2.0)
            halt_mins = max(halt_mins, self.halt_minutes_on_trigger)
            reasons.append(f"loss_streak_halt({consecutive_losses})")
        elif self.loss_streak_tighten > 0 and consecutive_losses >= self.loss_streak_tighten:
            size_mult = min(size_mult, 0.5)
            conc_cap = min(conc_cap, max(1, base_conc // 2))
            cd_mult = max(cd_mult, 1.5)
            reasons.append(f"loss_streak_tighten({consecutive_losses})")

        # Tighten on volatility spike or model degradation alarm
        if volatility_zscore >= 2.5:
            size_mult = min(size_mult, 0.5)
            conc_cap = min(conc_cap, 1)
            reasons.append(f"vol_spike(z={volatility_zscore:.2f})")
        if model_degraded:
            size_mult = min(size_mult, 0.5)
            cd_mult = max(cd_mult, 1.5)
            reasons.append("model_degraded")

        # Enforce strict tighten-only invariants even if external proposals are passed
        if requested_size_multiplier is not None:
            size_mult = min(size_mult, float(requested_size_multiplier))
        if requested_max_concurrent is not None:
            conc_cap = min(conc_cap, int(requested_max_concurrent))
        if requested_cooldown_multiplier is not None:
            cd_mult = max(cd_mult, float(requested_cooldown_multiplier))

        size_mult = max(0.0, min(1.0, float(size_mult)))
        conc_cap = max(1, min(base_conc, int(conc_cap)))
        cd_mult = max(1.0, float(cd_mult))

        reason_str = ("governor_veto:" + ",".join(reasons)) if veto else (",".join(reasons) if reasons else None)
        return GovernorDirective(
            size_multiplier=size_mult,
            max_concurrent_cap=conc_cap,
            extra_cooldown_multiplier=cd_mult,
            trigger_halt_minutes=halt_mins,
            veto=veto,
            reason=reason_str,
        )
