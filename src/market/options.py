"""
ATM Option Selector (`ATMSelector`), Black-Scholes-Merton pricing & Greeks
(`black_scholes_price_and_greeks`), synthetic option chain generator
(`SyntheticOptionsChainProvider`), and option-premium `ExitPlan` / cost sizing
(Phase 15 T15.3 / Decision D5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..state.contracts import ExitPlan, TrailSpec


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price_and_greeks(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    sigma: float,
    option_type: str = "call",
) -> Tuple[float, float, float, float, float]:
    """Return `(price, delta, gamma, theta_per_day, vega_per_1pct)` for a European call or put."""
    s = max(float(spot), 1e-6)
    k = max(float(strike), 1e-6)
    t = max(float(t_years), 1e-6)
    vol = max(float(sigma), 1e-4)
    r = float(rate)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = math.exp(-r * t)
    pdf_d1 = _norm_pdf(d1)

    is_call = str(option_type).lower().startswith("c")
    if is_call:
        price = s * _norm_cdf(d1) - k * disc * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = -(s * pdf_d1 * vol) / (2.0 * sqrt_t) - r * k * disc * _norm_cdf(d2)
    else:
        price = k * disc * _norm_cdf(-d2) - s * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta_annual = -(s * pdf_d1 * vol) / (2.0 * sqrt_t) + r * k * disc * _norm_cdf(-d2)

    gamma = pdf_d1 / (s * vol * sqrt_t)
    vega = s * pdf_d1 * sqrt_t * 0.01
    theta_day = theta_annual / 365.0
    return float(max(price, 1e-4)), float(delta), float(gamma), float(theta_day), float(vega)


@dataclass(frozen=True)
class OptionContract:
    option_symbol: str
    underlying_symbol: str
    strike: float
    expiry_timestamp: int
    dte_days: float
    option_type: str  # "call" | "put"
    multiplier: float
    implied_vol: float
    bid: float
    ask: float
    mid: float
    delta: float
    gamma: float
    theta_per_day: float
    vega: float
    open_interest: float = 1000.0

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid * 100.0 if self.mid > 0 else 0.0


class SyntheticOptionsChainProvider:
    """Deterministic Black-Scholes option chain generator anchored to the underlying chart."""

    def __init__(
        self,
        underlying_symbol: str = "SPY",
        multiplier: float = 100.0,
        rate: float = 0.04,
        base_spread_pct: float = 1.2,
    ) -> None:
        self.underlying_symbol = underlying_symbol
        self.multiplier = float(multiplier)
        self.rate = float(rate)
        self.base_spread_pct = float(base_spread_pct)

    def get_chain(
        self,
        timestamp: int,
        spot_price: float,
        implied_vol: float = 0.25,
        dte_days_list: Sequence[int] = (7, 14, 30),
        strike_offsets_pct: Sequence[float] = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
    ) -> List[OptionContract]:
        chain: List[OptionContract] = []
        for dte in dte_days_list:
            t_years = max(float(dte), 0.5) / 365.0
            exp_ts = int(timestamp + int(dte * 86400))
            for off_pct in strike_offsets_pct:
                raw_strike = spot_price * (1.0 + float(off_pct) / 100.0)
                strike = round(raw_strike, 0) if spot_price >= 50 else round(raw_strike, 2)
                for opt_type in ("call", "put"):
                    mid, delta, gamma, theta, vega = black_scholes_price_and_greeks(
                        spot_price, strike, t_years, self.rate, implied_vol, opt_type
                    )
                    half_sp = (self.base_spread_pct / 200.0) * mid
                    bid = max(0.01, mid - half_sp)
                    ask = mid + half_sp
                    sym = f"{self.underlying_symbol}-{dte}D-{int(strike)}-{'C' if opt_type == 'call' else 'P'}"
                    chain.append(
                        OptionContract(
                            option_symbol=sym,
                            underlying_symbol=self.underlying_symbol,
                            strike=float(strike),
                            expiry_timestamp=exp_ts,
                            dte_days=float(dte),
                            option_type=opt_type,
                            multiplier=self.multiplier,
                            implied_vol=float(implied_vol),
                            bid=float(bid),
                            ask=float(ask),
                            mid=float((bid + ask) / 2.0),
                            delta=delta,
                            gamma=gamma,
                            theta_per_day=theta,
                            vega=vega,
                            open_interest=1500.0,
                        )
                    )
        return chain


class ATMSelector:
    """Selects the ATM derivative contract for a main-chart directional signal (`+1` -> Call, `-1` -> Put)."""

    def __init__(
        self,
        *,
        min_dte_days: float = 3.0,
        max_dte_days: float = 45.0,
        target_dte_days: float = 14.0,
        min_abs_delta: float = 0.35,
        max_abs_delta: float = 0.65,
        max_spread_pct: float = 5.0,
        min_open_interest: float = 100.0,
        direction_mapping: str = "long_call_long_put",
    ) -> None:
        self.min_dte_days = float(min_dte_days)
        self.max_dte_days = float(max_dte_days)
        self.target_dte_days = float(target_dte_days)
        self.min_abs_delta = float(min_abs_delta)
        self.max_abs_delta = float(max_abs_delta)
        self.max_spread_pct = float(max_spread_pct)
        self.min_open_interest = float(min_open_interest)
        self.direction_mapping = str(direction_mapping)

    def select_contract(
        self,
        chain: Sequence[OptionContract],
        underlying_direction: int,
        spot_price: float,
    ) -> Optional[OptionContract]:
        if underlying_direction not in (1, -1):
            return None
        desired_type = "call" if underlying_direction == 1 else "put"

        eligible = [
            c
            for c in chain
            if c.option_type == desired_type
            and self.min_dte_days <= c.dte_days <= self.max_dte_days
            and self.min_abs_delta <= abs(c.delta) <= self.max_abs_delta
            and c.spread_pct <= self.max_spread_pct
            and c.open_interest >= self.min_open_interest
        ]
        if not eligible:
            return None

        # Rank by closeness to target DTE and closeness of |delta| to 0.50 (ATM)
        return min(
            eligible,
            key=lambda c: (
                abs(c.dte_days - self.target_dte_days),
                abs(abs(c.delta) - 0.50),
                abs(c.strike - spot_price),
                c.option_symbol,
            ),
        )

    @staticmethod
    def underlying_plan_to_option_premium_plan(
        underlying_plan: ExitPlan,
        contract: OptionContract,
        spot_price: float,
    ) -> ExitPlan:
        """Convert an underlying `%` `ExitPlan` into an option-premium `%` `ExitPlan` (D5:
        `option_stop_reference: option_premium`, capped at 100% max loss for long options)."""
        leverage = max(1.0, abs(contract.delta) * spot_price / max(contract.mid, 1e-4))
        opt_sl = min(99.0, max(1.0, underlying_plan.effective_stop_loss_pct * leverage))
        opt_tp = max(opt_sl * 1.1, underlying_plan.take_profit_pct * leverage)
        opt_hard = (
            min(99.0, underlying_plan.hard_max_loss_pct * leverage)
            if underlying_plan.hard_max_loss_pct is not None
            else None
        )
        opt_trail = None
        if underlying_plan.trailing is not None and underlying_plan.trailing.enabled:
            opt_trail = TrailSpec(
                enabled=True,
                activation_pct=underlying_plan.trailing.activation_pct * leverage,
                distance_pct=min(90.0, underlying_plan.trailing.distance_pct * leverage),
                step_pct=underlying_plan.trailing.step_pct * leverage,
            )
        return ExitPlan(
            stop_loss_pct=float(opt_sl),
            take_profit_pct=float(opt_tp),
            max_holding_bars=underlying_plan.max_holding_bars,
            trailing=opt_trail,
            hard_max_loss_pct=opt_hard,
            source="option_premium",
        )

    @staticmethod
    def compute_option_order_cost_usdt(
        contract: OptionContract,
        n_contracts: int,
        per_contract_fee: float = 0.65,
        slippage_pct: float = 0.5,
    ) -> float:
        """Total entry cost in settlement currency for `n_contracts` of `contract`."""
        n = max(0, int(n_contracts))
        notional = n * contract.mid * contract.multiplier
        half_spread_cost = n * ((contract.ask - contract.bid) / 2.0) * contract.multiplier
        slip_cost = notional * (float(slippage_pct) / 100.0)
        fees = n * float(per_contract_fee)
        return float(half_spread_cost + slip_cost + fees)
