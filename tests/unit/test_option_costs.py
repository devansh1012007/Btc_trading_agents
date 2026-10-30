"""
Unit tests for Phase 15 T15.3: option-premium `ExitPlan` conversion and
per-contract fee + spread cost calculation.
"""
import pytest

from src.market.options import ATMSelector, SyntheticOptionsChainProvider
from src.state.contracts import ExitPlan, TrailSpec

pytestmark = [pytest.mark.unit]


def test_underlying_plan_to_option_premium_plan_and_contract_costs():
    provider = SyntheticOptionsChainProvider("SPY", multiplier=100.0)
    chain = provider.get_chain(timestamp=1_700_000_000, spot_price=400.0, implied_vol=0.22)
    selector = ATMSelector(target_dte_days=14.0)
    contract = selector.select_contract(chain, underlying_direction=1, spot_price=400.0)
    assert contract is not None

    u_plan = ExitPlan(
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_holding_bars=6,
        trailing=TrailSpec(enabled=True, activation_pct=1.0, distance_pct=0.5, step_pct=0.1),
        hard_max_loss_pct=0.8,
    )
    opt_plan = ATMSelector.underlying_plan_to_option_premium_plan(u_plan, contract, spot_price=400.0)
    assert opt_plan.source == "option_premium"
    # Option stop loss is scaled by option delta leverage and strictly capped below 100% of premium
    assert 10.0 < opt_plan.effective_stop_loss_pct <= 99.0
    assert opt_plan.take_profit_pct > opt_plan.stop_loss_pct

    cost_usdt = ATMSelector.compute_option_order_cost_usdt(contract, n_contracts=2, per_contract_fee=0.65, slippage_pct=0.5)
    assert cost_usdt > 2 * 0.65
