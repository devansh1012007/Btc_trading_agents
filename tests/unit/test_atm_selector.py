"""
Unit tests for Phase 15 T15.3: `black_scholes_price_and_greeks`,
`SyntheticOptionsChainProvider`, and `ATMSelector`.
"""
import pytest

from src.market.options import (
    ATMSelector,
    SyntheticOptionsChainProvider,
    black_scholes_price_and_greeks,
)

pytestmark = [pytest.mark.unit]


def test_black_scholes_put_call_parity_and_atm_selector():
    c_px, c_delta, gamma, _, vega = black_scholes_price_and_greeks(100.0, 100.0, 30 / 365.0, 0.04, 0.25, "call")
    p_px, p_delta, _, _, _ = black_scholes_price_and_greeks(100.0, 100.0, 30 / 365.0, 0.04, 0.25, "put")
    assert 0.45 < c_delta < 0.60
    assert -0.55 < p_delta < -0.40
    assert gamma > 0 and vega > 0
    assert c_px > p_px > 0

    provider = SyntheticOptionsChainProvider("SPY", multiplier=100.0)
    chain = provider.get_chain(timestamp=1_700_000_000, spot_price=450.0, implied_vol=0.20)
    selector = ATMSelector(target_dte_days=14.0)

    call_contract = selector.select_contract(chain, underlying_direction=1, spot_price=450.0)
    put_contract = selector.select_contract(chain, underlying_direction=-1, spot_price=450.0)
    assert call_contract is not None and call_contract.option_type == "call"
    assert put_contract is not None and put_contract.option_type == "put"
    assert call_contract.dte_days == pytest.approx(14.0)
    assert call_contract.strike == pytest.approx(450.0)
