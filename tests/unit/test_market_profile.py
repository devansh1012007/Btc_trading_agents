"""
Unit tests for Phase 15 T15.1: `MarketProfile` presets (`crypto_spot`,
`crypto_perp`, `equity`, `option`), session hours, and lot/tick rounding.
"""
import pytest

from src.market.profile import MarketProfile

pytestmark = [pytest.mark.unit]


def test_market_profile_presets_and_session_calendar():
    spot = MarketProfile.preset("crypto_spot")
    assert spot.is_24_7 and not spot.allow_short

    eq = MarketProfile.preset("equity")
    assert not eq.is_24_7 and eq.settlement_currency == "USD"
    # Wednesday 2023-11-15 15:00 UTC (10:00 AM ET) -> open
    assert eq.is_session_open(1_700_060_400) is True
    # Wednesday 2023-11-15 03:00 UTC (10:00 PM ET) -> closed
    assert eq.is_session_open(1_700_017_200) is False
    # Saturday 2023-11-18 15:00 UTC -> closed
    assert eq.is_session_open(1_700_319_600) is False

    assert eq.round_price(123.456) == pytest.approx(123.46)
    assert eq.round_quantity(17.8) == pytest.approx(17.0)
