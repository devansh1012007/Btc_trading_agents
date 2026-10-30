"""
Unit tests for Phase 15 T15.2: `OHLCVFileAdapter` and `detect_unadjusted_splits`.
"""
import pandas as pd
import pytest

from src.market.adapters import OHLCVFileAdapter, detect_unadjusted_splits
from src.market.profile import MarketProfile

pytestmark = [pytest.mark.unit]


def test_detect_unadjusted_splits_flags_2_for_1_jump_and_adapter_filters_session(tmp_path):
    df = pd.DataFrame({
        "timestamp": [1_700_060_400, 1_700_060_460, 1_700_017_200],
        "open": [200.0, 100.0, 100.0],
        "high": [202.0, 101.0, 101.0],
        "low": [199.0, 99.0, 99.0],
        "close": [200.0, 100.0, 100.0],
        "volume": [1000.0, 2000.0, 500.0],
    })
    suspects = detect_unadjusted_splits(df, split_jump_threshold_pct=35.0)
    assert len(suspects) == 1
    assert suspects[0].inferred_ratio == "2:1"

    csv_path = tmp_path / "eq.csv"
    df.to_csv(csv_path, index=False)
    adapter = OHLCVFileAdapter("SPY", profile=MarketProfile.preset("equity"), filter_session=True)
    loaded = adapter.load_dataframe(csv_path)
    # Only the 2 in-session bars are kept; the 03:00 UTC bar is filtered out
    assert len(loaded) == 2
