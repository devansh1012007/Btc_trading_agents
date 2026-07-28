"""
PHASE 1 unit tests.
Verify: state contracts, data loading, feature computation.
"""
import pytest
from datetime import datetime
import numpy as np
import pandas as pd

from src.state.contracts import (
    MarketSnapshot, FeatureVector, SystemState,
    RegimeType, ExpertAction
)
from src.layers.market_data import MarketDataLayer
from src.layers.feature_engine import FeatureEngine
from src.connectors.data_source import CCXTConnector


class TestStateContracts:
    """Test state schema."""
    
    def test_market_snapshot_creation(self):
        """MarketSnapshot can be created and serialized."""
        snapshot = MarketSnapshot(
            timestamp=1000000,
            symbol="BTC/USDT",
            open=50000,
            high=50500,
            low=49900,
            close=50200,
            volume=100,
            bid=50190,
            ask=50210,
            bid_volume=5,
            ask_volume=5,
            spread_pct=0.04,
        )
        assert snapshot.timestamp == 1000000
        assert snapshot.close == 50200
        
        # Serialization
        d = snapshot.to_dict()
        assert d['close'] == 50200
    
    def test_system_state_immutable(self):
        """SystemState is frozen (immutable)."""
        snapshot = MarketSnapshot(
            timestamp=1000000,
            symbol="BTC/USDT",
            open=50000,
            high=50500,
            low=49900,
            close=50200,
            volume=100,
            bid=50190,
            ask=50210,
            bid_volume=5,
            ask_volume=5,
            spread_pct=0.04,
        )
        
        state = SystemState(
            timestamp=1000000,
            symbol="BTC/USDT",
            market_snapshot=snapshot,
        )
        
        # Try to modify (should fail)
        with pytest.raises(Exception):
            state.timestamp = 2000000


class TestMarketDataLayer:
    """Test market snapshot creation."""
    
    def test_create_from_bar(self):
        """Create snapshot from bar data."""
        ohlcv = {
            'open': 50000,
            'high': 50500,
            'low': 49900,
            'close': 50200,
            'volume': 100,
        }
        
        snapshot = MarketDataLayer.create_from_bar(
            symbol="BTC/USDT",
            timestamp_unix=1000000,
            ohlcv_row=ohlcv,
        )
        
        assert snapshot.close == 50200
        assert snapshot.volume == 100
        assert snapshot.spread_pct > 0


class TestFeatureEngine:
    """Test feature computation."""
    
    def test_momentum_calculation(self):
        """Momentum calculation is correct."""
        prices = np.array([100, 102, 104, 106, 108, 110])
        momentum = FeatureEngine._momentum(prices, periods=5)
        
        expected = (110 / 100 - 1) * 100  # 10%
        assert abs(momentum - expected) < 0.01
    
    def test_feature_vector_creation(self):
        """FeatureVector can be created."""
        fv = FeatureVector(
            timestamp=1000000,
            symbol="BTC/USDT",
            momentum_30s=2.5,
            momentum_5m=1.2,
            volatility_30s=0.8,
            volatility_5m=1.1,
            returns_5=0.5,
            rsi_14=55.0,
            macd_signal=10.0,
            sma_9=50100,
            sma_21=50050,
            momentum_10m=0.3,
            volatility_10m=1.5,
            imbalance_ratio=0.5,
            orderflow_momentum=1.0,
            spread_pct=0.04,
            depth_10_level=100000,
        )
        
        assert fv.momentum_30s == 2.5
        assert fv.timestamp == 1000000
    
    def test_feature_engine_requires_history(self):
        """FeatureEngine returns None without sufficient history."""
        engine = FeatureEngine(lookback_bars=100)
        
        snapshot = MarketSnapshot(
            timestamp=1000000,
            symbol="BTC/USDT",
            open=50000,
            high=50500,
            low=49900,
            close=50200,
            volume=100,
            bid=50190,
            ask=50210,
            bid_volume=5,
            ask_volume=5,
            spread_pct=0.04,
        )
        
        engine.add_bar(snapshot)
        fv = engine.compute_features(snapshot)
        
        assert fv is None  # Not enough history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
