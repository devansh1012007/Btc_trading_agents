"""
Feature Engineering Layer: Transform raw market data into features.
CRITICAL: No look-ahead bias. All features use only past data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..state.contracts import FeatureVector, MarketSnapshot

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute engineered features from price data."""
    
    def __init__(self, lookback_bars: int = 100):
        """
        Initialize feature engine.
        
        Args:
            lookback_bars: How many bars to keep for feature calculation
        """
        self.lookback_bars = lookback_bars
        self.price_history = []  # Rolling window of (timestamp, close, volume, spread)
    
    def add_bar(self, snapshot: MarketSnapshot) -> None:
        """Add a new bar to rolling window."""
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'open': snapshot.open,
            'high': snapshot.high,
            'low': snapshot.low,
            'close': snapshot.close,
            'volume': snapshot.volume,
            'spread_pct': snapshot.spread_pct,
        })
        
        # Keep only lookback_bars
        if len(self.price_history) > self.lookback_bars:
            self.price_history.pop(0)
    
    def compute_features(self, snapshot: MarketSnapshot) -> Optional[FeatureVector]:
        """
        Compute all features for current bar.
        Returns None if not enough history.
        """
        if len(self.price_history) < 10:
            logger.warning(f"Not enough history: {len(self.price_history)} bars")
            return None
        
        # Convert to arrays for calculation
        closes = np.array([b['close'] for b in self.price_history])
        volumes = np.array([b['volume'] for b in self.price_history])
        
        # ===== SHORT-HORIZON FEATURES (30s-2m) =====
        
        # Momentum
        momentum_30s = self._momentum(closes, periods=1)  # Last 1 bar
        momentum_5m = self._momentum(closes, periods=5)
        
        # Volatility
        volatility_30s = self._volatility(closes, periods=1)
        volatility_5m = self._volatility(closes, periods=5)
        
        # Returns
        returns_5 = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0.0
        
        # RSI
        rsi_14 = self._rsi(closes, periods=14)
        
        # MACD
        macd_signal = self._macd_signal(closes)
        
        # ===== MEDIUM-HORIZON FEATURES (5-10m) =====
        
        # Moving averages
        sma_9 = self._sma(closes, periods=9)
        sma_21 = self._sma(closes, periods=21)
        
        # Momentum medium-term
        momentum_10m = self._momentum(closes, periods=10)
        volatility_10m = self._volatility(closes, periods=10)
        
        # ===== ORDERFLOW FEATURES =====
        
        # Simple imbalance (bid vs ask volume)
        total_depth = snapshot.bid_volume + snapshot.ask_volume
        if total_depth > 0:
            imbalance_ratio = snapshot.ask_volume / total_depth
        else:
            imbalance_ratio = 0.5  # Neutral when no data
        
        # Orderflow momentum (approximated from volume)
        orderflow_momentum = self._orderflow_momentum(closes, volumes)
        
        # ===== LIQUIDITY FEATURES =====
        spread_pct = snapshot.spread_pct
        depth_10_level = snapshot.bid_volume + snapshot.ask_volume
        
        # ===== STRUCTURAL FEATURES =====
        # (Will be filled by zone engine later)
        
        # Create feature vector
        fv = FeatureVector(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            momentum_30s=momentum_30s,
            momentum_5m=momentum_5m,
            volatility_30s=volatility_30s,
            volatility_5m=volatility_5m,
            returns_5=returns_5,
            rsi_14=rsi_14,
            macd_signal=macd_signal,
            sma_9=sma_9,
            sma_21=sma_21,
            momentum_10m=momentum_10m,
            volatility_10m=volatility_10m,
            imbalance_ratio=imbalance_ratio,
            orderflow_momentum=orderflow_momentum,
            spread_pct=spread_pct,
            depth_10_level=depth_10_level,
        )
        
        return fv
    
    # ===== HELPER FUNCTIONS =====
    
    @staticmethod
    def _momentum(prices: np.ndarray, periods: int = 5) -> float:
        """Compute momentum (% change over N periods)."""
        if len(prices) < periods + 1:
            return 0.0
        return float((prices[-1] / prices[-(periods + 1)] - 1) * 100)
    
    @staticmethod
    def _volatility(prices: np.ndarray, periods: int = 5) -> float:
        """Compute volatility (std dev of returns)."""
        if len(prices) < periods + 1:
            return 0.0
        returns = np.diff(prices[-(periods+1):]) / prices[-(periods+1):-1]
        volatility = np.std(returns) * 100
        return float(volatility) if not np.isnan(volatility) else 0.0
    
    @staticmethod
    def _sma(prices: np.ndarray, periods: int = 9) -> float:
        """Simple moving average."""
        if len(prices) < periods:
            return prices[-1]
        return float(np.mean(prices[-periods:]))
    
    @staticmethod
    def _rsi(prices: np.ndarray, periods: int = 14) -> float:
        """Relative Strength Index."""
        if len(prices) < periods + 1:
            return 50.0
        
        deltas = np.diff(prices[-periods:])
        seed = deltas[:periods]
        up = seed[seed >= 0].sum() / periods
        down = -seed[seed < 0].sum() / periods
        rs = up / (down + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)
    
    @staticmethod
    def _macd_signal(prices: np.ndarray) -> float:
        """MACD signal line (simplified)."""
        if len(prices) < 26:
            return 0.0
        exp1 = pd.Series(prices[-26:]).ewm(span=12).mean().iloc[-1]
        exp2 = pd.Series(prices[-26:]).ewm(span=26).mean().iloc[-1]
        macd = exp1 - exp2
        return float(macd)
    
    @staticmethod
    def _orderflow_momentum(closes: np.ndarray, volumes: np.ndarray) -> float:
        """Simple orderflow momentum from volume."""
        if len(volumes) < 5:
            return 0.0
        # Up volume vs down volume
        up_volume = np.sum(np.diff(closes[-5:]) > 0) * np.mean(volumes[-5:])
        down_volume = np.sum(np.diff(closes[-5:]) < 0) * np.mean(volumes[-5:])
        total = up_volume + down_volume + 1e-8
        return float((up_volume - down_volume) / total * 100)
