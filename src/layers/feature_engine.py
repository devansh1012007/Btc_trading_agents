"""
Feature Engineering Layer: Transform raw market data into features.
CRITICAL: No look-ahead bias. All features use only past data.
Phase 2 added a volume z-score (stored in raw_features) for the regime
detector's HMM observation set, and refactored onto the shared
RollingBarBuffer.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..state.contracts import FeatureVector, MarketSnapshot
from ..utils.rolling_buffer import RollingBarBuffer

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute engineered features from price data."""

    def __init__(self, lookback_bars: int = 100):
        self.lookback_bars = lookback_bars
        self.price_history = RollingBarBuffer(maxlen=lookback_bars)

    def add_bar(self, snapshot: MarketSnapshot) -> None:
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'open': snapshot.open,
            'high': snapshot.high,
            'low': snapshot.low,
            'close': snapshot.close,
            'volume': snapshot.volume,
            'spread_pct': snapshot.spread_pct,
        })

    def compute_features(self, snapshot: MarketSnapshot) -> Optional[FeatureVector]:
        if len(self.price_history) < 10:
            logger.warning(f"Not enough history: {len(self.price_history)} bars")
            return None

        closes = np.array([b['close'] for b in self.price_history])
        volumes = np.array([b['volume'] for b in self.price_history])

        momentum_30s = self._momentum(closes, periods=1)
        momentum_5m = self._momentum(closes, periods=5)

        volatility_30s = self._volatility(closes, periods=1)
        volatility_5m = self._volatility(closes, periods=5)

        returns_5 = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0.0

        rsi_14 = self._rsi(closes, periods=14)
        macd_signal = self._macd_signal(closes)

        sma_9 = self._sma(closes, periods=9)
        sma_21 = self._sma(closes, periods=21)

        momentum_10m = self._momentum(closes, periods=10)
        volatility_10m = self._volatility(closes, periods=10)

        total_depth = snapshot.bid_volume + snapshot.ask_volume
        if total_depth > 0:
            imbalance_ratio = snapshot.ask_volume / total_depth
        else:
            imbalance_ratio = 0.5

        orderflow_momentum = self._orderflow_momentum(closes, volumes)

        spread_pct = snapshot.spread_pct
        depth_10_level = snapshot.bid_volume + snapshot.ask_volume

        volume_zscore = self._volume_zscore(volumes)

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
            raw_features={"volume_zscore": volume_zscore},
        )

        return fv

    @staticmethod
    def _momentum(prices: np.ndarray, periods: int = 5) -> float:
        if len(prices) < periods + 1:
            return 0.0
        return float((prices[-1] / prices[-(periods + 1)] - 1) * 100)

    @staticmethod
    def _volatility(prices: np.ndarray, periods: int = 5) -> float:
        if len(prices) < periods + 1:
            return 0.0
        returns = np.diff(prices[-(periods+1):]) / prices[-(periods+1):-1]
        volatility = np.std(returns) * 100
        return float(volatility) if not np.isnan(volatility) else 0.0

    @staticmethod
    def _sma(prices: np.ndarray, periods: int = 9) -> float:
        if len(prices) < periods:
            return prices[-1]
        return float(np.mean(prices[-periods:]))

    @staticmethod
    def _rsi(prices: np.ndarray, periods: int = 14) -> float:
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
        if len(prices) < 26:
            return 0.0
        exp1 = pd.Series(prices[-26:]).ewm(span=12).mean().iloc[-1]
        exp2 = pd.Series(prices[-26:]).ewm(span=26).mean().iloc[-1]
        macd = exp1 - exp2
        return float(macd)

    @staticmethod
    def _orderflow_momentum(closes: np.ndarray, volumes: np.ndarray) -> float:
        if len(volumes) < 5:
            return 0.0
        up_volume = np.sum(np.diff(closes[-5:]) > 0) * np.mean(volumes[-5:])
        down_volume = np.sum(np.diff(closes[-5:]) < 0) * np.mean(volumes[-5:])
        total = up_volume + down_volume + 1e-8
        return float((up_volume - down_volume) / total * 100)

    @staticmethod
    def _volume_zscore(volumes: np.ndarray, periods: int = 20) -> float:
        window = volumes[-periods:] if len(volumes) >= periods else volumes
        if len(window) < 2:
            return 0.0
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-9:
            return 0.0
        return float((volumes[-1] - mean) / std)
