"""
Feature Engineering Layer: Transform raw market data into features.
CRITICAL: No look-ahead bias. All features use only past data.
Phase 2 added a volume z-score (stored in raw_features) for the regime
detector's HMM observation set, and refactored onto the shared
RollingBarBuffer.

Phase 5: periods are now config-driven via feature_engine.* config block.
Field names are period-agnostic (short/medium/long) to avoid lying about
the actual period when config changes.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..state.contracts import FeatureVector, MarketSnapshot
from ..utils.rolling_buffer import RollingBarBuffer

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute engineered features from price data."""

    def __init__(self, lookback_bars: int = 100, config: Optional[dict] = None):
        self.lookback_bars = lookback_bars
        self.config = config or {}
        self.price_history = RollingBarBuffer(maxlen=lookback_bars)

        # Read periods from config (defaults match previous hardcoded values)
        fe_cfg = self.config.get("feature_engine", self.config.get("features", {}))
        mp = fe_cfg.get("momentum_periods", {})
        self.momentum_short = mp.get("short", 1) if isinstance(mp, dict) else (mp[0] if len(mp) > 0 else 1)
        self.momentum_medium = mp.get("medium", 5) if isinstance(mp, dict) else (mp[1] if len(mp) > 1 else 5)
        self.momentum_long = mp.get("long", 10) if isinstance(mp, dict) else (mp[2] if len(mp) > 2 else 10)
        vp = fe_cfg.get("volatility_periods", {})
        self.volatility_short = vp.get("short", 1) if isinstance(vp, dict) else (vp[0] if len(vp) > 0 else 1)
        self.volatility_medium = vp.get("medium", 5) if isinstance(vp, dict) else (vp[1] if len(vp) > 1 else 5)
        self.volatility_long = vp.get("long", 10) if isinstance(vp, dict) else (vp[2] if len(vp) > 2 else 10)
        self.rsi_period = fe_cfg.get("rsi_period", 14)
        self.sma_periods = fe_cfg.get("sma_periods", [9, 21])
        self.volume_zscore_period = fe_cfg.get("volume_zscore_period", 20)

        # Optional time-horizon window specification (`features.windows_seconds`)
        win_sec = fe_cfg.get("windows_seconds")
        if isinstance(win_sec, dict) and win_sec:
            from ..labeling.hold_policy import parse_granularity_seconds
            gran = (self.config.get("system") or {}).get("data_granularity", "1m")
            bar_s = max(1.0, parse_granularity_seconds(str(gran)))
            if "short" in win_sec:
                self.momentum_short = max(1, int(round(float(win_sec["short"]) / bar_s)))
                self.volatility_short = max(2, int(round(float(win_sec["short"]) / bar_s)))
            if "medium" in win_sec:
                self.momentum_medium = max(2, int(round(float(win_sec["medium"]) / bar_s)))
                self.volatility_medium = max(2, int(round(float(win_sec["medium"]) / bar_s)))
            if "long" in win_sec:
                self.momentum_long = max(3, int(round(float(win_sec["long"]) / bar_s)))
                self.volatility_long = max(3, int(round(float(win_sec["long"]) / bar_s)))
            required_len = max(self.lookback_bars, self.momentum_long + 5, self.volatility_long + 5, 30)
            if required_len > self.lookback_bars:
                self.lookback_bars = required_len
                self.price_history = RollingBarBuffer(maxlen=required_len)

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

        momentum_short = self._momentum(closes, periods=self.momentum_short)
        momentum_medium = self._momentum(closes, periods=self.momentum_medium)

        volatility_short = self._volatility(closes, periods=self.volatility_short)
        volatility_medium = self._volatility(closes, periods=self.volatility_medium)

        returns_5 = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0.0

        rsi_14 = self._rsi(closes, periods=self.rsi_period)
        macd_signal = self._macd_signal(closes)

        sma_9 = self._sma(closes, periods=self.sma_periods[0])
        sma_21 = self._sma(closes, periods=self.sma_periods[1])

        momentum_long = self._momentum(closes, periods=self.momentum_long)
        volatility_long = self._volatility(closes, periods=self.volatility_long)

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
            momentum_short=momentum_short,
            momentum_medium=momentum_medium,
            volatility_short=volatility_short,
            volatility_medium=volatility_medium,
            returns_5=returns_5,
            rsi_14=rsi_14,
            macd_signal=macd_signal,
            sma_9=sma_9,
            sma_21=sma_21,
            momentum_long=momentum_long,
            volatility_long=volatility_long,
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
        window = np.asarray(prices[-26:], dtype=np.float64)
        idx = np.arange(len(window) - 1, -1, -1, dtype=np.float64)
        w1 = (1.0 - 2.0 / 13.0) ** idx
        w2 = (1.0 - 2.0 / 27.0) ** idx
        exp1 = float(np.dot(window, w1) / w1.sum())
        exp2 = float(np.dot(window, w2) / w2.sum())
        return float(exp1 - exp2)

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
