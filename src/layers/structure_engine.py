"""
Dynamic Structure Engine: Detect and maintain support/resistance zones.

Phase 2 completes this engine (§6): price-action interaction detection,
pivot-confirmation look-ahead fix (confirmed_at_bar), Beta-Bernoulli
survival update, regime_relevance tagging, and optional zone-survival
model blending. Phase 3 adds a data_granularity guard (§4.7) on the
zone-survival model load.
"""
import logging
import pickle
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..state.contracts import Zone, ZoneMap, MarketSnapshot, FeatureVector, RegimeState
from ..utils.rolling_buffer import RollingBarBuffer
from .zone_survival_features import zone_survival_features, features_to_row

logger = logging.getLogger(__name__)


class DynamicStructureEngine:
    """Identify and manage support/resistance zones."""

    def __init__(
        self,
        lookback_bars: int = 50,
        pivot_window: int = 5,
        min_zone_separation_pct: float = 1.0,
        zone_decay_bars: int = 100,
        breakout_survival_penalty: float = 0.3,
        zone_survival_model_path: Optional[str] = None,
        zone_survival_blend_weight: float = 0.0,
        data_granularity: str = "1m",
    ):
        """
        Args:
            data_granularity: current bar granularity; compared against
                the zone-survival model's recorded training granularity
                at load time (Phase 3 §4.7) -- a mismatch is treated as
                if the model file were absent.
        """
        self.lookback_bars = lookback_bars
        self.pivot_window = pivot_window
        self.min_zone_separation_pct = min_zone_separation_pct
        self.zone_decay_bars = zone_decay_bars
        self.breakout_survival_penalty = breakout_survival_penalty
        self.zone_survival_blend_weight = zone_survival_blend_weight
        self.data_granularity = data_granularity

        self.price_history = RollingBarBuffer(maxlen=lookback_bars)
        self.zones: Dict[str, Zone] = {}
        self.bar_counter = 0

        self._confirmed_at_bar: Dict[str, int] = {}
        self._latest_regime_state: Optional[RegimeState] = None
        self.touch_events: List[Dict] = []

        self._zone_survival_model = None
        self._zone_survival_feature_names = None
        if zone_survival_model_path:
            self._load_zone_survival_model(zone_survival_model_path)

    def add_bar(self, snapshot: MarketSnapshot) -> None:
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'high': snapshot.high,
            'low': snapshot.low,
            'close': snapshot.close,
            'volume': snapshot.volume,
            'hour_utc': snapshot.hour_utc,
            'day_of_week': snapshot.day_of_week,
        })
        self.bar_counter += 1

    def compute_zones(
        self,
        current_price: float,
        symbol: str = "BTC/USDT",
        regime_state: Optional[RegimeState] = None,
    ) -> ZoneMap:
        self._latest_regime_state = regime_state

        if len(self.price_history) < self.pivot_window + 1:
            return ZoneMap(
                timestamp=int(self.price_history[-1]['timestamp']) if len(self.price_history) else 0,
                symbol=symbol,
                current_price=current_price,
                zones=[],
            )

        self._generate_zones_from_pivots()
        self._generate_zones_from_volume()

        self._process_price_action()

        self._update_zone_ages()
        self._decay_untouched_zones()
        self._merge_nearby_zones(current_price)

        active_zones = [z for z in self.zones.values() if z.survival_probability > 0.1]
        active_zones.sort(key=lambda z: abs(z.mid - current_price))

        support_zones = [z for z in active_zones if z.zone_type == "support" and z.mid < current_price]
        resistance_zones = [z for z in active_zones if z.zone_type == "resistance" and z.mid > current_price]

        nearest_support = support_zones[-1] if support_zones else None
        nearest_resistance = resistance_zones[0] if resistance_zones else None

        return ZoneMap(
            timestamp=int(self.price_history[-1]['timestamp']),
            symbol=symbol,
            current_price=current_price,
            zones=active_zones,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

    # ------------------------------------------------------------------
    # §6.1 Price-action interaction detector
    # ------------------------------------------------------------------

    def _process_price_action(self) -> None:
        if len(self.price_history) == 0:
            return
        bar = self.price_history[-1]

        for zone_id, zone in list(self.zones.items()):
            touched = bar['low'] <= zone.upper_bound and bar['high'] >= zone.lower_bound
            if not touched:
                continue

            broke_through = (
                (zone.zone_type == "resistance" and bar['close'] > zone.upper_bound) or
                (zone.zone_type == "support" and bar['close'] < zone.lower_bound)
            )
            self._apply_touch_outcome(zone_id, broke_through)

    def _apply_touch_outcome(self, zone_id: str, broke_through: bool) -> None:
        zone = self.zones.get(zone_id)
        if zone is None:
            return

        self._log_touch_event(zone_id, zone, "breakout" if broke_through else "rejection")

        rejection_count = zone.rejection_count + (0 if broke_through else 1)
        breakout_count = zone.breakout_count + (1 if broke_through else 0)
        rejection_score = self._beta_bernoulli_rejection_score(rejection_count, breakout_count)

        rule_based_survival = rejection_score * self._time_decay_factor(zone)
        if broke_through:
            rule_based_survival *= self.breakout_survival_penalty

        survival_probability = self._blend_survival_probability(zone_id, rule_based_survival)

        updated = replace(
            zone,
            touch_count=zone.touch_count + 1,
            rejection_count=rejection_count,
            breakout_count=breakout_count,
            rejection_score=rejection_score,
            breakout_score=1.0 - rejection_score,
            survival_probability=survival_probability,
            last_touched_at=self.bar_counter,
        )
        self.zones[zone_id] = updated
        self._update_regime_relevance(zone_id, self._latest_regime_state)

    def _record_rejection(self, zone_id: str) -> None:
        self._apply_touch_outcome(zone_id, broke_through=False)

    def _record_breakout(self, zone_id: str) -> None:
        self._apply_touch_outcome(zone_id, broke_through=True)

    def _log_touch_event(self, zone_id: str, zone: Zone, event_type: str) -> None:
        bar = self.price_history[-1] if len(self.price_history) else {}
        self.touch_events.append({
            "bar_index": self.bar_counter,
            "timestamp": bar.get("timestamp"),
            "zone_id": zone_id,
            "event_type": event_type,
            "features": zone_survival_features(
                zone=zone,
                recent_volatility_pct=self._recent_volatility_pct(),
                hour_utc=bar.get("hour_utc", 0),
                day_of_week=bar.get("day_of_week", 0),
                regime_state=self._latest_regime_state,
            ),
        })

    @staticmethod
    def _beta_bernoulli_rejection_score(rejection_count: int, breakout_count: int) -> float:
        return (rejection_count + 1) / (rejection_count + breakout_count + 2)

    def _time_decay_factor(self, zone: Zone) -> float:
        bars_since_touch = self.bar_counter - zone.last_touched_at
        if bars_since_touch <= self.zone_decay_bars:
            return 1.0
        decay_factor = 1.0 - (bars_since_touch / (self.zone_decay_bars * 2))
        return max(decay_factor, 0.1)

    def _recent_volatility_pct(self, periods: int = 20) -> float:
        if len(self.price_history) < 2:
            return 0.0
        closes = np.array([b['close'] for b in self.price_history])
        window = closes[-periods:] if len(closes) >= periods else closes
        if len(window) < 2:
            return 0.0
        returns = np.diff(window) / window[:-1]
        vol = np.std(returns) * 100
        return float(vol) if not np.isnan(vol) else 0.0

    # ------------------------------------------------------------------
    # §6.5 Zone survival model blending
    # ------------------------------------------------------------------

    def _load_zone_survival_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info(f"No zone survival model found at {p}; using pure rule-based survival probability")
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load zone survival model at {p}: {e}")
            return

        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"Zone survival model at {p} trained on granularity={trained_granularity}, "
                f"running on {self.data_granularity}. Treating as absent (Phase 3 §4.7)."
            )
            return

        self._zone_survival_model = payload["model"]
        self._zone_survival_feature_names = payload.get("feature_names")
        logger.info(f"Loaded zone survival model from {p}")

    def _blend_survival_probability(self, zone_id: str, rule_based_survival: float) -> float:
        if self._zone_survival_model is None or self.zone_survival_blend_weight <= 0:
            return rule_based_survival

        zone = self.zones.get(zone_id)
        if zone is None:
            return rule_based_survival

        bar = self.price_history[-1] if len(self.price_history) else {}
        features = zone_survival_features(
            zone=zone,
            recent_volatility_pct=self._recent_volatility_pct(),
            hour_utc=bar.get('hour_utc', 0),
            day_of_week=bar.get('day_of_week', 0),
            regime_state=self._latest_regime_state,
        )
        row = features_to_row(features)

        try:
            ml_prediction = float(self._zone_survival_model.predict_proba(row)[0][1])
        except Exception as e:
            logger.warning(f"Zone survival model inference failed, using rule-based only: {e}")
            return rule_based_survival

        w = self.zone_survival_blend_weight
        return (1 - w) * rule_based_survival + w * ml_prediction

    # ------------------------------------------------------------------
    # §6.6 regime_relevance
    # ------------------------------------------------------------------

    def _update_regime_relevance(self, zone_id: str, regime_state: Optional[RegimeState]) -> None:
        if regime_state is None:
            return
        zone = self.zones.get(zone_id)
        if zone is None:
            return
        regime_key = regime_state.primary_regime.value
        relevance = dict(zone.regime_relevance)
        relevance[regime_key] = relevance.get(regime_key, 0.0) + 1.0
        self.zones[zone_id] = replace(zone, regime_relevance=relevance)

    # ------------------------------------------------------------------
    # §6.2 confirmed_at_bar accessor
    # ------------------------------------------------------------------

    def get_confirmed_at_bar(self, zone_id: str) -> Optional[int]:
        return self._confirmed_at_bar.get(zone_id)

    # ------------------------------------------------------------------
    # Zone generation
    # ------------------------------------------------------------------

    def _generate_zones_from_pivots(self) -> None:
        if len(self.price_history) < self.pivot_window + 1:
            return

        highs = np.array([b['high'] for b in self.price_history])
        lows = np.array([b['low'] for b in self.price_history])
        timestamps = np.array([b['timestamp'] for b in self.price_history])

        for i in range(self.pivot_window, len(highs) - self.pivot_window):
            if highs[i] == np.max(highs[i - self.pivot_window:i + self.pivot_window + 1]):
                zone_price = highs[i]
                zone_id = f"pivot_high_{timestamps[i]}"
                if zone_id not in self.zones:
                    zone_width = zone_price * 0.5 / 100
                    self._add_or_merge_zone(
                        zone_id=zone_id, lower=zone_price - zone_width, upper=zone_price + zone_width,
                        zone_type="resistance", created_at=timestamps[i],
                    )

        for i in range(self.pivot_window, len(lows) - self.pivot_window):
            if lows[i] == np.min(lows[i - self.pivot_window:i + self.pivot_window + 1]):
                zone_price = lows[i]
                zone_id = f"pivot_low_{timestamps[i]}"
                if zone_id not in self.zones:
                    zone_width = zone_price * 0.5 / 100
                    self._add_or_merge_zone(
                        zone_id=zone_id, lower=zone_price - zone_width, upper=zone_price + zone_width,
                        zone_type="support", created_at=timestamps[i],
                    )

    def _generate_zones_from_volume(self) -> None:
        if len(self.price_history) < 10:
            return

        closes = np.array([b['close'] for b in self.price_history])
        volumes = np.array([b['volume'] for b in self.price_history])
        timestamps = np.array([b['timestamp'] for b in self.price_history])

        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        for i in range(max(0, len(volumes) - 10), len(volumes)):
            if volumes[i] > avg_volume * 1.5:
                zone_price = closes[i]
                zone_id = f"volume_zone_{timestamps[i]}"
                if zone_id not in self.zones:
                    zone_width = zone_price * 1.0 / 100
                    zone_type = "support" if i % 2 == 0 else "resistance"
                    self._add_or_merge_zone(
                        zone_id=zone_id, lower=zone_price - zone_width, upper=zone_price + zone_width,
                        zone_type=zone_type, created_at=timestamps[i],
                    )

    def _add_or_merge_zone(self, zone_id: str, lower: float, upper: float, zone_type: str, created_at: int) -> None:
        mid = (lower + upper) / 2

        for existing_id, existing_zone in list(self.zones.items()):
            if existing_zone.zone_type != zone_type:
                continue
            distance_pct = abs(existing_zone.mid - mid) / mid * 100 if mid != 0 else 0
            if distance_pct < self.min_zone_separation_pct:
                new_lower = min(existing_zone.lower_bound, lower)
                new_upper = max(existing_zone.upper_bound, upper)
                new_mid = (new_lower + new_upper) / 2
                merged_zone = replace(
                    existing_zone, lower_bound=new_lower, upper_bound=new_upper, mid=new_mid,
                    strength=min(existing_zone.strength + 0.1, 1.0),
                    touch_count=existing_zone.touch_count + 1,
                    zone_type=zone_type, last_touched_at=self.bar_counter,
                )
                self.zones[existing_id] = merged_zone
                return

        zone = Zone(
            zone_id=zone_id, lower_bound=lower, upper_bound=upper, mid=mid,
            strength=0.5, age=0, touch_count=1, rejection_count=0, breakout_count=0,
            rejection_score=0.5, breakout_score=0.5, liquidity_score=0.5,
            survival_probability=0.6, zone_type=zone_type, created_at=created_at,
            last_touched_at=self.bar_counter,
        )
        self.zones[zone_id] = zone
        self._confirmed_at_bar[zone_id] = self.bar_counter

    def _update_zone_ages(self) -> None:
        for zone_id, zone in list(self.zones.items()):
            self.zones[zone_id] = replace(zone, age=zone.age + 1)

    def _decay_untouched_zones(self) -> None:
        for zone_id in list(self.zones.keys()):
            zone = self.zones[zone_id]
            bars_since_touch = self.bar_counter - zone.last_touched_at

            if bars_since_touch > self.zone_decay_bars:
                decay_factor = self._time_decay_factor(zone)
                self.zones[zone_id] = replace(
                    zone, strength=zone.strength * decay_factor,
                    survival_probability=zone.survival_probability * decay_factor,
                )

            if self.zones[zone_id].survival_probability < 0.05:
                del self.zones[zone_id]
                self._confirmed_at_bar.pop(zone_id, None)

    def _merge_nearby_zones(self, current_price: float) -> None:
        zones_list = list(self.zones.values())
        zones_to_remove = set()

        for i, z1 in enumerate(zones_list):
            for j, z2 in enumerate(zones_list):
                if i >= j or z1.zone_type != z2.zone_type:
                    continue
                if z1.zone_id in zones_to_remove or z2.zone_id in zones_to_remove:
                    continue

                distance_pct = abs(z1.mid - z2.mid) / z1.mid * 100 if z1.mid != 0 else 0
                if distance_pct < self.min_zone_separation_pct * 0.5:
                    merged = replace(
                        z1,
                        lower_bound=min(z1.lower_bound, z2.lower_bound),
                        upper_bound=max(z1.upper_bound, z2.upper_bound),
                        mid=(z1.mid + z2.mid) / 2,
                        strength=max(z1.strength, z2.strength),
                        age=min(z1.age, z2.age),
                        touch_count=z1.touch_count + z2.touch_count,
                        rejection_count=z1.rejection_count + z2.rejection_count,
                        breakout_count=z1.breakout_count + z2.breakout_count,
                        rejection_score=(z1.rejection_score + z2.rejection_score) / 2,
                        breakout_score=(z1.breakout_score + z2.breakout_score) / 2,
                        liquidity_score=max(z1.liquidity_score, z2.liquidity_score),
                        survival_probability=max(z1.survival_probability, z2.survival_probability),
                        created_at=min(z1.created_at, z2.created_at),
                        last_touched_at=max(z1.last_touched_at, z2.last_touched_at),
                    )
                    self.zones[z1.zone_id] = merged
                    zones_to_remove.add(z2.zone_id)

        for zone_id in zones_to_remove:
            if zone_id in self.zones:
                del self.zones[zone_id]
                self._confirmed_at_bar.pop(zone_id, None)
