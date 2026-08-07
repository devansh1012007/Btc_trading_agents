"""
Dynamic Structure Engine: Detect and maintain support/resistance zones.
This layer identifies structural levels from market data.

Phase 2 completes this engine (§6 of the Phase 2 plan):
  - §6.1 Price-action interaction detector: zones are now actually checked
    against each bar's high/low/close, so rejection_count/breakout_count/
    rejection_score/breakout_score/survival_probability move off their
    init values.
  - §6.2 Pivot-confirmation look-ahead fix: an engine-side shadow dict
    tracks confirmed_at_bar (the bar count at which a zone actually
    entered self.zones), distinct from created_at (the pivot's own,
    earlier timestamp). Downstream consumers must join on
    confirmed_at_bar, never created_at.
  - §6.3 Beta-Bernoulli survival update, replacing the old
    "hardcoded 0.6, decays with time only" logic.
  - §6.6 Zone.regime_relevance populated from real touch events.
  - §6.5 Optional blended serving: rule-based survival probability can be
    blended with a trained zone-survival LightGBM model, with a
    zone_survival_blend_weight config knob and a pure rule-based fallback
    if no model file is present.
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
    ):
        """
        Initialize zone engine.

        Args:
            lookback_bars: How many bars to keep for zone calculation
            pivot_window: Window for finding pivot highs/lows
            min_zone_separation_pct: Zones closer than this merge
            zone_decay_bars: Bars before untouched zone decays away
            breakout_survival_penalty: Multiplicative penalty applied to
                survival_probability the instant a zone breaks out, so it
                drops out of active_zones quickly instead of waiting for
                time-decay alone (§6.3)
            zone_survival_model_path: Optional path to a trained
                zone-survival LightGBM model (§6.5). If missing or absent,
                the engine falls back to pure rule-based survival.
            zone_survival_blend_weight: 0 = pure rule-based,
                1 = pure ML-predicted survival probability
        """
        self.lookback_bars = lookback_bars
        self.pivot_window = pivot_window
        self.min_zone_separation_pct = min_zone_separation_pct
        self.zone_decay_bars = zone_decay_bars
        self.breakout_survival_penalty = breakout_survival_penalty
        self.zone_survival_blend_weight = zone_survival_blend_weight

        # Rolling history (§5.1 shared RollingBarBuffer, replacing the
        # hand-rolled list + manual pop-to-lookback that used to be
        # duplicated here and in feature_engine.py)
        self.price_history = RollingBarBuffer(maxlen=lookback_bars)
        self.zones: Dict[str, Zone] = {}
        self.bar_counter = 0

        # §6.2: engine-side shadow tracking of when each zone_id actually
        # entered self.zones -- the fix for the pivot-confirmation
        # look-ahead trap. Never exposed via the frozen Zone contract;
        # always query via get_confirmed_at_bar().
        self._confirmed_at_bar: Dict[str, int] = {}

        # Regime state passed into the most recent compute_zones() call,
        # used for regime_relevance tagging (§6.6) and the zone-survival
        # feature set (§6.5).
        self._latest_regime_state: Optional[RegimeState] = None

        # Full touch-event log (§6.5 labeling script reads this directly
        # after a replay, rather than re-detecting touches itself).
        self.touch_events: List[Dict] = []

        self._zone_survival_model = None
        self._zone_survival_feature_names = None
        if zone_survival_model_path:
            self._load_zone_survival_model(zone_survival_model_path)

    def add_bar(self, snapshot: MarketSnapshot) -> None:
        """Add a new bar and update zones."""
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
        """
        Compute active zone map for current price.

        Args:
            current_price: Latest close price
            symbol: Trading pair
            regime_state: The current bar's RegimeState (§3 ordering:
                regime is computed before structure so zones can be tagged
                with the regime active at the moment they're touched).

        Returns:
            ZoneMap with all active zones
        """
        self._latest_regime_state = regime_state

        if len(self.price_history) < self.pivot_window + 1:
            return ZoneMap(
                timestamp=int(self.price_history[-1]['timestamp']) if len(self.price_history) else 0,
                symbol=symbol,
                current_price=current_price,
                zones=[],
            )

        # Generate new zones from current bar
        self._generate_zones_from_pivots()
        self._generate_zones_from_volume()

        # §6.1: classify this bar's price action against every active zone
        self._process_price_action()

        # Update zone metadata
        self._update_zone_ages()
        self._decay_untouched_zones()
        self._merge_nearby_zones(current_price)

        # Filter to active zones
        active_zones = [z for z in self.zones.values() if z.survival_probability > 0.1]
        active_zones.sort(key=lambda z: abs(z.mid - current_price))

        # Find nearest support/resistance
        support_zones = [z for z in active_zones if z.zone_type == "support" and z.mid < current_price]
        resistance_zones = [z for z in active_zones if z.zone_type == "resistance" and z.mid > current_price]

        nearest_support = support_zones[-1] if support_zones else None
        nearest_resistance = resistance_zones[0] if resistance_zones else None

        zone_map = ZoneMap(
            timestamp=int(self.price_history[-1]['timestamp']),
            symbol=symbol,
            current_price=current_price,
            zones=active_zones,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

        return zone_map

    # ------------------------------------------------------------------
    # §6.1 Price-action interaction detector
    # ------------------------------------------------------------------

    def _process_price_action(self) -> None:
        """
        Check the latest bar's price action against every active zone and
        classify it as a rejection or a breakout. Before this, no code
        anywhere compared a bar's high/low against a zone's bounds, so
        rejection_count/breakout_count/rejection_score/breakout_score
        never moved off their initial values (finding #1).
        """
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
        """Shared update path for both rejection and breakout outcomes."""
        zone = self.zones.get(zone_id)
        if zone is None:
            return

        self._log_touch_event(zone_id, zone, "breakout" if broke_through else "rejection")

        rejection_count = zone.rejection_count + (0 if broke_through else 1)
        breakout_count = zone.breakout_count + (1 if broke_through else 0)
        rejection_score = self._beta_bernoulli_rejection_score(rejection_count, breakout_count)

        rule_based_survival = rejection_score * self._time_decay_factor(zone)
        if broke_through:
            # Immediate penalty so a broken zone drops out of active_zones
            # quickly, rather than waiting for time-decay alone (§6.3).
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

        # §6.6: tag which regime this zone tends to matter in.
        self._update_regime_relevance(zone_id, self._latest_regime_state)

    def _record_rejection(self, zone_id: str) -> None:
        self._apply_touch_outcome(zone_id, broke_through=False)

    def _record_breakout(self, zone_id: str) -> None:
        self._apply_touch_outcome(zone_id, broke_through=True)

    def _log_touch_event(self, zone_id: str, zone: Zone, event_type: str) -> None:
        """Record a touch event with a full feature snapshot, so
        scripts/label_zone_events.py can build training labels without
        re-implementing touch detection."""
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
        """Beta(1,1) (Laplace) posterior mean for P(rejection) -- replaces
        the old hardcoded-0.6 survival logic (§6.3)."""
        return (rejection_count + 1) / (rejection_count + breakout_count + 2)

    def _time_decay_factor(self, zone: Zone) -> float:
        """Same time-decay logic as _decay_untouched_zones, evaluated for
        a single zone (shared so the two paths can't drift apart)."""
        bars_since_touch = self.bar_counter - zone.last_touched_at
        if bars_since_touch <= self.zone_decay_bars:
            return 1.0
        decay_factor = 1.0 - (bars_since_touch / (self.zone_decay_bars * 2))
        return max(decay_factor, 0.1)

    def _recent_volatility_pct(self, periods: int = 20) -> float:
        """Recent realized volatility (%) used as the denominator for the
        zone-survival feature 'width_to_volatility_ratio'."""
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
            self._zone_survival_model = payload["model"]
            self._zone_survival_feature_names = payload.get("feature_names")
            logger.info(f"Loaded zone survival model from {p}")
        except Exception as e:
            logger.warning(f"Failed to load zone survival model at {p}: {e}")

    def _blend_survival_probability(self, zone_id: str, rule_based_survival: float) -> float:
        """
        Blend rule-based survival_probability with a trained model's
        prediction. Always falls back to pure rule-based if no model is
        loaded, blend weight is 0, or inference fails for any reason --
        same defensive pattern used elsewhere in the codebase.
        """
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
        """
        Bar count at which this zone actually entered self.zones -- the
        causally-correct join key for any downstream consumer, as opposed
        to Zone.created_at (the underlying pivot's own, earlier
        timestamp). Returns None if zone_id is unknown.
        """
        return self._confirmed_at_bar.get(zone_id)

    # ------------------------------------------------------------------
    # Zone generation (unchanged from Phase 1 except confirmed_at_bar
    # bookkeeping)
    # ------------------------------------------------------------------

    def _generate_zones_from_pivots(self) -> None:
        """Detect pivot highs and lows."""
        if len(self.price_history) < self.pivot_window + 1:
            return

        highs = np.array([b['high'] for b in self.price_history])
        lows = np.array([b['low'] for b in self.price_history])
        timestamps = np.array([b['timestamp'] for b in self.price_history])

        # Find pivot highs
        for i in range(self.pivot_window, len(highs) - self.pivot_window):
            if highs[i] == np.max(highs[i - self.pivot_window:i + self.pivot_window + 1]):
                zone_price = highs[i]
                zone_id = f"pivot_high_{timestamps[i]}"

                if zone_id not in self.zones:
                    zone_width = zone_price * 0.5 / 100
                    self._add_or_merge_zone(
                        zone_id=zone_id,
                        lower=zone_price - zone_width,
                        upper=zone_price + zone_width,
                        zone_type="resistance",
                        created_at=timestamps[i],
                    )

        # Find pivot lows
        for i in range(self.pivot_window, len(lows) - self.pivot_window):
            if lows[i] == np.min(lows[i - self.pivot_window:i + self.pivot_window + 1]):
                zone_price = lows[i]
                zone_id = f"pivot_low_{timestamps[i]}"

                if zone_id not in self.zones:
                    zone_width = zone_price * 0.5 / 100
                    self._add_or_merge_zone(
                        zone_id=zone_id,
                        lower=zone_price - zone_width,
                        upper=zone_price + zone_width,
                        zone_type="support",
                        created_at=timestamps[i],
                    )

    def _generate_zones_from_volume(self) -> None:
        """Detect zones of high volume concentration."""
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
                        zone_id=zone_id,
                        lower=zone_price - zone_width,
                        upper=zone_price + zone_width,
                        zone_type=zone_type,
                        created_at=timestamps[i],
                    )

    def _add_or_merge_zone(
        self,
        zone_id: str,
        lower: float,
        upper: float,
        zone_type: str,
        created_at: int,
    ) -> None:
        """Add zone or merge with nearby zone."""
        mid = (lower + upper) / 2

        # Check for merge with existing zones
        for existing_id, existing_zone in list(self.zones.items()):
            if existing_zone.zone_type != zone_type:
                continue

            distance_pct = abs(existing_zone.mid - mid) / mid * 100 if mid != 0 else 0
            if distance_pct < self.min_zone_separation_pct:
                # Merge zones -- existing_id's confirmed_at_bar is left
                # untouched (it was already confirmed earlier; that's the
                # correct, earliest confirmation time for this zone_id).
                new_lower = min(existing_zone.lower_bound, lower)
                new_upper = max(existing_zone.upper_bound, upper)
                new_mid = (new_lower + new_upper) / 2

                merged_zone = replace(
                    existing_zone,
                    lower_bound=new_lower,
                    upper_bound=new_upper,
                    mid=new_mid,
                    strength=min(existing_zone.strength + 0.1, 1.0),
                    touch_count=existing_zone.touch_count + 1,
                    zone_type=zone_type,
                    last_touched_at=self.bar_counter,
                )

                self.zones[existing_id] = merged_zone
                return

        # Create new zone
        zone = Zone(
            zone_id=zone_id,
            lower_bound=lower,
            upper_bound=upper,
            mid=mid,
            strength=0.5,
            age=0,
            touch_count=1,
            rejection_count=0,
            breakout_count=0,
            rejection_score=0.5,
            breakout_score=0.5,
            liquidity_score=0.5,
            survival_probability=0.6,
            zone_type=zone_type,
            created_at=created_at,
            last_touched_at=self.bar_counter,
        )
        self.zones[zone_id] = zone
        # §6.2: this is the moment the zone actually became knowable to
        # the system -- record it separately from created_at.
        self._confirmed_at_bar[zone_id] = self.bar_counter

    def _update_zone_ages(self) -> None:
        """Increment zone ages."""
        for zone_id, zone in list(self.zones.items()):
            self.zones[zone_id] = replace(zone, age=zone.age + 1)

    def _decay_untouched_zones(self) -> None:
        """Decay zones that haven't been touched recently."""
        for zone_id in list(self.zones.keys()):
            zone = self.zones[zone_id]
            bars_since_touch = self.bar_counter - zone.last_touched_at

            if bars_since_touch > self.zone_decay_bars:
                decay_factor = self._time_decay_factor(zone)
                self.zones[zone_id] = replace(
                    zone,
                    strength=zone.strength * decay_factor,
                    survival_probability=zone.survival_probability * decay_factor,
                )

            if self.zones[zone_id].survival_probability < 0.05:
                del self.zones[zone_id]
                self._confirmed_at_bar.pop(zone_id, None)

    def _merge_nearby_zones(self, current_price: float) -> None:
        """Merge zones that are very close."""
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
                    # Merge z2 into z1
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
