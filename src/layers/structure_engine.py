"""
Dynamic Structure Engine: Detect and maintain support/resistance zones.
This layer identifies structural levels from market data.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from ..state.contracts import Zone, ZoneMap, MarketSnapshot, FeatureVector

logger = logging.getLogger(__name__)


class DynamicStructureEngine:
    """Identify and manage support/resistance zones."""
    
    def __init__(
        self,
        lookback_bars: int = 50,
        pivot_window: int = 5,
        min_zone_separation_pct: float = 1.0,
        zone_decay_bars: int = 100,
    ):
        """
        Initialize zone engine.
        
        Args:
            lookback_bars: How many bars to keep for zone calculation
            pivot_window: Window for finding pivot highs/lows
            min_zone_separation_pct: Zones closer than this merge
            zone_decay_bars: Bars before untouched zone decays away
        """
        self.lookback_bars = lookback_bars
        self.pivot_window = pivot_window
        self.min_zone_separation_pct = min_zone_separation_pct
        self.zone_decay_bars = zone_decay_bars
        
        # Rolling history
        self.price_history = []
        self.zones = {}  # {zone_id: Zone}
        self.bar_counter = 0
    
    def add_bar(self, snapshot: MarketSnapshot, feature: Optional[FeatureVector] = None) -> None:
        """Add a new bar and update zones."""
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'high': snapshot.high,
            'low': snapshot.low,
            'close': snapshot.close,
            'volume': snapshot.volume,
        })
        
        # Keep rolling window
        if len(self.price_history) > self.lookback_bars:
            self.price_history.pop(0)
        
        self.bar_counter += 1
    
    def compute_zones(self, current_price: float, symbol: str = "BTC/USDT") -> ZoneMap:
        """
        Compute active zone map for current price.
        
        Returns:
            ZoneMap with all active zones
        """
        if len(self.price_history) < self.pivot_window + 1:
            return ZoneMap(
                timestamp=int(self.price_history[-1]['timestamp']) if self.price_history else 0,
                symbol=symbol,
                current_price=current_price,
                zones=[],
            )
        
        # Generate new zones from current bar
        self._generate_zones_from_pivots()
        self._generate_zones_from_volume()
        
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
                # Merge zones
                new_lower = min(existing_zone.lower_bound, lower)
                new_upper = max(existing_zone.upper_bound, upper)
                new_mid = (new_lower + new_upper) / 2
                
                merged_zone = Zone(
                    zone_id=existing_id,
                    lower_bound=new_lower,
                    upper_bound=new_upper,
                    mid=new_mid,
                    strength=min(existing_zone.strength + 0.1, 1.0),
                    age=existing_zone.age,
                    touch_count=existing_zone.touch_count + 1,
                    rejection_count=existing_zone.rejection_count,
                    breakout_count=existing_zone.breakout_count,
                    rejection_score=existing_zone.rejection_score,
                    breakout_score=existing_zone.breakout_score,
                    liquidity_score=existing_zone.liquidity_score,
                    survival_probability=existing_zone.survival_probability,
                    zone_type=zone_type,
                    created_at=existing_zone.created_at,
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
    
    def _update_zone_ages(self) -> None:
        """Increment zone ages."""
        for zone_id in self.zones:
            zone = self.zones[zone_id]
            self.zones[zone_id] = Zone(
                zone_id=zone.zone_id,
                lower_bound=zone.lower_bound,
                upper_bound=zone.upper_bound,
                mid=zone.mid,
                strength=zone.strength,
                age=zone.age + 1,
                touch_count=zone.touch_count,
                rejection_count=zone.rejection_count,
                breakout_count=zone.breakout_count,
                rejection_score=zone.rejection_score,
                breakout_score=zone.breakout_score,
                liquidity_score=zone.liquidity_score,
                survival_probability=zone.survival_probability,
                zone_type=zone.zone_type,
                created_at=zone.created_at,
                last_touched_at=zone.last_touched_at,
            )
    
    def _decay_untouched_zones(self) -> None:
        """Decay zones that haven't been touched recently."""
        for zone_id in list(self.zones.keys()):
            zone = self.zones[zone_id]
            bars_since_touch = self.bar_counter - zone.last_touched_at
            
            if bars_since_touch > self.zone_decay_bars:
                decay_factor = 1.0 - (bars_since_touch / (self.zone_decay_bars * 2))
                decay_factor = max(decay_factor, 0.1)
                
                decayed_zone = Zone(
                    zone_id=zone.zone_id,
                    lower_bound=zone.lower_bound,
                    upper_bound=zone.upper_bound,
                    mid=zone.mid,
                    strength=zone.strength * decay_factor,
                    age=zone.age,
                    touch_count=zone.touch_count,
                    rejection_count=zone.rejection_count,
                    breakout_count=zone.breakout_count,
                    rejection_score=zone.rejection_score,
                    breakout_score=zone.breakout_score,
                    liquidity_score=zone.liquidity_score,
                    survival_probability=zone.survival_probability * decay_factor,
                    zone_type=zone.zone_type,
                    created_at=zone.created_at,
                    last_touched_at=zone.last_touched_at,
                )
                self.zones[zone_id] = decayed_zone
            
            if self.zones[zone_id].survival_probability < 0.05:
                del self.zones[zone_id]
    
    def _merge_nearby_zones(self, current_price: float) -> None:
        """Merge zones that are very close."""
        zones_list = list(self.zones.values())
        zones_to_remove = set()
        
        for i, z1 in enumerate(zones_list):
            for j, z2 in enumerate(zones_list):
                if i >= j or z1.zone_type != z2.zone_type:
                    continue
                
                distance_pct = abs(z1.mid - z2.mid) / z1.mid * 100 if z1.mid != 0 else 0
                if distance_pct < self.min_zone_separation_pct * 0.5:
                    # Merge z2 into z1
                    merged = Zone(
                        zone_id=z1.zone_id,
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
                        zone_type=z1.zone_type,
                        created_at=min(z1.created_at, z2.created_at),
                        last_touched_at=max(z1.last_touched_at, z2.last_touched_at),
                    )
                    
                    self.zones[z1.zone_id] = merged
                    zones_to_remove.add(z2.zone_id)
        
        for zone_id in zones_to_remove:
            if zone_id in self.zones:
                del self.zones[zone_id]
