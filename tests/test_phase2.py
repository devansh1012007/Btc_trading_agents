"""
Phase 2 test suite (§10).

Covers the Dynamic Structure Engine completion (zone touch/breakout
classification, no-look-ahead, idempotency, structural feature wiring),
the Regime Detection Layer (probability normalization, causal filtering,
model-absent fallback, panic-score smoothness), and the shared
walk-forward splitter's embargo guarantee.
"""
import numpy as np
import pytest

from src.state.contracts import MarketSnapshot, FeatureVector, RegimeType
from src.layers.structure_engine import DynamicStructureEngine
from src.layers.structural_features import attach_structural_features
from src.layers.regime_detector import RegimeDetector, panic_score
from src.backtest.walk_forward import WalkForwardSplitter


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def make_snapshot(timestamp, open_, high, low, close, volume=100.0, symbol="BTC/USDT"):
    return MarketSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        open=open_, high=high, low=low, close=close, volume=volume,
        bid=close * 0.9995, ask=close * 1.0005,
        bid_volume=50.0, ask_volume=50.0,
        spread_pct=0.01,
        hour_utc=12, day_of_week=2,
    )


def make_feature_vector(timestamp, momentum_5m=0.0, momentum_10m=0.0,
                         volatility_5m=0.1, volatility_10m=0.1,
                         volatility_30s=0.05, momentum_30s=0.0,
                         volume_zscore=0.0, symbol="BTC/USDT"):
    return FeatureVector(
        timestamp=timestamp, symbol=symbol,
        momentum_30s=momentum_30s, momentum_5m=momentum_5m,
        volatility_30s=volatility_30s, volatility_5m=volatility_5m,
        returns_5=0.0, rsi_14=50.0, macd_signal=0.0,
        sma_9=100.0, sma_21=100.0,
        momentum_10m=momentum_10m, volatility_10m=volatility_10m,
        imbalance_ratio=0.5, orderflow_momentum=0.0,
        spread_pct=0.01, depth_10_level=100.0,
        raw_features={"volume_zscore": volume_zscore},
    )


def _build_pivot_high(engine, base_price=100.0, pivot_offset=2.0):
    """Feed bars so a single pivot-high (resistance) zone gets created."""
    w = engine.pivot_window
    t = 0
    for i in range(w):
        price = base_price + i
        engine.add_bar(make_snapshot(t, price, price + 0.1, price - 0.1, price))
        t += 1

    pivot_price = base_price + pivot_offset
    engine.add_bar(make_snapshot(t, pivot_price, pivot_price + 0.5, pivot_price - 0.1, pivot_price))
    t += 1

    last_price = base_price
    for i in range(w):
        price = base_price + pivot_offset - (i + 1)
        last_price = price
        engine.add_bar(make_snapshot(t, price, price + 0.1, price - 0.1, price))
        engine.compute_zones(current_price=price)
        t += 1

    return t, pivot_price


# --------------------------------------------------------------------
# Dynamic Structure Engine
# --------------------------------------------------------------------

def test_zone_touch_classification():
    """A price approach that bounces off a zone should increment
    rejection_count, not breakout_count."""
    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=1000)
    t, _ = _build_pivot_high(engine)

    assert len(engine.zones) >= 1
    zone_id = next(iter(engine.zones))
    zone_before = engine.zones[zone_id]
    assert zone_before.rejection_count == 0 and zone_before.breakout_count == 0

    upper = zone_before.upper_bound
    lower = zone_before.lower_bound
    approach_price = (upper + lower) / 2

    engine.add_bar(make_snapshot(t, approach_price, upper + 0.01, approach_price - 0.2, upper - 0.02))
    engine.compute_zones(current_price=upper - 0.02)

    zone_after = engine.zones[zone_id]
    assert zone_after.rejection_count == zone_before.rejection_count + 1
    assert zone_after.breakout_count == zone_before.breakout_count


def test_zone_breakout_classification():
    """A close through a zone's upper bound should increment breakout_count
    and drop survival_probability sharply."""
    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=1000)
    t, _ = _build_pivot_high(engine)

    zone_id = next(iter(engine.zones))
    zone_before = engine.zones[zone_id]
    survival_before = zone_before.survival_probability
    upper = zone_before.upper_bound
    lower = zone_before.lower_bound

    breakout_close = upper + 1.0
    engine.add_bar(make_snapshot(t, lower, breakout_close + 0.2, lower - 0.1, breakout_close))
    engine.compute_zones(current_price=breakout_close)

    zone_after = engine.zones[zone_id]
    assert zone_after.breakout_count == zone_before.breakout_count + 1
    assert zone_after.survival_probability < survival_before


def test_zone_no_lookahead():
    """A zone must never appear in compute_zones()'s output for a bar
    before its confirmed_at_bar."""
    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=1000)

    t = 0
    prices = [100, 101, 102, 105, 104, 103, 102, 101, 100, 99, 100, 101]
    for price in prices:
        engine.add_bar(make_snapshot(t, price, price + 0.3, price - 0.3, price))
        zone_map = engine.compute_zones(current_price=price)
        for zone in zone_map.zones:
            confirmed_at = engine.get_confirmed_at_bar(zone.zone_id)
            assert confirmed_at is not None
            assert confirmed_at <= engine.bar_counter
        t += 1


def test_zone_idempotent_add_bar():
    """Calling compute_zones repeatedly on the same underlying data
    shouldn't duplicate or double-count a zone."""
    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=1000)
    _build_pivot_high(engine)

    zone_ids_before = set(engine.zones.keys())
    zone_map_1 = engine.compute_zones(current_price=100.0)
    zone_map_2 = engine.compute_zones(current_price=100.0)

    assert set(engine.zones.keys()) == zone_ids_before
    assert len(zone_map_1.zones) == len(zone_map_2.zones)


def test_structural_features_populated():
    """FeatureVector's structural fields become non-None once a ZoneMap
    with at least one zone exists."""
    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=1000)
    t, pivot_price = _build_pivot_high(engine)
    zone_map = engine.compute_zones(current_price=pivot_price - 1.0)

    fv = make_feature_vector(timestamp=t)
    merged = attach_structural_features(fv, zone_map)

    assert zone_map.nearest_resistance is not None
    assert merged.distance_to_nearest_resistance is not None
    assert merged.resistance_strength is not None


# --------------------------------------------------------------------
# Regime Detection Layer
# --------------------------------------------------------------------

def test_regime_state_probabilities_sum_to_one():
    detector = RegimeDetector(hmm_model_path="nonexistent.pkl", classifier_model_path="nonexistent2.pkl")
    fv = make_feature_vector(timestamp=0, momentum_5m=0.2)
    detector.add_bar(fv)
    state = detector.compute_regime(fv)

    total = (
        state.prob_trending_up + state.prob_trending_down
        + state.prob_chop_range + state.prob_panic_spike
    )
    assert abs(total - 1.0) < 1e-6


def test_regime_filtering_is_causal():
    """Output for bar N must be identical whether or not bars N+1..N+k
    exist -- i.e. no smoothing/look-ahead in the serving path."""
    np.random.seed(0)
    momentums = np.random.normal(0, 0.1, 30)
    n_cutoff = 15

    detector_full = RegimeDetector(hmm_model_path="nonexistent.pkl", classifier_model_path="nonexistent2.pkl")
    state_at_n = None
    for i, m in enumerate(momentums):
        fv = make_feature_vector(timestamp=i, momentum_5m=float(m))
        detector_full.add_bar(fv)
        state = detector_full.compute_regime(fv)
        if i == n_cutoff:
            state_at_n = state

    detector_short = RegimeDetector(hmm_model_path="nonexistent.pkl", classifier_model_path="nonexistent2.pkl")
    state_short = None
    for i, m in enumerate(momentums[:n_cutoff + 1]):
        fv = make_feature_vector(timestamp=i, momentum_5m=float(m))
        detector_short.add_bar(fv)
        state_short = detector_short.compute_regime(fv)

    assert state_at_n.prob_trending_up == pytest.approx(state_short.prob_trending_up)
    assert state_at_n.prob_trending_down == pytest.approx(state_short.prob_trending_down)
    assert state_at_n.prob_chop_range == pytest.approx(state_short.prob_chop_range)
    assert state_at_n.prob_panic_spike == pytest.approx(state_short.prob_panic_spike)
    assert state_at_n.primary_regime == state_short.primary_regime


def test_regime_fallback_without_model_file():
    """RegimeDetector degrades gracefully (rule-based) if .pkl files are absent."""
    detector = RegimeDetector(
        hmm_model_path="data/models/does_not_exist_hmm.pkl",
        classifier_model_path="data/models/does_not_exist_clf.pkl",
    )
    fv = make_feature_vector(timestamp=0, momentum_5m=0.1)
    detector.add_bar(fv)
    state = detector.compute_regime(fv)

    assert state is not None
    assert isinstance(state.primary_regime, RegimeType)


def test_panic_override_smooth():
    """panic_score should be continuous (no discontinuous jump) across the threshold."""
    threshold = 3.0
    xs = np.linspace(threshold - 1.0, threshold + 1.0, 200)
    scores = [panic_score(x, 0.0, 0.0, z_threshold=threshold) for x in xs]

    diffs = np.abs(np.diff(scores))
    assert np.max(diffs) < 0.1
    assert scores[0] < 0.5 < scores[-1]


# --------------------------------------------------------------------
# Walk-forward splitter
# --------------------------------------------------------------------

def test_walk_forward_embargo():
    """No index from a test fold should appear within embargo_bars of any
    train-fold index."""
    splitter = WalkForwardSplitter(train_bars=100, test_bars=20, step_bars=20, embargo_bars=10)
    n_samples = 500

    folds = list(splitter.split(n_samples))
    assert len(folds) > 0

    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert (test_idx.min() - train_idx.max()) > splitter.embargo_bars
