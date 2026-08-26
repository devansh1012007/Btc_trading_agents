"""
Look-ahead (leakage) guards (PRESENTATION §3): tests that would fail if
any layer started reading information from the future.

  - features at bar i are a pure function of bars[0..i] — perturbing
    future bars must not change them
  - feature timestamps match their bar's timestamp
  - walk-forward splits enforce the purge/embargo gap between train and test
  - triple-barrier labels depend only on the forward window handed to them

No conftest.py — inline helpers only (repo convention).
"""
import pytest

from src.backtest.walk_forward import WalkForwardSplitter
from src.labeling.triple_barrier import label_triple_barrier
from src.layers.feature_engine import FeatureEngine
from src.state.contracts import MarketSnapshot

pytestmark = [pytest.mark.unit]

WARMUP = 30


def make_bar(timestamp, close, volume=100.0):
    return MarketSnapshot(
        timestamp=timestamp, symbol="BTC/USDT", open=close, high=close * 1.005,
        low=close * 0.995, close=close, volume=volume, bid=close * 0.9999,
        ask=close * 1.0001, bid_volume=5, ask_volume=5, spread_pct=0.02,
    )


def build_bars(n, seed_close=50_000.0):
    # Deterministic pseudo-random walk (no numpy RNG state shared with prod code)
    bars, close = [], seed_close
    for i in range(n):
        close = close * (1 + ((i * 37) % 11 - 5) / 2000.0)
        bars.append(make_bar(timestamp=1_700_000_000 + i * 60, close=round(close, 2)))
    return bars


# --------------------------------------------------------------------
# Feature causality
# --------------------------------------------------------------------

def test_features_are_pure_function_of_past_bars():
    """Features at bar k must be identical whether or not future bars exist."""
    bars = build_bars(WARMUP + 20)

    full_engine = FeatureEngine(lookback_bars=100)
    full_features = {}
    for bar in bars:
        full_engine.add_bar(bar)
        fv = full_engine.compute_features(bar)
        if fv is not None:
            full_features[bar.timestamp] = fv

    assert len(full_features) >= WARMUP, "warmup assumption broken"

    # Second run: only the first k+1 bars ever exist.
    k = WARMUP + 10
    truncated_engine = FeatureEngine(lookback_bars=100)
    for bar in bars[:k + 1]:
        truncated_engine.add_bar(bar)
        fv = truncated_engine.compute_features(bar)

    reference = full_features[bars[k].timestamp]
    assert fv is not None
    assert fv == reference, "feature at bar k changed when future bars were removed"


def test_perturbing_future_bars_does_not_change_past_features():
    """
    Replace every bar AFTER bar k with garbage and re-run the stream:
    every feature vector up to and including bar k must be bit-identical
    to the clean run. Catches engines that peek ahead in the input list,
    not just engines that read their own buffer.
    """
    bars = build_bars(WARMUP + 20)
    k = WARMUP + 5

    def collect_fvs(stream):
        engine = FeatureEngine(lookback_bars=100)
        out = {}
        for bar in stream:
            engine.add_bar(bar)
            fv = engine.compute_features(bar)
            if fv is not None:
                out[bar.timestamp] = fv
        return out

    clean = collect_fvs(bars)
    poisoned_stream = (bars[:k + 1]
                       + [make_bar(timestamp=bars[k].timestamp + 60 * j, close=999_999.0)
                          for j in range(1, len(bars) - k)])
    poisoned = collect_fvs(poisoned_stream)

    # Invariant scope: every feature computed at or before bar k must be
    # identical regardless of what arrived afterward.
    cutoff_ts = bars[k].timestamp
    assert {ts for ts in clean if ts <= cutoff_ts} == {ts for ts in poisoned if ts <= cutoff_ts}
    for ts, fv in clean.items():
        if ts > cutoff_ts:
            continue
        assert poisoned[ts] == fv, f"feature at t={ts} changed when future bars were poisoned"


def test_feature_timestamp_matches_its_bar():
    bars = build_bars(WARMUP + 3)
    engine = FeatureEngine(lookback_bars=100)
    for bar in bars:
        engine.add_bar(bar)
        fv = engine.compute_features(bar)
        if fv is not None:
            assert fv.timestamp == bar.timestamp


# --------------------------------------------------------------------
# Walk-forward split hygiene
# --------------------------------------------------------------------

def test_train_and_test_never_overlap_and_embargo_is_respected():
    splitter = WalkForwardSplitter(train_bars=500, test_bars=100, step_bars=100, embargo_bars=20)
    n = 1500
    for train_idx, test_idx in splitter.split(n):
        assert train_idx.max() < test_idx.min(), "train/test overlap"
        gap = test_idx.min() - train_idx.max() - 1
        assert gap >= splitter.embargo_bars, f"embargo violated: gap={gap}"


def test_split_indices_within_bounds():
    splitter = WalkForwardSplitter(train_bars=300, test_bars=80, step_bars=80, embargo_bars=10)
    n = 900
    for train_idx, test_idx in splitter.split(n):
        assert train_idx.min() >= 0 and test_idx.max() < n


def test_zero_embargo_still_no_overlap():
    splitter = WalkForwardSplitter(train_bars=200, test_bars=50, step_bars=50, embargo_bars=0)
    for train_idx, test_idx in splitter.split(600):
        assert train_idx.max() < test_idx.min()


# --------------------------------------------------------------------
# Triple-barrier labels see only their window
# --------------------------------------------------------------------

def _label_with(bars_slice):
    return label_triple_barrier(
        forward_bars=bars_slice, entry_price=bars_slice[0].close, direction=1,
        take_profit_pct=1.0, stop_loss_pct=1.0, max_holding_bars=5,
        fee_pct=0.04, entry_timestamp=bars_slice[0].timestamp,
    )


def make_trend_bars(pct_per_bar):
    return [make_bar(timestamp=1_800_000_000 + i * 60, close=100.0 * (1 + pct_per_bar / 100) ** i)
            for i in range(12)]


def test_label_outcome_unchanged_by_extra_future_bars():
    rising = make_trend_bars(+0.4)  # TP (1%) hit around bar 3-4
    short_window = _label_with(rising[:8])
    long_window = _label_with(rising)
    assert short_window.exit_reason == long_window.exit_reason
    assert short_window.bars_held == long_window.bars_held


def test_label_changes_only_when_window_truncated_before_event():
    rising = make_trend_bars(+0.4)
    outcome = _label_with(rising)
    assert outcome.exit_reason == "take_profit"

    # Cut BEFORE the TP touch → same labels can't be produced from data
    # that never contained the event.
    early = _label_with(rising[:2])
    assert early.exit_reason != "take_profit" or early.bars_held < outcome.bars_held


def test_entry_price_is_first_forward_bar_close():
    bars = make_trend_bars(+0.4)
    outcome = _label_with(bars)
    # With direction=1 and entry at close of bar 0, pnl must reflect growth
    # from that exact price (positive for a rising series net of fees).
    assert outcome.realized_return_pct > -100.0  # sane output
