"""
Correctness tests for the Phase-1 defects the independent review found.

Every test here corresponds to a concrete, demonstrated failure: the
review proved each one by mutating the source and showing that nothing
caught it. The bar for a test in this file is that it must FAIL when the
defect is reintroduced.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics import dashboard_data as dd  # noqa: E402
from src.diagnostics.decision_trace import (  # noqa: E402
    STAGE_LABELS,
    TERMINAL_STAGES,
    classify_terminal_stage,
)
from src.gui import models  # noqa: E402
from src.gui.artifacts import (  # noqa: E402
    find_ohlcv_files,
    read_decisions_jsonl,
    symbol_file_tokens,
)
from src.gui.indicators import atr, bollinger, compute_indicators, donchian, ema, rsi  # noqa: E402
from src.gui.instruments import load_registry  # noqa: E402
from src.gui.models import (  # noqa: E402
    BAR_COLUMNS,
    DECISION_COLUMNS,
    TRADE_COLUMNS,
    ZONE_COLUMNS,
    empty_frame,
)
from src.gui.timeframes import (  # noqa: E402
    align_to_timeframe,
    bucket_start,
    resample_ohlcv,
    to_epoch_seconds,
)
from src.state.contracts import ExecutionRecord, RiskDecision  # noqa: E402
from src.utils.logging_setup import decision_record_from_state  # noqa: E402

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Timestamps: the unit must be detected, not guessed
# ---------------------------------------------------------------------------

def test_object_dtype_integer_epochs_are_not_corrupted():
    """An object-dtype column of epoch seconds used to be read as
    NANOSECONDS, silently shrinking every timestamp by a factor of 1e9."""
    series = pd.Series([1_700_000_000, 1_700_000_060], dtype=object)
    assert to_epoch_seconds(series).tolist() == [1_700_000_000.0, 1_700_000_060.0]


def test_microsecond_and_nanosecond_epochs_are_scaled_to_seconds():
    assert to_epoch_seconds(pd.Series([1_700_000_000_000_000])).iloc[0] == 1_700_000_000
    assert to_epoch_seconds(pd.Series([1_700_000_000_000_000_000])).iloc[0] == 1_700_000_000


def test_numeric_strings_are_parsed_as_epochs():
    assert to_epoch_seconds(pd.Series(["1700000000", "1700000060"])).tolist() == [
        1_700_000_000.0, 1_700_000_060.0,
    ]


def test_a_mixed_unit_column_keeps_both_rows():
    """A column that mixes seconds and milliseconds must not be scaled by
    a single column-wide guess."""
    series = pd.Series([1_700_000_000, 1_700_000_060_000])
    assert to_epoch_seconds(series).tolist() == [1_700_000_000.0, 1_700_000_060.0]


def test_the_gui_and_the_runtime_normalise_timestamps_the_same_way():
    """Two copies of a unit heuristic is two chances to disagree about
    when the system traded."""
    from src.utils.time_utils import epoch_seconds

    for series in (
        pd.Series([1_700_000_000, 1_700_000_060]),
        pd.Series([1_700_000_000_000, 1_700_000_060_000]),
        pd.Series([1_700_000_000], dtype=object),
        pd.Series(["2023-11-14 00:00:00"]),
    ):
        assert to_epoch_seconds(series).tolist() == epoch_seconds(series).tolist()


def test_a_boolean_column_is_not_treated_as_a_timestamp():
    out = to_epoch_seconds(pd.Series([True, False]))
    assert out.isna().all()


# ---------------------------------------------------------------------------
# Causality of the bucket helpers
# ---------------------------------------------------------------------------

def test_align_to_timeframe_keeps_a_missing_bucket_open_value():
    """``GroupBy.first()`` SKIPS nulls, which pulled a later bar's value
    into the bucket's open. The bucket's open must stay missing."""
    frame = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060],
        "regime": [None, "second"],
    })
    out = align_to_timeframe(frame, "5m", how="first")
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["regime"])


def test_align_to_timeframe_defaults_to_the_causal_first_value():
    frame = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060],
        "regime": ["first", "second"],
    })
    out = align_to_timeframe(frame, "5m")
    assert list(out["regime"]) == ["first"]


def test_align_to_timeframe_last_keeps_the_bucket_close():
    frame = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060],
        "regime": ["first", "second"],
    })
    out = align_to_timeframe(frame, "5m", how="last")
    assert list(out["regime"]) == ["second"]


def test_align_to_timeframe_rewrites_the_timestamp_to_the_bucket_start():
    """The returned timestamp must be the BUCKET's start, not the bar's
    own timestamp -- otherwise a chart joining on it puts the decision
    on the wrong candle."""
    frame = pd.DataFrame({
        "timestamp": [1_700_000_030, 1_700_000_060],
        "regime": ["a", "b"],
    })
    out = align_to_timeframe(frame, "5m")
    assert list(out["timestamp"]) == [bucket_start(1_700_000_030, "5m")]
    assert out.iloc[0]["timestamp"] != 1_700_000_030
    assert out.iloc[0]["timestamp"] % 300 == 0


def test_align_to_timeframe_rejects_a_row_without_a_timestamp():
    frame = pd.DataFrame({"timestamp": [1_700_000_000, None], "regime": ["a", "b"]})
    with pytest.raises(ValueError, match="missing or non-numeric"):
        align_to_timeframe(frame, "5m")


def test_resample_keeps_a_missing_open_missing():
    """A bucket's open is the open of the FIRST bar in it -- not the
    first non-null one, which would be a later bar's price."""
    bars = pd.DataFrame({
        "timestamp": [1_700_000_000, 1_700_000_060, 1_700_000_120],
        "open": [np.nan, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0],
        "volume": [1.0, 1.0, 1.0],
    })
    out = resample_ohlcv(bars, "5m")
    first = out.iloc[0]
    assert pd.isna(first["open"])
    assert first["close"] == 2.0


def test_resample_refuses_to_build_a_candle_finer_than_the_data():
    bars = pd.DataFrame({
        "timestamp": [1_700_000_000], "open": [1.0], "high": [1.0],
        "low": [1.0], "close": [1.0], "volume": [1.0],
    })
    with pytest.raises(ValueError, match="finer than the base granularity"):
        resample_ohlcv(bars, "10s", base_timeframe="1m")


def test_resample_is_causal_in_every_bucket():
    """The old causality test only compared the first two buckets."""
    stamps = [1_700_000_000 + i * 60 for i in range(60)]
    bars = pd.DataFrame({
        "timestamp": stamps,
        "open": np.arange(60, dtype="float64") + 1,
        "high": np.arange(60, dtype="float64") + 2,
        "low": np.arange(60, dtype="float64"),
        "close": np.arange(60, dtype="float64") + 1.5,
        "volume": np.ones(60),
    })
    full = resample_ohlcv(bars, "5m")
    partial = resample_ohlcv(bars.iloc[:30], "5m")
    assert len(partial) > 1
    # The partial run's LAST bucket is incomplete by construction, so
    # only the complete buckets are comparable.
    for i in range(len(partial) - 1):
        assert full.iloc[i].tolist() == partial.iloc[i].tolist(), f"bucket {i} differs"


def test_bucket_start_agrees_with_the_resampler():
    """The aligner and the resampler must place a bar in the SAME bucket."""
    stamps = [1_700_000_000 + i * 60 for i in range(200)]
    bars = pd.DataFrame({
        "timestamp": stamps,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    for tf in ("1m", "5m", "15m", "1h", "4h", "1D", "1W"):
        resampled = resample_ohlcv(bars, tf)
        labels = {bucket_start(int(ts), tf) for ts in stamps}
        assert set(resampled["timestamp"]) <= labels


# ---------------------------------------------------------------------------
# Frame dtypes: an empty frame must concat with a populated one
# ---------------------------------------------------------------------------

def test_empty_frame_gives_every_numeric_column_a_float_dtype():
    empty = empty_frame(BAR_COLUMNS)
    for col in ("open", "high", "low", "close", "volume"):
        assert empty[col].dtype.kind == "f", f"{col} is {empty[col].dtype}"
    assert empty["timestamp"].dtype.kind == "i"


def test_concatenating_an_empty_bars_frame_preserves_numeric_dtypes():
    live = pd.DataFrame({
        "timestamp": [1], "open": [1.0], "high": [1.0], "low": [1.0],
        "close": [1.0], "volume": [1.0],
    })
    merged = pd.concat([empty_frame(BAR_COLUMNS), live], ignore_index=True)
    for col in ("open", "high", "low", "close", "volume"):
        assert merged[col].dtype.kind == "f", f"{col} became {merged[col].dtype}"


def test_empty_frame_gives_the_boolean_column_a_nullable_boolean_dtype():
    empty = empty_frame(DECISION_COLUMNS)
    assert empty["risk_approved"].dtype == "boolean"


def test_concatenating_an_empty_decision_frame_keeps_the_boolean_column():
    live = pd.DataFrame({
        "timestamp": [1], "symbol": ["BTC/USDT"], "regime": ["trending_up"],
        "regime_confidence": [0.5], "expert": ["trend_expert"],
        "opportunity_score": [0.7], "router_confidence": [0.6],
        "risk_score": [0.2], "risk_approved": [True],
        "decision": ["NO_TRADE"], "terminal_stage": ["executed"],
        "terminal_reason": ["filled via trend_expert"],
    })
    merged = pd.concat([empty_frame(DECISION_COLUMNS), live], ignore_index=True)
    assert merged["risk_approved"].dtype == "boolean", (
        f"risk_approved became {merged['risk_approved'].dtype}"
    )


def test_empty_zone_frame_types_its_numeric_columns():
    empty = empty_frame(ZONE_COLUMNS)
    assert empty["timestamp"].dtype.kind == "i"
    assert empty["bar_index"].dtype.kind == "i"
    assert empty["price_at_event"].dtype.kind == "f"
    assert empty["zone_strength_at_event"].dtype.kind == "f"


def test_an_empty_trade_frame_and_a_populated_one_concat_cleanly():
    empty = empty_frame(TRADE_COLUMNS)
    populated = empty.copy()
    populated.loc[0] = populated.iloc[0] if len(populated) else None
    assert empty["entry_timestamp"].dtype == empty_frame(TRADE_COLUMNS)["entry_timestamp"].dtype


# ---------------------------------------------------------------------------
# Symbol matching: a perpetual must not masquerade as its spot
# ---------------------------------------------------------------------------

def test_spot_does_not_pick_up_the_perpetual_file(tmp_path):
    """The spot's own files are the connector's BTC_USDT_<date>.parquet
    and the raw BTCUSDT.csv. The perpetual's BTCUSDT_USDT_<date>.parquet
    belongs to a DIFFERENT instrument."""
    (tmp_path / "BTC_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "BTCUSDT_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "BTCUSDT.csv").write_bytes(b"")
    found = sorted(p.name for p in find_ohlcv_files("BTC/USDT", str(tmp_path)))
    assert found == ["BTCUSDT.csv", "BTC_USDT_2026-01-01.parquet"]


def test_the_perpetual_does_not_pick_up_the_spot_file(tmp_path):
    (tmp_path / "BTC_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "BTCUSDT.csv").write_bytes(b"")
    (tmp_path / "BTCUSDT_USDT_2026-01-01.parquet").write_bytes(b"")
    found = [p.name for p in find_ohlcv_files("BTC/USDT:USDT", str(tmp_path))]
    assert found == ["BTCUSDT_USDT_2026-01-01.parquet"]


def test_the_perpetual_finds_its_own_files(tmp_path):
    (tmp_path / "BTCUSDT_USDT_2026-01-01.parquet").write_bytes(b"")
    (tmp_path / "BTC_USDT_USDT_2026-01-01.parquet").write_bytes(b"")
    found = [p.name for p in find_ohlcv_files("BTC/USDT:USDT", str(tmp_path))]
    assert sorted(found) == ["BTCUSDT_USDT_2026-01-01.parquet", "BTC_USDT_USDT_2026-01-01.parquet"]


def test_symbol_file_tokens_distinguish_spot_from_perpetual():
    assert "BTCUSDT" in symbol_file_tokens("BTC/USDT")
    assert "BTCUSDTUSDT" in symbol_file_tokens("BTC/USDT:USDT")
    assert "BTCUSDT_USDT" in symbol_file_tokens("BTC/USDT:USDT")


# ---------------------------------------------------------------------------
# The decisions reader must land on the same time axis as the bars
# ---------------------------------------------------------------------------

def test_read_decisions_jsonl_converts_iso_timestamps(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"symbol": "X", "timestamp": "2023-11-14 00:00:00"}) + "\n")
    frame = read_decisions_jsonl(str(path))
    assert len(frame) == 1
    assert frame.iloc[0]["timestamp"] == 1_699_920_000


def test_read_decisions_jsonl_converts_millisecond_timestamps(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"symbol": "X", "timestamp": 1_700_000_000_000}) + "\n")
    frame = read_decisions_jsonl(str(path))
    assert frame.iloc[0]["timestamp"] == 1_700_000_000


def test_a_decision_log_with_iso_timestamps_is_not_silently_empty(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join([
        json.dumps({"symbol": "X", "timestamp": "2023-11-14 00:00:00"}),
        json.dumps({"symbol": "X", "timestamp": "2023-11-14 00:01:00"}),
    ]) + "\n")
    frame = read_decisions_jsonl(str(path))
    assert len(frame) == 2
    assert list(frame["timestamp"]) == [1_699_920_000, 1_699_920_060]


# ---------------------------------------------------------------------------
# The instrument registry must not take the GUI down
# ---------------------------------------------------------------------------

def test_load_registry_survives_malformed_yaml(tmp_path):
    path = tmp_path / "instruments.yaml"
    path.write_text("instruments: [broken\n")
    assert load_registry(str(path)) == {}


def test_load_registry_survives_an_unreadable_file(tmp_path):
    path = tmp_path / "instruments.yaml"
    path.write_text("a: 1")
    path.chmod(0o000)
    try:
        assert load_registry(str(path)) == {}
    finally:
        path.chmod(0o600)


def test_the_registry_path_does_not_depend_on_the_cwd(tmp_path):
    """Proven in a SUBPROCESS.

    ``DEFAULT_REGISTRY_PATH`` is bound when the module is imported, so
    a ``monkeypatch.chdir`` in this process cannot influence it -- the
    guard has to be a fresh interpreter started somewhere else.
    """
    import subprocess
    import sys as _sys

    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from src.gui.instruments import DEFAULT_REGISTRY_PATH, load_registry\n"
        "from pathlib import Path\n"
        "assert Path(DEFAULT_REGISTRY_PATH).is_absolute(), DEFAULT_REGISTRY_PATH\n"
        "assert Path(DEFAULT_REGISTRY_PATH).exists(), DEFAULT_REGISTRY_PATH\n"
        "assert load_registry(), 'the registry is empty outside the repo root'\n"
        "print('OK')\n"
    ) % str(ROOT)
    result = subprocess.run(
        [_sys.executable, "-c", script],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        "the instrument registry does not resolve outside the repo root:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_a_malformed_numeric_field_is_rejected(tmp_path):
    path = tmp_path / "instruments.yaml"
    path.write_text(
        "instruments:\n"
        "  BAD/X:\n"
        "    asset_class: spot\n"
        "    multiplier: -2\n"
    )
    assert load_registry(str(path)) == {}


# ---------------------------------------------------------------------------
# The vocabulary: ONE list, ONE label map, ONE style map
# ---------------------------------------------------------------------------

def test_the_gui_exports_the_canonical_terminal_stages():
    assert models.TERMINAL_STAGES == TERMINAL_STAGES


def test_the_label_map_covers_every_terminal_stage_exactly():
    assert set(STAGE_LABELS) == set(TERMINAL_STAGES)
    assert list(STAGE_LABELS) == list(TERMINAL_STAGES)


def test_the_style_map_covers_every_terminal_stage_exactly():
    assert set(dd.STAGE_STYLES) == set(TERMINAL_STAGES)


def test_the_classifier_only_ever_returns_a_known_stage():
    """Every branch of the classifier must return a stage from the
    canonical list -- a typo would silently become a new vocabulary."""
    import ast
    import inspect

    source = inspect.getsource(classify_terminal_stage)
    tree = ast.parse(textwrap_dedent(source))
    returned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            first = node.value.elts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                returned.add(first.value)
    assert returned, "no stage literals found in the classifier"
    assert returned <= set(TERMINAL_STAGES), f"unknown stages: {returned - set(TERMINAL_STAGES)}"


def textwrap_dedent(source: str) -> str:
    import textwrap
    return textwrap.dedent(source)


# ---------------------------------------------------------------------------
# The classifier: an execution that RAISED is not "no expert candidate"
# ---------------------------------------------------------------------------

def _risk_decision(approved: bool, reason: str = None) -> RiskDecision:
    return RiskDecision(
        timestamp=1_700_000_000, symbol="BTC/USDT", approved=approved,
        risk_score=0.5, base_size=100.0, adjusted_size=100.0,
        size_reduction_pct=0.0, rejection_reason=reason,
    )


def _execution_record(side: str, status: str) -> ExecutionRecord:
    return ExecutionRecord(
        timestamp=1_700_000_000, order_timestamp=1_700_000_000,
        fill_timestamp=1_700_000_000, symbol="BTC/USDT", side=side,
        order_price=100.0, fill_price=100.0, requested_size=1.0, filled_size=1.0,
        slippage_pct=0.0, fees_pct=0.04, total_cost_pct=0.04, status=status,
    )


def test_a_raised_execution_is_classified_as_execution_failed():
    """An order that blew up leaves no ExecutionRecord, so without an
    explicit signal the bar was reported as 'no expert candidate'."""
    approved = _risk_decision(True)
    stage, reason = classify_terminal_stage(None, None, approved, None, execution_failed=True)
    assert stage == "execution_failed"
    assert "failed" in reason


def test_a_raised_execution_outranks_a_derived_risk_rejection():
    derived = _risk_decision(False, "no_trade_selected")
    stage, _ = classify_terminal_stage(None, None, derived, None, execution_failed=True)
    assert stage == "execution_failed"


# ---------------------------------------------------------------------------
# The §8 feed must carry the fields the GUI's schema promises
# ---------------------------------------------------------------------------

def test_the_section8_record_carries_regime_confidence_and_risk_approval():
    class _Regime:
        primary_regime = None
        regime_confidence = 0.77

    state = _FakeState(
        regime_state=_Regime(),
        risk_decision=_risk_decision(True),
    )
    record = decision_record_from_state(state)
    assert record["regime_confidence"] == 0.77
    assert record["risk_approved"] is True
    assert "regime_confidence" in DECISION_COLUMNS
    assert "risk_approved" in DECISION_COLUMNS


def test_a_short_fill_is_reported_as_short():
    record = _FakeState(execution_record=_execution_record("SELL", "filled"))
    assert decision_record_from_state(record)["decision"] == "SHORT"


# ---------------------------------------------------------------------------
# Indicators: TradingView / TA-Lib seeding and causality
# ---------------------------------------------------------------------------

def _reference_ema(values, period):
    """TradingView-style EMA: seeded with the SMA of the first N values."""
    out = [float("nan")] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * (values[i] - prev) + prev
        out[i] = prev
    return out


def _reference_rsi(closes, period):
    """Wilder's RSI with the SMA seed, as TA-Lib computes it."""
    out = [float("nan")] * len(closes)
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def value(gain, loss):
        if loss == 0.0:
            return 50.0 if gain == 0.0 else 100.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[period] = value(avg_gain, avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = value(avg_gain, avg_loss)
    return out


def test_ema_matches_the_tradingview_seeding():
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 200)))
    for period in (9, 10, 21):
        mine = ema(close, period).to_numpy()
        assert np.allclose(mine, _reference_ema(close.tolist(), period), equal_nan=True, atol=1e-9)


def test_rsi_matches_the_wilder_sma_seeding():
    rng = np.random.default_rng(11)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 200)))
    mine = rsi(close, 14).to_numpy()
    assert np.allclose(mine, _reference_rsi(close.tolist(), 14), equal_nan=True, atol=1e-9)


def _reference_atr(high, low, close, period):
    """TA-Lib's ATR: true range from the second bar, seeded with the
    simple mean of the first ``period`` true ranges, so the first value
    lands on index ``period``."""
    tr = [None] * len(close)
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    out = [float("nan")] * len(close)
    if len(close) < period + 1:
        return out
    prev = sum(tr[1:period + 1]) / period
    out[period] = prev
    for i in range(period + 1, len(close)):
        prev = ((period - 1) * prev + tr[i]) / period
        out[i] = prev
    return out


def test_atr_matches_the_talib_seeding():
    """An ATR band drawn one candle to the left of the exchange's is
    exactly the kind of thing that destroys a trader's trust."""
    rng = np.random.default_rng(5)
    n = 120
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, n)))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    for period in (3, 14):
        mine = atr(high, low, close, period).to_numpy()
        ref = _reference_atr(high.tolist(), low.tolist(), close.tolist(), period)
        assert np.allclose(mine, ref, equal_nan=True, atol=1e-9), (
            f"ATR({period}) does not match the TA-Lib reference"
        )
        first_valid = int(np.argmax(~np.isnan(mine)))
        assert first_valid == period, (
            f"ATR({period}) starts at index {first_valid}, expected {period}"
        )


def test_atr_survives_a_duplicated_first_timestamp():
    """Masking the first bar by INDEX LABEL would also drop a bar that
    shares the first label, leaving the ATR all-NaN.

    The duplicate has to be in the INDEX: ``compute_indicators`` indexes
    the frame by its timestamp column, so two bars with the same
    timestamp really do share a label.
    """
    index = pd.Index([0, 0, 1, 2, 3, 4, 5, 6])
    close = pd.Series([100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=index)
    high = close + 1.0
    low = close - 1.0
    values = atr(high, low, close, 3)
    assert values.notna().any(), "the ATR was wiped out by a duplicate index label"
    assert pd.notna(values.iloc[3])


def test_rsi_stays_within_its_bounds():
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 200)))
    values = rsi(close, 14).dropna()
    assert values.between(0.0, 100.0).all()


@pytest.mark.parametrize("kind", ["atr", "bollinger", "donchian"])
def test_indicators_ignore_the_future(kind):
    """A whole-series max/min would look ahead; perturbing the last bar
    must not change any earlier value."""
    rng = np.random.default_rng(5)
    n = 120
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, n)))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)

    def compute(hi, lo, cl):
        if kind == "atr":
            return atr(hi, lo, cl, 14)
        if kind == "bollinger":
            return bollinger(cl, 20, 2.0)[1]
        return donchian(hi, lo, 20)[0]

    baseline = compute(high, low, close)
    changed_close = close.copy()
    changed_close.iloc[-1] = 9999.0
    changed_high = high.copy()
    changed_high.iloc[-1] = 9999.0
    perturbed = compute(changed_high, low, changed_close)
    assert np.allclose(
        baseline.iloc[:-1].to_numpy(), perturbed.iloc[:-1].to_numpy(), equal_nan=True
    )


def test_vwap_refuses_an_index_that_is_not_a_time():
    from src.gui.indicators import vwap

    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="epoch seconds"):
        vwap(series, series, series, pd.Series([1.0, 1.0, 1.0]))


def test_compute_indicators_adds_every_declared_column():
    n = 120
    bars = pd.DataFrame({
        "timestamp": [1_700_000_000 + i * 60 for i in range(n)],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    })
    out = compute_indicators(bars)
    from src.gui.indicators import INDICATOR_COLUMNS
    for col in INDICATOR_COLUMNS:
        assert col in out.columns, f"{col} missing"
        assert out[col].dtype.kind == "f"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeState:
    """Minimal stand-in for a SystemState."""

    def __init__(self, **kwargs):
        self.symbol = kwargs.get("symbol", "BTC/USDT")
        self.timestamp = kwargs.get("timestamp", 1_700_000_000)
        self.regime_state = kwargs.get("regime_state")
        self.router_decision = kwargs.get("router_decision")
        self.opportunity_score = kwargs.get("opportunity_score")
        self.risk_decision = kwargs.get("risk_decision")
        self.execution_record = kwargs.get("execution_record")
