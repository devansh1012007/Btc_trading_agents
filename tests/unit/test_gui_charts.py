"""
Figure builders (Phase 2.3): clean data, correct placement, determinism.

Skipped when the optional ``viz`` extra (plotly) is not installed.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from src.gui import charts, theme  # noqa: E402
from src.gui import overlays as ov  # noqa: E402
from src.gui.timeframes import bucket_start, resample_ohlcv  # noqa: E402
from tests.unit.gui_fixtures import make_bundle, ts  # noqa: E402

pytestmark = [pytest.mark.unit]

TIMEFRAMES = ["1m", "5m", "15m", "1h"]


@pytest.fixture(scope="module")
def bundle():
    return make_bundle()


def price_figure(bundle, timeframe, **kwargs):
    candles = resample_ohlcv(bundle.bars, timeframe, bundle.base_timeframe)
    overlays = ov.build_overlays(bundle, timeframe, candles)
    return charts.build_price_chart(
        bundle.bars, overlays, timeframe, bundle.instrument,
        base_timeframe=bundle.base_timeframe, **kwargs,
    ), candles


def trace(fig, uid):
    return next(t for t in fig.data if t.uid == uid)


def _values(array):
    return [] if array is None else list(array)


def assert_clean(fig):
    """No NaN / None in anything a trace plots."""
    for t in fig.data:
        for attr in ("x", "y", "open", "high", "low", "close"):
            for v in _values(getattr(t, attr, None)):
                assert v is not None, f"{t.uid}.{attr} contains None"
                assert not (isinstance(v, float) and np.isnan(v)), f"{t.uid}.{attr} contains NaN"


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_every_figure_is_free_of_nan_and_none(bundle, timeframe):
    fig, _ = price_figure(bundle, timeframe, indicators=["ema_fast", "bb_upper", "vwap"],
                          panes=["rsi", "macd"])
    assert_clean(fig)


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_the_candle_trace_is_the_resampled_frame(bundle, timeframe):
    fig, candles = price_figure(bundle, timeframe)
    candle = trace(fig, "candles")
    assert len(candle.x) == len(candles)
    assert list(candle.x) == charts.to_datetimes(candles["timestamp"])
    np.testing.assert_allclose(candle.close, candles["close"])


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_entry_markers_sit_on_the_candle_containing_the_fill(bundle, timeframe):
    fig, candles = price_figure(bundle, timeframe)
    longs = trace(fig, "entry-1")
    expected = sorted({bucket_start(ts(7), timeframe), bucket_start(ts(100), timeframe)})
    assert sorted(longs.x) == charts.to_datetimes(expected)
    assert set(longs.x) <= set(trace(fig, "candles").x)
    shorts = trace(fig, "entry--1")
    assert list(shorts.x) == charts.to_datetimes([bucket_start(ts(40), timeframe)])


def test_exit_markers_are_coloured_by_pnl(bundle):
    fig, _ = price_figure(bundle, "1m")
    exits = trace(fig, "exit")
    assert list(exits.marker.color) == [theme.WIN, theme.LOSS]      # +1.5% then -1.0%
    assert "take profit" in exits.customdata[0].lower()
    assert "stop loss" in exits.customdata[1].lower()


def test_the_entry_hover_does_not_leak_the_outcome(bundle):
    fig, _ = price_figure(bundle, "1m")
    for text in trace(fig, "entry-1").text:
        assert "PnL" not in text and "take profit" not in text.lower()
        assert "TP" in text and "SL" in text           # the levels are known at entry


def test_decision_dots_are_one_trace_per_stage_and_skip_quiet_stages(bundle):
    fig, _ = price_figure(bundle, "5m")
    uids = {t.uid for t in fig.data}
    assert "decision-opportunity_filter" in uids and "decision-executed" in uids
    assert "decision-position_open" not in uids           # quiet by default
    fig, _ = price_figure(bundle, "5m", decision_stages=["position_open"])
    assert {t.uid for t in fig.data if str(t.uid).startswith("decision-")} == {"decision-position_open"}


def test_decision_dots_use_the_shared_stage_colours(bundle):
    fig, _ = price_figure(bundle, "5m")
    assert trace(fig, "decision-executed").marker.color == theme.stage_colour("executed")
    assert theme.stage_colour("executed") == theme.STAGE_COLOURS["executed"]


def test_layers_can_be_switched_off(bundle):
    fig, _ = price_figure(bundle, "5m", layers=[])
    uids = {t.uid for t in fig.data}
    assert not any(str(u).startswith(("entry", "exit", "decision")) for u in uids)
    assert len(fig.layout.shapes) == 0


def test_spans_barriers_and_zones_are_shapes_not_none_separated_traces(bundle):
    fig, _ = price_figure(bundle, "5m")
    assert len(fig.layout.shapes) >= 3 + 2 * 3 + 2          # spans + TP/SL + zones
    assert_clean(fig)


def test_overlays_for_another_timeframe_are_refused(bundle):
    overlays = ov.build_overlays(bundle, "5m")
    with pytest.raises(ValueError, match="5m"):
        charts.build_price_chart(bundle.bars, overlays, "15m", bundle.instrument,
                                 base_timeframe="1m")


def test_unknown_indicators_and_panes_are_refused(bundle):
    overlays = ov.build_overlays(bundle, "5m")
    with pytest.raises(ValueError, match="indicator"):
        charts.build_price_chart(bundle.bars, overlays, "5m", bundle.instrument, indicators=["nope"])
    with pytest.raises(ValueError, match="pane"):
        charts.build_price_chart(bundle.bars, overlays, "5m", bundle.instrument, panes=["nope"])


def test_indicators_are_drawn_from_the_resampled_candles(bundle):
    fig, candles = price_figure(bundle, "5m", indicators=["ema_fast"])
    line = trace(fig, "ind-ema_fast")
    assert set(line.x) <= set(trace(fig, "candles").x)
    assert len(line.x) <= len(candles)


def test_the_same_session_renders_the_same_figure(bundle):
    a, _ = price_figure(bundle, "15m", indicators=["ema_fast"], panes=["rsi"])
    b, _ = price_figure(bundle, "15m", indicators=["ema_fast"], panes=["rsi"])
    assert a.to_json() == b.to_json()


def test_every_trace_has_an_explicit_deterministic_uid(bundle):
    fig, _ = price_figure(bundle, "5m", indicators=["ema_fast"], panes=["rsi", "macd"])
    uids = [t.uid for t in fig.data]
    assert all(uids) and len(uids) == len(set(uids))
    assert not any(len(str(u)) == 36 and str(u).count("-") == 4 for u in uids)   # no random uuid


def test_an_empty_session_renders_a_message_not_an_error():
    from src.gui.instruments import default_spec
    from src.gui.models import SessionBundle

    empty = SessionBundle(symbol="X", instrument=default_spec("X"))
    overlays = ov.build_overlays(empty, "5m")
    fig = charts.build_price_chart(empty.bars, overlays, "5m", empty.instrument)
    assert fig.layout.annotations and "No bars" in fig.layout.annotations[0].text


def test_the_decision_strip_has_one_row_per_stage(bundle):
    overlays = ov.build_overlays(bundle, "5m")
    fig = charts.build_decision_chart(bundle, overlays, "5m")
    assert {t.y[0] for t in fig.data} <= {theme.stage_name(s) for s in
                                          ["opportunity_filter", "position_open", "executed"]}
    assert_clean(fig)


def test_the_funnel_counts_every_decided_bar(bundle):
    fig = charts.build_stage_funnel(bundle)
    assert sum(fig.data[0].x) == len(bundle.decisions)


def test_the_equity_chart_plots_equity_and_drawdown(bundle):
    overlays = ov.build_overlays(bundle, "5m")
    fig = charts.build_equity_chart(overlays, "5m", initial_equity=10_000.0)
    assert {t.uid for t in fig.data} == {"equity", "drawdown"}
    assert max(trace(fig, "drawdown").y) <= 0
    assert_clean(fig)


def test_session_figures_cover_all_four_views(bundle):
    figures = charts.build_session_figures(bundle, "15m", indicators=["ema_fast"])
    assert set(figures) == {"price", "decisions", "equity", "funnel"}


def test_the_html_export_is_self_contained_and_stable(bundle):
    from src.gui.export import render_report_html
    from src.gui.sources import summarize_session

    summary = summarize_session(bundle)
    html_a = render_report_html(bundle, charts.build_session_figures(bundle, "15m"), summary, "15m")
    html_b = render_report_html(bundle, charts.build_session_figures(bundle, "15m"), summary, "15m")
    assert html_a == html_b
    assert html_a.count("plotly.js") >= 1
    assert "<script src=" not in html_a and "<link " not in html_a       # nothing fetched
    assert "ILLUSTRATIVE - fixture" in html_a                          # the warning banner
    assert "t7" in html_a and "take profit" in html_a


def test_to_datetimes_is_utc_and_ordered():
    out = charts.to_datetimes([0, 60])
    assert out == ["1970-01-01 00:00:00", "1970-01-01 00:01:00"]
    assert charts.to_datetimes([]) == []


def test_theme_colours_every_terminal_stage():
    from src.diagnostics.decision_trace import TERMINAL_STAGES
    assert set(theme.STAGE_COLOURS) == set(TERMINAL_STAGES)
    assert theme.stage_colour("not-a-stage") == theme.TEXT_MUTED


def test_the_price_chart_accepts_a_pandas_empty_overlay_set(bundle):
    overlays = ov.OverlaySet(timeframe="5m")
    fig = charts.build_price_chart(bundle.bars, overlays, "5m", bundle.instrument,
                                   base_timeframe="1m")
    assert isinstance(fig.data[0].x[0], str)
    assert pd.Timestamp(fig.data[0].x[0]).tzinfo is None
