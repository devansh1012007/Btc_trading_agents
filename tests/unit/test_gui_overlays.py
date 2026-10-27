"""
Overlay placement and causality (Phase 2.2 / 2.4).

The fixture's timestamps are chosen so every assertion names the exact
candle a marker must land on, at every timeframe.
"""
import pytest

from src.diagnostics.decision_trace import TERMINAL_STAGES
from src.gui import overlays as ov
from src.gui.timeframes import bucket_start, resample_ohlcv
from tests.unit.gui_fixtures import START, make_bundle, ts

pytestmark = [pytest.mark.unit]

TIMEFRAMES = ["1m", "5m", "15m", "1h"]


@pytest.fixture(scope="module")
def bundle():
    return make_bundle()


def build(bundle, timeframe):
    candles = resample_ohlcv(bundle.bars, timeframe, bundle.base_timeframe)
    return ov.build_overlays(bundle, timeframe, candles), candles


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_every_marker_lands_on_a_real_candle_at_every_timeframe(bundle, timeframe):
    overlays, candles = build(bundle, timeframe)
    candle_ts = set(candles["timestamp"])
    for marker in [*overlays.entries, *overlays.exits, *overlays.decisions]:
        assert marker.ts in candle_ts, f"{type(marker).__name__} at {marker.ts} is off the candle grid"


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_a_timeframe_switch_does_not_move_a_trade_off_its_bar(bundle, timeframe):
    overlays, _ = build(bundle, timeframe)
    by_id = {e.trade_id: e for e in overlays.entries}
    assert by_id["t7"].ts == bucket_start(ts(7), timeframe)
    assert by_id["t40"].ts == bucket_start(ts(40), timeframe)
    assert by_id["t100"].ts == bucket_start(ts(100), timeframe)
    exits = {x.trade_id: x for x in overlays.exits}
    assert exits["t7"].ts == bucket_start(ts(22), timeframe)
    assert exits["t40"].ts == bucket_start(ts(50), timeframe)
    # the exact fill time survives the re-bucketing
    assert by_id["t7"].entry_ts == ts(7)
    assert exits["t7"].exit_ts == ts(22)


def test_the_5m_entry_is_on_the_candle_containing_it(bundle):
    overlays, _ = build(bundle, "5m")
    entry = next(e for e in overlays.entries if e.trade_id == "t7")
    assert entry.ts == START + 5 * 60            # bar 7 is inside minutes 5-9
    assert entry.ts <= ts(7) < entry.ts + 300


def test_an_entry_marker_carries_no_outcome(bundle):
    """Causality: nothing on an entry may reveal how the trade ended."""
    fields = set(ov.EntryMarker.__dataclass_fields__)
    assert not fields & {"exit_ts", "exit_price", "pnl_usdt", "pnl_pct", "reason", "exit_reason"}
    overlays, _ = build(bundle, "1m")
    entry = next(e for e in overlays.entries if e.trade_id == "t7")
    assert entry.take_profit_price == pytest.approx(entry.price * 1.015)
    assert entry.stop_loss_price == pytest.approx(entry.price * 0.99)


def test_short_barriers_are_mirrored(bundle):
    overlays, _ = build(bundle, "1m")
    entry = next(e for e in overlays.entries if e.trade_id == "t40")
    assert entry.direction == -1
    assert entry.take_profit_price == pytest.approx(entry.price * 0.985)
    assert entry.stop_loss_price == pytest.approx(entry.price * 1.01)


def test_the_open_trade_has_an_entry_a_span_and_no_exit(bundle):
    overlays, _ = build(bundle, "1m")
    assert all(x.trade_id != "t100" for x in overlays.exits)
    span = next(s for s in overlays.spans if s.trade_id == "t100")
    assert span.is_open and span.exit_ts is None and span.exit_price is None


def test_a_trade_outside_the_visible_candles_draws_nothing(bundle):
    sliced = make_bundle()
    sliced.bars = sliced.bars[sliced.bars["timestamp"] >= ts(30)].reset_index(drop=True)
    overlays, _ = build(sliced, "1m")
    assert all(e.trade_id != "t7" for e in overlays.entries)
    assert any(e.trade_id == "t40" for e in overlays.entries)


def test_a_decision_candle_shows_the_decision_that_got_furthest(bundle):
    overlays, _ = build(bundle, "15m")
    first = next(d for d in overlays.decisions if d.ts == START)
    # minutes 0-14 contain the executed bar 7, which outranks the rest
    assert first.stage == "executed"
    assert first.n_bars == 15


def test_stage_rank_follows_the_vocabulary_order():
    assert list(ov.STAGE_RANK) == list(TERMINAL_STAGES)
    assert ov.STAGE_RANK["executed"] == max(ov.STAGE_RANK.values())


def test_a_candles_decision_uses_only_bars_inside_it(bundle):
    """Causality: the 5m candle at START must not see bar 7's decision."""
    overlays, _ = build(bundle, "5m")
    first = next(d for d in overlays.decisions if d.ts == START)
    assert first.stage == "opportunity_filter"
    assert first.n_bars == 5


def test_regime_is_the_regime_at_the_candles_open(bundle):
    # The regime flips to chop_range at bar 60, the first bar of the 1h
    # candle START+3600: the hour candle at START opens in trending_up.
    overlays, _ = build(bundle, "1h")
    assert overlays.regimes[0].regime == "trending_up"
    overlays, _ = build(bundle, "5m")
    kinds = [r.regime for r in overlays.regimes]
    assert kinds == ["trending_up", "chop_range"]
    assert overlays.regimes[0].end_ts == overlays.regimes[1].start_ts == START + 3600


def test_zones_are_levels_from_creation_to_the_ending_event(bundle):
    overlays, _ = build(bundle, "5m")
    zones = {z.zone_id: z for z in overlays.zones}
    assert zones["z1"].start_ts == bucket_start(ts(3), "5m")
    assert zones["z1"].end_event == "broke"
    assert zones["z1"].end_ts == bucket_start(ts(30), "5m") + 300
    assert zones["z2"].end_event is None                  # still alive
    assert zones["z2"].end_ts == START + 115 * 60 + 300   # to the end of the last candle
    assert zones["z1"].price == pytest.approx(101.0)


def test_equity_is_the_value_at_each_candles_close(bundle):
    overlays, candles = build(bundle, "5m")
    assert len(overlays.equity.ts) == len(candles)
    by_ts = dict(zip(overlays.equity.ts, overlays.equity.equity, strict=True))
    assert by_ts[bucket_start(ts(22), "5m")] == pytest.approx(10_007.5)   # exit at bar 22 is in this candle
    assert min(overlays.equity.drawdown_pct) == 0.0
    assert max(overlays.equity.drawdown_pct) == pytest.approx(5.0 / 10_007.5 * 100)


def test_no_overlay_carries_nan_or_none_in_plottable_fields(bundle):
    import math
    overlays, _ = build(bundle, "5m")
    for marker in [*overlays.entries, *overlays.exits, *overlays.decisions]:
        assert not math.isnan(marker.price)
    for zone in overlays.zones:
        assert not math.isnan(zone.price)
    assert all(not math.isnan(v) for v in overlays.equity.equity)


def test_building_twice_gives_identical_overlays(bundle):
    a, _ = build(bundle, "15m")
    b, _ = build(bundle, "15m")
    assert a == b


def test_an_empty_session_has_empty_overlays():
    from src.gui.instruments import default_spec
    from src.gui.models import SessionBundle

    empty = SessionBundle(symbol="X", instrument=default_spec("X"))
    overlays = ov.build_overlays(empty, "5m")
    assert not overlays.entries and not overlays.zones and not overlays.equity
