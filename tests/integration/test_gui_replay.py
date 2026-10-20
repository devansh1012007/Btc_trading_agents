"""
Integration test: the Trade Map GUI's replay engine.

The replay is the GUI's own copy of the deployment path -- same pipeline,
same risk layer, same paper-sim execution, same position lifecycle. This
test pins that down: the replay must produce the same trades as the
backtest harness, and it must not touch the production registry.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.gui.models import DECISION_COLUMNS, TRADE_COLUMNS  # noqa: E402
from src.gui.replay import replay_session, synthetic_ohlcv  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def config():
    return load_layered_config("paper")


@pytest.fixture(scope="module")
def session(config):
    """One replay over the same synthetic series the backtest harness uses."""
    bars = synthetic_ohlcv(900, seed=42)
    return replay_session(bars, "BTC/USDT", config, relax=True, synthetic=True)


def test_replay_produces_a_complete_session(session):
    assert not session.bars.empty
    assert not session.decisions.empty
    assert session.traces
    assert list(session.decisions.columns) == DECISION_COLUMNS
    assert list(session.trades.columns) == TRADE_COLUMNS
    assert session.meta["source"] == "replay"
    assert session.meta["synthetic"] is True


def test_replay_finds_trades_and_marks_them_executed(session):
    assert len(session.trades) > 0, "expected the replay to take at least one trade"
    executed = session.decisions[session.decisions["terminal_stage"] == "executed"]
    assert len(executed) == len(session.trades)

    entry_timestamps = set(session.trades["entry_timestamp"])
    assert entry_timestamps == set(executed["timestamp"])


def test_every_trade_lands_on_a_real_bar(session):
    bar_timestamps = set(session.bars["timestamp"])
    for trade in session.trades.itertuples():
        assert trade.entry_timestamp in bar_timestamps
        assert trade.exit_timestamp in bar_timestamps
        assert trade.exit_timestamp >= trade.entry_timestamp


def test_trade_exits_at_its_triple_barrier_levels(session):
    """The exit price must be the TP level, the SL level, or the bar close
    on timeout -- in the direction the trade was actually taken."""
    for trade in session.trades.itertuples():
        sign = 1 if trade.direction == 1 else -1
        tp_level = trade.entry_price * (1 + sign * trade.take_profit_pct / 100)
        sl_level = trade.entry_price * (1 - sign * trade.stop_loss_pct / 100)
        if trade.exit_reason == "take_profit":
            assert trade.exit_price == pytest.approx(tp_level, rel=1e-9)
        elif trade.exit_reason == "stop_loss":
            assert trade.exit_price == pytest.approx(sl_level, rel=1e-9)
        else:
            assert trade.exit_reason == "timeout"


def test_decisions_cover_every_processed_bar(session):
    assert set(session.decisions["timestamp"]) == set(session.traces.keys())
    assert len(session.decisions) <= len(session.bars)


def test_equity_curve_tracks_the_trades(session):
    assert not session.equity.empty
    assert list(session.equity.columns) == ["timestamp", "equity"]
    final = session.equity["equity"].iloc[-1]
    expected = session.meta["initial_equity"] + session.trades["pnl_usdt"].sum()
    assert final == pytest.approx(expected, rel=1e-9)


def test_replay_does_not_write_to_the_production_registry(config, tmp_path, monkeypatch):
    """The GUI is a viewer: replaying must never touch data/registry.db."""
    monkeypatch.chdir(tmp_path)
    bars = synthetic_ohlcv(200, seed=11)
    replay_session(bars, "BTC/USDT", config, relax=True)
    assert not (tmp_path / "data" / "registry.db").exists()


def test_replay_never_places_a_live_order(config, monkeypatch):
    """Even if the config asks for live mode, the replay forces paper_sim."""
    import src.gui.replay as replay_mod

    captured = {}
    original = replay_mod.build_execution_engine

    def spy(cfg, connector, symbol):
        captured["mode"] = cfg.get("execution", {}).get("mode")
        return original(cfg, connector, symbol)

    monkeypatch.setattr(replay_mod, "build_execution_engine", spy)

    live_cfg = {**config, "execution": {"mode": "live", "confirm_real_money": True}}
    bars = synthetic_ohlcv(150, seed=5)
    session = replay_session(bars, "BTC/USDT", live_cfg, relax=True)

    assert captured["mode"] == "paper_sim"
    # Exact, not a subset: a subset check passes vacuously when the
    # trade set is empty.
    assert set(session.trades["mode"].dropna().unique()) == {"paper_sim"}
    assert not session.trades.empty, "expected the replay to take trades"


def test_a_position_still_open_at_the_end_is_still_reported(config):
    """A trade that never closed is a trade the GUI must still show --
    with an entry marker and no exit marker."""
    # Unreachable barriers (and no regime-specific overrides) keep the
    # first position open for the whole session.
    hold = {
        **config,
        "labeling": {
            **config.get("labeling", {}),
            "max_holding_bars": 10_000,
            "k_tp": 1000.0,
            "k_sl": 1000.0,
            "regime_specific_barriers": {},
        },
    }
    bars = synthetic_ohlcv(900, seed=42)
    session = replay_session(bars, "BTC/USDT", hold, relax=True)

    assert not session.trades.empty
    open_rows = session.trades[session.trades["exit_reason"] == "open"]
    assert len(open_rows) == 1, "expected exactly one still-open position"
    row = open_rows.iloc[0]
    assert pd.isna(row["exit_timestamp"])
    assert pd.isna(row["exit_price"])

    # Every executed bar must still have a matching trade row.
    executed = session.decisions[session.decisions["terminal_stage"] == "executed"]
    assert set(executed["timestamp"]) == set(session.trades["entry_timestamp"])


def test_replay_leaves_the_production_registry_byte_identical(config):
    """The GUI is a viewer: the real registry must not change."""
    import hashlib

    registry = ROOT / "data" / "registry.db"
    before = hashlib.sha256(registry.read_bytes()).hexdigest() if registry.exists() else None

    bars = synthetic_ohlcv(200, seed=17)
    replay_session(bars, "BTC/USDT", config, relax=True)

    after = hashlib.sha256(registry.read_bytes()).hexdigest() if registry.exists() else None
    assert before == after


def test_synthetic_ohlcv_is_deterministic():
    a = synthetic_ohlcv(50, seed=3)
    b = synthetic_ohlcv(50, seed=3)
    assert a.equals(b)
    c = synthetic_ohlcv(50, seed=4)
    assert not a.equals(c)


def test_gui_replay_labels_a_position_open_bar_as_position_open(config):
    """A bar that happened while a position was open must say so.

    The runtime says "position already open"; if the GUI says
    "no expert candidate" it is naming a stage that had nothing to do
    with why the bar did not trade.
    """
    hold = {
        **config,
        "labeling": {
            **config.get("labeling", {}),
            "max_holding_bars": 10_000,
            "k_tp": 1000.0,
            "k_sl": 1000.0,
            "regime_specific_barriers": {},
        },
    }
    bars = synthetic_ohlcv(900, seed=42)
    session = replay_session(bars, "BTC/USDT", hold, relax=True)

    open_rows = session.trades[session.trades["exit_reason"] == "open"]
    assert len(open_rows) == 1
    entry_ts = int(open_rows.iloc[0]["entry_timestamp"])

    # The bar after the entry is the first bar that was blocked by the
    # open position.
    later = session.decisions[session.decisions["timestamp"] > entry_ts]
    assert not later.empty
    assert (later["terminal_stage"] == "position_open").any(), (
        "the GUI never reported an open position: "
        f"{later['terminal_stage'].value_counts().to_dict()}"
    )
    # And it must NOT claim there was no expert candidate for those bars.
    blocked = later[later["terminal_stage"] == "position_open"]
    assert set(blocked["expert"].dropna()) != {"no_trade"}


def test_gui_replay_never_reports_an_unfilled_order_as_a_trade(config, monkeypatch):
    """A rejected order is not a trade: the decision column must stay
    NO_TRADE so the chart does not draw a phantom entry marker."""
    from src.gui import replay as replay_mod

    original = replay_mod.build_execution_engine

    def spy(cfg, connector, symbol):
        engine = original(cfg, connector, symbol)
        backend = engine.backend
        # Capture the ORIGINAL bound method BEFORE patching it. A closure
        # that calls `backend.place_order(...)` after the patch calls
        # ITSELF, so the "rejected order" never happens -- the order
        # raises RecursionError instead, which exercises a different
        # branch entirely.
        place_order = backend.place_order

        def reject(*args, **kwargs):
            record = place_order(*args, **kwargs)
            return type(record)(
                timestamp=record.timestamp, order_timestamp=record.order_timestamp,
                fill_timestamp=record.fill_timestamp, symbol=record.symbol,
                side=record.side, order_price=record.order_price,
                fill_price=record.fill_price, requested_size=record.requested_size,
                filled_size=0.0, slippage_pct=record.slippage_pct,
                fees_pct=record.fees_pct, total_cost_pct=record.total_cost_pct,
                status="rejected", order_id=record.order_id,
            )

        monkeypatch.setattr(backend, "place_order", reject)
        return engine

    monkeypatch.setattr(replay_mod, "build_execution_engine", spy)
    # The same series the other tests use, so it does produce entries.
    bars = synthetic_ohlcv(900, seed=42)
    session = replay_session(bars, "BTC/USDT", config, relax=True)

    # The system DID want to trade -- the orders were attempted and
    # rejected -- and none of it may be reported as a trade.
    rejected = session.decisions[session.decisions["terminal_stage"] == "execution_failed"]
    assert len(rejected) > 0, "the replay never attempted an order; the test proves nothing"
    assert set(rejected["terminal_reason"]) == {"status=rejected"}, (
        f"expected a REJECTED order, got {set(rejected['terminal_reason'])}"
    )
    assert set(session.decisions["decision"].dropna().unique()) <= {"NO_TRADE"}
    assert not (session.decisions["terminal_stage"] == "executed").any()
    assert session.trades.empty, "a rejected order must not produce a trade"


def test_gui_replay_traces_carry_latency_measurements(session):
    """A replayed trace must be indistinguishable from a live one."""
    for trace in session.traces.values():
        assert trace["total_duration_ms"] is not None
        assert trace["total_duration_ms"] > 0
        assert trace["stage_timings_ms"], "no per-stage timings were recorded"
        assert "features_ms" in trace["stage_timings_ms"]
        break
    # And the stage records themselves carry their own duration.
    stages = next(iter(session.traces.values()))["stages"]
    assert stages
    assert any(s.get("duration_ms") for s in stages), (
        "no stage carried a duration_ms"
    )


def test_gui_replay_does_not_write_into_the_working_directory(config, tmp_path, monkeypatch):
    """A viewer must not scatter data directories into the CWD."""
    monkeypatch.chdir(tmp_path)
    bars = synthetic_ohlcv(120, seed=8)
    replay_session(bars, "BTC/USDT", config, relax=True)
    assert not (tmp_path / "data").exists(), "the replay created a data/ directory"


def test_gui_replay_accepts_iso_and_millisecond_timestamps(config):
    """The market data layer reads an INTEGER epoch, so a caller that
    hands the replay an ISO column (or a millisecond column) used to
    get a hard crash."""
    import pandas as pd

    bars = synthetic_ohlcv(300, seed=31)
    epoch = bars["timestamp"].astype("int64")
    iso = pd.to_datetime(epoch, unit="s", utc=True)
    millis = epoch * 1000

    results = []
    for frame in (
        bars,
        bars.assign(timestamp=iso),
        bars.assign(timestamp=millis),
    ):
        session = replay_session(frame, "BTC/USDT", config, relax=True)
        results.append((len(session.decisions), list(session.decisions["timestamp"])[:5]))

    assert results[0] == results[1] == results[2], (
        "the same bars produced different sessions depending on the "
        f"timestamp shape: {results}"
    )


def test_the_replay_frames_share_the_empty_frames_dtypes(session):
    """Every frame the GUI draws must concat with its empty counterpart
    without pandas upcasting a column."""
    from src.gui.models import (
        BAR_COLUMNS,
        DECISION_COLUMNS,
        EQUITY_COLUMNS,
        TRADE_COLUMNS,
        ZONE_COLUMNS,
        empty_frame,
    )

    for frame, columns in (
        (session.bars, BAR_COLUMNS),
        (session.decisions, DECISION_COLUMNS),
        (session.trades, TRADE_COLUMNS),
        (session.zones, ZONE_COLUMNS),
        (session.equity, EQUITY_COLUMNS),
    ):
        empty = empty_frame(columns)
        for col in columns:
            assert empty[col].dtype == frame[col].dtype, (
                f"{col}: the empty frame is {empty[col].dtype} but the "
                f"populated frame is {frame[col].dtype}"
            )
        merged = pd.concat([empty, frame], ignore_index=True)
        for col in columns:
            assert merged[col].dtype == frame[col].dtype, (
                f"{col}: concatenating changed the dtype to {merged[col].dtype}"
            )


def test_gui_replay_reports_a_raised_order_as_execution_failed(config, monkeypatch):
    """The GUI's own except branch must classify the bar as a failed
    execution -- the same label the runtime uses."""
    from src.gui import replay as replay_mod

    original = replay_mod.build_execution_engine

    def spy(cfg, connector, symbol):
        engine = original(cfg, connector, symbol)
        backend = engine.backend
        place_order = backend.place_order

        def explode(*args, **kwargs):
            place_order(*args, **kwargs)
            raise RuntimeError("exchange unreachable")

        monkeypatch.setattr(backend, "place_order", explode)
        return engine

    monkeypatch.setattr(replay_mod, "build_execution_engine", spy)
    bars = synthetic_ohlcv(900, seed=42)
    session = replay_session(bars, "BTC/USDT", config, relax=True)

    failed = session.decisions[
        session.decisions["terminal_stage"] == "execution_failed"
    ]
    assert len(failed) > 0, "the GUI never reported a failed execution"
    assert all(failed["decision"] == "NO_TRADE")
    assert any("exchange unreachable" in str(r) for r in failed["terminal_reason"])
