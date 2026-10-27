"""
Sources, the headline summary, the CLI and the app's safety rails
(Phase 2.5 / 3).

Nothing here runs the pipeline: sources are exercised with the hand-built
fixture bundle and monkeypatched readers.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.gui import sources
from tests.unit.gui_fixtures import make_bundle, ts

ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_the_summary_counts_closed_and_open_trades():
    summary = sources.summarize_session(make_bundle())
    assert summary["n_trades"] == 3
    assert summary["n_closed"] == 2 and summary["n_open"] == 1
    assert summary["win_rate_pct"] == pytest.approx(50.0)
    assert summary["total_pnl_usdt"] == pytest.approx(2.5)
    assert summary["profit_factor"] == pytest.approx(1.5)
    assert summary["expectancy_pct"] == pytest.approx(0.25)
    assert summary["max_drawdown_pct"] == pytest.approx(5.0 / 10_007.5 * 100)


def test_the_funnel_percentages_add_up():
    funnel = sources.summarize_session(make_bundle())["funnel"]
    assert sum(row["bars"] for row in funnel) == 120
    assert sum(row["pct"] for row in funnel) == pytest.approx(100.0)
    assert [row["stage"] for row in funnel] == [
        "opportunity_filter", "position_open", "executed"]


def test_an_empty_session_summarises_without_dividing_by_zero():
    from src.gui.instruments import default_spec
    from src.gui.models import SessionBundle

    summary = sources.summarize_session(SessionBundle(symbol="X", instrument=default_spec("X")))
    assert summary["n_trades"] == 0 and summary["win_rate_pct"] is None
    assert summary["profit_factor"] is None and summary["funnel"] == []


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def test_the_base_timeframe_is_inferred_from_the_bar_spacing():
    bars = make_bundle().bars
    assert sources.infer_base_timeframe(bars) == "1m"
    five = pd.DataFrame({"timestamp": [i * 300 for i in range(10)]})
    assert sources.infer_base_timeframe(five) == "5m"
    assert sources.infer_base_timeframe(five.head(2), default="30s") == "30s"


def test_with_no_data_the_auto_source_is_synthetic(monkeypatch):
    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: pd.DataFrame())
    assert sources.auto_source("BTC/USDT") == "synthetic"
    assert not sources.artifacts_available("BTC/USDT")


def test_raw_bars_without_artifacts_means_replay(monkeypatch):
    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: make_bundle().bars)
    monkeypatch.setattr(sources, "read_trade_journal", lambda: pd.DataFrame())
    monkeypatch.setattr(sources, "read_decisions_jsonl", lambda: pd.DataFrame())
    assert sources.auto_source("BTC/USDT") == "replay"


def test_artifacts_without_bars_is_a_clear_message_not_a_crash(monkeypatch):
    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: pd.DataFrame())
    with pytest.raises(sources.SourceUnavailable, match="OHLCV"):
        sources.artifacts_session("BTC/USDT", config={})


def test_artifacts_with_bars_but_no_run_is_a_clear_message(monkeypatch):
    from src.gui.models import DECISION_COLUMNS, TRADE_COLUMNS, empty_frame

    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: make_bundle().bars)
    monkeypatch.setattr(sources, "read_trade_journal", lambda: empty_frame(TRADE_COLUMNS))
    monkeypatch.setattr(sources, "read_decisions_jsonl", lambda: empty_frame(DECISION_COLUMNS))
    with pytest.raises(sources.SourceUnavailable, match="no trades"):
        sources.artifacts_session("BTC/USDT", config={})


@pytest.fixture
def artifacts(monkeypatch):
    fixture = make_bundle()
    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: fixture.bars)
    monkeypatch.setattr(sources, "read_trade_journal", lambda: fixture.trades)
    monkeypatch.setattr(sources, "read_decisions_jsonl", lambda: fixture.decisions)
    monkeypatch.setattr(sources, "read_decision_traces", lambda: fixture.traces)
    monkeypatch.setattr(sources, "read_zone_events", lambda: fixture.zones)
    return fixture


def test_an_artifacts_session_is_built_from_what_is_on_disk(artifacts):
    bundle = sources.artifacts_session("BTC/USDT", config={"system": {"data_granularity": "1m"}})
    assert bundle.meta["source"] == "artifacts" and not bundle.is_synthetic
    assert len(bundle.bars) == len(artifacts.bars) and len(bundle.trades) == 3
    assert set(bundle.traces) == set(artifacts.traces)


def test_derived_equity_steps_at_each_exit(artifacts):
    bundle = sources.artifacts_session("BTC/USDT", config={})
    by_ts = dict(zip(bundle.equity["timestamp"], bundle.equity["equity"], strict=True))
    assert by_ts[ts(21)] == pytest.approx(10_000.0)
    assert by_ts[ts(22)] == pytest.approx(10_007.5)
    assert by_ts[ts(50)] == pytest.approx(10_002.5)


def test_an_old_artifacts_session_is_flagged_stale_and_equity_as_derived(artifacts):
    bundle = sources.artifacts_session("BTC/USDT", config={})
    assert any(w.startswith("STALE") for w in bundle.warnings)
    assert any("DERIVED" in w for w in bundle.warnings)


def test_trades_outside_the_bar_range_are_dropped(artifacts, monkeypatch):
    late = artifacts.bars.iloc[30:].reset_index(drop=True)
    monkeypatch.setattr(sources, "load_symbol_ohlcv", lambda symbol: late)
    bundle = sources.artifacts_session("BTC/USDT", config={})
    assert list(bundle.trades["trade_id"]) == ["t40", "t100"]


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown source"):
        sources.load_session("nope", "BTC/USDT")


def test_the_sources_module_imports_no_ui_library():
    text = (ROOT / "src/gui/sources.py").read_text()
    assert "import streamlit" not in text and "import plotly" not in text


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def cli(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    module = importlib.import_module("trade_map")
    monkeypatch.setattr(sources, "load_session", lambda *a, **k: make_bundle())
    monkeypatch.setattr(sources, "auto_source", lambda symbol: "synthetic")
    monkeypatch.setattr(sources, "load_config", lambda env="paper": {})
    return module


def test_the_cli_prints_a_summary(cli, capsys):
    assert cli.main(["--source", "synthetic"]) == 0
    out = capsys.readouterr().out
    assert "BTC/USDT" in out and "win rate: 50.00%" in out
    assert "where each bar stopped" in out and "ILLUSTRATIVE" in out


def test_the_cli_writes_one_self_contained_html(cli, tmp_path, capsys):
    pytest.importorskip("plotly")
    out = tmp_path / "map.html"
    assert cli.main(["--source", "synthetic", "--out", str(out), "--timeframe", "5m"]) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "Plotly" in html
    assert "Wrote" in capsys.readouterr().out


def test_the_cli_refuses_to_overwrite_a_production_artifact(cli):
    for target in ("data/registry.db", "data/logs/decisions.jsonl", "data/raw/BTCUSDT.html"):
        with pytest.raises(SystemExit, match="production artifact|must be an .html"):
            cli.main(["--out", str(ROOT / target)])


def test_the_cli_only_writes_html(cli, tmp_path):
    with pytest.raises(SystemExit, match=".html"):
        cli.main(["--out", str(tmp_path / "map.csv")])


def test_the_cli_says_so_when_there_is_nothing_to_show(cli, monkeypatch, capsys):
    def nothing(*args, **kwargs):
        raise sources.SourceUnavailable("no OHLCV file")

    monkeypatch.setattr(sources, "load_session", nothing)
    assert cli.main(["--source", "replay"]) == 0
    assert "Nothing to show" in capsys.readouterr().out


def test_the_default_timeframe_keeps_the_candle_count_reasonable(cli):
    bundle = make_bundle()
    assert cli.default_timeframe(bundle) == "1m"        # 120 bars -> no resampling needed


# ---------------------------------------------------------------------------
# the viewer cannot trade
# ---------------------------------------------------------------------------

GUI_FILES = [*(ROOT / "src" / "gui").glob("*.py"), ROOT / "scripts" / "trade_map.py",
             ROOT / "scripts" / "trade_map_app.py"]

FORBIDDEN = ("create_order", "LiveBackend", "live_backend", "PaperLiveBackend", "paper_live_backend",
             "ccxt", "record_trade_entry", "record_trade_exit", "log_experiment")


@pytest.mark.parametrize("path", GUI_FILES, ids=lambda p: p.name)
def test_no_gui_module_can_reach_an_exchange_or_write_the_journal(path):
    text = path.read_text()
    for token in FORBIDDEN:
        assert token not in text, f"{path.name} mentions {token}"


def test_a_replay_is_forced_to_paper_sim_even_if_the_config_says_live(monkeypatch):
    from src.gui import replay as replay_mod

    seen = {}

    class Stop(Exception):
        pass

    def fake_pipeline(*args, **kwargs):
        seen["mode"] = kwargs["config"]["execution"]["mode"]
        raise Stop

    monkeypatch.setattr(replay_mod, "TradingPipeline", fake_pipeline)
    config = {"execution": {"mode": "live", "confirm_real_money": True}}
    with pytest.raises(Stop):
        replay_mod.replay_session(make_bundle().bars, "BTC/USDT", config)
    assert seen["mode"] == "paper_sim"


def test_the_replay_risk_store_is_not_persistent():
    text = (ROOT / "src/gui/replay.py").read_text()
    assert "RiskStateStore(persistent=False)" in text


# ---------------------------------------------------------------------------
# ports and packaging
# ---------------------------------------------------------------------------

def test_the_app_never_uses_streamlits_default_port():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(ROOT / "scripts"))
    app = importlib.import_module("trade_map_app")
    assert app.DEFAULT_PORT == 8572
    assert app.DEFAULT_PORT not in (8501, 8571)
    assert app.HOST == "127.0.0.1"


def test_the_port_is_overridable_by_environment(monkeypatch):
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(ROOT / "scripts"))
    app = importlib.import_module("trade_map_app")
    monkeypatch.setenv("TRADE_MAP_PORT", "8600")
    assert app.port() == 8600
    monkeypatch.delenv("TRADE_MAP_PORT")
    assert app.port() == 8572


def test_the_makefile_exposes_the_targets_and_the_port():
    makefile = (ROOT / "Makefile").read_text()
    assert "TRADE_MAP_PORT ?= 8572" in makefile
    assert "trade-map:" in makefile and "trade-map-export:" in makefile
    assert "--server.port=8501" not in makefile


def test_plotly_and_streamlit_are_optional_not_core():
    text = (ROOT / "pyproject.toml").read_text()
    core = text.split("[project.optional-dependencies]")[0]
    assert "plotly" not in core and "streamlit" not in core
    assert "plotly" in text.split("viz = [")[1].split("]")[0]
    assert "plotly" not in (ROOT / "requirements.txt").read_text()


def test_the_gui_package_imports_without_plotly():
    init = (ROOT / "src/gui/__init__.py").read_text()
    for forbidden in ("import plotly", "import streamlit", "from .charts", "from . import charts",
                      "from .export", "from . import export"):
        assert forbidden not in init, forbidden
    for name in ("models", "timeframes", "indicators", "artifacts", "instruments", "replay",
                 "sources", "overlays", "theme"):
        assert "plotly" not in (ROOT / "src" / "gui" / f"{name}.py").read_text(), name
