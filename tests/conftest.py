"""
Shared test builders.

Only the builders that several test files used to carry their own copy of
live here: the ``sys.path`` bootstrap, the journal database, the
synthetic series, the ``hold`` config and the contract objects. Fixtures
that belong to one file stay in that file.
"""
import dataclasses
import sys
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.registry.db import close_all_connections, close_connection, connect  # noqa: E402
from src.state.contracts import ExecutionRecord, RiskDecision  # noqa: E402
from src.utils.config_helpers import load_layered_config  # noqa: E402
from src.utils.synthetic import synthetic_ohlcv  # noqa: E402


@pytest.fixture(autouse=True)
def _release_pooled_sqlite_connections():
    """The registry pools one SQLite connection per file per thread. A
    test's tmp directory disappears when the test ends, so drop the
    pooled handles with it -- nothing may leak across tests."""
    yield
    close_all_connections()


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def paper_config() -> dict:
    return load_layered_config("paper")


@pytest.fixture(scope="session")
def hold_config(paper_config: dict) -> dict:
    """A config whose first position never exits.

    Unreachable barriers (and no regime-specific overrides) keep the
    first position open for the whole session, so a test can observe a
    still-open trade and the ``position_open`` bars that follow it.
    """
    return {
        **paper_config,
        "labeling": {
            **paper_config.get("labeling", {}),
            "max_holding_bars": 10_000,
            "k_tp": 1000.0,
            "k_sl": 1000.0,
            "regime_specific_barriers": {},
        },
    }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_bars() -> Callable[..., pd.DataFrame]:
    """Factory for the deterministic mixed-regime series the GUI demo uses."""
    return synthetic_ohlcv


_JOURNAL_DEFAULT_ROW = {
    "trade_id": "t1",
    "entry_timestamp": 1_700_000_000,
    "entry_price": 100.0,
    "exit_timestamp": 1_700_000_060,
    "exit_price": 101.5,
    "exit_reason": "take_profit",
    "expert_used": "trend_expert",
    "regime_at_entry": "trending_up",
    "direction": 1,
    "size": 250.0,
    "realized_pnl_pct": 1.35,
    "realized_pnl_usdt": 3.4,
    "fees_paid": 0.12,
    "slippage_realized": 0.01,
    "router_confidence": 0.7,
    "risk_score_at_entry": 0.2,
    "mode": "paper",
}


@pytest.fixture
def make_journal_db() -> Callable[..., Path]:
    """Factory: create a registry database holding ``trade_journal`` rows.

    Goes through ``src.registry.db.connect`` so the table is the REAL
    schema -- a hand-written ``CREATE TABLE`` would silently drift from it.
    ``rows`` is a list of partial dicts merged over a default row; pass
    ``rows=[]`` for an empty journal.
    """
    def _make(path, rows: Optional[List[dict]] = None) -> Path:
        path = Path(path)
        records = [{}] if rows is None else rows
        with connect(str(path)) as conn:
            for record in records:
                row = {**_JOURNAL_DEFAULT_ROW, **record}
                columns = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                conn.execute(
                    f"INSERT INTO trade_journal ({columns}) VALUES ({marks})",
                    tuple(row.values()),
                )
        close_connection(str(path))
        return path

    return _make


# ---------------------------------------------------------------------------
# contract objects
# ---------------------------------------------------------------------------

@pytest.fixture
def make_execution_record() -> Callable[..., ExecutionRecord]:
    def _make(status: str = "filled", side: str = "BUY", **overrides) -> ExecutionRecord:
        record = ExecutionRecord(
            timestamp=1_700_000_000, order_timestamp=1_700_000_000,
            fill_timestamp=1_700_000_000, symbol="BTC/USDT", side=side,
            order_price=100.0, fill_price=100.0, requested_size=100.0,
            filled_size=100.0, slippage_pct=0.0, fees_pct=0.04, total_cost_pct=0.04,
            status=status, order_id="x",
        )
        return dataclasses.replace(record, **overrides) if overrides else record

    return _make


@pytest.fixture
def make_risk_decision() -> Callable[..., RiskDecision]:
    def _make(approved: bool = True, reason: Optional[str] = None) -> RiskDecision:
        return RiskDecision(
            timestamp=1_700_000_000, symbol="BTC/USDT", approved=approved,
            risk_score=0.5, base_size=100.0, adjusted_size=100.0,
            size_reduction_pct=0.0, rejection_reason=reason,
        )

    return _make
