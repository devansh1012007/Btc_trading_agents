"""
Readers for the artifacts a live/paper run leaves behind.

Everything here is **read-only**: the GUI must never mutate the trade
journal, the risk state or the logs. The SQLite connection is opened
with the ``mode=ro`` URI so a missing or locked database degrades to an
empty result instead of creating a file.

The readers normalise into the column vocabularies declared in
:mod:`src.gui.models`, so the charts never care whether a row came from
the trade journal, a decision JSONL line or a backtest.
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from ..diagnostics.dashboard_data import load_decision_traces
from ..feedback.zone_event_log import load_zone_events
from ..registry.db import DEFAULT_DB_PATH
from .models import (
    BAR_COLUMNS,
    DECISION_COLUMNS,
    NUMERIC_DECISION_COLUMNS,
    NUMERIC_TRADE_COLUMNS,
    TRADE_COLUMNS,
    ZONE_COLUMNS,
    coerce_numeric,
    empty_frame,
    normalize_dtypes,
)
from .timeframes import to_epoch_seconds

logger = logging.getLogger(__name__)

#: Repository root, derived from this file's location so the artifacts
#: resolve the same way no matter which directory the GUI is launched
#: from -- exactly how ``instruments.DEFAULT_REGISTRY_PATH`` does it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DECISIONS_JSONL = "data/logs/decisions.jsonl"
TRACE_DIR = "data/logs/decision_traces"
ZONE_EVENTS_PATH = "data/logs/zone_events.parquet"
RAW_DIR = "data/raw"


def resolve_artifact_path(relative: str) -> str:
    """Resolve a repo-relative artifact path.

    The repository root wins; the current working directory is the
    fallback (a run launched elsewhere -- the e2e tests, a container --
    leaves its ``data/`` next to where it ran). An absolute path is
    returned untouched, and when neither location exists the repo-root
    path is returned so the "missing" log message names a stable place.
    """
    path = Path(relative)
    if path.is_absolute():
        return str(path)
    repo_candidate = _REPO_ROOT / path
    if repo_candidate.exists():
        return str(repo_candidate)
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str(repo_candidate)

#: trade_journal column -> normalised column.
_JOURNAL_COLUMN_MAP = {
    "trade_id": "trade_id",
    "entry_timestamp": "entry_timestamp",
    "exit_timestamp": "exit_timestamp",
    "entry_price": "entry_price",
    "exit_price": "exit_price",
    "direction": "direction",
    "expert_used": "expert",
    "regime_at_entry": "regime",
    "size": "size",
    "realized_pnl_pct": "pnl_pct",
    "realized_pnl_usdt": "pnl_usdt",
    "fees_paid": "fees_paid",
    "slippage_realized": "slippage",
    "exit_reason": "exit_reason",
    "router_confidence": "router_confidence",
    "risk_score_at_entry": "risk_score",
    "mode": "mode",
}

_OHLCV_ALIASES = {
    "timestamp": ("timestamp", "ts", "time", "date", "datetime", "open_time"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "v", "vol"),
}


def _read_only_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """Open a SQLite database read-only, or return None if that is impossible.

    A missing file is the normal "nothing has run yet" case and is logged
    quietly; a file that EXISTS but cannot be opened (locked, corrupt,
    permissions) is logged as a warning -- otherwise it would look
    exactly like "no trades".
    """
    if not Path(db_path).exists():
        logger.debug("No database at %s", db_path)
        return None
    try:
        # as_uri() percent-encodes the path, so a name containing '?' or
        # '#' cannot break the URI.
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        logger.warning("Could not open the database %s read-only: %s", db_path, exc)
        return None


def read_trade_journal(db_path: Optional[str] = None) -> pd.DataFrame:
    """Read ``trade_journal`` into the normalised trade schema.

    ``db_path`` defaults to the repo's ``data/registry.db`` (see
    :func:`resolve_artifact_path`). Returns an empty (but correctly
    shaped) frame when the database or the table is missing; a database
    that exists but cannot be read is logged as a warning.
    """
    db_path = db_path or resolve_artifact_path(DEFAULT_DB_PATH)
    conn = _read_only_connection(db_path)
    if conn is None:
        return empty_frame(TRADE_COLUMNS)

    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_journal'"
        ).fetchone()
        if row is None:
            logger.info("%s has no trade_journal table yet", db_path)
            return empty_frame(TRADE_COLUMNS)
        rows = conn.execute(
            "SELECT * FROM trade_journal ORDER BY entry_timestamp ASC"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Could not read the trade journal in %s: %s", db_path, exc)
        return empty_frame(TRADE_COLUMNS)
    finally:
        conn.close()

    if not rows:
        return empty_frame(TRADE_COLUMNS)

    raw = pd.DataFrame([dict(r) for r in rows])
    renamed = raw.rename(columns=_JOURNAL_COLUMN_MAP)
    keep = [c for c in TRADE_COLUMNS if c in renamed.columns]
    out = renamed[keep].copy()
    for col in TRADE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = coerce_numeric(out[TRADE_COLUMNS], NUMERIC_TRADE_COLUMNS)
    before = len(out)
    out = out.dropna(subset=["entry_timestamp"])
    if len(out) < before:
        logger.warning("Dropped %d trade-journal row(s) with no entry_timestamp",
                       before - len(out))
    out = normalize_dtypes(out)
    return out.reset_index(drop=True)


def read_decisions_jsonl(path: Optional[str] = None) -> pd.DataFrame:
    """Read the per-bar §8 decision records into the normalised schema.

    Tolerates the two shapes the log has taken (the canonical ``src/``
    record and the richer ``tests/src`` superset) by reading every key
    defensively, and normalises the timestamp the same way the OHLCV
    reader does -- epoch seconds, milliseconds, ISO strings and datetime
    objects all land on the same axis, so a decision can be joined to
    its bar.
    """
    # The deployment loop honours $DECISIONS_JSONL when it writes the
    # feed, so the reader must too or the two would look in different
    # places.
    p = Path(path or os.environ.get("DECISIONS_JSONL") or resolve_artifact_path(DECISIONS_JSONL))
    if not p.exists():
        return empty_frame(DECISION_COLUMNS)

    records: List[dict] = []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line mid-write
                if isinstance(record, dict):
                    records.append(record)
    except OSError as exc:
        logger.warning("Could not read the decision log %s: %s", p, exc)
        return empty_frame(DECISION_COLUMNS)

    if not records:
        return empty_frame(DECISION_COLUMNS)

    frame = pd.DataFrame(records)
    for col in DECISION_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = coerce_numeric(frame[DECISION_COLUMNS], NUMERIC_DECISION_COLUMNS)
    frame["timestamp"] = to_epoch_seconds(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"])
    frame = normalize_dtypes(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def read_decision_traces(base_dir: Optional[str] = None) -> Dict[int, dict]:
    """Raw decision traces keyed by bar timestamp."""
    traces = load_decision_traces(base_dir or resolve_artifact_path(TRACE_DIR))
    out: Dict[int, dict] = {}
    for trace in traces:
        ts = trace.get("timestamp")
        if ts is None:
            continue
        try:
            out[int(ts)] = trace
        except (TypeError, ValueError):
            continue
    return out


def read_zone_events(path: Optional[str] = None) -> pd.DataFrame:
    """Zone lifecycle events, normalised to the GUI's column vocabulary."""
    path = path or resolve_artifact_path(ZONE_EVENTS_PATH)
    try:
        frame = load_zone_events(path)
    except Exception as exc:
        logger.warning("Could not read the zone event log %s: %s", path, exc)
        return empty_frame(ZONE_COLUMNS)
    if frame is None or frame.empty:
        return empty_frame(ZONE_COLUMNS)
    for col in ZONE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[ZONE_COLUMNS].copy()
    if "timestamp" in out.columns:
        out["timestamp"] = to_epoch_seconds(out["timestamp"])
        out = out.dropna(subset=["timestamp"])
    return normalize_dtypes(out).reset_index(drop=True)


def _rename_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common OHLCV column spellings onto the canonical names."""
    rename: Dict[str, str] = {}
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for canonical, aliases in _OHLCV_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            if alias in lowered:
                rename[lowered[alias]] = canonical
                break
    return frame.rename(columns=rename) if rename else frame


def _coerce_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the timestamp column to integer unix seconds."""
    if "timestamp" not in frame.columns:
        return frame
    values = to_epoch_seconds(frame["timestamp"])
    out = frame.copy()
    out["timestamp"] = values
    return out.dropna(subset=["timestamp"])


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV bars from a CSV or Parquet file.

    Returns an empty frame (with the GUI's canonical dtypes) when the
    file is missing or unreadable.
    """
    p = Path(path)
    if not p.exists():
        return empty_frame(BAR_COLUMNS)

    try:
        if p.suffix.lower() in (".parquet", ".pq"):
            frame = pd.read_parquet(p)
        else:
            frame = pd.read_csv(p)
    except Exception as exc:
        logger.warning("Could not read the OHLCV file %s: %s", p, exc)
        return empty_frame(BAR_COLUMNS)

    frame = _rename_ohlcv_columns(frame)
    required = ["open", "high", "low", "close"]
    if not all(col in frame.columns for col in required):
        logger.warning("OHLCV file %s is missing the columns %s", p,
                       [c for c in required if c not in frame.columns])
        return empty_frame(BAR_COLUMNS)

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = _coerce_timestamps(frame)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return frame[BAR_COLUMNS].reset_index(drop=True)


def symbol_file_tokens(symbol: str) -> Set[str]:
    """File-name stems that can hold ``symbol``'s bars.

    The repo writes bars as ``{BASE}_{QUOTE}_{date}.parquet`` (see
    ``src/connectors/data_source.py``) and the raw CSVs are named
    ``{BASE}{QUOTE}.csv``. A perpetual adds its settlement currency
    (``BTCUSDT_USDT.csv``).

    The settlement currency is what tells a perpetual apart from its
    spot: a symbol that HAS one matches only the settlement-bearing
    forms, and a symbol that has NONE matches only the plain forms.
    Without that split, ``BTCUSDT.csv`` (the spot) would be handed to
    the perpetual and the GUI would draw the spot's prices as if they
    were the perpetual's.
    """
    raw = str(symbol).strip()
    base, _, rest = raw.partition("/")
    quote, _, settle = rest.partition(":")
    base = base.upper()
    quote = quote.upper()
    settle = settle.upper()

    if settle:
        return {
            base + quote + settle,
            f"{base}_{quote}_{settle}",
            f"{base}{quote}_{settle}",
        }
    return {base + quote, f"{base}_{quote}"}


def _stem_matches_token(stem: str, token: str) -> bool:
    """True when ``stem`` is ``token`` or ``token`` + a period suffix.

    The suffix must start with a digit, so ``BTCUSDT_USDT_2026-01-01``
    is NOT a match for the spot token ``BTCUSDT`` (its remainder starts
    with ``U``) but IS a match for the perpetual token ``BTCUSDT_USDT``.
    """
    if stem == token:
        return True
    if not stem.startswith(token):
        return False
    remainder = stem[len(token):]
    return bool(remainder.startswith("_") and remainder[1:2].isdigit())


def find_ohlcv_files(symbol: str, raw_dir: Optional[str] = None) -> List[Path]:
    """Every OHLCV file that holds ``symbol``'s bars.

    Handles both layouts the repo uses: per-day parquet written by the
    connector (``BTC_USDT_2026-01-01.parquet``) and a single CSV
    (``BTCUSDT.csv``). A perpetual's file is NOT returned for its spot
    symbol, and vice versa.
    """
    directory = Path(raw_dir or resolve_artifact_path(RAW_DIR))
    if not directory.exists():
        return []

    tokens = symbol_file_tokens(symbol)

    found: List[Path] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in (".csv", ".parquet", ".pq"):
            continue
        name = path.stem.upper()
        if any(_stem_matches_token(name, token) for token in tokens):
            found.append(path)
    return found


def load_symbol_ohlcv(symbol: str, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and merge every OHLCV file belonging to ``symbol``."""
    frames = [load_ohlcv(str(path)) for path in find_ohlcv_files(symbol, raw_dir)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_frame(BAR_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return merged.reset_index(drop=True)
