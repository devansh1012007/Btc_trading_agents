"""
Tick and top-of-book quote contracts (`TickRecord`, `QuoteRecord`), quarantine
validation (`validate_and_quarantine_ticks`), and `TickSource` readers
(Phase 13, T13.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..execution.price_path import PathStep, TickPath


@dataclass(frozen=True)
class TickRecord:
    timestamp_ms: int
    price: float
    size: float
    side: str = "BUY"

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ms / 1000.0


@dataclass(frozen=True)
class QuoteRecord:
    timestamp_ms: int
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ms / 1000.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 100.0 if m > 0 else 0.0


@dataclass
class TickValidationResult:
    valid_ticks: List[TickRecord] = field(default_factory=list)
    valid_quotes: List[QuoteRecord] = field(default_factory=list)
    quarantined: List[Dict[str, Any]] = field(default_factory=list)


def validate_and_quarantine_ticks(
    ticks: Iterable[TickRecord],
    quotes: Iterable[QuoteRecord] = (),
    *,
    max_out_of_order_ms: int = 0,
) -> TickValidationResult:
    """Validate ticks and quotes; quarantine malformed, non-positive, crossed-book,
    or out-of-order records without raising."""
    res = TickValidationResult()

    last_tick_ms: Optional[int] = None
    for idx, t in enumerate(ticks):
        if t.price <= 0 or not (t.price == t.price):
            res.quarantined.append({"kind": "tick", "index": idx, "reason": "non_positive_price", "record": t})
            continue
        if t.size <= 0 or not (t.size == t.size):
            res.quarantined.append({"kind": "tick", "index": idx, "reason": "non_positive_size", "record": t})
            continue
        if last_tick_ms is not None and t.timestamp_ms < last_tick_ms - max_out_of_order_ms:
            res.quarantined.append({"kind": "tick", "index": idx, "reason": "out_of_order_timestamp", "record": t})
            continue
        last_tick_ms = max(last_tick_ms, t.timestamp_ms) if last_tick_ms is not None else t.timestamp_ms
        res.valid_ticks.append(t)

    last_quote_ms: Optional[int] = None
    for idx, q in enumerate(quotes):
        if q.bid <= 0 or q.ask <= 0:
            res.quarantined.append({"kind": "quote", "index": idx, "reason": "non_positive_quote", "record": q})
            continue
        if q.bid >= q.ask:
            res.quarantined.append({"kind": "quote", "index": idx, "reason": "crossed_book", "record": q})
            continue
        if last_quote_ms is not None and q.timestamp_ms < last_quote_ms - max_out_of_order_ms:
            res.quarantined.append({"kind": "quote", "index": idx, "reason": "out_of_order_timestamp", "record": q})
            continue
        last_quote_ms = max(last_quote_ms, q.timestamp_ms) if last_quote_ms is not None else q.timestamp_ms
        res.valid_quotes.append(q)

    return res


class TickSource:
    """In-memory or file-backed chronological tick + quote source."""

    def __init__(
        self,
        ticks: Sequence[TickRecord],
        quotes: Sequence[QuoteRecord] = (),
        *,
        validate: bool = True,
    ) -> None:
        if validate:
            v = validate_and_quarantine_ticks(ticks, quotes)
            self.ticks = v.valid_ticks
            self.quotes = v.valid_quotes
            self.quarantined = v.quarantined
        else:
            self.ticks = list(ticks)
            self.quotes = list(quotes)
            self.quarantined = []

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, validate: bool = True) -> "TickSource":
        ticks: List[TickRecord] = []
        quotes: List[QuoteRecord] = []
        for row in df.to_dict("records"):
            ts_ms = int(row.get("timestamp_ms", int(float(row.get("timestamp", 0)) * 1000)))
            if "price" in row and pd.notna(row["price"]):
                ticks.append(
                    TickRecord(
                        timestamp_ms=ts_ms,
                        price=float(row["price"]),
                        size=float(row.get("size", row.get("volume", 1.0))),
                        side=str(row.get("side", "BUY")),
                    )
                )
            if "bid" in row and "ask" in row and pd.notna(row["bid"]) and pd.notna(row["ask"]):
                quotes.append(
                    QuoteRecord(
                        timestamp_ms=ts_ms,
                        bid=float(row["bid"]),
                        ask=float(row["ask"]),
                        bid_size=float(row.get("bid_size", row.get("bid_volume", 1.0))),
                        ask_size=float(row.get("ask_size", row.get("ask_volume", 1.0))),
                    )
                )
        return cls(ticks, quotes, validate=validate)

    @classmethod
    def from_file(cls, path: str | Path, *, validate: bool = True) -> "TickSource":
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        return cls.from_dataframe(df, validate=validate)

    def to_tick_path(self, default_spread_pct: float = 0.02) -> TickPath:
        steps: List[PathStep] = []
        q_idx = 0
        n_q = len(self.quotes)
        last_q: Optional[QuoteRecord] = None
        for t in self.ticks:
            while q_idx < n_q and self.quotes[q_idx].timestamp_ms <= t.timestamp_ms:
                last_q = self.quotes[q_idx]
                q_idx += 1
            if last_q is not None:
                bid, ask = last_q.bid, last_q.ask
                sp = last_q.spread_pct
                depth = (last_q.bid_size + last_q.ask_size) * last_q.mid
            else:
                half = default_spread_pct / 200.0
                bid = t.price * (1.0 - half)
                ask = t.price * (1.0 + half)
                sp = default_spread_pct
                depth = t.size * t.price
            steps.append(
                PathStep(
                    timestamp=t.timestamp_seconds,
                    open=t.price,
                    high=t.price,
                    low=t.price,
                    close=t.price,
                    bid=bid,
                    ask=ask,
                    spread_pct=sp,
                    volume=t.size,
                    depth_usdt=depth,
                )
            )
        return TickPath(steps)
