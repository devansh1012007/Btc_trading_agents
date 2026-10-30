"""
Market-agnostic OHLCV data adapters (`ParquetOHLCVAdapter`, `CSVOHLCVAdapter`)
and corporate-action split detector (`detect_unadjusted_splits`, Phase 15 T15.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from ..layers.market_data import MarketDataLayer
from ..state.contracts import MarketSnapshot
from .profile import MarketProfile


@dataclass(frozen=True)
class SplitSuspect:
    timestamp: int
    prev_close: float
    open_price: float
    jump_pct: float
    inferred_ratio: str


def detect_unadjusted_splits(
    df_ohlcv: pd.DataFrame,
    *,
    split_jump_threshold_pct: float = 35.0,
) -> List[SplitSuspect]:
    """Detect likely unadjusted stock splits / reverse splits from overnight close-to-open jumps."""
    if df_ohlcv.empty or len(df_ohlcv) < 2:
        return []
    closes = df_ohlcv["close"].to_numpy(dtype=float)
    opens = df_ohlcv["open"].to_numpy(dtype=float)
    timestamps = df_ohlcv["timestamp"].to_numpy(dtype=int)

    suspects: List[SplitSuspect] = []
    for i in range(1, len(df_ohlcv)):
        prev_c = closes[i - 1]
        curr_o = opens[i]
        if prev_c <= 0 or curr_o <= 0:
            continue
        jump_pct = abs(curr_o - prev_c) / prev_c * 100.0
        if jump_pct >= split_jump_threshold_pct:
            ratio = prev_c / curr_o
            ratio_str = f"{round(ratio)}:1" if ratio >= 1.0 else f"1:{round(1.0 / ratio)}"
            suspects.append(
                SplitSuspect(
                    timestamp=int(timestamps[i]),
                    prev_close=float(prev_c),
                    open_price=float(curr_o),
                    jump_pct=float(jump_pct),
                    inferred_ratio=ratio_str,
                )
            )
    return suspects


class OHLCVFileAdapter:
    """Loads CSV or Parquet OHLCV bars and filters by `MarketProfile.is_session_open` when configured."""

    def __init__(self, symbol: str, profile: MarketProfile | None = None, filter_session: bool = False) -> None:
        self.symbol = symbol
        self.profile = profile or MarketProfile.preset("crypto_perp")
        self.filter_session = bool(filter_session)

    def load_dataframe(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        if self.filter_session and not self.profile.is_24_7 and "timestamp" in df.columns:
            mask = [self.profile.is_session_open(int(ts)) for ts in df["timestamp"]]
            df = df.loc[mask].reset_index(drop=True)
        return df

    def load_snapshots(self, path: str | Path) -> Sequence[MarketSnapshot]:
        df = self.load_dataframe(path)
        return MarketDataLayer.batch_create_from_dataframe(df, self.symbol)
