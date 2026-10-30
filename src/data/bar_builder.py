"""
Tick-to-bar aggregator (`BarBuilder`, Phase 13 T13.2): builds `1s` (or `N`-second)
`MarketSnapshot` bars from `TickRecord` + `QuoteRecord` streams with real
top-of-book `bid`/`ask`/`spread_pct`, carries forward empty seconds with
`volume=0.0`, and never mixes ticks from `[t+1, ...)` into bar `t`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Sequence

from ..state.contracts import MarketSnapshot
from .ticks import QuoteRecord, TickRecord, TickSource


class BarBuilder:
    """Causal aggregator turning sub-second ticks and quotes into whole-second `MarketSnapshot` bars."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        bar_seconds: int = 1,
        carry_forward_empty: bool = True,
        default_spread_pct: float = 0.02,
    ) -> None:
        if bar_seconds < 1:
            raise ValueError(f"bar_seconds must be >= 1, got {bar_seconds}")
        self.symbol = symbol
        self.bar_seconds = int(bar_seconds)
        self.carry_forward_empty = bool(carry_forward_empty)
        self.default_spread_pct = float(default_spread_pct)

    def build_bars(
        self,
        ticks: Sequence[TickRecord],
        quotes: Sequence[QuoteRecord] = (),
    ) -> List[MarketSnapshot]:
        if not ticks:
            return []
        sorted_ticks = sorted(ticks, key=lambda t: t.timestamp_ms)
        sorted_quotes = sorted(quotes, key=lambda q: q.timestamp_ms)

        first_sec = (sorted_ticks[0].timestamp_ms // 1000 // self.bar_seconds) * self.bar_seconds
        last_sec = (sorted_ticks[-1].timestamp_ms // 1000 // self.bar_seconds) * self.bar_seconds

        t_idx = 0
        q_idx = 0
        n_t = len(sorted_ticks)
        n_q = len(sorted_quotes)

        last_close = sorted_ticks[0].price
        last_quote: Optional[QuoteRecord] = None
        bars: List[MarketSnapshot] = []

        for bucket_ts in range(first_sec, last_sec + self.bar_seconds, self.bar_seconds):
            bucket_end_ms = (bucket_ts + self.bar_seconds) * 1000
            bucket_ticks: List[TickRecord] = []
            while t_idx < n_t and sorted_ticks[t_idx].timestamp_ms < bucket_end_ms:
                if sorted_ticks[t_idx].timestamp_ms >= bucket_ts * 1000:
                    bucket_ticks.append(sorted_ticks[t_idx])
                t_idx += 1

            while q_idx < n_q and sorted_quotes[q_idx].timestamp_ms < bucket_end_ms:
                last_quote = sorted_quotes[q_idx]
                q_idx += 1

            if not bucket_ticks and not self.carry_forward_empty:
                continue

            if bucket_ticks:
                o = bucket_ticks[0].price
                h = max(x.price for x in bucket_ticks)
                low = min(x.price for x in bucket_ticks)
                c = bucket_ticks[-1].price
                vol = sum(x.size for x in bucket_ticks)
                last_close = c
            else:
                o = h = low = c = last_close
                vol = 0.0

            if last_quote is not None:
                bid = last_quote.bid
                ask = last_quote.ask
                bid_vol = last_quote.bid_size
                ask_vol = last_quote.ask_size
                spread_pct = last_quote.spread_pct
                liq_src = "real_orderbook"
            else:
                half = self.default_spread_pct / 200.0
                bid = c * (1.0 - half)
                ask = c * (1.0 + half)
                bid_vol = vol * 0.5
                ask_vol = vol * 0.5
                spread_pct = self.default_spread_pct
                liq_src = "synthetic"

            dt = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)
            bars.append(
                MarketSnapshot(
                    timestamp=int(bucket_ts),
                    symbol=self.symbol,
                    open=float(o),
                    high=float(h),
                    low=float(low),
                    close=float(c),
                    volume=float(vol),
                    bid=float(bid),
                    ask=float(ask),
                    bid_volume=float(bid_vol),
                    ask_volume=float(ask_vol),
                    spread_pct=float(spread_pct),
                    hour_utc=dt.hour,
                    day_of_week=dt.weekday(),
                    liquidity_source=liq_src,
                )
            )
        return bars

    def build_from_source(self, source: TickSource) -> List[MarketSnapshot]:
        return self.build_bars(source.ticks, source.quotes)
