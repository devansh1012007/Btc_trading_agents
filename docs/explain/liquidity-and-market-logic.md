# Liquidity & Market Logic

## Synthetic Liquidity Model (`src/liquidity/synthetic_liquidity_model.py`)

### Purpose

Feature-conditioned spread/depth proxy for backtesting. Historical replay has no real order-book history, so the crude `MarketDataLayer` fallback (zero depth, fixed ~0.2% spread) is replaced with a feature-conditional estimate.

### Two-Pass Design

The synthetic liquidity model needs `volatility_medium` from the `FeatureVector`, but the `FeatureVector` computation needs a `MarketSnapshot` with bid/ask/depth fields. Resolution:

1. `MarketDataLayer.create_from_bar()` creates snapshot with crude placeholders.
2. Feature engine computes `FeatureVector`.
3. `attach_synthetic_liquidity()` replaces `bid`, `ask`, `bid_volume`, `ask_volume`, `spread_pct` via `dataclasses.replace()`.
4. The corrected snapshot is used downstream.

The structure engine already consumed the pre-liquidity snapshot, which is safe — it never reads bid/ask/spread. The append to `structure_snapshots` happens after the liquidity attachment so everything references the identical corrected object.

### Model

```python
illiquidity_denom = max(abs(volume_zscore), min_volume_zscore_floor)
spread_pct = base_spread_pct + k_volatility * volatility_medium + k_illiquidity / illiquidity_denom
depth_usdt = k_depth * (bar_volume * bar_close) * depth_participation_rate   # quote-currency units
bid_volume = ask_volume = (depth_usdt / bar_close) / 2                        # split symmetrically, base-currency units
```

| Parameter | Default | Meaning |
|---|---|---|
| `base_spread_pct` | 0.015 | Base spread (%) |
| `k_volatility` | 0.8 | Spread widens with `volatility_medium` |
| `k_illiquidity` | 0.02 | Spread widens as |volume z-score| shrinks |
| `min_volume_zscore_floor` | 0.1 | Prevents division blow-up |
| `k_depth` | 3.0 | Depth scaling vs bar notional |
| `depth_participation_rate` | 0.15 | Fraction of bar notional assumed executable |

Depth is stored in **base currency (BTC)** on the snapshot — matching `MarketSnapshot.bid_volume/ask_volume` semantics; consumers multiply by `close` to get USDT before comparing against thresholds.

### `LiquidityEstimate`

```python
@dataclass(frozen=True)
class LiquidityEstimate:
    spread_pct: float
    bid_volume: float
    ask_volume: float
    source: str  # "synthetic"
```

### `attach_synthetic_liquidity()`

Returns a new `MarketSnapshot` (immutable) with updated liquidity fields. Sets `liquidity_source = "synthetic"`.

## Real Orderbook Adapter (`src/liquidity/real_orderbook_adapter.py`)

### `fetch_real_liquidity()`

CCXT `fetch_order_book(symbol, limit=depth_limit)`. Aggregates top N bids/asks:

```python
total_bid = sum(price × quantity for price, quantity in orderbook["bids"][:depth_limit])
total_ask = sum(price × quantity for price, quantity in orderbook["asks"][:depth_limit])
mid = (best_bid + best_ask) / 2
spread_pct = (best_ask - best_bid) / mid * 100
```

### `attach_real_liquidity()`

Returns a new `MarketSnapshot` with real liquidity fields. Sets `liquidity_source = "real_order_book"`.

### Failure Handling

PaperLiveBackend falls back gracefully on network failure:

```python
def get_liquidity(self, symbol, market):
    try:
        return fetch_real_liquidity(self.connector, symbol)
    except Exception as e:
        logger.warning(f"Real order-book fetch failed, falling back to snapshot's existing liquidity: {e}")
        return LiquidityEstimate(
            spread_pct=market.spread_pct,
            bid_volume=market.bid_volume,
            ask_volume=market.ask_volume,
            source=market.liquidity_source,
        )
```

## Liquidity Types (`src/liquidity/types.py`)

```python
@dataclass(frozen=True)
class LiquidityEstimate:
    spread_pct: float
    bid_volume: float
    ask_volume: float
    source: str  # "synthetic" | "real_order_book"

def to_ticker_data(liquidity: LiquidityEstimate, mid: float) -> Dict:
    """Converts to MarketDataLayer.create_from_bar() format."""
    half_spread = mid * liquidity.spread_pct / 200
    return {
        "bid": mid - half_spread,
        "ask": mid + half_spread,
        "bid_volume": liquidity.bid_volume,
        "ask_volume": liquidity.ask_volume,
    }
```

## Depth & Fill Model (`src/execution/execution_engine.py`)

### `simulate_fill()`

```python
spread_frac = spread_pct / 100
if side == "BUY":
    fill_price = mid * (1 + spread_frac/2 + slippage_pct/100)
else:
    fill_price = mid * (1 - spread_frac/2 - slippage_pct/100)

max_fillable_usdt = depth_usdt * participation_cap  # 15%
filled_size_usdt = min(requested_size_usdt, max_fillable_usdt)

status = "filled" if filled >= requested else ("partial" if filled > 0 else "rejected")
```

### Depth Participation Cap

`depth_participation_rate` (default 0.15) caps how much of a bar's synthetic/real depth we're willing to assume we could actually execute against. Prevents overestimating fillable size in thin markets.

### Partial Fill Policy

`execution.partial_fill_policy` in config:
- `"cancel_remainder"` (default): Partial fill only if depth supports full size
- `"market_fill_remainder"`: Fill whatever depth supports, rest at market

## Market Data (`src/connectors/data_source.py`)

### CCXTConnector

Wrapper around CCXT. `enableRateLimit: True` on all exchange instances.

```python
def __init__(self, exchange_name="binance", sandbox=True):
    exchange_class = getattr(ccxt, exchange_name)
    self.exchange = exchange_class({
        'enableRateLimit': True,
        'sandbox': sandbox,
    })
```

### Methods

| Method | Returns | Notes |
|---|---|---|
| `fetch_ohlcv(symbol, timeframe, limit)` | DataFrame | Normalized to columns: timestamp, open, high, low, close, volume, symbol |
| `fetch_ticker(symbol)` | Dict | bid, ask, last, bid_volume, ask_volume |
| `fetch_funding_rate(symbol)` | Optional[float] | None if exchange doesn't support it |
| `save_ohlcv_local(df, symbol, data_dir)` | — | Partitioned by date: `{symbol}_{date}.parquet` |
| `load_ohlcv_local(symbol, date_range, data_dir)` | DataFrame | Loads all matching parquet files, sorted by timestamp |

### Local Caching

Parquet files partitioned by date. `load_ohlcv_local()` glob-matches `{symbol}_{date}.parquet` files, loads and concatenates. Corrupt files are skipped with a warning.

## Calibration (`scripts/calibrate_synthetic_liquidity.py`)

OLS regression to fit liquidity model coefficients against real-vs-synthetic samples. Needs paired data (real order book vs. synthetic estimate) that `paper_live` mode produces.

**Phase 5**: `PaperLiveBackend` now logs paired (synthetic, real) liquidity samples to Parquet. Run `paper_live` mode for a stretch (at minimum a few hundred bars across varied conditions), then run the calibration script. Only after calibration produces fitted coefficients does `opportunity_filter.hard_rules.max_spread_pct` get revisited.

## Known Issues

### Spread Threshold Mismatch

Synthetic liquidity model's mean spread (~0.09%, using plan's own proposed coefficients) regularly exceeds `max_spread_pct: 0.05%` (Phase 1's original default). Two independently-written defaults, connected for the first time by Phase 4's fix, disagree. This needs real spread data for calibration — loosening `max_spread_pct` in shipped config would be tuning a risk gate to manufacture activity rather than reflect real conditions.

### Depth Unit Mismatch (Fixed in Phase 4)

`MarketSnapshot.bid_volume`/`ask_volume` are base-currency units (BTC), but `min_depth_usdt` is a quote-currency threshold. The check compared them directly with no price conversion — `44.8 < 10000` is true regardless of the fact that 44.8 BTC is worth millions of dollars. Fixed in both places it appeared (`evaluate()`'s hard rule and `_liquidity_score()`), both now multiply by `market.close` before comparing.
