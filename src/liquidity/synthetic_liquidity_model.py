"""
Synthetic liquidity model (Phase 4, §4.2) -- resolves Phase 3's Finding #2
for backtesting: historical replay has no real order-book history (there's
no way to know what BTC/USDT's spread and depth actually were three years
ago), so this produces a config-driven, feature-conditioned PROXY instead
of silently defaulting to zero depth / a fixed 0.2% spread (the bug
Finding #2 identified).

This is a heuristic, and it is labeled as one everywhere it's used --
every MarketSnapshot produced through attach_synthetic_liquidity() carries
liquidity_source="synthetic", so nothing downstream can mistake it for a
real order book. It exists to stop the Opportunity Filter and Risk Layer
from rejecting every single bar during backtesting, not to make backtested
PnL numbers trustworthy in an absolute sense.

Two-pass design note: estimate_synthetic_liquidity() takes a FeatureVector
(needs volatility_5m and volume_zscore), but FeatureVector is computed
FROM a MarketSnapshot downstream in the pipeline -- a snapshot has to
exist before features can be computed from it. attach_synthetic_liquidity()
resolves this the same way Phase 2's structural features do (§6.4 there):
build the snapshot with a crude placeholder first (MarketDataLayer's
uncorrected fallback), compute features from it, then replace the
snapshot's liquidity fields via dataclasses.replace() once real inputs
exist. Nothing that reads high/low/close/volume/timestamp is affected by
this second pass; only bid/ask/bid_volume/ask_volume/spread_pct/
liquidity_source change.
"""
from dataclasses import replace

from ..state.contracts import FeatureVector, MarketSnapshot
from .types import LiquidityEstimate


def estimate_synthetic_liquidity(fv: FeatureVector, bar: MarketSnapshot, cfg: dict) -> LiquidityEstimate:
    base_spread_pct = cfg.get("base_spread_pct", 0.015)
    k_volatility = cfg.get("k_volatility", 0.8)
    k_illiquidity = cfg.get("k_illiquidity", 0.02)
    min_vz_floor = cfg.get("min_volume_zscore_floor", 0.1)
    k_depth = cfg.get("k_depth", 3.0)
    participation = cfg.get("depth_participation_rate", 0.15)

    volume_zscore = (fv.raw_features or {}).get("volume_zscore", 0.0)
    illiquidity_denom = max(abs(volume_zscore), min_vz_floor)
    spread_pct = base_spread_pct + k_volatility * fv.volatility_5m + k_illiquidity / illiquidity_denom
    spread_pct = max(spread_pct, 0.0)

    depth_usdt = k_depth * (bar.volume * bar.close) * participation
    depth_base_ccy = (depth_usdt / bar.close) if bar.close else 0.0

    return LiquidityEstimate(
        spread_pct=spread_pct,
        bid_volume=depth_base_ccy / 2,
        ask_volume=depth_base_ccy / 2,
        source="synthetic",
    )


def attach_synthetic_liquidity(bar: MarketSnapshot, fv: FeatureVector, cfg: dict) -> MarketSnapshot:
    """Return a new MarketSnapshot with bid/ask/bid_volume/ask_volume/
    spread_pct/liquidity_source replaced by the synthetic estimate."""
    estimate = estimate_synthetic_liquidity(fv, bar, cfg)
    mid_price = bar.close
    half_spread_frac = (estimate.spread_pct / 100) / 2
    bid = mid_price * (1 - half_spread_frac)
    ask = mid_price * (1 + half_spread_frac)

    return replace(
        bar,
        bid=bid, ask=ask,
        bid_volume=estimate.bid_volume, ask_volume=estimate.ask_volume,
        spread_pct=estimate.spread_pct,
        liquidity_source="synthetic",
    )
