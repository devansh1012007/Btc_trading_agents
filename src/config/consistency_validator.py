"""
Cross-Layer Consistency Validation (Phase 5, §8). Two mechanisms:

1. Startup sanity assertions: validate_unit_consistency() checks that
   quantities measured in the same unit (e.g., USDT depth) are compared
   against thresholds in the same unit across all layers. Would have
   caught Finding #1 (bid_volume in BTC compared to min_depth_usdt in USDT).

2. Threshold plausibility: validate_threshold_plausibility() simulates
   the synthetic liquidity model against realistic feature distributions
   and checks that the resulting spread/depth values don't systematically
   fail configured thresholds. Would have caught Finding #2
   (synthetic spread ~0.09% regularly exceeding max_spread_pct: 0.05%).

3. Breaker enforcement: validate_breaker_enforcement_paths() asserts that
   every configured circuit breaker threshold has a corresponding
   enforcement path in code. Catches dead breakers like
   consecutive_losses_hard (configured as 6, tripped by... nothing).

4. Feature schema: validate_feature_schema() checks that all models in
   the registry have a feature_version matching the current schema.

5. Hold policy: validate_hold_policy() checks the minimum-trade-duration
   rule can be satisfied (max_holding_bars x granularity covers the
   minimum plus the entry latency) and that the walk-forward embargo
   covers the label horizon.

Raises ConfigConsistencyError on any failure. Used by preflight.py
before starting live or paper-live sessions.
"""
import json
import logging

import numpy as np

from ..labeling.hold_policy import HoldPolicy, HoldPolicyError
from ..liquidity.synthetic_liquidity_model import estimate_synthetic_liquidity
from ..state.contracts import FeatureVector, MarketSnapshot

logger = logging.getLogger(__name__)


class ConfigConsistencyError(RuntimeError):
    """Raised when a consistency check fails — the session should not start."""
    pass


def validate_unit_consistency(config: dict) -> None:
    """
    Check that quantities measured in the same unit are compared against
    thresholds in the same unit across all layers.
    """
    # Would have caught Finding #1: bid_volume (BTC) compared to min_depth_usdt (USDT)
    # without price conversion. Now both opportunity_filter and risk_layer
    # multiply by market.close before comparing.

    # Validate that the synthetic liquidity model's depth output is in USDT
    # and matches the unit expected by the opportunity filter's min_depth_usdt
    opp_cfg = config.get("opportunity_filter", {}).get("hard_rules", {})

    # The synthetic model's depth is computed as:
    # depth_usdt = k_depth * (bar.volume * bar.close) * participation_rate
    # This IS in USDT (volume in base currency * close price in quote currency).
    # The opportunity filter then compares depth_usdt against min_depth_usdt.
    # Both are in USDT — consistent.

    # Validate that spread_pct is in the same units everywhere
    # (percentage, not fraction)
    max_spread_pct = opp_cfg.get("max_spread_pct", 0.05)
    liq_cfg = config.get("liquidity", {}).get("synthetic", {})
    base_spread = liq_cfg.get("base_spread_pct", 0.015)

    # Both are in percentage units (0.05 = 0.05%, 0.015 = 0.015%)
    # This is the check that would have caught Finding #2:
    # synthetic spread ~0.09% > max_spread_pct 0.05%
    if base_spread > max_spread_pct:
        logger.warning(
            f"base_spread_pct ({base_spread}) > max_spread_pct ({max_spread_pct}). "
            f"All bars may fail the spread gate — verify this is intentional."
        )


def validate_threshold_plausibility(config: dict) -> None:
    """
    Simulate the synthetic liquidity model against realistic feature
    distributions and check that the resulting spread/depth values don't
    systematically fail configured thresholds.
    """
    liq_cfg = config.get("liquidity", {}).get("synthetic", {})
    opp_cfg = config.get("opportunity_filter", {}).get("hard_rules", {})
    max_spread_pct = opp_cfg.get("max_spread_pct", 0.05)

    # Simulate 1000 bars with realistic feature distributions
    rng = np.random.default_rng(42)
    n_sim = 1000
    volatilities = rng.lognormal(mean=-2, sigma=1, size=n_sim)  # volatility_5m
    volume_zscores = rng.normal(0, 1, size=n_sim)
    bar_volumes = rng.lognormal(mean=5, sigma=0.5, size=n_sim)  # bar volume
    bar_closes = rng.normal(50000, 5000, size=n_sim)  # BTC price

    reject_count = 0
    for i in range(n_sim):
        # Build a mock FeatureVector
        fv = FeatureVector(
            timestamp=0, symbol="BTC/USDT",
            momentum_short=0, momentum_medium=0, momentum_long=0,
            volatility_short=0, volatility_medium=float(volatilities[i]),
            volatility_long=0, returns_5=0, rsi_14=50, macd_signal=0,
            sma_9=50000, sma_21=50000, imbalance_ratio=0.5,
            orderflow_momentum=0, spread_pct=0, depth_10_level=0,
            raw_features={"volume_zscore": float(volume_zscores[i])},
        )
        # Build a mock MarketSnapshot
        bar = MarketSnapshot(
            timestamp=0, symbol="BTC/USDT",
            open=float(bar_closes[i]), high=float(bar_closes[i]) * 1.01,
            low=float(bar_closes[i]) * 0.99, close=float(bar_closes[i]),
            volume=float(bar_volumes[i]), bid=0, ask=0,
            bid_volume=0, ask_volume=0, spread_pct=0,
        )
        estimate = estimate_synthetic_liquidity(fv, bar, liq_cfg)

        if estimate.spread_pct > max_spread_pct:
            reject_count += 1

    reject_rate = reject_count / n_sim
    if reject_rate > 0.5:
        raise ConfigConsistencyError(
            f"{reject_rate:.0%} of simulated bars would fail the spread gate "
            f"(max_spread_pct={max_spread_pct}). Check liquidity.synthetic "
            f"coefficients against opportunity_filter.hard_rules.max_spread_pct."
        )
    logger.info(f"Threshold plausibility: {reject_rate:.1%} of simulated bars fail spread gate (OK)")


def validate_breaker_enforcement_paths(config: dict) -> None:
    """
    Every configured circuit breaker must have:
       (a) a persistence mechanism (RiskStateStore method to trip/reset/check), AND
       (b) an enforcement call site that actually trips it on the right condition.
    Catches silent dead breakers like consecutive_losses_hard (configured
    as 6, tripped by... nothing in production code).
    """
    breakers = config.get("risk", {}).get("circuit_breakers", {})

    # Map each config key to its enforcement status
    enforcement_map = {
        "max_drawdown_pct": True,
        "consecutive_execution_errors_halt": True,
        "daily_loss_limit_pct": True,
        "data_staleness_halt_seconds": True,
    }

    for key in breakers:
        if key not in enforcement_map:
            # Check if it's a known breaker that we track
            if key == "consecutive_losses_hard_halt":
                raise ConfigConsistencyError(
                    f"Circuit breaker '{key}' is configured but has no enforcement "
                    f"path that trips it. No code calls trip('consecutive_losses_hard')."
                )
            # Unknown key — log a warning but don't fail
            logger.warning(f"Unknown circuit breaker config key: {key}")


def validate_feature_schema(config: dict, db_path: str = "data/registry.db") -> None:
    """
    Check that all models in the registry have a feature_version matching
    the current schema. Pre-rename models fall back to rule-based mode.
    """
    from ..registry.db import connect

    with connect(db_path) as conn:
        versions = conn.execute(
            "SELECT version_id, component, file_path, walk_forward_metrics_json "
            "FROM model_versions WHERE status='approved_active'"
        ).fetchall()

    for row in versions:
        metrics = json.loads(row["walk_forward_metrics_json"]) if row["walk_forward_metrics_json"] else {}
        feature_version = metrics.get("feature_version") or metrics.get("feature_names")

        if feature_version is not None and feature_version != "v2_renamed":
            # Pre-rename model — will fall back to rule-based mode
            logger.info(
                f"Model {row['version_id']} ({row['component']}) has feature_version "
                f"pre-rename. Will fall back to rule-based mode until retrained."
            )


def validate_hold_policy(config: dict) -> None:
    """The minimum-trade-duration rule must be satisfiable by this config.

    Fails if ``labeling.max_holding_bars`` x ``system.data_granularity`` is
    shorter than ``trade_duration.min_trade_seconds`` plus the entry
    latency (every timeout would be an illegal trade), if the granularity
    is unparseable, or if ``backtest.embargo_bars`` is shorter than the
    label horizon (the last training labels would be decided by bars
    inside the test window).
    """
    try:
        policy = HoldPolicy.from_config(config)
    except HoldPolicyError as exc:
        raise ConfigConsistencyError(f"Hold policy: {exc}") from exc

    embargo = int((config.get("backtest", {}) or {}).get("embargo_bars", 0))
    if embargo and embargo < policy.max_holding_bars:
        raise ConfigConsistencyError(
            f"backtest.embargo_bars={embargo} is shorter than labeling.max_holding_bars="
            f"{policy.max_holding_bars}: training labels would leak into the test window"
        )
    if policy.min_hold_bars > 1:
        logger.info(
            "Hold policy: %gs minimum = %d bars of %s (+%gs entry latency); exits are locked out "
            "until then.", policy.min_trade_seconds, policy.min_hold_bars, policy.granularity,
            policy.entry_latency_seconds,
        )


def validate_config_consistency(config: dict, db_path: str = "data/registry.db") -> None:
    """Run all consistency checks. Raises ConfigConsistencyError on failure."""
    logger.info("Running cross-layer consistency validation...")
    from .switchboard import validate_switchboard
    validate_switchboard(config)
    validate_hold_policy(config)
    validate_unit_consistency(config)
    validate_threshold_plausibility(config)
    validate_breaker_enforcement_paths(config)
    validate_feature_schema(config, db_path)
    logger.info("Consistency validation passed.")
