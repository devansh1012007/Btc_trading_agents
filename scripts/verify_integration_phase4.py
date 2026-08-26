"""
Phase 4 integration test (§14, M4.12): full SystemState through
ExecutionRecord, paper_sim mode, 150+ mock bars, Parquet round trip,
every Decision + Risk + Execution field populated for at least the bars
where a trade actually happened.

Uses a diagnostic config relaxation for the Opportunity Filter's
liquidity thresholds (documented in PHASE4_COMPLETE.md): the synthetic
liquidity model's realistic output and the Phase 1-3 default thresholds
were never checked against each other before this phase connected them
for the first time, and turn out to disagree -- this test relaxes them
locally, the same way Phase 3's own integration test did for an
analogous reason, without touching the shipped config.

Run directly: python test_integration_phase4.py
"""
import copy
import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING)  # quiet the per-bar layer-fallback INFO/WARNING noise
logger = logging.getLogger(__name__)

import sys

from src.feedback.system_state_archive import load_system_state_archive
from src.feedback.trade_journal import get_recent_trades
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.pipeline import TradingPipeline
from src.registry.db import DEFAULT_DB_PATH

sys.path.insert(0, "scripts")
from run import build_execution_engine, run_historical


def generate_mock_ohlcv(n_bars: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(n_bars)]

    price = 50000.0
    rows = []
    for ts in timestamps:
        drift = rng.normal(0, 15)
        price = max(price + drift, 100.0)
        high = price + abs(rng.normal(0, 8))
        low = price - abs(rng.normal(0, 8))
        open_ = price + rng.normal(0, 4)
        close = price
        volume = abs(rng.normal(100, 30))
        rows.append({
            "timestamp": ts, "open": open_, "high": max(high, open_, close),
            "low": min(low, open_, close), "close": close, "volume": volume,
        })
    return pd.DataFrame(rows)


def main():
    import os
    if os.path.exists(DEFAULT_DB_PATH):
        os.remove(DEFAULT_DB_PATH)

    with open("config/settings.base.yaml") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    # Diagnostic-only relaxation -- see module docstring.
    config["opportunity_filter"]["hard_rules"]["max_spread_pct"] = 0.3
    config["opportunity_filter"]["hard_rules"]["edge_vs_fees_threshold"] = -1.0
    config["opportunity_filter"]["hard_rules"]["reject_if_best_confidence_below"] = 0.0
    config["opportunity_filter"]["hard_rules"]["no_consensus_ceiling"] = 1.0

    df_ohlcv = generate_mock_ohlcv(n_bars=400)
    print(f"Generated {len(df_ohlcv)} mock OHLCV bars")

    pipeline = TradingPipeline(symbol="BTC/USDT", config=config)
    risk_layer = RiskLayer(config=config.get("risk", {}), store=RiskStateStore(), data_granularity="1m")
    execution_engine = build_execution_engine(config, pipeline.connector, "BTC/USDT")
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)

    run_historical(pipeline, risk_layer, execution_engine, df_ohlcv, "paper", fee_pct, initial_equity=10000.0)

    trades = get_recent_trades(limit=1000)
    print(f"\n{len(trades)} trades recorded in the journal")
    assert len(trades) > 0, "No trades executed -- can't validate risk/execution field population"
    assert all(t["exit_reason"] is not None for t in trades), "Every recorded trade should have resolved"
    print("Trade journal populated correctly")

    archive = load_system_state_archive("BTC/USDT", "2026-01-01")
    print(f"\nSystemState archive: {len(archive)} rows, {len(archive.columns)} columns")
    assert len(archive) > 100

    for prefix in ["market_", "feature_", "regime_", "expert_", "opportunity_", "router_"]:
        assert any(c.startswith(prefix) for c in archive.columns), f"No {prefix}* columns in archive"
    print("Full-stack columns (market/feature/regime/expert/opportunity/router) present")

    risk_populated = archive["risk_approved"].notna().sum()
    execution_populated = archive["execution_status"].notna().sum()
    print(f"\nBars with a risk decision: {risk_populated}")
    print(f"Bars with an execution record: {execution_populated}")
    assert risk_populated > 0, "RiskDecision never populated on any bar"
    assert execution_populated > 0, "ExecutionRecord never populated on any bar"

    assert archive["realized_pnl"].notna().all(), "realized_pnl must be populated on every bar (0.0 for no-trade bars)"
    n_zero = (archive["realized_pnl"] == 0.0).sum()
    n_nonzero = (archive["realized_pnl"] != 0.0).sum()
    print(f"\nrealized_pnl: {n_zero} no-trade/no-exit bars (0.0), {n_nonzero} bars with a resolved trade's PnL")
    assert n_nonzero > 0, "No bar ever carried a nonzero realized_pnl"

    store = RiskStateStore()
    print(f"\nFinal persisted risk state: consecutive_losses={store.consecutive_losses()}, "
          f"daily_loss_pct={store.daily_loss_pct():.3f}%, drawdown={store.current_drawdown_pct():.3f}%")

    print("\nPHASE 4 INTEGRATION TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
