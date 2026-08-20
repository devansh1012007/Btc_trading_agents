"""
Health check CLI (§11, M4.11): one command, one glance. Current mode, all
circuit breaker states, time since last successfully processed bar, any
pending_approval registry entries, today's trade count and PnL.
"""
import argparse
import os
from datetime import datetime, timezone

from src.utils.config_helpers import load_layered_config
from src.layers.risk_layer import RiskStateStore
from src.registry.model_registry import get_pending
from src.feedback.trade_journal import summarize


def main():
    parser = argparse.ArgumentParser(description="One-glance system health check")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    args = parser.parse_args()

    config = load_layered_config(args.env)
    store = RiskStateStore()

    print(f"{'='*60}")
    print(f"  BTC Stage-3 Trading System -- Status ({args.env})")
    print(f"{'='*60}")

    print(f"\nMode: execution.mode={config.get('execution', {}).get('mode')}  "
          f"sandbox={config.get('system', {}).get('sandbox')}  "
          f"confirm_real_money={config.get('execution', {}).get('confirm_real_money')}")

    kill_switch_path = "data/control/EMERGENCY_STOP"
    print(f"\nManual kill switch: {'ACTIVE (' + kill_switch_path + ' exists)' if os.path.exists(kill_switch_path) else 'clear'}")

    print("\nCircuit breakers:")
    halts = store.active_halts()
    if not halts:
        print("  all clear")
    for h in halts:
        print(f"  TRIPPED: {h['breaker_type']}  reason={h['reason']}  since={h['tripped_at']}")

    breaker_cfg = config.get("risk", {}).get("circuit_breakers", {})
    print(f"\n  consecutive_execution_errors: {store.consecutive_execution_errors()} "
          f"(halts at {breaker_cfg.get('consecutive_execution_errors_halt', 3)})")
    print(f"  consecutive_losses: {store.consecutive_losses()} "
          f"(soft pause at {breaker_cfg.get('consecutive_losses_soft_pause', 3)}, "
          f"hard halt at {breaker_cfg.get('consecutive_losses_hard_halt', 6)})")
    print(f"  daily_loss_pct: {store.daily_loss_pct():.3f}% "
          f"(limit {breaker_cfg.get('daily_loss_limit_pct', 2.0)}%)")
    print(f"  current_drawdown_pct: {store.current_drawdown_pct():.3f}% "
          f"(limit {breaker_cfg.get('max_drawdown_pct', 8.0)}%)")

    staleness_threshold = breaker_cfg.get("data_staleness_halt_seconds", 120)
    is_stale = store.is_stale(staleness_threshold)
    print(f"  data feed: {'STALE (>' + str(staleness_threshold) + 's since last bar)' if is_stale else 'fresh'}")

    print("\nPending model approvals:")
    pending = get_pending()
    if not pending:
        print("  none")
    for p in pending:
        print(f"  {p['component']}: {p['version_id']} (trained {p['trained_at']})")

    print("\nToday's trading:")
    summary = summarize()
    print(f"  trades: {summary['trades_today']}  "
          f"PnL: {summary['total_pnl_usdt_today']:.2f} USDT  "
          f"win rate: {summary['win_rate_today']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
