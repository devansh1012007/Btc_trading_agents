"""
Reset a tripped circuit breaker (§5.3). Deliberately a separate, explicit
manual step -- breakers that require manual reset (max_drawdown,
consecutive_execution_errors, consecutive_losses_hard) don't clear
themselves just because conditions improved; you have to look and decide.
"""
import argparse

from src.layers.risk_layer import RiskStateStore


def main():
    parser = argparse.ArgumentParser(description="Reset a circuit breaker")
    parser.add_argument("--type", required=True,
                         choices=["consecutive_execution_errors", "max_drawdown", "consecutive_losses_hard",
                                  "daily_loss_limit", "data_staleness"],
                         dest="breaker_type")
    parser.add_argument("--db-path", default="data/registry.db")
    args = parser.parse_args()

    store = RiskStateStore(db_path=args.db_path)
    was_active = store.is_tripped(args.breaker_type)
    store.reset(args.breaker_type)
    print(f"Breaker '{args.breaker_type}': was_active={was_active} -> reset.")


if __name__ == "__main__":
    main()
