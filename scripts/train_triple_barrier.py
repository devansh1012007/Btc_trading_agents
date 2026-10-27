"""
Harvest every expert's training samples THE WAY THE HARNESS USES THEM and
label them with the harness's own exit rule.

For each expert (trend, mean-reversion, breakout), in ONE replay pass:

* population  -- bars where that expert's own ``propose_candidate()`` gate
                 fired AND the Opportunity Filter passed (as in the harness);
* entry price -- the simulated FILL (spread + slippage), not the bar close;
* barriers    -- regime-specific ``k_tp`` / ``k_sl`` (shared sizing code);
* exit        -- the shared exit rule incl. the minimum-trade-duration
                 lockout (``trade_duration.min_trade_seconds``, default 10 s);
* sampling    -- one position at a time per expert (+ order cool-down).

Output: one Parquet with an ``expert_name`` column AND a sidecar
``<output>.meta.json`` recording the contract it was labeled under.
``train_trend_expert.py`` / ``train_mean_reversion_expert.py`` /
``train_breakout_expert.py`` refuse a dataset whose sidecar does not match
the running config.

Usage:
    python scripts/train_triple_barrier.py --input data/raw/BTC_USDT_mock.parquet
    python scripts/train_triple_barrier.py --input ... --all-candidates   # legacy, NOT harness-like
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.backtest.harness_harvest import EXPERT_NAMES, harvest_expert_samples  # noqa: E402
from src.backtest.training_common import (  # noqa: E402
    add_alignment_args,
    load_training_config,
    read_ohlcv,
    require_filter_from_args,
    sample_policy_from_args,
    write_dataset_meta,
)
from src.labeling.hold_policy import POPULATION_EXPERT_GATE, HoldPolicyError  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def harvest_candidates(
    df_ohlcv: pd.DataFrame, symbol: str, config: dict, *,
    require_filter_pass: bool = True, sample_policy: str = "sequential",
) -> pd.DataFrame:
    """Labeled, harness-aligned samples for every expert (a DataFrame)."""
    df, _, _ = harvest_expert_samples(
        df_ohlcv, symbol, config, expert_names=EXPERT_NAMES,
        require_filter_pass=require_filter_pass, sample_policy=sample_policy,
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Harvest + harness-label all experts' candidates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/triple_barrier_labels.parquet")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default=None, help="explicit config file (overrides --env)")
    add_alignment_args(parser)
    args = parser.parse_args()

    config = load_training_config(args.config, args.env)
    df_ohlcv = read_ohlcv(args.input)
    require_filter = require_filter_from_args(args)
    sample_policy = sample_policy_from_args(args)
    if not require_filter or sample_policy != "sequential":
        logger.warning("Training population is NOT harness-like (--include-filtered-out / "
                       "--all-candidates). The model card will record it.")

    try:
        labeled, stats, ctx = harvest_expert_samples(
            df_ohlcv, args.symbol, config, expert_names=EXPERT_NAMES,
            require_filter_pass=require_filter, sample_policy=sample_policy,
        )
    except HoldPolicyError as exc:
        raise SystemExit(f"Hold policy is not satisfiable: {exc}") from exc

    logger.info(
        f"Hold policy: min {ctx.policy.min_trade_seconds:g}s -> {ctx.min_hold_bars} bar(s) at "
        f"{ctx.policy.granularity} (entry latency {ctx.policy.entry_latency_seconds:g}s)"
    )
    logger.info(
        f"Candidates {stats.n_candidates} -> filter-rejected {stats.n_filter_rejected} -> "
        f"unresolved {stats.n_unresolved} -> labeled {stats.n_labeled} -> sampled {stats.n_sampled}"
    )
    if len(labeled):
        logger.info(labeled.groupby("expert_name")["label"].agg(["count", "mean"]).to_string())
    elif stats.n_candidates and stats.n_filter_rejected == stats.n_candidates:
        logger.warning("Every candidate was rejected by the Opportunity Filter. With the shipped "
                       "synthetic liquidity use a relaxed config, or pass --include-filtered-out.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(args.output, compression="snappy")
    meta_path = write_dataset_meta(
        args.output, policy=ctx.policy, population=POPULATION_EXPERT_GATE, sample_policy=sample_policy,
        require_filter_pass=require_filter, stats=stats, df=labeled,
    )
    logger.info(f"Saved {len(labeled)} samples to {args.output} (contract: {meta_path})")


if __name__ == "__main__":
    main()
