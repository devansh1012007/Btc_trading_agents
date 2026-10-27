"""
Train the Trend Expert's scoring model (§6.6, M3.4) the way the
harness uses it.

The training set is the expert's OWN deployment population: bars where
``trend_expert.propose_candidate()`` fired and the Opportunity Filter
passed, labeled with the harness's exit rule (simulated-fill entry,
regime-specific barriers, the minimum-trade-duration lockout), one
position at a time. See ``src/backtest/harness_harvest.py``.

Two ways to supply it:

    # reuse the combined dataset (all experts, one replay) -- the default path
    python scripts/train_triple_barrier.py --input data/raw/BTC_USDT_mock.parquet
    python scripts/train_trend_expert.py --labels data/processed/triple_barrier_labels.parquet

    # or harvest just this expert straight from OHLCV
    python scripts/train_trend_expert.py --input data/raw/BTC_USDT_mock.parquet

The saved model carries a ``training_alignment`` contract; the running
system serves it only when its hold policy matches (otherwise the expert
falls back to its rule-based mode).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.harness_harvest import alignment_block_for, harvest_expert_samples  # noqa: E402
from src.backtest.training_common import (  # noqa: E402
    add_alignment_args,
    expert_dataset,
    load_training_config,
    read_ohlcv,
    require_filter_from_args,
    sample_policy_from_args,
    train_expert,
)
from src.experts.trend_expert import FEATURE_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERT = "trend_expert"


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--labels", default=None,
                        help="combined dataset from train_triple_barrier.py")
    source.add_argument("--input", default=None, help="historical OHLCV Parquet/CSV to harvest from")
    parser.add_argument("--output", default="data/models/experts/trend_expert.pkl")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default=None, help="explicit config file (overrides --env)")
    add_alignment_args(parser)
    args = parser.parse_args()

    config = load_training_config(args.config, args.env)

    if args.input:
        require_filter = require_filter_from_args(args)
        sample_policy = sample_policy_from_args(args)
        df, stats, ctx = harvest_expert_samples(
            read_ohlcv(args.input), args.symbol, config, expert_names=[EXPERT],
            require_filter_pass=require_filter, sample_policy=sample_policy,
        )
        if len(df) == 0:
            raise SystemExit(f"No {EXPERT} samples (candidates {stats.n_candidates}, "
                             f"filter-rejected {stats.n_filter_rejected}) -- need more history")
        policy = ctx.policy
        alignment = alignment_block_for(
            ctx, df, stats, expert=EXPERT, require_filter_pass=require_filter,
            sample_policy=sample_policy,
        )
    else:
        labels = args.labels or "data/processed/triple_barrier_labels.parquet"
        df, policy, alignment = expert_dataset(labels, config, EXPERT)

    train_expert(EXPERT, FEATURE_NAMES, df, config, policy, alignment, args.output)


if __name__ == "__main__":
    main()
