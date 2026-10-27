"""
Harvest expert training samples the way the harness uses the experts.

One replay pass through the full stack, then for each expert:

* **population** -- only bars where THAT expert's own ``propose_candidate()``
  gate fired (``ExpertOutput.action != NO_TRADE``). This is the population
  the expert is scored on in deployment. (Mean-reversion and breakout used
  to be trained on the zone touch-event stream instead -- a different
  population from the one they are deployed on.)
* **filter** -- by default only bars the Opportunity Filter passed
  (``opportunity_score.tradable``), because in the harness a rejected bar
  never reaches the router or the risk layer. ``require_filter_pass=False``
  keeps the rejected bars (the old population).
* **labels** -- :func:`src.labeling.harness_labels.label_candidate`.
* **sampling** -- ``"sequential"`` keeps one position at a time per expert
  (plus the order cool-down); ``"all"`` keeps every candidate on every bar
  (overlapping, the old behaviour).

The returned ``HarvestStats`` records how many candidates each stage
removed, so a model card can state exactly what population it was trained
on.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..evaluation.trade_duration import hold_stats
from ..labeling.harness_labels import SAMPLE_POLICIES, LabelContext, label_candidate, select_sequential
from ..labeling.hold_policy import POPULATION_EXPERT_GATE, training_alignment_block
from ..state.contracts import ExpertAction
from .replay import ReplayResult, replay_full_stack

EXPERT_NAMES = ["trend_expert", "mean_reversion_expert", "breakout_expert"]


def expert_feature_builders() -> Dict[str, Callable]:
    """``build_features`` per expert (imported lazily: experts import layers)."""
    from ..experts import breakout_expert as bo_mod
    from ..experts import mean_reversion_expert as mr_mod
    from ..experts import trend_expert as trend_mod
    return {
        "trend_expert": trend_mod.build_features,
        "mean_reversion_expert": mr_mod.build_features,
        "breakout_expert": bo_mod.build_features,
    }


@dataclass
class HarvestStats:
    n_candidates: int = 0           # bars where the expert's gate fired
    n_filter_rejected: int = 0      # ... removed because the Opportunity Filter said no
    n_unresolved: int = 0           # ... removed because the data ended before a legal exit
    n_labeled: int = 0              # ... labeled
    n_sampled: int = 0              # ... kept by the sample policy
    per_expert: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def bump(self, expert: str, key: str) -> None:
        setattr(self, key, getattr(self, key) + 1)
        self.per_expert.setdefault(expert, {}).setdefault(key, 0)
        self.per_expert[expert][key] += 1


def harvest_expert_samples(
    df_ohlcv: pd.DataFrame, symbol: str, config: dict, *,
    expert_names: Optional[Sequence[str]] = None,
    require_filter_pass: bool = True,
    sample_policy: str = "sequential",
    replay: Optional[ReplayResult] = None,
    with_lockout_free: bool = False,
) -> Tuple[pd.DataFrame, HarvestStats, LabelContext]:
    """Labeled, harness-aligned training samples for the given experts.

    Pass ``replay`` to reuse an existing :func:`replay_full_stack` result.
    Every row carries ``passed_filter``. ``with_lockout_free=True`` adds
    ``realized_return_free_pct`` / ``bars_held_free``: the same candidate
    labeled with NO minimum-hold lockout, for the lockout-cost eval.
    """
    if sample_policy not in SAMPLE_POLICIES:
        raise ValueError(f"sample_policy must be one of {SAMPLE_POLICIES}, got {sample_policy!r}")
    names = list(expert_names or EXPERT_NAMES)
    ctx = LabelContext.from_config(config)      # raises HoldPolicyError on an unsatisfiable policy
    builders = expert_feature_builders()
    result = replay if replay is not None else replay_full_stack(df_ohlcv, symbol, config, run_experts=True)
    max_hold = ctx.policy.max_holding_bars
    snapshots = result.structure_snapshots
    stats = HarvestStats()

    rows: List[Dict[str, Any]] = []
    for i, state in enumerate(result.system_states):
        if state.expert_outputs is None or state.regime_state is None:
            continue
        # The labeler reads at most ``max_holding_bars`` forward bars.
        forward = snapshots[i + 1: i + 1 + max_hold]

        for name in names:
            output = state.expert_outputs.get_by_name(name)
            if output is None or output.action == ExpertAction.NO_TRADE:
                continue
            stats.bump(name, "n_candidates")

            passed_filter = bool(state.opportunity_score and state.opportunity_score.tradable)
            if require_filter_pass and not passed_filter:
                stats.bump(name, "n_filter_rejected")
                continue

            direction = 1 if output.action == ExpertAction.BUY else -1
            labeled = label_candidate(ctx, state, forward, direction)
            if labeled is None:
                stats.bump(name, "n_unresolved")
                continue
            stats.bump(name, "n_labeled")

            features = builders[name](state.feature_vector, state.zone_map, state.regime_state, direction)
            row = {
                **features, "expert_name": name, "timestamp": state.timestamp, "bar_index": i,
                "direction": direction, "passed_filter": passed_filter, **labeled,
            }
            if with_lockout_free:
                free = label_candidate(ctx, state, forward, direction, min_hold_bars=1)
                row["realized_return_free_pct"] = free["realized_return_pct"] if free else float("nan")
                row["bars_held_free"] = free["bars_held"] if free else float("nan")
            rows.append(row)

    if sample_policy == "sequential" and rows:
        keep = select_sequential(rows, ctx.min_bars_between_orders)
        rows = [row for row, k in zip(rows, keep, strict=True) if k]
    for row in rows:
        stats.bump(row["expert_name"], "n_sampled")

    return pd.DataFrame(rows), stats, ctx


def apply_population(
    df: pd.DataFrame, ctx: LabelContext, *, require_filter_pass: bool = True, sample_policy: str = "sequential",
) -> pd.DataFrame:
    """Narrow a FULL harvest (``require_filter_pass=False, sample_policy="all"``)
    to a training population.

    The same two steps ``harvest_expert_samples`` applies while harvesting,
    exposed so the alignment eval can score one replay against both the
    complete label set and the population a model would actually be
    trained on.
    """
    if sample_policy not in SAMPLE_POLICIES:
        raise ValueError(f"sample_policy must be one of {SAMPLE_POLICIES}, got {sample_policy!r}")
    out = df
    if require_filter_pass and "passed_filter" in out.columns:
        out = out[out["passed_filter"]]
    out = out.reset_index(drop=True)
    if sample_policy == "sequential" and len(out):
        keep = select_sequential(out.to_dict("records"), ctx.min_bars_between_orders)
        out = out[keep].reset_index(drop=True)
    return out


def alignment_block_for(
    ctx: LabelContext, df: pd.DataFrame, stats: HarvestStats, *, expert: Optional[str] = None,
    require_filter_pass: bool = True, sample_policy: str = "sequential",
    population: str = POPULATION_EXPERT_GATE,
) -> Dict[str, Any]:
    """The ``training_alignment`` block for a model trained on ``df``."""
    stat = stats.per_expert.get(expert, {}) if expert else {}
    n_candidates = stat.get("n_candidates", stats.n_candidates)
    hold = hold_stats(df["hold_seconds"] if "hold_seconds" in df.columns else [], ctx.policy.min_trade_seconds)
    return training_alignment_block(
        ctx.policy, population=population, sample_policy=sample_policy,
        require_opportunity_filter_pass=require_filter_pass,
        n_candidates=n_candidates, n_samples=len(df),
        hold_stats=hold, min_duration_violations=hold["violations"],
    )
