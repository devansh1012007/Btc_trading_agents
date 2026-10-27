"""
Shared plumbing for the training scripts.

Everything the training scripts used to copy between themselves -- config
loading, OHLCV reading, the walk-forward fit, the model card -- plus the
two things that make a training run mirror the harness:

* the **dataset contract**: the labeled-candidates Parquet is written with
  a sidecar (``<file>.meta.json``) recording the hold policy, population
  and sampling it was labeled under, and every consumer re-checks that
  sidecar against the CURRENT config before it trains. A dataset labeled
  under a different hold policy is refused (a stale dataset would train a
  model on trades the running harness does not take);
* **bar-indexed walk-forward windows** with the label horizon enforced
  (``WalkForwardSplitter.split_by_bar_index``).
"""
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from ..evaluation.trade_duration import hold_stats
from ..labeling.hold_policy import (
    HoldPolicy,
    HoldPolicyError,
    training_alignment_block,
)
from .walk_forward import WalkForwardSplitter

logger = logging.getLogger(__name__)

DATASET_META_VERSION = 1
FEATURE_VERSION = "v2_renamed"


# ---------------------------------------------------------------------------
# config + data
# ---------------------------------------------------------------------------

def load_training_config(config_path: Optional[str] = None, env: str = "paper") -> dict:
    """The config a training run labels under.

    By default the SAME layered config the harness runs (``base`` +
    ``paper`` overlay): the old scripts read ``settings.base.yaml`` alone,
    so their slippage / latency / execution knobs were not the ones the
    harness uses. An explicit ``--config`` still wins.
    """
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise SystemExit(f"Config not found: {p}")
        with open(p) as f:
            return yaml.safe_load(f) or {}
    from ..utils.config_helpers import load_layered_config
    return load_layered_config(env)


def read_ohlcv(path: str) -> pd.DataFrame:
    """OHLCV from CSV / Parquet, sorted by time, timestamps left as given."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def add_alignment_args(parser) -> None:
    """The population / sampling switches every harvesting script shares."""
    parser.add_argument("--env", default="paper", choices=["paper", "live"],
                        help="config overlay to label under (default: paper, like the harness)")
    parser.add_argument("--all-candidates", action="store_true",
                        help="keep every candidate on every bar (overlapping) instead of one "
                             "position at a time. NOT how the harness trades; the model card says so.")
    parser.add_argument("--include-filtered-out", action="store_true",
                        help="also keep bars the Opportunity Filter rejected. NOT how the harness "
                             "trades; the model card says so.")


def sample_policy_from_args(args) -> str:
    return "all" if getattr(args, "all_candidates", False) else "sequential"


def require_filter_from_args(args) -> bool:
    return not getattr(args, "include_filtered_out", False)


# ---------------------------------------------------------------------------
# the dataset contract
# ---------------------------------------------------------------------------

def dataset_meta_path(dataset_path: str) -> Path:
    return Path(str(dataset_path) + ".meta.json")


def write_dataset_meta(
    dataset_path: str, *, policy: HoldPolicy, population: str, sample_policy: str,
    require_filter_pass: bool, stats: Any, df: pd.DataFrame,
) -> Path:
    """Sidecar describing exactly how ``dataset_path`` was labeled."""
    per_expert = {}
    for name, group in (df.groupby("expert_name") if len(df) else []):
        counts = dict(stats.per_expert.get(name, {}))
        per_expert[str(name)] = {
            "n_candidates": counts.get("n_candidates", 0),
            "n_filter_rejected": counts.get("n_filter_rejected", 0),
            "n_unresolved": counts.get("n_unresolved", 0),
            "n_labeled": counts.get("n_labeled", 0),
            "n_samples": int(len(group)),
            "hold_stats": hold_stats(group["hold_seconds"], policy.min_trade_seconds),
        }
    meta = {
        "version": DATASET_META_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "hold_policy": policy.fingerprint(),
        "population": population,
        "sample_policy": sample_policy,
        "require_opportunity_filter_pass": bool(require_filter_pass),
        "totals": {
            "n_candidates": stats.n_candidates, "n_filter_rejected": stats.n_filter_rejected,
            "n_unresolved": stats.n_unresolved, "n_labeled": stats.n_labeled,
            "n_sampled": stats.n_sampled,
        },
        "per_expert": per_expert,
    }
    path = dataset_meta_path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def read_dataset_meta(dataset_path: str) -> Optional[Dict[str, Any]]:
    path = dataset_meta_path(dataset_path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_dataset_matches_config(meta: Optional[Mapping[str, Any]], config: Mapping[str, Any]) -> HoldPolicy:
    """Refuse a labeled dataset that was not labeled under the running policy.

    Returns the running :class:`HoldPolicy`. Raises ``SystemExit`` with the
    reason otherwise -- a training script must never silently train on a
    stale dataset.
    """
    try:
        policy = HoldPolicy.from_config(config)
    except HoldPolicyError as exc:
        raise SystemExit(f"Hold policy is not satisfiable: {exc}") from exc
    if meta is None:
        raise SystemExit(
            "The labeled dataset has no .meta.json sidecar, so it is not known to be harness-aligned. "
            "Re-run scripts/train_triple_barrier.py."
        )
    labeled = meta.get("hold_policy") or {}
    running = policy.fingerprint()
    diffs = [f"{k}: dataset={labeled.get(k)!r} running={running[k]!r}" for k in running if labeled.get(k) != running[k]]
    if diffs:
        raise SystemExit(
            "The labeled dataset was labeled under a different hold policy (" + "; ".join(diffs) +
            "). Re-run scripts/train_triple_barrier.py with the current config."
        )
    return policy


def alignment_block_from_meta(
    meta: Mapping[str, Any], policy: HoldPolicy, expert: str, df: pd.DataFrame,
) -> Dict[str, Any]:
    """The ``training_alignment`` block for ``expert`` trained on ``df``."""
    stat = (meta.get("per_expert") or {}).get(expert, {})
    hold = hold_stats(df["hold_seconds"] if "hold_seconds" in df.columns else [], policy.min_trade_seconds)
    return training_alignment_block(
        policy, population=meta["population"], sample_policy=meta["sample_policy"],
        require_opportunity_filter_pass=meta["require_opportunity_filter_pass"],
        n_candidates=stat.get("n_candidates", len(df)), n_samples=len(df),
        hold_stats=hold, min_duration_violations=hold["violations"],
    )


# ---------------------------------------------------------------------------
# windows + fit
# ---------------------------------------------------------------------------

def splitter_for(config: Mapping[str, Any], df: pd.DataFrame, policy: HoldPolicy) -> WalkForwardSplitter:
    """Walk-forward windows in BARS, with the label horizon enforced.

    The configured ``train_bars`` / ``test_bars`` / ``step_bars`` are capped
    by the data's span (so a short dataset still gets folds), and the
    embargo must cover the label horizon (``labeling.max_holding_bars``).
    """
    backtest_cfg = config.get("backtest", {}) or {}
    bars = df["bar_index"].to_numpy(dtype="int64")
    span = int(bars.max() - bars.min() + 1) if bars.size else 1
    try:
        return WalkForwardSplitter(
            train_bars=min(backtest_cfg.get("train_bars", 5000), max(1, span // 2)),
            test_bars=min(backtest_cfg.get("test_bars", 500), max(1, span // 4)),
            step_bars=min(backtest_cfg.get("step_bars", 100), max(1, span // 4)),
            embargo_bars=backtest_cfg.get("embargo_bars", 50),
            label_horizon_bars=policy.max_holding_bars,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)


def walk_forward_fit(
    df: pd.DataFrame, feature_names: Sequence[str], splitter: WalkForwardSplitter,
    make_model=make_lgbm, multiclass: bool = False,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Fit by bar-indexed walk-forward CV, then on all the data.

    Needs ``bar_index``, ``bars_held`` (the label horizon of each row, used
    to purge) and ``label`` columns. ``multiclass`` scores accuracy only
    (the router); the default also reports AUC.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    X = df[list(feature_names)].values
    y = df["label"].values
    bar_index = df["bar_index"].to_numpy(dtype="int64")
    label_end = bar_index + df["bars_held"].to_numpy(dtype="int64")

    folds: List[Dict[str, Any]] = []
    windows = list(splitter.split_by_bar_index(bar_index, label_end))
    if not windows:
        logger.warning("Not enough samples for a walk-forward fold; fitting on all data without CV.")
    for fold_i, (train_idx, test_idx) in enumerate(windows):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        if multiclass:
            proba = model.predict_proba(X[test_idx])
            preds = model.classes_[np.argmax(proba, axis=1)]
            metric = {"fold": fold_i, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                      "accuracy": float(accuracy_score(y[test_idx], preds))}
        else:
            proba = model.predict_proba(X[test_idx])[:, 1]
            preds = (proba >= 0.5).astype(int)
            metric = {
                "fold": fold_i, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                "auc": float(roc_auc_score(y[test_idx], proba)) if len(set(y[test_idx])) > 1 else None,
                "accuracy": float(accuracy_score(y[test_idx], preds)),
            }
        folds.append(metric)
        logger.info(f"Fold {fold_i}: {metric}")

    final_model = make_model()
    final_model.fit(X, y)
    return final_model, folds


# ---------------------------------------------------------------------------
# one trainer for all three experts
# ---------------------------------------------------------------------------

def assert_no_short_trades(df: pd.DataFrame, policy: HoldPolicy, what: str) -> None:
    """Training must never see a label for a trade the rule forbids."""
    if "hold_seconds" not in df.columns:
        raise SystemExit(f"{what}: the dataset has no hold_seconds column (labeled before the "
                         f"minimum-trade-duration change); re-run scripts/train_triple_barrier.py")
    short = int((df["hold_seconds"] < policy.min_trade_seconds - 1e-3).sum()) if policy.min_trade_seconds > 0 else 0
    if short:
        raise SystemExit(f"{what}: {short} labeled trade(s) are shorter than {policy.min_trade_seconds:g}s")


def train_expert(
    expert_name: str, feature_names: Sequence[str], df: pd.DataFrame, config: Mapping[str, Any],
    policy: HoldPolicy, alignment: Mapping[str, Any], output: str, make_model=None,
) -> Path:
    """Train, validate, save ``expert_name``'s model + card. No hidden state.

    ``make_model`` (default: the shared LightGBM) is injectable so the
    contract round-trip can be tested without fitting a real model.
    """
    assert_no_short_trades(df, policy, expert_name)
    df = df.sort_values("bar_index").reset_index(drop=True)
    data_granularity = (config.get("system", {}) or {}).get("data_granularity", "1m")

    splitter = splitter_for(config, df, policy)
    model, fold_metrics = walk_forward_fit(df, feature_names, splitter, make_model or make_lgbm)

    favorable = df[df["label"] == 1]["realized_return_pct"]
    unfavorable = df[df["label"] == 0]["realized_return_pct"]
    payload = {
        "model": model, "feature_names": list(feature_names), "granularity": data_granularity,
        "feature_version": FEATURE_VERSION,
        "avg_favorable_return": float(favorable.mean()) if len(favorable) else 0.0,
        "avg_unfavorable_return": float(unfavorable.mean()) if len(unfavorable) else 0.0,
        "avg_adverse_excursion": float(df["max_adverse_excursion_pct"].mean()) if len(df) else 0.0,
        "training_alignment": dict(alignment),
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    card = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "granularity": data_granularity,
        "n_samples": int(len(df)), "n_favorable": int((df["label"] == 1).sum()),
        "n_unfavorable": int((df["label"] == 0).sum()),
        "avg_favorable_return": payload["avg_favorable_return"],
        "avg_unfavorable_return": payload["avg_unfavorable_return"],
        "avg_adverse_excursion": payload["avg_adverse_excursion"],
        "feature_names": list(feature_names), "walk_forward_folds": fold_metrics,
        "walk_forward_window_unit": "bars",
        "pinned_rate": float(df["pinned_to_min_hold"].mean()) if "pinned_to_min_hold" in df.columns and len(df) else None,
        "training_alignment": dict(alignment),
    }
    with open(output_path.with_suffix(".meta.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info(f"Saved {expert_name} model to {output_path}")
    return output_path


def expert_dataset(
    labels_path: str, config: Mapping[str, Any], expert_name: str,
) -> Tuple[pd.DataFrame, HoldPolicy, Dict[str, Any]]:
    """Load the labeled dataset, verify its contract, return ``expert_name``'s rows."""
    meta = read_dataset_meta(labels_path)
    policy = check_dataset_matches_config(meta, config)
    df_all = pd.read_parquet(labels_path)
    df = df_all[df_all["expert_name"] == expert_name].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"No {expert_name} samples found -- run scripts/train_triple_barrier.py first "
                         f"(or the gate/filter never fired on that data)")
    return df, policy, alignment_block_from_meta(meta, policy, expert_name, df)


def bar_level_alignment(
    policy: HoldPolicy, df: pd.DataFrame, *, population: str, require_filter_pass: bool,
    n_candidates: int,
) -> Dict[str, Any]:
    """``training_alignment`` for the bar-level models (router, opportunity
    filter, risk model): they label every bar of their population (no
    one-position sampling), so ``sample_policy`` is ``"bars"``."""
    hold = hold_stats(df["hold_seconds"] if "hold_seconds" in df.columns else [], policy.min_trade_seconds)
    return training_alignment_block(
        policy, population=population, sample_policy="bars",
        require_opportunity_filter_pass=require_filter_pass, n_candidates=n_candidates,
        n_samples=len(df), hold_stats=hold, min_duration_violations=hold["violations"],
    )


def labels_have_expected_rows(df: pd.DataFrame) -> bool:
    return bool(len(df)) and bool(np.all(np.isfinite(df["realized_return_pct"].to_numpy(dtype="float64"))))
