"""
Harness-ordered training DAG (`TrainingPlan`), content-addressed caching,
out-of-fold stacking (`generate_oof_predictions`, Leakage Control L9), and
per-fold refit walk-forward harness (`RefitWalkForwardHarness`, Phase 12 T12.1-T12.3).

Both the runtime pipeline and the training DAG derive their stage order from
`src/pipeline_stages.py` (`get_stages`), guaranteeing that training executes in
exact harness order:
`dataset -> regime -> zone_selection -> zone_survival -> barrier_sizing -> experts -> exit_models -> routers -> opportunity_confirm -> risk_sizing`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..backtest.walk_forward import WalkForwardSplitter
from ..pipeline_stages import StageSpec, get_stages


class OOFLeakageError(ValueError):
    """Raised when a downstream meta-model training set contains in-sample predictions (`is_oof=False`, L9)."""


#: Canonical 10-step training sequence mapped to `src/pipeline_stages.py`.
TRAINING_DAG_STAGES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("dataset", ()),
    ("regime", ("dataset",)),
    ("zone_selection", ("dataset", "regime")),
    ("zone_survival", ("zone_selection", "regime")),
    ("barrier_sizing", ("zone_survival", "regime")),
    ("experts", ("barrier_sizing", "zone_survival", "regime")),
    ("exit_models", ("experts", "barrier_sizing")),
    ("routers", ("experts", "exit_models")),
    ("opportunity_confirm", ("routers", "experts")),
    ("risk_sizing", ("opportunity_confirm", "routers", "barrier_sizing")),
)


@dataclass
class TrainingStageNode:
    name: str
    order_index: int
    enabled: bool
    upstream_deps: Tuple[str, ...]
    cache_key: str
    status: str = "planned"  # "planned" | "cached" | "executed" | "skipped"
    artefact_hash: Optional[str] = None


def _compute_stage_cache_key(
    stage_name: str,
    config: Mapping[str, Any],
    data_fingerprint: str,
    upstream_keys: Mapping[str, str],
) -> str:
    stage_cfg = (config.get("training_stage_configs") or {}).get(stage_name) or config.get(stage_name) or {}
    payload = {
        "stage": stage_name,
        "data_fingerprint": str(data_fingerprint),
        "stage_config": stage_cfg,
        "upstream_keys": {k: upstream_keys[k] for k in sorted(upstream_keys)},
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class TrainingPlan:
    """Topological training DAG derived from `src/pipeline_stages.py`."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        data_fingerprint: str = "default_data",
    ) -> None:
        self.config = dict(config or {})
        self.data_fingerprint = str(data_fingerprint)
        self.pipeline_stages: Tuple[StageSpec, ...] = get_stages("staged")
        self.nodes: List[TrainingStageNode] = self._build_nodes()

    def _build_nodes(self) -> List[TrainingStageNode]:
        stages_cfg = ((self.config.get("pipeline") or {}).get("stages") or {})
        nodes: List[TrainingStageNode] = []
        cache_keys: Dict[str, str] = {}

        for idx, (name, deps) in enumerate(TRAINING_DAG_STAGES):
            st_entry = stages_cfg.get(name, {})
            enabled = bool(st_entry.get("enabled", True)) if isinstance(st_entry, Mapping) else True
            dep_keys = {d: cache_keys[d] for d in deps if d in cache_keys}
            ck = _compute_stage_cache_key(name, self.config, self.data_fingerprint, dep_keys)
            cache_keys[name] = ck
            nodes.append(
                TrainingStageNode(
                    name=name,
                    order_index=idx,
                    enabled=enabled,
                    upstream_deps=deps,
                    cache_key=ck,
                    status="planned" if enabled else "skipped",
                )
            )
        return nodes

    def stage_names(self) -> List[str]:
        return [n.name for n in self.nodes]

    def cache_key_map(self) -> Dict[str, str]:
        return {n.name: n.cache_key for n in self.nodes}

    def render_plan_table(self) -> str:
        lines = [
            f"{'#':<3} {'Stage':<22} {'Enabled':<8} {'CacheKey':<18} {'Dependencies'}",
            "-" * 78,
        ]
        for n in self.nodes:
            deps_str = ", ".join(n.upstream_deps) if n.upstream_deps else "-"
            lines.append(
                f"{n.order_index:<3} {n.name:<22} {str(n.enabled):<8} {n.cache_key:<18} {deps_str}"
            )
        return "\n".join(lines)

    def execute(
        self,
        runners: Optional[Mapping[str, Callable[[TrainingStageNode, Dict[str, Any]], Any]]] = None,
        *,
        cache_store: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute stages in exact topological order, reusing `cache_store` hits when `cache_key` matches."""
        store = cache_store if cache_store is not None else {}
        runner_map = dict(runners or {})
        context: Dict[str, Any] = {}

        for node in self.nodes:
            if not node.enabled:
                node.status = "skipped"
                continue
            if node.cache_key in store:
                node.status = "cached"
                context[node.name] = store[node.cache_key]
                node.artefact_hash = node.cache_key
                continue

            fn = runner_map.get(node.name)
            res = fn(node, context) if fn is not None else {"stage": node.name, "cache_key": node.cache_key}
            store[node.cache_key] = res
            context[node.name] = res
            node.status = "executed"
            node.artefact_hash = node.cache_key

        return context


def verify_oof_no_in_sample_leakage(records: Sequence[Mapping[str, Any]]) -> None:
    """Verify every stacking record used by a downstream model is strictly out-of-fold (`is_oof=True`, L9)."""
    for idx, r in enumerate(records):
        if r.get("is_oof") is not True or r.get("split") == "train":
            raise OOFLeakageError(
                f"Row {idx} is not out-of-fold (is_oof={r.get('is_oof')!r}, split={r.get('split')!r}) — L9 violation"
            )
        train_max_bar = r.get("train_max_bar_index")
        bar_idx = r.get("bar_index")
        embargo = int(r.get("embargo_bars", 0))
        if train_max_bar is not None and bar_idx is not None:
            if int(bar_idx) <= int(train_max_bar) + embargo:
                raise OOFLeakageError(
                    f"Row {idx} at bar_index={bar_idx} violates train_max_bar={train_max_bar} + embargo={embargo} (L9)"
                )


def generate_oof_predictions(
    bar_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    train_bars: int = 100,
    test_bars: int = 40,
    step_bars: int = 40,
    embargo_bars: int = 5,
    label_horizon_bars: int = 5,
) -> List[Dict[str, Any]]:
    """Generate purged-and-embargoed out-of-fold predictions for downstream stacking (L9)."""
    splitter = WalkForwardSplitter(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        embargo_bars=embargo_bars,
        label_horizon_bars=label_horizon_bars,
    )
    bars_arr = np.asarray(bar_indices, dtype=int)
    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y)
    label_end = bars_arr + int(label_horizon_bars)

    oof_rows: List[Dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split_by_bar_index(bars_arr, label_end)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        preds = fit_predict_fn(X_arr[train_idx], y_arr[train_idx], X_arr[test_idx])
        t_max = int(np.max(bars_arr[train_idx]))
        for local_i, row_idx in enumerate(test_idx.tolist()):
            oof_rows.append({
                "row_index": int(row_idx),
                "bar_index": int(bars_arr[row_idx]),
                "fold_id": int(fold_id),
                "train_max_bar_index": t_max,
                "embargo_bars": int(embargo_bars),
                "prediction": float(preds[local_i]),
                "target": float(y_arr[row_idx]),
                "is_oof": True,
                "split": "val",
            })

    verify_oof_no_in_sample_leakage(oof_rows)
    return oof_rows


@dataclass(frozen=True)
class FoldRefitRecord:
    fold_id: int
    train_start_bar: int
    train_end_bar: int
    test_start_bar: int
    test_end_bar: int
    bundle_hash: str


class RefitWalkForwardHarness:
    """Walk-forward harness that refits the `TrainingPlan` per fold (`--refit`) with purged embargo."""

    def __init__(
        self,
        *,
        train_bars: int = 200,
        test_bars: int = 50,
        step_bars: int = 50,
        embargo_bars: int = 5,
        label_horizon_bars: int = 5,
    ) -> None:
        self.splitter = WalkForwardSplitter(
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            embargo_bars=embargo_bars,
            label_horizon_bars=label_horizon_bars,
        )
        self.label_horizon_bars = int(label_horizon_bars)
        self.fold_records: List[FoldRefitRecord] = []

    def run_folds(
        self,
        bar_indices: np.ndarray,
        fit_fold_fn: Callable[[int, np.ndarray], str],
        eval_fold_fn: Callable[[int, np.ndarray, str], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        bars_arr = np.asarray(bar_indices, dtype=int)
        label_end = bars_arr + self.label_horizon_bars
        all_test_outputs: List[Dict[str, Any]] = []

        for fold_id, (train_idx, test_idx) in enumerate(self.splitter.split_by_bar_index(bars_arr, label_end)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            t_start = int(np.min(bars_arr[train_idx]))
            t_end = int(np.max(bars_arr[train_idx]))
            v_start = int(np.min(bars_arr[test_idx]))
            v_end = int(np.max(bars_arr[test_idx]))
            assert t_end + self.splitter.embargo_bars < v_start + 1, "Embargo violated between train and test"

            bundle_hash = fit_fold_fn(fold_id, train_idx)
            self.fold_records.append(
                FoldRefitRecord(
                    fold_id=fold_id,
                    train_start_bar=t_start,
                    train_end_bar=t_end,
                    test_start_bar=v_start,
                    test_end_bar=v_end,
                    bundle_hash=bundle_hash,
                )
            )
            fold_out = eval_fold_fn(fold_id, test_idx, bundle_hash)
            all_test_outputs.extend(fold_out)

        return all_test_outputs
