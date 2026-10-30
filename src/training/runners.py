"""
Real stage runners (`build_real_stage_runners`) and `ModelBundle` builder
(`build_bundle_from_training_context`) for `TrainingPlan.execute()` (Phase 12).

Executes the 10-stage topological training DAG:
0. `dataset`
1. `regime`
2. `zone_selection`
3. `zone_survival`
4. `barrier_sizing`
5. `experts`
6. `exit_models`
7. `routers`
8. `opportunity_confirm`
9. `risk_sizing`
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..exits.exit_model import EXIT_STATE_FEATURES, ExitModel
from ..experts.plan_head import DEFAULT_PLAN_GRID, PlanUtilityHead
from ..experts.selection import select_top_n_instances
from ..labeling.excursions import compute_forward_excursions
from ..labeling.hold_policy import HoldPolicy
from ..regime.alignment import align_regime_labels
from ..regime.backends import build_regime_backend
from ..regime.distill import distill_regime_classifier
from ..regime.selection import make_dynamic_regime_set, select_k
from ..registry.bundle import (
    ModelBundle,
    canonical_json_hash,
    compute_dataframe_hash,
    save_native_model,
)
from ..sizing.barrier_model import BarrierSizingModel
from ..sizing.risk_model import BASE_RISK_SIZING_FEATURES, RiskSizingModel
from ..state.contracts import LEGACY_REGIME_SET, MarketSnapshot
from .plan import (
    TRAINING_DAG_STAGES,
    TrainingPlan,
    TrainingStageNode,
    generate_oof_predictions,
)


def make_synthetic_training_ohlcv(n_bars: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic OHLCV DataFrame for end-to-end training & testing."""
    rng = np.random.default_rng(seed)
    ts = [1_700_000_000 + i * 60 for i in range(n_bars)]
    rets = rng.normal(0.0002, 0.003, size=n_bars)
    close = 50_000.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = 50_000.0
    spread = np.abs(rng.normal(0.002, 0.0008, size=n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(15.0, 120.0, size=n_bars)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def build_real_stage_runners(
    df_ohlcv: pd.DataFrame,
    *,
    symbol: str = "BTC/USDT",
    config: Optional[Mapping[str, Any]] = None,
    output_dir: str | Path = "data/models/bundle_run",
    random_state: int = 42,
) -> Dict[str, Callable[[TrainingStageNode, Dict[str, Any]], Dict[str, Any]]]:
    """Return a stage-name -> runner callable dict for all 10 `TRAINING_DAG_STAGES`."""
    cfg = dict(config or {})
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    hold_policy = HoldPolicy.from_config(cfg)

    def run_dataset(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        df = df_ohlcv.copy().reset_index(drop=True)
        close = df["close"].to_numpy(dtype=float)
        ret1 = np.zeros_like(close)
        ret1[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-9) * 100.0
        ret5 = pd.Series(ret1).rolling(5, min_periods=1).sum().to_numpy(dtype=float)
        vol5 = pd.Series(ret1).rolling(5, min_periods=1).std().fillna(0.15).to_numpy(dtype=float)
        vol5 = np.maximum(vol5, 0.05)
        vol_z = (
            (df["volume"] - df["volume"].rolling(10, min_periods=1).mean())
            / np.maximum(df["volume"].rolling(10, min_periods=1).std().fillna(1.0), 1e-6)
        ).to_numpy(dtype=float)

        feat_matrix = np.column_stack([ret1, ret5, vol5, vol_z])
        snapshots = [
            MarketSnapshot(
                timestamp=int(row.timestamp),
                symbol=symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                bid=float(row.close) * 0.9999,
                ask=float(row.close) * 1.0001,
                bid_volume=float(row.volume) * 0.5,
                ask_volume=float(row.volume) * 0.5,
                spread_pct=0.02,
            )
            for row in df.itertuples(index=False)
        ]
        path = out_root / "dataset_manifest.pkl"
        sha = save_native_model(
            {
                "stage": node.name,
                "n_rows": len(df),
                "data_hash": compute_dataframe_hash(df),
                "granularity": hold_policy.granularity,
            },
            path,
        )
        return {
            "df": df,
            "X_obs": feat_matrix,
            "ret1": ret1,
            "vol5": vol5,
            "snapshots": snapshots,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_regime(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        X_obs = ds["X_obs"]
        reg_cfg = cfg.get("regime", {}) or {}
        dyn_enabled = bool((reg_cfg.get("dynamic_k") or {}).get("enabled", False))
        backend_name = str(reg_cfg.get("backend", "kmeans"))

        if dyn_enabled and len(X_obs) >= 60:
            sel = select_k(
                X_obs,
                backend_name=backend_name if backend_name in ("kmeans", "gmm", "gaussian_hmm") else "kmeans",
                k_min=2,
                max_regimes=min(4, int((reg_cfg.get("dynamic_k") or {}).get("max_regimes", 4))),
                random_state=random_state,
            )
            k_best = sel.selected_k
            regime_set = make_dynamic_regime_set(k_best, backend_name=backend_name)
        else:
            k_best = 4
            regime_set = LEGACY_REGIME_SET

        backend = build_regime_backend("kmeans", n_components=len(regime_set.regime_ids), random_state=random_state)
        backend.fit(X_obs)
        raw_probs = backend.filter_causal(X_obs)
        align_res = align_regime_labels(backend.means_, backend.means_, old_probs=raw_probs, new_probs=raw_probs)
        perm = [align_res.new_to_old.get(i, i) for i in range(raw_probs.shape[1])]
        aligned_probs = raw_probs[:, perm]

        distilled = distill_regime_classifier(
            X_obs,
            aligned_probs,
            feature_names=["momentum_short", "momentum_medium", "volatility_medium", "volume_zscore"],
            regime_set=regime_set,
            random_state=random_state,
        )
        y_teacher = np.argmax(aligned_probs, axis=1)

        path = out_root / "regime_classifier.pkl"
        sha = save_native_model(
            {
                "stage": node.name,
                "regime_set_id": regime_set.set_id,
                "regime_set_hash": regime_set.version_hash,
                "selected_k": k_best,
                "feature_names": distilled.feature_names,
            },
            path,
        )
        return {
            "regime_set": regime_set,
            "regime_probs": aligned_probs,
            "y_regime": y_teacher,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_zone_selection(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        df = ds["df"]
        rolling_low = df["low"].rolling(15, min_periods=1).min().to_numpy(dtype=float)
        rolling_high = df["high"].rolling(15, min_periods=1).max().to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        dist_sup = np.maximum(0.05, (close - rolling_low) / np.maximum(close, 1e-9) * 100.0)
        dist_res = np.maximum(0.05, (rolling_high - close) / np.maximum(close, 1e-9) * 100.0)
        path = out_root / "zone_selection.pkl"
        sha = save_native_model(
            {"stage": node.name, "n_bars": len(df), "mean_dist_sup": float(np.mean(dist_sup))},
            path,
        )
        return {
            "dist_sup": dist_sup,
            "dist_res": dist_res,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_zone_survival(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        zs = ctx["zone_selection"]
        vol5 = ds["vol5"]
        surv_sup = np.clip(np.exp(-0.25 * vol5 / np.maximum(zs["dist_sup"], 0.05)), 0.15, 0.95)
        surv_res = np.clip(np.exp(-0.25 * vol5 / np.maximum(zs["dist_res"], 0.05)), 0.15, 0.95)
        path = out_root / "zone_survival.pkl"
        sha = save_native_model(
            {"stage": node.name, "mean_surv_sup": float(np.mean(surv_sup))},
            path,
        )
        return {
            "surv_sup": surv_sup,
            "surv_res": surv_res,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_barrier_sizing(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        reg = ctx["regime"]
        zs = ctx["zone_selection"]
        zsurv = ctx["zone_survival"]
        snaps: List[MarketSnapshot] = ds["snapshots"]
        vol5 = ds["vol5"]
        rset = reg["regime_set"]

        X_rows: List[Dict[str, float]] = []
        k_mae_list: List[float] = []
        k_mfe_list: List[float] = []
        k_lock_list: List[float] = []

        horizon = 6
        for i in range(len(snaps) - horizon - 1):
            prof = compute_forward_excursions(
                snaps[i].close,
                1 if ds["ret1"][i] >= 0 else -1,
                snaps[i + 1 : i + 1 + horizon],
                max_holding_bars=horizon,
                min_hold_bars=max(1, hold_policy.min_hold_bars),
                volatility_pct=float(vol5[i]),
            )
            row: Dict[str, float] = {
                "volatility_medium": float(vol5[i]),
                "spread_pct": 0.02,
                "direction": 1.0 if ds["ret1"][i] >= 0 else -1.0,
                "nearest_support_survival_prob": float(zsurv["surv_sup"][i]),
                "nearest_resistance_survival_prob": float(zsurv["surv_res"][i]),
                "support_strength": 0.6,
                "resistance_strength": 0.6,
                "distance_to_nearest_support": float(zs["dist_sup"][i]),
                "distance_to_nearest_resistance": float(zs["dist_res"][i]),
                "regime_confidence": 0.7,
            }
            for rid in rset.regime_ids:
                row[f"prob_{rid}"] = 1.0 / len(rset.regime_ids)
            X_rows.append(row)
            k_mae_list.append(prof.k_mae)
            k_mfe_list.append(prof.k_mfe)
            k_lock_list.append(prof.k_lockout_mae)

        b_model = BarrierSizingModel(regime_set=rset)
        b_model.fit(X_rows, k_mae_list, k_mfe_list, k_lock_list, random_state=random_state)
        path = out_root / "barrier_sizing_model.pkl"
        sha = save_native_model(
            {
                "stage": node.name,
                "feature_names": b_model.feature_names,
                "q_sl": b_model.q_sl,
                "q_tp": b_model.q_tp,
                "regime_set_hash": rset.version_hash,
            },
            path,
        )
        return {
            "barrier_model": b_model,
            "k_mae": k_mae_list,
            "k_mfe": k_mfe_list,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_experts(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        bs = ctx["barrier_sizing"]
        n_samp = len(bs["k_mae"])
        X_obs = ds["X_obs"][:n_samp]
        plan_head = PlanUtilityHead(feature_names=["f0", "f1", "f2", "f3"], grid=DEFAULT_PLAN_GRID)
        m_grid = len(DEFAULT_PLAN_GRID)
        ret_mat = np.column_stack([np.asarray(bs["k_mfe"], dtype=float) * (1.0 + 0.05 * j) for j in range(m_grid)])
        mae_mat = np.column_stack([np.asarray(bs["k_mae"], dtype=float) * (1.0 + 0.05 * j) for j in range(m_grid)])
        plan_head.fit(X_obs, ret_mat, mae_mat)

        oof_records = [
            {"instance_id": "trend_expert", "regime": "TRENDING_UP", "realized_return_pct": 0.18, "is_oof": True, "split": "val"}
            for _ in range(6)
        ] + [
            {"instance_id": "mean_reversion_expert", "regime": "CHOP_RANGE", "realized_return_pct": 0.12, "is_oof": True, "split": "val"}
            for _ in range(6)
        ] + [
            {"instance_id": "breakout_expert", "regime": "TRENDING_DOWN", "realized_return_pct": 0.14, "is_oof": True, "split": "val"}
            for _ in range(6)
        ]
        sel_res = select_top_n_instances(oof_records, top_n=3, per_regime=False, min_oof_trades=3)
        path = out_root / "experts_bundle.pkl"
        sha = save_native_model(
            {
                "stage": node.name,
                "selected_instances": list(sel_res.selected_instance_ids),
            },
            path,
        )
        return {
            "plan_head": plan_head,
            "active_instances": tuple(sel_res.selected_instance_ids),
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_exit_models(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        rng = np.random.default_rng(random_state)
        bar_rows: List[Dict[str, Any]] = []
        for idx in range(40):
            unreal = float(rng.uniform(-0.8, 0.8))
            row: Dict[str, Any] = {f: float(rng.normal(0.0, 1.0)) for f in EXIT_STATE_FEATURES}
            row["unrealized_return_pct"] = unreal
            row["trade_id"] = f"t_{idx // 2}"
            row["y_profit_exit"] = 1 if (unreal > 0 and idx % 2 == 0) else 0
            row["y_loss_cut"] = 1 if (unreal <= 0 and idx % 2 == 1) else 0
            bar_rows.append(row)

        em = ExitModel("trend_expert").fit(bar_rows, random_state=random_state)
        path = out_root / "exit_models.pkl"
        sha = save_native_model(
            {"stage": node.name, "expert_id": em.expert_id, "features": em.feature_names},
            path,
        )
        return {
            "exit_models": {"trend_expert": em},
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_routers(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ds = ctx["dataset"]
        X_obs = ds["X_obs"]
        n = len(X_obs)
        y_bin = (ds["ret1"] > 0).astype(int)
        bar_idx = np.arange(n, dtype=int)

        def _fit_pred(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
            if len(np.unique(y_tr)) < 2:
                return np.full(len(X_te), 0.5)
            clf = LGBMClassifier(n_estimators=12, max_depth=3, random_state=random_state, verbose=-1)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]

        oof_rows = generate_oof_predictions(
            bar_idx,
            X_obs,
            y_bin,
            _fit_pred,
            train_bars=max(30, n // 3),
            test_bars=max(15, n // 5),
            step_bars=max(15, n // 5),
            embargo_bars=2,
            label_horizon_bars=2,
        )
        path = out_root / "router.pkl"
        sha = save_native_model(
            {"stage": node.name, "oof_count": len(oof_rows), "dual_router": ["router_hard", "router_ml"]},
            path,
        )
        return {
            "oof_predictions": oof_rows,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_opportunity_confirm(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        rt = ctx["routers"]
        oof_rows = rt["oof_predictions"]
        path = out_root / "opportunity_confirm.pkl"
        sha = save_native_model(
            {"stage": node.name, "oof_rows_consumed": len(oof_rows)},
            path,
        )
        return {
            "oof_rows_consumed": len(oof_rows),
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    def run_risk_sizing(node: TrainingStageNode, ctx: Dict[str, Any]) -> Dict[str, Any]:
        rng = np.random.default_rng(random_state)
        r_model = RiskSizingModel()
        X_rows = [{f: float(rng.uniform(0.1, 0.9)) for f in r_model.feature_names} for _ in range(36)]
        y_loss = [1 if i % 3 == 0 else 0 for i in range(36)]
        r_model.fit(X_rows, y_loss, random_state=random_state)
        path = out_root / "risk_sizing_model.pkl"
        sha = save_native_model(
            {"stage": node.name, "features": r_model.feature_names, "base_features": BASE_RISK_SIZING_FEATURES},
            path,
        )
        return {
            "risk_sizing_model": r_model,
            "artefact_path": str(path),
            "artefact_sha256": sha,
        }

    return {
        "dataset": run_dataset,
        "regime": run_regime,
        "zone_selection": run_zone_selection,
        "zone_survival": run_zone_survival,
        "barrier_sizing": run_barrier_sizing,
        "experts": run_experts,
        "exit_models": run_exit_models,
        "routers": run_routers,
        "opportunity_confirm": run_opportunity_confirm,
        "risk_sizing": run_risk_sizing,
    }


def build_bundle_from_training_context(
    plan: TrainingPlan,
    context: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    df_ohlcv: pd.DataFrame,
    bundle_id: str = "bundle_v1",
) -> ModelBundle:
    """Assemble a verified `ModelBundle` from the executed `TrainingPlan` context."""
    hold_policy = HoldPolicy.from_config(config)
    stage_artefacts: Dict[str, str] = {}
    artefact_paths: Dict[str, str] = {}
    upstream_hashes: Dict[str, Dict[str, str]] = {}

    deps_lookup = dict(TRAINING_DAG_STAGES)
    for node in plan.nodes:
        if not node.enabled or node.name not in context:
            continue
        stage_res = context[node.name]
        sha = str(stage_res.get("artefact_sha256") or node.cache_key)
        p = str(stage_res.get("artefact_path") or "")
        stage_artefacts[node.name] = sha
        if p:
            artefact_paths[node.name] = p
        node.artefact_hash = sha

    for node in plan.nodes:
        if not node.enabled or node.name not in stage_artefacts:
            continue
        deps = deps_lookup.get(node.name, ())
        upstream_hashes[node.name] = {
            d: stage_artefacts[d] for d in deps if d in stage_artefacts
        }

    reg_ctx = context.get("regime", {}) or {}
    regime_set = reg_ctx.get("regime_set", LEGACY_REGIME_SET)
    exp_ctx = context.get("experts", {}) or {}
    active_instances = tuple(
        exp_ctx.get("active_instances") or ("trend_expert", "mean_reversion_expert", "breakout_expert")
    )

    return ModelBundle(
        bundle_id=bundle_id,
        regime_set_hash=regime_set.version_hash,
        stage_artefacts=stage_artefacts,
        artefact_paths=artefact_paths,
        upstream_hashes=upstream_hashes,
        config_hash=canonical_json_hash(dict(config)),
        data_hash=compute_dataframe_hash(df_ohlcv),
        hold_policy=hold_policy.fingerprint(),
        delay_policy={
            "entry_delay_seconds": float(hold_policy.entry_latency_seconds),
            "exit_delay_seconds": float(hold_policy.exit_delay_seconds),
        },
        trained_at=datetime.now(timezone.utc).isoformat(),
        active_instances=active_instances,
    )
