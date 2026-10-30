#!/usr/bin/env python
"""
Full-stack walk-forward backtest + performance report (Phase 5, Module B /
§5, M5.2). Closes §2.2: computes the system's realized performance against
docs/Project_Plan.md §1's original success criteria — which four phases of
tests confirmed mechanics but never measured.

What "full-stack" means here: every bar goes through Features → Regime →
Structure → Experts → Opportunity Filter → Router → RISK LAYER → EXECUTION
(paper_sim fills, real position lifecycle, TP/SL/timeout exits) — the same
code paths scripts/run.py uses at deployment. Metrics are computed on the
REALIZED trade sequence, not any single layer's predictions, so fee drag,
risk-layer size reduction, partial fills, and compounding are all priced in.

Walk-forward design decision (documented): models are FIXED across the run
(currently rule-based fallback until manual retrain post-rename), so there
is no per-fold retraining to sequence. Instead of fake re-warmups per fold,
the system runs ONE CONTINUOUS replay and the timeline is segmented into
contiguous, non-overlapping evaluation WINDOWS (default 1440 bars ≈ one day
at 1m) for per-period stability. This avoids double-counted overlapping
test segments while answering the same question: is performance stable
across time periods?

Baselines (§5.2 — rigor checks on component complexity):
  vs_buy_and_hold                     one long episode per window
  vs_simple_momentum                  Project_Plan.md's own stated baseline
  vs_always_highest_raw_confidence    Router replaced by naive argmax-confidence
  vs_trade_everything_experts_propose Opportunity Filter gate removed

Calibration (§5.3): every expert candidate is labeled via the triple-barrier
proxy (independent of routing/filtering) for reliability buckets; router
calibration uses executed trades' selection_confidence vs realized win.

Output (§5.4): data/reports/backtest_<timestamp>.{json,md} + a row in
data/registry.db::experiments.

Usage:
    python scripts/full_stack_walk_forward_backtest.py \
        --input data/raw/BTCUSDT.csv --env paper

    # Smoke test without a data file:
    python scripts/full_stack_walk_forward_backtest.py --synthetic 900 \
        --relax-liquidity --window-bars 300
"""
import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run import OpenPosition, build_execution_engine, process_bar  # noqa: E402

from src.evaluation.trade_duration import duration_report  # noqa: E402
from src.labeling.harness_labels import LabelContext, entry_price_for  # noqa: E402
from src.labeling.hold_policy import HoldPolicy, alignment_settings  # noqa: E402
from src.labeling.triple_barrier import (  # noqa: E402
    barrier_volatility_pct,
    barriers_for_regime,
    label_triple_barrier,
)
from src.layers.market_data import MarketDataLayer  # noqa: E402
from src.layers.risk_layer import RiskLayer, RiskStateStore, compute_base_size  # noqa: E402
from src.pipeline import TradingPipeline  # noqa: E402
from src.registry.db import log_experiment  # noqa: E402
from src.reporting.performance_report import (  # noqa: E402
    _sanitize,
    baseline_verdicts,
    buy_and_hold_episodes,
    chained_equity_curve,
    compute_breakdowns,
    compute_trade_stats,
    compute_turnover,
    confidence_calibration,
    equity_curve_drawdown_pct,
    miscalibration_score,
    momentum_baseline_episodes,
    render_markdown_report,
    success_criteria_check,
    total_return_pct,
)
from src.state.contracts import (  # noqa: E402
    ExpertAction,
    ExpertOutputs,
    OpportunityScore,
    RiskDecision,
    RouterDecision,
)
from src.utils.config_helpers import load_layered_config  # noqa: E402
from src.utils.synthetic import synthetic_ohlcv  # noqa: E402

# ---------------------------------------------------------------------------
# Baseline component swaps
# ---------------------------------------------------------------------------

class NaiveConfidenceRouter:
    """
    Baseline router (§5.2 'vs_always_highest_raw_confidence'): identical
    interface to Router.decide(), but selection is always the highest raw
    candidate confidence above the configured floor — no blending, no model,
    no switching-context features. Isolates what the real Router adds.
    """

    def __init__(self, fallback_cfg: dict):
        self.floor = float(fallback_cfg.get("confidence_floor_to_trade", 0.3))

    def decide(self, expert_outputs, regime, zone_map, opportunity_score) -> RouterDecision:
        ts, sym = opportunity_score.timestamp, opportunity_score.symbol
        if not opportunity_score.tradable:
            return RouterDecision(
                timestamp=ts, symbol=sym, selected_expert="no_trade",
                selection_confidence=1.0 - opportunity_score.tradable_probability,
                no_trade_probability=1.0, expert_ranking={},
                reasoning="opportunity_filter_rejected")
        cands = [o for o in expert_outputs.all() if o.action != ExpertAction.NO_TRADE]
        ranking = {o.expert_name: o.confidence for o in cands}
        if not cands:
            return RouterDecision(timestamp=ts, symbol=sym, selected_expert="no_trade",
                                  selection_confidence=0.0, no_trade_probability=1.0,
                                  expert_ranking=ranking, reasoning="no_candidates")
        best = max(cands, key=lambda o: o.confidence)
        if best.confidence < self.floor:
            return RouterDecision(timestamp=ts, symbol=sym, selected_expert="no_trade",
                                  selection_confidence=best.confidence,
                                  no_trade_probability=1.0, expert_ranking=ranking,
                                  reasoning="below_naive_floor")
        return RouterDecision(timestamp=ts, symbol=sym, selected_expert=best.expert_name,
                              selection_confidence=best.confidence,
                              no_trade_probability=1.0 - best.confidence,
                              expert_ranking=ranking, reasoning="naive_highest_confidence")

    def _update_switch_state(self, *_):  # interface compat (unused)
        pass


class UngatedOpportunityFilter:
    """
    Baseline filter (§5.2 'vs_trade_everything_experts_propose'): delegates
    to the real filter so breakdown scores stay populated, but forces
    tradable=True — the gate is removed, downstream risk layer still vets.
    """

    def __init__(self, inner):
        self._inner = inner

    def evaluate(self, *args, **kwargs) -> OpportunityScore:
        score = self._inner.evaluate(*args, **kwargs)
        return dc_replace(score, tradable=True, tradable_probability=max(score.tradable_probability, 0.999))


class SingleExpertLayer:
    """
    Ablation (--single-expert <name>, §13): keep only one expert's output,
    so the router chooses among that expert and no-trade. Isolates what
    each individual expert contributes versus the full ensemble.
    """

    def __init__(self, inner, expert_name: str):
        self._inner = inner
        self._expert_name = expert_name

    def evaluate(self, fv, zone_map, regime_state) -> ExpertOutputs:
        outputs = self._inner.evaluate(fv, zone_map, regime_state)
        keep = outputs.get_by_name(self._expert_name)
        if keep is None:
            return ExpertOutputs(timestamp=outputs.timestamp, symbol=outputs.symbol, outputs={})
        return ExpertOutputs(timestamp=outputs.timestamp, symbol=outputs.symbol,
                             outputs={self._expert_name: keep})


class AutoApproveRiskLayer:
    """
    Ablation (--no-risk-layer, §13 'what does the Risk Layer buy?'): every
    candidate the router selects is approved at risk-formula base size
    (equity × risk_pct ÷ stop distance) with NO liquidity reduction, NO
    caps, NO breakers gating entries. The persistent store stays live so
    equity/drawdown accounting in the loop keeps working identically.
    """

    def __init__(self, inner):
        self._inner = inner
        self.store = inner.store          # loop reads/writes store directly
        self.breaker_cfg = inner.breaker_cfg

    def evaluate(self, router_decision, expert_outputs, regime_state, zone_map, fv, market, equity) -> RiskDecision:
        ts, sym = router_decision.timestamp, router_decision.symbol
        if router_decision.selected_expert == "no_trade":
            return RiskDecision(timestamp=ts, symbol=sym, approved=False, risk_score=0.0,
                                base_size=0.0, adjusted_size=0.0, size_reduction_pct=100.0,
                                rejection_reason="no_trade_selected")
        expert_output = expert_outputs.get_by_name(router_decision.selected_expert)
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            return RiskDecision(timestamp=ts, symbol=sym, approved=False, risk_score=0.0,
                                base_size=0.0, adjusted_size=0.0, size_reduction_pct=100.0,
                                rejection_reason="selected_expert_has_no_candidate")
        base = compute_base_size(equity, expert_output, self._inner.sizing_cfg)
        return RiskDecision(timestamp=ts, symbol=sym, approved=True, risk_score=0.0,
                            base_size=base, adjusted_size=base, size_reduction_pct=0.0,
                            rejection_reason=None)

    def record_trade(self, timestamp: int) -> None:
        self._inner.record_trade(timestamp)

    def advance_bar(self) -> None:
        self._inner.advance_bar()

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# One full-stack pass
# ---------------------------------------------------------------------------

def _fresh_risk_layer(config: dict, tmp_dir: str) -> RiskLayer:
    hold_policy, require_aligned = alignment_settings(config)
    return RiskLayer(
        hold_policy=hold_policy, require_aligned_models=require_aligned,
        config=config.get("risk", {}),
        store=RiskStateStore(db_path=str(Path(tmp_dir) / "registry.db")),
        model_path=config.get("risk", {}).get("ml_model", {}).get("model_path"),
        data_granularity=config.get("system", {}).get("data_granularity", "1m"),
        kill_switch_path=str(Path(tmp_dir) / "NO_SUCH_KILL_SWITCH"),
    )


def run_full_stack_pass(df_ohlcv: pd.DataFrame, symbol: str, config: dict,
                        initial_equity: float,
                        router_override=None,
                        bypass_filter: bool = False,
                        collect_expert_candidates: bool = False,
                        single_expert: Optional[str] = None,
                        disable_risk_layer: bool = False) -> dict:
    """
    Run the complete stack bar-by-bar over df_ohlcv (mirrors run.py's
    run_historical loop body exactly — exits first, config-driven position
    limit, rate limits, breaker trips) WITHOUT touching production stores.
    Returns trades, bar-aligned equity curve, expert candidates, router
    calibration pairs.

    Ablation hooks (§13 ladder rows): single_expert keeps one expert only;
    disable_risk_layer auto-approves at base size.
    """
    pipeline = TradingPipeline(symbol=symbol, config=copy.deepcopy(config))
    if router_override is not None:
        pipeline.router = router_override
    if bypass_filter:
        pipeline.opportunity_filter = UngatedOpportunityFilter(pipeline.opportunity_filter)
    if single_expert is not None:
        pipeline.expert_layer = SingleExpertLayer(pipeline.expert_layer, single_expert)

    tmp_dir = tempfile.mkdtemp(prefix="fsb_pass_")
    risk_layer = _fresh_risk_layer(config, tmp_dir)
    if disable_risk_layer:
        risk_layer = AutoApproveRiskLayer(risk_layer)
    exec_engine = build_execution_engine(config, pipeline.connector, symbol)
    fee_pct = config.get("labeling", {}).get("fee_pct", 0.04)

    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
    equity = initial_equity
    open_pos: Optional[OpenPosition] = None
    trades: List[dict] = []
    equity_curve: List[float] = []
    bar_ts: List[int] = []
    candidates: List[dict] = []   # expert-candidate records for calibration
    routed_pairs: List[Tuple[float, bool]] = []

    current_idx = {"i": 0}

    def on_open(position: OpenPosition) -> str:
        return f"bt-{current_idx['i']}"

    def on_close(position: OpenPosition, exit_timestamp: int, exit_info: dict) -> None:
        win = exit_info["realized_pnl_pct"] > 0
        if position.selection_confidence is not None:
            routed_pairs.append((position.selection_confidence, win))
        dt = datetime.fromtimestamp(position.entry_timestamp, tz=timezone.utc)
        trades.append({
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": exit_timestamp,
            "expert": position.expert,
            "regime": position.regime,
            "direction": position.direction,
            "size_usdt": position.size_usdt,
            "entry_price": position.entry_price,
            "pnl_pct": exit_info["realized_pnl_pct"],
            "pnl_usdt": exit_info["realized_pnl_usdt"],
            "exit_reason": exit_info["exit_reason"],
            "bars_held": exit_info["bars_held"],
            # Wall-clock fill-to-exit time: the number the minimum-trade-
            # duration rule is measured on (src/labeling/hold_policy.py).
            "hold_seconds": exit_info["hold_seconds"],
            "min_hold_bars": position.min_hold_bars,
            "hour_utc": dt.hour,
            "day_of_week": dt.weekday(),
            "selection_confidence": position.selection_confidence,
            "risk_score": position.risk_score,
        })

    try:
        for snap_idx, snapshot in enumerate(snapshots):
            state = pipeline.run_live_bar(snapshot)
            if state is None:
                continue  # warmup
            current_idx["i"] = snap_idx

            # The per-bar block -- exits first (never gated by the Risk
            # Layer), then the single entry path, the lockout, the
            # barriers -- is the SAME function the live loop, the
            # historical loop and the GUI replay call. The harness used to
            # carry its own copy; a rule added to one (the minimum trade
            # duration) silently missed the other.
            outcome = process_bar(
                state, risk_layer=risk_layer, execution_engine=exec_engine,
                position=open_pos, equity=equity, fee_pct=fee_pct, config=config,
                on_trade_open=on_open, on_trade_close=on_close,
            )
            open_pos, equity = outcome.position, outcome.equity
            state = outcome.state

            # Candidate harvesting for per-expert calibration (upstream of
            # filter/router -- every proposal gets an outcome sample).
            if collect_expert_candidates and state.expert_outputs is not None:
                fv = state.feature_vector
                for out in state.expert_outputs.all():
                    if out.action == ExpertAction.NO_TRADE:
                        continue
                    candidates.append({
                        "snapshot_index": snap_idx,
                        "entry_timestamp": state.timestamp,
                        "expert": out.expert_name,
                        "confidence": out.confidence,
                        "direction": 1 if out.action == ExpertAction.BUY else -1,
                        "entry_price": state.market_snapshot.close,
                        "volatility_medium": barrier_volatility_pct(fv),
                        "regime": state.regime_state.primary_regime.name,
                    })

            equity_curve.append(equity)
            bar_ts.append(state.timestamp)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "trades": trades,
        "snapshots": snapshots,
        "equity_curve": equity_curve,
        "bar_ts": bar_ts,
        "candidates": candidates,
        "routed_pairs": routed_pairs,
        "final_equity": equity,
    }


# ---------------------------------------------------------------------------
# Expert calibration labeling (triple-barrier proxy, §5.3)
# ---------------------------------------------------------------------------

def label_expert_candidates(candidates: List[dict], snapshots: list,
                            config: dict) -> Dict[str, List[Tuple[float, bool]]]:
    """Label every harvested candidate the way the harness would trade it.

    Independent of routing/filtering -- fair per-expert reliability data --
    but identical to a harness trade in everything else: simulated-fill
    entry price, regime-specific barriers, the shared exit rule and the
    minimum-hold lockout (``src/labeling/harness_labels.py``).
    """
    ctx = LabelContext.from_config(config)
    max_hold = ctx.policy.max_holding_bars
    pairs: Dict[str, List[Tuple[float, bool]]] = {}
    for cand in candidates:
        idx = cand["snapshot_index"]
        fwd = snapshots[idx + 1: idx + 1 + max_hold]
        if not fwd:
            continue
        tp, sl = barriers_for_regime(config, cand["regime"], cand["volatility_medium"])
        entry_price = entry_price_for(ctx, snapshots[idx], cand["direction"])
        try:
            outcome = label_triple_barrier(
                forward_bars=fwd, entry_price=entry_price,
                direction=cand["direction"], take_profit_pct=tp, stop_loss_pct=sl,
                max_holding_bars=max_hold, fee_pct=ctx.fee_pct,
                entry_timestamp=cand["entry_timestamp"], min_hold_bars=ctx.min_hold_bars)
        except ValueError:
            continue
        if outcome.bars_held < ctx.min_hold_bars:
            continue                      # data ended inside the lockout: no legal outcome
        pairs.setdefault(cand["expert"], []).append((cand["confidence"], outcome.realized_return_pct > 0))
    return pairs


# ---------------------------------------------------------------------------
# Metric finalization helpers
# ---------------------------------------------------------------------------

def finalize(trades: List[dict], curve: List[float], initial_equity: float) -> dict:
    stats = compute_trade_stats(trades)
    stats["total_return_pct"] = total_return_pct(initial_equity, curve[-1] if curve else initial_equity)
    stats["max_drawdown_pct"] = round(equity_curve_drawdown_pct(curve), 4)
    turnover_x, notional = compute_turnover(trades, initial_equity)
    stats["turnover_x"] = round(turnover_x, 3) if turnover_x is not None else None
    stats["total_notional_usdt"] = round(notional, 2)
    return stats


def window_slices(bar_ts: List[int], window_bars: int) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """Contiguous non-overlapping index windows over the processed-bar axis."""
    n = len(bar_ts)
    slices, bounds = [], []
    start = 0
    idx = 0
    while start < n:
        end = min(start + window_bars, n)
        bounds.append((start, end))
        slices.append({
            "index": idx,
            "start_ts": bar_ts[start],
            "end_ts": bar_ts[end - 1],
            "n_bars": end - start,
        })
        idx += 1
        start = end
    return slices, bounds


def trades_in_window(trades: List[dict], t0: int, t1: int) -> List[dict]:
    return [t for t in trades if t0 <= t["entry_timestamp"] <= t1]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

RELAX_KEYS = [
    ("opportunity_filter", "hard_rules", "max_spread_pct"),
    ("opportunity_filter", "hard_rules", "min_depth_usdt"),
    ("opportunity_filter", "hard_rules", "edge_vs_fees_threshold"),
]


def maybe_relax_liquidity(config: dict, relax: bool) -> dict:
    cfg = copy.deepcopy(config)
    if not relax:
        return cfg
    hr = cfg.setdefault("opportunity_filter", {}).setdefault("hard_rules", {})
    hr["max_spread_pct"] = 0.5          # matches market.max_spread_pct ceiling
    hr["min_depth_usdt"] = 0.0
    hr["edge_vs_fees_threshold"] = -10.0
    logger.warning("=" * 70)
    logger.warning("LIQUIDITY GATES RELAXED FOR THIS RUN (--relax-liquidity):")
    for path in RELAX_KEYS:
        logger.warning(f"  {'.'.join(path)} overridden")
    logger.warning("This mirrors docs/test_integration_phase4.py's documented local")
    logger.warning("relaxation — the shipped synthetic spread (~0.09% mean) exceeds")
    logger.warning("shipped max_spread_pct (0.05%), which would reject ~all bars.")
    logger.warning("Numbers from a relaxed run measure PIPELINE BEHAVIOR, not")
    logger.warning("production edge. Do not read expectancy from this mode.")
    logger.warning("=" * 70)
    return cfg


def generate_synthetic_ohlcv(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Mixed-regime synthetic BTCUSDT (trend/chop/trend/panic-ish segments).

    The generator itself lives in ``src/utils/synthetic.py`` (shared with
    the latency benchmark and the Trade Map GUI); the harness's own
    contract -- bars ending around "now" -- is just its start timestamp.
    """
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return synthetic_ohlcv(n_bars, seed=seed, start=start)


def strip_regime_gating(config: dict) -> dict:
    """
    Ablation (--no-regime-gating, §13): remove every regime-dependent gate
    — experts accept any regime at any confidence, the opportunity filter
    and risk layer stop rejecting on regime. Regime FEATURES still flow
    (they inform models); only the hard gating is removed.
    """
    cfg = copy.deepcopy(config)
    for expert_cfg in (cfg.get("experts", {}) or {}).values():
        if isinstance(expert_cfg, dict):
            expert_cfg["regime_requirement"] = []
            expert_cfg["regime_confidence_floor"] = 0.0
    for section in ("opportunity_filter", "risk"):
        hard = ((cfg.get(section, {}) or {}).get("hard_rules", {}) or {})
        if "reject_on_regime" in hard or hard != {}:
            hard["reject_on_regime"] = []
    return cfg


def run_backtest(df_ohlcv: pd.DataFrame, symbol: str, config: dict,
                 initial_equity: float = 10000.0, window_bars: int = 1440,
                 skip_baselines: bool = False,
                 single_expert: Optional[str] = None,
                 disable_risk_layer: bool = False,
                 disable_regime_gating: bool = False) -> dict:
    labeling_fee = config.get("labeling", {}).get("fee_pct", 0.04)
    fee_rt_pct = labeling_fee * 2

    ablation_applied = bool(single_expert or disable_risk_layer or disable_regime_gating)
    if disable_regime_gating:
        config = strip_regime_gating(config)

    logger.info("Pass 1/3: full stack (main)...")
    main_pass_kwargs = {}
    if single_expert:
        main_pass_kwargs["single_expert"] = single_expert
    if disable_risk_layer:
        main_pass_kwargs["disable_risk_layer"] = True
    main = run_full_stack_pass(df_ohlcv, symbol, config, initial_equity,
                               collect_expert_candidates=True, **main_pass_kwargs)
    snapshots: list = main["snapshots"]

    naive_res = unfiltered_res = None
    if not skip_baselines:
        logger.info("Pass 2/3: naive highest-confidence router baseline...")
        naive_router = NaiveConfidenceRouter(config.get("router", {}).get("fallback_behavior", {}))
        naive_res = run_full_stack_pass(df_ohlcv, symbol, config, initial_equity,
                                        router_override=naive_router)

        logger.info("Pass 3/3: no-opportunity-filter baseline...")
        unfiltered_res = run_full_stack_pass(df_ohlcv, symbol, config, initial_equity,
                                             bypass_filter=True)

    aggregate = finalize(main["trades"], main["equity_curve"], initial_equity)
    aggregate_breakdowns = compute_breakdowns(main["trades"])

    # Windows over the continuous replay
    slices, bounds = window_slices(main["bar_ts"], window_bars)
    windows = []
    for s, (i0, i1) in zip(slices, bounds, strict=False):
        w_trades = trades_in_window(main["trades"], s["start_ts"], s["end_ts"])
        seg_curve = main["equity_curve"][i0:i1]
        wm = finalize(w_trades, seg_curve, initial_equity)
        s["metrics"] = wm
        windows.append(s)

    # Price-series baselines
    closes = np.array([s.close for s in snapshots])
    ts_all = [s.timestamp for s in snapshots]
    fe_cfg = config.get("feature_engine", {}) or {}
    mp = fe_cfg.get("momentum_periods", {}) or {}
    if isinstance(mp, dict):
        mom_period = int(mp.get("medium", 5))
    elif isinstance(mp, (list, tuple)) and len(mp) > 1:
        mom_period = int(mp[1])   # legacy positional list [short, medium, long]
    else:
        mom_period = 5
    thr = config.get("experts", {}).get("trend", {}).get("momentum_threshold", 0.05)
    momentum_eps = momentum_baseline_episodes(ts_all, closes, mom_period, thr, fee_rt_pct)
    bh_bounds_idx = [(b[0], b[1] - 1) for b in bounds if b[1] > b[0]]
    bh_eps = buy_and_hold_episodes(ts_all, closes, bh_bounds_idx, fee_rt_pct)

    baselines: Dict[str, dict] = {}
    if not skip_baselines:
        baselines["naive_router"] = finalize(naive_res["trades"], naive_res["equity_curve"], initial_equity)
        baselines["unfiltered"] = finalize(unfiltered_res["trades"], unfiltered_res["equity_curve"], initial_equity)
    m_curve, m_final = chained_equity_curve(momentum_eps, initial_equity)
    baselines["simple_momentum"] = finalize(momentum_eps, m_curve, initial_equity)
    b_curve, b_final = chained_equity_curve(bh_eps, initial_equity)
    baselines["buy_and_hold"] = finalize(bh_eps, b_curve, initial_equity)

    # Calibration (§5.3)
    expert_pairs = label_expert_candidates(main["candidates"], snapshots, config)
    expert_cal = {name: confidence_calibration(pairs) for name, pairs in sorted(expert_pairs.items())}
    expert_scores = {name: miscalibration_score(cal) for name, cal in expert_cal.items()}
    router_cal = confidence_calibration(main["routed_pairs"])
    report_calibration = {
        "experts": expert_cal,
        "expert_miscalibration_scores": expert_scores,
        "router": router_cal,
        "router_miscalibration_score": miscalibration_score(router_cal),
        "note": "Expert outcomes labeled via triple-barrier proxy on every "
                "candidate bar (independent of routing/filtering); router "
                "outcomes are executed trades only.",
    }

    verdicts = baseline_verdicts(aggregate, baselines.get("naive_router", {}),
                                 baselines.get("unfiltered", {})) if not skip_baselines else {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "granularity": config.get("system", {}).get("data_granularity", "1m"),
        "symbol": symbol,
        "initial_equity": initial_equity,
        "ablation": {
            "single_expert": single_expert,
            "risk_layer_disabled": disable_risk_layer,
            "regime_gating_disabled": disable_regime_gating,
            "applied": ablation_applied,
        },
        "input": {
            "n_bars": int(len(snapshots)),
            "start_ts": ts_all[0] if ts_all else None,
            "end_ts": ts_all[-1] if ts_all else None,
        },
        "windows": windows,
        "aggregate": aggregate,
        "aggregate_breakdowns": aggregate_breakdowns,
        "baselines": baselines,
        "baseline_verdicts": verdicts,
        "calibration": report_calibration,
        # Minimum trade duration: measured on every trade the main pass took.
        # ``passed`` is False if ANY trade was held shorter than the minimum.
        "trade_duration": duration_report(main["trades"], HoldPolicy.from_config(config)),
        "success_criteria": success_criteria_check(aggregate, baselines["simple_momentum"]),
    }
    return report


def config_hash(config: dict) -> str:
    blob = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default=None, help="OHLCV CSV/Parquet")
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Generate N synthetic bars instead of --input")
    parser.add_argument("--env", default="paper", choices=["paper", "live"])
    parser.add_argument("--config", default=None, help="Explicit config file (overrides --env)")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--refit", action="store_true",
                        help="Refit the harness-ordered TrainingPlan per walk-forward fold")
    parser.add_argument("--window-bars", type=int, default=1440)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--relax-liquidity", action="store_true",
                        help="Relax spread/depth/edge hard rules in-memory (documented local "
                             "relaxation; measures pipeline behavior, NOT production edge)")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--single-expert", default=None, metavar="NAME",
                        help="Ablation: keep only this expert's output (e.g. trend_expert)")
    parser.add_argument("--no-risk-layer", action="store_true",
                        help="Ablation: auto-approve at base size — no caps/breakers/liquidity sizing")
    parser.add_argument("--no-regime-gating", action="store_true",
                        help="Ablation: experts/filter/risk stop gating on regime")
    parser.add_argument("--report-dir", default="data/reports")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = load_layered_config(args.env)

    if args.relax_liquidity:
        config = maybe_relax_liquidity(config, True)

    if args.synthetic:
        df = generate_synthetic_ohlcv(args.synthetic)
        logger.info(f"Generated {len(df)} synthetic bars")
    elif args.input:
        path = Path(args.input)
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        if "timestamp" in df.columns and not np.issubdtype(df["timestamp"].dtype, np.integer):
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        raise SystemExit("Provide --input <ohlcv file> or --synthetic <n_bars>")

    symbol = args.symbol or config.get("system", {}).get("symbol", "BTC/USDT")

    report = run_backtest(df, symbol, config, args.initial_equity,
                          args.window_bars, args.skip_baselines,
                          single_expert=args.single_expert,
                          disable_risk_layer=args.no_risk_layer,
                          disable_regime_gating=args.no_regime_gating)
    report["config_hash"] = config_hash(config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"backtest_{stamp}.json"
    md_path = out_dir / f"backtest_{stamp}.md"

    safe = _sanitize(report)
    with open(json_path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    md_text = render_markdown_report(safe)
    with open(md_path, "w") as f:
        f.write(md_text)

    core_metrics = {k: report["aggregate"].get(k) for k in
                    ("n_trades", "total_return_pct", "win_rate_pct",
                     "expectancy_per_trade_pct", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown_pct")}
    try:
        exp_id = log_experiment(run_type="backtest", config=config,
                                metrics={"aggregate": core_metrics,
                                         "success_criteria_passed":
                                             sum(1 for c in report["success_criteria"] if c["pass"])},
                                component="full_stack")
        logger.info(f"Logged experiment {exp_id}")
    except Exception as e:
        logger.warning(f"Experiment logging skipped: {e}")

    print(md_text)
    logger.info(f"Report written: {json_path} , {md_path}")


if __name__ == "__main__":
    main()
