"""
Performance reporting (Phase 5, Module B / §5): compute the full metric
suite from a realized trade sequence and compare against Project_Plan.md's
original success criteria.

Metrics computed directly against docs/Project_Plan.md §1:
  - Positive expectancy after fees/slippage vs baseline simple momentum
  - Win rate + risk-adjusted return better than simple momentum
  - Max drawdown < 8% in backtests

All functions here are pure: trades in, dicts out. No pipeline imports —
this module is deliberately independent so tests can exercise it without
constructing the stack.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import math

import numpy as np

# ---------------------------------------------------------------------------
# Trade-sequence metrics
# ---------------------------------------------------------------------------

def compute_trade_stats(trades: List[dict]) -> dict:
    """
    Per-trade statistics from a list of closed-trade dicts. Each trade dict
    must carry: pnl_pct (net, %), pnl_usdt. Optional: anything else (expert,
    regime, hour_utc, ...) used only by breakdowns.

    Sharpe/Sortino here are PER-TRADE ratios (mean/std of trade returns),
    not annualized — annualizing assumes a trade frequency that backtests
    don't guarantee. Documented as such wherever surfaced.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate_pct": None, "profit_factor": None,
            "expectancy_per_trade_pct": None, "avg_win_pct": None, "avg_loss_pct": None,
            "sharpe_ratio": None, "sortino_ratio": None,
            "gross_win_pct": 0.0, "gross_loss_pct": 0.0,
        }

    returns = np.array([float(t["pnl_pct"]) for t in trades], dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=0)) if n > 1 else 0.0

    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt((downside ** 2).mean())) if n > 0 else 0.0

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    profit_factor = None
    if gross_loss > 1e-12:
        profit_factor = gross_win / gross_loss
    elif gross_win > 1e-12:
        profit_factor = float("inf")  # serialized as string downstream

    return {
        "n_trades": int(n),
        "win_rate_pct": float(len(wins) / n * 100),
        "profit_factor": profit_factor,
        "expectancy_per_trade_pct": mean_r,
        "avg_win_pct": float(wins.mean()) if len(wins) else None,
        "avg_loss_pct": float(losses.mean()) if len(losses) else None,
        "sharpe_ratio": (mean_r / std_r) if std_r > 1e-12 else None,
        "sortino_ratio": (mean_r / downside_std) if downside_std > 1e-12 else None,
        "gross_win_pct": gross_win,
        "gross_loss_pct": gross_loss,
    }


def compute_turnover(trades: List[dict], initial_equity: float) -> Tuple[Optional[float], float]:
    """
    Compute turnover multiple (total_notional / initial_equity) and total
    notional in USDT from a list of trade dicts.

    Each trade dict should carry ``entry_price`` and ``size_usdt`` (or
    ``filled_size``) — defaults to 0 when absent.
    """
    notional = 0.0
    for t in trades:
        price = float(t.get("entry_price", 0.0))
        size = float(t.get("size_usdt", t.get("filled_size", 0.0)))
        notional += price * size
    turnover_x = notional / initial_equity if initial_equity > 0 else None
    return turnover_x, notional


def equity_curve_drawdown_pct(equity_curve: List[float]) -> float:
    """
    Max peak-to-trough drawdown as a POSITIVE percentage (e.g. 6.5 means
    equity fell 6.5% from its peak at some point). Returns 0.0 for empty or
    monotonic curves.
    """
    if not equity_curve:
        return 0.0
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    valid = running_max > 1e-12
    if not valid.any():
        return 0.0
    dd = (running_max[valid] - eq[valid]) / running_max[valid] * 100
    return float(dd.max())


def total_return_pct(initial_equity: float, final_equity: float) -> float:
    if initial_equity <= 0:
        return 0.0
    return float((final_equity / initial_equity - 1) * 100)


def strategy_metrics(trades: List[dict], initial_equity: float,
                     final_equity: float) -> dict:
    """Full headline block for one strategy pass."""
    stats = compute_trade_stats(trades)
    stats["total_return_pct"] = total_return_pct(initial_equity, final_equity)
    return stats


# ---------------------------------------------------------------------------
# Breakdowns (§5.2)
# ---------------------------------------------------------------------------

def _group_stats(subset: List[dict]) -> dict:
    s = compute_trade_stats(subset)
    return {
        "n_trades": s["n_trades"],
        "win_rate_pct": s["win_rate_pct"],
        "expectancy_per_trade_pct": s["expectancy_per_trade_pct"],
        "total_pnl_usdt": float(sum(t.get("pnl_usdt", 0.0) for t in subset)),
    }


def compute_breakdowns(trades: List[dict]) -> dict:
    def group_by(key_fn, label):
        groups: Dict[str, List[dict]] = {}
        for t in trades:
            k = key_fn(t)
            groups.setdefault(k, []).append(t)
        return {str(k): _group_stats(v) for k, v in sorted(groups.items())}

    return {
        "by_expert": group_by(lambda t: t.get("expert", "unknown"), "expert"),
        "by_regime": group_by(lambda t: t.get("regime", "unknown"), "regime"),
        "by_hour_of_day": group_by(lambda t: int(t.get("hour_utc", -1)), "hour"),
        "by_day_of_week": group_by(lambda t: int(t.get("day_of_week", -1)), "day"),
    }


# ---------------------------------------------------------------------------
# Calibration reliability data (§5.3)
# ---------------------------------------------------------------------------

def confidence_calibration(pairs: List[Tuple[float, bool]], n_buckets: int = 10) -> List[dict]:
    """
    Reliability-diagram buckets: predicted confidence bucketed into fixed
    0.1-wide bins over [0, 1]; realized win rate compared to mean predicted
    confidence per bucket. A well-calibrated model has realized ≈ predicted.
    """
    buckets = []
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        subset = [(c, w) for c, w in pairs if lo <= c < hi] if b < n_buckets - 1 \
            else [(c, w) for c, w in pairs if lo <= c <= hi]
        n = len(subset)
        buckets.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "avg_predicted_confidence": float(np.mean([c for c, _ in subset])) if n else None,
            "realized_win_rate_pct": float(sum(1 for _, w in subset if w) / n * 100) if n else None,
        })
    return buckets


def miscalibration_score(calibration: List[dict]) -> Optional[float]:
    """
    Mean |predicted − realized| weighted by bucket count (None if no data).
    Single number summarizing the reliability diagram's gap.
    """
    total_n = sum(b["n"] for b in calibration)
    if total_n == 0:
        return None
    acc = 0.0
    for b in calibration:
        if b["n"] == 0 or b["avg_predicted_confidence"] is None:
            continue
        gap = abs(b["avg_predicted_confidence"] - b["realized_win_rate_pct"] / 100.0)
        acc += gap * b["n"]
    return float(acc / total_n)


# ---------------------------------------------------------------------------
# Baseline strategies (§5.2 baseline_comparisons)
# ---------------------------------------------------------------------------

def momentum_baseline_episodes(timestamps: List[int], closes: np.ndarray,
                               momentum_period: int, threshold_pct: float,
                               fee_rt_pct: float) -> List[dict]:
    """
    Simple momentum strategy (Project_Plan.md's stated baseline):
      momentum > +threshold → long; < −threshold → short; otherwise flat.
    Positions persist while the signal holds. One round-trip fee applied on
    each open and each close. Episodes returned in the same dict shape as
    full-stack trades so identical metric functions apply.
    """
    episodes: List[dict] = []
    target = 0
    ep_start_i = None
    ep_ret = 1.0
    ep_dir = 0

    def close_ep(exit_i: int):
        nonlocal ep_ret, ep_dir, ep_start_i
        if ep_start_i is None:
            return
        pnl_pct = (ep_ret * (1 - fee_rt_pct / 100) - 1) * 100
        episodes.append({
            "entry_timestamp": timestamps[ep_start_i],
            "exit_timestamp": timestamps[exit_i],
            "expert": "simple_momentum", "regime": "n/a",
            "direction": ep_dir, "pnl_pct": pnl_pct,
            "pnl_usdt": 0.0,  # notional-free comparison; pct metrics only
            "hour_utc": datetime.fromtimestamp(timestamps[ep_start_i], tz=timezone.utc).hour,
            "day_of_week": datetime.fromtimestamp(timestamps[ep_start_i], tz=timezone.utc).weekday(),
            "selection_confidence": None, "risk_score": None,
            "exit_reason": "signal_flip",
        })
        ep_ret, ep_dir, ep_start_i = 1.0, 0, None

    for i in range(momentum_period, len(closes)):
        ret = closes[i] / closes[i - 1]
        if target != 0:
            ep_ret *= (ret if target > 0 else 1.0 / ret)

        mom = (closes[i] / closes[i - momentum_period] - 1) * 100
        new_target = 1 if mom > threshold_pct else (-1 if mom < -threshold_pct else 0)

        if new_target != target:
            close_ep(i)
            target = new_target
            if target != 0:
                ep_dir = target
                ep_start_i = i
                ep_ret *= (1 - fee_rt_pct / 100)

    close_ep(len(closes) - 1)
    return episodes


def buy_and_hold_episodes(timestamps: List[int], closes: np.ndarray,
                          window_bounds: List[Tuple[int, int]],
                          fee_rt_pct: float) -> List[dict]:
    """One long episode per window — buy at window start, sell at end."""
    episodes = []
    for (start_i, end_i) in window_bounds:
        if end_i <= start_i:
            continue
        pnl_pct = (closes[end_i] / closes[start_i] * (1 - fee_rt_pct / 100) ** 2 - 1) * 100
        ts0 = timestamps[start_i]
        episodes.append({
            "entry_timestamp": ts0,
            "exit_timestamp": timestamps[end_i],
            "expert": "buy_and_hold", "regime": "n/a",
            "direction": 1, "pnl_pct": pnl_pct, "pnl_usdt": 0.0,
            "hour_utc": datetime.fromtimestamp(ts0, tz=timezone.utc).hour,
            "day_of_week": datetime.fromtimestamp(ts0, tz=timezone.utc).weekday(),
            "selection_confidence": None, "risk_score": None,
            "exit_reason": "window_end",
        })
    return episodes


def chained_equity_curve(episodes_or_trades: List[dict], initial_equity: float) -> Tuple[List[float], float]:
    """
    Compound notional through an episode sequence chronologically. For
    pct-only baselines pnl_usdt is derived from compounding notional.
    """
    eq = initial_equity
    curve = [eq]
    for t in sorted(episodes_or_trades, key=lambda x: x["entry_timestamp"]):
        eq *= (1 + float(t["pnl_pct"]) / 100)
        curve.append(eq)
    return curve, eq


# ---------------------------------------------------------------------------
# Success criteria & baseline verdicts (§2.2 closure)
# ---------------------------------------------------------------------------

SUCCESS_CRITERIA = [
    "positive_expectancy_after_fees",
    "max_drawdown_below_8pct",
    "win_rate_beats_simple_momentum",
    "risk_adjusted_return_beats_simple_momentum",
]


def success_criteria_check(strategy: dict, momentum: dict,
                           max_drawdown_target_pct: float = 8.0) -> List[dict]:
    exp_ret = strategy.get("expectancy_per_trade_pct")
    mom_exp = momentum.get("expectancy_per_trade_pct")
    checks = [
        {
            "criterion": "positive_expectancy_after_fees",
            "target": "expectancy_per_trade_pct > 0",
            "actual": exp_ret,
            "pass": bool(exp_ret is not None and exp_ret > 0),
        },
        {
            "criterion": "max_drawdown_below_8pct",
            "target": f"max_drawdown_pct < {max_drawdown_target_pct}",
            "actual": strategy.get("max_drawdown_pct"),
            "pass": bool(strategy.get("max_drawdown_pct") is not None
                         and strategy["max_drawdown_pct"] < max_drawdown_target_pct),
        },
        {
            "criterion": "win_rate_beats_simple_momentum",
            "target": "strategy win_rate > simple_momentum win_rate",
            "actual": strategy.get("win_rate_pct"),
            "baseline_actual": momentum.get("win_rate_pct"),
            "pass": bool(strategy.get("win_rate_pct") is not None
                         and momentum.get("win_rate_pct") is not None
                         and strategy["win_rate_pct"] > momentum["win_rate_pct"]),
        },
        {
            "criterion": "risk_adjusted_return_beats_simple_momentum",
            "target": "strategy sharpe > simple_momentum sharpe",
            "actual": strategy.get("sharpe_ratio"),
            "baseline_actual": momentum.get("sharpe_ratio"),
            "pass": bool(strategy.get("sharpe_ratio") is not None
                         and momentum.get("sharpe_ratio") is not None
                         and strategy["sharpe_ratio"] > momentum["sharpe_ratio"]),
        },
    ]
    _ = mom_exp  # kept for signature clarity
    return checks


def baseline_verdicts(strategy: dict, naive_router: dict, unfiltered: dict) -> dict:
    """
    Rigor checks on whether Router and Opportunity Filter earn their
    complexity (plan §5.2): statistically-indistinguishable performance from
    the naive alternative would be a real finding.
    """

    def verdict(a: Optional[float], b: Optional[float], higher_better=True) -> str:
        if a is None or b is None:
            return "INSUFFICIENT_DATA"
        if abs(a - b) < 1e-9:
            return "INDISTINGUISHABLE"
        better = (a > b) if higher_better else (a < b)
        return "ADDS_VALUE" if better else "SUBTRACTS_VALUE"

    return {
        "router_vs_naive_expectancy": verdict(
            strategy.get("expectancy_per_trade_pct"), naive_router.get("expectancy_per_trade_pct")),
        "router_vs_naive_sharpe": verdict(
            strategy.get("sharpe_ratio"), naive_router.get("sharpe_ratio")),
        "filter_vs_unfiltered_expectancy": verdict(
            strategy.get("expectancy_per_trade_pct"), unfiltered.get("expectancy_per_trade_pct")),
        "filter_vs_unfiltered_sharpe": verdict(
            strategy.get("sharpe_ratio"), unfiltered.get("sharpe_ratio")),
    }


# ---------------------------------------------------------------------------
# JSON sanitization & markdown rendering (§5.4)
# ---------------------------------------------------------------------------


def _sanitize(obj):
    """Recursively convert numpy scalars and infinities to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _fmt(v, suffix="", nd=2):
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return f"{v:{'.%df' % nd}}{suffix}"


def render_markdown_report(report: dict) -> str:
    """Human-readable summary of the structured report dict (§5.4)."""
    L = []
    L.append("# Full-Stack Walk-Forward Backtest Report")
    L.append("")
    L.append(f"Generated: {report['generated_at']}")
    L.append(f"Input: {report['input']['n_bars']} bars "
             f"({report['input']['start_ts']} → {report['input']['end_ts']}), "
             f"granularity={report['granularity']}")
    L.append(f"Windows: {len(report['windows'])} contiguous evaluation segments "
             f"of a single continuous replay (models fixed — rule-based fallback "
             f"until manual retrain post-rename)")
    L.append("")
    L.append("> Sharpe/Sortino are per-trade ratios (mean/std of net trade returns), "
             "not annualized.")
    L.append("")

    agg = report["aggregate"]
    L.append("## Aggregate Strategy Performance (full stack)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Total return | {_fmt(agg.get('total_return_pct'), '%')} |")
    L.append(f"| Trades | {agg.get('n_trades', 0)} |")
    L.append(f"| Win rate | {_fmt(agg.get('win_rate_pct'), '%', 1)} |")
    L.append(f"| Expectancy / trade | {_fmt(agg.get('expectancy_per_trade_pct'), '%', 4)} |")
    L.append(f"| Profit factor | {_fmt(agg.get('profit_factor'))} |")
    L.append(f"| Avg win / avg loss | {_fmt(agg.get('avg_win_pct'), '%', 3)} / {_fmt(agg.get('avg_loss_pct'), '%', 3)} |")
    L.append(f"| Sharpe (per-trade) | {_fmt(agg.get('sharpe_ratio'))} |")
    L.append(f"| Sortino (per-trade) | {_fmt(agg.get('sortino_ratio'))} |")
    L.append(f"| Max drawdown | {_fmt(agg.get('max_drawdown_pct'), '%')} |")
    L.append("")

    L.append("## Baseline Comparisons")
    L.append("")
    L.append("| Strategy | Trades | Total ret % | Win rate % | Expectancy % | Sharpe | Sortino | Max DD % |")
    L.append("|---|---|---|---|---|---|---|---|")
    rows = [("Full stack (main)", agg)]
    for name, key in [
        ("Buy & hold", "buy_and_hold"),
        ("Simple momentum", "simple_momentum"),
        ("Naive highest-confidence router", "naive_router"),
        ("No opportunity filter", "unfiltered"),
    ]:
        b = report["baselines"].get(key)
        if b:
            rows.append((name, b))
    for name, m in rows:
        L.append(f"| {name} | {m.get('n_trades', 0)} | {_fmt(m.get('total_return_pct'))} | "
                 f"{_fmt(m.get('win_rate_pct'), '', 1)} | {_fmt(m.get('expectancy_per_trade_pct'), '', 3)} | "
                 f"{_fmt(m.get('sharpe_ratio'))} | {_fmt(m.get('sortino_ratio'))} | "
                 f"{_fmt(m.get('max_drawdown_pct'))} |")
    L.append("")

    L.append("### Component rigor verdicts")
    L.append("")
    for k, v in report["baseline_verdicts"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## Project_Plan.md Success Criteria")
    L.append("")
    L.append("| Criterion | Target | Actual | Verdict |")
    L.append("|---|---|---|---|")
    for c in report["success_criteria"]:
        actual = c.get("actual")
        base = c.get("baseline_actual")
        pct_like = "pct" in c["criterion"] or "expectancy" in c["criterion"]
        fmt = (lambda v: _fmt(v, "%", 3)) if pct_like else (lambda v: _fmt(v))
        actual_s = fmt(actual) if isinstance(actual, float) else str(actual)
        if base is not None:
            actual_s += f" (momentum: {fmt(base) if isinstance(base, float) else base})"
        L.append(f"| {c['criterion']} | {c['target']} | {actual_s} | "
                 f"{'✅ PASS' if c['pass'] else '❌ FAIL'} |")
    L.append("")

    L.append("## Per-Window Breakdown")
    L.append("")
    L.append("| Window | Bars | Trades | Return % | Win rate % | Sharpe | Max DD % |")
    L.append("|---|---|---|---|---|---|---|")
    for w in report["windows"]:
        m = w["metrics"]
        L.append(f"| {w['index']} ({w['start_ts']}–{w['end_ts']}) | {w['n_bars']} | "
                 f"{m.get('n_trades', 0)} | {_fmt(m.get('total_return_pct'))} | "
                 f"{_fmt(m.get('win_rate_pct'), '', 1)} | {_fmt(m.get('sharpe_ratio'))} | "
                 f"{_fmt(m.get('max_drawdown_pct'))} |")
    L.append("")

    bd = report["aggregate_breakdowns"]
    for title, key in [("By Expert", "by_expert"), ("By Regime", "by_regime")]:
        L.append(f"### {title}")
        L.append("")
        L.append("| Group | Trades | Win rate % | Expectancy % | PnL (USDT) |")
        L.append("|---|---|---|---|---|")
        for g, s in (bd.get(key) or {}).items():
            L.append(f"| {g} | {s['n_trades']} | {_fmt(s['win_rate_pct'], '', 1)} | "
                     f"{_fmt(s['expectancy_per_trade_pct'], '', 3)} | {_fmt(s['total_pnl_usdt'])} |")
        L.append("")

    L.append("## Confidence Calibration (reliability data)")
    L.append("")
    cal = report["calibration"]
    for expert_name, buckets in cal.get("experts", {}).items():
        score = cal.get("expert_miscalibration_scores", {}).get(expert_name)
        L.append(f"### {expert_name} (weighted |pred−realized| = {_fmt(score, '', 3)})")
        L.append("")
        L.append("| Conf bucket | n | Avg predicted | Realized win % |")
        L.append("|---|---|---|---|")
        for b in buckets:
            if b["n"] == 0:
                continue
            L.append(f"| {b['bucket']} | {b['n']} | "
                     f"{_fmt(b['avg_predicted_confidence'], '', 3)} | "
                     f"{_fmt(b['realized_win_rate_pct'], '', 1)} |")
        L.append("")
    rb = cal.get("router") or []
    rs = cal.get("router_miscalibration_score")
    L.append(f"### Router (executed trades; weighted |pred−realized| = {_fmt(rs, '', 3)})")
    L.append("")
    L.append("| Conf bucket | n | Avg predicted | Realized win % |")
    L.append("|---|---|---|---|")
    for b in rb:
        if b["n"] == 0:
            continue
        L.append(f"| {b['bucket']} | {b['n']} | "
                 f"{_fmt(b['avg_predicted_confidence'], '', 3)} | "
                 f"{_fmt(b['realized_win_rate_pct'], '', 1)} |")
    L.append("")

    passed = sum(1 for c in report["success_criteria"] if c["pass"])
    L.append("---")
    L.append(f"**Success criteria: {passed}/{len(report['success_criteria'])} met.** "
             f"All numbers above are realized, net of fees/slippage, through the "
             f"complete stack including risk-layer sizing and execution simulation.")
    L.append("")
    return "\n".join(L)
