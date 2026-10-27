"""
Metrics and evals for the minimum-trade-duration rule and for how closely
training labels reproduce what the harness actually does.

Pure functions over plain lists / dicts / DataFrames -- no pipeline, no
I/O -- so the same code scores a harness run, a labeled training set and a
unit-test fixture.

Metric glossary (every number the reports print):

``violations``
    trades whose fill-to-exit time is below ``min_trade_seconds``. MUST be 0.
``pinned_rate``
    share of non-timeout exits that happened on the FIRST bar the lockout
    allowed. High => the barriers sit inside the lockout window, i.e. the
    rule is what ends those trades, not the market.
``lockout_cost``
    expectancy with the lockout minus expectancy without it, on the same
    candidates: what the 10 s rule costs (or earns) in net-of-fee percent.
``label_harness_agreement``
    for every trade the harness took, does the training label for the same
    (expert, entry bar) have the same exit reason / bars held / net PnL?
"""
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..labeling.hold_policy import HoldPolicy


def _clean(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                     dtype="float64")
    return arr


def hold_stats(hold_seconds: Iterable[float], min_trade_seconds: float) -> Dict[str, Any]:
    """Distribution of fill-to-exit times and the number below the minimum."""
    arr = _clean(hold_seconds)
    if arr.size == 0:
        return {"n": 0, "violations": 0, "min_s": None, "p05_s": None, "median_s": None,
                "p95_s": None, "max_s": None, "min_trade_seconds": float(min_trade_seconds)}
    violations = int((arr < float(min_trade_seconds) - 1e-3).sum()) if min_trade_seconds > 0 else 0
    return {
        "n": int(arr.size), "violations": violations,
        "min_s": float(arr.min()), "p05_s": float(np.percentile(arr, 5)),
        "median_s": float(np.median(arr)), "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(arr.max()), "min_trade_seconds": float(min_trade_seconds),
    }


def _frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(list(rows))


def pinned_rate(rows: Any) -> Optional[float]:
    """Share of non-timeout exits on the first lockout-eligible bar.

    Needs ``bars_held``, ``min_hold_bars`` and ``exit_reason`` columns
    (harness trades carry all three). ``None`` when no lockout was active
    or there were no barrier exits.
    """
    df = _frame(rows)
    if df.empty or not {"bars_held", "min_hold_bars", "exit_reason"} <= set(df.columns):
        return None
    locked = df[(df["min_hold_bars"] > 1) & (df["exit_reason"] != "timeout")]
    if locked.empty:
        return None
    return float((locked["bars_held"] == locked["min_hold_bars"]).mean())


def duration_report(trades: Any, policy: HoldPolicy) -> Dict[str, Any]:
    """The ``trade_duration`` section of a harness report.

    ``trades`` are harness trade dicts (``hold_seconds``, ``expert``,
    ``bars_held``, ``min_hold_bars``, ``exit_reason``).
    """
    df = _frame(trades)
    minimum = policy.min_trade_seconds
    report: Dict[str, Any] = {
        "policy": policy.fingerprint(),
        "shortest_legal_hold_seconds": policy.shortest_legal_hold_seconds,
        "overall": hold_stats(df["hold_seconds"] if "hold_seconds" in df else [], minimum),
        "pinned_rate": pinned_rate(df),
        "by_expert": {},
        "by_exit_reason": {},
    }
    report["passed"] = report["overall"]["violations"] == 0
    if df.empty or "hold_seconds" not in df.columns:
        return report
    if "expert" in df.columns:
        for name, group in df.groupby("expert", dropna=True):
            report["by_expert"][str(name)] = {
                **hold_stats(group["hold_seconds"], minimum), "pinned_rate": pinned_rate(group),
            }
    if "exit_reason" in df.columns:
        for reason, group in df.groupby("exit_reason", dropna=True):
            report["by_exit_reason"][str(reason)] = hold_stats(group["hold_seconds"], minimum)
    return report


def lockout_cost(locked_returns: Sequence[float], free_returns: Sequence[float]) -> Dict[str, Any]:
    """What the minimum hold costs on the SAME candidates.

    ``free_returns`` are the net-of-fee returns the labeler produces with no
    lockout; ``locked_returns`` with it. Aligned element-for-element.
    """
    locked = np.asarray(locked_returns, dtype="float64")
    free = np.asarray(free_returns, dtype="float64")
    if locked.shape != free.shape:
        raise ValueError("locked and free outcomes must be aligned candidate-for-candidate")
    if locked.size == 0:
        return {"n": 0, "changed_share": None, "expectancy_locked_pct": None,
                "expectancy_free_pct": None, "lockout_cost_pct": None, "win_rate_locked": None,
                "win_rate_free": None}
    return {
        "n": int(locked.size),
        "changed_share": float((~np.isclose(locked, free)).mean()),
        "expectancy_locked_pct": float(locked.mean()),
        "expectancy_free_pct": float(free.mean()),
        "lockout_cost_pct": float(free.mean() - locked.mean()),
        "win_rate_locked": float((locked > 0).mean()),
        "win_rate_free": float((free > 0).mean()),
    }


def alignment_report(
    labels: pd.DataFrame, trades: Any, *, pnl_tolerance_pct: float = 1e-6,
) -> Dict[str, Any]:
    """How faithfully the training labels reproduce the harness's trades.

    ``labels`` has one row per labeled candidate (``expert_name``,
    ``timestamp``, ``exit_reason``, ``bars_held``, ``realized_return_pct``).
    ``trades`` are harness trades (``expert``, ``entry_timestamp``,
    ``exit_reason``, ``bars_held``, ``pnl_pct``). A harness trade whose
    (expert, entry bar) has no label is a COVERAGE miss: the training
    population did not contain a decision the harness actually made.
    """
    trades_df = _frame(trades)
    result: Dict[str, Any] = {
        "n_trades": int(len(trades_df)), "n_labeled": int(len(labels)),
        "matched": 0, "coverage": None, "exit_reason_agreement": None,
        "bars_held_agreement": None, "pnl_mae_pct": None, "pnl_max_abs_error_pct": None,
        "unmatched": [], "disagreements": [],
    }
    if trades_df.empty:
        return result

    index = {(str(r.expert_name), int(r.timestamp)): r for r in labels.itertuples()} if len(labels) else {}
    reason_ok = bars_ok = 0
    pnl_errors: List[float] = []
    for trade in trades_df.to_dict("records"):
        key = (str(trade["expert"]), int(trade["entry_timestamp"]))
        label = index.get(key)
        if label is None:
            result["unmatched"].append({"expert": key[0], "entry_timestamp": key[1]})
            continue
        result["matched"] += 1
        same_reason = str(label.exit_reason) == str(trade["exit_reason"])
        same_bars = int(label.bars_held) == int(trade["bars_held"])
        reason_ok += same_reason
        bars_ok += same_bars
        error = abs(float(label.realized_return_pct) - float(trade["pnl_pct"]))
        pnl_errors.append(error)
        if not (same_reason and same_bars and error <= pnl_tolerance_pct):
            result["disagreements"].append({
                "expert": key[0], "entry_timestamp": key[1],
                "label": {"exit_reason": str(label.exit_reason), "bars_held": int(label.bars_held),
                          "return_pct": float(label.realized_return_pct)},
                "harness": {"exit_reason": str(trade["exit_reason"]), "bars_held": int(trade["bars_held"]),
                            "return_pct": float(trade["pnl_pct"])},
            })

    matched = result["matched"]
    result["coverage"] = matched / len(trades_df)
    if matched:
        result["exit_reason_agreement"] = reason_ok / matched
        result["bars_held_agreement"] = bars_ok / matched
        result["pnl_mae_pct"] = float(np.mean(pnl_errors))
        result["pnl_max_abs_error_pct"] = float(np.max(pnl_errors))
    return result


def alignment_passed(report: Mapping[str, Any], *, min_coverage: float = 0.99, min_agreement: float = 0.99) -> bool:
    """The pass/fail the eval script and the retrain gate use."""
    if report.get("n_trades", 0) == 0:
        return True          # nothing traded: nothing to contradict
    if (report.get("coverage") or 0.0) < min_coverage:
        return False
    return (report.get("exit_reason_agreement") or 0.0) >= min_agreement and \
           (report.get("bars_held_agreement") or 0.0) >= min_agreement
