"""
Training-as-deployed: the harness-aligned labeler, population selection,
the dataset contract, the metrics and the end-to-end model contract.

No real model is trained: the one test that exercises the trainer injects a
constant stub model.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import training_common as tc
from src.backtest.harness_harvest import HarvestStats, apply_population
from src.evaluation.trade_duration import (
    alignment_passed,
    alignment_report,
    duration_report,
    hold_stats,
    lockout_cost,
    pinned_rate,
)
from src.execution.execution_engine import entry_fill_price, total_slippage_pct
from src.experts.trend_expert import FEATURE_NAMES as TREND_FEATURES
from src.experts.trend_expert import TrendExpert
from src.labeling.harness_labels import (
    LabelContext,
    entry_price_for,
    label_candidate,
    select_sequential,
)
from src.labeling.hold_policy import POPULATION_EXPERT_GATE, HoldPolicy
from src.labeling.triple_barrier import barrier_volatility_pct, barriers_for_regime
from src.state.contracts import MarketSnapshot, RegimeType
from tests.unit.test_phase3 import make_feature_vector, make_regime_state

pytestmark = [pytest.mark.unit]


def bar(ts, o, h, low, c, spread=0.01):
    return MarketSnapshot(
        timestamp=ts, symbol="BTC/USDT", open=o, high=h, low=low, close=c, volume=10.0,
        bid=c * 0.9995, ask=c * 1.0005, bid_volume=50.0, ask_volume=50.0, spread_pct=spread,
        hour_utc=12, day_of_week=2,
    )


def config(granularity="10s", max_bars=5, latency_ms=0, min_seconds=10):
    return {
        "system": {"data_granularity": granularity},
        "labeling": {"max_holding_bars": max_bars, "fee_pct": 0.04, "k_tp": 1.5, "k_sl": 1.0,
                     "regime_specific_barriers": {"CHOP_RANGE": {"k_tp": 1.0, "k_sl": 0.5}}},
        "execution": {"slippage_pct": 0.01, "simulated_latency_ms": latency_ms, "min_bars_between_orders": 2},
        "trade_duration": {"min_trade_seconds": min_seconds},
    }


def state(close=100.0, spread=0.01, regime=RegimeType.TRENDING_UP, vol=0.2):
    return SimpleNamespace(
        timestamp=1_000, market_snapshot=bar(1_000, close, close, close, close, spread),
        regime_state=make_regime_state(regime), feature_vector=make_feature_vector(volatility_medium=vol),
    )


# ---------------------------------------------------------------------------
# the labeler pays what the harness pays
# ---------------------------------------------------------------------------

def test_the_entry_price_is_the_simulated_fill_not_the_close():
    ctx = LabelContext.from_config(config())
    snap = bar(1, 100, 100, 100, 100.0, spread=0.02)
    long_px = entry_price_for(ctx, snap, 1)
    short_px = entry_price_for(ctx, snap, -1)
    assert long_px == pytest.approx(entry_fill_price("BUY", 100.0, 0.02, 0.01))
    assert short_px == pytest.approx(entry_fill_price("SELL", 100.0, 0.02, 0.01))
    assert long_px > 100.0 > short_px


def test_the_slippage_includes_the_simulated_latency_like_the_paper_backend():
    cfg = config(latency_ms=500)
    cfg["execution"]["latency_adverse_pct_per_100ms"] = 0.002
    assert total_slippage_pct(cfg["execution"]) == pytest.approx(0.01 + 5 * 0.002)
    assert LabelContext.from_config(cfg).slippage_pct == pytest.approx(0.02)


def test_the_context_carries_the_policy_the_fees_and_the_cooldown():
    ctx = LabelContext.from_config(config("5s", max_bars=6))
    assert ctx.min_hold_bars == 2 and ctx.fee_pct == 0.04 and ctx.min_bars_between_orders == 2


def test_a_label_uses_the_shared_barrier_sizing():
    cfg = config()
    ctx = LabelContext.from_config(cfg)
    fwd = [bar(1_010 + 10 * i, 100, 100.0, 100.0, 100.0) for i in range(5)]
    for regime in (RegimeType.TRENDING_UP, RegimeType.CHOP_RANGE):
        st = state(regime=regime)
        row = label_candidate(ctx, st, fwd, 1)
        tp, sl = barriers_for_regime(cfg, regime.name, barrier_volatility_pct(st.feature_vector))
        assert (row["take_profit_pct"], row["stop_loss_pct"]) == (tp, sl)
        assert row["regime"] == regime.name


def test_a_label_is_the_harness_outcome_entry_barriers_exit_and_fees():
    ctx = LabelContext.from_config(config())
    st = state()
    entry = entry_price_for(ctx, st.market_snapshot, 1)
    tp, _ = barriers_for_regime(ctx.config, "TRENDING_UP", barrier_volatility_pct(st.feature_vector))
    tp_level = entry * (1 + tp / 100)
    fwd = [bar(1_010, 100, tp_level + 1, 99.9, tp_level), bar(1_020, 100, 100, 100, 100)] + \
          [bar(1_030 + 10 * i, 100, 100, 100, 100) for i in range(3)]
    row = label_candidate(ctx, st, fwd, 1)
    assert row["exit_reason"] == "take_profit" and row["bars_held"] == 1
    assert row["realized_return_pct"] == pytest.approx(tp - 2 * 0.04)
    assert row["label"] == 1 and row["hold_seconds"] == 10.0


def test_the_lockout_changes_the_label_not_just_a_flag():
    cfg = config("5s", max_bars=6)                       # 10 s minimum = 2 bars of 5 s
    ctx = LabelContext.from_config(cfg)
    st = state()
    entry = entry_price_for(ctx, st.market_snapshot, 1)
    # a target-hit on bar 1, then a stop-hit on bar 2:
    tp, sl = barriers_for_regime(cfg, "TRENDING_UP", barrier_volatility_pct(st.feature_vector))
    fwd = [bar(1_005, entry, entry * (1 + tp / 100 + 0.01), entry, entry),
           bar(1_010, entry, entry, entry * (1 - sl / 100 - 0.01), entry * (1 - sl / 100 - 0.005))] + \
          [bar(1_015 + 5 * i, entry, entry, entry, entry) for i in range(4)]
    locked = label_candidate(ctx, st, fwd, 1)
    free = label_candidate(ctx, st, fwd, 1, min_hold_bars=1)
    assert free["exit_reason"] == "take_profit" and free["bars_held"] == 1       # a 5 s trade
    assert locked["exit_reason"] == "stop_loss" and locked["bars_held"] == 2     # the rule forbids it
    assert locked["hold_seconds"] == 10.0 and free["hold_seconds"] == 5.0
    assert locked["pinned_to_min_hold"] and not free["pinned_to_min_hold"]


def test_a_candidate_with_no_legal_outcome_in_the_data_is_dropped():
    cfg = config("5s", max_bars=6)
    ctx = LabelContext.from_config(cfg)
    st = state()
    assert label_candidate(ctx, st, [], 1) is None
    one_bar = [bar(1_005, 100, 100, 100, 100)]                     # data ends inside the 2-bar lockout
    assert label_candidate(ctx, st, one_bar, 1) is None
    short_tail = [bar(1_005 + 5 * i, 100, 100, 100, 100) for i in range(3)]   # unresolved timeout cut off
    assert label_candidate(ctx, st, short_tail, 1) is None
    full = [bar(1_005 + 5 * i, 100, 100, 100, 100) for i in range(6)]
    assert label_candidate(ctx, st, full, 1)["exit_reason"] == "timeout"


@pytest.mark.parametrize("granularity", ["5s", "10s", "30s", "1m"])
def test_no_label_is_shorter_than_ten_seconds_at_any_granularity(granularity):
    cfg = config(granularity, max_bars=40, latency_ms=2000)
    ctx = LabelContext.from_config(cfg)
    step = int(ctx.policy.granularity_seconds)
    rng = np.random.default_rng(3)
    fwd, price = [], 100.0
    for i in range(40):
        o, c = price, price * (1 + rng.normal(0, 0.003))
        fwd.append(bar(1_000 + step * (i + 1), o, max(o, c) * 1.001, min(o, c) * 0.999, c))
        price = c
    for direction in (1, -1):
        row = label_candidate(ctx, state(), fwd, direction)
        if row is not None:
            assert row["hold_seconds"] >= 10.0


# ---------------------------------------------------------------------------
# population: one position at a time
# ---------------------------------------------------------------------------

def rows(*specs):
    return [{"expert_name": n, "bar_index": i, "bars_held": h} for n, i, h in specs]


def test_sequential_keeps_one_position_at_a_time_per_expert():
    data = rows(*[("a", i, 3) for i in (0, 1, 2, 3, 4, 5, 6, 10)])
    keep = select_sequential(data, min_bars_between_orders=2)
    assert [r["bar_index"] for r, k in zip(data, keep, strict=True) if k] == [0, 3, 6, 10]


def test_an_exit_bar_can_host_the_next_entry():
    data = rows(("a", 0, 3), ("a", 3, 3))
    assert select_sequential(data, 1) == [True, True]


def test_the_order_cooldown_binds_when_trades_are_short():
    data = rows(*[("a", i, 1) for i in range(5)])
    keep = select_sequential(data, min_bars_between_orders=2)
    assert [r["bar_index"] for r, k in zip(data, keep, strict=True) if k] == [0, 2, 4]


def test_experts_are_sequenced_independently_and_input_order_does_not_matter():
    data = rows(("b", 2, 3), ("a", 1, 3), ("a", 0, 3), ("b", 0, 3), ("a", 2, 3))
    keep = select_sequential(data, 2)
    kept = sorted((r["expert_name"], r["bar_index"]) for r, k in zip(data, keep, strict=True) if k)
    assert kept == [("a", 0), ("b", 0)]


def test_apply_population_filters_then_samples():
    ctx = LabelContext.from_config(config())
    df = pd.DataFrame({
        "expert_name": "a", "bar_index": [0, 1, 2, 3, 4, 5], "bars_held": 2,
        "passed_filter": [True, True, False, True, True, True],
    })
    everything = apply_population(df, ctx, require_filter_pass=False, sample_policy="all")
    assert len(everything) == 6
    filtered = apply_population(df, ctx, require_filter_pass=True, sample_policy="all")
    assert filtered["bar_index"].tolist() == [0, 1, 3, 4, 5]
    seq = apply_population(df, ctx, require_filter_pass=True, sample_policy="sequential")
    assert seq["bar_index"].tolist() == [0, 3, 5]
    with pytest.raises(ValueError):
        apply_population(df, ctx, sample_policy="whatever")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_hold_stats_counts_violations_and_ignores_nan():
    s = hold_stats([10, 12, 5, float("nan"), None, 30], 10)
    assert s["n"] == 4 and s["violations"] == 1 and s["min_s"] == 5 and s["max_s"] == 30
    empty = hold_stats([], 10)
    assert empty["n"] == 0 and empty["min_s"] is None
    assert hold_stats([1, 2], 0)["violations"] == 0                  # rule disabled


def trade(expert="trend_expert", hold=20.0, held_bars=2, min_bars=2, reason="stop_loss", ts=1000, pnl=0.1):
    return {"expert": expert, "hold_seconds": hold, "bars_held": held_bars, "min_hold_bars": min_bars,
            "exit_reason": reason, "entry_timestamp": ts, "pnl_pct": pnl}


def test_the_duration_report_passes_only_with_zero_violations():
    p = HoldPolicy(10.0, "10s", 10.0, 5, 0.0)
    good = duration_report([trade(), trade("breakout_expert", hold=30.0, held_bars=3)], p)
    assert good["passed"] and good["overall"]["violations"] == 0
    assert set(good["by_expert"]) == {"trend_expert", "breakout_expert"}
    bad = duration_report([trade(), trade(hold=4.0, held_bars=1, min_bars=1)], p)
    assert not bad["passed"] and bad["overall"]["violations"] == 1
    assert duration_report([], p)["passed"]


def test_pinned_rate_is_the_share_of_barrier_exits_on_the_first_legal_bar():
    data = [trade(held_bars=2), trade(held_bars=2), trade(held_bars=4), trade(reason="timeout", held_bars=5)]
    assert pinned_rate(data) == pytest.approx(2 / 3)
    assert pinned_rate([trade(min_bars=1)]) is None                  # no lockout was active
    assert pinned_rate([]) is None


def test_the_lockout_cost_is_free_minus_locked_expectancy():
    cost = lockout_cost([0.1, -0.2, 0.3], [0.5, -0.2, 0.3])
    assert cost["n"] == 3 and cost["changed_share"] == pytest.approx(1 / 3)
    assert cost["lockout_cost_pct"] == pytest.approx((0.5 - 0.2 + 0.3) / 3 - (0.1 - 0.2 + 0.3) / 3)
    with pytest.raises(ValueError):
        lockout_cost([1], [1, 2])
    assert lockout_cost([], [])["n"] == 0


def labels(*specs):
    return pd.DataFrame([{"expert_name": e, "timestamp": t, "exit_reason": r, "bars_held": b,
                          "realized_return_pct": p} for e, t, r, b, p in specs])


def test_alignment_is_perfect_when_labels_reproduce_every_trade():
    lab = labels(("trend_expert", 1000, "stop_loss", 2, 0.1), ("trend_expert", 2000, "take_profit", 3, 0.5))
    trades = [trade(ts=1000, pnl=0.1), trade(ts=2000, reason="take_profit", held_bars=3, pnl=0.5)]
    rep = alignment_report(lab, trades)
    assert rep["coverage"] == 1.0 and rep["exit_reason_agreement"] == 1.0 and rep["bars_held_agreement"] == 1.0
    assert rep["pnl_max_abs_error_pct"] == pytest.approx(0.0) and not rep["disagreements"]
    assert alignment_passed(rep)


def test_alignment_reports_disagreements_and_missing_decisions():
    lab = labels(("trend_expert", 1000, "take_profit", 2, 0.1))
    trades = [trade(ts=1000, reason="stop_loss"), trade(ts=9999)]
    rep = alignment_report(lab, trades)
    assert rep["coverage"] == 0.5 and rep["exit_reason_agreement"] == 0.0
    assert rep["unmatched"] == [{"expert": "trend_expert", "entry_timestamp": 9999}]
    assert rep["disagreements"][0]["label"]["exit_reason"] == "take_profit"
    assert not alignment_passed(rep)


def test_alignment_with_no_trades_is_vacuously_fine_and_a_pnl_mismatch_fails():
    assert alignment_passed(alignment_report(labels(), []))
    lab = labels(("trend_expert", 1000, "stop_loss", 2, 0.1))
    rep = alignment_report(lab, [trade(ts=1000, pnl=0.2)])
    assert rep["pnl_mae_pct"] == pytest.approx(0.1) and rep["disagreements"]


# ---------------------------------------------------------------------------
# the dataset contract
# ---------------------------------------------------------------------------

def dataset_frame(n=60, hold=20.0):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=n) for name in TREND_FEATURES})
    df["expert_name"] = "trend_expert"
    df["bar_index"] = np.arange(n) * 10
    df["bars_held"] = 3
    df["hold_seconds"] = hold
    df["label"] = (np.arange(n) % 2).astype(int)
    df["realized_return_pct"] = np.where(df["label"] == 1, 0.2, -0.2)
    df["max_adverse_excursion_pct"] = 0.1
    df["pinned_to_min_hold"] = False
    return df


def stats_for(df):
    st = HarvestStats()
    for _ in range(len(df) * 3):
        st.bump("trend_expert", "n_candidates")
    for _ in range(len(df)):
        st.bump("trend_expert", "n_sampled")
    return st


def test_the_dataset_sidecar_round_trips_and_matches_its_config(tmp_path):
    cfg = config()
    policy = HoldPolicy.from_config(cfg)
    df = dataset_frame()
    path = str(tmp_path / "labels.parquet")
    meta_path = tc.write_dataset_meta(
        path, policy=policy, population=POPULATION_EXPERT_GATE, sample_policy="sequential",
        require_filter_pass=True, stats=stats_for(df), df=df)
    assert meta_path.exists()
    meta = tc.read_dataset_meta(path)
    assert meta["hold_policy"] == policy.fingerprint()
    assert meta["per_expert"]["trend_expert"]["n_candidates"] == 180
    assert meta["per_expert"]["trend_expert"]["hold_stats"]["violations"] == 0
    assert tc.check_dataset_matches_config(meta, cfg) == policy


def test_a_dataset_without_a_sidecar_is_refused():
    with pytest.raises(SystemExit, match="meta.json"):
        tc.check_dataset_matches_config(None, config())


def test_a_dataset_labeled_under_another_policy_is_refused(tmp_path):
    df = dataset_frame()
    old = HoldPolicy.from_config(config(min_seconds=3))
    path = str(tmp_path / "labels.parquet")
    tc.write_dataset_meta(path, policy=old, population=POPULATION_EXPERT_GATE, sample_policy="sequential",
                          require_filter_pass=True, stats=stats_for(df), df=df)
    with pytest.raises(SystemExit, match="different hold policy"):
        tc.check_dataset_matches_config(tc.read_dataset_meta(path), config())


def test_an_unsatisfiable_policy_stops_a_training_run():
    with pytest.raises(SystemExit, match="not satisfiable"):
        tc.check_dataset_matches_config({}, config("1s", max_bars=5))


def test_training_refuses_a_label_shorter_than_the_minimum():
    p = HoldPolicy.from_config(config())
    tc.assert_no_short_trades(dataset_frame(hold=20.0), p, "trend")
    with pytest.raises(SystemExit, match="shorter than 10s"):
        tc.assert_no_short_trades(dataset_frame(hold=4.0), p, "trend")
    with pytest.raises(SystemExit, match="no hold_seconds"):
        tc.assert_no_short_trades(dataset_frame().drop(columns=["hold_seconds"]), p, "trend")


def test_the_walk_forward_embargo_must_cover_the_label_horizon():
    cfg = config(max_bars=8)
    cfg["backtest"] = {"embargo_bars": 3}
    p = HoldPolicy.from_config(cfg)
    with pytest.raises(SystemExit, match="label horizon"):
        tc.splitter_for(cfg, dataset_frame(), p)
    cfg["backtest"] = {"embargo_bars": 8, "train_bars": 5000, "test_bars": 500, "step_bars": 100}
    assert tc.splitter_for(cfg, dataset_frame(), p).label_horizon_bars == 8


def test_the_alignment_block_counts_candidates_samples_and_violations():
    cfg = config()
    p = HoldPolicy.from_config(cfg)
    df = dataset_frame(n=40)
    meta = {"population": POPULATION_EXPERT_GATE, "sample_policy": "sequential",
            "require_opportunity_filter_pass": True, "per_expert": {"trend_expert": {"n_candidates": 400}}}
    blk = tc.alignment_block_from_meta(meta, p, "trend_expert", df)
    assert blk["n_candidates"] == 400 and blk["n_samples"] == 40 and blk["min_duration_violations"] == 0
    assert blk["hold_policy"] == p.fingerprint() and blk["population"] == POPULATION_EXPERT_GATE


def test_the_bar_level_alignment_block_has_its_own_sampling_label():
    p = HoldPolicy.from_config(config())
    blk = tc.bar_level_alignment(p, dataset_frame(), population="router_selected_bars",
                                 require_filter_pass=True, n_candidates=60)
    assert blk["sample_policy"] == "bars" and blk["population"] == "router_selected_bars"


# ---------------------------------------------------------------------------
# the contract, end to end (stub model: nothing is trained)
# ---------------------------------------------------------------------------

class ConstantModel:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def test_a_model_saved_by_the_trainer_is_served_by_the_expert_under_the_same_policy(tmp_path):
    pytest.importorskip("sklearn")
    cfg = config(max_bars=5)
    cfg["backtest"] = {"train_bars": 400, "test_bars": 100, "step_bars": 100, "embargo_bars": 5}
    policy = HoldPolicy.from_config(cfg)
    df = dataset_frame(n=120)
    meta = {"population": POPULATION_EXPERT_GATE, "sample_policy": "sequential",
            "require_opportunity_filter_pass": True, "per_expert": {"trend_expert": {"n_candidates": 500}}}
    alignment = tc.alignment_block_from_meta(meta, policy, "trend_expert", df)

    out = tmp_path / "trend_expert.pkl"
    tc.train_expert("trend_expert", TREND_FEATURES, df, cfg, policy, alignment, str(out),
                    make_model=ConstantModel)

    card = json.loads(out.with_suffix(".meta.json").read_text())
    assert card["training_alignment"]["hold_policy"] == policy.fingerprint()
    assert card["walk_forward_window_unit"] == "bars" and card["walk_forward_folds"]

    expert_cfg = {"momentum_threshold": 0.05, "regime_requirement": ["TRENDING_UP"],
                  "regime_confidence_floor": 0.4, "persistence_requirement": 0.5,
                  "fallback_confidence": 0.6, "fallback_expected_return": 0.08, "fallback_expected_risk": 0.12}
    served = TrendExpert(expert_cfg, str(out), "10s", policy, True)
    assert served._model is not None

    other = HoldPolicy.from_config(config(min_seconds=3))
    refused = TrendExpert(expert_cfg, str(out), "10s", other, True)
    assert refused._model is None


def test_the_trainer_refuses_a_dataset_that_breaks_the_rule(tmp_path):
    cfg = config()
    policy = HoldPolicy.from_config(cfg)
    with pytest.raises(SystemExit, match="shorter than"):
        tc.train_expert("trend_expert", TREND_FEATURES, dataset_frame(hold=3.0), cfg, policy, {},
                        str(tmp_path / "m.pkl"), make_model=ConstantModel)
