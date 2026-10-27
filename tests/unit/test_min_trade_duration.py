"""
The minimum-trade-duration rule (default 10 s) and the training contract.

Everything here is pure: no pipeline, no model is trained. The harness-level
guarantees live in tests/integration/test_min_hold_harness.py.
"""
import copy
import pickle

import numpy as np
import pytest

from src.backtest.walk_forward import WalkForwardSplitter
from src.config.consistency_validator import ConfigConsistencyError, validate_hold_policy
from src.execution.position_lifecycle import OpenPosition, check_position_exit
from src.experts.breakout_expert import BreakoutExpert
from src.experts.expert_registry import build_experts
from src.experts.mean_reversion_expert import MeanReversionExpert
from src.experts.trend_expert import TrendExpert
from src.labeling.hold_policy import (
    POPULATION_EXPERT_GATE,
    HoldPolicy,
    HoldPolicyError,
    alignment_settings,
    model_alignment_problem,
    parse_granularity_seconds,
    registry_alignment_status,
    training_alignment_block,
)
from src.labeling.triple_barrier import (
    barrier_exit_step,
    barriers_for_regime,
    compute_barrier_pcts,
    label_triple_barrier,
)
from src.layers.opportunity_filter import OpportunityFilter
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.layers.router import Router
from src.state.contracts import MarketSnapshot

pytestmark = [pytest.mark.unit]


def bar(ts, o, h, low, c):
    return MarketSnapshot(
        timestamp=ts, symbol="BTC/USDT", open=o, high=h, low=low, close=c, volume=10.0,
        bid=c * 0.9995, ask=c * 1.0005, bid_volume=50.0, ask_volume=50.0, spread_pct=0.01,
        hour_utc=12, day_of_week=2,
    )


def policy(granularity="10s", min_seconds=10.0, max_bars=5, latency=0.0):
    return HoldPolicy(min_seconds, granularity, parse_granularity_seconds(granularity), max_bars, latency)


# ---------------------------------------------------------------------------
# HoldPolicy arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("granularity,expected", [("1s", 1), ("5s", 5), ("10s", 10), ("30s", 30),
                                                  ("1m", 60), ("5m", 300), ("1.5m", 90), ("1h", 3600)])
def test_granularity_parses_to_seconds(granularity, expected):
    assert parse_granularity_seconds(granularity) == expected


@pytest.mark.parametrize("bad", ["", "abc", "10", "m", "-5s", "0s"])
def test_a_bad_granularity_is_refused(bad):
    with pytest.raises(HoldPolicyError):
        parse_granularity_seconds(bad)


@pytest.mark.parametrize("granularity,latency,expected_bars", [
    ("1s", 0, 10), ("5s", 0, 2), ("10s", 0, 1), ("30s", 0, 1), ("1m", 0, 1), ("5m", 0, 1),
    ("10s", 2, 2),      # 2 s latency eats into the first 10 s bar: it lasts 8 s -> not allowed
    ("10s", 10, 2), ("10s", 10.5, 3), ("5s", 1, 3), ("1m", 30, 1),
])
def test_min_hold_bars_covers_the_minimum_plus_latency(granularity, latency, expected_bars):
    p = policy(granularity, latency=latency, max_bars=50)
    assert p.min_hold_bars == expected_bars
    assert p.held_seconds(p.min_hold_bars) >= 10 - 1e-9          # the rule holds at the boundary
    if p.min_hold_bars > 1:
        assert p.held_seconds(p.min_hold_bars - 1) < 10           # and ONE bar fewer would break it


def test_a_zero_minimum_disables_the_lockout():
    p = policy("1s", min_seconds=0.0)
    assert p.min_hold_bars == 1 and not p.is_violation(0.5)


def test_is_violation_is_strict_below_the_minimum():
    p = policy()
    assert p.is_violation(9.9) and not p.is_violation(10.0) and not p.is_violation(12.0)


def test_held_seconds_subtracts_the_entry_latency():
    p = policy("10s", latency=2.0)
    assert p.held_seconds(1) == 8.0 and p.held_seconds(2) == 18.0


def test_the_requirement_is_the_default_when_the_config_has_no_section():
    p = HoldPolicy.from_config({"system": {"data_granularity": "5s"}, "labeling": {"max_holding_bars": 6}})
    assert p.min_trade_seconds == 10.0 and p.min_hold_bars == 2


def test_latency_defaults_to_the_paper_simulators():
    cfg = {"system": {"data_granularity": "10s"}, "labeling": {"max_holding_bars": 5},
           "execution": {"simulated_latency_ms": 2000}}
    assert HoldPolicy.from_config(cfg).entry_latency_seconds == 2.0
    cfg["trade_duration"] = {"assumed_entry_latency_seconds": 0.5}
    assert HoldPolicy.from_config(cfg).entry_latency_seconds == 0.5


def test_a_policy_with_a_too_short_timeout_is_unsatisfiable():
    """5 bars of 1 s is a 5 s trade: every timeout would be illegal."""
    cfg = {"system": {"data_granularity": "1s"}, "labeling": {"max_holding_bars": 5}}
    with pytest.raises(HoldPolicyError, match="shorter than the 10s minimum"):
        HoldPolicy.from_config(cfg)
    assert HoldPolicy.from_config(cfg, validate=False).problems()


def test_latency_longer_than_the_whole_hold_is_unsatisfiable():
    with pytest.raises(HoldPolicyError, match="latency"):
        policy("10s", max_bars=1, latency=10.0).validate()


def test_with_latency_returns_a_new_policy():
    base = policy("10s")
    slow = base.with_latency(3.0)
    assert base.entry_latency_seconds == 0.0 and slow.entry_latency_seconds == 3.0
    assert slow.min_hold_bars == 2


def test_the_fingerprint_names_every_number_a_model_depends_on():
    assert set(policy().fingerprint()) == {
        "min_trade_seconds", "granularity", "max_holding_bars", "min_hold_bars", "entry_latency_seconds"}


# ---------------------------------------------------------------------------
# The one exit rule
# ---------------------------------------------------------------------------

ENTRY, TP, SL = 100.0, 2.0, 1.0          # TP level 102, SL level 99 for a long


def step(bars_held, b, min_hold=3, direction=1, max_hold=6):
    return barrier_exit_step(direction, ENTRY, TP, SL, b, bars_held, max_hold, min_hold)


def test_nothing_can_exit_inside_the_lockout():
    crash = bar(1, 99.5, 99.6, 90.0, 91.0)               # crosses the stop
    spike = bar(1, 100.5, 110.0, 100.4, 109.0)           # crosses the target
    for held in (1, 2):
        assert step(held, crash) is None
        assert step(held, spike) is None


def test_the_timeout_is_locked_out_too():
    assert step(2, bar(1, 100, 100.1, 99.9, 100.0), min_hold=3, max_hold=2) is None


def test_a_stop_on_the_first_legal_bar_fills_at_the_worse_of_stop_and_open():
    gap = bar(1, 95.0, 96.0, 90.0, 91.0)                  # opened far below the 99 stop
    assert step(3, gap) == ("stop_loss", 95.0)
    touch = bar(1, 99.5, 99.8, 98.0, 98.5)                # opened above the stop, traded through it
    assert step(3, touch) == ("stop_loss", 99.0)


def test_a_short_stop_on_the_first_legal_bar_mirrors_the_long():
    gap = bar(1, 105.0, 106.0, 104.0, 105.5)              # short: stop level is 101
    assert barrier_exit_step(-1, ENTRY, TP, SL, gap, 3, 6, 3) == ("stop_loss", 105.0)


def test_the_gap_fill_only_applies_on_the_first_legal_bar():
    gap = bar(1, 95.0, 96.0, 90.0, 91.0)
    assert step(4, gap) == ("stop_loss", 99.0)            # later bars: the stop level, as before


def test_a_target_is_never_improved_by_the_lockout():
    spike = bar(1, 105.0, 110.0, 104.0, 109.0)
    assert step(3, spike) == ("take_profit", 102.0)


def test_stop_wins_when_a_bar_touches_both_barriers():
    both = bar(1, 100.0, 103.0, 98.0, 100.0)
    assert step(3, both)[0] == "stop_loss"


def test_with_no_lockout_the_stop_fills_at_the_stop_level_even_through_a_gap():
    """min_hold_bars=1 is the original behaviour, unchanged."""
    gap = bar(1, 95.0, 96.0, 90.0, 91.0)
    assert step(1, gap, min_hold=1) == ("stop_loss", 99.0)


def _legacy_label(forward, entry, direction, tp_pct, sl_pct, max_hold, fee):
    """The pre-change labeler, verbatim in behaviour."""
    if direction == 1:
        tp, sl = entry * (1 + tp_pct / 100), entry * (1 - sl_pct / 100)
    else:
        tp, sl = entry * (1 - tp_pct / 100), entry * (1 + sl_pct / 100)
    bars = forward[:max_hold]
    for i, b in enumerate(bars):
        hit_tp = b.high >= tp if direction == 1 else b.low <= tp
        hit_sl = b.low <= sl if direction == 1 else b.high >= sl
        if hit_sl:
            exit_p, reason = sl, "stop_loss"
        elif hit_tp:
            exit_p, reason = tp, "take_profit"
        else:
            continue
        raw = (exit_p - entry) / entry * 100 if direction == 1 else (entry - exit_p) / entry * 100
        return reason, i + 1, raw - fee * 2
    last = bars[-1]
    raw = (last.close - entry) / entry * 100 if direction == 1 else (entry - last.close) / entry * 100
    return "timeout", len(bars), raw - fee * 2


def random_bars(rng, n, start=1_000, step_s=10):
    price, out = 100.0, []
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.004))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
        low = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
        out.append(bar(start + i * step_s, o, h, low, c))
        price = c
    return out


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("direction", [1, -1])
def test_with_no_lockout_the_labels_are_exactly_the_legacy_labels(seed, direction):
    rng = np.random.default_rng(seed)
    fwd = random_bars(rng, 6)
    outcome = label_triple_barrier(fwd, 100.0, direction, 0.6, 0.4, 6, 0.04, min_hold_bars=1)
    reason, held, ret = _legacy_label(fwd, 100.0, direction, 0.6, 0.4, 6, 0.04)
    assert (outcome.exit_reason, outcome.bars_held) == (reason, held)
    assert outcome.realized_return_pct == pytest.approx(ret)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("min_hold", [1, 2, 3, 5])
@pytest.mark.parametrize("direction", [1, -1])
def test_a_label_and_a_live_position_always_exit_identically(seed, min_hold, direction):
    """THE guarantee: the labeler and ``check_position_exit`` share one rule."""
    rng = np.random.default_rng(1000 + seed)
    fwd = random_bars(rng, 6)
    outcome = label_triple_barrier(fwd, 100.0, direction, 0.5, 0.35, 6, 0.04, min_hold_bars=min_hold)

    pos = OpenPosition(
        trade_id="t", entry_timestamp=fwd[0].timestamp - 10, entry_price=100.0, direction=direction,
        size_usdt=1000.0, take_profit_pct=0.5, stop_loss_pct=0.35, max_holding_bars=6,
        min_hold_bars=min_hold,
    )
    exit_info = None
    for b in fwd:
        exit_info = check_position_exit(pos, b, 0.04)
        if exit_info:
            break
    assert exit_info is not None
    assert exit_info["exit_reason"] == outcome.exit_reason
    assert exit_info["bars_held"] == outcome.bars_held >= min_hold
    assert exit_info["realized_pnl_pct"] == pytest.approx(outcome.realized_return_pct)


@pytest.mark.parametrize("seed", range(30))
def test_no_label_is_ever_shorter_than_the_minimum_hold(seed):
    rng = np.random.default_rng(seed)
    fwd = random_bars(rng, 8, step_s=5)
    p = policy("5s", max_bars=8)
    outcome = label_triple_barrier(fwd, 100.0, 1, 0.2, 0.2, 8, 0.04, min_hold_bars=p.min_hold_bars)
    assert p.held_seconds(outcome.bars_held) >= 10


def test_the_lockout_is_reported_on_the_label():
    fwd = [bar(1, 100, 100.1, 98.0, 98.5), bar(2, 98.5, 98.6, 97.0, 97.5), bar(3, 97.5, 98.0, 96.0, 96.5)]
    out = label_triple_barrier(fwd, 100.0, 1, 2.0, 1.0, 5, 0.04, min_hold_bars=3)
    assert out.exit_reason == "stop_loss" and out.bars_held == 3 and out.pinned_to_min_hold
    assert out.lockout_adverse_excursion_pct == pytest.approx(3.0)        # worst low inside the lockout
    free = label_triple_barrier(fwd, 100.0, 1, 2.0, 1.0, 5, 0.04, min_hold_bars=1)
    assert free.bars_held == 1 and not free.pinned_to_min_hold


def test_a_label_refuses_an_impossible_policy():
    fwd = random_bars(np.random.default_rng(0), 4)
    with pytest.raises(ValueError, match="illegal trade"):
        label_triple_barrier(fwd, 100.0, 1, 1.0, 1.0, max_holding_bars=2, fee_pct=0.04, min_hold_bars=3)
    with pytest.raises(ValueError):
        label_triple_barrier(fwd, 100.0, 1, 1.0, 1.0, 5, 0.04, min_hold_bars=0)


def test_barrier_sizing_has_one_implementation():
    cfg = {"labeling": {"k_tp": 1.5, "k_sl": 1.0,
                        "regime_specific_barriers": {"CHOP_RANGE": {"k_tp": 1.0, "k_sl": 0.5}}}}
    assert barriers_for_regime(cfg, "TRENDING_UP", 0.2) == compute_barrier_pcts(0.2, 1.5, 1.0)
    assert barriers_for_regime(cfg, "CHOP_RANGE", 0.2) == compute_barrier_pcts(0.2, 1.0, 0.5)
    assert barriers_for_regime({}, "ANY", 0.2) == compute_barrier_pcts(0.2, 1.5, 1.0)


# ---------------------------------------------------------------------------
# check_position_exit: the wall-clock rule
# ---------------------------------------------------------------------------

def locked_position(**kw):
    defaults = dict(
        trade_id="t", entry_timestamp=1000, entry_price=100.0, direction=1, size_usdt=1000.0,
        take_profit_pct=2.0, stop_loss_pct=1.0, max_holding_bars=6,
        min_hold_bars=1, min_hold_seconds=10.0, entry_latency_seconds=2.0,
    )
    defaults.update(kw)
    return OpenPosition(**defaults)


def test_a_one_bar_trade_with_latency_is_held_until_the_wall_clock_allows():
    """10 s bars, 2 s latency: bar 1 is only 8 s after the fill -> no exit."""
    pos = locked_position()
    stopped = bar(1010, 99.5, 99.6, 90.0, 91.0)
    assert check_position_exit(pos, stopped, 0.04) is None
    later = bar(1020, 91.0, 92.0, 89.0, 90.0)
    info = check_position_exit(pos, later, 0.04)
    assert info is not None and info["exit_reason"] == "stop_loss"
    assert info["hold_seconds"] == pytest.approx(18.0) and info["bars_held"] == 2


def test_a_duplicated_bar_cannot_shorten_a_trade():
    pos = locked_position(entry_latency_seconds=0.0, min_hold_seconds=10.0)
    crash = bar(1010, 99.5, 99.6, 90.0, 91.0)
    assert check_position_exit(pos, crash, 0.04) is not None      # 10 s after entry: legal
    pos2 = locked_position(entry_latency_seconds=0.0)
    same_time = bar(1000, 99.5, 99.6, 90.0, 91.0)                 # same stamp as the entry
    assert check_position_exit(pos2, same_time, 0.04) is None


def test_a_position_with_no_policy_behaves_exactly_as_before():
    pos = OpenPosition(trade_id="t", entry_timestamp=1000, entry_price=100.0, direction=1, size_usdt=1000.0,
                       take_profit_pct=2.0, stop_loss_pct=1.0, max_holding_bars=5)
    assert pos.min_hold_bars == 1 and pos.min_hold_seconds == 0.0 and pos.entry_latency_seconds == 0.0
    info = check_position_exit(pos, bar(1000, 99.5, 99.6, 90.0, 91.0), 0.04)   # same timestamp: still exits
    assert info is not None and info["exit_price"] == pytest.approx(99.0)


def test_a_gap_in_the_data_only_lengthens_a_trade():
    pos = locked_position(entry_latency_seconds=0.0)
    far_later = bar(5000, 99.5, 99.6, 90.0, 91.0)
    info = check_position_exit(pos, far_later, 0.04)
    assert info["hold_seconds"] == 4000


# ---------------------------------------------------------------------------
# the model's training contract
# ---------------------------------------------------------------------------

def block(p, **over):
    base = training_alignment_block(
        p, population=POPULATION_EXPERT_GATE, sample_policy="sequential",
        require_opportunity_filter_pass=True, n_candidates=100, n_samples=40,
    )
    base.update(over)
    return base


def test_a_matching_contract_is_accepted():
    p = policy()
    assert model_alignment_problem({"training_alignment": block(p)}, p, require_aligned=True,
                                   expected_population=POPULATION_EXPERT_GATE) is None


def test_a_legacy_model_is_refused_only_when_alignment_is_required():
    p = policy()
    assert "no training_alignment" in model_alignment_problem({}, p, require_aligned=True)
    assert model_alignment_problem({}, p, require_aligned=False) is None
    assert model_alignment_problem({}, None, require_aligned=True) is None      # no policy: no check


@pytest.mark.parametrize("trained,running,fragment", [
    (policy("10s"), policy("5s", max_bars=5), "granularity"),
    (policy("10s", min_seconds=5.0), policy("10s", min_seconds=10.0), "min_trade_seconds"),
    (policy("10s", max_bars=5), policy("10s", max_bars=8), "max_holding_bars"),
    (policy("10s", latency=0.0), policy("10s", latency=2.0), "entry_latency_seconds"),
])
def test_a_model_trained_under_another_policy_is_refused(trained, running, fragment):
    problem = model_alignment_problem({"training_alignment": block(trained)}, running, require_aligned=True)
    assert problem is not None and fragment in problem


def test_a_model_whose_labels_broke_the_rule_is_refused():
    p = policy()
    problem = model_alignment_problem({"training_alignment": block(p, min_duration_violations=3)}, p,
                                      require_aligned=True)
    assert "3 trade" in problem


def test_a_model_trained_on_another_population_is_refused():
    p = policy()
    problem = model_alignment_problem({"training_alignment": block(p, population="zone_touch_events")}, p,
                                      require_aligned=True, expected_population=POPULATION_EXPERT_GATE)
    assert "zone_touch_events" in problem


def test_the_registry_gate_reports_ok_and_why_not():
    p = policy()
    assert registry_alignment_status({"training_alignment": block(p)}, p) == (True, None)
    ok, why = registry_alignment_status({}, p)
    assert not ok and "retrain" in why


# ---------------------------------------------------------------------------
# every model loader enforces the contract
# ---------------------------------------------------------------------------

def write_model(path, p=None, legacy=False, **block_over):
    payload = {"model": object(), "feature_names": ["f"], "granularity": "10s", "feature_version": "v2_renamed",
               "avg_favorable_return": 0.1, "avg_unfavorable_return": -0.1, "avg_adverse_excursion": 0.1}
    if not legacy:
        payload["training_alignment"] = block(p, **block_over)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return str(path)


EXPERTS = [("trend", TrendExpert), ("mean_reversion", MeanReversionExpert), ("breakout", BreakoutExpert)]


@pytest.fixture
def ten_second_policy():
    return policy("10s", max_bars=5)


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_every_expert_serves_a_matching_model(tmp_path, paper_config, ten_second_policy, key, cls):
    path = write_model(tmp_path / "m.pkl", ten_second_policy)
    expert = cls(paper_config["experts"][key], path, "10s", ten_second_policy, True)
    assert expert._model is not None and not expert.use_fallback_mode


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_every_expert_refuses_a_legacy_model(tmp_path, paper_config, ten_second_policy, key, cls):
    path = write_model(tmp_path / "m.pkl", legacy=True)
    expert = cls(paper_config["experts"][key], path, "10s", ten_second_policy, True)
    assert expert._model is None and expert.use_fallback_mode


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_every_expert_still_accepts_a_legacy_model_when_not_required(tmp_path, paper_config,
                                                                      ten_second_policy, key, cls):
    path = write_model(tmp_path / "m.pkl", legacy=True)
    expert = cls(paper_config["experts"][key], path, "10s", ten_second_policy, False)
    assert expert._model is not None


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_every_expert_refuses_a_model_from_another_policy(tmp_path, paper_config, ten_second_policy, key, cls):
    other = policy("10s", min_seconds=3.0, max_bars=5)
    path = write_model(tmp_path / "m.pkl", other)
    expert = cls(paper_config["experts"][key], path, "10s", ten_second_policy, True)
    assert expert._model is None


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_every_expert_refuses_a_touch_event_model(tmp_path, paper_config, ten_second_policy, key, cls):
    path = write_model(tmp_path / "m.pkl", ten_second_policy, population="zone_touch_events")
    expert = cls(paper_config["experts"][key], path, "10s", ten_second_policy, True)
    assert expert._model is None


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_an_expert_built_without_a_policy_is_unchanged(tmp_path, paper_config, key, cls):
    path = write_model(tmp_path / "m.pkl", legacy=True)
    expert = cls(paper_config["experts"][key], path, "10s")
    assert expert._model is not None and expert.hold_policy is None


@pytest.mark.parametrize("key,cls", EXPERTS)
def test_an_expert_fails_closed_under_an_unsatisfiable_policy(paper_config, key, cls, monkeypatch):
    from src.state.contracts import ExpertAction
    from tests.unit.test_phase3 import make_feature_vector, make_regime_state, make_zone_map

    args = (make_feature_vector(), make_zone_map(), make_regime_state())

    open_expert = cls(paper_config["experts"][key], None, "1m")
    monkeypatch.setattr(open_expert, "propose_candidate", lambda *a: 1)       # the gate fires
    assert open_expert.evaluate(*args).action == ExpertAction.BUY              # control

    broken = HoldPolicy(10.0, "1s", 1.0, 5, 0.0)                               # 5 x 1 s < 10 s
    blocked = cls(paper_config["experts"][key], None, "1s", broken, False)
    monkeypatch.setattr(blocked, "propose_candidate", lambda *a: 1)
    assert blocked.duration_blocked_reason is not None
    assert blocked.evaluate(*args).action == ExpertAction.NO_TRADE


def test_build_experts_hands_every_expert_the_policy(paper_config):
    experts = build_experts(paper_config, "1m")
    assert {e.hold_policy.min_trade_seconds for e in experts.values()} == {10.0}
    assert all(e.require_aligned_models for e in experts.values())


def test_build_experts_refuses_an_unsatisfiable_config(paper_config):
    cfg = copy.deepcopy(paper_config)
    cfg["system"]["data_granularity"] = "1s"
    cfg["labeling"]["max_holding_bars"] = 5
    with pytest.raises(HoldPolicyError):
        build_experts(cfg, "1s")


def test_the_opportunity_filter_router_and_risk_model_enforce_it_too(tmp_path, paper_config, ten_second_policy):
    legacy = write_model(tmp_path / "legacy.pkl", legacy=True)
    good = write_model(tmp_path / "good.pkl", ten_second_policy)

    for path, expect_loaded in ((legacy, False), (good, True)):
        opp = OpportunityFilter({}, {}, path, "10s", ten_second_policy, True)
        router = Router({}, path, "10s", ten_second_policy, True)
        risk = RiskLayer({}, store=RiskStateStore(persistent=False), model_path=path, data_granularity="10s",
                         hold_policy=ten_second_policy, require_aligned_models=True)
        for component in (opp, router, risk):
            assert (component._model is not None) is expect_loaded, type(component).__name__


def test_the_shipped_config_requires_aligned_models_and_ten_seconds(paper_config):
    assert paper_config["trade_duration"]["min_trade_seconds"] == 10
    policy_, required = alignment_settings(paper_config)
    assert required and policy_.min_trade_seconds == 10.0


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_the_shipped_config_passes_the_hold_policy_validation(paper_config):
    validate_hold_policy(paper_config)


def test_validation_rejects_a_timeout_shorter_than_the_minimum(paper_config):
    cfg = copy.deepcopy(paper_config)
    cfg["system"]["data_granularity"] = "5s"
    cfg["labeling"]["max_holding_bars"] = 1
    with pytest.raises(ConfigConsistencyError, match="Hold policy"):
        validate_hold_policy(cfg)


def test_validation_rejects_an_embargo_shorter_than_the_label_horizon(paper_config):
    cfg = copy.deepcopy(paper_config)
    cfg["backtest"]["embargo_bars"] = 2
    cfg["labeling"]["max_holding_bars"] = 5
    with pytest.raises(ConfigConsistencyError, match="embargo"):
        validate_hold_policy(cfg)


def test_validation_rejects_an_unparseable_granularity(paper_config):
    cfg = copy.deepcopy(paper_config)
    cfg["system"]["data_granularity"] = "tenseconds"
    with pytest.raises(ConfigConsistencyError):
        validate_hold_policy(cfg)


# ---------------------------------------------------------------------------
# walk-forward windows
# ---------------------------------------------------------------------------

def test_an_embargo_shorter_than_the_label_horizon_is_refused():
    with pytest.raises(ValueError, match="label horizon"):
        WalkForwardSplitter(100, 20, 10, embargo_bars=3, label_horizon_bars=5)
    WalkForwardSplitter(100, 20, 10, embargo_bars=5, label_horizon_bars=5)


def test_bar_index_windows_are_measured_in_bars_not_rows():
    rng = np.random.default_rng(0)
    bars = np.sort(rng.choice(np.arange(0, 1000), size=300, replace=False))
    sp = WalkForwardSplitter(train_bars=300, test_bars=100, step_bars=100, embargo_bars=10, label_horizon_bars=5)
    windows = list(sp.split_by_bar_index(bars))
    assert windows
    for train, test in windows:
        assert bars[train].max() + 10 < bars[test].min() + 1          # embargo respected
        assert bars[test].max() - bars[test].min() < 100              # a test window is 100 BARS wide
        assert bars[train].max() - bars[train].min() < 300


def test_the_purge_drops_training_rows_whose_label_reaches_into_the_test_window():
    bars = np.arange(0, 200)
    label_end = bars + 5
    label_end[59] = 70            # one label decided deep inside the test window
    sp = WalkForwardSplitter(train_bars=60, test_bars=20, step_bars=50, embargo_bars=5, label_horizon_bars=5)
    (train, test), *_ = list(sp.split_by_bar_index(bars, label_end))
    assert 59 not in train.tolist()
    assert 58 in train.tolist()
    assert bars[test].min() == 65


def test_windows_never_overlap_train_and_test():
    bars = np.arange(0, 500)
    sp = WalkForwardSplitter(100, 50, 25, embargo_bars=5, label_horizon_bars=5)
    for train, test in sp.split_by_bar_index(bars):
        assert not set(train.tolist()) & set(test.tolist())
        assert bars[train].max() < bars[test].min()


def test_the_legacy_row_splitter_is_unchanged():
    sp = WalkForwardSplitter(100, 20, 20, 5)
    first = next(iter(sp.split(400)))
    assert len(first[0]) == 100 and len(first[1]) == 20
