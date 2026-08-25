"""
Phase 4 test suite (§14): synthetic liquidity, Risk Layer sizing/hard
rules/circuit breakers (including SQLite persistence across a simulated
restart), Execution Engine mode dispatch/idempotency, live-mode safety
gates, reconciliation, SystemState archive completeness, zone lifecycle
coverage, and model registry version tracking.
"""
import numpy as np
import pytest

from src.execution.backends.live_backend import LiveBackend, LiveTradingNotConfirmedError
from src.execution.backends.paper_sim_backend import PaperSimBackend
from src.execution.execution_engine import ExecutionEngine
from src.execution.reconciliation import reconcile
from src.layers.risk_layer import RiskLayer, RiskStateStore, compute_base_size, liquidity_size_multiplier
from src.layers.structure_engine import DynamicStructureEngine
from src.liquidity.synthetic_liquidity_model import attach_synthetic_liquidity, estimate_synthetic_liquidity
from src.monitoring.drift_monitor import population_stability_index
from src.registry import model_registry as reg
from src.state.contracts import (
    ExpertAction,
    ExpertOutput,
    ExpertOutputs,
    FeatureVector,
    MarketSnapshot,
    RegimeState,
    RegimeType,
    RiskDecision,
    RouterDecision,
    SystemState,
)
from src.state.serialization import system_state_to_dict

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def make_snapshot(timestamp=0, close=100.0, bid_volume=30.0, ask_volume=30.0, spread_pct=0.02, symbol="BTC/USDT"):
    return MarketSnapshot(
        timestamp=timestamp, symbol=symbol, open=close, high=close * 1.01, low=close * 0.99, close=close,
        volume=50, bid=close * (1 - spread_pct / 200), ask=close * (1 + spread_pct / 200),
        bid_volume=bid_volume, ask_volume=ask_volume, spread_pct=spread_pct,
        hour_utc=12, day_of_week=2,
    )


def make_feature_vector(timestamp=0, volatility_medium=0.1, volume_zscore=0.0, symbol="BTC/USDT"):
    return FeatureVector(
        timestamp=timestamp, symbol=symbol, momentum_short=0, momentum_medium=0.1, volatility_short=0.0,
        volatility_medium=volatility_medium, returns_5=0, rsi_14=50, macd_signal=0, sma_9=101, sma_21=100,
        momentum_long=0.1, volatility_long=0.1, imbalance_ratio=0.5, orderflow_momentum=0,
        spread_pct=0.02, depth_10_level=100, raw_features={"volume_zscore": volume_zscore},
    )


def make_regime_state(primary_regime=RegimeType.TRENDING_UP, confidence=0.8, persistence=0.8):
    return RegimeState(
        timestamp=0, symbol="BTC/USDT", prob_trending_up=confidence, prob_trending_down=0.0,
        prob_chop_range=1 - confidence, prob_panic_spike=0.0, primary_regime=primary_regime,
        regime_confidence=confidence, regime_age=10, regime_persistence_score=persistence,
    )


def make_expert_output(name="trend_expert", action=ExpertAction.BUY, confidence=0.7,
                        invalidation_distance=0.5, expected_return=0.1, expected_risk=0.1):
    return ExpertOutput(
        expert_name=name, timestamp=0, symbol="BTC/USDT", action=action, confidence=confidence,
        no_trade_probability=1 - confidence, expected_return=expected_return, expected_risk=expected_risk,
        reward_to_risk=max(expected_return / max(expected_risk, 1e-6), 0.0),
        invalidation_distance=invalidation_distance, zone_aware=True, model_output=confidence,
        regime_used=RegimeType.TRENDING_UP,
    )


def make_router_decision(selected="trend_expert", confidence=0.7):
    return RouterDecision(timestamp=0, symbol="BTC/USDT", selected_expert=selected,
                           selection_confidence=confidence, no_trade_probability=1 - confidence)


HARD_RULES = {
    "min_invalidation_distance_pct": 0.1, "max_invalidation_distance_pct": 3.0,
    "min_bars_between_switches": 0, "reject_on_regime": [],
    "liquidity_size_reduction_curve": {"thin_threshold_usdt": 1000, "full_size_threshold_usdt": 5000},
}
SIZING = {"risk_pct_per_trade": 0.5, "max_position_pct_of_equity": 20.0}
BREAKERS = {"daily_loss_limit_pct": 2.0, "data_staleness_halt_seconds": 999999, "consecutive_losses_soft_pause": 3}


def make_risk_layer(db_path, hard_rules=None, breakers=None):
    return RiskLayer(
        config={"sizing": SIZING, "hard_rules": hard_rules or HARD_RULES, "circuit_breakers": breakers or BREAKERS},
        store=RiskStateStore(db_path=str(db_path)),
        kill_switch_path=str(db_path) + ".NO_SUCH_KILL_SWITCH",
    )


# --------------------------------------------------------------------
# Synthetic liquidity
# --------------------------------------------------------------------

def test_synthetic_liquidity_bounds():
    bar = make_snapshot(close=50000.0)
    low_vol_fv = make_feature_vector(volatility_medium=0.01, volume_zscore=2.0)
    high_vol_fv = make_feature_vector(volatility_medium=1.0, volume_zscore=2.0)
    cfg = {"base_spread_pct": 0.015, "k_volatility": 0.8, "k_illiquidity": 0.02,
           "min_volume_zscore_floor": 0.1, "k_depth": 3.0, "depth_participation_rate": 0.15}

    low_vol_estimate = estimate_synthetic_liquidity(low_vol_fv, bar, cfg)
    high_vol_estimate = estimate_synthetic_liquidity(high_vol_fv, bar, cfg)

    assert low_vol_estimate.spread_pct >= 0
    assert high_vol_estimate.spread_pct > low_vol_estimate.spread_pct  # spread widens with volatility
    assert low_vol_estimate.bid_volume > 0 and low_vol_estimate.ask_volume > 0
    assert low_vol_estimate.source == "synthetic"


def test_liquidity_source_correctly_flagged():
    bar = make_snapshot()
    fv = make_feature_vector()
    cfg = {"base_spread_pct": 0.015, "k_volatility": 0.8, "k_illiquidity": 0.02,
           "min_volume_zscore_floor": 0.1, "k_depth": 3.0, "depth_participation_rate": 0.15}

    updated = attach_synthetic_liquidity(bar, fv, cfg)
    assert updated.liquidity_source == "synthetic"
    assert updated.bid_volume > 0  # no longer the old zero-depth fallback


# --------------------------------------------------------------------
# Risk Layer: sizing
# --------------------------------------------------------------------

def test_risk_position_sizing_formula():
    expert_output = make_expert_output(invalidation_distance=0.5)
    size = compute_base_size(equity=10000, expert_output=expert_output, cfg=SIZING)
    # risk_amount = 10000 * 0.5% = 50; size = 50 / (0.5/100) = 10000
    assert size == pytest.approx(10000.0, rel=1e-6)

    zero_stop = make_expert_output(invalidation_distance=0.0)
    assert compute_base_size(equity=10000, expert_output=zero_stop, cfg=SIZING) == 0.0


def test_liquidity_size_multiplier_smooth_ramp():
    cfg = {"liquidity_size_reduction_curve": {"thin_threshold_usdt": 1000, "full_size_threshold_usdt": 5000}}
    assert liquidity_size_multiplier(500, cfg) == 0.0
    assert liquidity_size_multiplier(10000, cfg) == 1.0
    mid = liquidity_size_multiplier(3000, cfg)
    assert 0.0 < mid < 1.0


# --------------------------------------------------------------------
# Risk Layer: hard rules veto ML, same pattern as Phase 3's Opportunity Filter
# --------------------------------------------------------------------

def test_risk_hard_rules_veto_ml(tmp_path):
    db_path = tmp_path / "registry.db"
    strict_rules = dict(HARD_RULES)
    strict_rules["max_invalidation_distance_pct"] = 0.2  # candidate below will violate this
    risk = make_risk_layer(db_path, hard_rules=strict_rules)

    expert_output = make_expert_output(invalidation_distance=1.0)  # exceeds max
    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(expert_output)
    decision = risk.evaluate(make_router_decision(), outputs, make_regime_state(), None,
                              make_feature_vector(), make_snapshot(), equity=10000)

    assert decision.approved is False
    assert "invalidation_distance_too_wide" in decision.rejection_reason


def test_opportunity_filter_regime_gate_still_enforced_in_risk_layer(tmp_path):
    """Defense in depth (§5.2): the Risk Layer independently rejects
    PANIC_SPIKE too, not just the Opportunity Filter."""
    db_path = tmp_path / "registry.db"
    risk = make_risk_layer(db_path, hard_rules={**HARD_RULES, "reject_on_regime": ["PANIC_SPIKE"]})

    expert_output = make_expert_output()
    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(expert_output)
    decision = risk.evaluate(make_router_decision(), outputs, make_regime_state(RegimeType.PANIC_SPIKE, 0.9),
                              None, make_feature_vector(), make_snapshot(), equity=10000)

    assert decision.approved is False
    assert "regime_blocked" in decision.rejection_reason


# --------------------------------------------------------------------
# Circuit breakers
# --------------------------------------------------------------------

def test_circuit_breaker_daily_loss_halts_entries_not_exits(tmp_path):
    db_path = tmp_path / "registry.db"
    risk = make_risk_layer(db_path, breakers={**BREAKERS, "daily_loss_limit_pct": 1.0})
    risk.store.record_trade_result(pnl_pct=-1.5)  # exceeds the 1.0% daily limit

    expert_output = make_expert_output()
    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(expert_output)
    decision = risk.evaluate(make_router_decision(), outputs, make_regime_state(), None,
                              make_feature_vector(), make_snapshot(), equity=10000)

    assert decision.approved is False
    assert "daily_loss_limit" in decision.rejection_reason
    # "Not exits" is architectural, not a code path: RiskLayer.evaluate()
    # is only ever called for new-entry candidates (see run.py's loop --
    # open-position exit checking never calls into RiskLayer at all), so
    # there is no exit call path here to separately test against.


def test_circuit_breaker_persists_across_restart(tmp_path):
    db_path = tmp_path / "registry.db"
    store1 = RiskStateStore(db_path=str(db_path))
    store1.trip("max_drawdown", "drawdown=9.0%")
    del store1  # simulate the process ending

    store2 = RiskStateStore(db_path=str(db_path))  # simulate a fresh process restarting
    assert store2.is_tripped("max_drawdown") is True
    assert any(h["breaker_type"] == "max_drawdown" for h in store2.active_halts())


def test_circuit_breaker_independent_triggers(tmp_path):
    db_path = tmp_path / "registry.db"
    store = RiskStateStore(db_path=str(db_path))

    store.trip("consecutive_execution_errors", "3 failures")
    assert store.is_tripped("consecutive_execution_errors") is True
    assert store.is_tripped("consecutive_losses_hard") is False  # independent

    store.reset("consecutive_execution_errors")
    assert store.is_tripped("consecutive_execution_errors") is False

    store.trip("consecutive_losses_hard", "6 losses")
    assert store.is_tripped("consecutive_losses_hard") is True
    assert store.is_tripped("consecutive_execution_errors") is False  # resetting one didn't affect the other


def test_manual_kill_switch_checked_first(tmp_path):
    kill_switch = tmp_path / "EMERGENCY_STOP"
    kill_switch.write_text("")
    risk = RiskLayer(
        config={"sizing": SIZING, "hard_rules": HARD_RULES, "circuit_breakers": BREAKERS},
        store=RiskStateStore(db_path=str(tmp_path / "registry.db")),
        kill_switch_path=str(kill_switch),
    )

    expert_output = make_expert_output()
    outputs = ExpertOutputs(timestamp=0, symbol="BTC/USDT").add_expert(expert_output)
    # Even a candidate that would pass every other check must be rejected.
    decision = risk.evaluate(make_router_decision(), outputs, make_regime_state(), None,
                              make_feature_vector(), make_snapshot(), equity=10000)

    assert decision.approved is False
    assert decision.rejection_reason == "manual_kill_switch_active"


# --------------------------------------------------------------------
# Execution Engine
# --------------------------------------------------------------------

def test_execution_mode_dispatch():
    backend = PaperSimBackend(config={"slippage_pct": 0.01, "depth_participation_rate": 0.15}, fee_pct=0.04)
    engine = ExecutionEngine(backend, mode="paper_sim")
    assert engine.mode == "paper_sim"
    assert isinstance(engine.backend, PaperSimBackend)

    expert_output = make_expert_output()
    risk_decision = RiskDecision(timestamp=0, symbol="BTC/USDT", approved=True, risk_score=0.1,
                                  base_size=1000, adjusted_size=1000, size_reduction_pct=0.0, rejection_reason=None)
    record = engine.place_order(risk_decision, make_router_decision(), expert_output, make_snapshot())
    assert record is not None
    assert record.status in ("filled", "partial", "rejected")

    # A rejected RiskDecision must produce no order at all, regardless of backend.
    rejected = RiskDecision(timestamp=0, symbol="BTC/USDT", approved=False, risk_score=1.0,
                             base_size=0, adjusted_size=0, size_reduction_pct=100.0, rejection_reason="test")
    assert engine.place_order(rejected, make_router_decision(), expert_output, make_snapshot()) is None


def test_execution_idempotent_order_placement(tmp_path, monkeypatch):
    """Simulated crash-and-restart: the same decision hashes to the same
    clientOrderId, and LiveBackend must not re-place it if it's already
    recorded (mocked exchange -- create_order should never be called on
    the second attempt)."""
    db_path = str(tmp_path / "registry.db")

    class MockExchange:
        def __init__(self):
            self.create_order_calls = 0

        def create_order(self, **kwargs):
            self.create_order_calls += 1
            return {"id": "exch-123", "status": "closed", "average": 100.0, "filled": 10.0}

        def fetch_order_book(self, symbol, limit=10):
            return {"bids": [[99.9, 5.0]], "asks": [[100.1, 5.0]]}

    class MockConnector:
        def __init__(self):
            self.exchange = MockExchange()

    connector = MockConnector()
    config = {"mode": "live", "confirm_real_money": True, "order_type": "limit"}
    backend = LiveBackend(config, connector, "BTC/USDT", db_path=db_path)

    expert_output = make_expert_output()
    risk_decision = RiskDecision(timestamp=0, symbol="BTC/USDT", approved=True, risk_score=0.1,
                                  base_size=1000, adjusted_size=1000, size_reduction_pct=0.0, rejection_reason=None)
    market = make_snapshot()
    router_decision = make_router_decision()

    record1 = backend.place_order(risk_decision, router_decision, expert_output, market)
    assert connector.exchange.create_order_calls == 1

    # Simulate a restart: a NEW backend instance, same DB, same decision.
    backend2 = LiveBackend(config, connector, "BTC/USDT", db_path=db_path)
    record2 = backend2.place_order(risk_decision, router_decision, expert_output, market)

    assert connector.exchange.create_order_calls == 1  # NOT called again
    assert record1.order_id == record2.order_id


def test_live_mode_requires_both_flags():
    class MockConnector:
        exchange = None

    with pytest.raises(LiveTradingNotConfirmedError):
        LiveBackend({"mode": "live", "confirm_real_money": False}, MockConnector(), "BTC/USDT")

    with pytest.raises(LiveTradingNotConfirmedError):
        LiveBackend({"mode": "paper_sim", "confirm_real_money": True}, MockConnector(), "BTC/USDT")

    # Both required, together -- this must succeed.
    backend = LiveBackend({"mode": "live", "confirm_real_money": True}, MockConnector(), "BTC/USDT")
    assert backend is not None


def test_reconciliation_detects_drift(tmp_path):
    from src.registry.db import connect

    db_path = str(tmp_path / "registry.db")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO orders (client_order_id, symbol, side, requested_size, status, exchange_order_id, created_at, updated_at) "
            "VALUES ('bot-order-1', 'BTC/USDT', 'BUY', 1000, 'open', 'exch-999', 'now', 'now')"
        )

    class MockExchange:
        def fetch_open_orders(self, symbol):
            return []  # exchange has NO open orders -- bot-order-1 must have filled/cancelled while bot was down

    class MockConnector:
        exchange = MockExchange()

    result = reconcile(MockConnector(), "BTC/USDT", db_path=db_path)
    assert result["status"] == "ok"
    assert len(result["discrepancies"]) == 1
    assert result["discrepancies"][0]["exchange_order_id"] == "exch-999"


# --------------------------------------------------------------------
# SystemState archive completeness + realized_pnl on every bar
# --------------------------------------------------------------------

def test_full_systemstate_archive_completeness():
    import dataclasses

    from src.state import contracts as c

    src_text = open("src/state/serialization.py").read()
    for dc_class in [c.MarketSnapshot, c.FeatureVector, c.RegimeState, c.ExpertOutput,
                      c.OpportunityScore, c.RouterDecision, c.RiskDecision, c.ExecutionRecord]:
        for f in dataclasses.fields(dc_class):
            present = f"'{f.name}'" in src_text or f".{f.name}" in src_text or f"{f.name}_json" in src_text
            assert present, f"{dc_class.__name__}.{f.name} not found in serialization.py"


def test_realized_pnl_populated_on_no_trade_bars():
    market = make_snapshot()
    state = SystemState(timestamp=0, symbol="BTC/USDT", market_snapshot=market)  # no trade at all
    d = system_state_to_dict(state)
    assert d["realized_pnl"] == 0.0  # not missing/None -- explicit 0.0, per §10's PPO/bandit requirement


# --------------------------------------------------------------------
# Zone lifecycle log
# --------------------------------------------------------------------

def test_zone_event_log_covers_full_lifecycle():
    from tests.test_phase2 import _build_pivot_high
    from tests.test_phase2 import make_snapshot as make_market_snapshot

    engine = DynamicStructureEngine(lookback_bars=50, pivot_window=3, zone_decay_bars=3)
    t, _ = _build_pivot_high(engine)
    assert any(e["event_type"] == "created" for e in engine.lifecycle_events)

    zone_id = next(iter(engine.zones))
    zone_before = engine.zones[zone_id]
    upper, lower = zone_before.upper_bound, zone_before.lower_bound
    approach = (upper + lower) / 2
    engine.add_bar(make_market_snapshot(t, approach, upper + 0.01, approach - 0.2, upper - 0.02))
    engine.compute_zones(current_price=upper - 0.02)
    assert any(e["event_type"] == "rejected" for e in engine.lifecycle_events)

    # Exercise decay directly and deterministically: advance the bar
    # counter far past zone_decay_bars and call the decay step directly a
    # few times, rather than relying on many more compute_zones() calls
    # (which can generate/merge new pivot zones from the price path and
    # confound which zone_id decays when).
    engine.bar_counter += 1000
    for _ in range(5):
        engine._decay_untouched_zones()
    assert any(e["event_type"] == "decayed" for e in engine.lifecycle_events)


# --------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------

def test_model_registry_version_tracking(tmp_path):
    db_path = str(tmp_path / "registry.db")
    v1 = reg.register_version("router", "router_v1.pkl", "1m", {"avg_auc": 0.6}, db_path=db_path)
    assert reg.get_pending("router", db_path=db_path)[0]["version_id"] == v1

    reg.approve(v1, db_path=db_path)
    assert reg.get_active("router", db_path=db_path) == "router_v1.pkl"
    assert reg.get_pending("router", db_path=db_path) == []

    v2 = reg.register_version("router", "router_v2.pkl", "1m", {"avg_auc": 0.7}, db_path=db_path)
    reg.approve(v2, db_path=db_path)
    versions = {v["version_id"]: v["status"] for v in reg.list_versions("router", db_path=db_path)}
    assert versions[v1] == "superseded"
    assert versions[v2] == "approved_active"


def test_challenger_never_auto_promotes(tmp_path):
    db_path = str(tmp_path / "registry.db")
    v1 = reg.register_version("router", "router_v1.pkl", "1m", {"avg_auc": 0.6}, db_path=db_path)
    reg.approve(v1, db_path=db_path)

    # Register several challengers -- none of this should ever touch approved_active.
    for i in range(3):
        reg.register_version("router", f"router_v{i+2}.pkl", "1m", {"avg_auc": 0.9}, db_path=db_path)

    assert reg.get_active("router", db_path=db_path) == "router_v1.pkl"
    assert len(reg.get_pending("router", db_path=db_path)) == 3


def test_rollback_restores_previous_version(tmp_path):
    db_path = str(tmp_path / "registry.db")
    v1 = reg.register_version("router", "router_v1.pkl", "1m", {}, db_path=db_path)
    reg.approve(v1, db_path=db_path)
    v2 = reg.register_version("router", "router_v2.pkl", "1m", {}, db_path=db_path)
    reg.approve(v2, db_path=db_path)
    assert reg.get_active("router", db_path=db_path) == "router_v2.pkl"

    reg.rollback("router", v1, db_path=db_path)
    assert reg.get_active("router", db_path=db_path) == "router_v1.pkl"
    versions = {v["version_id"]: v["status"] for v in reg.list_versions("router", db_path=db_path)}
    assert versions[v2] == "superseded"
    assert versions[v1] == "approved_active"


# --------------------------------------------------------------------
# Drift monitor
# --------------------------------------------------------------------

def test_drift_monitor_flags_shifted_distribution():
    np.random.seed(0)
    reference = np.random.normal(0, 1, 2000)
    stable = np.random.normal(0, 1, 500)
    shifted = np.random.normal(4, 1, 500)

    assert population_stability_index(reference, stable) < 0.25
    assert population_stability_index(reference, shifted) >= 0.25
