"""
Financial invariants (PRESENTATION §7): tests that prevent the system from
doing stupid things with capital. These are not component behavior tests
(test_phase4 covers those) — each test here pins an invariant that must
hold for ANY input, forever:

  - position size never exceeds the configured cap
  - risk layer only ever shrinks a position, never grows it
  - kill switch + circuit breakers can always block a trade
  - trade rate limits are enforced
  - duplicate orders are impossible, even across restarts

Extend this file when adding any new risk control.
No conftest.py — inline helpers only (repo convention).
"""
import pytest

from src.execution.backends.live_backend import LiveBackend
from src.layers.risk_layer import RiskLayer, RiskStateStore, liquidity_size_multiplier
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
)

pytestmark = [pytest.mark.unit]

# --------------------------------------------------------------------
# Helpers (inline copies of the test_phase4 patterns)
# --------------------------------------------------------------------

def make_snapshot(timestamp=0, close=100.0, bid_volume=5000.0, ask_volume=5000.0,
                  spread_pct=0.02, symbol="BTC/USDT"):
    return MarketSnapshot(
        timestamp=timestamp, symbol=symbol, open=close, high=close * 1.01, low=close * 0.99,
        close=close, volume=50, bid=close * (1 - spread_pct / 200), ask=close * (1 + spread_pct / 200),
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


def make_regime_state(primary_regime=RegimeType.TRENDING_UP, confidence=0.8):
    return RegimeState(
        timestamp=0, symbol="BTC/USDT", prob_trending_up=confidence, prob_trending_down=0.0,
        prob_chop_range=1 - confidence, prob_panic_spike=0.0, primary_regime=primary_regime,
        regime_confidence=confidence, regime_age=10, regime_persistence_score=0.8,
    )


def make_expert_outputs(name="trend_expert", action=ExpertAction.BUY, confidence=0.7,
                        invalidation_distance=0.5):
    output = ExpertOutput(
        expert_name=name, timestamp=0, symbol="BTC/USDT", action=action, confidence=confidence,
        no_trade_probability=1 - confidence, expected_return=0.1, expected_risk=0.1,
        reward_to_risk=max(0.1 / max(0.1, 1e-6), 0.0), invalidation_distance=invalidation_distance,
        zone_aware=True, model_output=confidence, regime_used=RegimeType.TRENDING_UP,
    )
    return ExpertOutputs(timestamp=0, symbol="BTC/USDT", outputs={name: output})


def make_router_decision(selected="trend_expert", confidence=0.7, timestamp=0):
    return RouterDecision(timestamp=timestamp, symbol="BTC/USDT", selected_expert=selected,
                          selection_confidence=confidence, no_trade_probability=1 - confidence)


SIZING = {"risk_pct_per_trade": 0.5, "max_position_pct_of_equity": 20.0}
HARD_RULES = {
    "min_invalidation_distance_pct": 0.1, "max_invalidation_distance_pct": 3.0,
    "min_bars_between_switches": 0, "reject_on_regime": [],
    "liquidity_size_reduction_curve": {"thin_threshold_usdt": 1000, "full_size_threshold_usdt": 5000},
}
BREAKERS = {"daily_loss_limit_pct": 2.0, "data_staleness_halt_seconds": 999999,
            "consecutive_losses_soft_pause": 3}


def make_risk_layer(db_path, sizing=None, hard_rules=None, breakers=None):
    return RiskLayer(
        config={
            "sizing": sizing or SIZING,
            "hard_rules": hard_rules or HARD_RULES,
            "circuit_breakers": breakers or BREAKERS,
        },
        store=RiskStateStore(db_path=str(db_path)),
        kill_switch_path=str(db_path) + ".NO_SUCH_KILL_SWITCH",
    )


def evaluate_approved(layer, equity=10_000.0, invalidation_distance=0.5, timestamp=0):
    decision = layer.evaluate(
        router_decision=make_router_decision(timestamp=timestamp),
        expert_outputs=make_expert_outputs(invalidation_distance=invalidation_distance),
        regime_state=make_regime_state(),
        zone_map=None,
        fv=make_feature_vector(),
        market=make_snapshot(),
        equity=equity,
    )
    return decision


# --------------------------------------------------------------------
# Invariant: position size is capped
# --------------------------------------------------------------------

@pytest.mark.parametrize("equity", [1_000.0, 10_000.0, 250_000.0])
@pytest.mark.parametrize("invalidation_distance", [0.11, 0.5, 2.9])
def test_position_never_exceeds_max_position_pct(tmp_path, equity, invalidation_distance):
    """Tight stops inflate risk-based sizing enormously — the cap must hold."""
    layer = make_risk_layer(tmp_path / f"inv_{equity}_{invalidation_distance}.db")
    decision = evaluate_approved(layer, equity=equity, invalidation_distance=invalidation_distance)
    assert decision.approved
    cap = equity * (SIZING["max_position_pct_of_equity"] / 100)
    assert decision.adjusted_size <= cap + 1e-9, (
        f"size {decision.adjusted_size} breached cap {cap} "
        f"(equity={equity}, stop={invalidation_distance})"
    )


# --------------------------------------------------------------------
# Invariant: risk only shrinks
# --------------------------------------------------------------------

@pytest.mark.parametrize("depth_usdt", [0.0, 999.0, 2500.0, 4999.0, 50_000.0])
def test_adjusted_size_never_exceeds_base_size(tmp_path, depth_usdt):
    layer = make_risk_layer(tmp_path / "shrink.db")
    decision = layer.evaluate(
        router_decision=make_router_decision(),
        expert_outputs=make_expert_outputs(),
        regime_state=make_regime_state(),
        zone_map=None, fv=make_feature_vector(),
        market=make_snapshot(bid_volume=depth_usdt / 2, ask_volume=depth_usdt / 2),
        equity=10_000.0,
    )
    assert decision.approved
    assert decision.adjusted_size <= decision.base_size + 1e-9
    assert decision.size_reduction_pct >= 0.0


def test_liquidity_multiplier_bounded_0_to_1():
    curve = {"thin_threshold_usdt": 1000, "full_size_threshold_usdt": 5000}
    for depth in (0.0, 500.0, 3000.0, 8000.0, 10**9):
        mult = liquidity_size_multiplier(depth, curve)
        assert 0.0 <= mult <= 1.0
    assert liquidity_size_multiplier(0.0, curve) == 0.0
    assert liquidity_size_multiplier(50_000.0, curve) == 1.0


# --------------------------------------------------------------------
# Invariant: the risk layer can always say NO
# --------------------------------------------------------------------

def test_kill_switch_blocks_any_trade(tmp_path):
    kill_path = str(tmp_path / "EMERGENCY_STOP")
    layer = RiskLayer(
        config={"sizing": SIZING, "hard_rules": HARD_RULES, "circuit_breakers": BREAKERS},
        store=RiskStateStore(db_path=str(tmp_path / "kill.db")),
        kill_switch_path=kill_path,
    )
    # Baseline sanity: without the file, trade passes.
    assert evaluate_approved(layer).approved

    open(kill_path, "w").close()
    blocked = evaluate_approved(layer)
    assert not blocked.approved
    assert blocked.rejection_reason == "manual_kill_switch_active"
    assert blocked.adjusted_size == 0.0


def test_tripped_breaker_blocks_and_persists_across_restart(tmp_path):
    db_path = str(tmp_path / "breaker.db")

    layer = make_risk_layer(db_path)
    assert evaluate_approved(layer).approved

    layer.store.trip("max_drawdown", reason="invariant-test")
    blocked = evaluate_approved(layer)
    assert not blocked.approved
    assert blocked.rejection_reason == "circuit_breaker_active:max_drawdown"

    # Simulated process restart: fresh RiskLayer over the same SQLite store.
    layer_after_restart = make_risk_layer(db_path)
    still_blocked = evaluate_approved(layer_after_restart)
    assert not still_blocked.approved, "breaker state did not survive restart"


def test_daily_loss_limit_halts_new_entries(tmp_path):
    layer = make_risk_layer(tmp_path / "daily.db", breakers={**BREAKERS, "daily_loss_limit_pct": 2.0})
    assert evaluate_approved(layer).approved

    layer.store.record_trade_result(pnl_pct=-2.5)  # breach the 2% daily limit
    blocked = evaluate_approved(layer)
    assert not blocked.approved
    assert blocked.rejection_reason == "circuit_breaker_active:daily_loss_limit"


def test_no_trade_router_is_zero_risk_rejection(tmp_path):
    layer = make_risk_layer(tmp_path / "notrade.db")
    decision = layer.evaluate(
        router_decision=make_router_decision(selected="no_trade", confidence=0.0),
        expert_outputs=make_expert_outputs(),
        regime_state=make_regime_state(),
        zone_map=None, fv=make_feature_vector(), market=make_snapshot(), equity=10_000.0,
    )
    assert not decision.approved
    assert decision.base_size == 0.0 and decision.adjusted_size == 0.0


# --------------------------------------------------------------------
# Invariant: trade rate limits hold
# --------------------------------------------------------------------

def test_max_trades_per_hour_enforced(tmp_path):
    layer = make_risk_layer(
        tmp_path / "rate.db",
        breakers={**BREAKERS, "max_trades_per_hour": 2, "min_bars_between_orders": 0},
    )

    t0 = 1_000_000
    assert evaluate_approved(layer, timestamp=t0).approved
    layer.record_trade(timestamp=t0)
    assert evaluate_approved(layer, timestamp=t0 + 60).approved
    layer.record_trade(timestamp=t0 + 60)

    # Third trade inside the same rolling hour → rejected...
    third = evaluate_approved(layer, timestamp=t0 + 120)
    assert not third.approved and third.rejection_reason.startswith("max_trades_per_hour")

    # ...but allowed again once the window slides past the oldest trade.
    assert evaluate_approved(layer, timestamp=t0 + 3601).approved


def test_min_bars_between_orders_enforced(tmp_path):
    cfg_hard_rules = {**HARD_RULES}
    layer = RiskLayer(
        config={"sizing": SIZING, "hard_rules": cfg_hard_rules,
                "circuit_breakers": BREAKERS, "min_bars_between_orders": 3},
        store=RiskStateStore(db_path=str(tmp_path / "bars.db")),
        kill_switch_path=str(tmp_path / "no_kill_switch"),
    )
    t0 = 2_000_000
    assert evaluate_approved(layer, timestamp=t0).approved
    layer.record_trade(timestamp=t0)

    too_soon = evaluate_approved(layer, timestamp=t0 + 1)
    assert not too_soon.approved and too_soon.rejection_reason.startswith("min_bars_between_orders")

    for _ in range(3):
        layer.advance_bar()
    assert evaluate_approved(layer, timestamp=t0 + 180).approved


# --------------------------------------------------------------------
# Invariant: duplicate orders are impossible (crash-restart safe)
# --------------------------------------------------------------------

class _CountingExchange:
    def __init__(self):
        self.create_order_calls = 0

    def create_order(self, **kwargs):
        self.create_order_calls += 1
        return {"id": "exch-1", "status": "closed", "average": 100.0, "filled": 10.0}

    def fetch_order_book(self, symbol, limit=10):
        return {"bids": [[99.9, 5.0]], "asks": [[100.1, 5.0]]}


def test_same_decision_never_places_two_orders_even_across_restart(tmp_path):
    class _Connector:
        def __init__(self, exchange):
            self.exchange = exchange

    exchange = _CountingExchange()
    db_path = str(tmp_path / "registry.db")
    config = {"mode": "live", "confirm_real_money": True, "order_type": "limit"}

    approved = RiskDecision(timestamp=0, symbol="BTC/USDT", approved=True, risk_score=0.1,
                            base_size=1000, adjusted_size=1000, size_reduction_pct=0.0,
                            rejection_reason=None)
    backend1 = LiveBackend(config, _Connector(exchange), "BTC/USDT", db_path=db_path)
    record1 = backend1.place_order(approved, make_router_decision(),
                                   make_expert_outputs().get_by_name("trend_expert"), make_snapshot())
    assert exchange.create_order_calls == 1

    backend2 = LiveBackend(config, _Connector(exchange), "BTC/USDT", db_path=db_path)
    record2 = backend2.place_order(approved, make_router_decision(),
                                   make_expert_outputs().get_by_name("trend_expert"), make_snapshot())

    assert exchange.create_order_calls == 1, "duplicate order placed after restart"
    assert record1.order_id == record2.order_id
