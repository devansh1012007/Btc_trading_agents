"""
Unit tests for Phase 4 T4.1 & T4.4: `TrailSpec`, `ExitPlan`, `resolve_exit_plan`,
and the tighten-only `hard_max_loss_pct` cap (C2).
"""
import pytest

from src.labeling.triple_barrier import barrier_exit_step, resolve_exit_plan
from src.state.contracts import ExitPlan, ExpertAction, ExpertOutput, MarketSnapshot, RegimeType, TrailSpec

pytestmark = [pytest.mark.unit]


def _bar(ts: int, o: float, h: float, low: float, c: float) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=o,
        high=h,
        low=low,
        close=c,
        volume=10.0,
        bid=c * 0.9999,
        ask=c * 1.0001,
        bid_volume=5.0,
        ask_volume=5.0,
        spread_pct=0.02,
        hour_utc=12,
        day_of_week=2,
    )


def _expert_out(plan=None) -> ExpertOutput:
    return ExpertOutput(
        expert_name="trend",
        timestamp=1000,
        symbol="BTC/USDT",
        action=ExpertAction.BUY,
        confidence=0.8,
        no_trade_probability=0.2,
        expected_return=0.5,
        expected_risk=0.2,
        reward_to_risk=2.5,
        invalidation_distance=0.4,
        zone_aware=True,
        model_output=0.8,
        regime_used=RegimeType.TRENDING_UP,
        exit_plan=plan,
    )


def test_exit_plan_round_trip_dict():
    plan = ExitPlan(
        stop_loss_pct=0.8,
        take_profit_pct=1.6,
        max_holding_bars=8,
        trailing=TrailSpec(enabled=True, activation_pct=0.5, distance_pct=0.3, step_pct=0.1),
        hard_max_loss_pct=0.6,
        source="expert",
    )
    restored = ExitPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.effective_stop_loss_pct == pytest.approx(0.6)


def test_resolve_exit_plan_precedence_ladder():
    cfg = {
        "labeling": {"k_tp": 2.0, "k_sl": 1.0, "max_holding_bars": 6},
        "exits": {"plan_source": "auto", "hard": {"max_loss_pct": 0.9}},
    }
    exp_plan = ExitPlan(stop_loss_pct=1.2, take_profit_pct=2.4, max_holding_bars=10, source="expert")
    bm_plan = ExitPlan(stop_loss_pct=0.7, take_profit_pct=1.4, max_holding_bars=7, source="barrier_model")

    # 1. Expert plan wins when present
    r1 = resolve_exit_plan(cfg, expert_output=_expert_out(exp_plan), barrier_model_plan=bm_plan, volatility_pct=0.5)
    assert r1.source == "expert"
    assert r1.stop_loss_pct == pytest.approx(1.2)
    # Tighten-only hard cap (0.9 < 1.2) tightens effective stop loss to 0.9
    assert r1.effective_stop_loss_pct == pytest.approx(0.9)

    # 2. Barrier model plan wins when expert has no plan
    r2 = resolve_exit_plan(cfg, expert_output=_expert_out(None), barrier_model_plan=bm_plan, volatility_pct=0.5)
    assert r2.source == "barrier_model"
    assert r2.stop_loss_pct == pytest.approx(0.7)
    # Tighten-only hard cap (0.9 > 0.7) does NOT widen 0.7!
    assert r2.effective_stop_loss_pct == pytest.approx(0.7)

    # 3. Regime-specific k*vol fallback when neither expert nor barrier model provides a plan
    r3 = resolve_exit_plan(cfg, expert_output=_expert_out(None), barrier_model_plan=None, volatility_pct=0.5)
    assert r3.source == "regime_specific"
    assert r3.take_profit_pct == pytest.approx(1.0)
    assert r3.stop_loss_pct == pytest.approx(0.5)


def test_hard_loss_cap_triggers_before_wider_stop_loss():
    plan = ExitPlan(
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_bars=5,
        hard_max_loss_pct=1.0,
        source="expert",
    )
    b = _bar(1060, 100.0, 100.2, 98.8, 98.9)  # crosses -1.0% (99.0), does not cross -2.0% (98.0)
    step = barrier_exit_step(1, 100.0, 4.0, 2.0, b, bars_held=1, max_holding_bars=5, min_hold_bars=1, exit_plan=plan)
    assert step == ("hard_loss_cap", pytest.approx(99.0))
