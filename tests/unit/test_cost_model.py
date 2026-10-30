"""
Unit tests for Phase 3 T3.0: `CostModel` (`src/execution/cost_model.py`)
and AST gate forbidding direct `fee_pct`/`slippage_pct` config dict reads
outside `src/execution/cost_model.py`.
"""
import ast
from pathlib import Path

import pytest

from src.execution.cost_model import CostModel
from src.labeling.hold_policy import HoldPolicy
from src.state.contracts import FeatureVector, MarketSnapshot

pytestmark = [pytest.mark.unit]


def _bar(ts: int = 1000, close: float = 100.0, spread_pct: float = 0.02, depth_btc: float = 10.0) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        symbol="BTC/USDT",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        bid=close * (1 - spread_pct / 200),
        ask=close * (1 + spread_pct / 200),
        bid_volume=depth_btc / 2,
        ask_volume=depth_btc / 2,
        spread_pct=spread_pct,
        hour_utc=12,
        day_of_week=2,
    )


def _fv(vol_med: float = 0.25) -> FeatureVector:
    return FeatureVector(
        timestamp=1000,
        symbol="BTC/USDT",
        momentum_short=0.1,
        momentum_medium=0.1,
        momentum_long=0.1,
        volatility_short=0.1,
        volatility_medium=vol_med,
        volatility_long=0.2,
        returns_5=0.05,
        rsi_14=55.0,
        macd_signal=0.01,
        sma_9=100.0,
        sma_21=99.5,
        imbalance_ratio=0.1,
        orderflow_momentum=0.05,
        spread_pct=0.02,
        depth_10_level=100000.0,
    )


def test_legacy_aliases_map_cleanly_to_cost_model():
    cfg = {
        "labeling": {"fee_pct": 0.06, "slippage_buffer_pct": 0.03},
        "execution": {
            "slippage_pct": 0.02,
            "simulated_latency_ms": 2000,
            "latency_adverse_pct_per_100ms": 0.001,
            "depth_participation_rate": 0.2,
            "order_rejection_rate": 0.05,
        },
    }
    cm = CostModel.from_config(cfg)
    assert cm.fee_pct("entry") == pytest.approx(0.06)
    assert cm.fee_pct("exit") == pytest.approx(0.06)
    assert cm.round_trip_fee_pct() == pytest.approx(0.12)
    assert cm.entry_delay_seconds == pytest.approx(2.0)
    # 0.02 base + 2.0s * (0.001 * 10 /s) = 0.04
    assert cm.slippage_pct("entry") == pytest.approx(0.04)
    assert cm.impact.participation_cap == pytest.approx(0.2)
    assert cm.rejects.order_rejection_rate == pytest.approx(0.05)
    assert cm.opportunity_filter_cost_pct() == pytest.approx(0.15)


def test_slippage_models_fixed_volatility_and_depth_based():
    b = _bar(close=100.0, spread_pct=0.02, depth_btc=10.0)  # depth_usdt = 1000
    fv = _fv(vol_med=0.4)

    cm_vol = CostModel.from_config({
        "costs": {
            "slippage": {
                "model": "volatility_scaled",
                "vol_scaled": {"base_pct": 0.01, "k_vol": 0.1},
                "exit_slippage_multiplier": 1.5,
                "stop_slippage_multiplier": 2.0,
            }
        }
    })
    assert cm_vol.slippage_pct("entry", fv=fv, market=b) == pytest.approx(0.01 + 0.1 * 0.4)
    assert cm_vol.slippage_pct("exit", fv=fv, market=b, is_stop=True) == pytest.approx(0.05 * 1.5 * 2.0)

    cm_depth = CostModel.from_config({
        "costs": {
            "slippage": {
                "model": "depth_based",
                "depth_based": {"base_pct": 0.005, "k_participation": 0.2},
                "max_slippage_pct": 0.03,
            }
        }
    })
    # size=100 on depth=1000 -> ratio=0.1 -> 0.005 + 0.2*0.1 = 0.025
    assert cm_depth.slippage_pct("entry", market=b, size_usdt=100.0) == pytest.approx(0.025)
    # size=500 -> ratio=0.5 -> 0.105 clamped to max_slippage_pct=0.03
    assert cm_depth.slippage_pct("entry", market=b, size_usdt=500.0) == pytest.approx(0.03)


def test_spread_impact_and_carry_calculations():
    b = _bar(close=100.0, spread_pct=0.01, depth_btc=10.0)
    cm = CostModel.from_config({
        "costs": {
            "spread": {"source": "snapshot", "floor_pct": 0.02, "exit_spread_multiplier": 1.25},
            "impact": {"enabled": True, "k_sqrt": 0.1},
            "carry": {"enabled": True, "funding_interval_hours": 8.0, "default_funding_rate_pct": 0.01, "borrow_apr_pct": 3.65},
        }
    })
    assert cm.spread_pct(b, leg="entry") == pytest.approx(0.02)
    assert cm.spread_pct(b, leg="exit") == pytest.approx(0.025)
    assert cm.impact_pct(size_usdt=250.0, depth_usdt=1000.0) == pytest.approx(0.1 * 0.5)
    # 8 hours hold on long (+1): funding = 0.01%, borrow = (8 / 8760) * 3.65
    carry_long = cm.carry_pct(direction=1, hold_seconds=8 * 3600)
    carry_short = cm.carry_pct(direction=-1, hold_seconds=8 * 3600)
    assert carry_long > carry_short


def test_hold_policy_includes_cost_fingerprint_when_non_default():
    default_pol = HoldPolicy.from_config({"system": {"data_granularity": "1m"}, "labeling": {"max_holding_bars": 5}})
    assert "cost_fingerprint" not in default_pol.fingerprint()

    custom_pol = HoldPolicy.from_config({
        "system": {"data_granularity": "1m"},
        "labeling": {"max_holding_bars": 5},
        "costs": {"fees": {"taker_pct": 0.08}, "latency": {"exit_delay_seconds": 2.0}},
    })
    fp = custom_pol.fingerprint()
    assert fp["exit_delay_seconds"] == 2.0
    assert "cost_fingerprint" in fp


def test_no_direct_fee_or_slippage_config_reads_outside_cost_model():
    """AST gate: no module in src/ outside src/execution/cost_model.py may read
    'fee_pct', 'slippage_pct', 'slippage_buffer_pct', or 'assumed_taker_fee_pct'
    via dict .get(...) or subscript."""
    forbidden_keys = {"fee_pct", "slippage_pct", "slippage_buffer_pct", "assumed_taker_fee_pct"}
    src_root = Path("src")
    allow = {src_root / "execution" / "cost_model.py"}
    violations = []

    for py_file in sorted(src_root.rglob("*.py")):
        if py_file in allow:
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in forbidden_keys
            ):
                violations.append(f"{py_file}:{node.lineno} .get({node.args[0].value!r})")
    assert not violations, f"Direct cost config reads outside cost_model.py: {violations}"
