"""
Unit tests for generic regime identity, DynamicRegime, RegimeSet, consumer
accessors, schema versioning, and AST structural gate (Phase 2, T2.1–T2.3).
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.config.regime_adaptive import resolve_regime_param
from src.feedback.system_state_archive import (
    load_system_state_archive,
    write_system_state_archive,
)
from src.gui import theme
from src.layers.router import FEATURE_NAMES, Router, build_features, router_feature_names
from src.layers.structure_engine import DynamicStructureEngine
from src.layers.zone_survival_features import (
    ZONE_SURVIVAL_FEATURE_NAMES,
    features_to_row,
    zone_survival_feature_names,
    zone_survival_features,
)
from src.state.contracts import (
    LEGACY_REGIME_SET,
    DynamicRegime,
    ExpertOutputs,
    MarketSnapshot,
    OpportunityScore,
    RegimeSet,
    RegimeState,
    RegimeType,
    SystemState,
    Zone,
    coerce_regime,
    regime_prob_features,
)
from src.state.serialization import system_state_to_dict
from src.utils.config_helpers import parse_regime_list

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parent.parent.parent

FORBIDDEN_PROB_ATTRS = frozenset({
    "prob_trending_up",
    "prob_trending_down",
    "prob_chop_range",
    "prob_panic_spike",
})


def _seven_regime_set() -> RegimeSet:
    ids = tuple(f"r{i}" for i in range(7))
    return RegimeSet(
        set_id="dyn7",
        regime_ids=ids,
        display_names={rid: rid.upper() for rid in ids},
    )


def test_dynamic_regime_duck_compatibility_and_dict_keys() -> None:
    r0 = DynamicRegime(id="r0", display_name="R0", semantic_tag="bull_low_vol")
    assert r0.value == "r0"
    assert r0.name == "R0"

    # Legacy alias equality & hash compatibility
    dt_up = DynamicRegime(id="trending_up")
    assert dt_up == RegimeType.TRENDING_UP
    assert hash(dt_up) == hash(RegimeType.TRENDING_UP)
    assert dt_up in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]

    d = {r0: 1.5, RegimeType.TRENDING_UP: 2.0}
    assert d[DynamicRegime("r0")] == 1.5
    assert d[dt_up] == 2.0

    # parse_regime_list supports both legacy and dynamic ids
    parsed = parse_regime_list(["TRENDING_UP", "r0", "r3"])
    assert parsed[0] == RegimeType.TRENDING_UP
    assert parsed[1] == DynamicRegime("r0")
    assert DynamicRegime("r3") in parsed


@pytest.mark.parametrize("use_seven", [False, True])
def test_regime_state_router_and_zone_features_parametrized(use_seven: bool) -> None:
    rset = _seven_regime_set() if use_seven else LEGACY_REGIME_SET
    probs = {rid: 1.0 / rset.k for rid in rset.regime_ids}
    primary = coerce_regime(rset.regime_ids[0])
    rs = RegimeState(
        timestamp=1_700_000_000,
        symbol="BTC/USDT",
        primary_regime=primary,
        regime_confidence=0.65,
        regime_set_id=rset.set_id,
        regime_probs=probs,
    )
    for rid in rset.regime_ids:
        assert rs.get_prob(rid) == pytest.approx(1.0 / rset.k)

    # Router features
    opp = OpportunityScore(
        timestamp=1_700_000_000, symbol="BTC/USDT",
        tradable=True, tradable_probability=0.8,
        environment_quality=0.7, opportunity_strength=1.2,
    )
    r_feats = build_features(ExpertOutputs(1_700_000_000, "BTC/USDT"), rs, opp, 2, regime_set=rset)
    expected_r_names = router_feature_names(rset)
    assert set(expected_r_names) <= set(r_feats.keys())
    if not use_seven:
        assert expected_r_names == FEATURE_NAMES

    # Zone survival features
    zone = Zone(
        zone_id="z1", lower_bound=99.0, upper_bound=101.0, mid=100.0,
        strength=0.7, age=5, touch_count=2, rejection_count=1, breakout_count=0,
        rejection_score=0.6, breakout_score=0.4, liquidity_score=0.5, survival_probability=0.7,
    )
    z_feats = zone_survival_features(zone, 0.5, 12, 2, rs, regime_set=rset)
    z_names = zone_survival_feature_names(rset)
    row = features_to_row(z_feats, z_names)
    assert row.shape == (1, 8 + rset.k)
    if not use_seven:
        assert z_names == ZONE_SURVIVAL_FEATURE_NAMES


def test_serialization_and_archive_schema_backward_compatibility(tmp_path) -> None:
    rset = _seven_regime_set()
    probs = {rid: 1.0 / 7.0 for rid in rset.regime_ids}
    snap = MarketSnapshot(
        timestamp=1_700_000_000, symbol="BTC/USDT",
        open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
        bid=100.4, ask=100.6, bid_volume=5.0, ask_volume=5.0, spread_pct=0.02,
    )
    rs = RegimeState(
        timestamp=1_700_000_000, symbol="BTC/USDT",
        primary_regime=DynamicRegime("r2"), regime_confidence=0.5,
        regime_set_id="dyn7", regime_probs=probs,
    )
    state = SystemState(timestamp=1_700_000_000, symbol="BTC/USDT", market_snapshot=snap, regime_state=rs)
    flat = system_state_to_dict(state)
    assert flat["regime_primary"] == "r2"
    assert flat["regime_set_id"] == "dyn7"
    assert json.loads(flat["regime_probs_json"])["r2"] == pytest.approx(1.0 / 7.0)

    # Write new archive & read back
    write_system_state_archive([state], base_dir=str(tmp_path / "archive"))
    df_new = load_system_state_archive("BTC/USDT", "2023-11-14", base_dir=str(tmp_path / "archive"))
    assert len(df_new) == 1
    assert int(df_new["schema_version"].iloc[0]) >= 2

    # Simulate old archive written before P2 (missing schema_version, regime_set_id, regime_probs_json)
    old_dir = tmp_path / "old_archive" / "BTC_USDT"
    old_dir.mkdir(parents=True)
    pd.DataFrame([{
        "timestamp": 1_700_000_000,
        "symbol": "BTC/USDT",
        "regime_primary": "trending_up",
        "regime_prob_up": 0.7,
        "regime_prob_down": 0.1,
        "regime_prob_chop": 0.15,
        "regime_prob_panic": 0.05,
    }]).to_parquet(old_dir / "2023-11-14.parquet")
    df_old = load_system_state_archive("BTC/USDT", "2023-11-14", base_dir=str(tmp_path / "old_archive"))
    assert int(df_old["schema_version"].iloc[0]) == 1
    assert df_old["regime_set_id"].iloc[0] == "legacy4"
    recovered = json.loads(df_old["regime_probs_json"].iloc[0])
    assert recovered["trending_up"] == pytest.approx(0.7)


def test_gui_theme_renders_seven_regimes_deterministically() -> None:
    rset = _seven_regime_set()
    pal_a = theme.regime_palette(rset.regime_ids)
    pal_b = theme.regime_palette(rset.regime_ids)
    assert pal_a == pal_b
    assert len(pal_a) == 7
    # Legacy four keep their exact colours
    for leg in ("trending_up", "trending_down", "chop_range", "panic_spike"):
        assert theme.regime_colour(leg) == theme.REGIME_COLOURS[leg]


def test_regime_adaptive_supports_dynamic_regime() -> None:
    block = {"default": 1.0, "by_regime": {"r3": 2.5, "R4": 3.0}}
    assert resolve_regime_param(block, DynamicRegime("r3")) == 2.5
    assert resolve_regime_param(block, DynamicRegime("r4")) == 3.0
    assert resolve_regime_param(block, DynamicRegime("r0")) == 1.0


def test_loaders_refuse_regime_set_hash_mismatch(tmp_path) -> None:
    rtr_path = tmp_path / "router.pkl"
    with open(rtr_path, "wb") as f:
        pickle.dump({
            "model": object(),
            "feature_names": FEATURE_NAMES,
            "granularity": "1m",
            "feature_version": "v2_renamed",
            "regime_set_hash": "set_A_hash",
        }, f)
    rtr = Router(config={}, model_path=str(rtr_path), data_granularity="1m", expected_regime_set_hash="set_B_hash")
    assert rtr._model is None

    z_path = tmp_path / "zone.pkl"
    with open(z_path, "wb") as f:
        pickle.dump({
            "model": object(),
            "feature_names": ZONE_SURVIVAL_FEATURE_NAMES,
            "granularity": "1m",
            "regime_set_hash": "set_A_hash",
        }, f)
    se = DynamicStructureEngine(
        zone_survival_model_path=str(z_path),
        zone_survival_blend_weight=0.5,
        data_granularity="1m",
        expected_regime_set_hash="set_B_hash",
    )
    assert se._zone_survival_model is None


def test_no_direct_regime_prob_fields_outside_contracts() -> None:
    """AST structural gate (T2.2): no file in src/ or scripts/ outside
    src/state/contracts.py may read .prob_trending_up / .prob_trending_down /
    .prob_chop_range / .prob_panic_spike directly.
    """
    violations = []
    for folder in (ROOT / "src", ROOT / "scripts"):
        for py_file in sorted(folder.rglob("*.py")):
            if py_file.resolve() == (ROOT / "src" / "state" / "contracts.py").resolve():
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_PROB_ATTRS:
                    violations.append(f"{py_file.relative_to(ROOT)}:{node.lineno}:{node.attr}")
    assert not violations, f"Direct regime prob attribute access outside contracts.py: {violations}"
