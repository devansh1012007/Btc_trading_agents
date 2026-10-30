"""
Unit tests for ModelBundle manifest and upstream artefact hashing (Phase 1, T1.3).
"""
from __future__ import annotations

import pickle
import pytest

from src.labeling.hold_policy import HoldPolicy, model_alignment_problem, training_alignment_block
from src.layers.opportunity_filter import OpportunityFilter
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.layers.router import Router
from src.registry.bundle import (
    ModelBundle,
    compute_artefact_sha256,
    save_native_model,
    verify_upstream_hashes,
)
from src.utils.config_helpers import load_layered_config

pytestmark = [pytest.mark.unit]


def test_bundle_roundtrip_and_hash_stability(tmp_path) -> None:
    exp_path = tmp_path / "trend_expert.pkl"
    sha_exp = save_native_model({"model": None, "feature_names": ["a"]}, exp_path)
    sha_exp_2 = compute_artefact_sha256(exp_path)
    assert sha_exp == sha_exp_2

    rtr_path = tmp_path / "router.pkl"
    sha_rtr = save_native_model({"model": None, "feature_names": ["b"]}, rtr_path)

    bundle = ModelBundle(
        bundle_id="b-001",
        regime_set_hash="legacy4",
        stage_artefacts={"trend_expert": sha_exp, "router": sha_rtr},
        artefact_paths={"trend_expert": str(exp_path), "router": str(rtr_path)},
        upstream_hashes={"router": {"trend_expert": sha_exp}},
        config_hash="cfg123",
        data_hash="dat123",
        hold_policy={"min_trade_seconds": 10.0},
        delay_policy={"entry_delay_seconds": 0.0, "exit_delay_seconds": 0.0},
        trained_at="2026-01-01T00:00:00Z",
    )
    assert bundle.verify_consistency() is None

    saved = bundle.save(tmp_path / "bundle.json")
    loaded = ModelBundle.load(saved)
    assert loaded.bundle_hash == bundle.bundle_hash
    assert loaded.verify_consistency() is None

    # Mutate 1 byte of the upstream expert file's native sidecar -> bundle refuses
    sidecar = exp_path.with_suffix(".pkl.native.json")
    sidecar.write_text(sidecar.read_text() + " ")
    problem = loaded.verify_consistency()
    assert problem is not None and "trend_expert" in problem


def test_downstream_loaders_refuse_mismatched_upstream_hashes(tmp_path) -> None:
    cfg = load_layered_config("paper")
    policy = HoldPolicy.from_config(cfg)

    exp_path = tmp_path / "trend_expert.pkl"
    exp_sha = save_native_model({"model": None, "feature_names": ["a"]}, exp_path)

    alignment = training_alignment_block(
        policy,
        population="bars",
        sample_policy="bars",
        require_opportunity_filter_pass=True,
        n_candidates=100,
        n_samples=100,
        bundle_hash="bundle-abc",
        upstream_hashes={"trend_expert": exp_sha},
    )
    payload = {
        "model": object(),
        "feature_names": ["a"],
        "granularity": "1m",
        "feature_version": "v2_renamed",
        "training_alignment": alignment,
    }

    # Matching upstream hash -> accepted
    assert model_alignment_problem(
        payload, policy, require_aligned=True,
        loaded_upstream_hashes={"trend_expert": exp_sha},
        require_bundle_consistency=True,
    ) is None

    # Changed upstream hash -> refused
    prob = model_alignment_problem(
        payload, policy, require_aligned=True,
        loaded_upstream_hashes={"trend_expert": "0" * 64},
        require_bundle_consistency=True,
    )
    assert prob is not None and "trend_expert" in prob

    # Disk upstream modified -> refused
    sidecar = exp_path.with_suffix(".pkl.native.json")
    sidecar.write_text(sidecar.read_text() + " ")
    prob_disk = verify_upstream_hashes(payload, upstream_paths={"trend_expert": exp_path})
    assert prob_disk is not None and "trend_expert" in prob_disk
