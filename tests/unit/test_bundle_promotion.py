"""
Unit tests for Phase 12 T12.3: atomic `ModelBundle` promotion (`promote_bundle`
in `scripts/approve_model.py`).
"""
import importlib
import sys
from pathlib import Path

import pytest

from src.registry.bundle import ModelBundle, compute_artefact_sha256
from src.registry.db import connect

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = [pytest.mark.unit]


def test_promote_bundle_validates_hashes_and_persists_to_sqlite(tmp_path):
    approve_mod = importlib.import_module("approve_model")
    db_path = str(tmp_path / "reg.db")
    f_path = tmp_path / "m.json"
    f_path.write_text('{"ok": true}')
    sha = compute_artefact_sha256(f_path)

    bundle = ModelBundle(
        bundle_id="bundle-test-1",
        regime_set_hash="rset_1",
        stage_artefacts={"trend_expert": sha},
        artefact_paths={"trend_expert": str(f_path)},
        upstream_hashes={},
        config_hash="cfg_1",
        data_hash="data_1",
        hold_policy={"min_trade_seconds": 10.0},
        delay_policy={"entry_delay_seconds": 0.0, "exit_delay_seconds": 0.0},
        trained_at="2026-01-01T00:00:00Z",
    )

    # Mismatched regime_set_hash is refused
    ok_bad, msg_bad = approve_mod.promote_bundle(
        bundle, db_path=db_path, expected_regime_set_hash="different_hash"
    )
    assert not ok_bad and "regime_set_hash" in msg_bad

    # Matching hashes promote cleanly into bundles table
    ok, _ = approve_mod.promote_bundle(
        bundle,
        db_path=db_path,
        expected_regime_set_hash="rset_1",
    )
    assert ok
    with connect(db_path) as conn:
        row = conn.execute("SELECT status, bundle_hash FROM bundles WHERE bundle_id=?", (bundle.bundle_id,)).fetchone()
        assert row is not None and row["status"] == "approved_active"
        assert row["bundle_hash"] == bundle.bundle_hash
