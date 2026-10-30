"""
End-to-end tests for real stage runners (`src/training/runners.py`),
`scripts/train_pipeline.py --execute`, and `scripts/evaluate_bundle.py --run-replay`.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_bundle import main as eval_bundle_main
from scripts.train_pipeline import main as train_pipeline_main
from src.registry.bundle import ModelBundle
from src.training.plan import TRAINING_DAG_STAGES, TrainingPlan


def test_train_pipeline_execute_produces_verified_model_bundle(tmp_path: Path) -> None:
    artefacts_dir = tmp_path / "artefacts"
    bundle_out = tmp_path / "bundle.json"

    rc = train_pipeline_main(
        [
            "--env",
            "paper",
            "--execute",
            "--synthetic-bars",
            "100",
            "--artefacts-dir",
            str(artefacts_dir),
            "--bundle-out",
            str(bundle_out),
        ]
    )
    assert rc == 0
    assert bundle_out.exists()

    bundle = ModelBundle.load(bundle_out)
    expected_stages = [s[0] for s in TRAINING_DAG_STAGES]
    assert TrainingPlan().stage_names() == expected_stages
    assert set(bundle.stage_artefacts.keys()) == set(expected_stages)
    assert len(bundle.stage_artefacts) == 10
    assert bundle.verify_consistency(check_files_on_disk=True) is None


def test_evaluate_bundle_run_replay_executes_ablation_rungs(tmp_path: Path) -> None:
    out_json = tmp_path / "eval_report.json"
    rc = eval_bundle_main(
        [
            "--bundle-hash",
            "replay_bundle_test",
            "--run-replay",
            "--synthetic-bars",
            "110",
            "--out",
            str(out_json),
        ]
    )
    assert rc in (0, 1)
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["bundle_hash"] == "replay_bundle_test"
    assert len(data["rungs"]) == 11
    assert data["rungs"][0]["rung_id"] == "L0_baseline"
    assert data["rungs"][-1]["rung_id"] == "L10_full_bundle"
