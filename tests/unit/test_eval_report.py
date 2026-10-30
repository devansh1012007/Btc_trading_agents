"""
Unit tests for Phase 17 T17.2: `scripts/evaluate_bundle.py` CLI.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = [pytest.mark.unit]


def test_evaluate_bundle_cli_writes_complete_ablation_report(tmp_path):
    eval_mod = importlib.import_module("evaluate_bundle")
    out_file = tmp_path / "eval_report.json"
    rc = eval_mod.main(["--bundle-hash", "test_bundle_99", "--out", str(out_file)])
    assert rc == 0
    payload = json.loads(out_file.read_text())
    assert payload["bundle_hash"] == "test_bundle_99"
    assert payload["passed"] is True
    assert len(payload["rungs"]) == 11
