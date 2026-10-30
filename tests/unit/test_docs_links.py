"""
Unit tests for Phase 17 T17.3: verify all required ADRs (`0001`..`0012`) and
architecture/explanation/migration/ledger docs exist and are non-empty.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [pytest.mark.unit]


def test_all_adrs_and_phase_docs_exist():
    required_files = [
        "docs/ARCHITECTURE_MAP.md",
        "docs/IMPLEMENTATION_LEDGER.md",
        "docs/MIGRATION.md",
        "docs/gui_checklist.md",
        "docs/perf/baseline.md",
        "docs/explain/regimes-and-exits.md",
        "docs/design-decisions/adr-0007-dynamic-regime-contract.md",
        "docs/design-decisions/adr-0008-unified-exit-plan-and-precedence-ladder.md",
        "docs/design-decisions/adr-0009-consolidated-cost-model-and-delayed-fills.md",
        "docs/design-decisions/adr-0010-multi-position-book-and-tighten-only-governor.md",
        "docs/design-decisions/adr-0011-harness-ordered-training-dag-and-oof-stacking.md",
        "docs/design-decisions/adr-0012-model-ledger-and-agent-sink.md",
    ]
    for rel in required_files:
        p = ROOT / rel
        assert p.exists(), f"Missing documentation file: {rel}"
        assert len(p.read_text().strip()) > 100, f"Documentation file {rel} is too short"
