"""
Golden regression oracle (Phase 0, T0.2).

Verifies that:
1. The default configuration produces the exact canonical JSON digest
   (zero tolerance) across both 60 s and 10 s scenarios.
2. Three consecutive runs produce identical SHA-256 digests (determinism).
3. Changing a known-sensitive parameter (e.g. fee_pct or momentum_threshold)
   changes the digest (guards against a vacuous oracle).
"""
from __future__ import annotations

import pytest

from tests.golden.digest import (
    SCENARIOS,
    compute_digest,
    diff_digests,
    golden_path,
    load_golden_digest,
)

pytestmark = [pytest.mark.integration]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_default_config_matches_golden_digest_exactly(scenario: str) -> None:
    path = golden_path(scenario)
    assert path.exists(), f"Missing golden file {path}"
    assert path.stat().st_size <= 200 * 1024, f"Golden file {path} exceeds 200 KB cap"

    expected = load_golden_digest(scenario)
    actual = compute_digest(**SCENARIOS[scenario])
    diff = diff_digests(expected, actual)
    assert expected == actual, f"Golden digest mismatch for {scenario}:\n{diff}"


def test_golden_digest_is_deterministic_across_three_consecutive_runs() -> None:
    params = SCENARIOS["paper_sim_1m_900"]
    runs = [compute_digest(**params) for _ in range(3)]
    assert runs[0]["sha256"] == runs[1]["sha256"] == runs[2]["sha256"]
    assert runs[0] == runs[1] == runs[2]


def test_golden_oracle_detects_sensitive_parameter_change() -> None:
    base = compute_digest(**SCENARIOS["paper_sim_1m_900"])
    altered = compute_digest(
        **SCENARIOS["paper_sim_1m_900"],
        config_overrides={"labeling": {"fee_pct": 0.12}},
    )
    assert base["sha256"] != altered["sha256"]
    assert base["final_equity"] != altered["final_equity"]
    assert diff_digests(base, altered) != ""
