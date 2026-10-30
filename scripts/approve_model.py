"""
Human approval CLI (§8.3, M4.9). The only code path in this system that
sets a model version to approved_active. Nothing here requires a UI --
appropriate for a solo trader, upgradeable to a small dashboard later
without changing the underlying registry.
"""
import argparse
import json

from src.labeling.hold_policy import ALIGNED_COMPONENTS
from src.registry.model_registry import approve, get_champion, get_pending, reject


def format_metrics(metrics_json: str) -> str:
    try:
        m = json.loads(metrics_json) if metrics_json else {}
    except json.JSONDecodeError:
        return "(unparseable metrics)"
    avg_auc = m.get("avg_auc")
    n_samples = m.get("n_samples")
    parts = []
    if avg_auc is not None:
        parts.append(f"AUC: {avg_auc:.4f}")
    if n_samples is not None:
        parts.append(f"Samples: {n_samples}")
    return "  |  ".join(parts) if parts else "(no summary metrics)"


def main():
    parser = argparse.ArgumentParser(description="Review and approve pending model versions")
    parser.add_argument("--component", default=None, help="Limit to one component (default: all with pending versions)")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm every prompt (non-interactive use, e.g. scripted approval)")
    parser.add_argument("--db-path", default="data/registry.db")
    parser.add_argument("--force", action="store_true",
                        help="Promote even a version that is NOT harness-aligned (trained under a "
                             "different minimum-trade-duration policy, or before the contract existed)")
    args = parser.parse_args()

    pending = get_pending(component=args.component, db_path=args.db_path)
    if not pending:
        print("No pending model versions to review.")
        return

    for version in pending:
        component = version["component"]
        champion = get_champion(component, db_path=args.db_path)

        print(f"\n{'='*70}")
        print(f"Component: {component}")
        if champion:
            print(f"  Champion ({champion['version_id']}, promoted {champion['promoted_at']}):")
            print(f"    {format_metrics(champion['walk_forward_metrics_json'])}")
        else:
            print("  Champion: none yet -- this would be the first approved version")

        print(f"  Challenger ({version['version_id']}, trained {version['trained_at']}):")
        print(f"    {format_metrics(version['walk_forward_metrics_json'])}")

        challenger_metrics = json.loads(version["walk_forward_metrics_json"] or "{}")
        clears = challenger_metrics.get("clears_promotion_margin")
        if clears is not None:
            print(f"  Clears configured promotion margin: {clears}")

        aligned = challenger_metrics.get("alignment_ok")
        if component in ALIGNED_COMPONENTS:
            if aligned is True:
                print("  Harness-aligned: yes (hold policy, population and horizon match the running config)")
            else:
                reason = challenger_metrics.get("alignment_problem") or "no alignment record (legacy version)"
                print(f"  Harness-aligned: NO - {reason}")
                if not args.force:
                    print("  -> Refused. Retrain with scripts/retrain_all.py, or pass --force to override.")
                    continue

        if args.yes:
            answer = "y"
        else:
            answer = input(f"\n  Promote {version['version_id']} to active? [y/N]: ").strip().lower()

        if answer == "y":
            approve(version["version_id"], promoted_by="cli", db_path=args.db_path)
            print(f"  -> Promoted {version['version_id']} to approved_active.")
        else:
            reject(version["version_id"], db_path=args.db_path)
            print("  -> Left as reviewed/rejected. Re-run retrain_all.py to produce a new challenger later.")


if __name__ == "__main__":
    main()


def promote_bundle(
    bundle,
    *,
    db_path: str = "data/registry.db",
    expected_regime_set_hash: str | None = None,
    check_files_on_disk: bool = True,
) -> tuple[bool, str]:
    """Validate and atomically record an approved `ModelBundle` in `bundles` table."""
    from datetime import datetime, timezone

    from src.registry.db import connect

    if expected_regime_set_hash is not None and bundle.regime_set_hash not in (None, "", expected_regime_set_hash):
        return False, f"Bundle regime_set_hash={bundle.regime_set_hash!r} != expected={expected_regime_set_hash!r}"

    if not bundle.stage_artefacts:
        return False, "Bundle has no stage_artefacts to promote"

    err = bundle.verify_consistency(check_files_on_disk=check_files_on_disk)
    if err is not None:
        return False, f"Bundle consistency check failed: {err}"

    now_iso = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bundles "
            "(bundle_id, bundle_hash, regime_set_hash, manifest_json, status, trained_at, promoted_at, promoted_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bundle.bundle_id,
                bundle.bundle_hash,
                bundle.regime_set_hash,
                json.dumps(bundle.to_dict(), sort_keys=True),
                "approved_active",
                bundle.trained_at,
                now_iso,
                "cli",
            ),
        )
    return True, f"Promoted bundle {bundle.bundle_id} ({bundle.bundle_hash})"
