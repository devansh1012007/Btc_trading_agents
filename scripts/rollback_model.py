"""
Rollback CLI (§8.4). Every version stays on disk and in the registry, so
rollback is just re-promoting an older version_id through the same
approve() mechanism -- no special-cased undo logic (see model_registry.rollback()).
"""
import argparse

from src.registry.model_registry import list_versions, rollback


def main():
    parser = argparse.ArgumentParser(description="Roll a component back to a previously-approved version")
    parser.add_argument("--component", required=True)
    parser.add_argument("--to-version", default=None, help="version_id to restore; if omitted, lists available versions")
    parser.add_argument("--db-path", default="data/registry.db")
    args = parser.parse_args()

    versions = list_versions(args.component, db_path=args.db_path)
    if not versions:
        print(f"No versions found for component '{args.component}'.")
        return

    if args.to_version is None:
        print(f"Versions for {args.component}:")
        for v in versions:
            print(f"  {v['version_id']}  status={v['status']}  trained_at={v['trained_at']}")
        print("\nRe-run with --to-version <version_id> to roll back.")
        return

    rollback(args.component, args.to_version, promoted_by="cli", db_path=args.db_path)
    print(f"Rolled back {args.component} to {args.to_version} (now approved_active).")


if __name__ == "__main__":
    main()
