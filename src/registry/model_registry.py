"""
Model Registry (Phase 4, §8): versioned models, champion/challenger
comparison, human-approved promotion, rollback. Extends (not replaces)
the .meta.json model-card convention already established in Phase 2/3 --
those files still get written by every train_*.py script; this table is
what model-loading code actually resolves the active file path from,
rather than a fixed filename.

Status lifecycle: pending_approval -> approved_active -> superseded
                                    -> rejected
Never auto-promotes: retrain_all.py only ever inserts pending_approval
rows (§8.2); approve_model.py is the only code path that sets
approved_active (§8.3).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict
import json

from .db import connect, DEFAULT_DB_PATH


def register_version(
    component: str, file_path: str, data_granularity: str,
    walk_forward_metrics: dict, status: str = "pending_approval",
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    version_id = f"{component}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO model_versions (version_id, component, trained_at, data_granularity, "
            "file_path, walk_forward_metrics_json, status, promoted_at, promoted_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            (version_id, component, datetime.now(timezone.utc).isoformat(), data_granularity,
             file_path, json.dumps(walk_forward_metrics), status),
        )
    return version_id


def get_active(component: str, db_path: str = DEFAULT_DB_PATH) -> Optional[str]:
    """Return the active model's file_path for a component, or None if
    nothing has ever been approved (caller should fall back to whatever
    default path convention it already uses)."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT file_path FROM model_versions WHERE component=? AND status='approved_active'",
            (component,),
        ).fetchone()
        return row["file_path"] if row else None


def get_pending(component: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    with connect(db_path) as conn:
        if component:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE component=? AND status='pending_approval' ORDER BY trained_at",
                (component,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE status='pending_approval' ORDER BY component, trained_at"
            ).fetchall()
        return [dict(r) for r in rows]


def get_champion(component: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE component=? AND status='approved_active'", (component,)
        ).fetchone()
        return dict(row) if row else None


def approve(version_id: str, promoted_by: str = "cli", db_path: str = DEFAULT_DB_PATH) -> None:
    """Promote a pending_approval version to approved_active, superseding
    whatever was previously active for that component. This is the ONLY
    function in the registry that sets approved_active -- retrain_all.py
    never calls it."""
    with connect(db_path) as conn:
        row = conn.execute("SELECT component FROM model_versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown version_id: {version_id}")
        component = row["component"]

        conn.execute(
            "UPDATE model_versions SET status='superseded' WHERE component=? AND status='approved_active'",
            (component,),
        )
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE model_versions SET status='approved_active', promoted_at=?, promoted_by=? WHERE version_id=?",
            (now, promoted_by, version_id),
        )


def reject(version_id: str, db_path: str = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE model_versions SET status='rejected' WHERE version_id=?", (version_id,))


def rollback(component: str, to_version_id: str, promoted_by: str = "cli", db_path: str = DEFAULT_DB_PATH) -> None:
    """Same mechanism as approve() run in reverse -- every version stays
    on disk and in the registry, so rollback is just re-promoting an
    older version_id, no special-cased undo logic needed (§8.4)."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT component FROM model_versions WHERE version_id=? AND component=?",
            (to_version_id, component),
        ).fetchone()
        if row is None:
            raise ValueError(f"version_id {to_version_id} not found for component {component}")
    approve(to_version_id, promoted_by=promoted_by, db_path=db_path)


def list_versions(component: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM model_versions WHERE component=? ORDER BY trained_at DESC", (component,)
        ).fetchall()
        return [dict(r) for r in rows]
