"""
ModelBundle Manifest and Upstream Artefact Hashing (Phase 1, T1.3 / §4.2, §7.1).

Tracks the cohesive set of artefacts produced by a training run:
- `regime_set_hash`: hash of the active `RegimeSet`
- `stage_artefacts`: mapping of stage/component name -> SHA-256 of the artefact
- `upstream_hashes`: per stage, the mapping of upstream stage -> SHA-256 it was trained against
- `config_hash`: canonical JSON hash of training-relevant config sections
- `data_hash`: hash of the training dataset's timestamp range, row count, and column list
- `hold_policy`: fingerprint of `HoldPolicy`
- `active_instances`: list of enabled/selected expert instances in the bundle

Supports native, non-pickle serialization (LightGBM booster JSON/text,
NumPy `.npz`/`.json`) so hashes are deterministic across runs, while keeping
legacy pickle hashing behind an explicit fallback flag.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def canonical_json_hash(obj: Any) -> str:
    """Deterministic SHA-256 hex digest of a JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_artefact_sha256(path: str | Path) -> str:
    """Compute the deterministic SHA-256 of an artefact file or native model."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artefact not found: {p}")
    # If a native .booster.json or .native.json sidecar exists, hash that for determinism
    native_sidecar = p.with_suffix(p.suffix + ".native.json")
    if native_sidecar.exists():
        return hashlib.sha256(native_sidecar.read_bytes()).hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a training frame's timestamps, row count, and columns."""
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = sorted(str(c) for c in df.columns)
    first_ts = int(df["timestamp"].iloc[0]) if "timestamp" in df.columns else 0
    last_ts = int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else len(df)
    summary = {
        "n_rows": int(len(df)),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "columns": cols,
    }
    return canonical_json_hash(summary)


def save_native_model(payload: Mapping[str, Any], path: str | Path) -> str:
    """Save a model payload with a deterministic native sidecar alongside legacy pickle."""
    import pickle

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(dict(payload), f)

    model = payload.get("model")
    booster_str = None
    if model is not None:
        if hasattr(model, "booster_") and getattr(model, "booster_", None) is not None:
            booster_str = model.booster_.model_to_string()
        elif hasattr(model, "model_to_string"):
            booster_str = model.model_to_string()

    native_meta = {
        k: v for k, v in payload.items() if k != "model"
    }
    if booster_str is not None:
        native_meta["booster_model_str"] = booster_str
    elif model is not None and hasattr(model, "__dict__"):
        native_meta["model_repr"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in model.__dict__.items()
            if not k.startswith("_")
        }

    sidecar = p.with_suffix(p.suffix + ".native.json")
    sidecar.write_text(json.dumps(native_meta, sort_keys=True, default=str), encoding="utf-8")
    return compute_artefact_sha256(p)


@dataclass(frozen=True)
class ModelBundle:
    """Manifest binding all trained stage artefacts from a single pipeline run."""
    bundle_id: str
    regime_set_hash: str
    stage_artefacts: Dict[str, str]          # stage_or_component -> sha256
    artefact_paths: Dict[str, str]           # stage_or_component -> file path
    upstream_hashes: Dict[str, Dict[str, str]]  # stage -> {upstream_stage: sha256}
    config_hash: str
    data_hash: str
    hold_policy: Dict[str, Any]
    delay_policy: Dict[str, float]
    trained_at: str
    active_instances: Tuple[str, ...] = ("trend_expert", "mean_reversion_expert", "breakout_expert")
    disabled_instances: Dict[str, str] = field(default_factory=dict)

    @property
    def bundle_hash(self) -> str:
        core = {
            "regime_set_hash": self.regime_set_hash,
            "stage_artefacts": self.stage_artefacts,
            "upstream_hashes": self.upstream_hashes,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "hold_policy": self.hold_policy,
            "delay_policy": self.delay_policy,
            "active_instances": list(self.active_instances),
        }
        return canonical_json_hash(core)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "bundle_hash": self.bundle_hash,
            "regime_set_hash": self.regime_set_hash,
            "stage_artefacts": dict(self.stage_artefacts),
            "artefact_paths": dict(self.artefact_paths),
            "upstream_hashes": {k: dict(v) for k, v in self.upstream_hashes.items()},
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "hold_policy": dict(self.hold_policy),
            "delay_policy": dict(self.delay_policy),
            "trained_at": self.trained_at,
            "active_instances": list(self.active_instances),
            "disabled_instances": dict(self.disabled_instances),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            regime_set_hash=str(data.get("regime_set_hash", "legacy4")),
            stage_artefacts=dict(data.get("stage_artefacts") or {}),
            artefact_paths=dict(data.get("artefact_paths") or {}),
            upstream_hashes={k: dict(v) for k, v in (data.get("upstream_hashes") or {}).items()},
            config_hash=str(data.get("config_hash", "")),
            data_hash=str(data.get("data_hash", "")),
            hold_policy=dict(data.get("hold_policy") or {}),
            delay_policy=dict(data.get("delay_policy") or {"entry_delay_seconds": 0.0, "exit_delay_seconds": 0.0}),
            trained_at=str(data.get("trained_at") or datetime.now(timezone.utc).isoformat()),
            active_instances=tuple(data.get("active_instances") or ("trend_expert", "mean_reversion_expert", "breakout_expert")),
            disabled_instances=dict(data.get("disabled_instances") or {}),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def verify_consistency(self, check_files_on_disk: bool = True) -> Optional[str]:
        """Verify that every stage's recorded upstream_hashes match stage_artefacts
        and (optionally) the actual files on disk. Returns None if consistent,
        or a descriptive error string on mismatch.
        """
        for stage, upstreams in self.upstream_hashes.items():
            for up_name, expected_sha in upstreams.items():
                actual_in_bundle = self.stage_artefacts.get(up_name)
                if actual_in_bundle is not None and actual_in_bundle != expected_sha:
                    return (
                        f"stage {stage!r} was trained against {up_name}={expected_sha[:12]}..., "
                        f"but bundle carries {up_name}={actual_in_bundle[:12]}..."
                    )
                if check_files_on_disk and up_name in self.artefact_paths:
                    up_path = Path(self.artefact_paths[up_name])
                    if not up_path.exists():
                        return f"upstream artefact {up_name!r} missing on disk at {up_path}"
                    disk_sha = compute_artefact_sha256(up_path)
                    if disk_sha != expected_sha:
                        return (
                            f"upstream artefact {up_name!r} on disk ({disk_sha[:12]}...) "
                            f"differs from what {stage!r} was trained against ({expected_sha[:12]}...)"
                        )
        if check_files_on_disk:
            for comp, fpath in self.artefact_paths.items():
                expected = self.stage_artefacts.get(comp)
                if expected is None:
                    continue
                p = Path(fpath)
                if not p.exists():
                    return f"artefact {comp!r} missing on disk at {p}"
                disk_sha = compute_artefact_sha256(p)
                if disk_sha != expected:
                    return (
                        f"artefact {comp!r} on disk ({disk_sha[:12]}...) "
                        f"differs from bundle manifest ({expected[:12]}...)"
                    )
        return None


def verify_upstream_hashes(
    payload: Mapping[str, Any],
    loaded_upstream_hashes: Optional[Mapping[str, str]] = None,
    upstream_paths: Optional[Mapping[str, str | Path]] = None,
    *,
    require_bundle_consistency: bool = False,
) -> Optional[str]:
    """Check a model payload's `training_alignment.upstream_hashes` against currently loaded upstreams."""
    alignment = payload.get("training_alignment") or {}
    recorded = alignment.get("upstream_hashes")
    if not recorded:
        if require_bundle_consistency:
            return "model has no upstream_hashes in training_alignment (require_bundle_consistency=True)"
        return None

    if loaded_upstream_hashes:
        for up_key, exp_hash in recorded.items():
            if up_key in loaded_upstream_hashes and loaded_upstream_hashes[up_key] != exp_hash:
                return (
                    f"upstream {up_key!r} hash mismatch: model expects {exp_hash[:12]}..., "
                    f"loaded is {loaded_upstream_hashes[up_key][:12]}..."
                )

    if upstream_paths:
        for up_key, exp_hash in recorded.items():
            if up_key in upstream_paths:
                p = Path(upstream_paths[up_key])
                if not p.exists():
                    return f"upstream artefact {up_key!r} at {p} does not exist"
                disk_hash = compute_artefact_sha256(p)
                if disk_hash != exp_hash:
                    return (
                        f"upstream {up_key!r} on disk ({disk_hash[:12]}...) does not match "
                        f"recorded training upstream ({exp_hash[:12]}...)"
                    )
    return None
