from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .runtime import attach_runtime_manifest
from .normalization import apply_normalization_manifest
from environment.observation_schema import OBSERVATION_SCHEMA_VERSION, SUPPORTED_OBSERVATION_SCHEMAS
from environment.workload import observation_schema_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Resolve a project-relative path without depending on the working directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalars and lists."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_path(
    config_path: Path,
    *,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    resolved = config_path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(path) for path in (*stack, resolved))
        raise ValueError(f"configuration extends cycle detected: {cycle}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be an object: {resolved}")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw, (resolved,)
    if not isinstance(parent, str) or not parent.strip():
        raise TypeError(
            f"configuration extends must be one non-empty string: {resolved}"
        )
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base, chain = _load_config_path(
        parent_path,
        stack=(*stack, resolved),
    )
    merged = _deep_merge(base, raw)
    # A saved schema-10 snapshot keeps its own variance contract even if its
    # parent now selects the modern workload objective.
    raw_manifest = raw.get("runtime_manifest")
    legacy_snapshot = isinstance(raw_manifest, Mapping) and raw_manifest.get("observation_schema") == 10
    if isinstance(raw_manifest, Mapping) and raw_manifest.get("observation_schema") in {10, 11}:
        merged.setdefault("network", {}).setdefault("observation_schema_version", raw_manifest["observation_schema"])
        # A saved snapshot must not acquire the new parent's encoding or version.
        if "observation_schema_version" not in raw.get("network", {}):
            merged["network"]["observation_schema_version"] = raw_manifest["observation_schema"]
        if "global_objective_encoding" not in raw.get("network", {}):
            merged["network"].pop("global_objective_encoding", None)
    if legacy_snapshot and "workload_objective" not in raw:
        merged.pop("workload_objective", None)
        if "observation_schema_version" not in raw.get("network", {}):
            merged.setdefault("network", {})["observation_schema_version"] = 10
        for field in ("workload_mode", "workload_q", "workload_order", "workload_weight_rule", "workload_version", "workload_scale", "objective_scales", "objective_names"):
            if field not in raw.get("network", {}):
                merged.get("network", {}).pop(field, None)
    return merged, (*chain, resolved)


def load_config(path: str | Path, *, allow_observation_migration: bool = False) -> dict[str, Any]:
    """Load a JSON config with optional single-parent ``extends`` support."""

    config_path = project_path(path).resolve()
    config, chain = _load_config_path(config_path, stack=())
    saved_manifest = config.pop("runtime_manifest", None)
    if saved_manifest is not None and not isinstance(saved_manifest, Mapping):
        raise ValueError("saved runtime_manifest must be an object")
    if saved_manifest is not None and saved_manifest.get("observation_schema") not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ValueError(f"saved runtime_manifest is incompatible: schema {OBSERVATION_SCHEMA_VERSION} requires retraining; older observation snapshots cannot be migrated")
    if saved_manifest is not None and saved_manifest.get("observation_schema") in {10, 11}:
        config.setdefault("network", {}).setdefault("observation_schema_version", saved_manifest["observation_schema"])
    if "workload_objective" not in config:
        if (saved_manifest is None or saved_manifest.get("observation_schema") != 10
                or int(config.get("network", {}).get("observation_schema_version", 10)) != 10):
            raise ValueError("missing workload_objective is allowed only in a verified saved schema-10 configuration")
        config.setdefault("network", {}).setdefault("observation_schema_version", 10)
    selected_schema = observation_schema_version(config)
    if saved_manifest is not None and saved_manifest.get("observation_schema") != selected_schema:
        raise ValueError("saved runtime_manifest observation schema disagrees with workload_objective")
    apply_normalization_manifest(config, project_root=PROJECT_ROOT)
    attach_runtime_manifest(config)
    if saved_manifest is not None and config["runtime_manifest"].get("flow_mode") == "raw_v1":
        saved_manifest = dict(saved_manifest)
        saved_manifest.setdefault("flow_mode", "raw_v1")
        saved_manifest.setdefault("normalization_manifest_sha256", config["objective_scalarizer"]["normalization_manifest_sha256"])
    if saved_manifest is not None and saved_manifest != config["runtime_manifest"]:
        raise ValueError("saved runtime_manifest does not match the current implementation; retraining is required")
    config["_config_path"] = str(config_path)
    config["_config_chain"] = tuple(str(item) for item in chain)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable copy without loader-only metadata."""
    copy = deepcopy(config)
    copy.pop("_config_path", None)
    copy.pop("_config_chain", None)
    return copy
