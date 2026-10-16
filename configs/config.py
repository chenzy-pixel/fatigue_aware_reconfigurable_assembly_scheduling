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

    saved = raw.get("runtime_manifest")
    if isinstance(saved, Mapping) and saved.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("only observation schema 12 is supported; older models require retraining")
    raw_network = raw.get("network", {})
    if not isinstance(raw_network, Mapping):
        raise TypeError("network config must be an object")
    declared = raw_network.get("observation_schema_version", OBSERVATION_SCHEMA_VERSION)
    if isinstance(declared, bool) or declared != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("only observation schema 12 is supported; older models require retraining")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw, (resolved,)
    if not isinstance(parent, str) or not parent.strip():
        raise TypeError(
            f"configuration extends must be one non-empty string: {resolved}"
        )
    snapshot_fields = {
        "paths", "dataset", "generator", "environment", "network", "ppo",
        "reward", "training", "objective_scalarizer", "preference", "evaluation",
    }
    if isinstance(raw.get("runtime_manifest"), Mapping) and snapshot_fields <= raw.keys():
        # Training writes complete effective snapshots. An old `extends` is
        # provenance, not permission to import new defaults (or a dependency on
        # an experimental preset that has since been retired).
        return raw, (resolved,)
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base, chain = _load_config_path(
        parent_path,
        stack=(*stack, resolved),
    )
    merged = _deep_merge(base, raw)
    raw_manifest = raw.get("runtime_manifest")
    if isinstance(raw_manifest, Mapping):
        # Preserve a saved architecture identity before inheriting new defaults.
        if "actor_head_variant" not in raw.get("network", {}) and "actor_head_variant" in raw_manifest:
            merged.setdefault("network", {})["actor_head_variant"] = raw_manifest["actor_head_variant"]
        if "flow_mode" not in raw.get("objective_scalarizer", {}):
            if raw_manifest.get("flow_mode", "raw_v1") == "raw_v1":
                merged.setdefault("objective_scalarizer", {}).pop("flow_mode", None)
            else:
                merged.setdefault("objective_scalarizer", {})["flow_mode"] = raw_manifest["flow_mode"]
    # Historical snapshots may still extend a parent whose new default changed.
    # Do not add even an explicit legacy field: historical effective hashes stay valid.
    if isinstance(raw_manifest, Mapping) and "sample_aggregation" not in raw.get("training", {}).get("formal_evaluation", {}):
        merged.get("training", {}).get("formal_evaluation", {}).pop("sample_aggregation", None)
    return merged, (*chain, resolved)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config with optional single-parent ``extends`` support."""

    config_path = project_path(path).resolve()
    config, chain = _load_config_path(config_path, stack=())
    saved_manifest = config.pop("runtime_manifest", None)
    if saved_manifest is not None and not isinstance(saved_manifest, Mapping):
        raise ValueError("saved runtime_manifest must be an object")
    if saved_manifest is not None and saved_manifest.get("observation_schema") not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ValueError(f"saved runtime_manifest is incompatible: schema {OBSERVATION_SCHEMA_VERSION} requires retraining; older observation snapshots cannot be migrated")
    if "workload_objective" not in config:
        raise ValueError("workload_objective is required for observation schema 12")
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
