from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .runtime import attach_runtime_manifest
from .normalization import apply_normalization_manifest
from environment.observation_schema import OBSERVATION_SCHEMA_VERSION


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
    return _deep_merge(base, raw), (*chain, resolved)


def load_config(path: str | Path, *, allow_observation_migration: bool = False) -> dict[str, Any]:
    """Load a JSON config with optional single-parent ``extends`` support."""

    config_path = project_path(path).resolve()
    config, chain = _load_config_path(config_path, stack=())
    saved_manifest = config.pop("runtime_manifest", None)
    if saved_manifest is not None and not isinstance(saved_manifest, Mapping):
        raise ValueError("saved runtime_manifest must be an object")
    if saved_manifest is not None and saved_manifest.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"saved runtime_manifest is incompatible: schema {OBSERVATION_SCHEMA_VERSION} requires retraining; older observation snapshots cannot be migrated")
    apply_normalization_manifest(config, project_root=PROJECT_ROOT)
    attach_runtime_manifest(config)
    if saved_manifest is not None and saved_manifest != config["runtime_manifest"]:
        raise ValueError(f"saved runtime_manifest does not match schema {OBSERVATION_SCHEMA_VERSION}; retraining is required and legacy observations cannot be migrated")
    config["_config_path"] = str(config_path)
    config["_config_chain"] = tuple(str(item) for item in chain)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable copy without loader-only metadata."""
    copy = deepcopy(config)
    copy.pop("_config_path", None)
    copy.pop("_config_chain", None)
    return copy
