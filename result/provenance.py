"""Canonical experiment provenance and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from configs.config import PROJECT_ROOT, project_path, public_config
from data.dataset import canonical_json_bytes, sha256_file, template_sha256
from data.models import load_instance_yaml
from result.metrics import (
    evaluation_quality_metric,
    quality_metric_sha256,
    result_schema_version,
)
from utils import SAMPLED_EVALUATION_RNG_VERSION


PROVENANCE_SCHEMA_VERSION = "1.0.0"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def network_weights_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU storage bytes."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        descriptor = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _checkpoint_network_weights_sha256(path: str | Path) -> str:
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    state_dict = payload.get("network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint payload has no network state mapping")
    if any(
        not isinstance(tensor, torch.Tensor)
        for tensor in state_dict.values()
    ):
        raise TypeError("checkpoint network state must contain only tensors")
    return network_weights_sha256(state_dict)


def provenance_with_network_weights(
    provenance: Mapping[str, Any],
    weights_sha256: str,
) -> dict[str, Any]:
    updated = dict(provenance)
    updated["network_weights_sha256"] = str(weights_sha256)
    components = {
        key: value
        for key, value in updated.items()
        if key not in {
            "provenance_schema_version",
            "experiment_fingerprint_sha256",
        }
    }
    updated["experiment_fingerprint_sha256"] = _sha256(
        canonical_json_bytes(components)
    )
    return updated


def _normalized_source_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def source_files(root: str | Path = PROJECT_ROOT) -> list[Path]:
    base = Path(root).resolve()
    selected = set(base.glob("*.py"))
    for package in ("agent", "configs", "data", "environment"):
        directory = base / package
        if directory.is_dir():
            selected.update(directory.rglob("*.py"))
    result_directory = base / "result"
    if result_directory.is_dir():
        selected.update(result_directory.glob("*.py"))
    return sorted(
        (path for path in selected if path.is_file()),
        key=lambda path: path.relative_to(base).as_posix(),
    )


def source_state_snapshot(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    base = Path(root).resolve()
    digest = hashlib.sha256()
    paths: list[str] = []
    for path in source_files(base):
        relative = path.relative_to(base).as_posix()
        content = _normalized_source_bytes(path)
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        paths.append(relative)
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "paths": paths,
    }


def effective_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = public_config(dict(config))
    return {
        "sha256": _sha256(canonical_json_bytes(normalized)),
        "config": normalized,
    }


def dataset_manifest_snapshot(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    return {
        "path": manifest_path.as_posix(),
        "sha256": _sha256(canonical_json_bytes(manifest)),
    }


def _fixed_instance_snapshot(config: Mapping[str, Any]) -> dict[str, Any] | None:
    configured = config.get("paths", {}).get("fixed_instance")
    if configured is None:
        return None
    path = project_path(configured).resolve()
    if not path.is_file():
        return {"path": path.as_posix(), "missing": True}
    instance = load_instance_yaml(path)
    return {
        "path": path.as_posix(),
        "file_sha256": sha256_file(path),
        "template_sha256": template_sha256(instance),
    }


def git_state(root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    base = Path(root).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=base,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_provenance(
    config: Mapping[str, Any],
    *,
    dataset_manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    source = source_state_snapshot(root)
    effective = effective_config_snapshot(config)
    manifest = dataset_manifest_snapshot(dataset_manifest_path)
    fixed = _fixed_instance_snapshot(config)
    quality_metric = evaluation_quality_metric(config)
    checkpoint_hash = (
        sha256_file(checkpoint_path) if checkpoint_path is not None else None
    )
    checkpoint_protocol = None
    weights_hash = None
    if checkpoint_metadata is not None:
        checkpoint_protocol = checkpoint_metadata.get(
            "experiment_suite_version", "legacy"
        )
        weights_hash = checkpoint_metadata.get("network_weights_sha256")
    if checkpoint_path is not None:
        computed_weights_hash = _checkpoint_network_weights_sha256(
            checkpoint_path
        )
        if weights_hash is None:
            weights_hash = computed_weights_hash
        elif str(weights_hash) != computed_weights_hash:
            raise ValueError(
                "checkpoint metadata network_weights_sha256 does not match "
                "the checkpoint network state"
            )
    provenance: dict[str, Any] = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_state_sha256": source["sha256"],
        "source_file_count": source["file_count"],
        "effective_config_sha256": effective["sha256"],
        "dataset_manifest_sha256": None if manifest is None else manifest["sha256"],
        "fixed_instance_sha256": (
            None if fixed is None else fixed.get("file_sha256")
        ),
        "template_sha256": None if fixed is None else fixed.get("template_sha256"),
        "git": git_state(root),
        "checkpoint_sha256": checkpoint_hash,
        "network_weights_sha256": weights_hash,
        "checkpoint_protocol_version": checkpoint_protocol,
        "evaluator_protocol_version": config.get(
            "experiment_suite_version", "legacy"
        ),
        "result_schema_version": result_schema_version(config),
        "quality_metric_version": quality_metric["version"],
        "quality_metric_sha256": quality_metric_sha256(quality_metric),
        "sampled_rng_version": SAMPLED_EVALUATION_RNG_VERSION,
    }
    fingerprint_components = {
        key: value
        for key, value in provenance.items()
        if key not in {"provenance_schema_version"}
    }
    provenance["experiment_fingerprint_sha256"] = _sha256(
        canonical_json_bytes(fingerprint_components)
    )
    return provenance
