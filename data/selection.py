"""Stable, strategy-independent instance subsets and explicit evaluation indices."""

from __future__ import annotations

from collections import Counter
import operator
from typing import Any, Sequence

from data.dataset import InstanceDataset, canonical_json_bytes, sha256_bytes
from data.distribution import PRESSURE_TYPES, stable_seed, weighted_labels


def resolve_instance_indices(dataset: InstanceDataset, *, instance_indices: Sequence[int] | None = None,
                             instance_limit: int | None = None, instance_offset: int | None = None) -> list[int]:
    if instance_indices is not None:
        if instance_limit is not None or instance_offset is not None:
            raise ValueError("instance_indices and offset/limit are mutually exclusive")
        indices = [operator.index(index) for index in instance_indices]
    else:
        offset = 0 if instance_offset is None else operator.index(instance_offset)
        count = len(dataset) - offset if instance_limit is None else operator.index(instance_limit)
        indices = list(range(offset, offset + count))
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("instance selection must be nonempty, unique, and within the dataset")
    return indices


def subset_snapshot(dataset: InstanceDataset, indices: Sequence[int], *, role: str) -> dict[str, Any]:
    payload = {
        "selection_version": "stratified_v2",
        "dataset_manifest_sha256": sha256_bytes(canonical_json_bytes(dataset.manifest)),
        "instance_indices": list(indices),
        "files": [dataset.manifest["files"][index] for index in indices],
    }
    return {**payload, "role": role, "subset_sha256": sha256_bytes(canonical_json_bytes(payload))}


def select_validation_subsets(dataset: InstanceDataset, weights: dict[str, float], *,
                              target_count: int = 50, diagnostic_count: int = 49) -> dict[str, dict[str, Any]]:
    if diagnostic_count < 0:
        raise ValueError("diagnostic_count must be nonnegative")
    digest = sha256_bytes(canonical_json_bytes(dataset.manifest))
    groups = {name: [] for name in PRESSURE_TYPES}
    for index, record in enumerate(dataset):
        groups[record.metadata["pressure_type"]].append(index)
    chosen: set[int] = set()
    selections = {}
    for role, count, distribution in (
        ("target", target_count, weights),
        ("diagnostic", diagnostic_count, {name: 1.0 for name in PRESSURE_TYPES}),
    ):
        if count == 0:
            continue
        quotas = Counter(weighted_labels(count, distribution))
        indices = []
        for name in PRESSURE_TYPES:
            candidates = sorted((index for index in groups[name] if index not in chosen),
                                key=lambda index: (stable_seed(digest, dataset.manifest["files"][index]["seed"], role), index))
            if len(candidates) < quotas[name]:
                raise ValueError(f"insufficient {name} instances for {role} quota {quotas[name]}")
            indices.extend(candidates[:quotas[name]])
        indices.sort()
        chosen.update(indices)
        selections[role] = {**subset_snapshot(dataset, indices, role=role), "pressure_counts": dict(quotas)}
    return selections
