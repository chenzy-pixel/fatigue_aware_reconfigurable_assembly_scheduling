"""Expand schema-5 input weights for schema-6 order time context."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch

from environment import CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE
from .network_v8 import assert_network_config_matches_spec, infer_checkpoint_network_spec


def migrate_time_context_checkpoint(
    checkpoint: Mapping[str, Any], target_spec: Mapping[str, Any], parameter_names: Sequence[str],
) -> dict[str, Any]:
    saved = infer_checkpoint_network_spec(checkpoint)
    if saved["observation_schema_version"] != 5:
        return dict(checkpoint)
    spec = deepcopy(saved)
    spec["feature_dimensions"]["order"] += 1
    spec["edge_feature_dimensions"][CAPABLE_EDGE] += 1
    spec["edge_feature_dimensions"][SERVICE_CANDIDATE_EDGE] += 2
    spec["action_set_feature_names"] = tuple(spec["action_set_feature_names"]) + tuple(
        target_spec["action_set_feature_names"][-2:]
    )
    for field in ("observation_schema_version", "time_context_version", "time_context_feature_schema"):
        spec[field] = deepcopy(target_spec[field])
    # Check every original architecture/normalization contract before migration.
    assert_network_config_matches_spec(target_spec, spec)
    widths = {
        "node_projectors.order.0.weight": 1,
        "production_edge_encoder.0.weight": 1,
        "worker_edge_encoder.0.weight": 2,
        "wait_feature_encoder.0.weight": 2,
    }
    for name in checkpoint["network"]:
        if name.startswith("message_layers."):
            if name.endswith("transforms.operation__capable_on__machine.weight"):
                widths[name] = 1
            elif name.endswith("transforms.machine__service_candidate__worker.weight"):
                widths[name] = 2

    def expand(name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in widths:
            return value
        if value.ndim != 2:
            raise ValueError(f"invalid checkpoint input weight: {name}")
        return torch.cat((value, value.new_zeros((value.shape[0], widths[name]))), dim=1)

    migrated = dict(checkpoint)
    migrated["network"] = {name: expand(name, value) for name, value in checkpoint["network"].items()}
    migrated["network_spec"] = spec
    if "optimizer" in checkpoint:
        optimizer = deepcopy(checkpoint["optimizer"])
        ids = [value for group in optimizer["param_groups"] for value in group["params"]]
        if len(ids) != len(parameter_names):
            raise ValueError("checkpoint optimizer parameter order is incompatible")
        for identifier, name in zip(ids, parameter_names, strict=True):
            for field, value in optimizer["state"].get(identifier, {}).items():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    optimizer["state"][identifier][field] = expand(name, value)
        migrated["optimizer"] = optimizer
    metadata = dict(checkpoint.get("metadata", {}))
    metadata["checkpoint_load_migration"] = {
        "source_observation_schema": 5, "target_observation_schema": 6,
        "time_context_version": target_spec["time_context_version"],
        "new_input_weight_initialization": "zeros",
    }
    migrated["metadata"] = metadata
    return migrated
