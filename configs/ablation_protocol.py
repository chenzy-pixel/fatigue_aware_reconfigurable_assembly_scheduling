"""Matched V8 ablation run identities and control-variable checks."""
from __future__ import annotations

import json
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .config import load_config, project_path
from .formal_preferences import formal_preferences
from .network_contract import MESSAGE_IDENTITY_FIELDS

ABLATION_MANIFEST = "configs/manifests/ablation_seed11.json"
ABLATION_PROTOCOL = "v8_matched_ablation_failure2_frozen_v1"


def training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Comparable algorithm, problem, normalization, and selection settings."""
    from agent.ppo.network import normalize_network_config
    from data.dataset import template_sha256
    from data.models import load_instance_yaml

    training = config["training"]
    scalarizer = dict(config["objective_scalarizer"])
    scalarizer.pop("normalization_manifest", None)  # Location is not identity.
    return deepcopy({
        "algorithm_seed": int(config["seed"]),
        "suite": config["experiment_suite_version"],
        "template_sha256": template_sha256(load_instance_yaml(project_path(config["paths"]["fixed_instance"]))),
        "dataset": config["dataset"],
        "generator": config["generator"],
        "environment": {**config["environment"], "fatigue_mode": config["environment"].get("fatigue_mode", "full")},
        "reward": config["reward"],
        "scalarizer": scalarizer,
        "preference": config["preference"],
        "network": normalize_network_config(config["network"]),
        "ppo": config["ppo"],
        "episodes": int(training["episodes"]),
        "parallel_envs": int(training["parallel_envs"]),
        "validation_parallel_envs": int(training["validation_parallel_envs"]),
        "validation_selection": training.get("validation_selection"),
        "policy_execution_version": training.get("policy_execution_version"),
        "policy_precision": training.get("policy_precision"),
        "online_instances": training.get("online_instances"),
        "initial_checkpoint": training.get("initial_checkpoint"),
        "episodes_per_update": int(training.get("episodes_per_update", training["parallel_envs"])),
        "validation_split": training["validation_split"],
        "validation_interval": training["validation_interval_episodes"],
        "validation_instance_limit": training["validation_instance_limit"],
        "formal_evaluation": training["formal_evaluation"],
        "validation_preferences": [p.as_tuple() for p in formal_preferences(dict(config), "validation")],
        "final_preferences": [p.as_tuple() for p in formal_preferences(dict(config), "final_test")],
        "validation_control": training["validation_control"],
        "forced_action_compression": training.get("forced_action_compression", False),
        "worker_local_physical_forced_actions": training.get("worker_local_physical_forced_actions", True),
    })


def assert_matching_training_config(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    first, second = training_contract(saved), training_contract(expected)
    differences = [key for key in first if first[key] != second[key]]
    if differences:
        raise ValueError("training configuration mismatch: " + ", ".join(differences))


def assert_paired_configs(baseline: Mapping[str, Any], variant: Mapping[str, Any], changed_field: str) -> None:
    first, second = training_contract(baseline), training_contract(variant)
    if changed_field == "fatigue_mode":
        if (first["environment"].pop("fatigue_mode"), second["environment"].pop("fatigue_mode")) != ("full", "neutral"):
            raise ValueError("fatigue pair must compare full and neutral environments")
    elif changed_field in {"encoder_variant", "actor_head_variant"}:
        expected = (("hetero_gnn", "node_mlp_pool") if changed_field == "encoder_variant"
                    else ("objective_experts", "shared_preference"))
        if (first["network"].pop(changed_field), second["network"].pop(changed_field)) != expected:
            raise ValueError(f"unexpected structural pair: {changed_field}")
        if changed_field == "encoder_variant":
            first["network"].pop("encoder_type")
            second["network"].pop("encoder_type")
            for name in MESSAGE_IDENTITY_FIELDS:
                first["network"].pop(name)
                second["network"].pop(name)
        else:
            first["network"].pop("expert_weight_parameterization")
            second["network"].pop("expert_weight_parameterization")
    else:
        raise ValueError(f"unknown ablation field: {changed_field}")
    differences = [key for key in first if first[key] != second[key]]
    if differences:
        raise ValueError("unmatched ablation control variables: " + ", ".join(differences))


def load_ablation_manifest(path: str | Path = ABLATION_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(project_path(path).read_text(encoding="utf-8"))
    if manifest.get("protocol") != ABLATION_PROTOCOL:
        raise ValueError("unsupported ablation protocol")
    for pair in manifest["pairs"]:
        configs = [load_config(manifest["runs"][pair[label]]["config"]) for label in ("baseline", "variant")]
        if any(int(c["seed"]) != int(manifest["algorithm_seed"]) for c in configs):
            raise ValueError("ablation manifest algorithm seed mismatch")
        if any(c["reward"]["terminal_failure_penalty"] != 2.0 or
               c["objective_scalarizer"]["scale_source"] != "frozen_manifest" for c in configs):
            raise ValueError("ablation protocol requires failure penalty 2 and frozen scales")
        assert_paired_configs(*configs, pair["changed_field"])
    return manifest


def validate_training_run(directory: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Reject partial, historical, or mismatched checkpoints before evaluation."""
    import torch
    from .runtime import assert_checkpoint_fatigue_mode
    from agent.ppo.network import infer_checkpoint_network_spec, normalize_network_config
    from result.provenance import dataset_manifest_snapshot, network_weights_sha256

    saved = load_config(directory / "config.json")
    assert_matching_training_config(saved, expected)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if int(summary["episodes"]) != int(expected["training"]["episodes"]):
        raise ValueError("run did not complete the configured episode budget")
    payload = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    assert_matching_training_config(metadata["effective_config"], expected)
    assert_checkpoint_fatigue_mode(metadata, expected)
    if not metadata.get("selection_eligible", True) or metadata.get("validation_evaluation_complete") is False:
        raise ValueError("checkpoint selection came from an incomplete evaluation")
    if metadata.get("checkpoint_role") != "best" or not summary["checkpoint_selection"]["has_best"]:
        raise ValueError("run has no selected best checkpoint")
    if metadata.get("network_weights_sha256") != network_weights_sha256(payload["network"]):
        raise ValueError("checkpoint network state hash mismatch")
    if int(metadata["checkpoint_episode"]) != int(summary["checkpoint_selection"]["best_episode"]):
        raise ValueError("checkpoint episode differs from run selection")
    spec = infer_checkpoint_network_spec(payload)
    if normalize_network_config(spec) != normalize_network_config(expected["network"]):
        raise ValueError("checkpoint network specification differs from expected configuration")
    manifest_path = project_path(expected["paths"]["manifests_root"]) / expected["training"]["validation_split"] / "manifest.json"
    if metadata["validation_dataset_manifest"]["sha256"] != dataset_manifest_snapshot(manifest_path)["sha256"]:
        raise ValueError("training validation dataset manifest changed")
    return metadata


def validate_evaluation_run(directory: Path, expected: Mapping[str, Any], checkpoint: Path) -> None:
    """Validate all test cells and fingerprints before reusing an evaluation."""
    from data.dataset import load_dataset_split, sha256_file
    from result.provenance import dataset_manifest_snapshot, effective_config_snapshot
    from utils import configured_formal_evaluation_sampling_seeds, derive_evaluation_sampling_seed

    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    effective = load_config(directory / "config.json")
    assert_matching_training_config(effective, expected)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not metrics.get("evaluation_complete", True) or metrics.get("sampling_truncated_count", 0):
        raise ValueError("incomplete evaluations cannot be used for matched ablations")
    provenance = metrics.get("provenance", {})
    if provenance.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("evaluation uses another checkpoint")
    if provenance.get("effective_config_sha256") != effective_config_snapshot(raw)["sha256"]:
        raise ValueError("evaluation effective configuration hash mismatch")
    dataset = load_dataset_split(dict(expected), "test")
    manifest_hash = dataset_manifest_snapshot(dataset.manifest_path)["sha256"]
    if metrics.get("dataset") != "test" or provenance.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("evaluation test manifest changed")
    seeds = configured_formal_evaluation_sampling_seeds(dict(expected), "final_test")
    preferences = formal_preferences(dict(expected), "final_test")
    ids = {record.instance.instance_id for record in dataset}
    required = {(instance_id, point.key, repeat) for instance_id in ids for point in preferences for repeat in range(len(seeds))}
    with (directory / "instance_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = {(r["instance_id"], r["preference_key"], int(r["sampling_repeat"])) for r in rows}
    if len(observed) != len(rows) or observed != required:
        raise ValueError("evaluation has duplicate or incomplete test cells")
    universal = expected["preference"]["quality"]["mode"] == "universal_sobol_v1"
    for row in rows:
        if str(row.get("sampling_truncated", row.get("truncated", "False"))).lower() in {"true", "1"}:
            raise ValueError("externally truncated test cells cannot be used for matched ablations")
        seed = seeds[int(row["sampling_repeat"])]
        key = row["preference_key"] if universal else None
        if int(row["sampling_seed"]) != seed or int(row["derived_sampling_seed"]) != derive_evaluation_sampling_seed(seed, row["instance_id"], key):
            raise ValueError("evaluation sampling identity mismatch")
