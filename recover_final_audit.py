"""Recover the final fixed-manifest audit for a completed training run.

This entry point is intentionally audit-only: it never updates model weights,
renames checkpoints, or rewrites the original training failure evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from configs import project_path
from data import load_dataset_split
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from result import dataset_manifest_snapshot, effective_config_snapshot
from result.io import write_json
from train import (
    _assert_single_objective_checkpoint_evaluation,
    _reevaluate_checkpoint_with_parallel_runner,
    _validate_single_objective_validation_protocol,
)
from training import TrainingPhaseController
from utils import set_seed


RECOVERY_AUDIT_VERSION = "final_checkpoint_audit_recovery_v1"
OUTPUT_NAME = "recovered_final_checkpoint_audit.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("accepted checkpoint metadata is missing")
    return dict(metadata)


def _require_equal_identity(
    name: str, expected: object, observed: object
) -> None:
    if expected is None or observed is None or str(expected) != str(observed):
        raise RuntimeError(
            f"{name} identity mismatch: expected={expected}, observed={observed}"
        )


def recover_final_audit(
    run_directory_value: str | Path,
    *,
    parallel_envs: int | None = None,
    force: bool = False,
) -> Path:
    run_directory = Path(run_directory_value)
    if not run_directory.is_absolute():
        run_directory = project_path(run_directory)
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)

    output_path = run_directory / OUTPUT_NAME
    if output_path.exists() and not force:
        raise FileExistsError(
            f"recovery audit already exists; pass --force to replace it: {output_path}"
        )

    config_path = run_directory / "config.json"
    accepted_checkpoint = run_directory / "accepted_checkpoint.pt"
    best_checkpoint = run_directory / "best_checkpoint.pt"
    for path in (config_path, accepted_checkpoint, best_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _read_json(config_path)
    set_seed(int(config["seed"]))
    validation_limit = int(config["training"]["validation_instance_limit"])
    _validate_single_objective_validation_protocol(
        config,
        smoke=False,
        validation_limit=validation_limit,
    )
    worker_count = (
        int(config["training"]["validation_parallel_envs"])
        if parallel_envs is None
        else int(parallel_envs)
    )
    if worker_count < 1:
        raise ValueError("parallel_envs must be positive")

    accepted_sha256 = _sha256_file(accepted_checkpoint)
    best_sha256 = _sha256_file(best_checkpoint)
    if accepted_sha256 != best_sha256:
        raise RuntimeError("accepted and best checkpoint hashes diverged")

    metadata = _checkpoint_metadata(accepted_checkpoint)
    if metadata.get("formal_eligible") is not True:
        raise RuntimeError("accepted checkpoint is not marked formal_eligible")
    controller = TrainingPhaseController.from_config(config)
    expected_target = controller.single_objective_name
    _require_equal_identity(
        "single-objective target",
        expected_target,
        metadata.get("single_objective_name"),
    )
    expected_audit_value = metadata.get(
        "accepted_single_objective_audit_value"
    )
    if expected_audit_value is None:
        raise RuntimeError(
            "accepted checkpoint has no accepted_single_objective_audit_value"
        )
    controller.accepted_single_objective_audit_value = float(
        expected_audit_value
    )

    validation_split = str(config["training"]["validation_split"])
    manifest_path = (
        project_path(config["paths"]["manifests_root"])
        / validation_split
        / "manifest.json"
    )
    current_config = effective_config_snapshot(config)
    current_manifest = dataset_manifest_snapshot(manifest_path)
    if current_manifest is None:
        raise FileNotFoundError(manifest_path)
    _require_equal_identity(
        "effective config",
        metadata.get("effective_config_sha256"),
        current_config["sha256"],
    )
    _require_equal_identity(
        "validation manifest",
        metadata.get("dataset_manifest_sha256"),
        current_manifest["sha256"],
    )

    validation_dataset = load_dataset_split(config, validation_split)
    audit_limit = controller.single_objective_audit_instance_limit
    if len(validation_dataset) < audit_limit:
        raise RuntimeError(
            f"validation dataset has {len(validation_dataset)} instances; "
            f"the final audit requires {audit_limit}"
        )
    environment = AssemblySchedulingEnv(config)
    bootstrap_observation = environment.reset(
        validation_dataset[0].instance
    )
    template = load_instance_yaml(
        project_path(config["paths"]["fixed_instance"])
    )
    started_at = datetime.now(timezone.utc).isoformat()
    original_failure_path = run_directory / "failure.json"
    original_failure = (
        _read_json(original_failure_path)
        if original_failure_path.is_file()
        else None
    )
    common = {
        "version": RECOVERY_AUDIT_VERSION,
        "run_directory": str(run_directory),
        "started_at": started_at,
        "original_failure": original_failure,
        "config": str(config_path),
        "accepted_checkpoint": str(accepted_checkpoint),
        "accepted_checkpoint_sha256": accepted_sha256,
        "best_checkpoint_sha256": best_sha256,
        "parallel_envs": worker_count,
        "audit_instance_limit": audit_limit,
        "checkpoint_source_state_sha256": metadata.get(
            "source_state_sha256"
        ),
        "effective_config_sha256": current_config["sha256"],
        "dataset_manifest_sha256": current_manifest["sha256"],
        "network_weights_sha256": metadata.get("network_weights_sha256"),
    }
    evaluation: dict[str, object] | None = None
    try:
        evaluation = _reevaluate_checkpoint_with_parallel_runner(
            config,
            checkpoint=accepted_checkpoint,
            bootstrap_observation=bootstrap_observation,
            dataset_name=validation_split,
            instance_limit=audit_limit,
            sampling_seeds=[],
            greedy_only=True,
            template=template,
            episode_count=int(config["training"]["episodes"]),
            parallel_worker_count=worker_count,
        )
        evaluation_provenance = evaluation.get("provenance")
        if not isinstance(evaluation_provenance, dict):
            raise RuntimeError("final evaluation provenance is missing")
        _require_equal_identity(
            "evaluation config",
            metadata.get("effective_config_sha256"),
            evaluation_provenance.get("effective_config_sha256"),
        )
        _require_equal_identity(
            "evaluation validation manifest",
            metadata.get("dataset_manifest_sha256"),
            evaluation_provenance.get("dataset_manifest_sha256"),
        )
        _require_equal_identity(
            "evaluation network weights",
            metadata.get("network_weights_sha256"),
            evaluation_provenance.get("network_weights_sha256"),
        )
        _assert_single_objective_checkpoint_evaluation(
            controller,
            evaluation,
        )
    except BaseException as error:
        write_json(
            output_path,
            {
                **common,
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "evaluation": evaluation,
            },
        )
        raise

    write_json(
        output_path,
        {
            **common,
            "status": "passed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "evaluation": evaluation,
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = recover_final_audit(
        args.run_directory,
        parallel_envs=args.parallel_envs,
        force=args.force,
    )
    print(json.dumps({"recovered_final_audit": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
