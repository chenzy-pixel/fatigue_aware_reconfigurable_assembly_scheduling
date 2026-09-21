from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from configs.normalization import build_normalization_manifest, write_immutable_manifest


OBJECTIVES = ("flow", "cost", "variance")
SEEDS = (11, 23, 37, 53, 71)
OBJECTIVE_FIELDS = {
    "flow": "mean_flow_time_objective",
    "cost": "mean_reconfiguration_cost",
    "variance": "mean_worker_load_variance",
}


def collect_specialist_checkpoints(runs_root: str | Path) -> list[dict[str, object]]:
    """Read the selected endpoint value from all 15 specialist checkpoints."""

    root = Path(runs_root)
    rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            checkpoint = root / f"v8_specialist_{objective}_seed{seed}" / "best_checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            spec = payload.get("network_spec", {})
            metadata = payload.get("metadata", {})
            if (
                int(spec.get("policy_head_version", 0)) != 8
                or int(spec.get("observation_schema_version", 0)) != 5
                or spec.get("expert_weight_parameterization") != "simplex_softplus_v8"
            ):
                raise ValueError(f"specialist checkpoint is not V8: {checkpoint}")
            if (
                metadata.get("objective_name") != objective
                or int(metadata.get("algorithm_seed", -1)) != seed
                or metadata.get("checkpoint_role") != "best"
                or int(metadata.get("validation_instance_limit", -1)) != 50
            ):
                raise ValueError(
                    f"specialist checkpoint metadata is inconsistent: {checkpoint}"
                )
            validation = metadata.get("validation") or {}
            raw_mean = validation.get(OBJECTIVE_FIELDS[objective])
            snapshot = metadata.get("validation_dataset_manifest") or {}
            validation_sha = snapshot.get("sha256")
            if raw_mean is None or not validation_sha:
                raise ValueError(
                    f"specialist checkpoint lacks validation provenance: {checkpoint}"
                )
            rows.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "raw_objective_mean": float(raw_mean),
                    "validation_dataset_sha256": str(validation_sha),
                    "validation_instance_offset": 0,
                    "validation_instance_count": 50,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable V8 normalization manifest from 15 selected "
            "specialist validation records."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--specialist-records")
    source.add_argument("--specialist-runs-root")
    parser.add_argument("--validation-dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.specialist_runs_root:
        rows = collect_specialist_checkpoints(args.specialist_runs_root)
    else:
        rows = json.loads(Path(args.specialist_records).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("specialist record input must be a JSON list")
    manifest = build_normalization_manifest(
        rows,
        validation_dataset_path=args.validation_dataset_manifest,
    )
    sha256 = write_immutable_manifest(args.output, manifest)
    print(json.dumps({"path": str(Path(args.output).resolve()), "sha256": sha256}))


if __name__ == "__main__":
    main()
