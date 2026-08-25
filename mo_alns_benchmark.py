"""Execute the formal multi-seed MO-ALNS manifest without touching E1/E2 runs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from configs import load_config, project_path
from mo_alns import PROTOCOL_VERSION, run_mo_alns_dataset
from result import build_provenance
from result.io import write_config, write_csv, write_json


def run_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    instance_limit: int | None = None,
    parallel_envs: int | None = None,
) -> dict[str, Any]:
    path = project_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if str(manifest.get("protocol")) != PROTOCOL_VERSION.replace("mo_alns", "e1_e2_mo_alns"):
        raise ValueError("manifest does not declare e1_e2_mo_alns_solver_budget_v1")
    config = load_config(manifest["config"])
    datasets = tuple(str(value) for value in manifest["datasets"])
    seeds = tuple(int(value) for value in manifest["algorithm_seeds"])
    if not datasets or not seeds:
        raise ValueError("manifest must contain non-empty datasets and algorithm_seeds")
    requested_limit = instance_limit if instance_limit is not None else manifest.get("instance_limit")
    requested_parallelism = parallel_envs if parallel_envs is not None else manifest.get("parallel_envs")
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    reconfigurations: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    operators: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset in datasets:
            result = run_mo_alns_dataset(
                config,
                dataset_name=dataset,
                algorithm_seed=seed,
                instance_limit=requested_limit,
                parallel_envs=requested_parallelism,
            )
            rows.extend(result["rows"])
            schedules.extend(result["schedules"])
            reconfigurations.extend(result["reconfigurations"])
            archives.extend(result["archive"])
            operators.extend(result["operators"])
            search_log.extend(result["search_log"])
            artifacts.append(result["aggregate"])
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_config(output, config)
    write_json(output / "run_manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "protocol": manifest["protocol"],
            "datasets": list(datasets),
            "algorithm_seeds": list(seeds),
            "run_count": len(artifacts),
            "artifacts": artifacts,
            "provenance": build_provenance(config),
            "e1_e2_candidates": manifest.get("e1_e2_candidates"),
        },
    )
    write_csv(output / "candidates.csv", rows)
    write_csv(output / "instance_metrics.csv", rows)
    write_csv(output / "schedule.csv", schedules)
    write_csv(output / "reconfigurations.csv", reconfigurations)
    write_csv(output / "pareto_archive.csv", archives)
    write_csv(output / "search_log.csv", search_log)
    write_csv(output / "operator_statistics.csv", operators)
    return {"output": str(output), "run_count": len(artifacts), "candidate_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal MO-ALNS benchmark manifest")
    parser.add_argument("--manifest", default="configs/baselines/mo_alns_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--parallel-envs", type=int)
    args = parser.parse_args()
    summary = run_manifest(
        args.manifest,
        args.output_dir,
        instance_limit=args.instance_limit,
        parallel_envs=args.parallel_envs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
