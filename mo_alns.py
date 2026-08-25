"""Run the environment-decoded MO-ALNS baseline on a persisted dataset split."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.mo_alns import MOALNSSolver, decode_solution
from agent.mo_alns.types import preference_key
from configs import load_config, project_path
from data import load_dataset_split
from data.dataset import PERSISTED_SPLITS, validate_algorithm_seed
from environment import CANONICAL_PREFERENCE, PreferenceVector, simplex_lattice
from environment.types import proxy_return_from_metrics
from eval import build_evaluation_row
from result import (
    aggregate_evaluation_rows,
    build_provenance,
    create_run_directory,
    dataset_manifest_snapshot,
    evaluation_quality_metric,
)
from result.io import write_config, write_csv, write_json


PROTOCOL_VERSION = "mo_alns_solver_budget_v1"
RESULT_SCHEMA_VERSION = "5.1.0"


def mo_alns_preference_grid() -> tuple[PreferenceVector, ...]:
    points = tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
    if len(points) != 22:
        raise RuntimeError("MO-ALNS full grid must contain 22 preferences")
    return points


def _candidate_id(preference: PreferenceVector) -> str:
    return "mo_alns_w_" + preference_key(preference)


def _objective_replay_matches(first: tuple[float, float, float], second: tuple[float, float, float]) -> bool:
    return all(abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right)) for left, right in zip(first, second, strict=True))


def _run_record(
    config: dict[str, Any],
    record: Any,
    dataset_name: str,
    algorithm_seed: int,
    preferences: tuple[PreferenceVector, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Top-level process worker: solve one instance deterministically."""

    worker_config = deepcopy(config)
    worker_config["seed"] = int(algorithm_seed)
    solver = MOALNSSolver(
        worker_config,
        algorithm_seed=algorithm_seed,
        dataset_name=dataset_name,
    )
    grid = solver.solve_grid(record.instance, preferences)
    searches = {preference_key(search.preference): search for search in grid.searches}
    quality_metric = evaluation_quality_metric(worker_config)
    rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    reconfigurations: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []

    for preference in preferences:
        key = preference_key(preference)
        endpoint = grid.endpoints[key]
        replay = decode_solution(
            worker_config,
            record.instance,
            endpoint.solution,
            preference,
            capture_logs=True,
        )
        if not _objective_replay_matches(endpoint.objectives, replay.objectives):
            raise RuntimeError(
                f"final replay changed objectives for {record.instance.instance_id}/{key}"
            )
        if endpoint.action_trace_sha256 != replay.action_trace_sha256:
            raise RuntimeError(
                f"final replay changed action trace for {record.instance.instance_id}/{key}"
            )
        search = searches[key]
        metrics = dict(replay.metrics)
        metrics["preference"] = preference.as_dict()
        metrics["solve_time_seconds"] = search.search_time_seconds
        metrics["mo_alns_endpoint_replay_seconds"] = replay.metrics["solve_time_seconds"]
        metrics["mo_alns_environment_evaluations"] = search.environment_evaluations
        metrics["mo_alns_cache_hits"] = search.cache_hits
        metrics["mo_alns_proposal_count"] = search.proposal_count
        metrics["mo_alns_archive_size"] = len(grid.archive)
        metrics["mo_alns_initial_best_tchebycheff"] = search.initial_best_tchebycheff
        metrics["mo_alns_tchebycheff"] = replay.tchebycheff
        metrics["feasibility_proxy_return"] = proxy_return_from_metrics(
            metrics,
            worker_config["reward"],
            "feasibility",
            preference=preference,
        )
        row = build_evaluation_row(record, metrics, worker_config["reward"], quality_metric)
        candidate_id = _candidate_id(preference)
        rows.append(
            {
                **row,
                "arm": "mo_alns",
                "algorithm_seed": int(algorithm_seed),
                "dataset": dataset_name,
                "candidate_id": candidate_id,
                "candidate_source": "pareto_archive_endpoint",
                "endpoint_preference_key": key,
                "source_search_preference_key": preference_key(endpoint.preference),
                "environment_evaluation_count": search.environment_evaluations,
                "cache_hit_count": search.cache_hits,
                "proposal_count": search.proposal_count,
                "archive_size": len(grid.archive),
                "tchebycheff": replay.tchebycheff,
                "initial_best_tchebycheff": search.initial_best_tchebycheff,
                "replay_verified": True,
            }
        )
        schedules.extend(
            {
                "instance_id": record.instance.instance_id,
                "candidate_id": candidate_id,
                "endpoint_preference_key": key,
                **value,
            }
            for value in replay.schedule_log
        )
        reconfigurations.extend(
            {
                "instance_id": record.instance.instance_id,
                "candidate_id": candidate_id,
                "endpoint_preference_key": key,
                **value,
            }
            for value in replay.reconfiguration_log
        )
        search_log.extend(
            {
                "instance_id": record.instance.instance_id,
                "algorithm_seed": int(algorithm_seed),
                "dataset": dataset_name,
                "preference_key": key,
                **value,
            }
            for value in search.search_log
        )
        operator_rows.append(
            {
                "instance_id": record.instance.instance_id,
                "algorithm_seed": int(algorithm_seed),
                "dataset": dataset_name,
                "preference_key": key,
                "environment_evaluations": search.environment_evaluations,
                "cache_hits": search.cache_hits,
                "proposals": search.proposal_count,
                "search_time_seconds": search.search_time_seconds,
                "initial_best_tchebycheff": search.initial_best_tchebycheff,
                **search.operator_statistics,
            }
        )

    archive_rows = [
        {
            "instance_id": record.instance.instance_id,
            "algorithm_seed": int(algorithm_seed),
            "dataset": dataset_name,
            "solution_digest": candidate.solution_digest,
            "action_trace_sha256": candidate.action_trace_sha256,
            "flow_time_objective": candidate.objectives[0],
            "reconfiguration_cost": candidate.objectives[1],
            "worker_load_variance": candidate.objectives[2],
            "w_flow": candidate.preference.flow,
            "w_cost": candidate.preference.cost,
            "w_variance": candidate.preference.variance,
            "source_preference_key": preference_key(candidate.preference),
        }
        for candidate in grid.archive
    ]
    return {
        "rows": rows,
        "schedules": schedules,
        "reconfigurations": reconfigurations,
        "archive": archive_rows,
        "search_log": search_log,
        "operators": operator_rows,
    }


def run_mo_alns_dataset(
    config: Mapping[str, Any],
    *,
    dataset_name: str,
    algorithm_seed: int,
    instance_limit: int | None = None,
    preferences: Sequence[PreferenceVector] | None = None,
    parallel_envs: int | None = None,
) -> dict[str, Any]:
    """Run all requested preferences for a persisted split and return artifacts."""

    if dataset_name not in PERSISTED_SPLITS:
        raise ValueError(f"unknown persisted split {dataset_name!r}")
    effective_config = deepcopy(dict(config))
    effective_config["seed"] = validate_algorithm_seed(effective_config, int(algorithm_seed))
    dataset = load_dataset_split(effective_config, dataset_name)
    count = len(dataset) if instance_limit is None else int(instance_limit)
    if count < 1 or count > len(dataset):
        raise ValueError(f"instance_limit must be in [1, {len(dataset)}]")
    records = [dataset[index] for index in range(count)]
    points = tuple(mo_alns_preference_grid() if preferences is None else preferences)
    if not points:
        raise ValueError("at least one preference is required")
    configured_workers = int(effective_config.get("mo_alns", {}).get("parallel_workers", 20))
    workers = configured_workers if parallel_envs is None else int(parallel_envs)
    workers = max(1, min(workers, len(records)))
    results: list[dict[str, list[dict[str, Any]]]] = []
    if workers == 1:
        results = [
            _run_record(effective_config, record, dataset_name, int(algorithm_seed), points)
            for record in records
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_record,
                    effective_config,
                    record,
                    dataset_name,
                    int(algorithm_seed),
                    points,
                )
                for record in records
            ]
            for future in as_completed(futures):
                results.append(future.result())
    rows = sorted(
        [row for result in results for row in result["rows"]],
        key=lambda value: (str(value["instance_id"]), str(value["candidate_id"])),
    )
    manifest = str(dataset.manifest_path)
    aggregate = aggregate_evaluation_rows(
        rows,
        dataset=dataset_name,
        policy="mo_alns",
        manifest=manifest,
        quality_metric=evaluation_quality_metric(effective_config),
        schema_version=RESULT_SCHEMA_VERSION,
    )
    aggregate.update(
        {
            "protocol": PROTOCOL_VERSION,
            "algorithm_seed": int(algorithm_seed),
            "preference_count_per_instance": len(points),
            "candidate_budget_per_preference": int(effective_config.get("mo_alns", {}).get("max_evaluations_per_preference", 300)),
            "candidate_budget_per_instance": len(points)
            * int(effective_config.get("mo_alns", {}).get("max_evaluations_per_preference", 300)),
            "dataset_manifest_sha256": dataset_manifest_snapshot(dataset.manifest_path)["sha256"],
            "parallel_envs": workers,
            "archive_entry_count": sum(len(result["archive"]) for result in results),
        }
    )
    return {
        "config": effective_config,
        "rows": rows,
        "schedules": [row for result in results for row in result["schedules"]],
        "reconfigurations": [row for result in results for row in result["reconfigurations"]],
        "archive": [row for result in results for row in result["archive"]],
        "search_log": [row for result in results for row in result["search_log"]],
        "operators": [row for result in results for row in result["operators"]],
        "aggregate": aggregate,
        "manifest_path": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MO-ALNS scheduling baseline")
    parser.add_argument("--config", default="configs/baselines/mo_alns.json")
    parser.add_argument("--dataset", choices=PERSISTED_SPLITS, required=True)
    parser.add_argument("--algorithm-seed", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--run-name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--canonical-only", action="store_true")
    group.add_argument("--preference", nargs=3, type=float, metavar=("FLOW", "COST", "VARIANCE"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = deepcopy(config)
        config.setdefault("mo_alns", {})["max_evaluations_per_preference"] = 8
        config["mo_alns"]["max_proposals_multiplier"] = 2
        args.instance_limit = 1 if args.instance_limit is None else args.instance_limit
    seed = int(config["seed"] if args.algorithm_seed is None else args.algorithm_seed)
    preferences: tuple[PreferenceVector, ...] | None
    if args.canonical_only:
        preferences = (PreferenceVector(*CANONICAL_PREFERENCE),)
    elif args.preference is not None:
        preferences = (PreferenceVector(*args.preference),)
    else:
        preferences = None
    result = run_mo_alns_dataset(
        config,
        dataset_name=args.dataset,
        algorithm_seed=seed,
        instance_limit=args.instance_limit,
        preferences=preferences,
        parallel_envs=args.parallel_envs,
    )
    run_directory = create_run_directory(
        project_path(result["config"]["paths"]["result_root"]),
        label=f"mo_alns_{args.dataset}",
        run_name=args.run_name,
    )
    metrics = dict(result["aggregate"])
    metrics["provenance"] = build_provenance(
        result["config"], dataset_manifest_path=result["manifest_path"]
    )
    write_config(run_directory, result["config"])
    write_json(run_directory / "metrics.json", metrics)
    write_csv(run_directory / "instance_metrics.csv", result["rows"])
    write_csv(run_directory / "candidates.csv", result["rows"])
    write_csv(run_directory / "schedule.csv", result["schedules"])
    write_csv(run_directory / "reconfigurations.csv", result["reconfigurations"])
    write_csv(run_directory / "pareto_archive.csv", result["archive"])
    write_csv(run_directory / "search_log.csv", result["search_log"])
    write_csv(run_directory / "operator_statistics.csv", result["operators"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results: {run_directory}")


if __name__ == "__main__":
    main()
