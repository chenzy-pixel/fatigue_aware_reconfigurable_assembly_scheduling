"""Audit current C0/E1 checkpoints under the E1 protocol v2 evaluator."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config, project_path
from data import load_dataset_split
from data.dataset import canonical_json_bytes
from environment import AssemblySchedulingEnv
from eval import evaluate_dataset, evaluate_dataset_parallel
from result import build_provenance
from result.io import write_csv, write_json


SAMPLING_SEEDS = (100011, 100012, 100013)
AUDIT_ARMS = {
    "c0": (
        "configs/v7/c0_v6_control.json",
        "result/runs/v7_2000_c0_seed11/accepted_checkpoint.pt",
    ),
    "e1": (
        "configs/v7/e1_context_exception.json",
        "result/runs/v7_2000_e1_seed11/accepted_checkpoint.pt",
    ),
}
TIMING_FIELDS = {
    "inference_time_seconds",
    "solve_time_seconds",
    "inference_time_per_decision_ms",
}


def _non_timing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in TIMING_FIELDS}
        for row in rows
    ]


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(_non_timing_rows(rows))).hexdigest()


def _assert_rng_unchanged(
    python_state,
    numpy_state,
    torch_state: torch.Tensor,
) -> None:
    if random.getstate() != python_state:
        raise RuntimeError("sampled evaluation modified Python RNG state")
    after_numpy = np.random.get_state()
    if (
        after_numpy[0] != numpy_state[0]
        or not np.array_equal(after_numpy[1], numpy_state[1])
        or after_numpy[2:] != numpy_state[2:]
    ):
        raise RuntimeError("sampled evaluation modified NumPy RNG state")
    if not torch.equal(torch.get_rng_state(), torch_state):
        raise RuntimeError("sampled evaluation modified global Torch RNG state")


def _load_agent(config: dict[str, Any], checkpoint: Path):
    dataset = load_dataset_split(config, "validation")
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(dataset[0].instance)
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    metadata = agent.load(checkpoint, load_optimizer=False)
    return agent, metadata, dataset


def audit_arm(
    arm: str,
    config_path: str,
    checkpoint_value: str,
    *,
    instance_limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"audit arm={arm}: loading checkpoint", flush=True)
    config = deepcopy(load_config(config_path))
    config["seed"] = 11
    checkpoint = project_path(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    agent, checkpoint_metadata, dataset = _load_agent(config, checkpoint)
    count = len(dataset) if instance_limit is None else int(instance_limit)
    if count < 10:
        raise ValueError("the 1/10 parallel audit requires at least 10 instances")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    greedy_rows, _, _, greedy = evaluate_dataset(
        config,
        dataset_name="validation",
        policy_name="ppo",
        ppo_agent=agent,
        instance_limit=count,
        decode_mode="greedy",
    )
    print(f"audit arm={arm}: greedy complete", flush=True)
    sampled_results: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = [
        {"arm": arm, "mode": "greedy_serial", **row}
        for row in greedy_rows
    ]
    template = dataset[0].instance
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=count,
        worker_count=10,
    ) as runner:
        for sampling_seed in SAMPLING_SEEDS:
            print(
                f"audit arm={arm}: sampled seed={sampling_seed} serial",
                flush=True,
            )
            serial_rows, _, _, serial = evaluate_dataset(
                config,
                dataset_name="validation",
                policy_name="ppo",
                ppo_agent=agent,
                instance_limit=count,
                decode_mode="sampled",
                sampling_seed=sampling_seed,
            )
            serial_non_timing = _non_timing_rows(serial_rows)
            parallel_results: dict[str, Any] = {}
            for parallel_envs in (1, 10):
                print(
                    f"audit arm={arm}: sampled seed={sampling_seed} "
                    f"parallel_envs={parallel_envs}",
                    flush=True,
                )
                config["training"][
                    "validation_parallel_envs"
                ] = parallel_envs
                parallel_rows, parallel = evaluate_dataset_parallel(
                    config,
                    dataset_name="validation",
                    ppo_agent=agent,
                    runner=runner,
                    instance_limit=count,
                    decode_mode="sampled",
                    sampling_seed=sampling_seed,
                )
                if _non_timing_rows(parallel_rows) != serial_non_timing:
                    raise RuntimeError(
                        f"{arm} seed {sampling_seed} differs at "
                        f"parallel_envs={parallel_envs}"
                    )
                parallel_results[str(parallel_envs)] = {
                    "aggregate": parallel,
                    "non_timing_rows_sha256": _rows_sha256(parallel_rows),
                    "action_trace_sha256": [
                        row["action_trace_sha256"] for row in parallel_rows
                    ],
                }
                audit_rows.extend(
                    {
                        "arm": arm,
                        "mode": f"sampled_parallel_{parallel_envs}",
                        "sampling_seed": sampling_seed,
                        **row,
                    }
                    for row in parallel_rows
                )
            sampled_results[str(sampling_seed)] = {
                "serial": serial,
                "serial_non_timing_rows_sha256": _rows_sha256(serial_rows),
                "serial_action_trace_sha256": [
                    row["action_trace_sha256"] for row in serial_rows
                ],
                "parallel": parallel_results,
                "parallel_1_10_exact_match": True,
            }
            audit_rows.extend(
                {
                    "arm": arm,
                    "mode": "sampled_serial",
                    "sampling_seed": sampling_seed,
                    **row,
                }
                for row in serial_rows
            )
            print(
                f"audit arm={arm}: sampled seed={sampling_seed} exact",
                flush=True,
            )
    _assert_rng_unchanged(python_state, numpy_state, torch_state)
    provenance = build_provenance(
        config,
        dataset_manifest_path=dataset.manifest_path,
        checkpoint_path=checkpoint,
        checkpoint_metadata=checkpoint_metadata,
    )
    return {
        "arm": arm,
        "config_path": str(project_path(config_path)),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "instance_count": count,
        "greedy": greedy,
        "sampled": sampled_results,
        "sampled_global_rng_unchanged": True,
        "provenance": provenance,
    }, audit_rows


def run_audit(*, instance_limit: int | None = None) -> Path:
    root = project_path("result/audits")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".e1_protocol_v2_{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        arm_results: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for arm, (config_path, checkpoint) in AUDIT_ARMS.items():
            result, arm_rows = audit_arm(
                arm,
                config_path,
                checkpoint,
                instance_limit=instance_limit,
            )
            arm_results[arm] = result
            rows.extend(arm_rows)
        payload = {
            "audit_protocol_version": "v7_e1_protocol_v2",
            "result_schema_version": "4.1.0",
            "sampling_seeds": list(SAMPLING_SEEDS),
            "parallel_envs": [1, 10],
            "all_checks_passed": True,
            "arms": arm_results,
        }
        write_json(staging / "audit.json", payload)
        write_csv(staging / "instance_metrics.csv", rows)
        name = f"{datetime.now():%Y%m%d_%H%M%S}_c0_e1_validation"
        final = root / name
        staging.replace(final)
        return final
    except BaseException:
        failure = staging / "FAILED.txt"
        failure.write_text("audit failed; staging preserved\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-limit", type=int)
    args = parser.parse_args()
    output = run_audit(instance_limit=args.instance_limit)
    print(json.dumps({"audit": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
