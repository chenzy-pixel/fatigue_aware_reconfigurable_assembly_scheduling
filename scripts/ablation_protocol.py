"""Matched ablation run discovery and seed-specific training/evaluation entry points."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from configs import load_config, project_path
from configs.ablation_protocol import validate_training_run, validate_evaluation_run
from environment.observation_schema import OBSERVATION_SCHEMA_VERSION
from environment.time_context import TIME_CONTEXT_VERSION
from data.dataset import sha256_file, validate_algorithm_seed
from result.io import write_json
from result.provenance import dataset_manifest_snapshot

ROOT = Path(__file__).resolve().parents[1]
ROLE_CONFIGS = {
    "universal": "configs/default.json",
    "no_graph": "configs/ablations/no_graph.json",
    "full_flow": "configs/e1/single_flow.json",
    "neutral_flow": "configs/ablations/neutral_flow.json",
    "full_cost": "configs/e1/single_cost.json",
    "neutral_cost": "configs/ablations/neutral_cost.json",
    "full_workload": "configs/e1/single_workload.json",
    "neutral_workload": "configs/ablations/neutral_workload.json",
}
STRUCTURAL = ("universal", "no_graph")
NEUTRAL = ("neutral_flow", "neutral_cost", "neutral_workload")
FATIGUE = ("full_flow", "neutral_flow", "full_cost", "neutral_cost", "full_workload", "neutral_workload")
VARIANTS = ("no_graph",) + FATIGUE


def batch_directory() -> Path:
    return project_path(load_config("configs/default.json")["paths"]["result_root"]) / "ablation_batches"


def save_batch_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)


def protocol_profile(config: dict) -> dict:
    """Compare solver identity and the controlled training/evaluation protocol."""
    from agent.ppo.network import normalize_network_config
    from environment.workload import workload_config

    training = deepcopy(config["training"])
    training.setdefault("episodes_per_update", training["parallel_envs"])
    training.setdefault("initial_checkpoint", None)
    network = deepcopy(config["network"])
    # Serialized tuples become JSON lists. Canonical network identity preserves
    # every selected objective/scale while making saved and live profiles equal.
    network.update(normalize_network_config(network))
    return {
        "network": network,
        **({"workload_objective": workload_config(config)}),
        "fatigue_mode": config.get("environment", {}).get("fatigue_mode", "full"),
        "environment": {k: v for k, v in config["environment"].items() if k != "fatigue_mode"},
        "reward": config["reward"], "ppo": config["ppo"],
        "preference": config["preference"], "generator": config["generator"], "dataset": config["dataset"],
        "objective_scalarizer": config["objective_scalarizer"],
        "experiment_suite_version": config["experiment_suite_version"],
        "quality_metric": config["evaluation"]["quality_metric"],
        "training": {key: training[key] for key in (
            "episodes", "parallel_envs", "validation_parallel_envs", "episodes_per_update",
            "validation_interval_episodes", "validation_instance_limit", "validation_selection",
            "policy_execution_version", "policy_precision", "formal_evaluation",
            "online_instances", "initial_checkpoint", "validation_split", "validation_control",
            "forced_action_compression", "worker_local_physical_forced_actions")},
    }


def discover_training_run(role: str, seed: int, run_root: Path | None = None) -> Path:
    expected = load_config(ROLE_CONFIGS[role])
    expected["seed"] = seed
    root = run_root or project_path(expected["paths"]["result_root"])
    validation_manifest = project_path(expected["paths"]["manifests_root"]) / "validation/manifest.json"
    validation_hash = dataset_manifest_snapshot(validation_manifest)["sha256"]
    candidates = []
    for run in root.iterdir() if root.is_dir() else ():
        if not run.is_dir() or not all((run / name).is_file() for name in ("config.json", "summary.json", "best_checkpoint.pt")):
            continue
        try:
            actual = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8-sig"))
            runtime = actual.get("runtime_manifest", {})
            if (int(actual["seed"]) != seed or int(summary["episodes"]) != expected["training"]["episodes"]
                    or not summary["checkpoint_selection"]["has_best"]
                    or runtime.get("observation_schema") != OBSERVATION_SCHEMA_VERSION
                    or runtime.get("time_context") != TIME_CONTEXT_VERSION
                    or summary["provenance"].get("dataset_manifest_sha256") != validation_hash
                    or protocol_profile(actual) != protocol_profile(expected)):
                continue
        except (ValueError, TypeError, KeyError, OSError):
            continue
        if sha256_file(run / "best_checkpoint.pt") != summary["provenance"].get("checkpoint_sha256"):
            continue
        try:
            validate_training_run(run, expected)
        except (ValueError, TypeError, KeyError, OSError):
            continue
        candidates.append(run)
    if not candidates:
        raise FileNotFoundError(f"No completed compatible {role} run for seed {seed}; train {ROLE_CONFIGS[role]} first")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def latest_evaluation_manifest(seed: int = 11) -> Path:
    candidates = []
    for path in batch_directory().glob(f"evaluation_seed{seed}_*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if manifest.get("status") == "complete" and set(manifest.get("runs", {})) == set(ROLE_CONFIGS):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed ablation evaluation manifest for seed {seed}; run python -m scripts.ablation_protocol")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def evaluate_group() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matched baseline/ablation checkpoints")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed = validate_algorithm_seed(load_config("configs/default.json"), args.seed)
    sources = {role: discover_training_run(role, seed) for role in ROLE_CONFIGS}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = batch_directory() / f"evaluation_seed{seed}_{stamp}.json"
    manifest = {"schema": "matched_ablation_evaluation_v1", "seed": seed, "status": "running", "runs": {}}
    for role, config_path in ROLE_CONFIGS.items():
        name = f"ablation_eval_{role}_seed{seed}_{stamp}"
        checkpoint = sources[role] / "best_checkpoint.pt"
        command = [sys.executable, str(ROOT / "eval.py"), "--config", config_path, "--policy", "ppo",
            "--checkpoint", str(checkpoint), "--dataset", "test", "--decode-mode", "sampled",
            "--algorithm-seed", str(seed), "--run-name", name]
        if role in ("universal", *STRUCTURAL):
            command += ["--preference-set", "final_test"]
        if args.device:
            command += ["--device", args.device]
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        save_batch_manifest(manifest_path, manifest)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            expected = load_config(config_path)
            expected["seed"] = seed
            output = project_path(expected["paths"]["result_root"]) / name
            validate_evaluation_run(output, expected, checkpoint)
        except Exception as error:
            manifest.update(status="failed", failed_role=role, error=str(error))
            save_batch_manifest(manifest_path, manifest)
            raise
        manifest["runs"][role] = {"config": config_path, "training_run": str(sources[role]),
            "checkpoint": str(checkpoint), "evaluation_run": str(output)}
        save_batch_manifest(manifest_path, manifest)
    if not args.dry_run:
        manifest["status"] = "complete"
        save_batch_manifest(manifest_path, manifest)
        print(f"Evaluation manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    evaluate_group()
