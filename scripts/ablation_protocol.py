"""Matched ablation run discovery and seed-specific training/evaluation entry points."""
from __future__ import annotations

import argparse
import json
import subprocess
import hashlib
import tempfile
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from configs import load_config, project_path
from data.dataset import sha256_file, validate_algorithm_seed
from result.io import write_json
from result.provenance import dataset_manifest_snapshot

ROOT = Path(__file__).resolve().parents[1]
ROLE_CONFIGS = {
    "universal": "configs/v8/universal.json",
    "no_graph": "configs/ablations/no_graph.json",
    "shared_head": "configs/ablations/shared_head.json",
    "full_flow": "configs/e1/single_flow.json",
    "neutral_flow": "configs/ablations/neutral_flow.json",
    "full_cost": "configs/e1/single_cost.json",
    "neutral_cost": "configs/ablations/neutral_cost.json",
    "full_variance": "configs/e1/single_variance.json",
    "neutral_variance": "configs/ablations/neutral_variance.json",
}
STRUCTURAL = ("no_graph", "shared_head")
NEUTRAL = ("neutral_flow", "neutral_cost", "neutral_variance")
VARIANTS = STRUCTURAL + NEUTRAL


def batch_directory() -> Path:
    return project_path(load_config("configs/default.json")["paths"]["result_root"]) / "ablation_batches"


def save_batch_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)


def protocol_profile(config: dict) -> dict:
    """Compare solver identity and the controlled training/evaluation protocol."""
    training = deepcopy(config["training"])
    training.setdefault("episodes_per_update", training["parallel_envs"])
    training.setdefault("initial_checkpoint", None)
    network = deepcopy(config["network"])
    network.setdefault("encoder_variant", "hetero_gnn")
    network.setdefault("actor_head_variant", "objective_experts")
    return {
        "network": network,
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
                    or runtime.get("observation_schema") != 6
                    or runtime.get("time_context") != "order_chain_action_context_v1"
                    or summary["provenance"].get("dataset_manifest_sha256") != validation_hash
                    or protocol_profile(actual) != protocol_profile(expected)):
                continue
        except (ValueError, TypeError, KeyError, OSError):
            continue
        if sha256_file(run / "best_checkpoint.pt") != summary["provenance"].get("checkpoint_sha256"):
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
        raise FileNotFoundError(f"No completed ablation evaluation manifest for seed {seed}; run scripts/run_13_evaluate_ablations.py")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def train_group(roles: tuple[str, ...], *, smoke: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Train matched ablation variants")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed = validate_algorithm_seed(load_config("configs/default.json"), args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    group = "smoke" if smoke else ("structural" if roles == STRUCTURAL else "neutral")
    manifest_path = batch_directory() / f"training_{group}_seed{seed}_{stamp}.json"
    manifest = {"schema": "matched_ablation_batch_v1", "seed": seed, "group": group, "status": "running", "runs": {}}
    for role in roles:
        config = load_config(ROLE_CONFIGS[role])
        if args.device:
            config["device"] = args.device
        if smoke:
            import torch
            cache_key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:10]
            config["paths"]["training_instances_cache"] = str(Path(tempfile.gettempdir()) / f"assembly_smoke_{cache_key}")
            config["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            config["training"].update({"smoke_episodes": 2, "smoke_rollout_steps": 16,
                "smoke_parallel_envs": 2, "validation_parallel_envs": 2,
                "smoke_validation_instance_limit": 1, "torch_num_threads": 2})
            formal = config["training"]["formal_evaluation"]
            formal["validation_repeats"] = formal["final_test_repeats"] = 1
            formal["validation_preferences"] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            formal["final_test_lattice_denominator"] = 1
        name = f"ablation_{group}_{role}_seed{seed}_{stamp}"
        print(f"{role}: {ROLE_CONFIGS[role]}, seed={seed}, episodes={2 if smoke else config['training']['episodes']}, run={name}", flush=True)
        if args.dry_run:
            continue
        from train import train
        save_batch_manifest(manifest_path, manifest)
        try:
            output = train(config, smoke=smoke, algorithm_seed=seed, run_name=name)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8-sig"))
            if not summary["checkpoint_selection"]["has_best"] or not (output / "best_checkpoint.pt").is_file():
                raise RuntimeError(f"No eligible best checkpoint for {role}")
        except Exception as error:
            manifest.update(status="failed", failed_role=role, error=str(error))
            save_batch_manifest(manifest_path, manifest)
            raise
        manifest["runs"][role] = {"config": ROLE_CONFIGS[role], "training_run": str(output), "checkpoint": str(output / "best_checkpoint.pt")}
        save_batch_manifest(manifest_path, manifest)
    if not args.dry_run:
        manifest["status"] = "complete"
        save_batch_manifest(manifest_path, manifest)
        print(f"Batch manifest: {manifest_path}", flush=True)


def evaluate_group() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the nine matched baseline/ablation checkpoints")
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
        except subprocess.CalledProcessError as error:
            manifest.update(status="failed", failed_role=role, error=str(error))
            save_batch_manifest(manifest_path, manifest)
            raise
        output = project_path(load_config(config_path)["paths"]["result_root"]) / name
        manifest["runs"][role] = {"config": config_path, "training_run": str(sources[role]),
            "checkpoint": str(checkpoint), "evaluation_run": str(output)}
        save_batch_manifest(manifest_path, manifest)
    if not args.dry_run:
        manifest["status"] = "complete"
        save_batch_manifest(manifest_path, manifest)
        print(f"Evaluation manifest: {manifest_path}", flush=True)
