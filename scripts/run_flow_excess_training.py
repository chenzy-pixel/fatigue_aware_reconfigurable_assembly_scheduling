"""Portable launch/preflight for the modified Flow and Universal experiments."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import load_config
from configs.config import public_config
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from data.dataset import validate_algorithm_seed
from environment.types import FLOW_EXCESS, flow_mode, flow_reward_version
from result.io import write_json
from result.provenance import dataset_manifest_snapshot

CONFIGS = {
    "flow": "configs/flow_excess/single_flow.json",
    "universal": "configs/flow_excess/universal.json",
}


def startup_preflight(configs, *, probe_network=False, config_paths=None,
                      allowed_flow_modes=(FLOW_EXCESS,)):
    """Check native data, contracts and hardware; never collect/train episodes."""
    import torch
    reports = {}
    for label, config in configs.items():
        if flow_mode(config) not in allowed_flow_modes or config["reward"]["mode"] != flow_reward_version(config):
            raise ValueError("experiment Flow mode and reward identity disagree with the launch contract")
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is configured but unavailable; install matching PyTorch/driver or specify --device cpu")
        if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError("configured CUDA device index is unavailable")
        datasets = {split: load_dataset_split(config, split) for split in ("validation", "test")}
        formal = config["training"]["formal_evaluation"]
        reports[label] = {
            "config": (config_paths or CONFIGS)[label], "flow_mode": flow_mode(config), "reward_version": config["reward"]["mode"],
            "actor_head_variant": config["network"].get("actor_head_variant", "objective_experts"),
            "scales": config["objective_scalarizer"]["scales"], "device": str(device),
            "episodes": config["training"]["episodes"],
            "episodes_per_update": config["training"].get("episodes_per_update") or config["training"]["parallel_envs"],
            "parallel_envs": config["training"]["parallel_envs"],
            "validation_parallel_envs": config["training"]["validation_parallel_envs"],
            "validation_interval_episodes": config["training"]["validation_interval_episodes"],
            "validation_instances": config["training"]["validation_instance_limit"],
            "validation_preferences": len(formal_preferences(config, "validation")),
            "validation_repeats": formal["validation_repeats"],
            "test_instances": len(datasets["test"]),
            "final_preferences": len(formal_preferences(config, "final_test")),
            "final_repeats": formal["final_test_repeats"],
            "dataset_manifest_sha256": {split: dataset_manifest_snapshot(data.manifest_path)["sha256"]
                                        for split, data in datasets.items()},
            "normalization_manifest_sha256": config["objective_scalarizer"]["normalization_manifest_sha256"],
        }
        if probe_network:
            from agent.ppo import build_actor_critic
            from environment import AssemblySchedulingEnv
            env = AssemblySchedulingEnv(config)
            observation = env.reset(datasets["validation"][0].instance)
            mask = env.get_action_mask()
            network = build_actor_critic(observation, config["network"]).to(device)
            with torch.no_grad():
                logits, values = network.forward_batch([observation, observation], [mask, mask], device=device)
                assert bool(torch.isfinite(values).all())
                assert bool(torch.isfinite(logits[:, :len(mask)][:, ~torch.as_tensor(mask, device=device)]).all())
            spec = network.network_spec()
            reports[label]["network_identity"] = {
                key: spec[key] for key in ("flow_mode", "reward_version", "normalization_manifest_sha256",
                                           "hidden_dim", "message_passing_layers", "observation_schema_version",
                                           "actor_head_variant", "encoder_variant")}
            reports[label]["network_probe_finite"] = True
            del network, logits, values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return reports


def launch_argument_parser(*, preflight_only_default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device")
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--validation-parallel-envs", type=int)
    parser.add_argument("--preflight-only", action="store_true", default=preflight_only_default)
    return parser


def launch_configurations(config_paths, args, *, prefix="flow_excess", allowed_flow_modes=(FLOW_EXCESS,)):
    labels = tuple(config_paths)
    configs = {label: load_config(ROOT / config_paths[label]) for label in labels}
    for config in configs.values():
        config["seed"] = validate_algorithm_seed(config, args.seed)
        # Worker count is a runtime control; keep the original PPO update budget.
        if config["training"].get("episodes_per_update") is None:
            config["training"]["episodes_per_update"] = config["training"]["parallel_envs"]
        if args.device is not None:
            config["device"] = args.device
        for argument in ("parallel_envs", "validation_parallel_envs"):
            value = getattr(args, argument)
            if value is not None:
                if value <= 0:
                    raise ValueError(f"--{argument.replace('_', '-')} must be positive")
                config["training"][argument] = value
    reports = startup_preflight(configs, probe_network=args.preflight_only,
                                config_paths=config_paths, allowed_flow_modes=allowed_flow_modes)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{prefix}_seed{args.seed}_{stamp}"
    directory = ROOT / ("result/analysis" if args.preflight_only else "result/runs") / f"{prefix}_startup"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "preflight.json", reports)
    if args.preflight_only:
        print(f"Startup preflight passed for {', '.join(labels)}: {directory}", flush=True)
        return
    jobs = []
    for label in labels:
        path = directory / f"{label}_config.json"
        write_json(path, public_config(configs[label]))
        jobs.append({"experiment": label, "run_name": f"{prefix}_{label}",
                     "config": str(path), "status": "pending"})
    write_json(directory / "launch.json", {"jobs": jobs})
    for job in jobs:
        job["status"] = "running"
        write_json(directory / "launch.json", {"jobs": jobs})
        command = [sys.executable, "-u", str(ROOT / "train.py"), "--config", job["config"],
                   "--algorithm-seed", str(args.seed), "--run-name", job["run_name"]]
        result = subprocess.run(command, cwd=ROOT)
        job.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
        write_json(directory / "launch.json", {"jobs": jobs})
        if result.returncode:
            raise SystemExit(result.returncode)


def main(default_experiment="both", *, preflight_only_default=False):
    parser = launch_argument_parser(preflight_only_default=preflight_only_default)
    parser.add_argument("--experiment", choices=("flow", "universal", "both"), default=default_experiment)
    args = parser.parse_args()
    labels = tuple(CONFIGS) if args.experiment == "both" else (args.experiment,)
    launch_configurations({label: CONFIGS[label] for label in labels}, args)


if __name__ == "__main__":
    main()
