from __future__ import annotations

import argparse
import json
import multiprocessing
import time

import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config, project_path
from data.dataset import OnlineInstanceDataset, validate_algorithm_seed
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from train import _collect_serial_batch
from utils import set_seed


def benchmark(
    config: dict,
    *,
    episodes: int,
    steps: int,
    workers: int,
) -> dict:
    if episodes < 1 or steps < 1 or workers < 2:
        raise ValueError("episodes/steps must be positive and workers >= 2")
    if episodes > workers:
        raise ValueError("episodes cannot exceed workers in a hot batch")
    torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    seed = validate_algorithm_seed(config, int(config["seed"]))
    template = load_instance_yaml(
        project_path(config["paths"]["fixed_instance"])
    )
    bootstrap = AssemblySchedulingEnv(config).reset(template)
    set_seed(seed)
    reference_network = build_actor_critic(bootstrap, config["network"])
    reference_state = {
        name: value.detach().clone()
        for name, value in reference_network.state_dict().items()
    }

    serial_network = build_actor_critic(bootstrap, config["network"])
    serial_network.load_state_dict(reference_state)
    serial_agent = PPOAgent(
        serial_network,
        config["ppo"],
        device=config["device"],
    )
    serial_dataset = OnlineInstanceDataset(
        config=config,
        template=template,
        episode_count=episodes,
    )
    serial_environment = AssemblySchedulingEnv(config)
    set_seed(seed)
    serial_start = time.perf_counter()
    serial_transitions = 0
    for episode_index in range(episodes):
        generation_start = time.perf_counter()
        record = serial_dataset[episode_index]
        generation_time = time.perf_counter() - generation_start
        rollout = _collect_serial_batch(
            config=config,
            agent=serial_agent,
            environment=serial_environment,
            instance=record.instance,
            record=record,
            episode_index=episode_index,
            sampling_start=generation_start,
            generation_time_seconds=generation_time,
            step_limit=steps,
        )
        serial_transitions += rollout.transition_count
    serial_seconds = time.perf_counter() - serial_start

    parallel_network = build_actor_critic(bootstrap, config["network"])
    parallel_network.load_state_dict(reference_state)
    parallel_agent = PPOAgent(
        parallel_network,
        config["ppo"],
        device=config["device"],
    )
    startup_start = time.perf_counter()
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=episodes,
        worker_count=workers,
    ) as runner:
        startup_seconds = time.perf_counter() - startup_start
        set_seed(seed)
        parallel_rollout = runner.collect_training_batch(
            parallel_agent,
            list(range(episodes)),
            gamma=float(config["ppo"]["gamma"]),
            gae_lambda=float(config["ppo"]["gae_lambda"]),
            step_limit=steps,
        )
    parallel_seconds = parallel_rollout.sampling_wall_time_seconds
    return {
        "episodes": episodes,
        "steps_per_episode": steps,
        "workers": workers,
        "serial_transitions": serial_transitions,
        "parallel_transitions": parallel_rollout.transition_count,
        "serial_sampling_seconds": serial_seconds,
        "parallel_hot_sampling_seconds": parallel_seconds,
        "worker_startup_seconds": startup_seconds,
        "hot_sampling_speedup": (
            serial_seconds / parallel_seconds
            if parallel_seconds > 0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare serial and parallel rollout throughput"
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    metrics = benchmark(
        load_config(args.config),
        episodes=args.episodes,
        steps=args.steps,
        workers=args.workers,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if (
        metrics["hot_sampling_speedup"] is not None
        and metrics["hot_sampling_speedup"] < 2.0
    ):
        raise SystemExit(
            "parallel hot sampling speedup is below the 2x target"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
