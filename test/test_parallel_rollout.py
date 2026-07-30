from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, TypedActorCritic
from agent.ppo.parallel import (
    ParallelEpisodeRunner,
    ParallelWorkerError,
)
from data.dataset import load_dataset_split
from environment import AssemblySchedulingEnv
from eval import evaluate_dataset, evaluate_dataset_parallel


def _agent(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    network = TypedActorCritic(
        observation.feature_dimensions,
        int(config["network"]["hidden_dim"]),
    )
    return PPOAgent(network, config["ppo"], device="cpu")


def test_parallel_training_seeds_and_cleanup(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    agent = _agent(effective_config, fixed_instance)
    processes = []
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
    ) as runner:
        processes = list(runner._processes)
        rollout = runner.collect_training_batch(
            agent,
            [0, 1],
            gamma=float(effective_config["ppo"]["gamma"]),
            gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
            step_limit=1,
        )
        assert [value.episode_index for value in rollout.episodes] == [0, 1]
        assert [value.metadata["seed"] for value in rollout.episodes] == [
            1_000_000,
            1_000_001,
        ]
        assert rollout.transition_count == 2
        assert all(len(value.buffer) == 1 for value in rollout.episodes)
        assert all(
            value.reward_phase == "feasibility"
            for value in rollout.episodes
        )
        assert all(
            value.reward_sum == pytest.approx(value.expected_reward)
            for value in rollout.episodes
        )
        assert all(
            set(value.reward_components)
            == {
                "flow",
                "cost",
                "variance",
                "completion_progress",
                "completion_bonus",
                "quality",
            }
            for value in rollout.episodes
        )
    assert processes
    assert all(not process.is_alive() for process in processes)


def test_parallel_validation_matches_serial_and_preserves_rng(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    effective_config["environment"]["max_decisions"] = 50
    agent = _agent(effective_config, fixed_instance)
    dataset = load_dataset_split(effective_config, "validation")
    processes = []
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
    ) as runner:
        processes = list(runner._processes)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        agent.network.train()
        parallel_rows, parallel = evaluate_dataset_parallel(
            effective_config,
            dataset_name="validation",
            ppo_agent=agent,
            runner=runner,
            instance_limit=2,
        )
        serial_rows, _, _, serial = evaluate_dataset(
            effective_config,
            dataset_name="validation",
            policy_name="ppo",
            ppo_agent=agent,
            instance_limit=2,
        )
        assert agent.network.training
        assert random.getstate() == python_state
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_state[0]
        assert np.array_equal(after_numpy[1], numpy_state[1])
        assert after_numpy[2:] == numpy_state[2:]
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert [row["instance_id"] for row in parallel_rows] == [
            row["instance_id"] for row in serial_rows
        ]
        for parallel_row, serial_row in zip(
            parallel_rows,
            serial_rows,
        ):
            for field in (
                "terminated",
                "truncated",
                "termination_reason",
                "decisions",
                "makespan",
                "total_flow_time",
                "flow_time_objective",
                "reconfiguration_cost",
                "worker_load_variance",
                "schedule_violation_count",
            ):
                expected = serial_row[field]
                if isinstance(expected, float):
                    assert parallel_row[field] == pytest.approx(
                        expected,
                        abs=1e-8,
                    )
                else:
                    assert parallel_row[field] == expected
        assert parallel["completion_rate"] == serial["completion_rate"]
        assert parallel["truncated_count"] == serial["truncated_count"]
        assert dataset.manifest_path.exists()
    assert all(not process.is_alive() for process in processes)


def test_parallel_worker_error_is_reported_and_all_workers_exit(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 30
    processes = []
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
    ) as runner:
        processes = list(runner._processes)
        with pytest.raises(
            ParallelWorkerError,
            match="unknown worker command",
        ):
            runner._exchange({0: ("invalid-command", None)})
    assert all(not process.is_alive() for process in processes)
