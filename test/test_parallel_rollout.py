from __future__ import annotations

import random
import time
from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, TypedActorCritic
from agent.ppo.parallel import (
    ParallelEpisodeRunner,
    ParallelWorkerError,
    _worker_roll_forward,
    physical_forced_action_from_mask,
)
from configs import load_config
from data.dataset import OnlineInstanceDataset, load_dataset_split
from environment import AssemblySchedulingEnv, RewardVector
from eval import evaluate_dataset, evaluate_dataset_parallel
from train import _collect_serial_batch


def _agent(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    network = TypedActorCritic(
        observation.feature_dimensions,
        int(config["network"]["hidden_dim"]),
    )
    return PPOAgent(network, config["ppo"], device="cpu")


class _FakeObservation:
    def __init__(self, state: int):
        self.state = state

    def copy(self):
        return _FakeObservation(self.state)


class _LocalForcedChainEnvironment:
    def __init__(self, masks, physical, *, terminate_at=None):
        self.masks = [np.asarray(mask, dtype=np.bool_) for mask in masks]
        self.physical = list(physical)
        self.terminate_at = terminate_at
        self.state = 0
        self.terminated = False
        self.truncated = False
        self.build_observation_flags: list[bool] = []

    def get_action_mask(self):
        return self.masks[self.state].copy()

    def forced_action_diagnostic(self, action_mask):
        legal = np.flatnonzero(~np.asarray(action_mask, dtype=np.bool_))
        if legal.size != 1:
            return None
        physical = bool(self.physical[self.state])
        action_kind = (
            "advance"
            if int(legal[0]) == len(action_mask) - 1
            else "pair"
        )
        return {
            "non_delay_blocked_advance": not physical,
            "advance_physically_unavailable": (
                physical and action_kind == "pair"
            ),
            "pair_physically_unavailable": (
                physical and action_kind == "advance"
            ),
        }

    def step(self, action, *, build_observation=True):
        assert not self.get_action_mask()[int(action)]
        self.build_observation_flags.append(bool(build_observation))
        self.state += 1
        self.terminated = self.state == self.terminate_at
        observation = (
            _FakeObservation(self.state)
            if build_observation and not self.terminated
            else None
        )
        reward = RewardVector(
            flow=float(self.state),
            cost=-float(self.state),
            variance=0.5,
        )
        return observation, reward, self.terminated, False, {}

    def observe(self):
        return _FakeObservation(self.state)

    def metrics(self):
        return {"state": self.state}

    def validate_schedule(self):
        return []


def test_physical_forced_detection_excludes_non_delay_singletons():
    physical_pair = _LocalForcedChainEnvironment(
        [[False, True]],
        [True],
    )
    physical_advance = _LocalForcedChainEnvironment(
        [[True, False]],
        [True],
    )
    non_delay_pair = _LocalForcedChainEnvironment(
        [[False, True]],
        [False],
    )

    assert physical_forced_action_from_mask(
        physical_pair,
        physical_pair.get_action_mask(),
    ) == 0
    assert physical_forced_action_from_mask(
        physical_advance,
        physical_advance.get_action_mask(),
    ) == 1
    assert physical_forced_action_from_mask(
        non_delay_pair,
        non_delay_pair.get_action_mask(),
    ) is None


def test_worker_local_roll_forward_stops_at_policy_or_step_budget():
    environment = _LocalForcedChainEnvironment(
        [
            [True, False],
            [False, True],
            [False, False],
        ],
        [True, True, False],
    )
    response = _worker_roll_forward(
        0,
        environment,
        _FakeObservation(0),
        preserve_graph=True,
        requested_action=None,
        drain_physical_forced_actions=True,
        max_environment_steps=None,
    )
    assert response.environment_step_count == 2
    assert response.local_physical_forced_action_count == 2
    assert response.observation.state == 2
    assert np.count_nonzero(~response.action_mask) == 2
    assert response.reward_vector.flow == pytest.approx(3.0)
    assert response.reward_vector.cost == pytest.approx(-3.0)
    assert environment.build_observation_flags == [False, False]

    budgeted = _LocalForcedChainEnvironment(
        [
            [True, False],
            [False, True],
            [False, False],
        ],
        [True, True, False],
    )
    response = _worker_roll_forward(
        0,
        budgeted,
        _FakeObservation(0),
        preserve_graph=True,
        requested_action=None,
        drain_physical_forced_actions=True,
        max_environment_steps=1,
    )
    assert response.environment_step_count == 1
    assert response.local_physical_forced_action_count == 1
    assert response.observation.state == 1
    assert budgeted.build_observation_flags == [False]


def test_worker_local_roll_forward_aggregates_terminal_suffix():
    environment = _LocalForcedChainEnvironment(
        [
            [False, False],
            [True, False],
        ],
        [False, True],
        terminate_at=2,
    )
    response = _worker_roll_forward(
        0,
        environment,
        _FakeObservation(0),
        preserve_graph=True,
        requested_action=0,
        drain_physical_forced_actions=True,
        max_environment_steps=2,
    )
    assert response.terminated
    assert response.environment_step_count == 2
    assert response.local_physical_forced_action_count == 1
    assert response.reward_vector.flow == pytest.approx(3.0)
    assert response.metrics["state"] == 2
    assert response.metrics["schedule_violations"] == []
    assert environment.build_observation_flags == [False, False]


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
            value.base_reward_sum == pytest.approx(value.expected_reward)
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
                "truncation",
                "unfinished",
                "feasibility_shaping",
            }
            for value in rollout.episodes
        )
    assert processes
    assert all(not process.is_alive() for process in processes)


def test_parallel_compression_skips_singleton_policy_masks(
    config,
    fixed_instance,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    effective_config["training"]["forced_action_compression"] = True
    effective_config["ppo"]["gamma"] = 1.0
    effective_config["ppo"]["gae_lambda"] = 0.995
    agent = _agent(effective_config, fixed_instance)
    original_act_batch = agent.act_batch
    policy_legal_action_counts: list[int] = []

    def tracked_act_batch(observations, action_masks, **kwargs):
        policy_legal_action_counts.extend(
            int(np.count_nonzero(~mask)) for mask in action_masks
        )
        return original_act_batch(observations, action_masks, **kwargs)

    monkeypatch.setattr(agent, "act_batch", tracked_act_batch)
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=1,
        worker_count=2,
    ) as runner:
        rollout = runner.collect_training_batch(
            agent,
            [0],
            gamma=1.0,
            gae_lambda=0.995,
            step_limit=64,
        )

    episode = rollout.episodes[0]
    assert policy_legal_action_counts
    assert min(policy_legal_action_counts) > 1
    assert episode.forced_action_count > 0
    assert episode.worker_local_physical_forced_action_count > 0
    assert episode.worker_step_command_count < episode.step_count
    assert episode.step_count == (
        episode.policy_step_count + episode.forced_action_count
    )
    assert episode.step_count == (
        episode.worker_step_command_count
        + episode.worker_local_physical_forced_action_count
    )
    assert rollout.environment_step_count == episode.step_count
    assert rollout.transition_count == episode.policy_step_count
    assert rollout.worker_step_command_count == (
        episode.worker_step_command_count
    )
    assert rollout.worker_local_physical_forced_action_count == (
        episode.worker_local_physical_forced_action_count
    )
    attributed_reward = sum(
        transition.reward for transition in episode.buffer.transitions
    )
    assert (
        attributed_reward + episode.unattributed_forced_reward
        == pytest.approx(episode.reward_sum, abs=1e-8)
    )


def test_worker_local_rollout_matches_round_trip_compression(
    config,
    fixed_instance,
):
    local_config = deepcopy(config)
    local_config["training"]["worker_timeout_seconds"] = 120
    local_config["training"]["forced_action_compression"] = True
    local_config["training"][
        "worker_local_physical_forced_actions"
    ] = True
    local_config["ppo"]["gamma"] = 1.0
    round_trip_config = deepcopy(local_config)
    round_trip_config["training"][
        "worker_local_physical_forced_actions"
    ] = False
    agent = _agent(local_config, fixed_instance)

    with ParallelEpisodeRunner(
        config=round_trip_config,
        template=fixed_instance,
        episode_count=1,
        worker_count=2,
    ) as runner:
        torch.manual_seed(24680)
        round_trip = runner.collect_training_batch(
            agent,
            [0],
            gamma=1.0,
            gae_lambda=float(local_config["ppo"]["gae_lambda"]),
            step_limit=64,
        ).episodes[0]
    with ParallelEpisodeRunner(
        config=local_config,
        template=fixed_instance,
        episode_count=1,
        worker_count=2,
    ) as runner:
        torch.manual_seed(24680)
        local = runner.collect_training_batch(
            agent,
            [0],
            gamma=1.0,
            gae_lambda=float(local_config["ppo"]["gae_lambda"]),
            step_limit=64,
        ).episodes[0]

    assert local.step_count == round_trip.step_count
    assert local.policy_step_count == round_trip.policy_step_count
    assert local.forced_action_count == round_trip.forced_action_count
    assert local.reward_sum == pytest.approx(round_trip.reward_sum, abs=1e-10)
    assert local.reward_components == pytest.approx(
        round_trip.reward_components,
        abs=1e-10,
    )
    assert local.metrics == round_trip.metrics
    assert local.worker_local_physical_forced_action_count > 0
    assert round_trip.worker_local_physical_forced_action_count == 0
    assert local.worker_step_command_count < (
        round_trip.worker_step_command_count
    )
    assert len(local.buffer) == len(round_trip.buffer)
    for local_transition, round_trip_transition in zip(
        local.buffer.transitions,
        round_trip.buffer.transitions,
    ):
        assert local_transition.action == round_trip_transition.action
        assert np.array_equal(
            local_transition.action_mask,
            round_trip_transition.action_mask,
        )
        for field in (
            "log_probability",
            "value",
            "reward",
            "advantage",
            "return_value",
        ):
            assert getattr(local_transition, field) == pytest.approx(
                getattr(round_trip_transition, field),
                abs=1e-10,
            )
        assert local_transition.done == round_trip_transition.done


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


def test_sampled_validation_is_parallelism_invariant_and_preserves_rng(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    effective_config["environment"]["max_decisions"] = 30
    agent = _agent(effective_config, fixed_instance)
    sampling_seed = 100011
    instance_limit = 10
    timing_fields = {
        "inference_time_seconds",
        "solve_time_seconds",
        "inference_time_per_decision_ms",
    }

    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    serial_rows, _, _, _ = evaluate_dataset(
        effective_config,
        dataset_name="validation",
        policy_name="ppo",
        ppo_agent=agent,
        instance_limit=instance_limit,
        decode_mode="sampled",
        sampling_seed=sampling_seed,
    )
    expected = [
        {
            key: value
            for key, value in row.items()
            if key not in timing_fields
        }
        for row in serial_rows
    ]

    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=instance_limit,
        worker_count=10,
    ) as runner:
        for parallel_envs in (1, 2, 10):
            effective_config["training"][
                "validation_parallel_envs"
            ] = parallel_envs
            parallel_rows, aggregate = evaluate_dataset_parallel(
                effective_config,
                dataset_name="validation",
                ppo_agent=agent,
                runner=runner,
                instance_limit=instance_limit,
                decode_mode="sampled",
                sampling_seed=sampling_seed,
            )
            actual = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in timing_fields
                }
                for row in parallel_rows
            ]
            assert actual == expected
            assert aggregate["parallel_envs"] == parallel_envs
            assert all(row["action_trace_sha256"] for row in parallel_rows)

    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_serial_and_parallel_shaping_are_identical(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    effective_config["training"]["forced_action_compression"] = True
    effective_config["training"][
        "worker_local_physical_forced_actions"
    ] = True
    effective_config["ppo"]["gamma"] = 1.0
    agent = _agent(effective_config, fixed_instance)
    dataset = OnlineInstanceDataset(
        config=effective_config,
        template=fixed_instance,
        episode_count=1,
    )
    record = dataset[0]
    serial_environment = AssemblySchedulingEnv(effective_config)

    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=1,
        worker_count=2,
    ) as runner:
        torch.manual_seed(12345)
        serial = _collect_serial_batch(
            config=effective_config,
            agent=agent,
            environment=serial_environment,
            episode_index=0,
            instance=record.instance,
            record=record,
            sampling_start=time.perf_counter(),
            generation_time_seconds=0.0,
            reward_phase="feasibility",
            step_limit=20,
        ).episodes[0]
        torch.manual_seed(12345)
        parallel = runner.collect_training_batch(
            agent,
            [0],
            gamma=float(effective_config["ppo"]["gamma"]),
            gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
            reward_phase="feasibility",
            step_limit=20,
        ).episodes[0]

    assert serial.reward_components == pytest.approx(
        parallel.reward_components,
        abs=1e-10,
    )
    assert serial.reward_sum == pytest.approx(parallel.reward_sum, abs=1e-10)
    assert serial.base_reward_sum == pytest.approx(
        parallel.base_reward_sum,
        abs=1e-10,
    )
    assert serial.step_count == parallel.step_count
    assert serial.policy_step_count == parallel.policy_step_count
    assert serial.forced_action_count == parallel.forced_action_count
    assert parallel.worker_local_physical_forced_action_count > 0
    assert parallel.worker_step_command_count + (
        parallel.worker_local_physical_forced_action_count
    ) == parallel.step_count


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


def test_training_cache_manifest_and_worker_progress_are_persistent(
    config,
    fixed_instance,
    tmp_path,
):
    effective = deepcopy(config)
    effective["paths"]["training_instances_cache"] = str(
        tmp_path / "cache"
    )
    effective["training"]["worker_timeout_seconds"] = 120
    effective["training"]["worker_stall_timeout_seconds"] = 30
    run_directory = tmp_path / "run"
    with ParallelEpisodeRunner(
        config=effective,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
        diagnostic_directory=run_directory,
    ) as runner:
        first = runner.pre_generate_training_instances()
    assert first["instance_count"] == 2
    assert first["cache_hit_count"] == 0
    assert first["cache_fingerprint"]
    assert (run_directory / "training_instance_manifest.json").exists()
    assert (run_directory / "training_instance_manifest.sha256").exists()
    assert (run_directory / "worker_progress.jsonl").exists()

    second_run = tmp_path / "second_run"
    with ParallelEpisodeRunner(
        config=effective,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
        diagnostic_directory=second_run,
    ) as runner:
        second = runner.pre_generate_training_instances()
    assert second["cache_hit_count"] == 2
    assert [entry["sha256"] for entry in first["files"]] == [
        entry["sha256"] for entry in second["files"]
    ]


@pytest.mark.slow
def test_training_indices_220_239_repeat_three_times_with_twenty_workers(
    fixed_instance,
    tmp_path,
):
    effective = load_config("configs/v7/e1_single_flow.json")
    effective["device"] = "cpu"
    effective["paths"]["training_instances_cache"] = str(
        tmp_path / "cache"
    )
    effective["training"]["worker_timeout_seconds"] = 600
    effective["training"]["worker_stall_timeout_seconds"] = 60
    with ParallelEpisodeRunner(
        config=effective,
        template=fixed_instance,
        episode_count=2000,
        worker_count=20,
        diagnostic_directory=tmp_path / "diagnostics",
    ) as runner:
        reports = [
            runner.pre_generate_training_instances(range(220, 240))
            for _ in range(3)
        ]
    hashes = [
        [entry["sha256"] for entry in report["files"]]
        for report in reports
    ]
    assert hashes[0] == hashes[1] == hashes[2]
    assert all(report["temporal_unknown_count"] == 0 for report in reports)
    assert all(
        report["temporal_budget_termination_reasons"] == {}
        for report in reports
    )
    assert reports[1]["cache_hit_count"] == 20
    assert reports[2]["cache_hit_count"] == 20
