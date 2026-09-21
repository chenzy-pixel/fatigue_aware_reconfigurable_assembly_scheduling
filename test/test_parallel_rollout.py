from __future__ import annotations

import random
import time
from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.buffer import RolloutBuffer
from agent.ppo.parallel import (
    ParallelEpisodeRunner,
    ParallelWorkerError,
    _worker_roll_forward,
    physical_forced_action_from_mask,
)
from configs import load_config
from data.dataset import OnlineInstanceDataset, load_dataset_split
from environment import AssemblySchedulingEnv, RewardVector
from eval import (
    EvaluationPolicy,
    evaluate_dataset,
    evaluate_dataset_parallel,
    evaluate_instance,
)


def _agent(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    network = build_actor_critic(observation, config["network"])
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
            "wait"
            if int(legal[0]) == len(action_mask) - 1
            else "pair"
        )
        return {
            "wait_physically_unavailable": (
                physical and action_kind == "pair"
            ),
            "pair_physically_unavailable": (
                physical and action_kind == "wait"
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


def test_physical_forced_detection_accepts_pair_and_wait_singletons():
    physical_pair = _LocalForcedChainEnvironment(
        [[False, True]],
        [True],
    )
    physical_wait = _LocalForcedChainEnvironment(
        [[True, False]],
        [True],
    )
    non_physical_pair = _LocalForcedChainEnvironment(
        [[False, True]],
        [False],
    )

    assert physical_forced_action_from_mask(
        physical_pair,
        physical_pair.get_action_mask(),
    ) == 0
    assert physical_forced_action_from_mask(
        physical_wait,
        physical_wait.get_action_mask(),
    ) == 1
    assert physical_forced_action_from_mask(
        non_physical_pair,
        non_physical_pair.get_action_mask(),
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
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    agent = _agent(effective_config, fixed_instance)
    monkeypatch.setattr(
        agent,
        "value_batch",
        lambda observations, masks: [7.0] * len(observations),
    )
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
            episode.buffer.transitions[0].done is False
            and episode.buffer.transitions[0].return_value
            == pytest.approx(episode.buffer.transitions[0].reward + 7.0)
            for episode in rollout.episodes
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
                "operation_progress",
                "quality",
                "feasibility_shaping",
            }
            for value in rollout.episodes
        )
    assert processes
    assert all(not process.is_alive() for process in processes)


def test_environment_failure_marks_done_and_disables_critic_bootstrap(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["environment"]["max_decisions"] = 1
    agent = _agent(effective_config, fixed_instance)
    environment = AssemblySchedulingEnv(effective_config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    action, log_probability, value = agent.act(observation, mask)
    _, reward, terminated, truncated, _ = environment.step(action)
    assert terminated is False
    assert truncated is True
    buffer = RolloutBuffer()
    buffer.add(
        observation,
        mask,
        action,
        log_probability,
        value,
        reward.scalarize(effective_config["reward"]),
        done=terminated or truncated,
    )
    buffer.compute_gae(last_value=123.0, gamma=1.0, gae_lambda=0.95)
    transition = buffer.transitions[0]
    assert environment.metrics()["task_failed"] is True
    assert transition.done is True
    assert transition.return_value == pytest.approx(transition.reward)


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
    for field in (
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
        "terminal_reason",
        "completed_operations",
    ):
        assert local.metrics[field] == round_trip.metrics[field]
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
                abs=1e-4,
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


def test_mixed_preference_quality_rollout_uses_episode_contexts(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["training"]["worker_timeout_seconds"] = 120
    agent = _agent(effective_config, fixed_instance)
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
    ) as runner:
        batch = runner.collect_training_batch(
            agent,
            [0, 1],
            gamma=1.0,
            gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
            step_limit=6,
            preferences=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
    expected = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert len(batch.episodes) == 2
    for episode, preference in zip(batch.episodes, expected, strict=True):
        assert episode.metadata["preference"] == preference
        assert episode.metadata["preference_key"]
        assert episode.reward_sum == pytest.approx(
            episode.reward_components["operation_progress"]
            + episode.reward_components["quality"],
            abs=1e-8,
        )
        assert all(
            transition.observation.preference.tolist() == preference
            for transition in episode.buffer.transitions
        )


def test_serial_and_parallel_ppo_preserve_the_same_preference(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["device"] = "cpu"
    effective_config["environment"]["max_decisions"] = 50
    effective_config["training"]["worker_timeout_seconds"] = 120
    agent = _agent(effective_config, fixed_instance)
    agent.network.eval()
    record = load_dataset_split(effective_config, "validation")[0]
    preference = (7.0, 2.0, 1.0)
    with ParallelEpisodeRunner(
        config=effective_config,
        template=fixed_instance,
        episode_count=1,
        worker_count=1,
    ) as runner:
        parallel = runner.evaluate_records(
            agent,
            [record],
            max_parallelism=1,
            deterministic=True,
            preferences=[preference],
        )[0]
    prepared = EvaluationPolicy(
        effective_config,
        policy_name="ppo",
        bootstrap_observation=AssemblySchedulingEnv(effective_config).reset(
            record.instance, preference=preference
        ),
        ppo_agent=agent,
        decode_mode="greedy",
    )
    _, serial = evaluate_instance(
        effective_config,
        instance=record.instance,
        policy_name="ppo",
        prepared_policy=prepared,
        preference=preference,
    )
    assert parallel.action_trace_sha256 == serial["action_trace_sha256"]
    assert parallel.metrics["preference"] == serial["preference"] == {
        "flow": 0.7,
        "cost": 0.2,
        "variance": pytest.approx(0.1),
    }
    assert parallel.metrics["preference_key"] == serial["preference_key"]
    for field in (
        "terminated",
        "truncated",
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
    ):
        assert parallel.metrics[field] == pytest.approx(serial[field])


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
    effective = load_config("configs/e1/single_flow.json")
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
    assert all(report["generator_version"] == "1.3.0" for report in reports)
    assert reports[1]["cache_hit_count"] == 20
    assert reports[2]["cache_hit_count"] == 20
