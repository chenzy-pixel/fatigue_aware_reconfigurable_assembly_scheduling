from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config
from e2_preference_analysis import (
    E1_SAMPLING_SEEDS,
    analyze_candidate_rows,
    e2_preference_grid,
)
from environment import (
    AssemblySchedulingEnv,
    PreferenceVector,
    normalize_preference,
    preference_config,
    proxy_return_from_metrics,
    sample_episode_preference,
)
from eval import evaluate_dataset
from result import result_schema_version


@pytest.fixture()
def e2_config():
    return load_config("configs/v7/e2_preference_conditioned.json")


def test_preference_validation_config_and_grid(e2_config):
    assert normalize_preference([0.5, 0.3, 0.2]).as_dict() == {
        "flow": 0.5,
        "cost": 0.3,
        "variance": 0.2,
    }
    for invalid in (
        [0.5, 0.5],
        [0.5, 0.3, 0.3],
        [-0.1, 0.6, 0.5],
        [float("nan"), 0.5, 0.5],
    ):
        with pytest.raises((ValueError, TypeError)):
            normalize_preference(invalid)
    normalized = preference_config(e2_config)
    assert normalized["enabled"] is True
    assert normalized["sampler"]["dirichlet_probability"] == 0.7
    invalid_config = deepcopy(e2_config)
    invalid_config["network"]["preference_conditioning"] = "none"
    with pytest.raises(ValueError, match="separate_encoder_v1"):
        preference_config(invalid_config)
    grid = e2_preference_grid()
    assert len(grid) == 22
    assert PreferenceVector(0.5, 0.3, 0.2) in grid
    assert sum(point.flow == 1.0 for point in grid) == 1
    assert len(E1_SAMPLING_SEEDS) + 1 == len(grid)
    assert result_schema_version(e2_config) == "4.2.0"


def test_preference_sampling_is_episode_stable_and_rng_isolated(e2_config):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    first = sample_episode_preference(
        e2_config, algorithm_seed=11, episode_index=0
    )
    second = sample_episode_preference(
        e2_config, algorithm_seed=11, episode_index=0
    )
    assert first == second
    assert first[1] == "dirichlet"
    assert first[0].as_tuple() == pytest.approx(
        (0.2799854953164442, 0.5887001724786334, 0.1313143322049223)
    )
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "preference",
    [
        PreferenceVector(1.0, 0.0, 0.0),
        PreferenceVector(0.0, 1.0, 0.0),
        PreferenceVector(0.0, 0.0, 1.0),
        PreferenceVector(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        PreferenceVector(0.5, 0.3, 0.2),
    ],
)
def test_observation_copy_and_reward_identity_follow_episode_preference(
    e2_config,
    fixed_instance,
    preference,
):
    environment = AssemblySchedulingEnv(e2_config)
    observation = environment.reset(fixed_instance, preference=preference)
    copied = observation.copy()
    assert copied.preference.tolist() == pytest.approx(preference.as_tuple())
    assert not np.shares_memory(copied.preference, observation.preference)

    policy = HeuristicPolicy()
    reward_sum = 0.0
    shaping_sum = 0.0
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        observation, reward, _, _, _ = environment.step(action)
        reward_sum += reward.scalarize(e2_config["reward"], "quality")
        shaping_sum += reward.feasibility_shaping
    metrics = environment.metrics()
    expected = proxy_return_from_metrics(
        metrics,
        e2_config["reward"],
        "quality",
        preference=preference,
    )
    assert reward_sum - shaping_sum == pytest.approx(expected, abs=1e-8)
    assert metrics["preference"] == preference.as_dict()


def test_e2_network_conditions_every_actor_path_and_critic(
    e2_config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(e2_config)
    observation = environment.reset(
        fixed_instance, preference=PreferenceVector(1.0, 0.0, 0.0)
    )
    network = build_actor_critic(observation, e2_config["network"])
    hidden = int(e2_config["network"]["hidden_dim"])
    assert network.preference_encoder is not None
    assert network.production_scorer[0].in_features == 5 * hidden
    assert network.worker_scorer[0].in_features == 6 * hidden
    assert network.production_defer[0].in_features == 8 * hidden
    assert network.worker_advance[0].in_features == 8 * hidden
    assert network.critic[0].in_features == 8 * hidden
    assert network.network_spec()["observation_schema_version"] == 5

    alternative = replace(
        observation,
        preference=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    mask = environment.get_action_mask()
    first_logits, first_value = network.forward(observation, mask, device="cpu")
    second_logits, second_value = network.forward(alternative, mask, device="cpu")
    assert not torch.allclose(first_logits, second_logits)
    assert not torch.allclose(first_value, second_value)
    finite_logits = first_logits[torch.as_tensor(~mask)]
    (finite_logits.sum() + first_value).backward()
    gradients = [
        parameter.grad
        for parameter in network.preference_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_e1_and_e2_checkpoints_are_explicitly_incompatible(
    e2_config,
    fixed_instance,
    tmp_path,
):
    e2_environment = AssemblySchedulingEnv(e2_config)
    observation = e2_environment.reset(fixed_instance)
    e2_agent = PPOAgent(
        build_actor_critic(observation, e2_config["network"]),
        e2_config["ppo"],
        device="cpu",
    )
    checkpoint = tmp_path / "e2.pt"
    e2_agent.save(checkpoint)
    reloaded_e2 = PPOAgent(
        build_actor_critic(observation, e2_config["network"]),
        e2_config["ppo"],
        device="cpu",
    )
    reloaded_e2.load(checkpoint)
    e1_config = load_config("configs/v7/e1_context_exception.json")
    e1_agent = PPOAgent(
        build_actor_critic(observation, e1_config["network"]),
        e1_config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="architecture"):
        e1_agent.load(checkpoint)
    e1_checkpoint = tmp_path / "e1.pt"
    e1_agent.save(e1_checkpoint)
    with pytest.raises(ValueError, match="architecture"):
        e2_agent.load(e1_checkpoint)


def test_parallel_rollout_uses_episode_index_preferences(
    e2_config,
    fixed_instance,
):
    bootstrap = AssemblySchedulingEnv(e2_config)
    observation = bootstrap.reset(fixed_instance)
    agent = PPOAgent(
        build_actor_critic(observation, e2_config["network"]),
        e2_config["ppo"],
        device="cpu",
    )
    with ParallelEpisodeRunner(
        config=e2_config,
        template=fixed_instance,
        episode_count=2,
        worker_count=2,
    ) as runner:
        rollout = runner.collect_training_batch(
            agent,
            [0, 1],
            gamma=float(e2_config["ppo"]["gamma"]),
            gae_lambda=float(e2_config["ppo"]["gae_lambda"]),
            step_limit=1,
            reward_phase="quality",
        )
    assert [episode.preference for episode in rollout.episodes] == [
        sample_episode_preference(
            e2_config, algorithm_seed=11, episode_index=index
        )[0]
        for index in (0, 1)
    ]
    assert all(
        abs(episode.base_reward_sum - episode.expected_reward) <= 1e-8
        for episode in rollout.episodes
    )


def test_e2_training_rollouts_are_parallelism_invariant(
    e2_config,
    fixed_instance,
):
    effective = deepcopy(e2_config)
    effective["training"]["worker_timeout_seconds"] = 120
    bootstrap = AssemblySchedulingEnv(effective)
    observation = bootstrap.reset(fixed_instance)
    agent = PPOAgent(
        build_actor_critic(observation, effective["network"]),
        effective["ppo"],
        device="cpu",
    )

    def signature(episode):
        return {
            "preference": episode.preference,
            "source": episode.preference_source,
            "actions": tuple(
                transition.action for transition in episode.buffer.transitions
            ),
            "step_count": episode.step_count,
            "base_reward": episode.base_reward_sum,
            "expected_reward": episode.expected_reward,
        }

    with ParallelEpisodeRunner(
        config=effective,
        template=fixed_instance,
        episode_count=10,
        worker_count=10,
    ) as runner:
        by_parallelism = {}
        for parallelism in (1, 2, 10):
            collected = {}
            for start in range(0, 10, parallelism):
                batch = runner.collect_training_batch(
                    agent,
                    list(range(start, min(start + parallelism, 10))),
                    gamma=float(effective["ppo"]["gamma"]),
                    gae_lambda=float(effective["ppo"]["gae_lambda"]),
                    step_limit=4,
                    reward_phase="feasibility",
                )
                collected.update(
                    {
                        episode.episode_index: signature(episode)
                        for episode in batch.episodes
                    }
                )
            by_parallelism[parallelism] = collected

    assert by_parallelism[1] == by_parallelism[2] == by_parallelism[10]
    assert all(
        abs(item["base_reward"] - item["expected_reward"]) <= 1e-8
        for item in by_parallelism[10].values()
    )


def test_e2_terminal_truncation_proxy_uses_episode_preference(
    e2_config,
    fixed_instance,
):
    effective = deepcopy(e2_config)
    effective["environment"]["max_decisions"] = 1
    preference = PreferenceVector(0.0, 1.0, 0.0)
    environment = AssemblySchedulingEnv(effective)
    environment.reset(fixed_instance, preference=preference)
    _, reward, _, truncated, _ = environment.step(
        HeuristicPolicy().select_action(environment)
    )
    assert truncated
    metrics = environment.metrics()
    expected = proxy_return_from_metrics(
        metrics,
        effective["reward"],
        "quality",
        preference=preference,
    )
    assert (
        reward.scalarize(effective["reward"], "quality")
        - reward.feasibility_shaping
    ) == pytest.approx(expected, abs=1e-8)


def test_e2_evaluation_schema_records_requested_preference(e2_config):
    rows, _, _, aggregate = evaluate_dataset(
        e2_config,
        dataset_name="validation",
        policy_name="heuristic",
        instance_limit=1,
        preference=PreferenceVector(0.0, 1.0, 0.0),
    )
    assert aggregate["evaluation_schema_version"] == "4.2.0"
    assert rows[0]["w_flow"] == 0.0
    assert rows[0]["w_cost"] == 1.0
    assert rows[0]["w_variance"] == 0.0
    assert rows[0]["quality_score"] != rows[0]["preference_quality_score"]


def _analysis_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grid = e2_preference_grid()
    for arm in ("e1", "e2"):
        for index in range(22):
            preference = grid[index]
            is_e1_greedy = arm == "e1" and index == 0
            rows.append(
                {
                    "dataset": "test",
                    "algorithm_seed": 11,
                    "instance_id": "case-1",
                    "arm": arm,
                    "candidate_id": (
                        "e1_greedy"
                        if is_e1_greedy
                        else f"{arm}_{index}"
                    ),
                    "candidate_source": (
                        "greedy"
                        if is_e1_greedy
                        else "sampled"
                        if arm == "e1"
                        else "preference_greedy"
                    ),
                    "terminated": True,
                    "truncated": False,
                    "schedule_violation_count": 0,
                    "w_flow": (
                        0.5 if arm == "e1" else preference.flow
                    ),
                    "w_cost": 0.3 if arm == "e1" else preference.cost,
                    "w_variance": (
                        0.2 if arm == "e1" else preference.variance
                    ),
                    "flow_time_objective": 100.0
                    + (index if arm == "e1" else 20.0 * preference.cost),
                    "reconfiguration_cost": 80.0
                    + (21 - index if arm == "e1" else 20.0 * preference.flow),
                    "worker_load_variance": 5.0
                    + (index % 3 if arm == "e1" else 2.0 * preference.flow),
                    "quality_score": 0.25 + (0.01 if arm == "e2" else 0.0),
                    "action_trace_sha256": f"{arm}-trace-{index}",
                }
            )
    canonical_index = grid.index(PreferenceVector(0.5, 0.3, 0.2))
    rows[canonical_index]["candidate_id"] = f"e1_{canonical_index}"
    return rows


def test_equal_budget_analysis_reports_fronts_and_canonical_score():
    annotated, instances, seeds, summary = analyze_candidate_rows(
        _analysis_rows()
    )
    assert len(annotated) == 44
    assert len(instances) == 1
    assert len(seeds) == 1
    assert summary["true_pareto_claim"] is False
    assert summary["candidate_design"]["candidate_budget_per_instance"] == 22
    assert instances[0]["e2_unique_action_traces"] == 22
    assert "e2_flow_normalized_front_span" in instances[0]
    assert "win_tie_loss" in summary["statistics"]["test"]["hypervolume"]
    assert "preference_response_spearman" in summary["statistics"]["test"]


def test_equal_budget_analysis_filters_invalid_rollouts_without_changing_budget():
    rows = _analysis_rows()
    invalid = next(row for row in rows if row["candidate_id"] == "e2_1")
    invalid["terminated"] = False
    invalid["truncated"] = True

    annotated, instances, _, summary = analyze_candidate_rows(rows)

    assert len(annotated) == 44
    assert summary["valid_candidate_row_count"] == 43
    assert summary["invalid_candidate_row_count"] == 1
    assert instances[0]["e2_valid_candidates"] == 21
    rejected = next(row for row in annotated if row["candidate_id"] == "e2_1")
    assert rejected["valid_candidate"] is False
    assert rejected["is_union_pareto"] is False
