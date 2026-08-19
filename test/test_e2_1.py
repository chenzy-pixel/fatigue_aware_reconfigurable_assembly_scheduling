from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import (
    training_base_instance_count,
    training_episode_assignment,
)
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv, PreferenceVector
from train import TrainingPhaseController, _pareto_snapshot, _preference_key


def _e2_1_network():
    config = load_config("configs/v7/e2_1_preference_pareto.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=(1.0, 0.0, 0.0))
    network = build_actor_critic(observation, config["network"])
    return config, environment, observation, network


def test_direct_preference_vertices_rank_corresponding_objectives() -> None:
    _, _, _, network = _e2_1_network()
    objectives = torch.tensor(
        [[1.0, 5.0, 9.0], [5.0, 1.0, 5.0], [9.0, 9.0, 1.0]]
    )
    feasible = torch.tensor([True, True, True])
    expected = (0, 1, 2)
    for column, action in enumerate(expected):
        preference = torch.zeros(3)
        preference[column] = 1.0
        logits = network._direct_preference_logits(
            objectives, feasible, preference
        )
        assert int(torch.argmax(logits)) == action


def test_direct_preference_scale_is_positive_trainable_and_singleton_safe() -> None:
    _, _, _, network = _e2_1_network()
    assert network.preference_action_scale().item() == pytest.approx(1.0)
    objectives = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    logits = network._direct_preference_logits(
        objectives,
        torch.tensor([True, True]),
        torch.tensor([1.0, 0.0, 0.0]),
    )
    logits[0].backward()
    assert network.preference_action_scale_raw.grad is not None
    assert torch.isfinite(network.preference_action_scale_raw.grad)
    with torch.no_grad():
        network.preference_action_scale_raw.fill_(-100.0)
    assert network.preference_action_scale().item() >= 0.1
    singleton = network._direct_preference_logits(
        objectives,
        torch.tensor([True, False]),
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert torch.equal(singleton, torch.zeros_like(singleton))


def test_e2_1_forward_and_checkpoint_round_trip(tmp_path) -> None:
    config, environment, observation, network = _e2_1_network()
    logits, value = network(
        observation, environment.get_action_mask(), device="cpu"
    )
    assert torch.isfinite(logits[~torch.as_tensor(environment.get_action_mask())]).all()
    assert torch.isfinite(value)
    agent = PPOAgent(network, config["ppo"], device="cpu")
    checkpoint = tmp_path / "e2_1.pt"
    agent.save(checkpoint)
    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    clone.load(checkpoint)
    assert clone.policy_head_diagnostics()[
        "policy_head_preference_action_scale"
    ] == pytest.approx(1.0)


def test_grouped_training_assignment_is_complete_and_deterministic() -> None:
    config = load_config("configs/v7/e2_1_preference_pareto.json")
    assert training_base_instance_count(config, 2000) == 400
    first = [training_episode_assignment(config, index) for index in range(10)]
    second = [training_episode_assignment(config, index) for index in range(10)]
    assert first == second
    assert [value.base_instance_index for value in first] == [0] * 5 + [1] * 5
    assert [value.preference_slot for value in first] == list(range(5)) * 2
    assert len({value.preference.as_tuple() for value in first[:5]}) == 5
    with pytest.raises(ValueError, match="divisible"):
        training_base_instance_count(config, 11)


def _candidate_row(
    preference: PreferenceVector,
    objectives: tuple[float, float, float],
) -> dict:
    return {
        "instance_id": "instance-1",
        "terminated": True,
        "truncated": False,
        "schedule_violation_count": 0,
        "maximum_worker_fatigue": 0.5,
        "safe_fatigue_limit": 0.8,
        "flow_time_objective": objectives[0],
        "reconfiguration_cost": objectives[1],
        "worker_load_variance": objectives[2],
        "preference_quality_score": 0.25,
        "preference_key": _preference_key(preference),
    }


def test_pareto_snapshot_deduplicates_and_promotion_guards() -> None:
    config = load_config("configs/v7/e2_1_preference_pareto.json")
    canonical = PreferenceVector(0.5, 0.3, 0.2)
    rows = [
        _candidate_row(canonical, (800.0, 400.0, 10.0)),
        _candidate_row(PreferenceVector(1.0, 0.0, 0.0), (700.0, 500.0, 12.0)),
        _candidate_row(PreferenceVector(0.0, 1.0, 0.0), (800.0, 400.0, 10.0)),
    ]
    snapshot = _pareto_snapshot(
        rows,
        config=config,
        scope="anchors_5",
        update_id=1,
        completed_episodes=10,
        fatigue_tolerance=1e-9,
    )
    assert snapshot["mean_unique_objective_count"] == pytest.approx(2.0)
    controller = TrainingPhaseController.from_config(config)
    controller.phase = "quality"
    assert controller.observe_pareto_snapshot(
        snapshot, completed_episodes=10
    ) == "accepted"

    improved = deepcopy(snapshot)
    improved["mean_hypervolume"] = float(snapshot["mean_hypervolume"]) + 0.01
    improved["canonical_quality"] = 0.25 * 1.005
    assert controller.observe_pareto_snapshot(
        improved, completed_episodes=20
    ) == "promoted"

    bad_quality = deepcopy(improved)
    bad_quality["mean_hypervolume"] = float(improved["mean_hypervolume"]) + 0.01
    bad_quality["canonical_quality"] = 0.30
    assert controller.observe_pareto_snapshot(
        bad_quality, completed_episodes=30
    ) == "not_promoted"

    unsafe = deepcopy(improved)
    unsafe["all_safe"] = False
    assert controller.observe_pareto_snapshot(
        unsafe, completed_episodes=40
    ) == "rejected"
