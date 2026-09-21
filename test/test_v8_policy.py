from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.network import infer_checkpoint_network_spec
from agent.ppo.network_v8 import ObjectiveExpert, SimplexMonotoneRanker
from environment import (
    AssemblySchedulingEnv,
    bounded_quality_score,
    proxy_return_from_metrics,
    quality_preference_for_episode,
)


@pytest.mark.parametrize("feature_count", (1, 3, 4))
def test_equal_direct_inputs_have_feature_count_invariant_scale(feature_count):
    ranker = SimplexMonotoneRanker([-1] * feature_count)
    value = ranker(torch.full((1, feature_count), 0.6))
    assert value.item() == pytest.approx(-np.tanh(0.6), abs=1e-7)
    weights = ranker.normalized_weights()
    assert torch.all(weights > 0)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-7)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / feature_count))


def test_objective_expert_ranges_and_parameterization():
    expert = ObjectiveExpert([-1, 1, -1], hidden_dim=8)
    action = torch.randn(17, 8)
    features = torch.rand(17, 3)
    direct, context, combined = expert(action, features)
    assert torch.all(direct >= -1) and torch.all(direct <= 1)
    assert torch.all(context >= -1) and torch.all(context <= 1)
    assert torch.all(combined >= -2) and torch.all(combined <= 2)
    assert expert.direct_ranker._parameters.keys() == {"theta"}
    assert not any(
        token in name
        for name, _ in expert.direct_ranker.named_parameters()
        for token in ("bias", "temperature", "scale")
    )


def test_one_hot_mixer_ignores_inactive_experts():
    experts = torch.tensor([[0.25, -0.4, 1.2]], requires_grad=True)
    preference = torch.tensor([1.0, 0.0, 0.0])
    base = (experts * preference).sum()
    base.backward()
    assert base.item() == pytest.approx(0.25)
    assert experts.grad.tolist() == [[1.0, 0.0, 0.0]]


def test_preference_is_outside_hgnn_but_conditions_actor_and_critic(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance, preference=(1, 0, 0))
    cost_observation = replace(observation, preference=np.asarray([0, 1, 0]))
    mask = environment.get_action_mask()
    network = build_actor_critic(observation, config["network"])
    network.eval()

    _, flow_nodes, flow_global, flow_graph = network.encode_graph(
        [observation], device="cpu"
    )
    _, cost_nodes, cost_global, cost_graph = network.encode_graph(
        [cost_observation], device="cpu"
    )
    assert torch.equal(flow_global, cost_global)
    assert torch.equal(flow_graph, cost_graph)
    assert all(torch.equal(flow_nodes[name], cost_nodes[name]) for name in flow_nodes)

    flow_logits, flow_value = network(observation, mask, device="cpu")
    cost_logits, cost_value = network(cost_observation, mask, device="cpu")
    assert not torch.allclose(flow_logits[~torch.as_tensor(mask)], cost_logits[~torch.as_tensor(mask)])
    assert not torch.allclose(flow_value, cost_value)


def test_single_stage_reward_is_unshaped_and_telescopes(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, preference=(7, 2, 1))
    policy = HeuristicPolicy()
    reward_return = 0.0
    shaping_seen = 0.0
    while not (environment.terminated or environment.truncated):
        _, reward, _, _, _ = environment.step(policy.select_action(environment))
        reward_return += reward.scalarize(config["reward"])
        shaping_seen += abs(reward.feasibility_shaping)
    metrics = environment.metrics()
    assert metrics["terminated"] is True
    assert metrics["truncated"] is False
    initial = metrics["initial_objectives"]
    initial_score = bounded_quality_score(
        initial["flow"], initial["cost"], initial["variance"], config, preference=(7, 2, 1)
    )
    terminal_score = bounded_quality_score(
        metrics["flow_time_objective"],
        metrics["reconfiguration_cost"],
        metrics["worker_load_variance"],
        config,
        preference=(7, 2, 1),
    )
    expected = (
        metrics["operation_progress"]
        - metrics["initial_progress"]
        - terminal_score
        + initial_score
    )
    assert shaping_seen == 0.0
    assert reward_return == pytest.approx(expected, abs=1e-8)
    assert metrics["preference_quality_score"] == pytest.approx(terminal_score)
    assert metrics["raw_preference_quality_score"] == pytest.approx(terminal_score)
    assert proxy_return_from_metrics(
        metrics,
        config,
        preference=(7, 2, 1),
    ) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "preference",
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)
def test_terminal_failure_uses_unit_quality_bound_and_telescopes(
    config,
    fixed_instance,
    preference,
):
    truncated_config = deepcopy(config)
    truncated_config["environment"]["max_decisions"] = 3
    environment = AssemblySchedulingEnv(truncated_config)
    environment.reset(fixed_instance, preference=preference)
    initial = environment.metrics()["initial_objectives"]
    initial_score = bounded_quality_score(
        initial["flow"],
        initial["cost"],
        initial["variance"],
        truncated_config,
        preference=preference,
    )

    policy = HeuristicPolicy()
    reward_return = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = environment.step(
            policy.select_action(environment)
        )
        reward_return += reward.scalarize(truncated_config["reward"])
    metrics = environment.metrics()

    assert terminated is False
    assert truncated is True
    assert metrics["preference_quality_score"] == 1.0
    assert metrics["raw_preference_quality_score"] < 1.0
    expected = (
        metrics["operation_progress"]
        - metrics["initial_progress"]
        + initial_score
        - 1.0
    )
    assert reward_return == pytest.approx(
        expected,
        abs=1e-8,
    )
    assert proxy_return_from_metrics(
        metrics,
        truncated_config,
        preference=preference,
    ) == pytest.approx(expected, abs=1e-8)


def test_quality_preference_quota_is_deterministic(config):
    points = [
        quality_preference_for_episode(
            config, algorithm_seed=11, quality_episode_index=index
        )
        for index in range(20)
    ]
    assert [point.as_tuple() for point in points[:6]] == [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]
    assert all(sum(point.as_tuple()) == pytest.approx(1.0) for point in points)
    repeated = quality_preference_for_episode(
        config, algorithm_seed=11, quality_episode_index=13
    )
    assert repeated == points[13]


def test_v8_rejects_v7_checkpoint_spec():
    with pytest.raises(ValueError, match="V7"):
        infer_checkpoint_network_spec(
            {"network_spec": {"policy_head_version": 7, "observation_schema_version": 4}}
        )


def test_v8_checkpoint_round_trip_preserves_normalization_manifest_hash(
    config, fixed_instance, tmp_path
):
    manifest_sha = "a" * 64
    network_config = dict(config["network"])
    network_config["normalization_manifest_sha256"] = manifest_sha
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = PPOAgent(
        build_actor_critic(observation, network_config),
        config["ppo"],
        device="cpu",
    )
    checkpoint = tmp_path / "v8.pt"
    agent.save(
        checkpoint,
        metadata={"normalization_manifest_sha256": manifest_sha},
    )
    clone = PPOAgent(
        build_actor_critic(observation, network_config),
        config["ppo"],
        device="cpu",
    )
    metadata = clone.load(checkpoint)
    assert metadata["normalization_manifest_sha256"] == manifest_sha
    assert clone.network.network_spec()["normalization_manifest_sha256"] == manifest_sha

    incompatible_config = dict(network_config)
    incompatible_config["normalization_manifest_sha256"] = "b" * 64
    incompatible = PPOAgent(
        build_actor_critic(observation, incompatible_config),
        config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="normalization_manifest_sha256"):
        incompatible.load(checkpoint)
