from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agent.ppo import HeteroGraphActorCritic, build_actor_critic, normalize_network_config
from environment import AssemblySchedulingEnv, HeterogeneousGraphObservation


def test_latest_hgnn_forward_contract(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    network = build_actor_critic(observation, config["network"])

    logits, value = network(observation, mask, device="cpu")

    assert isinstance(network, HeteroGraphActorCritic)
    assert isinstance(observation, HeterogeneousGraphObservation)
    assert logits.shape == (len(mask),)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert torch.all(logits[torch.as_tensor(mask)] == torch.finfo(logits.dtype).min)


def test_latest_hgnn_batch_matches_individual(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    network = build_actor_critic(observation, config["network"])
    network.eval()

    expected_logits, expected_value = network(observation, mask, device="cpu")
    logits, values = network.forward_batch(
        [observation, observation], [mask, mask], device="cpu"
    )

    assert torch.allclose(logits[0], expected_logits)
    assert torch.allclose(logits[1], expected_logits, atol=1e-6)
    assert torch.allclose(values, expected_value.expand(2))


def test_latest_network_config_is_fixed_to_v7(config, fixed_instance):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    network = build_actor_critic(observation, config["network"])
    assert network.policy_head_version == 7
    assert network.candidate_context_mode == "bounded_ranker_scale_v7"

    invalid = deepcopy(config["network"])
    invalid["policy_head_version"] = 6
    with pytest.raises(ValueError, match="policy_head_version"):
        build_actor_critic(observation, invalid)


def test_latest_network_rejects_removed_context_mode(config, fixed_instance):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    invalid = deepcopy(config["network"])
    invalid["candidate_context_mode"] = "removed_context_mode"
    with pytest.raises(ValueError, match="bounded_ranker_scale_v7"):
        build_actor_critic(observation, invalid)


def test_network_spec_records_public_observation_schema(config, fixed_instance):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    network = build_actor_critic(observation, config["network"])
    spec = network.network_spec()

    assert spec["observation_schema_version"] == 4
    assert spec["policy_head_version"] == 7
    assert spec["candidate_context_mode"] == "bounded_ranker_scale_v7"
