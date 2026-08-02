from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, RolloutBuffer, TypedActorCritic
from data.dataset import load_dataset_split
from environment import (
    AssemblySchedulingEnv,
    DecisionType,
    PolicyObservation,
)


def _find_worker_observation(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    policy = HeuristicPolicy()
    for _ in range(100):
        if observation.decision_type == DecisionType.WORKER:
            return observation, environment.get_action_mask()
        action = policy.select_action(environment)
        observation, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            break
    raise AssertionError("test instance did not reach a worker decision")


def test_mixed_variable_size_batch_matches_individual_forward(
    config,
    fixed_instance,
):
    validation_record = load_dataset_split(config, "validation")[0]
    first_environment = AssemblySchedulingEnv(config)
    first_observation = first_environment.reset(fixed_instance)
    first_mask = first_environment.get_action_mask()
    second_environment = AssemblySchedulingEnv(config)
    second_observation = second_environment.reset(
        validation_record.instance
    )
    second_mask = second_environment.get_action_mask()
    worker_observation, worker_mask = _find_worker_observation(
        config,
        fixed_instance,
    )
    observations = [
        first_observation,
        second_observation,
        worker_observation,
    ]
    masks = [first_mask, second_mask, worker_mask]
    network = TypedActorCritic(
        first_observation.feature_dimensions,
        int(config["network"]["hidden_dim"]),
    )
    network.eval()
    individual = [
        network(observation, mask, device="cpu")
        for observation, mask in zip(observations, masks)
    ]
    batch_logits, batch_values = network.forward_batch(
        observations,
        masks,
        device="cpu",
    )
    for index, ((logits, value), mask) in enumerate(
        zip(individual, masks)
    ):
        assert torch.allclose(
            logits,
            batch_logits[index, : len(mask)],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            value,
            batch_values[index],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.all(
            batch_logits[index, len(mask) :]
            == torch.finfo(batch_logits.dtype).min
        )
    agent = PPOAgent(network, config["ppo"], device="cpu")
    actions, _, values = agent.act_batch(
        observations,
        masks,
        deterministic=True,
    )
    assert all(not masks[index][action] for index, action in enumerate(actions))
    assert all(math.isfinite(value) for value in values)
    compact = PolicyObservation.from_observation(first_observation)
    assert not hasattr(compact, "relations")


def test_sampled_batch_uses_independent_reproducible_generator(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    network = TypedActorCritic(
        observation.feature_dimensions,
        int(config["network"]["hidden_dim"]),
    )
    agent = PPOAgent(network, config["ppo"], device="cpu")
    observations = [observation] * 32
    masks = [mask] * 32
    global_state = torch.random.get_rng_state().clone()
    first_generator = torch.Generator(device="cpu").manual_seed(12345)
    second_generator = torch.Generator(device="cpu").manual_seed(12345)

    first = agent.act_batch(
        observations,
        masks,
        generator=first_generator,
    )
    second = agent.act_batch(
        observations,
        masks,
        generator=second_generator,
    )

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_ppo_update_uses_one_batched_forward_per_minibatch(
    config,
    fixed_instance,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["ppo"]["batch_size"] = 4
    effective_config["ppo"]["epochs"] = 2
    environment = AssemblySchedulingEnv(effective_config)
    observation = environment.reset(fixed_instance)
    network = TypedActorCritic(
        observation.feature_dimensions,
        int(effective_config["network"]["hidden_dim"]),
    )
    agent = PPOAgent(network, effective_config["ppo"], device="cpu")
    buffer = RolloutBuffer()
    for _ in range(10):
        mask = environment.get_action_mask()
        action, log_probability, value = agent.act(observation, mask)
        next_observation, reward, terminated, truncated, _ = (
            environment.step(action)
        )
        buffer.add(
            observation,
            mask,
            action,
            log_probability,
            value,
            reward.scalarize(effective_config["reward"]),
            terminated or truncated,
        )
        observation = next_observation
        if terminated or truncated:
            break
    buffer.compute_gae(
        last_value=0.0,
        gamma=float(effective_config["ppo"]["gamma"]),
        gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
    )
    call_count = 0
    original_forward_batch = network.forward_batch

    def counting_forward_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward_batch(*args, **kwargs)

    monkeypatch.setattr(
        network,
        "forward_batch",
        counting_forward_batch,
    )
    metrics = agent.update(buffer)
    expected = (
        math.ceil(len(buffer) / effective_config["ppo"]["batch_size"])
        * effective_config["ppo"]["epochs"]
    )
    assert call_count == expected
    assert all(math.isfinite(value) for value in metrics.values())


def test_gae_is_computed_before_parallel_buffers_are_merged():
    first = RolloutBuffer()
    second = RolloutBuffer()
    observation = PolicyObservation(
        operations=np.zeros((1, 1), dtype=np.float32),
        machines=np.zeros((1, 1), dtype=np.float32),
        workers=np.zeros((1, 1), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        decision_type=DecisionType.PRODUCTION,
    )
    mask = np.array([False, True], dtype=np.bool_)
    first.add(observation, mask, 0, 0.0, 1.0, 1.0, True)
    second.add(observation, mask, 0, 0.0, 10.0, 5.0, True)
    first.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    second.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    combined = RolloutBuffer()
    combined.extend(first)
    combined.extend(second)
    assert [value.advantage for value in combined.transitions] == [
        pytest.approx(0.0),
        pytest.approx(-5.0),
    ]
