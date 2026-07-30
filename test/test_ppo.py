from __future__ import annotations

import math

import torch

from agent.ppo import PPOAgent, RolloutBuffer, TypedActorCritic
from environment import AssemblySchedulingEnv
from utils import set_seed


def test_ppo_update_changes_parameters_and_checkpoint_reloads(
    config, fixed_instance, tmp_path
):
    set_seed(config["seed"])
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    network = TypedActorCritic(
        observation.feature_dimensions, config["network"]["hidden_dim"]
    )
    agent = PPOAgent(network, config["ppo"], device="cpu")
    buffer = RolloutBuffer()
    for _ in range(16):
        mask = environment.get_action_mask()
        action, log_probability, value = agent.act(observation, mask)
        next_observation, reward, terminated, truncated, _ = environment.step(
            action
        )
        buffer.add(
            observation,
            mask,
            action,
            log_probability,
            value,
            reward.scalarize(config["reward"]),
            terminated or truncated,
        )
        observation = next_observation
        if terminated or truncated:
            break
    buffer.compute_gae(
        last_value=0.0,
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )
    before = [parameter.detach().clone() for parameter in network.parameters()]
    metrics = agent.update(buffer)
    after = list(network.parameters())
    assert all(math.isfinite(value) for value in metrics.values())
    assert {
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "gradient_norm",
        "gradient_norm_max",
        "gradient_clipped_fraction",
        "pre_update_explained_variance",
        "return_mean",
        "return_std",
        "advantage_mean",
        "advantage_std",
        "value_prediction_mean",
        "value_prediction_std",
        "learning_rate",
    }.issubset(metrics)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert 0.0 <= metrics["gradient_clipped_fraction"] <= 1.0
    assert metrics["gradient_norm"] >= 0.0
    assert metrics["gradient_norm_max"] >= metrics["gradient_norm"]
    assert any(
        not torch.equal(old, new.detach()) for old, new in zip(before, after)
    )

    checkpoint = tmp_path / "checkpoint.pt"
    agent.save(checkpoint, metadata={"test": True})
    reloaded_network = TypedActorCritic(
        observation.feature_dimensions, config["network"]["hidden_dim"]
    )
    reloaded = PPOAgent(reloaded_network, config["ppo"], device="cpu")
    metadata = reloaded.load(checkpoint)
    assert metadata == {"test": True}
    for original, restored in zip(
        network.parameters(), reloaded_network.parameters()
    ):
        assert torch.equal(original.detach(), restored.detach())
