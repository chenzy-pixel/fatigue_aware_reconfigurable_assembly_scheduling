from __future__ import annotations

import hashlib

import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv


CONFIG = "configs/e1/single_flow.json"
ACCEPTED = "result/runs/v7_2000_e1_seed11/accepted_checkpoint.pt"


def _environment():
    config = load_config(CONFIG)
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    return config, environment, observation


def test_v7_bounded_residual_has_gradient_and_ranker_scaled_bound():
    config, _, observation = _environment()
    network = build_actor_critic(observation, config["network"])
    relative = torch.tensor([-0.5, 0.5])
    raw = torch.tensor([-3.0, 3.0], requires_grad=True)
    feasible = torch.tensor([True, True])
    residual = network._context_residual(
        relative, raw, feasible, network.production_residual_context_gate
    )
    ranker_scale = relative.std(unbiased=False).clamp_min(1e-3)
    bound = (
        torch.sigmoid(network.production_residual_context_gate)
        * 2.0
        * ranker_scale
    )
    assert torch.max(torch.abs(residual)) <= bound + 1e-7
    residual.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert network.production_residual_context_gate.grad is not None


def test_current_checkpoint_strict_load_and_schema_alias():
    config, environment, observation = _environment()
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    metadata = agent.load(project_path(ACCEPTED), load_optimizer=False)
    assert metadata
    assert agent.network.network_spec()["observation_schema_version"] == 3
    logits, value = agent.network(
        observation, environment.get_action_mask(), device="cpu"
    )
    assert logits.shape == (environment.production_action_size,)
    assert torch.isfinite(logits).all()
    assert float(value) == pytest.approx(1.722648024559021)


def test_checkpoint_round_trip_and_incompatible_spec_rejected(tmp_path):
    config, _, observation = _environment()
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    checkpoint = tmp_path / "current.pt"
    agent.save(checkpoint, metadata={"line": "e1"})
    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    metadata = clone.load(checkpoint)
    assert len(metadata.pop("network_weights_sha256")) == 64
    assert metadata["line"] == "e1"

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["network_spec"]["policy_head_version"] = 6
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="policy_head_version"):
        clone.load(incompatible)


def test_current_checkpoint_hash_is_frozen():
    digest = hashlib.sha256(project_path(ACCEPTED).read_bytes()).hexdigest()
    assert digest == "084184afa53cebfb04d691f5f70a9abb630a1aa0d7f4ecfdb28caeca87299533"
