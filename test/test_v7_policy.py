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


def test_v8_residual_gates_initialize_at_half():
    config, _, observation = _environment()
    network = build_actor_critic(observation, config["network"])
    assert torch.sigmoid(network.production_residual_gate).item() == pytest.approx(0.5)
    assert torch.sigmoid(network.worker_residual_gate).item() == pytest.approx(0.5)
    assert network.residual_std_floor == pytest.approx(1e-3)


def test_previous_action_semantics_checkpoint_is_rejected():
    config, environment, observation = _environment()
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="V7|network_spec"):
        agent.load(project_path(ACCEPTED), load_optimizer=False)
    assert agent.network.network_spec()["observation_schema_version"] == 5


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
    payload["network_spec"]["policy_head_version"] = 7
    payload["network_spec"]["observation_schema_version"] = 4
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="V7"):
        clone.load(incompatible)


def test_current_checkpoint_hash_is_frozen():
    digest = hashlib.sha256(project_path(ACCEPTED).read_bytes()).hexdigest()
    assert digest == "084184afa53cebfb04d691f5f70a9abb630a1aa0d7f4ecfdb28caeca87299533"
