from __future__ import annotations

import torch
import pytest

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv
from result import result_schema_version
from train import resolve_summary_checkpoint


def _network():
    config = load_config("configs/v7/e2_5_safe_monotone_flow_gate.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=(0.5, 0.3, 0.2))
    return config, observation, build_actor_critic(observation, config["network"])


def test_e2_5_config_and_scales_are_bounded() -> None:
    config, _, network = _network()
    assert config["training"]["parallel_envs"] == 10
    assert result_schema_version(config) == "4.6.0"
    assert network.production_gate_version == "state_only_monotone_flow_commit_gate_v2"
    assert 0.5 < float(network.production_preference_action_scale()) < 3.0
    assert 0.1 < float(network.worker_preference_action_scale()) < 2.0
    for name, scale in (("k1", 1.0), ("k2", 2.0), ("k3", 3.0)):
        calibration = load_config(f"configs/v7/e2_5_calibration_{name}_seed101.json")
        assert calibration["seed"] == 101
        assert calibration["training"]["episodes"] == 500
        assert calibration["network"]["production_gate"]["flow_commit_residual"]["scale"] == scale


def test_e2_5_gate_is_identical_at_low_flow_and_monotone_above_threshold() -> None:
    _, observation, network = _network()
    base = network._state_only_production_gate_logits(
        observation, dtype=torch.float32, device="cpu"
    )
    network.set_production_flow_commit_residual_enabled(True)
    low, low_boost = network._final_production_gate_logits(
        base, torch.tensor([0.2, 0.7, 0.1])
    )
    high, high_boost = network._final_production_gate_logits(
        base, torch.tensor([0.8, 0.1, 0.1])
    )
    assert torch.equal(low, base)
    assert float(low_boost) == 0.0
    assert float(high[0] - high[1]) >= float(low[0] - low[1])
    assert float(high[1]) == float(base[1])
    assert float(high_boost) > 0.0


def test_e2_5_freeze_and_checkpoint_recovery(tmp_path) -> None:
    config, observation, network = _network()
    agent = PPOAgent(network, config["ppo"], device="cpu")
    network.set_production_state_gate_frozen(True)
    network.set_production_flow_commit_residual_enabled(True)
    checkpoint = tmp_path / "e2_5.pt"
    agent.save(checkpoint)
    clone_network = build_actor_critic(observation, config["network"])
    clone = PPOAgent(clone_network, config["ppo"], device="cpu")
    clone.load(checkpoint)
    assert clone_network.production_state_gate_frozen
    assert clone_network.production_flow_commit_residual_active
    assert all(not parameter.requires_grad for parameter in clone_network.production_state_gate.parameters())
    old = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    old_observation = AssemblySchedulingEnv(old).reset(
        load_instance_pickle(project_path(old["paths"]["instance_cache"])),
        preference=(0.5, 0.3, 0.2),
    )
    old_agent = PPOAgent(build_actor_critic(old_observation, old["network"]), old["ppo"], device="cpu")
    with pytest.raises(ValueError, match="checkpoint .*incompatible"):
        old_agent.load(checkpoint)


def test_e2_5_worker_direct_flow_and_cost_remain_zero() -> None:
    _, _, network = _network()
    components = network._safe_worker_variance_preference_logit_components(
        torch.tensor([2.0, 1.0]), torch.tensor([True, True]), torch.tensor([1.0, 1.0, 1.0])
    )
    assert torch.equal(components[:, :2], torch.zeros_like(components[:, :2]))


def test_e2_5_separate_scale_gradients_do_not_cross() -> None:
    _, _, network = _network()
    feasible = torch.tensor([True, True])
    production = network._direct_preference_logits(
        torch.tensor([[1.0, 3.0], [3.0, 1.0]]), feasible,
        torch.tensor([0.5, 0.5, 0.0]), scale=network.production_preference_action_scale(),
    ).sum()
    production.backward()
    assert network.production_preference_action_scale_raw.grad is not None
    assert network.worker_preference_action_scale_raw.grad is None
    network.zero_grad(set_to_none=True)
    worker = network._safe_worker_variance_preference_logit_components(
        torch.tensor([1.0, 3.0]), feasible, torch.tensor([0.0, 0.0, 1.0])
    ).sum()
    worker.backward()
    assert network.worker_preference_action_scale_raw.grad is not None
    assert network.production_preference_action_scale_raw.grad is None


def test_e2_5_summary_checkpoint_paths_support_relative_and_legacy_absolute(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    assert resolve_summary_checkpoint(summary, "accepted_checkpoint.pt") == tmp_path / "accepted_checkpoint.pt"
    absolute = tmp_path / "legacy.pt"
    assert resolve_summary_checkpoint(summary, str(absolute)) == absolute
