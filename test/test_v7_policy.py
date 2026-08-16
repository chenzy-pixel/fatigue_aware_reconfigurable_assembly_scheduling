from __future__ import annotations

import hashlib
import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv


def _environment(arm: str):
    config = load_config(f"configs/v7/{arm}.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    return config, environment, observation


def test_e1_bounded_residual_has_gradient_and_ranker_scaled_bound():
    config, environment, observation = _environment("e1_context_exception")
    network = build_actor_critic(observation, config["network"])
    relative = torch.tensor([-0.5, 0.5])
    raw = torch.tensor([-3.0, 3.0], requires_grad=True)
    feasible = torch.tensor([True, True])
    residual = network._context_residual(
        relative,
        raw,
        feasible,
        network.production_residual_context_gate,
    )
    ranker_scale = relative.std(unbiased=False).clamp_min(1e-3)
    bound = (
        torch.sigmoid(network.production_residual_context_gate)
        * 2.0
        * ranker_scale
    )
    assert torch.max(torch.abs(residual)) <= bound + 1e-7
    residual.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert network.production_residual_context_gate.grad is not None


def test_v7_checkpoint_round_trip_uses_exact_schema(tmp_path):
    arm = "e1_context_exception"
    config, _, observation = _environment(arm)
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    checkpoint = tmp_path / f"{arm}.pt"
    agent.save(checkpoint, metadata={"arm": arm})
    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    metadata = clone.load(checkpoint)
    assert len(metadata.pop("network_weights_sha256")) == 64
    assert metadata == {
        "arm": arm,
        "policy_head_diagnostics": agent.policy_head_diagnostics(),
    }
    v6_config = load_config("configs/v7/c0_v6_control.json")
    v6_clone = PPOAgent(
        build_actor_critic(observation, v6_config["network"]),
        v6_config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="not automatically converted"):
        v6_clone.load(checkpoint)


@pytest.mark.parametrize(
    "seed,action_hash,expected",
    (
        (11, "62471a63f5ea0e7e6924e84f43f612ea394c2de18dcd6aeb5a4401c8a6e861e4", (1160.4, 388.82, 0.9522222222222222)),
        (23, "171d548a41d7ba1cdf25fdd328dda36b52b798aef0a3cf23687b9961c2587f1e", (777.0, 497.5, 0.4966666666666666)),
        (37, "8674633a9d9dbce9ff27dba533fb497c8d1e40b63386f7fea4236d98fd1fa738", (723.1, 457.84, 3.11888888888889)),
        (53, "6939a3ba2ad8761ce6fb40316859266a85ab5ea5046c81c2e8cbea0184286cbc", (701.5, 279.3, 3.805555555555556)),
        (71, "4029cdaa83930d8a498f57b8d62eca713db4aebb69782354e05c0f5c490922bc", (843.5, 571.36, 1.1491666666666658)),
    ),
)
def test_historical_v6_checkpoint_fixed_instance_greedy_regression(
    seed, action_hash, expected
):
    run = project_path(f"result/runs/policy_head_v6_seed{seed}")
    config = load_config(run / "config.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    agent.load(run / "accepted_checkpoint.pt")
    actions = []
    while not (environment.terminated or environment.truncated):
        action, _, _ = agent.act(
            observation,
            environment.get_action_mask(),
            deterministic=True,
        )
        actions.append(action)
        observation, _, _, _, _ = environment.step(action)
    digest = hashlib.sha256(
        ",".join(str(action) for action in actions).encode("ascii")
    ).hexdigest()
    metrics = environment.metrics()
    assert digest == action_hash
    assert metrics["terminated"] is True
    assert metrics["truncated"] is False
    assert (
        metrics["flow_time_objective"],
        metrics["reconfiguration_cost"],
        metrics["worker_load_variance"],
    ) == pytest.approx(expected)
    assert environment.validate_schedule() == []
