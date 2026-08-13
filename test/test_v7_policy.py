from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from environment import (
    CAPABLE_EDGE,
    SERVICE_CANDIDATE_EDGE,
    AssemblySchedulingEnv,
)
from environment.env import WorkerTaskSnapshot
from environment.types import ReconfigurationStage


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


def test_e2_commit_set_common_shift_preserves_pair_order_and_zero_alignment():
    config, environment, observation = _environment("e2_commit_set")
    network = build_actor_critic(observation, config["network"])
    mask = environment.get_action_mask()
    logits_zero, _ = network(observation, mask, device="cpu")
    assert network.production_commit_set is not None
    output = network.production_commit_set[-1]
    assert torch.equal(output.weight, torch.zeros_like(output.weight))
    assert torch.equal(output.bias, torch.zeros_like(output.bias))
    with torch.no_grad():
        output.bias.fill_(1.25)
    logits_shifted, _ = network(observation, mask, device="cpu")
    feasible = torch.as_tensor(~mask[:-1])
    assert torch.allclose(
        logits_shifted[:-1][feasible] - logits_zero[:-1][feasible],
        torch.full_like(logits_zero[:-1][feasible], 1.25),
        atol=1e-6,
    )
    assert torch.equal(
        torch.argsort(logits_zero[:-1][feasible]),
        torch.argsort(logits_shifted[:-1][feasible]),
    )
    assert float(logits_shifted[-1].detach()) == pytest.approx(
        float(logits_zero[-1].detach())
    )


def test_e3_future_features_are_named_finite_and_monotone_directions_hold():
    config, _, observation = _environment("e3_future_value")
    capability = observation.relations[CAPABLE_EDGE]
    service = observation.relations[SERVICE_CANDIDATE_EDGE]
    for name in (
        "current_wave_target_demand_ratio",
        "future_wave_target_demand_ratio",
        "target_remaining_workload_norm",
        "configured_machine_support_ratio",
        "future_configuration_reuse_value_norm",
        "configuration_opportunity_cost_norm",
        "future_horizon_risk_norm",
    ):
        assert name in capability.feature_names
    for name in (
        "fatigue_headroom_ratio",
        "total_incremental_cost_norm",
        "qualification_opportunity_cost_norm",
        "recovery_eta_norm",
        "remaining_service_capacity_norm",
    ):
        assert name in service.feature_names
    assert np.isfinite(capability.edge_features).all()
    assert np.isfinite(service.edge_features).all()
    network = build_actor_critic(observation, config["network"])
    weights = network.effective_relative_cost_weights()
    assert weights["production"]["processing_time_norm"] < 0.0
    assert weights["production"]["future_configuration_reuse_value_norm"] > 0.0
    assert weights["worker"]["fatigue_headroom_ratio"] > 0.0
    assert weights["worker"]["qualification_opportunity_cost_norm"] < 0.0
    assert all(
        math.isfinite(value)
        for phase in weights.values()
        for value in phase.values()
    )


@pytest.mark.parametrize(
    "arm",
    ("e1_context_exception", "e2_commit_set", "e3_future_value", "full_v7"),
)
def test_v7_checkpoint_round_trip_uses_exact_schema(arm, tmp_path):
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
    assert clone.load(checkpoint) == {
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


def test_e4_conditional_wait_open_and_boundary_rejections(monkeypatch):
    _, environment, _ = _environment("e4_conditional_wait")
    task = WorkerTaskSnapshot(
        task_id="synthetic",
        machine_index=0,
        stage=ReconfigurationStage.WAIT_INS,
        module=environment.instance.modules[0],
    )
    next_tick = environment.current_tick + 1
    monkeypatch.setattr(
        environment, "_next_decision_event_tick", lambda: next_tick
    )
    monkeypatch.setattr(
        environment, "_current_worker_tasks", lambda: (task,)
    )
    monkeypatch.setattr(
        environment,
        "_projected_safe_edges_for_tasks",
        lambda tasks, tick: ((0,),) if tick == environment.current_tick else ((0, 1),),
    )
    monkeypatch.setattr(
        environment,
        "_projected_stage_duration_ticks",
        lambda task, worker, tick: 1,
    )
    monkeypatch.setattr(
        environment,
        "_best_projected_worker_candidate",
        lambda tasks, edges, tick=None: (
            (0.70, 10) if tick is None else (0.64, 9)
        ),
    )
    preview = environment._conditional_worker_wait_preview()
    assert preview is not None
    assert preview.wait_ticks == 1
    assert "legal_pair_gain" in preview.reason
    assert "fatigue_improvement" in preview.reason
    assert "duration_improvement" in preview.reason

    environment._consecutive_conditional_waits = 2
    assert environment._conditional_worker_wait_preview() is None
    environment._consecutive_conditional_waits = 0
    monkeypatch.setattr(
        environment,
        "_next_decision_event_tick",
        lambda: environment.current_tick,
    )
    assert environment._conditional_worker_wait_preview() is None


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
