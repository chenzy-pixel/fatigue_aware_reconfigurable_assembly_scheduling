from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from agent.baselines import HeuristicPolicy
from environment import (
    AssemblySchedulingEnv,
    RewardVector,
    bounded_quality_score,
    proxy_return_from_metrics,
)
from train import TrainingPhaseController
from result import evaluation_selection_key


def _run(environment: AssemblySchedulingEnv, phase: str) -> float:
    policy = HeuristicPolicy()
    total = 0.0
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        _, reward, _, _, _ = environment.step(action)
        total += reward.scalarize(
            environment.config["reward"],
            phase,
        )
    return total


@pytest.mark.parametrize("order_count", range(12, 19))
def test_complete_proxy_return_strictly_dominates_incomplete(
    config,
    order_count,
):
    reward_config = config["reward"]
    nearly_worst_complete = (
        1.0
        + 1.0
        - float(reward_config["quality_budget"]) * (1.0 - 1e-12)
    )
    best_incomplete = (order_count - 1) / order_count
    assert nearly_worst_complete > 1.5 - 1e-9
    assert best_incomplete < 1.0
    assert nearly_worst_complete > best_incomplete


def test_bounded_quality_score_uses_configured_weights(config):
    reward_config = config["reward"]
    assert bounded_quality_score(0.0, 0.0, 0.0, reward_config) == 0.0
    score = bounded_quality_score(
        reward_config["flow_scale"],
        reward_config["cost_scale"],
        reward_config["variance_scale"],
        reward_config,
    )
    assert score == pytest.approx(0.5)
    assert 0.0 <= bounded_quality_score(
        1e12,
        1e12,
        1e12,
        reward_config,
    ) < 1.0


@pytest.mark.parametrize("phase", ["feasibility", "quality"])
def test_proxy_reward_telescopes_for_complete_trajectory(
    config,
    fixed_instance,
    phase,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    reward_sum = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["terminated"]
    assert not metrics["truncated"]
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, config["reward"], phase)
    )
    if phase == "quality":
        assert reward_sum > 1.5


@pytest.mark.parametrize("phase", ["feasibility", "quality"])
def test_proxy_reward_telescopes_for_truncated_trajectory(
    config,
    fixed_instance,
    phase,
):
    limited = deepcopy(config)
    limited["environment"]["max_decisions"] = 1
    environment = AssemblySchedulingEnv(limited)
    environment.reset(fixed_instance)
    reward_sum = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["truncated"]
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, limited["reward"], phase)
    )
    assert reward_sum < 1.0


@pytest.mark.parametrize("phase", ["feasibility", "quality"])
def test_proxy_reward_telescopes_for_horizon_truncation(
    config,
    fixed_instance,
    phase,
):
    short_horizon = replace(fixed_instance, horizon=0.1)
    environment = AssemblySchedulingEnv(config)
    environment.reset(short_horizon)
    reward_sum = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["truncated"]
    assert metrics["terminal_reason"] == "horizon"
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, config["reward"], phase)
    )
    assert reward_sum < 1.0


def test_legacy_reward_configuration_remains_supported(config):
    legacy = deepcopy(config["reward"])
    legacy.pop("mode")
    vector = RewardVector(flow=-3600.0, cost=-1000.0, variance=-100.0)
    assert vector.scalarize(legacy) == pytest.approx(-3.0)


def test_checkpoint_selection_keeps_completion_as_highest_priority():
    def metric(value):
        return {"count": 2, "mean": value, "std": 0.0}

    def aggregate(completion_rate, flow, cost, variance):
        return {
            "completion_rate": completion_rate,
            "all_instance_metrics": {
                "flow_time_objective": metric(flow),
                "reconfiguration_cost": metric(cost),
                "worker_load_variance": metric(variance),
            },
        }

    complete = aggregate(1.0, 10_000.0, 10_000.0, 10_000.0)
    incomplete = aggregate(0.99, 0.0, 0.0, 0.0)
    assert evaluation_selection_key(complete) < evaluation_selection_key(
        incomplete
    )
    lower_flow = aggregate(1.0, 100.0, 1000.0, 1000.0)
    lower_cost = aggregate(1.0, 100.0, 10.0, 1000.0)
    assert evaluation_selection_key(lower_cost) < evaluation_selection_key(
        lower_flow
    )


def test_training_phase_requires_three_consecutive_successes(config):
    controller = TrainingPhaseController.from_config(config)
    assert controller.phase == "feasibility"
    assert (
        controller.observe_validation(1.0, completed_episodes=10)
        == "feasibility"
    )
    assert (
        controller.observe_validation(0.95, completed_episodes=20)
        == "feasibility"
    )
    for episode in (30, 40):
        assert (
            controller.observe_validation(1.0, completed_episodes=episode)
            == "feasibility"
        )
    assert (
        controller.observe_validation(1.0, completed_episodes=50)
        == "transition"
    )
    assert controller.phase == "quality"
    assert controller.phase_transition_episode == 50
    assert controller.should_validate(False)
    assert (
        controller.observe_validation(1.0, completed_episodes=60)
        == "accepted"
    )
    assert (
        controller.observe_validation(0.95, completed_episodes=70)
        == "rejected"
    )
    assert controller.accepted_quality_updates == 1
    assert controller.rejected_quality_updates == 1


def test_problem_unfinished_order_penalty_is_unchanged(fixed_instance):
    assert fixed_instance.unfinished_order_penalty == pytest.approx(240.0)
