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
from train import (
    TrainingPhaseController,
    ValidationStabilityController,
    _apply_ablation_variant,
)
from result import evaluation_selection_key


def _run(environment: AssemblySchedulingEnv, phase: str) -> dict[str, float]:
    policy = HeuristicPolicy()
    totals = {"training": 0.0, "base": 0.0, "shaping": 0.0}
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        _, reward, _, _, _ = environment.step(action)
        totals["training"] += reward.scalarize(
            environment.config["reward"],
            phase,
        )
        totals["base"] += reward.base_scalarize(
            environment.config["reward"],
            phase,
        )
        totals["shaping"] += reward.feasibility_shaping
    return totals


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
    completion = (order_count - 1) / order_count
    best_incomplete = completion - 0.5 - (1.0 - completion)
    assert nearly_worst_complete > 1.5 - 1e-9
    assert best_incomplete <= 0.39
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
    initial_potential = environment.feasibility_potential()
    rewards = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["terminated"]
    assert not metrics["truncated"]
    assert rewards["base"] == pytest.approx(
        proxy_return_from_metrics(metrics, config["reward"], phase)
    )
    assert rewards["training"] == pytest.approx(
        rewards["base"] + rewards["shaping"]
    )
    assert rewards["shaping"] == pytest.approx(
        -float(config["reward"]["feasibility_shaping"]["coefficient"])
        * initial_potential
    )
    if phase == "feasibility":
        assert rewards["base"] == pytest.approx(2.0)
    else:
        assert rewards["base"] > 1.5


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
    rewards = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["truncated"]
    assert rewards["base"] == pytest.approx(
        proxy_return_from_metrics(metrics, limited["reward"], phase)
    )
    assert rewards["training"] == pytest.approx(
        rewards["base"] + rewards["shaping"]
    )
    assert rewards["base"] < 1.0


@pytest.mark.parametrize("phase", ["feasibility", "quality"])
def test_proxy_reward_telescopes_for_horizon_truncation(
    config,
    fixed_instance,
    phase,
):
    short_horizon = replace(fixed_instance, horizon=0.1)
    environment = AssemblySchedulingEnv(config)
    environment.reset(short_horizon)
    rewards = _run(environment, phase)
    metrics = environment.metrics()
    assert metrics["truncated"]
    assert metrics["terminal_reason"] == "horizon"
    assert rewards["base"] == pytest.approx(
        proxy_return_from_metrics(metrics, config["reward"], phase)
    )
    assert rewards["training"] == pytest.approx(
        rewards["base"] + rewards["shaping"]
    )
    assert rewards["base"] < 1.0


def test_legacy_reward_configuration_remains_supported(config):
    legacy = deepcopy(config["reward"])
    legacy.pop("mode")
    vector = RewardVector(
        flow=-3600.0,
        cost=-1000.0,
        variance=-100.0,
        truncation=-1.0,
        unfinished=-1.0,
        feasibility_shaping=123.0,
    )
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


def test_validation_stability_rollback_thresholds(config):
    controller = ValidationStabilityController.from_config(config)
    best_score = (-1.0, 10.0, 10.0, 2.0)
    result = controller.observe_greedy(
        best_score,
        1.0,
        completed_episodes=10,
        feasibility_phase=True,
    )
    assert result["improved"]
    assert not result["rollback"]

    result = controller.observe_greedy(
        (-0.95, 10.0, 10.0, 2.0),
        0.95,
        completed_episodes=20,
        feasibility_phase=True,
    )
    assert not result["rollback"]

    result = controller.observe_greedy(
        (-0.90, 10.0, 10.0, 2.0),
        0.90,
        completed_episodes=30,
        feasibility_phase=True,
    )
    assert not result["rollback"]

    result = controller.observe_greedy(
        (-0.89, 10.0, 10.0, 2.0),
        0.89,
        completed_episodes=40,
        feasibility_phase=True,
    )
    assert result["rollback"]
    assert controller.feasibility_rollbacks == 1
    assert controller.best_episode == 10

    for episode in (50, 60, 70):
        result = controller.observe_greedy(
            (-0.80, 10.0, 10.0, 2.0),
            0.80,
            completed_episodes=episode,
            feasibility_phase=True,
        )
        assert not result["rollback"]
    assert controller.rollback_cooldown_remaining == 0
    assert controller.rollback_cooldown_validation_count == 3
    assert controller.rollback_cooldown_blocked_count == 2


def test_validation_plateau_decay_and_minimum_learning_rate(config):
    controller = ValidationStabilityController.from_config(config)
    controller.observe_greedy(
        (-1.0, 10.0, 10.0, 2.0),
        1.0,
        completed_episodes=1,
        feasibility_phase=False,
    )
    for validation in range(2, 17):
        result = controller.observe_greedy(
            (-0.9, 20.0, 10.0, 2.0),
            0.9,
            completed_episodes=validation,
            feasibility_phase=False,
        )
    assert result["learning_rate_decay_applied"]
    assert controller.current_learning_rate == pytest.approx(5e-5)

    for cycle in range(3):
        for offset in range(15):
            controller.observe_greedy(
                (-0.9, 20.0, 10.0, 2.0),
                0.9,
                completed_episodes=20 + cycle * 15 + offset,
                feasibility_phase=False,
            )
    assert controller.current_learning_rate == pytest.approx(2.5e-5)
    assert controller.learning_rate_decays == 2


def test_problem_unfinished_order_penalty_is_unchanged(fixed_instance):
    assert fixed_instance.unfinished_order_penalty == pytest.approx(240.0)


@pytest.mark.parametrize(
    (
        "variant", "shaping", "consecutive", "cooldown", "patience", "floor",
        "promotion",
    ),
    (
        ("E1", False, 1, 0, 10, 1e-5, "completion_only"),
        ("E2", True, 1, 0, 10, 1e-5, "completion_only"),
        ("E3", True, 2, 3, 15, 2.5e-5, "completion_only"),
        ("R11", False, 2, 3, 10, 1e-5, "completion_only"),
        ("S11", True, 2, 3, 10, 1e-5, "completion_only"),
        ("L11", True, 2, 3, 15, 2.5e-5, "completion_only"),
        ("Q11", True, 2, 3, 15, 2.5e-5, "score_improving"),
    ),
)
def test_ablation_variants_are_fixed_600_episode_seed11_runs(
    config,
    variant,
    shaping,
    consecutive,
    cooldown,
    patience,
    floor,
    promotion,
):
    effective = deepcopy(config)
    _apply_ablation_variant(effective, variant)
    rollback = effective["training"]["validation_control"][
        "feasibility_rollback"
    ]
    plateau = effective["training"]["validation_control"][
        "learning_rate_plateau"
    ]
    assert effective["seed"] == 11
    assert effective["training"]["episodes"] == 600
    assert effective["training"]["validation_interval_episodes"] == 10
    assert effective["reward"]["feasibility_shaping"]["enabled"] is shaping
    assert rollback["consecutive_validations"] == consecutive
    assert rollback["cooldown_validations"] == cooldown
    assert plateau["patience_validations"] == patience
    assert plateau["minimum"] == pytest.approx(floor)
    assert (
        effective["training"]["two_stage"]["quality_checkpoint_promotion"]
        == promotion
    )
    milestones = effective["training"]["validation_control"]["sampled"].get(
        "episode_milestones"
    )
    assert milestones == (
        [200, 400] if variant in {"R11", "S11", "L11", "Q11"} else None
    )


def test_score_improving_quality_promotion_rejects_non_improvement(config):
    effective = deepcopy(config)
    effective["training"]["two_stage"][
        "quality_checkpoint_promotion"
    ] = "score_improving"
    effective["training"]["two_stage"]["consecutive_validations"] = 1
    controller = TrainingPhaseController.from_config(effective)

    transition_score = (-1.0, 100.0, 10.0, 2.0)
    assert (
        controller.observe_validation(
            1.0, completed_episodes=10, score=transition_score
        )
        == "transition"
    )
    assert (
        controller.observe_validation(
            1.0,
            completed_episodes=20,
            score=(-1.0, 110.0, 1.0, 1.0),
        )
        == "rejected"
    )
    assert (
        controller.observe_validation(
            1.0,
            completed_episodes=30,
            score=(-1.0, 90.0, 20.0, 20.0),
        )
        == "accepted"
    )
    assert controller.accepted_quality_updates == 1
    assert controller.rejected_quality_updates == 1


def test_e0_ablation_cannot_be_retrained(config):
    with pytest.raises(ValueError, match="must not be retrained"):
        _apply_ablation_variant(deepcopy(config), "E0")
