from __future__ import annotations

import math
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
    _can_reuse_final_sampled_validation,
    _normalized_validation_quality_score,
    _validation_log_row,
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


def test_worker_load_variance_reward_is_emitted_at_assignment(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    policy = HeuristicPolicy()
    assignment_reward = None
    completion_advance_reward = None
    committed_task = None
    for _ in range(500):
        decision_type = observation.decision_type
        action = policy.select_action(environment)
        is_worker_pair = (
            decision_type.value == "WORKER"
            and action != environment.advance_action
        )
        is_worker_advance = (
            decision_type.value == "WORKER"
            and action == environment.advance_action
        )
        before_time = environment.current_time
        observation, reward, terminated, truncated, _ = environment.step(
            action
        )
        if is_worker_pair and assignment_reward is None:
            assignment_reward = reward.variance
            committed_task = next(
                iter(environment._active_committed_worker_tasks)
            )
        if (
            committed_task is not None
            and committed_task
            not in environment._active_committed_worker_tasks
        ):
            assert is_worker_advance
            assert environment.current_time > before_time
            completion_advance_reward = reward.variance
            break
        if terminated or truncated:
            break
    assert assignment_reward is not None
    assert not math.isclose(assignment_reward, 0.0, abs_tol=1e-12)
    assert completion_advance_reward == pytest.approx(0.0)
    metrics = environment.metrics()
    assert metrics["worker_assignment_count"] >= 1
    assert metrics[
        "worker_assignment_nonzero_variance_reward_count"
    ] >= 1
    assert metrics[
        "worker_assignment_variance_reward_abs_sum"
    ] >= abs(float(assignment_reward))


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
    legacy = {
        "flow_weight": 1.0,
        "flow_scale": 3600.0,
        "cost_weight": 1.0,
        "cost_scale": 1000.0,
        "variance_weight": 1.0,
        "variance_scale": 100.0,
    }
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


def test_validation_log_promotion_key_and_q12_score_are_identical():
    def metric(value):
        return {"count": 2, "mean": value, "std": 0.0}

    quality = (
        0.5 * 1200.0 / 2400.0
        + 0.3 * 1000.0 / 2000.0
        + 0.2 * 50.0 / 100.0
    )
    aggregate = {
        "dataset": "validation",
        "instance_count": 2,
        "completed_count": 2,
        "completion_rate": 1.0,
        "truncated_count": 0,
        "schedule_violation_count": 0,
        "completed_metrics": {
            "makespan": metric(1.0),
            "total_flow_time": metric(1200.0),
        },
        "all_instance_metrics": {
            "quality_score": metric(quality),
            "flow_time_objective": metric(1200.0),
            "reconfiguration_cost": metric(1000.0),
            "worker_load_variance": metric(50.0),
        },
        "gap_metrics": {
            name: metric(0.0)
            for name in (
                "relative_heuristic_gap_percent",
                "makespan_heuristic_gap_percent",
                "reconfiguration_cost_heuristic_gap_percent",
                "worker_load_variance_heuristic_gap_percent",
            )
        },
        "total_inference_time_seconds": 0.0,
        "total_solve_time_seconds": 0.0,
    }
    key = evaluation_selection_key(aggregate)
    row = _validation_log_row(aggregate, completed_episodes=10)
    assert key[1] == pytest.approx(quality)
    assert row["mean_quality_score"] == pytest.approx(quality)
    assert _normalized_validation_quality_score(key, {}) == pytest.approx(
        quality
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
        controller.observe_validation(
            1.0,
            completed_episodes=60,
            score=(-1.0, 0.4, 0.0, 0.0),
            normalized_quality_score=0.4,
        )
        == "promoted"
    )
    assert (
        controller.observe_validation(
            0.95,
            completed_episodes=70,
            score=(-0.95, 0.3, 0.0, 0.0),
            normalized_quality_score=0.3,
        )
        == "not_promoted"
    )
    assert controller.accepted_quality_updates == 1
    assert controller.not_promoted_quality_updates == 1


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
            (-0.95, 20.0, 10.0, 2.0),
            0.95,
            completed_episodes=validation,
            feasibility_phase=False,
        )
    assert result["learning_rate_decay_applied"]
    assert controller.current_learning_rate == pytest.approx(5e-5)

    for cycle in range(3):
        for offset in range(15):
            controller.observe_greedy(
                (-0.95, 20.0, 10.0, 2.0),
                0.95,
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
        ("Q12", True, 2, 3, 15, 2.5e-5, "score_improving"),
        ("Q13", True, 2, 3, 15, 2.5e-5, "constrained_weighted"),
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
        [200, 400]
        if variant in {"R11", "S11", "L11", "Q11", "Q12", "Q13"}
        else None
    )


def test_q12_only_changes_q11_quality_scales_and_weights(config):
    original = deepcopy(config)
    q11 = deepcopy(config)
    q12 = deepcopy(config)
    _apply_ablation_variant(q11, "Q11")
    _apply_ablation_variant(q12, "Q12")

    expected = deepcopy(q11)
    expected["training"]["ablation_variant"] = "Q12"
    expected["reward"]["flow_scale"] = 1200.0
    expected["reward"]["cost_scale"] = 1000.0
    expected["reward"]["variance_scale"] = 50.0
    expected["reward"]["quality_weights"] = {
        "flow": 0.5,
        "cost": 0.3,
        "variance": 0.2,
    }

    assert q12 == expected
    assert config == original


def test_q13_only_changes_q11_promotion_constraints(config):
    original = deepcopy(config)
    q11 = deepcopy(config)
    q13 = deepcopy(config)
    _apply_ablation_variant(q11, "Q11")
    _apply_ablation_variant(q13, "Q13")

    expected = deepcopy(q11)
    expected["training"]["ablation_variant"] = "Q13"
    expected["training"]["two_stage"][
        "quality_checkpoint_promotion"
    ] = "constrained_weighted"
    expected["training"]["two_stage"][
        "quality_promotion_constraints"
    ] = {
        "flow_relative_tolerance": 0.005,
        "cost_relative_tolerance": 0.0,
        "variance_relative_tolerance": 0.0,
        "minimum_normalized_score_improvement": 1e-12,
    }

    assert q13 == expected
    assert q13["reward"] == q11["reward"]
    assert config == original


def _q13_controller(config) -> TrainingPhaseController:
    effective = deepcopy(config)
    _apply_ablation_variant(effective, "Q13")
    effective["training"]["two_stage"]["consecutive_validations"] = 1
    return TrainingPhaseController.from_config(effective)


def _transition_q13_controller(config) -> TrainingPhaseController:
    controller = _q13_controller(config)
    assert (
        controller.observe_validation(
            1.0,
            completed_episodes=10,
            score=(-1.0, 100.0, 100.0, 10.0),
            normalized_quality_score=0.3,
        )
        == "transition"
    )
    return controller


def test_constrained_weighted_promotion_accepts_boundary_and_updates_anchor(
    config,
):
    controller = _transition_q13_controller(config)
    candidate = (-1.0, 100.5, 99.0, 9.0)
    assert (
        controller.observe_validation(
            1.0,
            completed_episodes=20,
            score=candidate,
            normalized_quality_score=0.29,
        )
        == "accepted"
    )
    assert controller.accepted_quality_score == candidate
    assert controller.accepted_normalized_quality_score == pytest.approx(0.29)
    assert controller.accepted_quality_episode == 20
    assert controller.last_promotion_diagnostics[
        "promotion_decision_reason"
    ] == "accepted"
    assert controller.last_promotion_diagnostics[
        "promotion_flow_constraint_pass"
    ] is True


@pytest.mark.parametrize(
    ("completion", "score", "quality_score", "reason"),
    (
        (
            0.95,
            (-0.95, 99.0, 99.0, 9.0),
            0.29,
            "completion_below_floor",
        ),
        (
            1.0,
            (-1.0, 100.5000001, 99.0, 9.0),
            0.29,
            "flow_tolerance_exceeded",
        ),
        (1.0, (-1.0, 99.0, 100.0000001, 9.0), 0.29, "cost_regressed"),
        (
            1.0,
            (-1.0, 99.0, 99.0, 10.0000001),
            0.29,
            "variance_regressed",
        ),
        (
            1.0,
            (-1.0, 99.0, 99.0, 9.0),
            0.3,
            "normalized_quality_not_improved",
        ),
        (
            1.0,
            (-1.0, math.inf, 99.0, 9.0),
            None,
            "missing_or_non_finite_promotion_metric",
        ),
    ),
)
def test_constrained_weighted_promotion_rejects_each_failed_guard_without_anchor_update(
    config,
    completion,
    score,
    quality_score,
    reason,
):
    controller = _transition_q13_controller(config)
    anchor = controller.accepted_quality_score
    anchor_quality = controller.accepted_normalized_quality_score
    assert (
        controller.observe_validation(
            completion,
            completed_episodes=20,
            score=score,
            normalized_quality_score=quality_score,
        )
        == "rejected"
    )
    assert controller.accepted_quality_score == anchor
    assert controller.accepted_normalized_quality_score == anchor_quality
    assert controller.accepted_quality_episode == 10
    assert controller.last_promotion_diagnostics[
        "promotion_decision_reason"
    ] == reason


def test_validation_quality_score_is_the_aligned_selection_metric(config):
    expected = bounded_quality_score(1200.0, 800.0, 40.0, config["reward"])
    score = (-1.0, expected, 0.0, 0.0)
    assert _normalized_validation_quality_score(
        score, config["reward"]
    ) == pytest.approx(expected)
    assert _normalized_validation_quality_score(
        (-1.0, math.inf, 1.0, 1.0), config["reward"]
    ) is None


def test_final_sampled_validation_is_reused_only_for_final_accepted_candidate():
    sampled = {"completion_rate": 1.0}
    assert _can_reuse_final_sampled_validation(
        final_episode=600,
        sampled_episode=600,
        validation_event="accepted",
        sampled_validation=sampled,
    )
    assert not _can_reuse_final_sampled_validation(
        final_episode=600,
        sampled_episode=600,
        validation_event="rejected",
        sampled_validation=sampled,
    )
    assert not _can_reuse_final_sampled_validation(
        final_episode=600,
        sampled_episode=400,
        validation_event="accepted",
        sampled_validation=sampled,
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
