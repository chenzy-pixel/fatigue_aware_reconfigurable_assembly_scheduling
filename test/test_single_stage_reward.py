from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.baselines import HeuristicPolicy
from deadlock_replay import _reward_audit
from environment import (
    AssemblySchedulingEnv,
    FAILURE_PENALTY_REWARD,
    LEGACY_PROGRESS_QUALITY_REWARD,
    proxy_return_from_metrics,
)
from environment.types import OperationState
from result.io import write_csv
from train import _episode_log_row


def _two_order_instance(fixed_instance, *, counts=(2, 4), second_release=100.0):
    first_source, second_source = fixed_instance.orders[:2]
    first = replace(
        first_source,
        release_time=0.0,
        operations=first_source.operations[: counts[0]],
    )
    second = replace(
        second_source,
        release_time=second_release,
        operations=second_source.operations[: counts[1]],
    )
    return replace(
        fixed_instance,
        instance_id="progress_denominator_case",
        orders=(first, second),
    )


def test_order_balanced_progress_uses_fixed_per_order_denominators(config, fixed_instance):
    instance = _two_order_instance(fixed_instance)
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    assert environment.operation_progress() == 0.0

    environment.operations[0].state = OperationState.DONE
    assert environment.operation_progress() == pytest.approx(1.0 / (2 * 2))
    environment.operations[2].state = OperationState.DONE
    assert environment.operation_progress() == pytest.approx(
        1.0 / (2 * 2) + 1.0 / (2 * 4)
    )


def test_unreleased_orders_are_in_denominator_and_release_has_no_progress(config, fixed_instance):
    instance = _two_order_instance(fixed_instance, counts=(1, 1), second_release=5.0)
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    assert environment._order_released[instance.orders[1].id] is False
    before = environment.operation_progress()
    _, reward, _, _, _ = environment.step(environment.wait_action)
    assert environment._order_released[instance.orders[1].id] is True
    assert environment.operation_progress() == before == 0.0
    assert reward.operation_progress == 0.0


def test_assignment_and_processing_start_never_create_progress(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    policy = HeuristicPolicy()
    saw_non_wait = False
    for _ in range(30):
        if environment.task_done:
            break
        action = policy.select_action(environment)
        action_type = environment._action_type(environment.decision_type, action)
        _, reward, _, _, _ = environment.step(action)
        if action_type != "WAIT":
            saw_non_wait = True
            assert reward.operation_progress == 0.0
    assert saw_non_wait


def test_one_wait_accumulates_all_simultaneous_operation_completions(config, fixed_instance):
    first_source, second_source = fixed_instance.orders[:2]
    first_operation = replace(first_source.operations[0], base_processing_time=10.0)
    second_operation = replace(second_source.operations[0], base_processing_time=10.0)
    instance = replace(
        fixed_instance,
        instance_id="simultaneous_completion_case",
        orders=(
            replace(first_source, release_time=0.0, operations=(first_operation,)),
            replace(second_source, release_time=0.0, operations=(second_operation,)),
        ),
    )
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    machine_indices = [
        index
        for index, machine in enumerate(environment.machines)
        if machine.spec.id in {"M5", "M7"}
    ]
    assert len(machine_indices) == 2
    for operation_index, machine_index in enumerate(machine_indices):
        action = environment.encode_production_action(operation_index, machine_index)
        _, reward, _, _, _ = environment.step(action)
        assert reward.operation_progress == 0.0
    _, reward, terminated, truncated, _ = environment.step(environment.wait_action)
    assert terminated is True
    assert truncated is False
    assert reward.operation_progress == pytest.approx(1.0)
    assert environment.operation_progress() == 1.0


def test_general_proxy_identity_supports_nonzero_initial_state(config):
    metrics = {
        "initial_progress": 0.25,
        "operation_progress": 0.75,
        "initial_preference_quality_score": 0.20,
        "preference_quality_score": 0.40,
        "raw_preference_quality_score": 0.40,
        "flow_time_objective": 1.0,
        "reconfiguration_cost": 1.0,
        "worker_load_variance": 1.0,
        "task_failed": False,
        "truncated": False,
    }
    assert proxy_return_from_metrics(metrics, config) == pytest.approx(0.30)


def test_standard_reset_and_successful_trajectory_satisfy_general_identity(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    initial = environment.metrics()
    assert initial["initial_progress"] == 0.0
    assert initial["initial_preference_quality_score"] == 0.0
    reward_sum = 0.0
    policy = HeuristicPolicy()
    while not environment.task_done:
        _, reward, _, _, _ = environment.step(policy.select_action(environment))
        reward_sum += reward.scalarize(config["reward"])
    metrics = environment.metrics()
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, config), abs=1e-8
    )


@pytest.mark.parametrize("reason", ("decision_limit", "horizon"))
def test_environment_failures_keep_actual_quality_and_apply_one_penalty(
    config, fixed_instance, reason
):
    effective = deepcopy(config)
    if reason == "decision_limit":
        effective["environment"]["max_decisions"] = 3
    else:
        fixed_instance = replace(fixed_instance, horizon=0.1)
    environment = AssemblySchedulingEnv(effective)
    environment.reset(fixed_instance)
    reward_sum = 0.0
    failure_sum = 0.0
    transitions = []
    policy = HeuristicPolicy()
    while not environment.task_done:
        _, reward, _, _, _ = environment.step(policy.select_action(environment))
        reward_sum += reward.scalarize(effective["reward"])
        failure_sum += reward.failure
        transitions.append({"reward": reward.as_dict()})
    metrics = environment.metrics()
    assert metrics["task_failed"] is True
    assert metrics["preference_quality_score"] == 1.0
    assert metrics["actual_preference_quality_score"] < 1.0
    assert metrics["terminal_failure_penalty_applied"] == pytest.approx(1.0)
    assert failure_sum == pytest.approx(-1.0)
    assert 0.0 <= metrics["operation_progress"] < 1.0
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, effective), abs=1e-8
    )
    audit = _reward_audit(environment, transitions)
    assert audit["component_sums"]["failure"] == pytest.approx(-1.0)
    assert audit["base_cumulative_reward"] == pytest.approx(
        metrics["base_cumulative_reward"], abs=1e-8
    )
    assert audit["scalar_training_return"] == pytest.approx(
        metrics["training_cumulative_reward"], abs=1e-8
    )
    assert audit["configured_identity_residual"] == pytest.approx(0.0, abs=1e-8)
    assert audit["recomputed_versions"][FAILURE_PENALTY_REWARD][
        "training_cumulative_reward"
    ] == pytest.approx(reward_sum, abs=1e-8)


def test_failure_v2_distinguishes_equal_progress_by_actual_quality(config):
    common = {
        "initial_progress": 0.0,
        "operation_progress": 0.94,
        "initial_preference_quality_score": 0.0,
        "preference_quality_score": 1.0,
        "flow_time_objective": 1.0,
        "reconfiguration_cost": 0.0,
        "worker_load_variance": 0.0,
        "task_failed": True,
        "truncated": True,
    }
    better = {**common, "raw_preference_quality_score": 0.65}
    worse = {**common, "raw_preference_quality_score": 0.80}
    better_return = proxy_return_from_metrics(better, config)
    worse_return = proxy_return_from_metrics(worse, config)
    assert better_return == pytest.approx(-0.71)
    assert worse_return == pytest.approx(-0.86)
    assert better_return - worse_return == pytest.approx(0.15)


def test_successful_action_trace_and_step_rewards_match_legacy_v1(
    config, fixed_instance
):
    current = AssemblySchedulingEnv(config)
    observation = current.reset(fixed_instance)
    del observation
    policy = HeuristicPolicy()
    actions: list[int] = []
    current_rewards: list[float] = []
    while not current.task_done:
        action = policy.select_action(current)
        actions.append(action)
        _, reward, _, _, _ = current.step(action)
        current_rewards.append(reward.scalarize(config["reward"]))
    assert current.task_succeeded

    legacy_config = deepcopy(config)
    legacy_config["reward"]["mode"] = LEGACY_PROGRESS_QUALITY_REWARD
    legacy = AssemblySchedulingEnv(legacy_config)
    legacy.reset(fixed_instance)
    legacy_rewards: list[float] = []
    for action in actions:
        _, reward, _, _, _ = legacy.step(action)
        legacy_rewards.append(reward.scalarize(legacy_config["reward"]))
    assert legacy.task_succeeded
    assert legacy_rewards == pytest.approx(current_rewards, abs=1e-12)
    for field in (
        "operation_progress",
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
        "terminal_reason",
    ):
        assert legacy.metrics()[field] == current.metrics()[field]


def test_episode_csv_row_reconstructs_failed_training_return(
    config, fixed_instance, tmp_path
):
    effective = deepcopy(config)
    effective["environment"]["max_decisions"] = 3
    environment = AssemblySchedulingEnv(effective)
    environment.reset(fixed_instance)
    policy = HeuristicPolicy()
    components = {
        "flow": 0.0,
        "cost": 0.0,
        "variance": 0.0,
        "operation_progress": 0.0,
        "quality": 0.0,
        "failure": 0.0,
        "feasibility_shaping": 0.0,
    }
    reward_sum = 0.0
    steps = 0
    while not environment.task_done:
        _, reward, _, _, _ = environment.step(policy.select_action(environment))
        reward_sum += reward.scalarize(effective["reward"])
        for name in components:
            components[name] += float(getattr(reward, name))
        steps += 1
    metrics = environment.metrics()
    expected = proxy_return_from_metrics(metrics, effective)
    episode = SimpleNamespace(
        episode_index=0,
        instance_id=metrics["instance_id"],
        reward_sum=reward_sum,
        base_reward_sum=components["operation_progress"] + components["quality"],
        unshaped_reward_sum=(
            components["operation_progress"]
            + components["quality"]
            + components["failure"]
        ),
        expected_reward=expected,
        metrics=metrics,
        step_count=steps,
        policy_step_count=steps,
        forced_action_count=0,
        forced_action_ratio=0.0,
        reward_components=components,
    )
    path = tmp_path / "episodes.csv"
    write_csv(path, [_episode_log_row(episode)])
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["reward_version"] == FAILURE_PENALTY_REWARD
    assert float(row["reward_failure"]) == pytest.approx(-1.0)
    assert float(row["base_reward"]) + float(row["reward_failure"]) == pytest.approx(
        float(row["reward"]), abs=1e-8
    )
    assert float(row["reward"]) == pytest.approx(float(row["expected_reward"]), abs=1e-8)
    assert float(row["reward_identity_error"]) == pytest.approx(0.0, abs=1e-8)
