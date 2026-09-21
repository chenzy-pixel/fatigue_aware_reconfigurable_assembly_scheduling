from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from agent.baselines import HeuristicPolicy
from environment import AssemblySchedulingEnv, proxy_return_from_metrics
from environment.types import OperationState


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
def test_environment_failures_keep_actual_progress_and_use_unit_terminal_quality(
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
    policy = HeuristicPolicy()
    while not environment.task_done:
        _, reward, _, _, _ = environment.step(policy.select_action(environment))
        reward_sum += reward.scalarize(effective["reward"])
    metrics = environment.metrics()
    assert metrics["task_failed"] is True
    assert metrics["preference_quality_score"] == 1.0
    assert 0.0 <= metrics["operation_progress"] < 1.0
    assert reward_sum == pytest.approx(
        proxy_return_from_metrics(metrics, effective), abs=1e-8
    )
