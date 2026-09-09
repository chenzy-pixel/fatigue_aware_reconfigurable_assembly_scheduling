from __future__ import annotations

import heapq

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from environment import AssemblySchedulingEnv, DecisionType
from environment.types import EventType, OperationState


def _step_requested_pair(environment, *, reconfiguration: bool) -> dict:
    policy = HeuristicPolicy()
    for _ in range(500):
        if environment.decision_type == DecisionType.PRODUCTION:
            for action in np.flatnonzero(~environment.get_action_mask()):
                if int(action) == environment.wait_action:
                    continue
                operation_index, machine_index = (
                    environment.decode_production_action(int(action))
                )
                operation = environment.operations[operation_index]
                machine = environment.machines[machine_index]
                mismatch = (
                    machine.current_module != operation.spec.required_module
                )
                if mismatch == reconfiguration:
                    return environment.step(int(action))[-1]
        environment.step(policy.select_action(environment))
        if environment.terminated or environment.truncated:
            break
    raise AssertionError("trajectory lacks the requested pair-action kind")


@pytest.mark.parametrize(
    ("reconfiguration", "expected"),
    [(False, "DIRECT_PROCESS"), (True, "COMMIT_RECONFIG")],
)
def test_pair_action_has_stable_direct_or_reconfiguration_semantics(
    config, fixed_instance, reconfiguration, expected
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    info = _step_requested_pair(
        environment, reconfiguration=reconfiguration
    )
    assert info["action_type"] == expected


def test_production_action_space_is_pairs_plus_one_wait(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    pair_count = len(environment.operations) * len(environment.machines)
    assert environment.wait_action == pair_count
    assert len(environment.get_action_mask()) == pair_count + 1


def test_initial_wait_advances_without_reconfiguration_cost(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    assert not environment.get_action_mask()[environment.wait_action]
    before_tick = environment.current_tick
    _, reward, _, _, info = environment.step(environment.wait_action)
    assert environment.current_tick > before_tick
    assert environment.metrics()["reconfiguration_cost"] == pytest.approx(0.0)
    assert reward.cost == pytest.approx(0.0)
    assert info["action_type"] == "WAIT"
    assert info["wait_certificate"]["allowed"] is True


def test_wait_allows_process_completion_exactly_at_horizon(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    pair_actions = [
        int(action)
        for action in np.flatnonzero(~environment.get_action_mask())
        if int(action) != environment.wait_action
    ]
    action = next(
        action
        for action in pair_actions
        if environment.machines[
            environment.decode_production_action(action)[1]
        ].current_module
        == environment.operations[
            environment.decode_production_action(action)[0]
        ].spec.required_module
    )
    operation_index, machine_index = environment.decode_production_action(action)
    environment.step(action, build_observation=False)

    for index, operation in enumerate(environment.operations):
        if index != operation_index:
            operation.state = OperationState.DONE
    operation_id = environment.operations[operation_index].spec.id
    environment._events = [
        event
        for event in environment._events
        if event[3] == EventType.PROCESS_COMPLETE
        and event[4].get("operation_id") == operation_id
    ]
    heapq.heapify(environment._events)
    completion_tick = environment.machines[machine_index].busy_until_tick
    assert completion_tick is not None
    environment.horizon_tick = completion_tick
    environment._invalidate_resource_snapshot()

    mask = environment.get_action_mask()
    certificate = environment._last_action_mask_analysis["wait"]
    assert np.flatnonzero(~mask).tolist() == [environment.wait_action]
    assert certificate["allowed"] is True
    assert certificate["next_tick"] == completion_tick
    assert certificate["estimated_completion_tick"] == completion_tick

    environment.step(environment.wait_action, build_observation=False)

    assert environment.terminated is True
    assert environment.truncated is False
    assert environment.terminal_reason == "completed"
