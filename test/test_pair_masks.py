from __future__ import annotations

import numpy as np

from agent.baselines import HeuristicPolicy
from environment import AssemblySchedulingEnv, DecisionType
from environment.types import MachineState, OperationState


def _first_mismatch_action(environment: AssemblySchedulingEnv) -> int:
    policy = HeuristicPolicy()
    for _ in range(500):
        if environment.decision_type == DecisionType.PRODUCTION:
            mask = environment.get_action_mask()
            for action in np.flatnonzero(~mask):
                action = int(action)
                if action == environment.wait_action:
                    continue
                operation_index, machine_index = (
                    environment.decode_production_action(action)
                )
                if (
                    environment.machines[machine_index].current_module
                    != environment.operations[
                        operation_index
                    ].spec.required_module
                ):
                    return action
        environment.step(policy.select_action(environment))
        if environment.terminated or environment.truncated:
            break
    raise AssertionError("test instance has no legal reconfiguration pair")


def test_production_pair_mask_is_exactly_instant_physical_legality(
    config, fixed_instance, monkeypatch
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    environment._invalidate_resource_snapshot()

    def unexpected_projection(*_args, **_kwargs):
        raise AssertionError("pair masking must not consult resource projection")

    monkeypatch.setattr(
        environment,
        "_production_resource_profile",
        unexpected_projection,
    )
    mask = environment.get_action_mask()
    for operation_index, operation in enumerate(environment.operations):
        for machine_index, machine in enumerate(environment.machines):
            action = environment.encode_production_action(
                operation_index, machine_index
            )
            expected = (
                operation.state == OperationState.READY
                and machine.state == MachineState.IDLE
                and machine.current_module
                != environment.instance.no_module_state
                and operation.spec.required_module
                in machine.spec.module_parameters
            )
            assert bool(not mask[action]) is expected


def test_worker_pair_mask_is_exactly_instant_safe_assignment(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    environment.step(
        _first_mismatch_action(environment), build_observation=False
    )
    before_tick = environment.current_tick
    _, _, _, _, info = environment.step(
        environment.wait_action, build_observation=False
    )
    assert environment.decision_type == DecisionType.WORKER
    assert environment.current_tick == before_tick
    assert info["wait_reason"] == "worker_phase_handoff"

    mask = environment.get_action_mask()
    for machine_index, machine in enumerate(environment.machines):
        task = environment._pending_reconfiguration(machine.spec.id)
        for worker_index, worker in enumerate(environment.workers):
            action = environment.encode_worker_action(
                machine_index, worker_index
            )
            expected = bool(
                task is not None
                and environment._worker_can_start(task, worker)
            )
            assert bool(not mask[action]) is expected


def test_wait_mask_uses_progress_even_when_completion_estimate_exceeds_horizon(
    config, fixed_instance, monkeypatch
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    monkeypatch.setattr(
        environment,
        "_wait_opportunity",
        lambda: (environment.current_tick + 1, "external_event:ORDER_RELEASE"),
    )
    monkeypatch.setattr(
        environment,
        "_remaining_completion_estimate_ticks",
        lambda: environment.horizon_tick + 1,
    )
    environment._invalidate_resource_snapshot()
    mask = environment.get_action_mask()
    certificate = environment._last_action_mask_analysis["wait"]
    assert not mask[environment.wait_action]
    assert certificate["allowed"] is True
    assert certificate["reason"] == "external_event:ORDER_RELEASE"
    assert (
        certificate["estimated_completion_tick"]
        > environment.horizon_tick
    )


def test_wait_mask_rejects_states_without_deterministic_progress(
    config, fixed_instance, monkeypatch
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    monkeypatch.setattr(environment, "_wait_opportunity", lambda: None)
    environment._invalidate_resource_snapshot()

    mask = environment.get_action_mask()
    certificate = environment._last_action_mask_analysis["wait"]

    assert mask[environment.wait_action]
    assert certificate["allowed"] is False
    assert certificate["reason"] == "no_state_progress"
