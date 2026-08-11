from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from environment import AssemblySchedulingEnv, DecisionType
from environment.types import MachineState, OperationState


def _defer_config(config, *, recovery_improvement: bool) -> dict:
    updated = deepcopy(config)
    updated["environment"]["worker_resource_control"]["mode"] = (
        "legacy_postcheck"
    )
    updated["environment"]["production_defer"] = {
        "allow_recovery_improvement": recovery_improvement,
    }
    return updated


def _small_instance(
    fixed_instance,
    *,
    mismatch: bool,
    future_release: bool = False,
    fast_recovery: bool = False,
):
    first_order = replace(fixed_instance.orders[0], release_time=0.0)
    required_module = first_order.operations[0].required_module
    machine = fixed_instance.machines[0]
    source_module = next(
        module for module in machine.module_parameters if module != required_module
    )
    machine = replace(
        machine,
        initial_module=source_module if mismatch else required_module,
    )
    worker = fixed_instance.workers[0]
    orders = [first_order]
    if future_release:
        orders.append(replace(fixed_instance.orders[1], release_time=0.5))
    order_ids_by_wave = {
        wave_id: [order.id for order in orders if order.wave == wave_id]
        for wave_id in fixed_instance.waves
    }
    waves = {
        wave_id: {
            **wave,
            "order_ids": order_ids_by_wave[wave_id],
        }
        for wave_id, wave in fixed_instance.waves.items()
    }
    fatigue = fixed_instance.fatigue
    if fast_recovery:
        fatigue = replace(
            fatigue,
            maximum_safe_fatigue=1.0,
            disassembly_time_coefficient=1.0,
            installation_time_coefficient=1.0,
            idle_recovery_rate_per_minute=1.0,
        )
        worker = replace(worker, initial_fatigue=0.8)
    return replace(
        fixed_instance,
        instance_id="production_defer_test",
        instance_type="test",
        horizon=20.0,
        machines=(machine,),
        workers=(worker,),
        orders=tuple(orders),
        waves=waves,
        fatigue=fatigue,
    )


def test_direct_and_commit_actions_keep_explicit_semantics(config, fixed_instance):
    local_config = _defer_config(config, recovery_improvement=False)

    direct_env = AssemblySchedulingEnv(local_config)
    direct_env.reset(_small_instance(fixed_instance, mismatch=False))
    direct_action = direct_env.encode_production_action(0, 0)
    assert direct_env.production_defer_action == direct_env.advance_action
    assert direct_env.get_action_mask()[direct_env.production_defer_action]
    _, _, _, _, direct_info = direct_env.step(direct_action)
    assert direct_info["action_type"] == "DIRECT_PROCESS"
    assert direct_env.operations[0].state == OperationState.PROCESSING
    assert direct_env.machines[0].state == MachineState.PROCESSING
    assert not direct_env.reconfigurations

    commit_env = AssemblySchedulingEnv(local_config)
    commit_env.reset(_small_instance(fixed_instance, mismatch=True))
    commit_action = commit_env.encode_production_action(0, 0)
    assert commit_env.get_action_mask()[commit_env.production_defer_action]
    _, _, _, _, commit_info = commit_env.step(commit_action)
    assert commit_info["action_type"] == "COMMIT_RECONFIG"
    assert commit_env.operations[0].state == OperationState.LOCKED
    assert commit_env.machines[0].state == MachineState.WAIT_DIS
    assert len(commit_env.reconfigurations) == 1

    before_tick = commit_env.current_tick
    before_reconfigurations = set(commit_env.reconfigurations)
    _, _, _, _, defer_info = commit_env.step(
        commit_env.production_defer_action
    )
    assert defer_info["action_type"] == "DEFER_PRODUCTION"
    assert defer_info["defer_reason"] == "worker_phase_handoff"
    assert commit_env.current_tick == before_tick
    assert commit_env.decision_type == DecisionType.WORKER
    assert set(commit_env.reconfigurations) == before_reconfigurations
    assert commit_env.worker_advance_action == commit_env.advance_action


def test_mismatch_pair_and_defer_are_both_policy_choices(config, fixed_instance):
    environment = AssemblySchedulingEnv(
        _defer_config(config, recovery_improvement=False)
    )
    environment.reset(
        _small_instance(
            fixed_instance,
            mismatch=True,
            future_release=True,
        )
    )
    pair_action = environment.encode_production_action(0, 0)
    mask = environment.get_action_mask()
    assert np.flatnonzero(~mask).tolist() == [
        pair_action,
        environment.production_defer_action,
    ]


def test_defer_waits_for_external_event_without_locking_or_cost(config, fixed_instance):
    environment = AssemblySchedulingEnv(
        _defer_config(config, recovery_improvement=False)
    )
    environment.reset(
        _small_instance(
            fixed_instance,
            mismatch=True,
            future_release=True,
        )
    )
    initial_fatigue = environment.workers[0].fatigue
    _, reward, terminated, truncated, info = environment.step(
        environment.production_defer_action
    )

    assert not terminated
    assert not truncated
    assert environment.current_tick == 5
    assert environment.decision_type == DecisionType.PRODUCTION
    assert environment.operations[0].state == OperationState.READY
    assert environment.machines[0].state == MachineState.IDLE
    assert environment.workers[0].fatigue < initial_fatigue
    assert not environment.reconfigurations
    assert environment.metrics()["reconfiguration_cost"] == pytest.approx(0.0)
    assert reward.flow < 0.0
    assert reward.cost == pytest.approx(0.0)
    assert info["action_type"] == "DEFER_PRODUCTION"
    assert info["defer_reason"] == "external_event:ORDER_RELEASE"
    assert info["wait_ticks"] == 5
    assert info["wait_time"] == pytest.approx(0.5)


def test_defer_masks_without_future_change(config, fixed_instance):
    environment = AssemblySchedulingEnv(
        _defer_config(config, recovery_improvement=False)
    )
    environment.reset(_small_instance(fixed_instance, mismatch=True))
    mask = environment.get_action_mask()
    assert mask[environment.production_defer_action]
    assert np.flatnonzero(~mask).tolist() == [
        environment.encode_production_action(0, 0)
    ]


def test_defer_stops_at_first_quantized_recovery_improvement(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(
        _defer_config(config, recovery_improvement=True)
    )
    environment.reset(
        _small_instance(
            fixed_instance,
            mismatch=True,
            fast_recovery=True,
        )
    )
    machine = environment.machines[0]
    target_module = environment.operations[0].spec.required_module
    duration_before = environment._idle_worker_reconfiguration_ticks_at(
        machine,
        target_module,
        environment.current_tick,
    )
    improvement_tick = (
        environment._earliest_production_defer_recovery_improvement_tick()
    )
    assert improvement_tick is not None
    mask = environment.get_action_mask()
    assert not mask[environment.production_defer_action]
    assert np.count_nonzero(~mask) == 2

    _, _, _, _, info = environment.step(environment.production_defer_action)
    duration_after = environment._idle_worker_reconfiguration_ticks_at(
        environment.machines[0],
        target_module,
        environment.current_tick,
    )
    assert environment.current_tick == improvement_tick
    assert duration_after is not None
    assert duration_before is not None
    assert duration_after < duration_before
    assert info["defer_reason"] == "reconfiguration_duration_improved"
    assert info["recovery_improvement"]
    metrics = environment.metrics()
    assert metrics["production_defer_recovery_improvement_count"] == 1
    assert metrics["action_type_counts"]["DEFER_PRODUCTION"] == 1
