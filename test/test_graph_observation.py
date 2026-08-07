from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from agent.ppo import RolloutBuffer
from environment import (
    ASSEMBLY_EDGE_TYPES,
    ASSEMBLY_NODE_TYPES,
    CAPABLE_EDGE,
    CAN_DISASSEMBLE_EDGE,
    CAN_INSTALL_EDGE,
    LOCKED_EDGE,
    SERVICE_CANDIDATE_EDGE,
    WAVE_MODULE_EDGE,
    PRECEDES_EDGE,
    AssemblySchedulingEnv,
)
from environment.types import ReconfigurationStage


def _edge_feature(
    observation,
    edge_type,
    source_index: int,
    target_index: int,
    feature_name: str,
) -> float:
    store = observation.relations[edge_type]
    matches = np.flatnonzero(
        (store.edge_index[0] == source_index)
        & (store.edge_index[1] == target_index)
    )
    assert len(matches) == 1
    feature_index = store.feature_names.index(feature_name)
    return float(store.edge_features[int(matches[0]), feature_index])


def _select_production_pair(environment, *, requires_reconfiguration: bool):
    mask = environment.get_action_mask()
    for action in np.flatnonzero(~mask):
        if int(action) == environment.advance_action:
            continue
        operation_index, machine_index = environment.decode_production_action(
            int(action)
        )
        operation = environment.operations[operation_index]
        machine = environment.machines[machine_index]
        mismatch = machine.current_module != operation.spec.required_module
        if mismatch == requires_reconfiguration:
            return int(action), operation_index, machine_index
    raise AssertionError("no suitable production pair found")


def _select_worker_for_machine(environment, machine_index: int) -> int:
    mask = environment.get_action_mask()
    for worker_index in range(len(environment.workers)):
        action = environment.encode_worker_action(machine_index, worker_index)
        if not mask[action]:
            return action
    raise AssertionError("no feasible worker found for machine")


def _reach_reconfiguration_pair(environment):
    for _ in range(500):
        try:
            return _select_production_pair(
                environment, requires_reconfiguration=True
            )
        except AssertionError:
            if environment.decision_type.value == "PRODUCTION":
                try:
                    action, _, _ = _select_production_pair(
                        environment, requires_reconfiguration=False
                    )
                except AssertionError:
                    action = environment.advance_action
            else:
                action = environment.advance_action
            environment.step(action)
    raise AssertionError("failed to reach a feasible reconfiguration pair")


def _advance_until_reconfiguration_stage(
    environment, reconfiguration, target_stage
):
    observation = environment.observe()
    for _ in range(500):
        if reconfiguration.stage == target_stage:
            return observation
        observation, _, terminated, truncated, _ = environment.step(
            environment.advance_action
        )
        assert not terminated and not truncated
    raise AssertionError(f"failed to reach reconfiguration stage {target_stage}")


def test_graph_observation_static_contract(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    observation.validate()

    assert observation.node_ids == {
        "operation": tuple(
            operation.id for operation in fixed_instance.operations
        ),
        "machine": tuple(machine.id for machine in fixed_instance.machines),
        "worker": tuple(worker.id for worker in fixed_instance.workers),
        "order": tuple(order.id for order in fixed_instance.orders),
        "module": tuple(fixed_instance.modules),
        "wave": tuple(fixed_instance.waves),
    }
    assert set(observation.node_features) == set(ASSEMBLY_NODE_TYPES)
    assert observation.operations.shape[0] == 60
    assert observation.machines.shape[0] == 8
    assert observation.workers.shape[0] == 6
    assert observation.global_feature_names == (
        "current_time_norm",
        "active_order_ratio",
        "ready_operation_ratio",
        "pending_reconfiguration_ratio",
        "completed_operation_ratio",
        "production_decision",
        "worker_decision",
        "safe_idle_worker_ratio",
        "worker_matching_deficit_norm",
        "minimum_worker_alternative_ratio",
        "minimum_candidate_horizon_slack",
    )
    assert observation.global_features.shape == (
        len(observation.global_feature_names),
    )
    assert set(observation.relations) == set(ASSEMBLY_EDGE_TYPES)

    precedence = observation.relations[PRECEDES_EDGE]
    assert precedence.num_edges == 45
    assert precedence.feature_names == ("precedence",)
    assert np.all(precedence.edge_features == 1.0)

    expected_capability = {
        (operation_index, machine_index)
        for operation_index, operation in enumerate(fixed_instance.operations)
        for machine_index, machine in enumerate(fixed_instance.machines)
        if operation.required_module in machine.module_parameters
    }
    capability = observation.relations[CAPABLE_EDGE]
    assert capability.bidirectional
    assert set(map(tuple, capability.edge_index.T)) == expected_capability
    assert capability.feature_names == (
        "processing_time_norm",
        "configuration_match",
        "earliest_start_time_norm",
        "resource_ready_time_norm",
        "predicted_finish_time_norm",
        "safe_disassembly_worker_ratio",
        "safe_installation_worker_ratio",
        "matching_deficit_after_commit_norm",
        "horizon_slack_norm",
        "reconfiguration_time_norm",
        "fixed_disassembly_cost_norm",
        "fixed_installation_cost_norm",
        "estimated_labor_cost_norm",
        "estimated_downtime_cost_norm",
    )
    assert np.all(capability.edge_features[:, [0, 2, 3, 4]] >= 0.0)
    assert np.all(capability.edge_features[:, [0, 2, 3, 4]] <= 2.0)
    assert np.all(capability.edge_features[:, [5, 6, 7]] >= 0.0)
    assert np.all(capability.edge_features[:, [5, 6, 7]] <= 1.0)
    assert np.all(capability.edge_features[:, 8] >= -1.0)
    assert np.all(capability.edge_features[:, 8] <= 1.0)

    repeated = environment.observe()
    assert np.array_equal(
        repeated.global_features,
        observation.global_features,
    )
    assert np.array_equal(
        repeated.relations[CAPABLE_EDGE].edge_features,
        capability.edge_features,
    )

    expected_installation = {
        (worker_index, operation_index)
        for worker_index, worker in enumerate(fixed_instance.workers)
        for operation_index, operation in enumerate(fixed_instance.operations)
        if operation.required_module in worker.qualified_modules
    }
    installation = observation.relations[CAN_INSTALL_EDGE]
    assert set(map(tuple, installation.edge_index.T)) == expected_installation

    expected_disassembly = {
        (worker_index, machine_index)
        for worker_index, worker in enumerate(fixed_instance.workers)
        for machine_index, machine in enumerate(fixed_instance.machines)
        if machine.initial_module in worker.qualified_modules
    }
    disassembly = observation.relations[CAN_DISASSEMBLE_EDGE]
    assert set(map(tuple, disassembly.edge_index.T)) == expected_disassembly

    locked = observation.relations[LOCKED_EDGE]
    assert locked.bidirectional
    assert locked.edge_index.shape == (2, 0)
    assert locked.edge_features.shape == (0, len(locked.feature_names))

    for edge_store in observation.relations.values():
        assert edge_store.edge_index.dtype == np.int64
        assert edge_store.edge_features.dtype == np.float32


def test_capability_est_for_idle_and_processing_machine(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)

    direct_action, direct_operation_index, machine_index = (
        _select_production_pair(
            environment, requires_reconfiguration=False
        )
    )
    assert _edge_feature(
        observation,
        CAPABLE_EDGE,
        direct_operation_index,
        machine_index,
        "earliest_start_time_norm",
    ) == pytest.approx(environment.current_tick / environment.horizon_tick)

    machine = environment.machines[machine_index]
    mismatch_operation_index = next(
        operation_index
        for operation_index, operation in enumerate(environment.operations)
        if operation.spec.required_module in machine.spec.module_parameters
        and operation.spec.required_module != machine.current_module
    )
    optimistic_reconfiguration = environment.estimate_reconfiguration_ticks(
        mismatch_operation_index, machine_index
    )
    assert _edge_feature(
        observation,
        CAPABLE_EDGE,
        mismatch_operation_index,
        machine_index,
        "earliest_start_time_norm",
    ) == pytest.approx(
        optimistic_reconfiguration / environment.horizon_tick
    )

    observation, _, _, _, _ = environment.step(direct_action)
    assert machine.busy_until_tick is not None
    assert _edge_feature(
        observation,
        CAPABLE_EDGE,
        direct_operation_index,
        machine_index,
        "earliest_start_time_norm",
    ) == pytest.approx(
        machine.busy_until_tick / environment.horizon_tick
    )
    assert _edge_feature(
        observation,
        CAPABLE_EDGE,
        mismatch_operation_index,
        machine_index,
        "earliest_start_time_norm",
    ) == pytest.approx(
        (
            machine.busy_until_tick
            + environment.estimate_reconfiguration_ticks(
                mismatch_operation_index, machine_index
            )
        )
        / environment.horizon_tick
    )


def test_dynamic_lock_and_worker_machine_edges(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    action, locked_operation_index, machine_index = (
        _reach_reconfiguration_pair(environment)
    )
    observation, _, _, _, _ = environment.step(action)
    machine = environment.machines[machine_index]
    reconfiguration = environment.reconfigurations[
        environment._machine_reconfiguration[machine.spec.id]
    ]

    locked = observation.relations[LOCKED_EDGE]
    assert locked.num_edges == 1
    assert tuple(locked.edge_index[:, 0]) == (
        locked_operation_index,
        machine_index,
    )
    assert _edge_feature(
        observation,
        LOCKED_EDGE,
        locked_operation_index,
        machine_index,
        "stage_WAIT_DIS",
    ) == 1.0

    locked_est = environment.estimate_earliest_start_tick(
        locked_operation_index, machine_index
    )
    same_target_operation_index = next(
        operation_index
        for operation_index, operation in enumerate(environment.operations)
        if operation_index != locked_operation_index
        and operation.spec.required_module == reconfiguration.target_module
    )
    assert environment.estimate_earliest_start_tick(
        same_target_operation_index, machine_index
    ) == (
        locked_est
        + environment.estimate_processing_ticks(
            locked_operation_index, machine_index
        )
    )

    environment.step(environment.advance_action)
    disassembly_action = _select_worker_for_machine(
        environment, machine_index
    )
    observation, _, _, _, _ = environment.step(disassembly_action)
    assert reconfiguration.stage == ReconfigurationStage.DIS
    assert _edge_feature(
        observation,
        LOCKED_EDGE,
        locked_operation_index,
        machine_index,
        "stage_DIS",
    ) == 1.0

    observation = _advance_until_reconfiguration_stage(
        environment, reconfiguration, ReconfigurationStage.WAIT_INS
    )
    assert reconfiguration.stage == ReconfigurationStage.WAIT_INS
    assert machine.current_module == fixed_instance.no_module_state
    assert _edge_feature(
        observation,
        LOCKED_EDGE,
        locked_operation_index,
        machine_index,
        "stage_WAIT_INS",
    ) == 1.0
    disassembly_edges = observation.relations[CAN_DISASSEMBLE_EDGE]
    assert not np.any(disassembly_edges.edge_index[1] == machine_index)

    environment.step(environment.advance_action)
    installation_action = _select_worker_for_machine(
        environment, machine_index
    )
    observation, _, _, _, _ = environment.step(installation_action)
    assert reconfiguration.stage == ReconfigurationStage.INS
    assert _edge_feature(
        observation,
        LOCKED_EDGE,
        locked_operation_index,
        machine_index,
        "stage_INS",
    ) == 1.0

    observation = _advance_until_reconfiguration_stage(
        environment, reconfiguration, ReconfigurationStage.DONE
    )
    assert reconfiguration.stage == ReconfigurationStage.DONE
    assert observation.relations[LOCKED_EDGE].num_edges == 0
    assert _edge_feature(
        observation,
        CAPABLE_EDGE,
        locked_operation_index,
        machine_index,
        "configuration_match",
    ) == 1.0
    expected_workers = {
        worker_index
        for worker_index, worker in enumerate(environment.workers)
        if machine.current_module in worker.spec.qualified_modules
    }
    actual_workers = {
        int(source)
        for source, target in observation.relations[
            CAN_DISASSEMBLE_EDGE
        ].edge_index.T
        if int(target) == machine_index
    }
    assert actual_workers == expected_workers
    observation.validate()


def test_graph_copy_buffer_and_terminal_observation(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    copied = observation.copy()
    copied.relations[CAPABLE_EDGE].edge_features[0, 0] = -123.0
    assert observation.relations[CAPABLE_EDGE].edge_features[0, 0] != -123.0

    buffer = RolloutBuffer()
    mask = environment.get_action_mask()
    action = int(np.flatnonzero(~mask)[0])
    buffer.add(observation, mask, action, 0.0, 0.0, 0.0, False)
    original_feature = buffer.transitions[0].observation.operations[0, 0]
    observation.operations[0, 0] = -456.0
    assert buffer.transitions[0].observation.operations[0, 0] == original_feature
    assert not hasattr(buffer.transitions[0].observation, "relations")

    while not (environment.terminated or environment.truncated):
        environment.step(HeuristicPolicy().select_action(environment))
    terminal = environment.observe()
    terminal.validate()
    assert terminal.relations[LOCKED_EDGE].edge_index.shape == (2, 0)


def test_fixed_cost_counterfactual_changes_state_and_candidate_edges(
    config,
    fixed_instance,
):
    changed_costs = {
        module: replace(
            costs,
            fixed_disassembly_cost=costs.fixed_disassembly_cost + 17.0,
            fixed_installation_cost=costs.fixed_installation_cost + 23.0,
        )
        for module, costs in fixed_instance.module_costs.items()
    }
    changed_instance = replace(
        fixed_instance,
        instance_id=f"{fixed_instance.instance_id}_changed_costs",
        module_costs=changed_costs,
    )
    baseline_env = AssemblySchedulingEnv(config)
    changed_env = AssemblySchedulingEnv(config)
    baseline = baseline_env.reset(fixed_instance)
    changed = changed_env.reset(changed_instance)

    assert np.array_equal(
        baseline_env.get_action_mask(), changed_env.get_action_mask()
    )
    assert not np.array_equal(baseline.modules, changed.modules)
    assert not np.array_equal(
        baseline.relations[CAPABLE_EDGE].edge_features,
        changed.relations[CAPABLE_EDGE].edge_features,
    )

    action = None
    policy = HeuristicPolicy()
    for _ in range(500):
        try:
            action, _, _ = _select_production_pair(
                baseline_env, requires_reconfiguration=True
            )
            break
        except AssertionError:
            assert np.array_equal(
                baseline_env.get_action_mask(), changed_env.get_action_mask()
            )
            shared_action = policy.select_action(baseline_env)
            baseline_env.step(shared_action)
            changed_env.step(shared_action)
    assert action is not None
    assert not changed_env.get_action_mask()[action]
    baseline_env.step(action)
    changed_env.step(action)
    baseline_worker, _, _, _, _ = baseline_env.step(
        baseline_env.advance_action
    )
    changed_worker, _, _, _, _ = changed_env.step(
        changed_env.advance_action
    )
    assert np.array_equal(
        baseline_env.get_action_mask(), changed_env.get_action_mask()
    )
    assert not np.array_equal(
        baseline_worker.relations[SERVICE_CANDIDATE_EDGE].edge_features,
        changed_worker.relations[SERVICE_CANDIDATE_EDGE].edge_features,
    )


def test_schema_v3_future_demand_order_progress_and_snapshot_stability(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    initial = environment.reset(fixed_instance)
    snapshot = initial.copy()
    assert all(
        np.isfinite(features).all()
        for features in initial.node_features.values()
    )
    assert np.isfinite(initial.global_features).all()
    assert all(
        np.isfinite(store.edge_features).all()
        for store in initial.relations.values()
    )

    is_last_index = initial.node_feature_names["operation"].index(
        "is_last_operation"
    )
    for order in fixed_instance.orders:
        rows = [
            fixed_instance.operation_index[operation.id]
            for operation in order.operations
        ]
        assert initial.operations[rows[-1], is_last_index] == 1.0
        assert np.count_nonzero(initial.operations[rows, is_last_index]) == 1

    future_index = initial.relations[WAVE_MODULE_EDGE].feature_names.index(
        "future_operation_ratio"
    )
    assert np.any(
        initial.relations[WAVE_MODULE_EDGE].edge_features[:, future_index]
        > 0.0
    )

    policy = HeuristicPolicy()
    changed_order_id = None
    for _ in range(1000):
        environment.step(policy.select_action(environment))
        for runtime in environment.operations:
            if runtime.state.value == "DONE":
                order = next(
                    value
                    for value in fixed_instance.orders
                    if value.id == runtime.spec.order_id
                )
                if order.id not in environment._order_completion_tick:
                    changed_order_id = order.id
                    break
        if changed_order_id is not None:
            break
        if environment.terminated or environment.truncated:
            break
    assert changed_order_id is not None
    progressed = environment.observe()
    order_index = next(
        index
        for index, order in enumerate(fixed_instance.orders)
        if order.id == changed_order_id
    )
    remaining_index = progressed.node_feature_names["order"].index(
        "remaining_operation_ratio"
    )
    age_index = progressed.node_feature_names["order"].index("age_norm")
    assert (
        progressed.orders[order_index, remaining_index]
        < initial.orders[order_index, remaining_index]
    )
    assert progressed.orders[order_index, age_index] > 0.0

    for node_type in snapshot.node_features:
        assert np.array_equal(
            snapshot.node_features[node_type],
            initial.node_features[node_type],
        )
    for edge_type in snapshot.relations:
        assert np.array_equal(
            snapshot.relations[edge_type].edge_features,
            initial.relations[edge_type].edge_features,
        )


def test_future_module_counterfactual_is_observable_without_mask_aliasing(
    config,
    fixed_instance,
):
    target_order = next(
        order for order in fixed_instance.orders if order.release_time > 0.0
    )
    target_operation = target_order.operations[0]
    replacement_module = next(
        module
        for module in fixed_instance.modules
        if module != target_operation.required_module
    )
    changed_operation = replace(
        target_operation,
        required_module=replacement_module,
    )
    changed_order = replace(
        target_order,
        operations=(changed_operation, *target_order.operations[1:]),
    )
    changed_orders = tuple(
        changed_order if order.id == target_order.id else order
        for order in fixed_instance.orders
    )
    changed_instance = replace(
        fixed_instance,
        instance_id=f"{fixed_instance.instance_id}_future_module",
        orders=changed_orders,
    )
    baseline_env = AssemblySchedulingEnv(config)
    changed_env = AssemblySchedulingEnv(config)
    baseline = baseline_env.reset(fixed_instance)
    changed = changed_env.reset(changed_instance)
    assert np.array_equal(
        baseline_env.get_action_mask(), changed_env.get_action_mask()
    )
    operation_index = fixed_instance.operation_index[target_operation.id]
    assert not np.array_equal(
        baseline.operations[operation_index],
        changed.operations[operation_index],
    )
    assert not np.array_equal(baseline.modules, changed.modules)
    assert not np.array_equal(
        baseline.relations[WAVE_MODULE_EDGE].edge_features,
        changed.relations[WAVE_MODULE_EDGE].edge_features,
    )
