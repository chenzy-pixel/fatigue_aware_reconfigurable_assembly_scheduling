"""Reference comparison for grouped capability edge construction."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from environment import AssemblySchedulingEnv, CAPABLE_EDGE
from environment.dynamics import quantize_to_ticks
from environment.resource_projection import ResourceProjector
from scripts.audit_state_sufficiency import worker_alias_instance, worker_histories


def _edge_by_edge_reference(env):
    edge_index = env._static_edge_indices[CAPABLE_EDGE]
    horizon = max(1, env.horizon_tick)
    worker_count = max(1, len(env.workers))
    scales = env.config["objective_scalarizer"]["scales"]
    projector = ResourceProjector(env)
    rows = np.empty((edge_index.shape[1], 16), dtype=np.float64)
    path_costs = {}
    order_slacks = {}
    for edge, (operation_index, machine_index) in enumerate(edge_index.T):
        oi, mi = int(operation_index), int(machine_index)
        operation, machine = env.operations[oi], env.machines[mi]
        candidate = env._candidate_resource_projection(oi, mi)
        profile = env._production_resource_profile(mi, operation.spec.required_module)
        path = candidate.path
        start = candidate.processing_start_tick if candidate.processing_start_tick is not None else horizon + 1
        finish = candidate.finish_tick if candidate.finish_tick is not None else horizon + 1
        key = (mi, id(path))
        if key not in path_costs:
            path_costs[key] = projector.path_costs(mi, path)
        dis, ins, labor, downtime, variance = path_costs[key]
        order_id = operation.spec.order_id
        if order_id not in order_slacks:
            order_slacks[order_id] = env.estimated_order_slack_norm(order_id)
        rows[edge] = [
            env._capability_processing_ticks[edge] / horizon,
            float(machine.current_module == operation.spec.required_module),
            start / horizon, candidate.resource_ready_tick / horizon,
            finish / horizon,
            (path.safe_disassembly_workers if path and path.stages else worker_count if path else 0) / worker_count,
            (path.safe_installation_workers if path and path.stages else worker_count if path else 0) / worker_count,
            profile.matching_deficit_after_commit / worker_count,
            (horizon - finish) / horizon,
            (path.end_tick - path.start_tick) / horizon if path else 0,
            dis / scales["cost"], ins / scales["cost"], labor / scales["cost"],
            downtime / scales["cost"], variance / scales["variance"], order_slacks[order_id],
        ]
    rows[:, (0, 2, 3, 4)] = np.clip(rows[:, (0, 2, 3, 4)], 0, 2)
    rows[:, 8] = np.clip(rows[:, 8], -1, 1)
    return edge_index, rows.astype(np.float32)


def test_grouped_capability_relation_matches_edge_by_edge_reference(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    grouped = env._build_capability_relation()
    env._invalidate_resource_snapshot()
    reference_index, reference_features = _edge_by_edge_reference(env)
    np.testing.assert_array_equal(grouped.edge_index, reference_index)
    np.testing.assert_array_equal(grouped.edge_features, reference_features)
    assert grouped.feature_names == (
        "processing_time_norm", "configuration_match", "earliest_start_time_norm",
        "resource_ready_time_norm", "predicted_finish_time_norm", "safe_disassembly_worker_ratio",
        "safe_installation_worker_ratio", "matching_deficit_after_commit_norm", "horizon_slack_norm",
        "reconfiguration_time_norm", "fixed_disassembly_cost_norm", "fixed_installation_cost_norm",
        "estimated_labor_cost_norm", "estimated_downtime_cost_norm",
        "estimated_worker_load_variance_delta_norm", "estimated_order_slack_norm",
    )


@pytest.mark.parametrize('prefix', [0, 9, 11, 13, 14])
def test_grouping_retains_actual_operation_identity(config, fixed_instance, prefix):
    instance = worker_alias_instance(fixed_instance)
    history = worker_histories(config, instance)[1][0]
    env = AssemblySchedulingEnv(config)
    env.reset(instance, build_observation=False)
    for action in history[:prefix]:
        env.step(action, build_observation=False)
    grouped = env._build_capability_relation()
    env._invalidate_resource_snapshot()
    index, features = _edge_by_edge_reference(env)
    np.testing.assert_array_equal(grouped.edge_index, index)
    np.testing.assert_array_equal(grouped.edge_features, features)


def test_grouping_uses_quantized_release_ticks(config, fixed_instance):
    orders = tuple(replace(order, release_time=1.01 if i % 2 else 1.02)
                   for i, order in enumerate(fixed_instance.orders))
    env = AssemblySchedulingEnv(config)
    env.reset(replace(fixed_instance, orders=orders), build_observation=False)
    expected_release = quantize_to_ticks(1.01, env.resolution)
    assert expected_release == quantize_to_ticks(1.02, env.resolution)
    assert {release for _, _, release, _ in env._capability_candidate_groups} == {expected_release}
    grouped = env._build_capability_relation()
    env._invalidate_resource_snapshot()
    _, features = _edge_by_edge_reference(env)
    np.testing.assert_array_equal(grouped.edge_features, features)
