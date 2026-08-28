from __future__ import annotations

from dataclasses import replace

from configs import load_config
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from environment.env import TemporalWorkerTask
from environment.types import ReconfigurationStage


def _single_worker_temporal_env(fixed_instance):
    config = load_config("configs/v7/e1_single_flow.json")
    orders = (
        replace(fixed_instance.orders[5], release_time=0.0),
        replace(fixed_instance.orders[10], release_time=0.0),
    )
    worker = replace(
        fixed_instance.workers[0],
        qualified_modules=fixed_instance.modules,
        initial_fatigue=0.1,
    )
    instance = replace(
        fixed_instance,
        instance_id="temporal_v3_serial_reuse",
        orders=orders,
        workers=(worker,),
    )
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    return environment


def test_temporal_v3_rescues_serial_worker_reuse(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    first = environment.encode_production_action(0, 0)
    second = environment.encode_production_action(4, 1)
    environment.step(first)
    mask = environment.get_action_mask()
    assert not mask[second]
    metrics = environment.metrics()
    assert metrics["temporal_oracle_feasible_count"] > 0
    assert metrics["temporal_future_installation_rescued_count"] > 0


def test_temporal_v3_precedence_and_cache_are_deterministic(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    module = next(iter(environment.workers[0].spec.qualified_modules))
    tasks = (
        TemporalWorkerTask(
            "dis:first", 0, ReconfigurationStage.WAIT_DIS, module, 0
        ),
        TemporalWorkerTask(
            "ins:first", 0, ReconfigurationStage.WAIT_INS, module, 0,
            predecessor_id="dis:first",
        ),
    )
    first = environment._run_temporal_feasibility_search(tasks)
    second = environment._run_temporal_feasibility_search(tasks)
    assert first.status == second.status == "feasible"
    assert first.candidate_completion_tick is None
    assert second.searched_nodes == first.searched_nodes
    assert environment.metrics()["temporal_oracle_cache_hit_count"] >= 1


def test_temporal_v3_node_budget_returns_unknown_and_allows_action(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    environment.config["environment"]["worker_resource_control"][
        "temporal_feasibility"
    ]["max_search_nodes"] = 1
    module = next(iter(environment.workers[0].spec.qualified_modules))
    tasks = (
        TemporalWorkerTask(
            "dis:first", 0, ReconfigurationStage.WAIT_DIS, module, 0
        ),
        TemporalWorkerTask(
            "ins:first", 0, ReconfigurationStage.WAIT_INS, module, 0,
            predecessor_id="dis:first",
        ),
    )
    result = environment._run_temporal_feasibility_search(tasks)
    assert result.status == "unknown"
    assert environment.metrics()["temporal_oracle_unknown_count"] == 1

