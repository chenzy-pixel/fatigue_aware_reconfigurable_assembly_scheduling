from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy, RandomPolicy
from environment import AssemblySchedulingEnv, DecisionType
from environment.env import quantize_to_ticks


def run_policy(environment, policy):
    action_trace = []
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        action_trace.append((environment.decision_type.value, action))
        environment.step(action)
    return action_trace


def start_isolated_disassembly(config, fixed_instance):
    order = replace(fixed_instance.orders[0], release_time=0.0)
    required_module = order.operations[0].required_module
    machine_index = next(
        index
        for index, machine in enumerate(fixed_instance.machines)
        if required_module in machine.module_parameters
        and any(
            module != required_module for module in machine.module_parameters
        )
    )
    machine = fixed_instance.machines[machine_index]
    source_module = next(
        module
        for module in machine.module_parameters
        if module != required_module
    )
    machines = list(fixed_instance.machines)
    machines[machine_index] = replace(machine, initial_module=source_module)
    waves = {}
    for wave_id, wave in fixed_instance.waves.items():
        updated = dict(wave)
        updated["order_ids"] = [order.id] if wave_id == order.wave else []
        if wave_id == order.wave:
            updated["release_interval"] = [0.0, 0.0]
        waves[wave_id] = updated
    instance = replace(
        fixed_instance,
        instance_id="isolated_reconfiguration_test",
        instance_type="test",
        machines=tuple(machines),
        orders=(order,),
        waves=waves,
    )
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    production_action = environment.encode_production_action(0, machine_index)
    assert not environment.get_action_mask()[production_action]
    environment.step(production_action)
    environment.step(environment.advance_action)
    worker_action = next(
        environment.encode_worker_action(machine_index, worker_index)
        for worker_index in range(len(environment.workers))
        if not environment.get_action_mask()[
            environment.encode_worker_action(machine_index, worker_index)
        ]
    )
    worker_index = environment.decode_worker_action(worker_action)[1]
    return environment, machine_index, worker_index, worker_action


def test_time_quantization_uses_ceil_with_tolerance():
    assert quantize_to_ticks(2.3, 0.1) == 23
    assert quantize_to_ticks(2.3000000000000003, 0.1) == 23
    assert quantize_to_ticks(2.31, 0.1) == 24
    with pytest.raises(ValueError):
        quantize_to_ticks(-0.1, 0.1)


def test_initial_action_mask_and_phase(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    assert observation.decision_type == DecisionType.PRODUCTION
    mask = environment.get_action_mask()
    assert mask.shape == (60 * 8 + 1,)
    assert np.count_nonzero(~mask) > 1
    for action in np.flatnonzero(~mask)[:-1]:
        operation_index, machine_index = environment.decode_production_action(
            int(action)
        )
        operation = environment.operations[operation_index]
        machine = environment.machines[machine_index]
        assert operation.spec.required_module in machine.spec.module_parameters


def test_heuristic_completes_fixed_case_and_rewards_telescope(
    config, fixed_instance
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    run_policy(environment, HeuristicPolicy())
    metrics = environment.metrics()
    assert metrics["terminated"]
    assert not metrics["truncated"]
    assert metrics["completed_operations"] == 60
    assert metrics["completed_orders"] == 15
    assert metrics["time"] == pytest.approx(154.4)
    assert metrics["time"] <= fixed_instance.horizon
    assert metrics["maximum_worker_fatigue"] <= (
        fixed_instance.fatigue.maximum_safe_fatigue + 1e-9
    )
    assert environment.validate_schedule() == []
    rewards = metrics["cumulative_reward"]
    assert rewards["flow"] == pytest.approx(-metrics["total_flow_time"])
    assert rewards["cost"] == pytest.approx(
        -metrics["reconfiguration_cost"]
    )
    assert rewards["variance"] == pytest.approx(
        -metrics["worker_load_variance"]
    )
    assert metrics["completed_reconfigurations"] > 0
    for reconfiguration in environment.reconfigurations.values():
        assert reconfiguration.disassembly_worker_id is not None
        assert reconfiguration.installation_worker_id is not None
        assert reconfiguration.disassembly_end_tick is not None
        assert reconfiguration.installation_start_tick is not None
        assert (
            reconfiguration.installation_start_tick
            >= reconfiguration.disassembly_end_tick
        )
    workers = {worker.id: worker for worker in fixed_instance.workers}
    for record in environment.reconfiguration_log:
        required_module = (
            record["source_module"]
            if record["stage"] == "DIS"
            else record["target_module"]
        )
        assert required_module in workers[record["worker_id"]].qualified_modules


def test_random_policy_is_reproducible_and_feasible(config, fixed_instance):
    traces = []
    metrics = []
    for _ in range(2):
        environment = AssemblySchedulingEnv(config)
        environment.reset(fixed_instance)
        traces.append(run_policy(environment, RandomPolicy(config["seed"])))
        metrics.append(environment.metrics())
        assert environment.validate_schedule() == []
    assert traces[0] == traces[1]
    assert metrics[0]["flow_time_objective"] == pytest.approx(
        metrics[1]["flow_time_objective"]
    )


def test_truncated_reward_identity(config, fixed_instance):
    limited_config = deepcopy(config)
    limited_config["environment"]["max_decisions"] = 1
    environment = AssemblySchedulingEnv(limited_config)
    environment.reset(fixed_instance)
    environment.step(HeuristicPolicy().select_action(environment))
    metrics = environment.metrics()
    assert metrics["truncated"]
    assert metrics["time"] == 0.0
    assert metrics["unfinished_orders"] > 0
    rewards = metrics["cumulative_reward"]
    assert rewards["flow"] == pytest.approx(
        -metrics["flow_time_objective"]
    )
    assert rewards["cost"] == pytest.approx(
        -metrics["reconfiguration_cost"]
    )
    assert rewards["variance"] == pytest.approx(
        -metrics["worker_load_variance"]
    )


def test_decision_limit_truncates_at_current_time(
    config, fixed_instance
):
    limited_config = deepcopy(config)
    environment, _, worker_index, worker_action = (
        start_isolated_disassembly(limited_config, fixed_instance)
    )
    start_time = environment.current_time
    limited_config["environment"]["max_decisions"] = (
        environment._decision_count + 1
    )
    environment.step(worker_action)

    metrics = environment.metrics()
    record = environment.reconfiguration_log[-1]
    assert metrics["truncated"]
    assert metrics["terminal_reason"] == "decision_limit"
    assert metrics["time"] == pytest.approx(start_time)
    assert environment.workers[worker_index].load == 0.0
    assert record["end"] == pytest.approx(start_time)
    assert record["duration"] == 0.0
    assert record["planned_end"] > record["end"]
    assert record["truncated"]


def test_horizon_truncation_settles_partial_worker_task(
    config, fixed_instance
):
    limited_config = deepcopy(config)
    environment, machine_index, worker_index, worker_action = (
        start_isolated_disassembly(limited_config, fixed_instance)
    )
    worker = environment.workers[worker_index]
    fatigue_before = worker.fatigue
    environment.step(worker_action)
    reconfiguration = environment._active_reconfiguration(
        environment.machines[machine_index].spec.id
    )
    total_ticks = (
        reconfiguration.disassembly_end_tick
        - reconfiguration.disassembly_start_tick
    )
    partial_ticks = max(1, total_ticks // 2)
    environment.horizon_tick = (
        reconfiguration.disassembly_start_tick + partial_ticks
    )
    expected_duration = partial_ticks * environment.resolution
    environment.step(environment.advance_action)

    metrics = environment.metrics()
    record = environment.reconfiguration_log[-1]
    expected_fatigue = fatigue_before + (
        fixed_instance.fatigue.disassembly_accumulation_rate_per_minute
        * expected_duration
    )
    assert metrics["truncated"]
    assert metrics["terminal_reason"] == "horizon"
    assert worker.load == pytest.approx(expected_duration)
    assert worker.fatigue == pytest.approx(expected_fatigue)
    assert record["end"] == pytest.approx(environment.current_time)
    assert record["duration"] == pytest.approx(expected_duration)
    assert record["planned_end"] > record["end"]
    rewards = metrics["cumulative_reward"]
    assert rewards["flow"] == pytest.approx(
        -metrics["flow_time_objective"]
    )
    assert rewards["cost"] == pytest.approx(
        -metrics["reconfiguration_cost"]
    )
    assert rewards["variance"] == pytest.approx(
        -metrics["worker_load_variance"]
    )
