"""Exact WAIT-recovery query against an exhaustive tick-by-tick oracle."""
from copy import deepcopy

import pytest

from agent.baselines import HeuristicPolicy
from environment import AssemblySchedulingEnv
from environment.types import MachineState, OperationState, ReconfigurationStage, WorkerState


def exhaustive_improvement(env):
    """Definition of the first duration decrease, without pruning or deduplication."""
    if env.instance.fatigue.idle_recovery_rate_per_minute <= 0:
        return None
    end = min(env.horizon_tick, min((e[0] for e in env._events if e[0] > env.current_tick), default=env.horizon_tick))
    mismatches = []
    stages = []
    for op in env.operations:
        if op.state != OperationState.READY:
            continue
        for machine in env.machines:
            if (machine.state != MachineState.IDLE or machine.current_module in (env.instance.no_module_state, op.spec.required_module)
                    or op.spec.required_module not in machine.spec.module_parameters):
                continue
            current = env._idle_worker_reconfiguration_ticks_at(machine, op.spec.required_module, env.current_tick)
            if current is not None:
                mismatches.append((machine, op.spec.required_module, current))
    for rec in env.reconfigurations.values():
        if rec.stage not in (ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS):
            continue
        module = rec.source_module if rec.stage == ReconfigurationStage.WAIT_DIS else rec.target_module
        for worker in env.workers:
            if worker.state == WorkerState.IDLE and module in worker.spec.qualified_modules and env._worker_can_start(rec, worker):
                stages.append((rec, worker, env._stage_duration_ticks(rec, worker)))
    for tick in range(env.current_tick + 1, end + 1):
        for machine, module, duration in mismatches:
            after = env._idle_worker_reconfiguration_ticks_at(machine, module, tick)
            if after is not None and after < duration:
                return tick
        for rec, worker, duration in stages:
            fatigue = max(0., worker.fatigue - env.instance.fatigue.idle_recovery_rate_per_minute * (tick - env.current_tick) * env.resolution)
            if env._stage_duration_ticks(rec, worker, fatigue_override=fatigue) < duration:
                return tick
    return None


@pytest.mark.parametrize("idle_fatigue", [0., 1e-8, .23, .749])
def test_recovery_pruning_matches_exhaustive_ticks_across_native_phases(config, fixed_instance, idle_fatigue):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    policy = HeuristicPolicy()
    checked = 0
    for step in range(100):
        if env.task_done:
            break
        if step in (0, 3, 10, 25, 50, 75):
            query = deepcopy(env)
            for worker in query.workers:
                if worker.state == WorkerState.IDLE:
                    worker.fatigue = idle_fatigue
            query._invalidate_resource_snapshot()
            expected = exhaustive_improvement(query)
            assert query._earliest_reconfiguration_duration_improvement_tick() == expected
            assert query._earliest_reconfiguration_duration_improvement_tick() == expected  # same-state cache
            checked += 1
        env.step(policy.select_action(env), build_observation=False)
    assert checked >= 3
