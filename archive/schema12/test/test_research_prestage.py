from dataclasses import replace

import numpy as np

from agent.baselines import HeuristicPolicy
from data.models import OperationSpec,OrderSpec
from environment import AssemblySchedulingEnv
from environment.types import MachineState,OperationState,ReconfigurationStage
from scripts.research_objectives_prestage import PrestagingEnv,rollout


def future_instance(template):
    # One future order, two source machines: one can be prepared while one remains.
    op = OperationSpec("PREP_OP","PREP_ORDER",1,"A2",8.0)
    order = OrderSpec("PREP_ORDER","W1",40.0,(op,))
    machines = tuple(replace(machine,initial_module="A1") for machine in template.machines[:2])
    return replace(template,instance_id="prestage_regression",horizon=100.0,
                   machines=machines,orders=(order,),waves={"W1":{"dominant_module":"A2","order_ids":[order.id]}},
                   workers=tuple(replace(worker,qualified_modules=("A1","A2","A3"),initial_fatigue=0.0) for worker in template.workers))


def test_future_module_is_unreachable_in_original_mask(config,fixed_instance):
    instance = future_instance(fixed_instance)
    env = AssemblySchedulingEnv(config)
    env.reset(instance,build_observation=False)
    assert np.flatnonzero(~env.get_action_mask()).tolist() == [env.wait_action]
    env.step(env.wait_action,build_observation=False)
    assert env.current_time == 40.0
    assert all(machine.current_module == "A1" for machine in env.machines)
    assert not env.reconfigurations


def test_prestage_installation_finishes_idle_with_order_unreleased(config,fixed_instance):
    env = PrestagingEnv(config)
    env.reset(future_instance(fixed_instance),build_observation=False)
    assert env.try_prestage(40.0)
    assert env.operations[0].state == OperationState.UNRELEASED
    assert env.operations[0].machine_id is None
    policy = HeuristicPolicy()
    for _ in range(100):
        env.step(policy.select_action(env),build_observation=False)
        if all(rec.stage == ReconfigurationStage.DONE for rec in env.reconfigurations.values()):
            break
    assert env.current_time < 40.0
    assert env.operations[0].state == OperationState.UNRELEASED
    assert not env.schedule_log
    prepared = next(machine for machine in env.machines if machine.current_module == "A2")
    assert prepared.state == MachineState.IDLE
    assert prepared.locked_operation_id is None
    assert env.metrics()["reconfiguration_cost"] > 0
    assert not env.validate_schedule()


def test_wakeup_reaches_window_and_resource_feasible_prestage_saves_flow(config,fixed_instance):
    instance = future_instance(fixed_instance)
    base,_ = rollout(config,instance)
    no_timer,_ = rollout(config,instance,20.0)
    prepared,_ = rollout(config,instance,20.0,wakeup=True)
    assert no_timer["prestage_count"] == 0
    assert prepared["prestage_count"] == 1
    assert base["success"] and prepared["success"]
    assert prepared["flow"] < base["flow"]
    assert prepared["cost"] > 0
    assert prepared["violations"] == 0
