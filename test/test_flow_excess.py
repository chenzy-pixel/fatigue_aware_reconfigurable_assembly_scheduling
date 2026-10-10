from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, build_actor_critic
from agent.mo_alns.types import metrics_objectives
from configs import load_config
from configs.config import public_config
from configs.normalization import apply_normalization_manifest
from environment import AssemblySchedulingEnv
from environment.time_context import project_wait_state
from environment.types import (FLOW_EXCESS, FLOW_RAW, flow_mode, metrics_flow_objective,
                               proxy_return_from_metrics, bounded_quality_score)
from data.models import OperationSpec, OrderSpec


@pytest.fixture
def excess_config():
    c = load_config("configs/flow_excess/universal.json")
    c["device"] = "cpu"
    return c


def one_order(template, *, slow=False, horizon=40):
    machines = tuple(replace(m, initial_module="A1", module_parameters={
        "A1": replace(m.module_parameters["A1"], processing_speed_factor=1.0 if i==0 else 2.0)})
        for i,m in enumerate(template.machines[:2]))
    orders = (OrderSpec("R", "W1", 0.0, (OperationSpec("O", "R", 1, "A1", 10.0),)),)
    return replace(template, instance_id="closed_form_flow", machines=machines,
                   orders=orders, horizon=float(horizon),
                   waves={"W1":{"dominant_module":"A1","order_ids":["R"]}})


def test_raw_default_and_unknown_modes(config):
    assert flow_mode(config) == FLOW_RAW
    with pytest.raises(ValueError, match="flow_mode"):
        flow_mode({"flow_mode":"other"})


@pytest.mark.parametrize("mi,elapsed,expected",[(0,5,0),(1,5,2.5),(1,10,5)])
def test_credit_reads_frozen_duration(excess_config,fixed_instance,mi,elapsed,expected):
    env=AssemblySchedulingEnv(excess_config); env.reset(one_order(fixed_instance),build_observation=False)
    assert not env._processing_lower_bound_ticks.flags.writeable
    env.step(env.encode_production_action(0,mi),build_observation=False)
    assert env.operations[0].planned_duration_ticks == (100 if mi==0 else 200)
    env._advance_interval(elapsed*10); env.current_tick=elapsed*10
    # Changing an unrelated resource field cannot change the frozen processing credit.
    env.machines[mi].busy_until_tick += 100
    assert env.flow_excess_objective() == expected
    assert env.flow_lower_bound_credit() == elapsed-expected


@pytest.mark.parametrize("preference",[(1,0,0),(.5,.3,.2),(0,1,0)])
def test_success_rollout_monotone_and_return_identity(excess_config,fixed_instance,preference):
    env=AssemblySchedulingEnv(excess_config); env.reset(fixed_instance,preference=preference,build_observation=False)
    policy=HeuristicPolicy(); last=env.flow_excess_objective(); total=0
    while not env.task_done:
        _,reward,_,_,_=env.step(policy.select_action(env),build_observation=False)
        now=env.flow_excess_objective(); assert now >= last-1e-8
        total += reward.scalarize(excess_config["reward"]); last=now
    assert env.task_succeeded
    metric=env.metrics()
    assert metric["flow_lower_bound_credit"] == env._total_processing_lower_bound_ticks*env.resolution
    assert metric["flow_excess_objective"] == pytest.approx(metric["flow_time_objective"]-metric["flow_processing_lower_bound"])
    assert total == pytest.approx(proxy_return_from_metrics(metric,excess_config,preference=preference))
    assert metrics_objectives(metric,excess_config)[0] == metric["flow_excess_objective"]
    fallback=dict(metric)
    for field in ("reward_preference_quality_score","actual_preference_quality_score","raw_preference_quality_score"):
        fallback.pop(field,None)
    assert proxy_return_from_metrics(fallback,excess_config,preference=preference) == pytest.approx(total)


def test_partial_processing_failure_and_sampling_cutoff(excess_config,fixed_instance):
    env=AssemblySchedulingEnv(excess_config); env.reset(one_order(fixed_instance,horizon=5),build_observation=False)
    env.step(env.encode_production_action(0,1),build_observation=False)
    env._advance_interval(50); env.current_tick=50; env._apply_truncation("horizon")
    assert env.operations[0].planned_duration_ticks==200
    assert env.flow_lower_bound_credit()==2.5
    assert env.flow_excess_objective()==2.5+fixed_instance.unfinished_order_penalty
    assert env.schedule_log[0]["planned_end"]==20
    assert env.schedule_log[0]["planned_duration_ticks"]==200
    cutoff=AssemblySchedulingEnv(excess_config); cutoff.reset(one_order(fixed_instance),build_observation=False)
    cutoff.step(cutoff.encode_production_action(0,1),build_observation=False)
    cutoff._advance_interval(50); cutoff.current_tick=50; cutoff._apply_sampling_truncation("decision_limit")
    assert cutoff.sampling_truncated and not cutoff.task_failed
    assert cutoff.flow_excess_objective()==2.5 and cutoff._flow_penalty==0


def test_raw_physical_and_reward_regression(config,excess_config,fixed_instance):
    raw=AssemblySchedulingEnv(config); new=AssemblySchedulingEnv(excess_config)
    raw.reset(fixed_instance,build_observation=False); new.reset(fixed_instance,build_observation=False)
    policy=HeuristicPolicy()
    while not raw.task_done:
        action=policy.select_action(raw)
        assert action==policy.select_action(new)
        raw.step(action,build_observation=False); new.step(action,build_observation=False)
        assert raw.current_tick==new.current_tick
    assert raw.schedule_log==new.schedule_log
    assert raw.reconfiguration_log==new.reconfiguration_log
    for field in ("flow_time_objective","reconfiguration_cost","worker_load_variance"):
        assert raw.metrics()[field]==new.metrics()[field]


def assert_wait_projection(env):
    certificate=env._wait_certificate()
    if not certificate["allowed"]:
        return False
    before=pickle.dumps(env)
    projected=project_wait_state(env,certificate["wait_ticks"],settle_terminal=True,certificate=certificate)
    assert pickle.dumps(env)==before
    delta=projected.flow_objective()-env.flow_objective()
    values,names=env._build_action_set_features({})
    assert values[names.index("estimated_flow_objective_delta_if_wait")] == pytest.approx(
        delta/env.config["objective_scalarizer"]["scales"]["flow"],abs=2e-6)
    current=env.flow_objective()
    env.step(env.wait_action,build_observation=False)
    assert env.flow_objective()-current==pytest.approx(delta,abs=1e-9)
    assert projected.terminal_reason==env.terminal_reason
    assert projected.decision_type==env.decision_type
    return True


def test_wait_projection_rollout_including_phase_handoff(excess_config,fixed_instance):
    env=AssemblySchedulingEnv(excess_config); env.reset(fixed_instance,build_observation=False)
    policy=HeuristicPolicy(); count=0
    while not env.task_done:
        action=policy.select_action(env)
        if action==env.wait_action:
            assert assert_wait_projection(env); count+=1
        else:
            env.step(action,build_observation=False)
    assert count>10 and env.task_succeeded


def test_wait_horizon_and_mask_deadlock(excess_config,fixed_instance):
    # A release at horizon makes WAIT legal; incomplete work then fails with a real penalty.
    base=one_order(fixed_instance,horizon=5)
    base=replace(base,orders=(replace(base.orders[0],release_time=5),))
    env=AssemblySchedulingEnv(excess_config); env.reset(base,build_observation=False)
    assert assert_wait_projection(env) and env.task_failed
    # A locked task with no qualified disassembly worker fails at worker handoff.
    dead=one_order(fixed_instance,horizon=40)
    op=replace(dead.orders[0].operations[0],required_module="A3")
    capable=next(m for m in fixed_instance.machines if {"A1","A3"} <= set(m.module_parameters))
    dead=replace(dead,machines=(replace(capable,initial_module="A1"),),
                 fatigue=replace(dead.fatigue,maximum_safe_fatigue=0.001),
                 orders=(replace(dead.orders[0],operations=(op,)),))
    env=AssemblySchedulingEnv(excess_config); env.reset(dead,build_observation=False)
    env.step(env.encode_production_action(0,0),build_observation=False)
    assert assert_wait_projection(env)
    assert env.task_failed and env.terminal_reason=="unrecoverable_deadlock"


def test_parallel_orders_same_tick_events_and_serial_zero_time_actions(excess_config,fixed_instance):
    instance=one_order(fixed_instance)
    fast=replace(instance.machines[1],module_parameters=instance.machines[0].module_parameters)
    orders=tuple(OrderSpec(f"R{i}","W1",0.0,tuple(
        OperationSpec(f"O{i}{j}",f"R{i}",j+1,"A1",10.0) for j in range(2))) for i in range(2))
    instance=replace(instance,machines=(instance.machines[0],fast),orders=orders)
    env=AssemblySchedulingEnv(excess_config); env.reset(instance,build_observation=False)
    env.step(env.encode_production_action(0,0),build_observation=False)
    env.step(env.encode_production_action(2,1),build_observation=False)
    assert env.current_tick==0 and env.flow_excess_objective()==0
    env._advance_interval(100); env.current_tick=100
    before=env.flow_excess_objective()
    env._process_events_at_current_tick()
    assert env.flow_excess_objective()==before==0
    assert env.flow_lower_bound_credit()==20
    env.step(env.encode_production_action(1,0),build_observation=False)
    env.step(env.encode_production_action(3,1),build_observation=False)
    assert env.current_tick==100 and env.flow_excess_objective()==0
    assert assert_wait_projection(env) and env.task_succeeded
    assert env.flow_lower_bound_credit()==40


@pytest.mark.parametrize("mode",[FLOW_RAW,FLOW_EXCESS])
@pytest.mark.parametrize("preference",[(1,0,0),(.5,.3,.2)])
def test_failed_physical_return_identity(mode,preference,config,excess_config,fixed_instance):
    selected=config if mode==FLOW_RAW else excess_config
    env=AssemblySchedulingEnv(selected)
    env.reset(one_order(fixed_instance,horizon=5),preference=preference,build_observation=False)
    total=0; policy=HeuristicPolicy()
    while not env.task_done:
        _,reward,_,_,_=env.step(policy.select_action(env),build_observation=False)
        total+=reward.scalarize(selected["reward"])
    assert env.task_failed
    assert total==pytest.approx(proxy_return_from_metrics(env.metrics(),selected,preference=preference))


def test_terminal_wait_has_neutral_slack(excess_config,fixed_instance):
    from environment.time_context import WAIT_TIME_FEATURES
    instance=replace(one_order(fixed_instance,horizon=5),orders=(
        replace(one_order(fixed_instance).orders[0],release_time=5),))
    env=AssemblySchedulingEnv(excess_config); env.reset(instance,build_observation=False)
    values,names=env._build_action_set_features({})
    assert all(values[names.index(name)]==0 for name in WAIT_TIME_FEATURES)


def test_uncached_wait_projection_without_supplied_certificate_is_isolated(excess_config,fixed_instance):
    base=one_order(fixed_instance)
    env=AssemblySchedulingEnv(excess_config)
    env.reset(replace(base,orders=(replace(base.orders[0],release_time=5),)),build_observation=False)
    env._invalidate_resource_snapshot()
    before=pickle.dumps(env)
    projected=project_wait_state(env,50,settle_terminal=True)
    assert pickle.dumps(env)==before
    assert projected.current_tick==50
    assert not projected._processing_lower_bound_ticks.flags.writeable


def test_full_projection_isolates_nested_logs(excess_config,fixed_instance):
    env=AssemblySchedulingEnv(excess_config)
    env.reset(one_order(fixed_instance),build_observation=False)
    env.step(env.encode_production_action(0,0),build_observation=False)
    env.schedule_log[0]["audit_extension"]={"flags":[False]}
    certificate=env._wait_certificate()
    before=pickle.dumps(env)
    projected=project_wait_state(env,certificate["wait_ticks"],settle_terminal=True,certificate=certificate)
    projected.schedule_log[0]["audit_extension"]["flags"][0]=True
    assert pickle.dumps(env)==before


def test_manifest_mismatch_and_old_raw_snapshot(config,excess_config,tmp_path):
    invalid=deepcopy(excess_config); invalid["objective_scalarizer"]["flow_mode"]=FLOW_RAW
    from configs.config import PROJECT_ROOT
    with pytest.raises(ValueError,match="flow_mode"):
        apply_normalization_manifest(invalid,project_root=PROJECT_ROOT)
    old=public_config(config)
    old["network"].pop("flow_mode",None); old["network"].pop("reward_version",None)
    old["runtime_manifest"].pop("flow_mode",None); old["runtime_manifest"].pop("normalization_manifest_sha256",None)
    path=tmp_path/"old_raw.json"; path.write_text(json.dumps(old))
    assert flow_mode(load_config(path))==FLOW_RAW


def test_checkpoint_modes_and_same_mode_reload(config,excess_config,fixed_instance,tmp_path):
    torch.set_num_threads(2)
    agents=[]
    for c in (config,excess_config):
        obs=AssemblySchedulingEnv(c).reset(fixed_instance)
        network=build_actor_critic(obs,c["network"])
        agents.append(PPOAgent(network,c["ppo"],device="cpu"))
    path=tmp_path/"excess.pt"; agents[1].save(path)
    agents[1].load(path)
    with pytest.raises(ValueError,match="incompatible"):
        agents[0].load(path)
    legacy=tmp_path/"raw.pt"; agents[0].save(legacy)
    payload=torch.load(legacy,weights_only=False)
    payload["network_spec"].pop("flow_mode"); payload["network_spec"].pop("reward_version")
    torch.save(payload,legacy); agents[0].load(legacy)
    with pytest.raises(ValueError,match="incompatible"):
        agents[1].load(legacy)
