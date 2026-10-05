from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.network import HeteroGraphActorCritic, infer_checkpoint_network_spec, normalize_network_config
from environment import AssemblySchedulingEnv, CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE
from environment.time_context import (
    ORDER_TIME_FEATURE, WAIT_TIME_FEATURES, WORKER_WAIT_FEATURE, project_wait_state,
)
from environment.types import OperationState, ReconfigurationStage


def _chain_instance(fixed_instance, *, mismatch=False, second_release=None):
    module = fixed_instance.modules[0]
    source = fixed_instance.orders[0]
    first = replace(source, release_time=0.0, operations=tuple(
        replace(op, required_module=module, base_processing_time=duration)
        for op, duration in zip(source.operations[:2], (10.0, 20.0), strict=True)
    ))
    machines = tuple(replace(machine, initial_module=(
        next(m for m in machine.module_parameters if m != module) if mismatch
        else module
    )) if module in machine.module_parameters else machine for machine in fixed_instance.machines)
    orders = (first,)
    if second_release is not None:
        second = fixed_instance.orders[1]
        orders += (replace(second, release_time=second_release, operations=(
            replace(second.operations[0], required_module=module, base_processing_time=15.0),
        )),)
    return replace(fixed_instance, orders=orders, machines=machines)


def _wait_columns(observation):
    return dict(zip(observation.action_set_feature_names, observation.action_set_features, strict=True))


def _commit(env):
    mask = env.get_action_mask()
    action = next(int(i) for i in np.flatnonzero(~mask[:-1])
                  if env.machines[env.decode_production_action(int(i))[1]].current_module
                  != env.operations[env.decode_production_action(int(i))[0]].spec.required_module)
    op, machine = env.decode_production_action(action)
    env.step(action)
    return op, machine


def test_order_slack_covers_whole_chain_and_remaining_processing(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    observation = env.reset(_chain_instance(fixed_instance))
    times = [min(env.estimate_processing_ticks(op, m) for m, machine in enumerate(env.machines)
                 if env.operations[op].spec.required_module in machine.spec.module_parameters)
             for op in range(2)]
    expected = sum(times)
    assert env.estimated_order_finish_ticks()[env.instance.orders[0].id] == expected
    slack = (env.horizon_tick - expected) / env.horizon_tick
    assert observation.orders[0, -1] == pytest.approx(slack)
    action = next(int(i) for i in np.flatnonzero(~env.get_action_mask()[:-1])
                  if env.estimate_processing_ticks(*env.decode_production_action(int(i))) == times[0])
    env.step(action, build_observation=False)
    env._advance_interval(times[0] // 2)
    env.current_tick = times[0] // 2
    env._invalidate_resource_snapshot()
    assert env.operations[0].state == OperationState.PROCESSING
    assert env.estimated_order_finish_ticks()[env.instance.orders[0].id] == expected
    assert env.estimated_order_slack_norm(env.instance.orders[0].id) == pytest.approx(slack)


def test_production_edges_carry_their_associated_order_slack(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    relation = observation.relations[CAPABLE_EDGE]
    for row, op in enumerate(relation.edge_index[0]):
        order_id = env.operations[int(op)].spec.order_id
        order = observation.node_ids["order"].index(order_id)
        assert relation.edge_features[row, -1] == observation.orders[order, -1]


@pytest.mark.parametrize("stage", [ReconfigurationStage.DIS, ReconfigurationStage.INS])
def test_running_reconfiguration_counts_only_its_remaining_duration(config, fixed_instance, stage):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance, mismatch=True))
    op, machine = _commit(env)
    env.step(env.wait_action)
    env.step(int(np.flatnonzero(~env.get_action_mask()[:-1])[0]))
    reconfiguration = env._active_reconfiguration(env.machines[machine].spec.id)
    if stage == ReconfigurationStage.INS:
        while reconfiguration.stage == ReconfigurationStage.DIS:
            env.step(env.wait_action)
        # The production phase hands the pending installation to the worker phase.
        if env.decision_type.value == "PRODUCTION":
            env.step(env.wait_action)
        env.step(int(np.flatnonzero(~env.get_action_mask()[:-1])[0]))
    assert reconfiguration.stage == stage
    order = env.operations[op].spec.order_id
    finish_before = env.estimated_order_finish_ticks()[order]
    end = env.machines[machine].busy_until_tick
    midpoint = env.current_tick + max(1, (end - env.current_tick) // 2)
    env._advance_interval(midpoint)
    env.current_tick = midpoint
    env._invalidate_resource_snapshot()
    assert env.estimated_order_finish_ticks()[order] == pytest.approx(finish_before, abs=1)
    certificate = env._wait_certificate()
    before = pickle.dumps(env)
    projected = project_wait_state(env, certificate["wait_ticks"])
    assert pickle.dumps(env) == before
    env.step(env.wait_action, build_observation=False)
    assert projected.minimum_active_order_slack_norm() == pytest.approx(env.minimum_active_order_slack_norm())
    assert [worker.fatigue for worker in projected.workers] == pytest.approx(
        [worker.fatigue for worker in env.workers])


def test_known_machine_occupancy_and_soft_negative_slack(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance))
    baseline = env.estimated_order_finish_ticks()[env.instance.orders[0].id]
    # Supply known commitments for every compatible resource to isolate waiting
    # from processing-time and configuration effects.
    from environment.types import MachineState
    for machine in env.machines:
        machine.state = MachineState.PROCESSING
        machine.busy_until_tick = env.horizon_tick + 30
    env._invalidate_resource_snapshot()
    delayed = env.estimated_order_finish_ticks()[env.instance.orders[0].id]
    assert delayed == env.horizon_tick + 30 + baseline
    assert env.estimated_order_slack_norm(env.instance.orders[0].id) < 0


def test_terminal_observation_does_not_advance_settled_worker_tasks(config, fixed_instance):
    settings = deepcopy(config)
    settings["environment"]["max_decisions"] = 3
    env = AssemblySchedulingEnv(settings)
    env.reset(_chain_instance(fixed_instance, mismatch=True))
    _commit(env)
    env.step(env.wait_action)
    observation, _, terminated, truncated, _ = env.step(int(np.flatnonzero(~env.get_action_mask()[:-1])[0]))
    assert truncated and not terminated
    observation.validate()
    assert _wait_columns(observation)["wait_duration_norm"] > 0
    assert np.isfinite(_wait_columns(observation)[WAIT_TIME_FEATURES[1]])


def test_wait_projection_advances_work_instead_of_subtracting_duration(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance))
    fastest = min(np.flatnonzero(~env.get_action_mask()[:-1]),
                  key=lambda i: env.estimate_processing_ticks(*env.decode_production_action(int(i))))
    env.step(int(fastest))
    before = env.minimum_active_order_slack_norm()
    observation = env.observe()
    columns = _wait_columns(observation)
    assert columns["wait_duration_norm"] > 0
    assert columns[WAIT_TIME_FEATURES[0]] == pytest.approx(before, abs=1e-7)
    assert columns[WAIT_TIME_FEATURES[1]] == pytest.approx(0.0, abs=1e-7)
    assert columns[WAIT_TIME_FEATURES[0]] != pytest.approx(before - columns["wait_duration_norm"])
    env.step(env.wait_action)
    assert env.operations[0].state == OperationState.DONE
    assert env.operations[1].state == OperationState.READY
    assert columns[WAIT_TIME_FEATURES[0]] == pytest.approx(env.minimum_active_order_slack_norm())


def test_wait_projection_release_and_completion_match_actual_events(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance, second_release=5.0))
    action = int(np.flatnonzero(~env.get_action_mask()[:-1])[0])
    env.step(action)
    certificate = env._wait_certificate()
    assert certificate["wait_ticks"] > 0
    before = pickle.dumps(env)
    projected = project_wait_state(env, certificate["wait_ticks"])
    assert pickle.dumps(env) == before
    env.step(env.wait_action, build_observation=False)
    assert projected.current_tick == env.current_tick
    assert projected._order_released == env._order_released
    assert projected._order_completion_tick == env._order_completion_tick
    assert [op.state for op in projected.operations] == [op.state for op in env.operations]
    assert projected.minimum_active_order_slack_norm() == pytest.approx(env.minimum_active_order_slack_norm())


def test_zero_time_handoff_preserves_slack_and_stage_wait_age(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance, mismatch=True))
    op, machine = _commit(env)
    observation = env.observe()
    columns = _wait_columns(observation)
    assert columns["wait_duration_norm"] == 0
    assert columns[WAIT_TIME_FEATURES[1]] == 0
    assert columns[WAIT_TIME_FEATURES[0]] == pytest.approx(env.minimum_active_order_slack_norm())
    env.step(env.wait_action)
    env._advance_interval(env.current_tick + 7)
    env.current_tick += 7
    env._invalidate_resource_snapshot()
    relation = env.observe().relations[SERVICE_CANDIDATE_EDGE]
    selected = relation.edge_index[0] == machine
    assert np.all(relation.edge_features[selected, -1] == pytest.approx(7 / env.horizon_tick))
    assert np.all(relation.edge_features[selected, -2] == pytest.approx(
        env.estimated_order_slack_norm(env.operations[op].spec.order_id)))
    env.step(int(np.flatnonzero(~env.get_action_mask()[:-1])[0]))
    reconfiguration = env._active_reconfiguration(env.machines[machine].spec.id)
    while reconfiguration.stage == ReconfigurationStage.DIS:
        env.step(env.wait_action)
    assert reconfiguration.stage == ReconfigurationStage.WAIT_INS
    relation = env.observe().relations[SERVICE_CANDIDATE_EDGE]
    assert np.all(relation.edge_features[relation.edge_index[0] == machine, -1] == 0)


@pytest.mark.parametrize("phase", ["production", "worker", "wait"])
def test_new_time_inputs_change_all_expert_contexts_but_not_direct_scores(config, fixed_instance, phase):
    env = AssemblySchedulingEnv(config)
    env.reset(_chain_instance(fixed_instance, mismatch=phase == "worker"))
    if phase == "worker":
        _commit(env)
        env.step(env.wait_action)
    observation = env.observe()
    mask = env.get_action_mask()
    settings = dict(config["network"], hidden_dim=8, message_passing_layers=1)
    network = build_actor_critic(observation, settings)
    prefix = "production_wait" if phase == "wait" else phase
    experts = getattr(network, prefix + "_experts")
    hidden = network.hidden_dim
    if phase == "wait":
        first, second, column, embedding_column = network.wait_feature_encoder[0], network.wait_action_encoder[0], -2, 0
    else:
        first = getattr(network, phase + "_edge_encoder")[0]
        second = getattr(network, phase + "_action_encoder")[0]
        column = -2 if phase == "worker" else -1
        embedding_column = (4 if phase == "worker" else 3) * hidden
    with torch.no_grad():
        first.weight.zero_(); first.bias.zero_(); first.weight[0, column] = 1
        second.weight.zero_(); second.bias.zero_(); second.weight[0, embedding_column] = 1
        for expert in experts.experts.values():
            expert.context[0].weight.zero_(); expert.context[0].bias.zero_(); expert.context[0].weight[0, 0] = 1
            expert.context[-1].weight.zero_(); expert.context[-1].bias.zero_(); expert.context[-1].weight[0, 0] = 1
    values = {}
    handles = [expert.register_forward_hook(
        lambda module, args, output, name=name: values.__setitem__(name, tuple(v.clone() for v in output))
    ) for name, expert in experts.experts.items()]
    if phase == "wait":
        one = observation.action_set_features.copy(); two = one.copy()
        one[-2] = 0.1; two[-2] = 0.6
        observations = [replace(observation, action_set_features=one), replace(observation, action_set_features=two)]
    else:
        relation_type = SERVICE_CANDIDATE_EDGE if phase == "worker" else CAPABLE_EDGE
        observations = []
        for amount in (0.1, 0.6):
            relations = dict(observation.relations)
            relation = relations[relation_type].copy()
            relation.edge_features[:, column] = amount
            relations[relation_type] = relation
            observations.append(replace(observation, relations=relations))
    try:
        captures = []
        with torch.no_grad():
            for item in observations:
                network(item, mask, device="cpu")
                captures.append(dict(values))
        for name in ("flow", "cost", "variance"):
            torch.testing.assert_close(captures[0][name][0], captures[1][name][0], rtol=0, atol=0)
            assert torch.all(captures[1][name][1] > captures[0][name][1])
        assert len({expert.context[0].weight.data_ptr() for expert in experts.experts.values()}) == 3
    finally:
        for handle in handles:
            handle.remove()


def _schema5_observation(observation):
    nodes = {key: value.copy() for key, value in observation.node_features.items()}
    nodes["order"] = nodes["order"][:, :-1]
    names = dict(observation.node_feature_names)
    names["order"] = names["order"][:-1]
    relations = dict(observation.relations)
    for edge, count in ((CAPABLE_EDGE, 1), (SERVICE_CANDIDATE_EDGE, 2)):
        relation = relations[edge]
        relations[edge] = replace(relation, edge_features=relation.edge_features[:, :-count],
                                  feature_names=relation.feature_names[:-count])
    return replace(observation, node_features=nodes, node_feature_names=names, relations=relations,
                   global_features=observation.global_features[:11],
                   global_feature_names=observation.global_feature_names[:11],
                   action_set_features=observation.action_set_features[:-2],
                   action_set_feature_names=observation.action_set_feature_names[:-2])


@pytest.mark.parametrize("variant", ["hetero_gnn", "node_mlp_pool", "shared_preference"])
def test_schema5_checkpoint_rejected_for_each_network_variant(config, fixed_instance, tmp_path, variant):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    settings = dict(config['network'], hidden_dim=16)
    if variant == 'shared_preference':
        settings['actor_head_variant'] = variant
    else:
        settings['encoder_variant'] = variant
    agent = PPOAgent(build_actor_critic(observation, settings), config['ppo'], device='cpu')
    path = tmp_path/'schema5.pt'
    agent.save(path)
    payload = torch.load(path,weights_only=False)
    payload['network_spec']['observation_schema_version'] = 5
    torch.save(payload,path)
    with pytest.raises(ValueError,match='requires retraining'):
        agent.load(path,load_optimizer=True,allow_observation_migration=True)


def test_schema10_requires_explicit_time_features(config, fixed_instance):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    with pytest.raises(ValueError, match="schema-10"):
        build_actor_critic(_schema5_observation(observation), config["network"])
    spec = build_actor_critic(observation, config["network"]).network_spec()
    spec["time_context_feature_schema"]["worker"] = []
    with pytest.raises(ValueError, match="time context feature schema"):
        infer_checkpoint_network_spec({"network_spec": spec})
