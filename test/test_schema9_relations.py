"""Actual allocation lifecycle, graph transport, encoding, and numeric boundaries."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import RolloutBuffer
from agent.ppo.network import build_actor_critic, infer_checkpoint_network_spec
from agent.ppo.parallel import _worker_roll_forward
from environment import AssemblySchedulingEnv, ASSEMBLY_EDGE_TYPES, EdgeStore, PROCESSING_ON_EDGE, SERVED_BY_EDGE
from environment.dynamics import EPSILON, quantize_to_ticks
from scripts.audit_observation_reward import build_alias_instance, reachable_histories
from scripts.audit_state_sufficiency import (
    MODES, effective, worker_alias_instance, worker_histories, virtual_observation,
    reconstructed_quantities, reconstructed_events, actual_events, encoding_checks, boundary_checks,
)


@pytest.mark.parametrize("mode", MODES)
def test_actual_relation_lifecycle_and_physical_reconstruction(config, fixed_instance, mode):
    env = AssemblySchedulingEnv(effective(config, mode))
    env.reset(fixed_instance)
    stages = set()
    saw_processing = saw_serving = False
    policy = HeuristicPolicy()
    while not env.task_done:
        obs = virtual_observation(env)
        obs.validate()
        physical = reconstructed_quantities(env, obs)
        assert physical["tick"] == env.current_tick
        assert reconstructed_events(env, obs, physical) == actual_events(env)
        np.testing.assert_allclose(physical["committed"]/env.instance.horizon,
                                   env._committed_worker_loads/env.instance.horizon, atol=1e-6, rtol=1e-6)
        saw_processing |= bool(obs.relations[PROCESSING_ON_EDGE].num_edges)
        saw_serving |= bool(obs.relations[SERVED_BY_EDGE].num_edges)
        stages.update(r.stage.value for r in env.reconfigurations.values())
        env.step(policy.select_action(env), build_observation=False)
    final = env.observe()
    final.validate()
    assert env.task_succeeded
    assert saw_processing and saw_serving
    assert {"WAIT_DIS", "DIS", "WAIT_INS", "INS", "DONE"} <= stages
    for kind in (PROCESSING_ON_EDGE, SERVED_BY_EDGE):
        assert final.relations[kind].edge_index.shape == (2, 0)
        assert final.relations[kind].edge_features.shape == (0, 0)


def test_actual_assignments_are_preserved_by_copy_pickle_buffer_and_batch(config, fixed_instance):
    instance = worker_alias_instance(fixed_instance)
    env = worker_histories(config, instance)[0][0]
    obs = env.observe()
    buffer = RolloutBuffer(preserve_graph=True)
    buffer.add(obs, env.get_action_mask(), env.wait_action, 0, 0, 0, False)
    for copied in (obs.copy(), pickle.loads(pickle.dumps(obs)), buffer.transitions[0].observation):
        copied.validate()
        for kind in (PROCESSING_ON_EDGE, SERVED_BY_EDGE):
            np.testing.assert_array_equal(copied.relations[kind].edge_index, obs.relations[kind].edge_index)
            assert copied.relations[kind].edge_features.shape == obs.relations[kind].edge_features.shape
    production = reachable_histories(config, build_alias_instance(fixed_instance))[0][0]
    observations = [obs, production.observe()]
    model = build_actor_critic(obs, {**config["network"], "hidden_dim": 16})
    fast = model._collate_graphs(observations, device="cpu")
    reference = model._collate_graphs_reference(observations, device="cpu")
    for kind in ASSEMBLY_EDGE_TYPES:
        for actual, expected in zip(fast.relations[kind][:2], reference.relations[kind][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fast.relations[SERVED_BY_EDGE][0].shape == (2, 2)
    assert fast.relations[PROCESSING_ON_EDGE][0].shape == (2, 7)
    masks = [env.get_action_mask(), production.get_action_mask()]

    def forward_and_gradients(mode):
        model.execution_mode = mode
        model.zero_grad(set_to_none=True)
        logits, values = model.forward_batch(observations, masks, device="cpu")
        loss = values.square().sum() + sum(
            logits[index, :len(mask)][~torch.as_tensor(mask)].square().sum()
            for index, mask in enumerate(masks))
        loss.backward()
        return logits.detach(), values.detach(), {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()}

    reference_logits, reference_values, reference_grads = forward_and_gradients("reference_v8")
    logits, values, gradients = forward_and_gradients("phase_batched_v1")
    torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(values, reference_values, atol=1e-5, rtol=1e-4)
    for name, gradient in gradients.items():
        expected = reference_grads[name]
        assert (gradient is None) == (expected is None)
        if gradient is not None:
            torch.testing.assert_close(gradient, expected, atol=1e-5, rtol=1e-4)
    for relation in ("operation__processing_on__machine", "machine__served_by__worker"):
        gradient = gradients[f"message_layers.0.transforms.{relation}.weight"]
        assert gradient is not None and gradient.abs().max() > 0


@pytest.mark.parametrize("corruption", ["direction", "attributes", "missing", "duplicate", "phase", "end", "qualification"])
def test_raw_actual_relations_reject_inconsistent_graphs(config, fixed_instance, corruption):
    env = worker_histories(config, worker_alias_instance(fixed_instance))[0][0]
    obs = env.observe().copy()
    store = obs.relations[SERVED_BY_EDGE]
    if corruption == "direction":
        obs.relations[SERVED_BY_EDGE] = replace(store, bidirectional=False)
    elif corruption == "attributes":
        obs.relations[SERVED_BY_EDGE] = replace(store, edge_features=np.ones((2, 1)), feature_names=("flag",))
    elif corruption == "missing":
        obs.relations[SERVED_BY_EDGE] = EdgeStore(np.empty((2, 0), dtype=np.int64), np.empty((0, 0)), (), True)
    elif corruption == "duplicate":
        pair = store.edge_index[:, :1]
        obs.relations[SERVED_BY_EDGE] = replace(store, edge_index=np.repeat(pair, 2, axis=1))
    else:
        wi = store.edge_index[1, 0]
        names = obs.node_feature_names["worker"]
        if corruption == "phase":
            obs.workers[wi, names.index("state_INS")] = 0
            obs.workers[wi, names.index("state_DIS")] = 1
        elif corruption == "qualification":
            obs.workers[wi, names.index("qualified_module_A2")] = 0
        else:
            obs.workers[wi, names.index("remaining_busy_time_norm")] += .01
    with pytest.raises(ValueError, match="served_by"):
        obs.validate()


def test_processing_relation_requires_all_actual_operations(config, fixed_instance):
    env = reachable_histories(config, build_alias_instance(fixed_instance))[0][0]
    obs = env.observe()
    relation = obs.relations[PROCESSING_ON_EDGE]
    missing = replace(relation, edge_index=relation.edge_index[:, 1:], edge_features=np.empty((6, 0)))
    with pytest.raises(ValueError, match="processing_on"):
        replace(obs, relations={**obs.relations, PROCESSING_ON_EDGE: missing}).validate()
    env.operations[0].machine_id = None
    with pytest.raises(RuntimeError, match="processing_on"):
        env._actual_resource_relations()


def test_busy_worker_end_time_is_checked_in_kernel(config, fixed_instance):
    env = worker_histories(config, worker_alias_instance(fixed_instance))[0][0]
    wi = env.observe().relations[SERVED_BY_EDGE].edge_index[1, 0]
    env.workers[wi].busy_until_tick += 1
    with pytest.raises(RuntimeError, match="served_by"):
        env._actual_resource_relations()


def test_relation_aware_encoding_distinguishes_both_counterexamples(config, fixed_instance):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        fixtures = {"processing": (build_alias_instance(fixed_instance), reachable_histories),
                    "serving": (worker_alias_instance(fixed_instance), worker_histories)}
        results = encoding_checks(config, fixtures)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(results) == 60
    for row in results:
        if row["variant"] == "node_mlp_pool":
            assert row["graph_difference_float64"] < 1e-12
        else:
            assert row["graph_difference_float64"] > 1e-8
            assert max(row["node_differences"].values()) > 1e-6


def test_worker_guard_preserves_actual_service_for_bootstrap(config, fixed_instance):
    instance = worker_alias_instance(fixed_instance)
    paths = worker_histories(config, instance)[1]
    settings = deepcopy(config)
    settings["environment"]["max_decisions"] = 11
    env = AssemblySchedulingEnv(settings)
    env.reset(instance)
    for action in paths[0][:10]:
        env.step(action, build_observation=False)
    response = _worker_roll_forward(0, env, env.observe(), preserve_graph=True,
        requested_action=paths[0][10], drain_physical_forced_actions=True, max_environment_steps=None)
    assert response.truncated and not response.terminated and not env.task_failed
    assert env.current_time == 34 and env._flow_penalty == 0
    response.observation.validate()
    assert response.observation.relations[SERVED_BY_EDGE].num_edges == 1
    recovered = reconstructed_quantities(env, response.observation)
    np.testing.assert_allclose(recovered["committed"], env._committed_worker_loads, atol=1e-5)
    assert response.action_mask is not None


@pytest.mark.parametrize("phase", ["DIS", "INS"])
def test_partial_real_failure_retains_runtime_relation(config, fixed_instance, phase):
    # Existing legal boundary probes exercise both partial phases and completion precedence.
    rows = boundary_checks(config, fixed_instance)
    matches = [row for row in rows if row.get("kind") == phase]
    assert len(matches) == 2
    assert all(row["legal_terminal_transition"] and row["failed"] and row["terminated"] for row in matches)
    assert all(row["failure_penalty"] == config["reward"]["terminal_failure_penalty"] for row in matches)
    assert all(row["actual_serving_edges"] == 1 and row["actual_processing_edges"] == 0 for row in matches)


@pytest.mark.parametrize("corruption", ["missing", "attributes"])
def test_checkpoint_spec_requires_actual_relation_contract(config, fixed_instance, corruption):
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    spec = build_actor_critic(obs, {**config["network"], "hidden_dim": 16}).network_spec()
    assert spec["observation_schema_version"] == 10 and len(spec["edge_feature_dimensions"]) == 14
    if corruption == "missing":
        del spec["edge_feature_dimensions"][SERVED_BY_EDGE]
    else:
        spec["edge_feature_dimensions"][PROCESSING_ON_EDGE] = 1
    with pytest.raises(ValueError, match="(edge feature|zero edge)"):
        infer_checkpoint_network_spec({"network_spec": spec})


def test_duration_rounding_and_safe_fatigue_boundaries(config, fixed_instance):
    assert quantize_to_ticks(1 + EPSILON/2, .1) == 10
    assert quantize_to_ticks(1 + 2*EPSILON, .1) == 11
    instance = worker_alias_instance(fixed_instance)
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    for action in worker_histories(config, instance)[1][0][:10]:
        env.step(action, build_observation=False)
    record = env._pending_reconfiguration(env.machines[5].spec.id)
    worker = env.workers[0]
    limit = instance.fatigue.maximum_safe_fatigue
    rate = env._stage_accumulation_rate(record)
    for _ in range(5):
        duration = env._stage_duration_ticks(record, worker)*env.resolution
        worker.fatigue = limit - rate*duration
    boundary = worker.fatigue
    worker.fatigue = boundary - 1e-6
    assert env._worker_can_start(record, worker)
    worker.fatigue = boundary + 1e-6
    assert not env._worker_can_start(record, worker)
