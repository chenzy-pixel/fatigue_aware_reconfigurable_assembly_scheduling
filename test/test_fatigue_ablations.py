from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.ppo import build_actor_critic
from agent.baselines import HeuristicPolicy
from agent.ppo.network_v8 import assert_network_config_matches_spec
from data.generate_orders import InstanceGenerator
from environment import AssemblySchedulingEnv
from environment import proxy_return_from_metrics
from environment import OPERATION_ORDER_EDGE, EdgeStore
from environment.fatigue_monitor import audit_fatigue


@pytest.mark.parametrize("variant", ["node_mlp_pool", "shared_preference"])
def test_network_variants_forward_and_reference_agree(config, fixed_instance, variant):
    settings = deepcopy(config["network"])
    settings["encoder_variant" if variant == "node_mlp_pool" else "actor_head_variant"] = variant
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    mask = env.get_action_mask()
    network = build_actor_critic(observation, settings)
    network.eval()
    with torch.no_grad():
        network.execution_mode = "phase_batched_v1"
        fast_logits, fast_value = network(observation, mask, device="cpu")
        network.execution_mode = "reference_v8"
        reference_logits, reference_value = network(observation, mask, device="cpu")
    torch.testing.assert_close(fast_logits, reference_logits)
    torch.testing.assert_close(fast_value, reference_value)
    assert_network_config_matches_spec(network.network_spec(), network.network_spec())
    with pytest.raises(ValueError):
        baseline = build_actor_critic(observation, config["network"])
        assert_network_config_matches_spec(baseline.network_spec(), network.network_spec())


def test_neutral_dynamics_hide_original_fatigue(config, fixed_instance):
    settings = deepcopy(config)
    settings["environment"]["fatigue_mode"] = "neutral"
    altered = replace(fixed_instance,
        workers=tuple(replace(worker, initial_fatigue=0.4) for worker in fixed_instance.workers),
        fatigue=replace(fixed_instance.fatigue,
            disassembly_accumulation_rate_per_minute=0.1,
            installation_accumulation_rate_per_minute=0.1))
    first = AssemblySchedulingEnv(settings)
    second = AssemblySchedulingEnv(settings)
    one = first.reset(fixed_instance)
    two = second.reset(altered)
    np.testing.assert_array_equal(first.get_action_mask(), second.get_action_mask())
    for name in one.node_features:
        np.testing.assert_array_equal(one.node_features[name], two.node_features[name])
    assert first.metrics()["fatigue_monitor_peak"] != second.metrics()["fatigue_monitor_peak"]
    assert first.metrics()["maximum_worker_fatigue"] == 0.0
    assert second.metrics()["maximum_worker_fatigue"] == 0.0


def test_node_mlp_encoder_does_not_read_graph_relation(config, fixed_instance):
    settings = {**config["network"], "encoder_variant": "node_mlp_pool"}
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    network = build_actor_critic(observation, settings)
    changed_relations = dict(observation.relations)
    original = changed_relations[OPERATION_ORDER_EDGE]
    changed_relations[OPERATION_ORDER_EDGE] = EdgeStore(
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_features=np.empty((0, original.edge_features.shape[1]), dtype=np.float32),
        feature_names=original.feature_names, bidirectional=original.bidirectional)
    altered = replace(observation, relations=changed_relations)
    with torch.no_grad():
        _, first_nodes, _, first_context = network.encode_graph([observation], device="cpu")
        _, second_nodes, _, second_context = network.encode_graph([altered], device="cpu")
    for name in first_nodes:
        torch.testing.assert_close(first_nodes[name], second_nodes[name])
    torch.testing.assert_close(first_context, second_context)


def test_fatigue_audit_integrates_threshold_crossing(fixed_instance):
    worker = replace(fixed_instance.workers[0], initial_fatigue=0.5)
    instance = replace(fixed_instance, workers=(worker,), fatigue=replace(fixed_instance.fatigue,
        maximum_safe_fatigue=0.75,
        disassembly_accumulation_rate_per_minute=0.1,
        idle_recovery_rate_per_minute=0.1))
    records = [{"worker_id": worker.id, "reconfiguration_id": "r1", "stage": "DIS",
                "start": 0.0, "end": 4.0}]
    metrics, segments = audit_fatigue(instance, records, 6.0)
    assert metrics["fatigue_monitor_peak"] == pytest.approx(0.9)
    assert metrics["fatigue_monitor_over_limit_minutes"] == pytest.approx(3.0)
    assert metrics["fatigue_monitor_over_limit_area"] == pytest.approx(0.225)
    assert segments[-1]["fatigue_end"] == pytest.approx(0.7)


def test_neutral_training_generator_keeps_full_benchmark_acceptance(config, fixed_instance):
    neutral = deepcopy(config)
    neutral["environment"]["fatigue_mode"] = "neutral"
    records = [InstanceGenerator(fixed_instance, settings["generator"], config=settings).generate(
        seed=1000000, split="train", pressure_type="fatigue_bottleneck",
        classify_reconfiguration_value=False)
        for settings in (config, neutral)]
    assert records[0].instance == records[1].instance
    assert records[0].metadata["generation_attempt"] == records[1].metadata["generation_attempt"]


def test_partial_worker_task_reward_identity_uses_committed_load(config, fixed_instance):
    settings = deepcopy(config)
    settings["environment"]["fatigue_mode"] = "neutral"
    settings["preference"]["quality"]["mode"] = "fixed"
    settings["preference"]["quality"]["fixed"] = [0.0, 0.0, 1.0]
    environment = AssemblySchedulingEnv(settings)
    environment.reset(fixed_instance)
    policy = HeuristicPolicy()
    for _ in range(500):
        if environment._active_committed_worker_tasks:
            break
        if environment.task_done:
            pytest.fail("no active worker task on the fixed instance")
        environment.step(policy.select_action(environment))
    metrics = environment.metrics()
    assert environment._active_committed_worker_tasks
    assert metrics["worker_load_variance"] != metrics["reward_objective_worker_load_variance"]
    expected = proxy_return_from_metrics(metrics, settings, preference=metrics["preference"])
    assert expected == pytest.approx(metrics["training_cumulative_reward"], abs=1e-8)
