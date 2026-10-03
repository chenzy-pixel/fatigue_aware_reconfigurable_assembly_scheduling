from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.network import (
    HeteroGraphActorCritic,
    SharedPreferenceHead,
    infer_checkpoint_network_spec,
    network_requires_graph_observation,
    normalize_network_config,
)
from environment import AssemblySchedulingEnv, CAPABLE_EDGE, DecisionType, SERVICE_CANDIDATE_EDGE


VARIANTS = tuple(product(
    ("hetero_gnn", "node_mlp_pool"), ("objective_experts", "shared_preference")
))


@pytest.fixture
def mixed_batch(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    production = environment.reset(fixed_instance)
    production_mask = environment.get_action_mask()
    small_environment = AssemblySchedulingEnv(config)
    small = small_environment.reset(replace(fixed_instance, orders=fixed_instance.orders[:1]))
    small_mask = small_environment.get_action_mask()
    policy = HeuristicPolicy()
    worker = production
    for _ in range(100):
        if worker.decision_type == DecisionType.WORKER:
            break
        worker, _, terminated, truncated, _ = environment.step(policy.select_action(environment))
        if terminated or truncated:
            pytest.fail("test instance must reach a worker decision")
    else:
        pytest.fail("test instance must reach a worker decision")
    worker_mask = environment.get_action_mask()
    assert np.any(~worker_mask[:-1])
    worker_wait = np.ones_like(worker_mask)
    worker_wait[-1] = False
    production_wait = np.ones_like(production_mask)
    production_wait[-1] = False
    worker_pair = np.ones_like(worker_mask)
    worker_pair[np.flatnonzero(~worker_mask[:-1])[0]] = False
    observations = [
        replace(production, preference=np.asarray([1.0, 0.0, 0.0])),
        replace(small, preference=np.asarray([0.0, 1.0, 0.0])),
        replace(worker, preference=np.asarray([0.0, 0.0, 1.0])),
        worker, production, worker,
    ]
    masks = [production_mask, small_mask, worker_mask, worker_wait, production_wait, worker_pair]
    assert len(masks[0]) != len(masks[1])
    return observations, masks


def _settings(config, encoder_variant, actor_head_variant):
    return dict(
        config["network"], hidden_dim=16, message_passing_layers=2, dropout=0.0,
        encoder_variant=encoder_variant, actor_head_variant=actor_head_variant,
    )


def _loss(logits, values, masks, device="cpu"):
    legal_logits = torch.cat([
        logits[index, :len(mask)][~torch.as_tensor(mask, device=device)]
        for index, mask in enumerate(masks)
    ])
    return values.square().sum() + legal_logits.square().sum()


@pytest.mark.parametrize("encoder_variant,actor_head_variant", VARIANTS)
def test_variant_spec_and_structure(config, mixed_batch, encoder_variant, actor_head_variant):
    observations, _ = mixed_batch
    settings = _settings(config, encoder_variant, actor_head_variant)
    network = build_actor_critic(observations[0], settings)
    spec = infer_checkpoint_network_spec({"network_spec": network.network_spec()})
    assert spec["encoder_type"] == spec["encoder_variant"] == encoder_variant
    assert spec["actor_head_variant"] == actor_head_variant
    assert spec["observation_schema_version"] == 6
    assert spec["normalization_manifest_sha256"] == config["network"]["normalization_manifest_sha256"]
    assert network_requires_graph_observation(settings)
    assert len(network.message_layers) == (2 if encoder_variant == "hetero_gnn" else 0)
    assert len(network.node_mlp_layers) == (2 if encoder_variant == "node_mlp_pool" else 0)
    shared_heads = [module for module in network.modules() if isinstance(module, SharedPreferenceHead)]
    assert len(shared_heads) == (4 if actor_head_variant == "shared_preference" else 0)
    expected_parameterization = (
        "simplex_softplus_v8" if actor_head_variant == "objective_experts"
        else "shared_preference_mlp_v1"
    )
    assert spec["expert_weight_parameterization"] == expected_parameterization


@pytest.mark.parametrize("name", ("encoder_variant", "actor_head_variant"))
def test_unknown_structural_variant_is_rejected(name):
    with pytest.raises(ValueError, match=name):
        normalize_network_config({name: "unknown"})


def test_default_variant_initialization_and_legacy_checkpoint(config, mixed_batch, tmp_path):
    observations, masks = mixed_batch
    implicit = _settings(config, "hetero_gnn", "objective_experts")
    implicit.pop("encoder_variant")
    implicit.pop("actor_head_variant")
    state = torch.random.get_rng_state()
    original = build_actor_critic(observations[0], implicit)
    torch.random.set_rng_state(state)
    explicit = build_actor_critic(observations[0], _settings(config, "hetero_gnn", "objective_experts"))
    assert tuple(original.state_dict()) == tuple(explicit.state_dict())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
    source_agent = PPOAgent(original, config["ppo"], device="cpu")
    path = tmp_path / "legacy_variants.pt"
    source_agent.save(path)
    checkpoint = torch.load(path, weights_only=False)
    for name in ("encoder_variant", "actor_head_variant"):
        checkpoint["network_spec"].pop(name)
    torch.save(checkpoint, path)
    saved = infer_checkpoint_network_spec(checkpoint)
    assert saved["encoder_variant"] == "hetero_gnn"
    assert saved["actor_head_variant"] == "objective_experts"
    clone_agent = PPOAgent(explicit, config["ppo"], device="cpu")
    clone_agent.load(path, load_optimizer=True)
    with torch.no_grad():
        expected = original.forward_batch(observations, masks, device="cpu")
        actual = explicit.forward_batch(observations, masks, device="cpu")
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ("reference_v8", "phase_batched_v1"))
def test_node_mlp_encoding_and_critic_are_independent_of_relation_topology(config, mixed_batch, mode):
    observations, masks = mixed_batch
    original = observations[0]
    relations = {
        edge: replace(store, edge_index=np.empty((2, 0), dtype=np.int64),
                      edge_features=np.empty((0, store.edge_features.shape[1]), dtype=np.float32))
        for edge, store in original.relations.items()
    }
    rewired = replace(original, relations=relations)
    rewired.validate()
    network = build_actor_critic(original, _settings(config, "node_mlp_pool", "objective_experts"))
    network.eval()
    network.execution_mode = mode
    with torch.no_grad():
        batch, expected_nodes, expected_global, expected_context = network.encode_graph([original], device="cpu")
        _, actual_nodes, actual_global, actual_context = network.encode_graph([rewired], device="cpu")
        expected_value = network.value_batch([original], [masks[0]], device="cpu")
        actual_value = network.value_batch([rewired], [masks[0]], device="cpu")
    for name, nodes in expected_nodes.items():
        torch.testing.assert_close(actual_nodes[name], nodes, atol=0, rtol=0)
        start, end = batch.node_slices[name][0]
        pooled = nodes[start:end].mean(0) if end > start else nodes.new_zeros(network.hidden_dim)
        offset = list(expected_nodes).index(name) * network.hidden_dim
        torch.testing.assert_close(expected_context[0, offset:offset + network.hidden_dim], pooled)
    torch.testing.assert_close(actual_global, expected_global, atol=0, rtol=0)
    torch.testing.assert_close(actual_context, expected_context, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)
    graph_network = build_actor_critic(original, _settings(config, "hetero_gnn", "objective_experts"))
    graph_network.eval()
    graph_network.execution_mode = mode
    with torch.no_grad():
        _, graph_nodes, _, _ = graph_network.encode_graph([original], device="cpu")
        _, rewired_nodes, _, _ = graph_network.encode_graph([rewired], device="cpu")
    assert any(not torch.allclose(graph_nodes[name], rewired_nodes[name]) for name in graph_nodes)


@pytest.mark.parametrize("phase,wait", tuple(product((DecisionType.PRODUCTION, DecisionType.WORKER), (False, True))))
def test_shared_scorer_conditions_action_scores_on_preference(config, mixed_batch, phase, wait):
    observations, _ = mixed_batch
    network = build_actor_critic(observations[0], _settings(config, "hetero_gnn", "shared_preference"))
    prefix = "production" if phase == DecisionType.PRODUCTION else "worker"
    head = getattr(network, prefix + ("_wait" if wait else "") + "_shared")
    with torch.no_grad():
        for parameter in network.preference_encoder.parameters():
            parameter.zero_()
        network.preference_encoder[0].weight[0, 0] = 1.0
        network.preference_encoder[0].bias[0] = 0.1
        network.preference_encoder[2].weight[0, 0] = 1.0
        for parameter in head.parameters():
            parameter.zero_()
        head.score[0].weight[0, 0] = 1.0
        head.score[0].weight[0, network.hidden_dim] = 1.0
        head.score[0].bias[0] = -0.5
        head.score[2].weight[0, 0] = 1.0
    actions = torch.zeros(2, network.hidden_dim)
    actions[:, 0] = torch.tensor([0.25, 0.75])
    direct = {objective: torch.zeros(2, len(fields)) for objective, fields in head.schema.items()}
    scores = []
    for weights in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        preference = torch.tensor([weights, weights], requires_grad=True)
        preference_embedding = network.preference_encoder(preference)
        _, _, _, base = network._score_head(actions, direct, preference, preference_embedding, phase, wait=wait)
        base.sum().backward()
        assert torch.count_nonzero(preference.grad) > 0
        scores.append(base.detach())
    assert not torch.allclose(scores[0], scores[1])
    assert not torch.allclose(scores[0].diff(), scores[1].diff())
    assert head.score[-1].out_features == 1
    assert network.effective_relative_cost_weights() == {}


@pytest.mark.parametrize("encoder_variant,actor_head_variant", VARIANTS)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_structural_variants_match_reference_logits_values_and_gradients(config, mixed_batch, encoder_variant, actor_head_variant, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    observations, masks = mixed_batch
    network = build_actor_critic(observations[0], _settings(config, encoder_variant, actor_head_variant)).to(device)
    network.eval()
    if actor_head_variant == "objective_experts":
        with torch.no_grad():
            for prefix in ("production", "worker", "production_wait", "worker_wait"):
                for expert in getattr(network, prefix + "_experts").experts.values():
                    expert.context[-1].weight.uniform_(-0.05, 0.05)
                    expert.context[-1].bias.uniform_(-0.05, 0.05)
    results = []
    for mode in ("reference_v8", "phase_batched_v1"):
        network.execution_mode = mode
        network.zero_grad(set_to_none=True)
        logits, values = network.forward_batch(observations, masks, device=device)
        _loss(logits, values, masks, device).backward()
        gradients = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in network.named_parameters()
        }
        with torch.no_grad():
            bootstrap = network.value_batch(observations, masks, device=device)
            network.forward_batch(observations, masks, device=device)
            diagnostics = network.consume_policy_decision_diagnostics()
        torch.testing.assert_close(bootstrap, values.detach(), atol=1e-6, rtol=1e-5)
        assert len(diagnostics) == len(observations)
        assert [row["decision_type"] for row in diagnostics] == [item.decision_type.value for item in observations]
        assert all(isinstance(value, (str, int, float, bool)) for row in diagnostics for value in row.values())
        if actor_head_variant == "shared_preference":
            assert all(not name.startswith(("direct_", "expert_", "contribution_")) for row in diagnostics for name in row)
            assert set(network.policy_head_diagnostics()) == {
                "policy_head_gate_production_residual", "policy_head_gate_worker_residual",
            }
        results.append((logits.detach(), values.detach(), gradients, diagnostics))
    expected_logits, expected_values, expected_gradients, expected_diagnostics = results[0]
    actual_logits, actual_values, actual_gradients, actual_diagnostics = results[1]
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual_values, expected_values, atol=1e-5, rtol=1e-4)
    for name, expected in expected_gradients.items():
        actual = actual_gradients[name]
        assert (actual is None) == (expected is None), name
        if actual is not None:
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4, msg=name)
    for actual, expected in zip(actual_diagnostics, expected_diagnostics, strict=True):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            if isinstance(value, float):
                assert actual[name] == pytest.approx(value, abs=2e-5, rel=1e-4)
            else:
                assert actual[name] == value


@pytest.mark.parametrize("encoder_variant,actor_head_variant", VARIANTS)
@pytest.mark.parametrize("mismatched_field", ("encoder_variant", "actor_head_variant"))
def test_checkpoint_structural_variant_mismatch_is_rejected(config, mixed_batch, tmp_path, encoder_variant, actor_head_variant, mismatched_field):
    observations, _ = mixed_batch
    settings = _settings(config, encoder_variant, actor_head_variant)
    source = PPOAgent(build_actor_critic(observations[0], settings), config["ppo"], device="cpu")
    path = tmp_path / "source.pt"
    source.save(path)
    choices = ("hetero_gnn", "node_mlp_pool") if mismatched_field == "encoder_variant" else ("objective_experts", "shared_preference")
    settings[mismatched_field] = next(value for value in choices if value != settings[mismatched_field])
    target = PPOAgent(build_actor_critic(observations[0], settings), config["ppo"], device="cpu")
    before = deepcopy(target.network.state_dict())
    with pytest.raises(ValueError, match=mismatched_field):
        target.load(path, load_optimizer=True)
    for name, value in before.items():
        torch.testing.assert_close(target.network.state_dict()[name], value, atol=0, rtol=0)


def _schema5_observation(observation):
    nodes = dict(observation.node_features)
    nodes["order"] = nodes["order"][:, :-1]
    names = dict(observation.node_feature_names)
    names["order"] = names["order"][:-1]
    relations = dict(observation.relations)
    for edge, count in ((CAPABLE_EDGE, 1), (SERVICE_CANDIDATE_EDGE, 2)):
        store = relations[edge]
        relations[edge] = replace(store, edge_features=store.edge_features[:, :-count],
                                  feature_names=store.feature_names[:-count])
    return replace(observation, node_features=nodes, node_feature_names=names, relations=relations,
                   action_set_features=observation.action_set_features[:-2],
                   action_set_feature_names=observation.action_set_feature_names[:-2])


@pytest.mark.parametrize("encoder_variant,actor_head_variant", VARIANTS)
def test_schema5_migration_preserves_variant_predictions_and_adam_state(config, mixed_batch, tmp_path, encoder_variant, actor_head_variant):
    batch, batch_masks = mixed_batch
    observations, masks = [batch[0], batch[2]], [batch_masks[0], batch_masks[2]]
    legacy = [_schema5_observation(item) for item in observations]
    settings = _settings(config, encoder_variant, actor_head_variant)
    normalized = normalize_network_config(settings)
    construction = {name: normalized[name] for name in (
        "hidden_dim", "message_passing_layers", "dropout", "residual_gate_initial_logit",
        "residual_std_floor", "normalization_manifest_sha256", "worker_flow_time_normalization",
        "worker_flow_time_std_floor", "encoder_variant", "actor_head_variant",
    )}
    old_network = HeteroGraphActorCritic(
        legacy[0].feature_dimensions, legacy[0].edge_feature_dimensions,
        legacy[0].action_set_feature_names, **construction,
    )
    old_agent = PPOAgent(old_network, config["ppo"], device="cpu")
    logits, values = old_network.forward_batch(legacy, masks, device="cpu")
    _loss(logits, values, masks).backward()
    old_agent.optimizer.step()
    path = tmp_path / "schema5.pt"
    old_agent.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["network_spec"]["observation_schema_version"] = 5
    checkpoint["network_spec"].pop("time_context_version")
    checkpoint["network_spec"].pop("time_context_feature_schema")
    if (encoder_variant, actor_head_variant) == ("hetero_gnn", "objective_experts"):
        checkpoint["network_spec"].pop("encoder_variant")
        checkpoint["network_spec"].pop("actor_head_variant")
    torch.save(checkpoint, path)
    target = PPOAgent(build_actor_critic(observations[0], settings), config["ppo"], device="cpu")
    metadata = target.load(path, load_optimizer=True)
    assert metadata["checkpoint_load_migration"]["target_observation_schema"] == 6
    assert metadata["source_network_weights_sha256"] == checkpoint["metadata"]["network_weights_sha256"]
    assert target.network.network_spec()["encoder_variant"] == encoder_variant
    assert target.network.network_spec()["actor_head_variant"] == actor_head_variant
    with torch.no_grad():
        expected = old_network.forward_batch(legacy, masks, device="cpu")
        actual = target.network.forward_batch(observations, masks, device="cpu")
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
    assert target.optimizer.state
    for parameter, state in target.optimizer.state.items():
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape
    target.optimizer.zero_grad(set_to_none=True)
    logits, values = target.network.forward_batch(observations, masks, device="cpu")
    _loss(logits, values, masks).backward()
    target.optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in target.network.parameters())
