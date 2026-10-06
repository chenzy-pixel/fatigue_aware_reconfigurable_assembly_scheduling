"""Message semantics, resource-pair expression, and computation identity."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.network import HeterogeneousMessagePassingLayer, NODE_TYPES, normalize_network_config
from configs import load_config, runtime_manifest
from configs.config import public_config
from configs.network_contract import MESSAGE_IDENTITY_FIELDS, message_identity
from environment import ASSEMBLY_EDGE_TYPES, CAPABLE_EDGE, PROCESSING_ON_EDGE, AssemblySchedulingEnv
from result.io import write_json
from scripts.audit_schema9_followup import attributed_edge_instances
from scripts.benchmark_joint_messages import _linear_reference


def _empty_relations(dimensions):
    return {kind: (torch.empty((2, 0), dtype=torch.long), torch.empty((0, width), dtype=torch.float64), False)
            for kind, width in dimensions.items()}


def test_attributed_relu_bidirectional_zero_attribute_and_total_mean():
    dimensions = dict.fromkeys(ASSEMBLY_EDGE_TYPES, 0)
    dimensions[CAPABLE_EDGE] = 1
    layer = HeterogeneousMessagePassingLayer(2, dimensions, 0).double()
    layer.norms = nn.ModuleDict({kind: nn.Identity() for kind in NODE_TYPES})
    with torch.no_grad():
        for transform in layer.transforms.values():
            transform.weight.zero_()
            transform.bias.zero_()
        attributed = layer.transforms["__".join(CAPABLE_EDGE)]
        attributed.weight[0, 0], attributed.weight[0, 2] = 1, -1
        attributed.bias[1] = 6
        layer.transforms["__".join(PROCESSING_ON_EDGE)].weight[0, 0] = -2
    nodes = {kind: torch.zeros((1, 2), dtype=torch.float64) for kind in NODE_TYPES}
    nodes["operation"] = torch.tensor([[2., 0.], [4., 0.]], dtype=torch.float64)
    nodes["machine"] = nodes["operation"].clone()
    relations = _empty_relations(dimensions)
    relations[CAPABLE_EDGE] = (torch.tensor([[0, 1], [0, 0]]), torch.tensor([[3.], [1.]], dtype=torch.float64), True)
    relations[PROCESSING_ON_EDGE] = (torch.tensor([[1], [0]]), torch.empty((1, 0), dtype=torch.float64), True)
    output = layer(nodes, relations)
    # Attributed messages to M0: (0,6), (3,6). Actual-edge message: (-8,0).
    # Three total incoming messages; the attribute-free negative message survives.
    torch.testing.assert_close(output["machine"], torch.tensor([[1/3, 4.], [4., 0.]], dtype=torch.float64))
    torch.testing.assert_close(output["operation"], torch.tensor([[2., 6.], [2.5, 3.]], dtype=torch.float64))


def test_empty_relations_preserve_residual_update():
    dimensions = dict.fromkeys(ASSEMBLY_EDGE_TYPES, 0)
    dimensions[CAPABLE_EDGE] = 1
    layer = HeterogeneousMessagePassingLayer(4, dimensions, 0).double()
    nodes = {kind: torch.randn((2, 4), dtype=torch.float64) for kind in NODE_TYPES}
    nodes["wave"] = torch.empty((0, 4), dtype=torch.float64)
    actual = layer(nodes, _empty_relations(dimensions))
    for kind, values in nodes.items():
        torch.testing.assert_close(actual[kind], torch.relu(layer.norms[kind](values)), atol=0, rtol=0)


def _double_context(model, observations):
    batch = model._collate_graphs(observations, device="cpu")
    nodes = {kind: model.node_projectors[kind](features.double()) for kind, features in batch.node_features.items()}
    relations = {kind: (index, features.double(), bidirectional) for kind, (index, features, bidirectional) in batch.relations.items()}
    for layer in model.message_layers:
        nodes = layer(nodes, relations)
    context = torch.cat(tuple(model._pool_slices(nodes[kind], batch.node_slices[kind]) for kind in NODE_TYPES)
                        + (model.global_encoder(batch.global_features.double()),), -1)
    preference = model.preference_encoder(torch.as_tensor(np.stack([o.preference for o in observations]), dtype=torch.float64))
    return context, model._critic_values(context, preference)


@pytest.mark.parametrize("seed", (0, 1, 11, 23, 37))
def test_legal_resource_pair_witness_is_distinguishable(config, fixed_instance, seed):
    observations = [AssemblySchedulingEnv(config).reset(instance, preference=(1, 0, 0))
                    for instance in attributed_edge_instances(fixed_instance)]
    for kind in NODE_TYPES:
        np.testing.assert_array_equal(observations[0].node_features[kind], observations[1].node_features[kind])
    torch.manual_seed(seed)
    network = build_actor_critic(observations[0], config["network"]).double().eval()
    reference = _linear_reference(network)
    assert tuple(network.state_dict()) == tuple(reference.state_dict())
    assert sum(p.numel() for p in network.parameters()) == 1197796
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
    with torch.no_grad():
        context, values = _double_context(network, observations)
        previous_context, previous_values = _double_context(reference, observations)
    assert float((previous_context[0]-previous_context[1]).abs().max()) < 1e-12
    assert float((previous_values[0]-previous_values[1]).abs()) < 1e-12
    assert float((context[0]-context[1]).abs().max()) > 1e-6
    assert float((values[0]-values[1]).abs()) > 1e-8


def test_node_reordering_is_equivariant_and_critic_is_invariant(config, fixed_instance):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    permutations = {kind: np.arange(len(values))[::-1].copy() for kind, values in observation.node_features.items()}
    inverse = {kind: np.argsort(indices) for kind, indices in permutations.items()}
    relations = {}
    for kind, store in observation.relations.items():
        source, _, target = kind
        index = np.stack((inverse[source][store.edge_index[0]], inverse[target][store.edge_index[1]]))
        order = np.lexsort((index[1], index[0]))
        relations[kind] = replace(store, edge_index=index[:, order], edge_features=store.edge_features[order])
    reordered = replace(observation,
        node_features={kind: values[permutations[kind]] for kind, values in observation.node_features.items()},
        node_ids={kind: tuple(ids[i] for i in permutations[kind]) for kind, ids in observation.node_ids.items()},
        relations=relations)
    reordered.validate()
    network = build_actor_critic(observation, {**config["network"], "hidden_dim": 16}).eval()
    with torch.no_grad():
        batch, nodes, _, context = network.encode_graph([observation, reordered], device="cpu")
        for kind in NODE_TYPES:
            start, end = batch.node_slices[kind][0]
            other_start, other_end = batch.node_slices[kind][1]
            torch.testing.assert_close(nodes[kind][other_start:other_end], nodes[kind][start:end][permutations[kind]], atol=1e-6, rtol=1e-5)
        preference = network.preference_encoder(torch.as_tensor(np.stack([observation.preference]*2)))
        values = network._critic_values(context, preference)
    torch.testing.assert_close(context[0], context[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", ("hetero_gnn", "node_mlp_pool"))
def test_generated_identity_is_consistent(config, fixed_instance, variant):
    settings = {**config["network"], "encoder_variant": variant}
    expected = message_identity(variant)
    normalized = normalize_network_config(settings)
    network = build_actor_critic(AssemblySchedulingEnv(config).reset(fixed_instance), settings)
    effective = deepcopy(config)
    effective["network"] = settings
    for actual in (normalized, network.network_spec(), runtime_manifest(effective)):
        assert {field: actual[field] for field in MESSAGE_IDENTITY_FIELDS} == expected
    for field in MESSAGE_IDENTITY_FIELDS:
        with pytest.raises(ValueError, match=field):
            normalize_network_config({**settings, field: "linear_v1"})
        with pytest.raises(ValueError, match=field):
            runtime_manifest({**effective, "network": {**settings, field: "linear_v1"}})


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("field", MESSAGE_IDENTITY_FIELDS)
@pytest.mark.parametrize("corruption", ("missing", "wrong"))
@pytest.mark.parametrize("allow_migration", (False, True))
def test_incompatible_identity_rejected_before_network_and_optimizer_mutation(
    config, fixed_instance, tmp_path, field, corruption, allow_migration,
):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    settings = {**config["network"], "hidden_dim": 16}
    source = PPOAgent(build_actor_critic(observation, settings), config["ppo"])
    destination = PPOAgent(build_actor_critic(observation, settings), config["ppo"])
    for agent in (source, destination):
        agent.optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in agent.network.parameters()).backward()
        agent.optimizer.step()
    path = tmp_path / "checkpoint.pt"
    source.save(path)
    payload = torch.load(path, weights_only=False)
    if corruption == "missing":
        payload["network_spec"].pop(field)
    else:
        payload["network_spec"][field] = "linear_v1"
    torch.save(payload, path)
    weights_before = deepcopy(destination.network.state_dict())
    optimizer_before = deepcopy(destination.optimizer.state_dict())
    with pytest.raises(ValueError, match=field):
        destination.load(path, load_optimizer=True, allow_observation_migration=allow_migration)
    _assert_nested_equal(destination.network.state_dict(), weights_before)
    _assert_nested_equal(destination.optimizer.state_dict(), optimizer_before)


def test_checkpoint_round_trip_restores_optimizer_and_outputs(config, fixed_instance, tmp_path):
    env = AssemblySchedulingEnv(config)
    observation, mask = env.reset(fixed_instance), env.get_action_mask()
    settings = {**config["network"], "hidden_dim": 16}
    source = PPOAgent(build_actor_critic(observation, settings), config["ppo"])
    source.optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in source.network.parameters()).backward()
    source.optimizer.step()
    path = tmp_path / "checkpoint.pt"
    source.save(path, metadata={"runtime_manifest": runtime_manifest(config)})
    clone = PPOAgent(build_actor_critic(observation, settings), config["ppo"])
    clone.load(path, load_optimizer=True)
    _assert_nested_equal(clone.optimizer.state_dict(), source.optimizer.state_dict())
    with torch.no_grad():
        expected = source.network(observation, mask, device="cpu")
        actual = clone.network(observation, mask, device="cpu")
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_snapshot_requires_current_generated_runtime_identity(config, tmp_path):
    path = tmp_path / "config.json"
    current = public_config(config)
    write_json(path, current)
    assert load_config(path)["runtime_manifest"] == current["runtime_manifest"]
    for field in MESSAGE_IDENTITY_FIELDS:
        for corruption in ("missing", "wrong"):
            old = deepcopy(current)
            if corruption == "missing":
                old["runtime_manifest"].pop(field)
            else:
                old["runtime_manifest"][field] = "linear_v1"
            write_json(path, old)
            with pytest.raises(ValueError, match="runtime_manifest"):
                load_config(path)
