from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from agent.baselines import HeuristicPolicy
from agent.ppo import (
    HeteroGraphActorCritic,
    PPOAgent,
    RolloutBuffer,
    TypedActorCritic,
    build_actor_critic,
    read_checkpoint_network_spec,
)
from agent.ppo.network import (
    BIDIRECTIONAL_EDGE_TYPES,
    HeterogeneousMessagePassingLayer,
)
from data.dataset import load_dataset_split
from environment import (
    ASSEMBLY_EDGE_TYPES,
    CAPABLE_EDGE,
    LOCKED_EDGE,
    AssemblySchedulingEnv,
    DecisionType,
    EdgeStore,
    HeterogeneousGraphObservation,
)
from utils import set_seed


def _small_gnn_config() -> dict[str, object]:
    return {
        "encoder_type": "hetero_gnn",
        "hidden_dim": 16,
        "message_passing_layers": 1,
        "dropout": 0.0,
    }


def _find_worker_observation(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    policy = HeuristicPolicy()
    for _ in range(200):
        if observation.decision_type == DecisionType.WORKER:
            mask = environment.get_action_mask()
            if np.any(~mask[:-1]):
                return observation, mask
        action = policy.select_action(environment)
        observation, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            break
    raise AssertionError("test instance did not reach a worker decision")


def test_network_factory_defaults_and_validation(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)

    network = build_actor_critic(observation, config["network"])
    assert isinstance(network, HeteroGraphActorCritic)
    assert network.network_spec() == {
        "encoder_type": "hetero_gnn",
        "hidden_dim": 128,
        "message_passing_layers": 2,
        "dropout": 0.0,
    }
    assert all(
        len(layer.relation_transforms) == len(ASSEMBLY_EDGE_TYPES)
        for layer in network.message_layers
    )

    legacy = build_actor_critic(observation, {"hidden_dim": 32})
    assert isinstance(legacy, TypedActorCritic)
    assert legacy.network_spec() == {
        "encoder_type": "typed_mlp",
        "hidden_dim": 32,
    }

    with pytest.raises(ValueError, match="encoder_type"):
        build_actor_critic(
            observation,
            {"encoder_type": "unknown", "hidden_dim": 16},
        )
    with pytest.raises(ValueError, match="message_passing_layers"):
        build_actor_critic(
            observation,
            {
                "encoder_type": "hetero_gnn",
                "hidden_dim": 16,
                "message_passing_layers": 0,
            },
        )
    with pytest.raises(ValueError, match="dropout"):
        build_actor_critic(
            observation,
            {
                "encoder_type": "hetero_gnn",
                "hidden_dim": 16,
                "message_passing_layers": 1,
                "dropout": 1.0,
            },
        )


def test_relation_layer_uses_edge_features_bidirectional_mean():
    edge_dimensions = {
        edge_type: 1 for edge_type in ASSEMBLY_EDGE_TYPES
    }
    layer = HeterogeneousMessagePassingLayer(
        hidden_dim=2,
        edge_feature_dimensions=edge_dimensions,
        dropout=0.0,
    )
    layer.layer_norms = nn.ModuleDict(
        {
            "operation": nn.Identity(),
            "machine": nn.Identity(),
            "worker": nn.Identity(),
        }
    )
    with torch.no_grad():
        for transform in layer.relation_transforms.values():
            transform.weight.zero_()
            transform.bias.zero_()
        capability_transform = layer.relation_transforms[
            "__".join(CAPABLE_EDGE)
        ]
        capability_transform.weight[0, 0] = 1.0
        capability_transform.weight[0, -1] = 1.0

    relations = {}
    for edge_type in ASSEMBLY_EDGE_TYPES:
        relations[edge_type] = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
            edge_type in BIDIRECTIONAL_EDGE_TYPES,
        )
    relations[CAPABLE_EDGE] = (
        torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        torch.tensor([[1.0], [3.0]], dtype=torch.float32),
        True,
    )
    updated = layer(
        {
            "operation": torch.tensor(
                [[1.0, 0.0], [3.0, 0.0]]
            ),
            "machine": torch.tensor([[2.0, 0.0]]),
            "worker": torch.tensor([[1.0, 0.0]]),
        },
        relations,
    )

    assert torch.allclose(
        updated["operation"],
        torch.tensor([[4.0, 0.0], [8.0, 0.0]]),
    )
    assert torch.allclose(
        updated["machine"],
        torch.tensor([[6.0, 0.0]]),
    )
    assert torch.allclose(
        updated["worker"],
        torch.tensor([[1.0, 0.0]]),
    )


def test_hetero_batch_matches_individual_and_masks(
    config,
    fixed_instance,
):
    first_environment = AssemblySchedulingEnv(config)
    first_observation = first_environment.reset(fixed_instance)
    first_mask = first_environment.get_action_mask()
    validation_instance = load_dataset_split(
        config, "validation"
    )[0].instance
    second_environment = AssemblySchedulingEnv(config)
    second_observation = second_environment.reset(validation_instance)
    second_mask = second_environment.get_action_mask()
    worker_observation, worker_mask = _find_worker_observation(
        config,
        fixed_instance,
    )
    observations = (
        first_observation,
        second_observation,
        worker_observation,
    )
    masks = (first_mask, second_mask, worker_mask)
    network = build_actor_critic(
        first_observation,
        _small_gnn_config(),
    )
    network.eval()

    individual = [
        network(observation, mask, device="cpu")
        for observation, mask in zip(observations, masks)
    ]
    batch_logits, batch_values = network.forward_batch(
        observations,
        masks,
        device="cpu",
    )
    minimum_logit = torch.finfo(batch_logits.dtype).min
    for index, ((logits, value), mask) in enumerate(
        zip(individual, masks)
    ):
        assert torch.allclose(
            logits,
            batch_logits[index, : len(mask)],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            value,
            batch_values[index],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.all(
            batch_logits[index, : len(mask)][
                np.asarray(mask, dtype=bool)
            ]
            == minimum_logit
        )
        assert torch.all(
            batch_logits[index, len(mask) :] == minimum_logit
        )

    selected_logits = []
    for index, mask in enumerate(masks):
        feasible_action = int(np.flatnonzero(~mask)[0])
        selected_logits.append(batch_logits[index, feasible_action])
    (batch_values.sum() + torch.stack(selected_logits).sum()).backward()
    relation_gradients = [
        parameter.grad
        for layer in network.message_layers
        for transform in layer.relation_transforms.values()
        for parameter in transform.parameters()
        if parameter.grad is not None
    ]
    assert relation_gradients
    assert all(torch.isfinite(gradient).all() for gradient in relation_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in relation_gradients)


def test_worker_head_requires_unique_locked_operation(
    config,
    fixed_instance,
):
    observation, mask = _find_worker_observation(config, fixed_instance)
    network = build_actor_critic(observation, _small_gnn_config())
    locked = observation.relations[LOCKED_EDGE]
    assert locked.num_edges >= 1

    missing_relations = dict(observation.relations)
    missing_relations[LOCKED_EDGE] = EdgeStore(
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_features=np.empty(
            (0, locked.edge_features.shape[1]),
            dtype=np.float32,
        ),
        feature_names=locked.feature_names,
        bidirectional=True,
    )
    missing = replace(observation, relations=missing_relations)
    with pytest.raises(ValueError, match="exactly one locked operation"):
        network(missing, mask, device="cpu")

    duplicated_relations = dict(observation.relations)
    duplicated_relations[LOCKED_EDGE] = EdgeStore(
        edge_index=np.repeat(locked.edge_index[:, :1], 2, axis=1),
        edge_features=np.repeat(
            locked.edge_features[:1],
            2,
            axis=0,
        ),
        feature_names=locked.feature_names,
        bidirectional=True,
    )
    duplicated = replace(observation, relations=duplicated_relations)
    with pytest.raises(ValueError, match="at most one locked operation"):
        network(duplicated, mask, device="cpu")


def test_graph_buffer_and_gnn_ppo_update(
    config,
    fixed_instance,
):
    effective_config = deepcopy(config)
    effective_config["network"] = _small_gnn_config()
    effective_config["ppo"]["epochs"] = 1
    effective_config["ppo"]["batch_size"] = 4
    set_seed(int(effective_config["seed"]))
    environment = AssemblySchedulingEnv(effective_config)
    observation = environment.reset(fixed_instance)
    network = build_actor_critic(
        observation,
        effective_config["network"],
    )
    agent = PPOAgent(network, effective_config["ppo"], device="cpu")
    buffer = RolloutBuffer(preserve_graph=True)
    first_observation = observation
    for _ in range(4):
        mask = environment.get_action_mask()
        action, log_probability, value = agent.act(observation, mask)
        next_observation, reward, terminated, truncated, _ = (
            environment.step(action)
        )
        buffer.add(
            observation,
            mask,
            action,
            log_probability,
            value,
            reward.scalarize(effective_config["reward"]),
            terminated or truncated,
        )
        observation = next_observation
        if terminated or truncated:
            break
    assert isinstance(
        buffer.transitions[0].observation,
        HeterogeneousGraphObservation,
    )
    original_edge_value = buffer.transitions[0].observation.relations[
        CAPABLE_EDGE
    ].edge_features[0, 0]
    first_observation.relations[CAPABLE_EDGE].edge_features[0, 0] = -99.0
    assert (
        buffer.transitions[0]
        .observation.relations[CAPABLE_EDGE]
        .edge_features[0, 0]
        == original_edge_value
    )
    buffer.compute_gae(
        last_value=0.0,
        gamma=float(effective_config["ppo"]["gamma"]),
        gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
    )
    before = [
        parameter.detach().clone() for parameter in network.parameters()
    ]
    metrics = agent.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, network.parameters())
    )


def test_new_and_legacy_checkpoint_compatibility(
    config,
    fixed_instance,
    tmp_path,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    gnn_config = _small_gnn_config()
    gnn_agent = PPOAgent(
        build_actor_critic(observation, gnn_config),
        config["ppo"],
        device="cpu",
    )
    gnn_checkpoint = tmp_path / "gnn.pt"
    gnn_agent.save(gnn_checkpoint, metadata={"kind": "gnn"})
    assert read_checkpoint_network_spec(gnn_checkpoint) == gnn_config
    gnn_clone = PPOAgent(
        build_actor_critic(observation, gnn_config),
        config["ppo"],
        device="cpu",
    )
    assert gnn_clone.load(gnn_checkpoint) == {"kind": "gnn"}

    typed_network = TypedActorCritic(
        observation.feature_dimensions,
        hidden_dim=16,
    )
    typed_agent = PPOAgent(
        typed_network,
        config["ppo"],
        device="cpu",
    )
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "network": typed_network.state_dict(),
            "optimizer": typed_agent.optimizer.state_dict(),
            "ppo_config": config["ppo"],
            "metadata": {"kind": "legacy"},
        },
        legacy_checkpoint,
    )
    assert read_checkpoint_network_spec(legacy_checkpoint) == {
        "encoder_type": "typed_mlp",
        "hidden_dim": 16,
    }
    typed_clone = PPOAgent(
        TypedActorCritic(observation.feature_dimensions, 16),
        config["ppo"],
        device="cpu",
    )
    assert typed_clone.load(legacy_checkpoint) == {"kind": "legacy"}
    with pytest.raises(ValueError, match="does not match"):
        gnn_clone.load(legacy_checkpoint)
