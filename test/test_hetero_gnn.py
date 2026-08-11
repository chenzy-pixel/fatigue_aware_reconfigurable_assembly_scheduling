from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

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
    SERVICE_CANDIDATE_EDGE,
    AssemblySchedulingEnv,
    DecisionType,
    EdgeStore,
    HeterogeneousGraphObservation,
)
from utils import set_seed
from m1_experiments import _state_loss


def _small_gnn_config() -> dict[str, object]:
    return {
        "encoder_type": "hetero_gnn",
        "hidden_dim": 16,
        "message_passing_layers": 1,
        "dropout": 0.0,
    }


def _small_m1_config() -> dict[str, object]:
    return {
        **_small_gnn_config(),
        "production_action_edge_features": True,
        "worker_action_edge_features": True,
        "production_candidate_relative_features": True,
        "worker_candidate_relative_features": True,
    }


def _find_worker_observation(config, instance, minimum_pair_actions=1):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    policy = HeuristicPolicy()
    for _ in range(200):
        if observation.decision_type == DecisionType.WORKER:
            mask = environment.get_action_mask()
            if int(np.count_nonzero(~mask[:-1])) >= minimum_pair_actions:
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
        "production_action_edge_features": True,
        "worker_action_edge_features": True,
        "production_candidate_relative_features": True,
        "worker_candidate_relative_features": True,
        "policy_head_version": 5,
        "production_action_semantics": "pair_plus_defer_v1",
        "production_relative_feature_names": (
            "processing_plus_reconfiguration_time_norm",
        ),
        "worker_relative_feature_names": (
            "projected_fatigue_ratio",
            "incremental_load_variance_norm",
        ),
        "candidate_context_mode": "common_offset_v4",
        "worker_relative_weight_sharing": "shared_mean_v4",
        "observation_schema_version": 3,
        "feature_dimensions": observation.feature_dimensions,
        "edge_feature_dimensions": observation.edge_feature_dimensions,
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
        "policy_head_version": 5,
        "production_action_semantics": "pair_plus_defer_v1",
        "observation_schema_version": 3,
        "feature_dimensions": {
            name: observation.feature_dimensions[name]
            for name in ("operation", "machine", "worker", "global")
        },
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


def test_action_edge_rows_align_with_flat_actions_and_change_logits(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    production_mask = environment.get_action_mask()
    machine_count = len(environment.machines)
    capable = observation.relations[CAPABLE_EDGE]
    capable_actions = (
        capable.edge_index[0] * machine_count + capable.edge_index[1]
    )
    assert np.all(np.diff(capable_actions) > 0)
    for row, action in enumerate(capable_actions):
        assert environment.decode_production_action(int(action)) == tuple(
            int(value) for value in capable.edge_index[:, row]
        )

    network = build_actor_critic(observation, _small_m1_config())
    network.eval()
    legal_action = next(
        int(action)
        for action in capable_actions
        if not production_mask[int(action)]
    )
    row = int(np.flatnonzero(capable_actions == legal_action)[0])
    changed = observation.copy()
    changed.relations[CAPABLE_EDGE].edge_features[row, 0] += 0.5
    original_logits, _ = network(observation, production_mask, device="cpu")
    changed_logits, _ = network(changed, production_mask, device="cpu")
    assert not torch.isclose(
        original_logits[legal_action], changed_logits[legal_action]
    )

    worker_observation, worker_mask = _find_worker_observation(
        config, fixed_instance
    )
    service = worker_observation.relations[SERVICE_CANDIDATE_EDGE]
    worker_count = worker_observation.workers.shape[0]
    service_actions = (
        service.edge_index[0] * worker_count + service.edge_index[1]
    )
    assert np.all(np.diff(service_actions) > 0)
    legal_worker_action = next(
        int(action)
        for action in service_actions
        if not worker_mask[int(action)]
    )
    service_row = int(
        np.flatnonzero(service_actions == legal_worker_action)[0]
    )
    changed_worker = worker_observation.copy()
    changed_worker.relations[SERVICE_CANDIDATE_EDGE].edge_features[
        service_row, 3
    ] += 0.25
    worker_logits, _ = network(
        worker_observation, worker_mask, device="cpu"
    )
    changed_worker_logits, _ = network(
        changed_worker, worker_mask, device="cpu"
    )
    assert not torch.isclose(
        worker_logits[legal_worker_action],
        changed_worker_logits[legal_worker_action],
    )


@pytest.mark.parametrize("seed", [1, 11, 101, 1009])
def test_relative_rankers_are_monotone_for_every_seed(
    config,
    fixed_instance,
    seed,
):
    set_seed(seed)
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    network = build_actor_critic(observation, _small_m1_config())
    weights = network.effective_relative_cost_weights()
    assert weights["production"] == pytest.approx((-1.0,))
    assert weights["worker"] == pytest.approx((-1.0, -1.0))
    assert all(value < 0.0 for values in weights.values() for value in values)

    with torch.no_grad():
        network.production_relative_ranker.weight.add_(100.0)
        network.worker_relative_ranker.weight[0, 0].add_(100.0)
        network.worker_relative_ranker.weight[0, 1].sub_(100.0)
    updated = network.effective_relative_cost_weights()
    assert all(
        value < 0.0 for values in updated.values() for value in values
    )
    assert updated["worker"][0] == pytest.approx(updated["worker"][1])


@pytest.mark.parametrize("seed", [1, 11, 101, 1009])
def test_worker_relative_ranker_prefers_lower_variance_at_equal_fatigue(
    config,
    fixed_instance,
    seed,
):
    set_seed(seed)
    observation, mask = _find_worker_observation(
        config,
        fixed_instance,
        minimum_pair_actions=2,
    )
    network = build_actor_critic(observation, _small_m1_config())
    with torch.no_grad():
        for parameter in network.worker_scorer.parameters():
            parameter.normal_(mean=0.0, std=10.0)
        network.worker_context_gate.fill_(10.0)
    service = observation.relations[SERVICE_CANDIDATE_EDGE]
    names = service.feature_names
    fatigue_column = names.index("projected_fatigue_ratio")
    variance_column = names.index("incremental_load_variance_norm")
    worker_count = observation.node_features["worker"].shape[0]
    service_actions = (
        service.edge_index[0] * worker_count + service.edge_index[1]
    )
    legal_actions = [
        int(action) for action in np.flatnonzero(~mask[:-1])
    ]
    first_action, second_action = legal_actions[:2]
    controlled = observation.copy()
    for action in legal_actions:
        row = int(np.flatnonzero(service_actions == action)[0])
        controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
            row, fatigue_column
        ] = 0.5
        controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
            row, variance_column
        ] = 0.0
    first_row = int(np.flatnonzero(service_actions == first_action)[0])
    second_row = int(np.flatnonzero(service_actions == second_action)[0])
    controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
        first_row, variance_column
    ] = -0.25
    controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
        second_row, variance_column
    ] = 0.25
    logits, _ = network(controlled, mask, device="cpu")
    assert logits[first_action] > logits[second_action]


@pytest.mark.parametrize("seed", [1, 11, 101, 1009])
def test_production_relative_ranker_prefers_shorter_duration_for_every_seed(
    config,
    fixed_instance,
    seed,
):
    set_seed(seed)
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    legal_actions = [
        int(action) for action in np.flatnonzero(~mask[:-1])
    ]
    assert len(legal_actions) >= 2
    first_action, second_action = legal_actions[:2]
    network = build_actor_critic(observation, _small_m1_config())
    with torch.no_grad():
        for parameter in network.production_scorer.parameters():
            parameter.normal_(mean=0.0, std=10.0)
        network.production_context_gate.fill_(10.0)
    capability = observation.relations[CAPABLE_EDGE]
    names = capability.feature_names
    processing_column = names.index("processing_time_norm")
    reconfiguration_column = names.index("reconfiguration_time_norm")
    machine_count = observation.node_features["machine"].shape[0]
    edge_actions = (
        capability.edge_index[0] * machine_count
        + capability.edge_index[1]
    )
    controlled = observation.copy()
    for action in legal_actions:
        row = int(np.flatnonzero(edge_actions == action)[0])
        controlled.relations[CAPABLE_EDGE].edge_features[
            row, processing_column
        ] = 0.5
        controlled.relations[CAPABLE_EDGE].edge_features[
            row, reconfiguration_column
        ] = 0.0
    first_row = int(np.flatnonzero(edge_actions == first_action)[0])
    second_row = int(np.flatnonzero(edge_actions == second_action)[0])
    controlled.relations[CAPABLE_EDGE].edge_features[
        first_row, processing_column
    ] = 0.25
    controlled.relations[CAPABLE_EDGE].edge_features[
        second_row, processing_column
    ] = 0.75

    logits, _ = network(controlled, mask, device="cpu")

    assert logits[first_action] > logits[second_action]


def test_behavior_cloning_top1_loss_treats_teacher_ties_as_equivalent():
    state = SimpleNamespace(
        action_mask=np.asarray([False, False, False, True]),
        legal_actions=(0, 1, 2),
        teacher_action=0,
        teacher_keys=((1.0,), (1.0,), (2.0,)),
        pairwise_actions=((0, 2), (1, 2)),
    )
    first = _state_loss(
        torch.tensor([4.0, 0.0, -2.0, -100.0]), state
    )
    second = _state_loss(
        torch.tensor([0.0, 4.0, -2.0, -100.0]), state
    )
    assert torch.allclose(first, second)


def test_direct_edge_heads_handle_only_advance(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    network = build_actor_critic(observation, _small_m1_config())
    production_mask = np.ones(environment.production_action_size, dtype=bool)
    production_mask[-1] = False
    relations = dict(observation.relations)
    capable = relations[CAPABLE_EDGE]
    relations[CAPABLE_EDGE] = EdgeStore(
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_features=np.empty(
            (0, capable.edge_features.shape[1]), dtype=np.float32
        ),
        feature_names=capable.feature_names,
        bidirectional=True,
    )
    only_advance = replace(observation, relations=relations)
    logits, value = network(only_advance, production_mask, device="cpu")
    assert torch.isfinite(logits[-1])
    assert torch.isfinite(value)
    assert torch.all(logits[:-1] == torch.finfo(logits.dtype).min)

    worker_observation, _ = _find_worker_observation(config, fixed_instance)
    worker_mask = np.ones(
        worker_observation.machines.shape[0]
        * worker_observation.workers.shape[0]
        + 1,
        dtype=bool,
    )
    worker_mask[-1] = False
    logits, value = network(
        worker_observation, worker_mask, device="cpu"
    )
    assert torch.isfinite(logits[-1])
    assert torch.isfinite(value)


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
    assert read_checkpoint_network_spec(gnn_checkpoint) == {
        **gnn_config,
        "production_action_edge_features": False,
        "worker_action_edge_features": False,
        "production_candidate_relative_features": False,
        "worker_candidate_relative_features": False,
        "policy_head_version": 5,
        "production_action_semantics": "pair_plus_defer_v1",
        "production_relative_feature_names": (),
        "worker_relative_feature_names": (),
        "candidate_context_mode": "common_offset_v4",
        "worker_relative_weight_sharing": "shared_mean_v4",
        "observation_schema_version": 3,
        "feature_dimensions": observation.feature_dimensions,
        "edge_feature_dimensions": observation.edge_feature_dimensions,
    }
    gnn_clone = PPOAgent(
        build_actor_critic(observation, gnn_config),
        config["ppo"],
        device="cpu",
    )
    assert gnn_clone.load(gnn_checkpoint) == {"kind": "gnn"}

    old_gnn_checkpoint = tmp_path / "old_gnn.pt"
    old_gnn_payload = torch.load(
        gnn_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    old_gnn_payload["network_spec"] = dict(gnn_config)
    torch.save(old_gnn_payload, old_gnn_checkpoint)
    inferred_old_gnn = read_checkpoint_network_spec(old_gnn_checkpoint)
    assert inferred_old_gnn["observation_schema_version"] == 1
    assert inferred_old_gnn["feature_dimensions"] == (
        observation.feature_dimensions
    )
    assert inferred_old_gnn["edge_feature_dimensions"] == (
        observation.edge_feature_dimensions
    )
    with pytest.raises(ValueError, match="policy head version is incompatible"):
        gnn_clone.load(old_gnn_checkpoint)

    v4_checkpoint = tmp_path / "v4_gnn.pt"
    v4_payload = torch.load(
        gnn_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    v4_spec = dict(v4_payload["network_spec"])
    v4_spec["policy_head_version"] = 4
    v4_spec.pop("production_action_semantics", None)
    v4_payload["network_spec"] = v4_spec
    torch.save(v4_payload, v4_checkpoint)
    with pytest.raises(
        ValueError,
        match="v4 and older checkpoints used production advance semantics",
    ):
        gnn_clone.load(v4_checkpoint)

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
        "observation_schema_version": 1,
        "feature_dimensions": {
            name: observation.feature_dimensions[name]
            for name in ("operation", "machine", "worker", "global")
        },
    }
    typed_clone = PPOAgent(
        TypedActorCritic(observation.feature_dimensions, 16),
        config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="observation schema is incompatible"):
        typed_clone.load(legacy_checkpoint)
    with pytest.raises(ValueError, match="does not match"):
        gnn_clone.load(legacy_checkpoint)
