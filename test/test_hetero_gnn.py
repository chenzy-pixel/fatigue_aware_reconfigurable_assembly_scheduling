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
        "policy_head_version": 6,
        "production_action_semantics": "pair_plus_defer_v1",
        "production_relative_feature_names": (
            "processing_time_norm",
            "reconfiguration_time_norm",
            "fixed_reconfiguration_cost_norm",
            "estimated_labor_cost_norm",
            "estimated_downtime_cost_norm",
            "horizon_slack_norm",
        ),
        "worker_relative_feature_names": (
            "stage_duration_norm",
            "projected_fatigue_ratio",
            "incremental_labor_cost_norm",
            "incremental_downtime_cost_norm",
            "incremental_load_variance_norm",
        ),
        "relative_weight_parameterization": "independent_softplus_signed_v6",
        "production_relative_initial_weights": (
            0.30,
            0.30,
            0.20,
            0.20,
            0.20,
            0.30,
        ),
        "worker_relative_initial_weights": (0.30, 0.30, 0.20, 0.20, 0.20),
        "candidate_context_mode": "common_plus_gated_residual_v6",
        "worker_relative_weight_sharing": "independent_softplus_v6",
        "common_context_gate_initial_logit": -4.0,
        "residual_context_gate_initial_logit": -4.0,
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
    v5_config = deepcopy(config["network"])
    v5_config["policy_head_version"] = 5
    with pytest.raises(ValueError, match="require retraining"):
        build_actor_critic(observation, v5_config)
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
    assert weights["production"] == pytest.approx(
        {
            "processing_time_norm": -0.30,
            "reconfiguration_time_norm": -0.30,
            "fixed_reconfiguration_cost_norm": -0.20,
            "estimated_labor_cost_norm": -0.20,
            "estimated_downtime_cost_norm": -0.20,
            "horizon_slack_norm": 0.30,
        }
    )
    assert weights["worker"] == pytest.approx(
        {
            "stage_duration_norm": -0.30,
            "projected_fatigue_ratio": -0.30,
            "incremental_labor_cost_norm": -0.20,
            "incremental_downtime_cost_norm": -0.20,
            "incremental_load_variance_norm": -0.20,
        }
    )

    with torch.no_grad():
        network.production_relative_ranker.weight[0, 0].add_(1.0)
        network.worker_relative_ranker.weight[0, 0].add_(1.0)
    updated = network.effective_relative_cost_weights()
    assert updated["production"]["processing_time_norm"] < -0.30
    assert updated["production"]["reconfiguration_time_norm"] == pytest.approx(
        -0.30
    )
    assert updated["worker"]["stage_duration_norm"] < -0.30
    assert updated["worker"]["projected_fatigue_ratio"] == pytest.approx(-0.30)


def test_candidate_context_residual_excludes_masked_and_degenerate_candidates():
    feasible = torch.tensor([True, True, False])
    common, residual = HeteroGraphActorCritic._candidate_context_components(
        torch.tensor([1.0, 3.0, 100.0]),
        feasible,
    )
    changed_common, changed_residual = (
        HeteroGraphActorCritic._candidate_context_components(
            torch.tensor([1.0, 3.0, -100.0]),
            feasible,
        )
    )
    assert common[:2].tolist() == pytest.approx([2.0, 2.0])
    assert residual[:2].tolist() == pytest.approx([-1.0, 1.0])
    assert torch.equal(common[:2], changed_common[:2])
    assert torch.equal(residual[:2], changed_residual[:2])

    single_common, single_residual = (
        HeteroGraphActorCritic._candidate_context_components(
            torch.tensor([2.0, 500.0]),
            torch.tensor([True, False]),
        )
    )
    assert torch.isfinite(single_common).all()
    assert torch.equal(single_residual, torch.zeros_like(single_residual))


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


@pytest.mark.parametrize(
    "feature_name",
    (
        "stage_duration_norm",
        "projected_fatigue_ratio",
        "incremental_labor_cost_norm",
        "incremental_downtime_cost_norm",
        "incremental_load_variance_norm",
    ),
)
def test_worker_v6_relative_features_independently_penalize_higher_values(
    config,
    fixed_instance,
    feature_name,
):
    observation, mask = _find_worker_observation(
        config,
        fixed_instance,
        minimum_pair_actions=2,
    )
    network = build_actor_critic(observation, _small_m1_config())
    service = observation.relations[SERVICE_CANDIDATE_EDGE]
    names = service.feature_names
    worker_count = observation.node_features["worker"].shape[0]
    edge_actions = service.edge_index[0] * worker_count + service.edge_index[1]
    legal_actions = [int(action) for action in np.flatnonzero(~mask[:-1])]
    first_action, second_action = legal_actions[:2]
    controlled = observation.copy()
    for name in (
        "stage_duration_norm",
        "projected_fatigue_ratio",
        "incremental_labor_cost_norm",
        "incremental_downtime_cost_norm",
        "incremental_load_variance_norm",
    ):
        column = names.index(name)
        for action in legal_actions:
            row = int(np.flatnonzero(edge_actions == action)[0])
            controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
                row, column
            ] = 0.5
    feature_column = names.index(feature_name)
    first_row = int(np.flatnonzero(edge_actions == first_action)[0])
    second_row = int(np.flatnonzero(edge_actions == second_action)[0])
    controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
        first_row, feature_column
    ] = 0.25
    controlled.relations[SERVICE_CANDIDATE_EDGE].edge_features[
        second_row, feature_column
    ] = 0.75

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


@pytest.mark.parametrize(
    ("feature_name", "higher_is_better"),
    (
        ("processing_time_norm", False),
        ("reconfiguration_time_norm", False),
        ("fixed_installation_cost_norm", False),
        ("estimated_labor_cost_norm", False),
        ("estimated_downtime_cost_norm", False),
        ("horizon_slack_norm", True),
    ),
)
def test_production_v6_relative_features_have_expected_monotone_direction(
    config,
    fixed_instance,
    feature_name,
    higher_is_better,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    legal_actions = [int(action) for action in np.flatnonzero(~mask[:-1])]
    assert len(legal_actions) >= 2
    first_action, second_action = legal_actions[:2]
    network = build_actor_critic(observation, _small_m1_config())
    capability = observation.relations[CAPABLE_EDGE]
    names = capability.feature_names
    machine_count = observation.node_features["machine"].shape[0]
    edge_actions = capability.edge_index[0] * machine_count + capability.edge_index[1]
    controlled = observation.copy()
    controlled_names = (
        "processing_time_norm",
        "reconfiguration_time_norm",
        "fixed_disassembly_cost_norm",
        "fixed_installation_cost_norm",
        "estimated_labor_cost_norm",
        "estimated_downtime_cost_norm",
        "horizon_slack_norm",
    )
    for name in controlled_names:
        column = names.index(name)
        for action in legal_actions:
            row = int(np.flatnonzero(edge_actions == action)[0])
            controlled.relations[CAPABLE_EDGE].edge_features[row, column] = 0.5
    feature_column = names.index(feature_name)
    first_row = int(np.flatnonzero(edge_actions == first_action)[0])
    second_row = int(np.flatnonzero(edge_actions == second_action)[0])
    first_value, second_value = (
        (0.75, 0.25) if higher_is_better else (0.25, 0.75)
    )
    controlled.relations[CAPABLE_EDGE].edge_features[
        first_row, feature_column
    ] = first_value
    controlled.relations[CAPABLE_EDGE].edge_features[
        second_row, feature_column
    ] = second_value

    logits, _ = network(controlled, mask, device="cpu")

    assert logits[first_action] > logits[second_action]


def test_v6_candidate_residual_can_rank_pairs_without_changing_common_offset(
    config,
    fixed_instance,
):
    class IndexedScorer(nn.Module):
        def forward(self, inputs):
            values = torch.arange(
                inputs[..., 0].numel(),
                dtype=inputs.dtype,
                device=inputs.device,
            ).reshape(inputs.shape[:-1])
            return values.unsqueeze(-1)

    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    legal_actions = [int(action) for action in np.flatnonzero(~mask[:-1])]
    assert len(legal_actions) >= 2
    network = build_actor_critic(observation, _small_m1_config())
    network.production_scorer = IndexedScorer()
    with torch.no_grad():
        network.production_context_gate.fill_(-10.0)
        network.production_residual_context_gate.fill_(10.0)
        network.production_relative_ranker.weight.fill_(-100.0)

    logits, _ = network(observation, mask, device="cpu")

    assert logits[legal_actions[-1]] > logits[legal_actions[0]]


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


def test_v6_policy_head_diagnostics_and_optimizer_checkpoint_restore(
    config,
    fixed_instance,
    tmp_path,
):
    effective_config = deepcopy(config)
    effective_config["network"] = _small_m1_config()
    effective_config["ppo"]["epochs"] = 1
    effective_config["ppo"]["batch_size"] = 4
    set_seed(int(effective_config["seed"]))

    production_environment = AssemblySchedulingEnv(effective_config)
    production_observation = production_environment.reset(fixed_instance)
    production_mask = production_environment.get_action_mask()
    assert np.count_nonzero(~production_mask[:-1]) >= 2
    worker_observation, worker_mask = _find_worker_observation(
        effective_config,
        fixed_instance,
        minimum_pair_actions=2,
    )
    network = build_actor_critic(
        production_observation,
        effective_config["network"],
    )
    agent = PPOAgent(network, effective_config["ppo"], device="cpu")

    initial_diagnostics = agent.policy_head_diagnostics()
    network_spec = network.network_spec()
    expected_weight_keys = {
        f"policy_head_weight_production_{name}"
        for name in network_spec["production_relative_feature_names"]
    } | {
        f"policy_head_weight_worker_{name}"
        for name in network_spec["worker_relative_feature_names"]
    }
    expected_gate_keys = {
        "policy_head_gate_production_common",
        "policy_head_gate_production_residual",
        "policy_head_gate_worker_common",
        "policy_head_gate_worker_residual",
    }
    assert set(initial_diagnostics) == expected_weight_keys | expected_gate_keys
    assert len(initial_diagnostics) == 15
    assert all(np.isfinite(value) for value in initial_diagnostics.values())
    expected_initial_gate = float(torch.sigmoid(torch.tensor(-4.0)))
    assert all(
        initial_diagnostics[name] == pytest.approx(expected_initial_gate)
        for name in expected_gate_keys
    )

    buffer = RolloutBuffer(preserve_graph=True)
    samples = (
        (production_observation, production_mask, 0.0),
        (worker_observation, worker_mask, 1.0),
        (production_observation, production_mask, -0.5),
        (worker_observation, worker_mask, 2.0),
    )
    for index, (observation, mask, reward) in enumerate(samples):
        action, log_probability, value = agent.act(observation, mask)
        buffer.add(
            observation,
            mask,
            action,
            log_probability,
            value,
            reward,
            index == len(samples) - 1,
        )
    buffer.compute_gae(
        last_value=0.0,
        gamma=float(effective_config["ppo"]["gamma"]),
        gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
    )
    metrics = agent.update(buffer)
    assert expected_weight_keys | expected_gate_keys <= set(metrics)
    assert all(np.isfinite(metrics[name]) for name in initial_diagnostics)
    for parameter in (
        network.production_relative_ranker.weight,
        network.worker_relative_ranker.weight,
        network.production_context_gate,
        network.production_residual_context_gate,
        network.worker_context_gate,
        network.worker_residual_context_gate,
    ):
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))

    checkpoint = tmp_path / "v6_optimizer.pt"
    agent.save(checkpoint, metadata={"kind": "v6"})
    clone_network = build_actor_critic(
        production_observation,
        effective_config["network"],
    )
    clone = PPOAgent(clone_network, effective_config["ppo"], device="cpu")
    metadata = clone.load(checkpoint, load_optimizer=True)
    assert metadata == {
        "kind": "v6",
        "policy_head_diagnostics": agent.policy_head_diagnostics(),
    }
    for original, restored in zip(
        network.parameters(), clone_network.parameters()
    ):
        assert torch.equal(original.detach(), restored.detach())
    original_optimizer = agent.optimizer.state_dict()
    restored_optimizer = clone.optimizer.state_dict()
    assert original_optimizer["param_groups"] == restored_optimizer["param_groups"]
    assert original_optimizer["state"].keys() == restored_optimizer["state"].keys()
    for parameter_id, original_state in original_optimizer["state"].items():
        for name, original_value in original_state.items():
            restored_value = restored_optimizer["state"][parameter_id][name]
            if torch.is_tensor(original_value):
                assert torch.equal(original_value, restored_value)
            else:
                assert original_value == restored_value


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
        "policy_head_version": 6,
        "production_action_semantics": "pair_plus_defer_v1",
        "production_relative_feature_names": (),
        "worker_relative_feature_names": (),
        "relative_weight_parameterization": "independent_softplus_signed_v6",
        "production_relative_initial_weights": (),
        "worker_relative_initial_weights": (),
        "candidate_context_mode": "common_plus_gated_residual_v6",
        "worker_relative_weight_sharing": "independent_softplus_v6",
        "common_context_gate_initial_logit": -4.0,
        "residual_context_gate_initial_logit": -4.0,
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

    v5_checkpoint = tmp_path / "v5_gnn.pt"
    v5_payload = torch.load(
        gnn_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    v5_spec = dict(v5_payload["network_spec"])
    v5_spec["policy_head_version"] = 5
    v5_payload["network_spec"] = v5_spec
    torch.save(v5_payload, v5_checkpoint)
    with pytest.raises(
        ValueError,
        match="not automatically converted to v6",
    ):
        gnn_clone.load(v5_checkpoint)

    wrong_features_checkpoint = tmp_path / "wrong_features_gnn.pt"
    wrong_features_payload = torch.load(
        gnn_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    wrong_features_spec = dict(wrong_features_payload["network_spec"])
    wrong_features_spec["worker_relative_feature_names"] = ("wrong",)
    wrong_features_payload["network_spec"] = wrong_features_spec
    torch.save(wrong_features_payload, wrong_features_checkpoint)
    with pytest.raises(ValueError, match="worker_relative_feature_names"):
        gnn_clone.load(wrong_features_checkpoint)

    wrong_context_checkpoint = tmp_path / "wrong_context_gnn.pt"
    wrong_context_payload = torch.load(
        gnn_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    wrong_context_spec = dict(wrong_context_payload["network_spec"])
    wrong_context_spec["candidate_context_mode"] = "common_offset_v4"
    wrong_context_payload["network_spec"] = wrong_context_spec
    torch.save(wrong_context_payload, wrong_context_checkpoint)
    with pytest.raises(ValueError, match="candidate_context_mode"):
        gnn_clone.load(wrong_context_checkpoint)

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
