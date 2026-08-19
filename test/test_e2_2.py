from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agent.ppo import (
    PPOAgent,
    build_actor_critic,
    summarize_policy_decision_diagnostics,
)
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv, PreferenceVector
from result import aggregate_evaluation_rows, aggregate_preference_diagnostics
from train import _pareto_snapshot, _validation_log_row


def _e2_2_network(preference=(1.0, 0.0, 0.0)):
    config = load_config("configs/v7/e2_2_hierarchical_preference.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=preference)
    network = build_actor_critic(observation, config["network"])
    return config, environment, observation, network


def _joint(
    network,
    pair_logits: torch.Tensor,
    *,
    commit: float = 0.4,
    defer: float = -0.2,
    mask: list[bool] | None = None,
) -> torch.Tensor:
    action_mask = torch.tensor(
        mask if mask is not None else [False] * (len(pair_logits) + 1),
        dtype=torch.bool,
    )
    return network._hierarchical_production_logits(
        pair_logits,
        pair_logits.new_tensor(commit),
        pair_logits.new_tensor(defer),
        action_mask,
    )


def test_e2_2_config_is_independent_and_keeps_e2_1_contract() -> None:
    e2_1 = load_config("configs/v7/e2_1_preference_pareto.json")
    e2_2 = load_config("configs/v7/e2_2_hierarchical_preference.json")

    assert e2_1["experiment_name"] == "v7_e2_1_preference_pareto"
    assert e2_1["network"]["production_action_semantics"] == (
        "pair_plus_defer_v1"
    )
    assert e2_2["experiment_name"] == "v7_e2_2_hierarchical_preference"
    assert e2_2["experiment_suite_version"] == (
        "v7_e2_2_pareto_protocol_v1"
    )
    assert e2_2["network"]["production_action_semantics"] == (
        "hierarchical_commit_then_pair_v2"
    )
    assert e2_2["network"]["production_commit_set_scorer"] is True
    assert e2_2["evaluation"]["result_schema_version"] == "4.3.0"
    assert e2_2["reward"] == e2_1["reward"]
    assert e2_2["training"]["preference_grouping"] == e2_1["training"][
        "preference_grouping"
    ]


def test_hierarchical_gate_is_invariant_to_conditional_pair_scores() -> None:
    _, _, _, network = _e2_2_network()
    gate_probability = torch.softmax(torch.tensor([0.4, -0.2]), dim=0)[0]

    baseline = _joint(network, torch.tensor([2.0, 0.0, -1.0]))
    shifted = _joint(network, torch.tensor([102.0, 100.0, 99.0]))
    rescaled = _joint(network, torch.tensor([20.0, 0.0, -10.0]))
    extra_candidate = _joint(network, torch.tensor([2.0, 0.0, -1.0, 8.0]))

    for joint in (baseline, shifted, rescaled, extra_candidate):
        assert torch.exp(joint[:-1]).sum() == pytest.approx(gate_probability)
        assert torch.exp(joint[-1]) == pytest.approx(1.0 - gate_probability)
        assert torch.exp(joint).sum() == pytest.approx(1.0)

    objectives = torch.tensor([[1.0, 5.0], [5.0, 1.0]])
    feasible = torch.tensor([True, True])
    flow_scores = network._direct_preference_logits(
        objectives,
        feasible,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    cost_scores = network._direct_preference_logits(
        objectives,
        feasible,
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert int(torch.argmax(flow_scores)) != int(torch.argmax(cost_scores))
    flow_joint = _joint(network, flow_scores)
    cost_joint = _joint(network, cost_scores)
    assert float(torch.exp(flow_joint[:-1]).sum().detach()) == pytest.approx(
        float(torch.exp(cost_joint[:-1]).sum().detach())
    )


def test_hierarchical_masks_and_greedy_decode_gate_first() -> None:
    _, _, _, network = _e2_2_network()
    pair_logits = torch.tensor([1.0, -1.0])

    defer_masked = _joint(
        network,
        pair_logits,
        mask=[False, False, True],
    )
    assert torch.exp(defer_masked[:-1]).sum() == pytest.approx(1.0)
    assert torch.exp(defer_masked[-1]) == 0.0

    commit_masked = _joint(
        network,
        pair_logits,
        mask=[True, True, False],
    )
    assert torch.exp(commit_masked[:-1]).sum() == 0.0
    assert torch.exp(commit_masked[-1]) == pytest.approx(1.0)

    singleton = _joint(
        network,
        pair_logits,
        mask=[False, True, False],
    )
    assert torch.isfinite(singleton).all()
    assert torch.exp(singleton).sum() == pytest.approx(1.0)

    commit_wins_but_no_pair_is_joint_top = torch.log(
        torch.tensor([0.30, 0.30, 0.40])
    )
    assert PPOAgent._hierarchical_greedy_action(
        commit_wins_but_no_pair_is_joint_top,
        np.array([False, False, False]),
    ) == 0
    assert PPOAgent._hierarchical_greedy_action(
        torch.log(torch.tensor([0.20, 0.20, 0.60])),
        np.array([False, False, False]),
    ) == 2
    assert PPOAgent._hierarchical_greedy_action(
        torch.log(torch.tensor([0.25, 0.25, 0.50])),
        np.array([False, False, False]),
    ) == 0


def test_hierarchical_joint_log_prob_and_heads_have_finite_gradients() -> None:
    config, environment, observation, network = _e2_2_network()
    action_mask = environment.get_action_mask()
    legal_pairs = np.flatnonzero(~action_mask[:-1])
    assert len(legal_pairs) >= 2

    logits, value = network(observation, action_mask, device="cpu")
    distribution = Categorical(logits=logits)
    action = torch.tensor(int(legal_pairs[0]))
    log_probability = distribution.log_prob(action)
    ratio = torch.exp(log_probability - log_probability.detach())
    loss = -log_probability + 0.0 * value
    loss.backward()

    assert torch.isfinite(logits[~torch.as_tensor(action_mask)]).all()
    assert torch.isfinite(distribution.entropy())
    assert float(ratio.detach()) == pytest.approx(1.0)
    assert network.preference_action_scale_raw.grad is not None
    assert torch.isfinite(network.preference_action_scale_raw.grad)
    assert network.production_commit_set is not None
    commit_gradients = [
        parameter.grad for parameter in network.production_commit_set.parameters()
    ]
    assert any(gradient is not None for gradient in commit_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in commit_gradients
        if gradient is not None
    )
    assert config["ppo"]["clip_epsilon"] > 0.0


def test_e2_2_checkpoint_round_trip_and_e2_1_rejection(tmp_path) -> None:
    config, _, observation, network = _e2_2_network()
    checkpoint = tmp_path / "e2_2.pt"
    agent = PPOAgent(network, config["ppo"], device="cpu")
    agent.save(checkpoint)

    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    clone.load(checkpoint)
    assert clone.network.network_spec()["production_action_semantics"] == (
        "hierarchical_commit_then_pair_v2"
    )

    e2_1_config = load_config("configs/v7/e2_1_preference_pareto.json")
    e2_1_agent = PPOAgent(
        build_actor_critic(observation, e2_1_config["network"]),
        e2_1_config["ppo"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="production action semantics"):
        e2_1_agent.load(checkpoint)


def _aggregate_row(**diagnostics) -> dict:
    return {
        "terminated": True,
        "truncated": False,
        "makespan": 100.0,
        "total_flow_time": 500.0,
        "flow_time_objective": 500.0,
        "reconfiguration_cost": 10.0,
        "worker_load_variance": 2.0,
        "inference_time_seconds": 0.2,
        "solve_time_seconds": 0.5,
        "inference_time_per_decision_ms": 2.0,
        "relative_heuristic_gap_percent": 5.0,
        "makespan_heuristic_gap_percent": 4.0,
        "reconfiguration_cost_heuristic_gap_percent": 3.0,
        "worker_load_variance_heuristic_gap_percent": 2.0,
        "maximum_worker_fatigue": 0.5,
        "mean_peak_worker_fatigue": 0.4,
        "safe_fatigue_limit": 0.75,
        "schedule_violation_count": 0,
        "decisions": 100,
        **diagnostics,
    }


def test_preference_diagnostics_are_weighted_and_persisted_in_aggregates() -> None:
    rows = [
        _aggregate_row(
            ranker_top_decision_count=2,
            preference_override_count=1,
            preference_override_rate=0.5,
            mean_preference_logit_std=0.2,
        ),
        _aggregate_row(
            ranker_top_decision_count=6,
            preference_override_count=3,
            preference_override_rate=0.5,
            mean_preference_logit_std=0.4,
        ),
    ]
    diagnostics = aggregate_preference_diagnostics(rows)
    assert diagnostics == {
        "ranker_top_decision_count": 8,
        "preference_override_count": 4,
        "preference_override_rate": pytest.approx(0.5),
        "mean_preference_logit_std": pytest.approx(0.35),
    }

    aggregate = aggregate_evaluation_rows(
        rows,
        dataset="validation",
        policy="ppo",
        manifest="manifest.json",
        schema_version="4.3.0",
    )
    assert aggregate["evaluation_schema_version"] == "4.3.0"
    assert aggregate["preference_override_rate"] == pytest.approx(0.5)
    assert aggregate["mean_preference_logit_std"] == pytest.approx(0.35)
    validation_row = _validation_log_row(aggregate, completed_episodes=10)
    assert validation_row["preference_override_count"] == 4
    assert validation_row["preference_override_rate"] == pytest.approx(0.5)
    assert validation_row["mean_preference_logit_std"] == pytest.approx(0.35)

    config = load_config("configs/v7/e2_2_hierarchical_preference.json")
    canonical = PreferenceVector(0.5, 0.3, 0.2)
    pareto_rows = []
    for index, row in enumerate(rows):
        candidate = deepcopy(row)
        candidate.update(
            {
                "instance_id": f"instance-{index}",
                "preference_key": "_".join(
                    f"{value:.12g}" for value in canonical.as_tuple()
                ),
                "preference_quality_score": 0.25,
            }
        )
        pareto_rows.append(candidate)
    snapshot = _pareto_snapshot(
        pareto_rows,
        config=config,
        scope="full_grid_22",
        update_id=1,
        completed_episodes=10,
        fatigue_tolerance=1e-9,
    )
    assert snapshot["preference_override_count"] == 4
    assert snapshot["preference_override_rate"] == pytest.approx(0.5)
    assert snapshot["mean_preference_logit_std"] == pytest.approx(0.35)


def test_policy_diagnostic_summary_contains_required_fields() -> None:
    summary = summarize_policy_decision_diagnostics(
        [
            {
                "decision_type": "production",
                "legal_pair_count": 2,
                "terminal_legal": True,
                "preference_overrode_relative_top": True,
                "preference_logit_std": 0.3,
                "ranker_top_selected": True,
                "context_overrode_top": False,
                "commit_set_logit": 0.1,
            }
        ]
    )
    assert summary["preference_override_count"] == 1
    assert summary["preference_override_rate"] == pytest.approx(1.0)
    assert summary["mean_preference_logit_std"] == pytest.approx(0.3)
