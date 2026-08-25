from __future__ import annotations

import csv
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agent.ppo import PPOAgent, build_actor_critic, summarize_policy_decision_diagnostics
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv, CANONICAL_PREFERENCE, simplex_lattice
from eval import _evaluation_row
from result import (
    aggregate_evaluation_rows,
    evaluation_quality_metric,
    result_schema_version,
)
from result.io import write_csv
from train import (
    ParetoSafetyGuard,
    TrainingPhaseController,
    _pareto_snapshot,
    _preference_key,
    _tiered_hard_failure_reason,
    _select_tiered_primary_role,
)


def _network(preference=(0.5, 0.3, 0.2)):
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=preference)
    network = build_actor_critic(observation, config["network"])
    return config, instance, environment, observation, network


def test_e2_4_config_and_checkpoint_contract(tmp_path) -> None:
    config, _, _, observation, network = _network()
    assert config["experiment_name"] == "v7_e2_4_neutral_gate_safe_variance"
    assert result_schema_version(config) == "5.0.0"
    assert network.network_spec()["production_gate"] == {
        "version": "state_only_action_set_gate_v1",
        "preference_conditioning": False,
        "tie_break": "commit",
    }
    checkpoint = tmp_path / "e2_4.pt"
    PPOAgent(network, config["ppo"], device="cpu").save(checkpoint)
    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    clone.load(checkpoint)

    for path in (
        "configs/v7/e2_preference_conditioned.json",
        "configs/v7/e2_1_preference_pareto.json",
        "configs/v7/e2_2_hierarchical_preference.json",
        "configs/v7/e2_3_safe_production_preference.json",
    ):
        old = load_config(path)
        old_agent = PPOAgent(
            build_actor_critic(observation, old["network"]),
            old["ppo"],
            device="cpu",
        )
        with pytest.raises(ValueError, match="production action semantics"):
            old_agent.load(checkpoint)


def test_state_only_gate_is_preference_and_pair_score_invariant() -> None:
    _, instance, environment, observation, network = _network()
    network.eval()
    first_gate = network._state_only_production_gate_logits(
        observation, dtype=torch.float32, device="cpu"
    )
    second_observation = environment.reset(
        instance, preference=(0.0, 1.0, 0.0)
    )
    second_gate = network._state_only_production_gate_logits(
        second_observation, dtype=torch.float32, device="cpu"
    )
    assert torch.equal(first_gate, second_gate)

    mask = torch.tensor([False, False, False, False])
    baseline = network._hierarchical_production_logits(
        torch.tensor([2.0, 0.0, -1.0]), first_gate[0], first_gate[1], mask
    )
    shifted = network._hierarchical_production_logits(
        torch.tensor([102.0, 100.0, 99.0]), first_gate[0], first_gate[1], mask
    )
    assert float(torch.exp(baseline[:-1]).sum().detach()) == pytest.approx(
        float(torch.exp(shifted[:-1]).sum().detach())
    )
    assert float(torch.exp(baseline[-1].detach())) == pytest.approx(
        float(torch.exp(shifted[-1].detach()))
    )

    objectives = torch.tensor([[1.0, 9.0], [9.0, 1.0]])
    feasible = torch.tensor([True, True])
    flow = network._direct_preference_logits(
        objectives, feasible, torch.tensor([1.0, 0.0, 0.0])
    )
    cost = network._direct_preference_logits(
        objectives, feasible, torch.tensor([0.0, 1.0, 0.0])
    )
    assert int(torch.argmax(flow)) != int(torch.argmax(cost))


def test_state_gate_and_preference_paths_have_finite_gradients() -> None:
    config, _, environment, observation, network = _network()
    network.train()
    mask = environment.get_action_mask()
    legal_pairs = np.flatnonzero(~mask[:-1])
    assert len(legal_pairs) >= 2 and not mask[-1]
    logits, value = network(observation, mask, device="cpu")
    distribution = Categorical(logits=logits)
    loss = -distribution.log_prob(torch.tensor(int(legal_pairs[0]))) + 0.0 * value
    loss.backward()
    assert torch.isfinite(distribution.entropy())
    assert network.preference_action_scale_raw.grad is not None
    assert torch.isfinite(network.preference_action_scale_raw.grad)
    assert network.production_state_gate is not None
    gate_gradients = [
        parameter.grad for parameter in network.production_state_gate.parameters()
    ]
    assert any(gradient is not None for gradient in gate_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gate_gradients
        if gradient is not None
    )
    assert config["ppo"]["clip_epsilon"] > 0.0


def test_safe_worker_variance_direct_term_has_no_flow_or_cost_component() -> None:
    _, _, _, _, network = _network()
    feasible = torch.tensor([False, True, True])
    components = network._safe_worker_variance_preference_logit_components(
        torch.tensor([9.0, 1.0, 5.0]),
        feasible,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert torch.allclose(components[:, :2], torch.zeros_like(components[:, :2]))
    assert int(torch.argmax(components.sum(dim=-1)[feasible])) == 0
    diagnostics = summarize_policy_decision_diagnostics(
        [
            {
                "decision_type": "worker",
                "action_count": 4,
                "legal_pair_count": 2,
                "terminal_legal": True,
                "preference_overrode_relative_top": True,
                "worker_variance_preference_overrode_relative_top": True,
                "preference_logit_std": 0.3,
                "worker_direct_preference_flow_logit_max_abs": 0.0,
                "worker_direct_preference_cost_logit_max_abs": 0.0,
                "worker_direct_preference_variance_logit_max_abs": 0.5,
            }
        ]
    )
    assert diagnostics["worker_variance_preference_override_count"] == 1
    assert diagnostics["worker_direct_preference_flow_logit_max_abs"] == 0.0
    assert diagnostics["worker_direct_preference_cost_logit_max_abs"] == 0.0


def _full_grid_rows() -> tuple[list[dict], tuple[str, ...], tuple[str, ...]]:
    preferences = tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
    instances = tuple(f"validation-{index}" for index in range(20))
    rows: list[dict] = []
    for instance_id in instances:
        for index, preference in enumerate(preferences):
            rows.append(
                {
                    "instance_id": instance_id,
                    "terminated": True,
                    "truncated": False,
                    "schedule_violation_count": 0,
                    "maximum_worker_fatigue": 0.5,
                    "safe_fatigue_limit": 0.75,
                    "flow_time_objective": 200.0 - 20.0 * preference.flow,
                    "reconfiguration_cost": 300.0 - 20.0 * preference.cost,
                    "worker_load_variance": 20.0 - 5.0 * preference.variance,
                    "preference_quality_score": 0.25,
                    "preference_key": _preference_key(preference),
                    "w_flow": preference.flow,
                    "w_cost": preference.cost,
                    "w_variance": preference.variance,
                    "action_trace_sha256": f"trace-{index % 8}",
                    "worker_direct_preference_flow_logit_max_abs": 0.0,
                    "worker_direct_preference_cost_logit_max_abs": 0.0,
                    "unsafe_worker_preference_selection_count": 0,
                }
            )
    return rows, instances, tuple(_preference_key(value) for value in preferences)


def test_e2_4_full_grid_requires_response_direction() -> None:
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    rows, instances, preferences = _full_grid_rows()
    snapshot = _pareto_snapshot(
        rows,
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=instances,
        expected_preference_keys=preferences,
    )
    assert snapshot["coverage_pass"] is True
    assert snapshot["all_safe"] is True
    assert snapshot["controllability_pass"] is True
    assert snapshot["preference_response_pass"] is True
    controller = TrainingPhaseController.from_config(config)
    controller.phase = "quality"
    assert controller.observe_pareto_snapshot(snapshot, completed_episodes=200) == "accepted"
    final_decision = controller.evaluate_final_pareto_snapshot(snapshot)
    assert final_decision["pass"] is True
    assert final_decision["checks"]["completion_pass"] is True

    failed = deepcopy(rows)
    for row in failed:
        row["worker_load_variance"] = 10.0
    failed_snapshot = _pareto_snapshot(
        failed,
        config=config,
        scope="full_grid_22",
        update_id=40,
        completed_episodes=400,
        fatigue_tolerance=1e-9,
        expected_instance_ids=instances,
        expected_preference_keys=preferences,
    )
    assert failed_snapshot["preference_response_spearman_variance"] == 0.0
    assert failed_snapshot["preference_response_pass"] is False


def test_tiered_snapshot_separates_completion_from_physical_safety() -> None:
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    rows, instances, preferences = _full_grid_rows()
    rows[0]["terminated"] = False
    rows[0]["truncated"] = True
    snapshot = _pareto_snapshot(
        rows,
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=instances,
        expected_preference_keys=preferences,
    )
    assert snapshot["physical_safety_pass"] is True
    assert snapshot["completion_pass"] is False
    assert snapshot["evaluation_integrity_pass"] is True
    assert snapshot["all_safe"] is False


def test_tiered_hard_metric_gate_rejects_nan_and_canonical_drift() -> None:
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    assert _tiered_hard_failure_reason({"loss": float("nan")}, config) == (
        "non_finite_training_metric:loss"
    )
    assert _tiered_hard_failure_reason(
        {"canonical_identity_max_abs_error": 2e-8}, config
    ) == "canonical_identity_failed"


@pytest.mark.parametrize(
    ("best", "last", "expected"),
    [
        ("passed", "passed", "best_safe"),
        ("passed", "failed", "best_safe"),
        ("failed", "passed", "last_safe"),
        ("failed", "failed", None),
    ],
)
def test_tiered_final_acceptance_any_candidate_matrix(
    best: str, last: str, expected: str | None
) -> None:
    reports = {
        "best_safe": {"acceptance_status": best, "checkpoint_sha256": "same"},
        "last_safe": {"acceptance_status": last, "checkpoint_sha256": "same"},
    }
    assert _select_tiered_primary_role(reports) == expected


def test_e2_4_tiered_safety_guard_only_rolls_back_physical_failures() -> None:
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    guard = ParetoSafetyGuard.from_config(config)
    assert guard is not None
    safe = {"scope": "anchors_5", "physical_safety_pass": True}
    incomplete = {
        "scope": "anchors_5",
        "physical_safety_pass": True,
        "completion_pass": False,
        "completion_rate": 0.99,
    }
    unsafe = {"scope": "anchors_5", "physical_safety_pass": False}
    assert guard.observe(incomplete) == "safe"
    assert guard.rollback_count == 0
    assert guard.observe(safe) == "safe"
    assert guard.consecutive_failures == 0
    assert guard.observe(unsafe) == "rollback"
    assert guard.rollback_count == 1
    guard.record_rollback("full_grid_safe")
    assert guard.consecutive_failures == 0
    assert guard.last_rollback_source == "full_grid_safe"


def test_missing_tiered_policy_retains_legacy_safety_guard_behavior() -> None:
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    del config["training"]["gate_policy"]
    guard = ParetoSafetyGuard.from_config(config)
    assert guard is not None
    unsafe = {"scope": "anchors_5", "coverage_pass": True, "all_safe": False}
    assert guard.observe(unsafe) == "warning"
    assert guard.observe(unsafe) == "rollback"


def test_e2_4_schema_aggregates_new_diagnostics() -> None:
    row = {
        "terminated": True,
        "truncated": False,
        "makespan": 10.0,
        "total_flow_time": 10.0,
        "flow_time_objective": 10.0,
        "reconfiguration_cost": 2.0,
        "worker_load_variance": 1.0,
        "inference_time_seconds": 0.1,
        "solve_time_seconds": 0.2,
        "inference_time_per_decision_ms": 1.0,
        "relative_heuristic_gap_percent": 0.0,
        "makespan_heuristic_gap_percent": 0.0,
        "reconfiguration_cost_heuristic_gap_percent": 0.0,
        "worker_load_variance_heuristic_gap_percent": 0.0,
        "maximum_worker_fatigue": 0.5,
        "mean_peak_worker_fatigue": 0.4,
        "safe_fatigue_limit": 0.75,
        "schedule_violation_count": 0,
        "decisions": 3,
        "production_ranker_top_decision_count": 2,
        "production_conditional_preference_override_count": 1,
        "worker_ranker_top_decision_count": 1,
        "worker_variance_preference_override_count": 1,
        "worker_direct_preference_flow_logit_max_abs": 0.0,
        "worker_direct_preference_cost_logit_max_abs": 0.0,
        "worker_direct_preference_variance_logit_max_abs": 0.4,
        "unsafe_worker_preference_selection_count": 0,
        "production_gate_state_count": 2,
        "production_gate_commit_selected_count": 1,
        "production_gate_defer_selected_count": 1,
        "mean_production_gate_commit_probability": 0.6,
        "mean_production_gate_defer_probability": 0.4,
        "mean_production_gate_logit_margin": 0.2,
    }
    aggregate = aggregate_evaluation_rows(
        [row],
        dataset="validation",
        policy="ppo",
        manifest="manifest.json",
        schema_version="4.5.0",
    )
    assert aggregate["evaluation_schema_version"] == "4.5.0"
    assert aggregate["production_gate_state_count"] == 2
    assert aggregate["worker_variance_preference_override_count"] == 1
    assert aggregate["mean_production_gate_commit_probability"] == 0.6


def test_e2_4_evaluation_row_and_csv_preserve_nonzero_diagnostics(tmp_path) -> None:
    metrics = defaultdict(lambda: 0.0)
    metrics.update(
        {
            "terminated": True,
            "truncated": False,
            "terminal_reason": "completed",
            "decisions": 7,
            "time": 10.0,
            "completed_orders": 1,
            "unfinished_orders": 0,
            "feasibility_proxy_return": 1.0,
            "total_flow_time": 10.0,
            "flow_time_objective": 10.0,
            "reconfiguration_cost": 2.0,
            "worker_load_variance": 1.0,
            "preference": (0.5, 0.3, 0.2),
            "inference_time_seconds": 0.1,
            "solve_time_seconds": 0.2,
            "inference_time_per_decision_ms": 1.0,
            "maximum_worker_fatigue": 0.5,
            "mean_peak_worker_fatigue": 0.4,
            "safe_fatigue_limit": 0.75,
            "schedule_violations": [],
            "worker_switch_ratio": 0.0,
            "production_gate_state_count": 7,
            "production_gate_commit_selected_count": 5,
            "production_gate_defer_selected_count": 2,
            "mean_production_gate_commit_probability": 0.7,
            "mean_production_gate_defer_probability": 0.3,
            "mean_production_gate_logit_margin": 0.9,
            "worker_variance_preference_override_count": 3,
            "worker_direct_preference_variance_logit_max_abs": 0.8,
            "maximum_worker_matching_deficit": 4,
            "future_installation_admission_masked_action_count": 6,
        }
    )
    record = SimpleNamespace(
        instance=SimpleNamespace(instance_id="validation-contract"),
        metadata={
            "seed": 11,
            "pressure_type": "test",
            "cost_profile": "test",
            "heuristic_metrics": {
                "heuristic_flow_time": 10.0,
                "heuristic_makespan": 10.0,
                "heuristic_reconfiguration_cost": 2.0,
                "worker_workload_variance": 1.0,
                "heuristic_completed": True,
                "ready_configuration_gap_ratio": 0.0,
                "heuristic_reconfiguration_ratio": 0.0,
                "mean_wave_overlap_ratio": 0.0,
            },
            "pressure_metrics": {
                "total_effective_load": 1.0,
                "max_module_load": 1.0,
            },
        },
    )
    config = load_config("configs/v7/e2_4_neutral_gate_safe_variance.json")
    row = _evaluation_row(
        record,
        metrics,
        config["reward"],
        evaluation_quality_metric(config),
    )
    assert row["production_gate_state_count"] == 7
    assert row["worker_variance_preference_override_count"] == 3
    assert row["maximum_worker_matching_deficit"] == 4
    output = tmp_path / "instance_metrics.csv"
    write_csv(output, [row])
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        persisted = next(csv.DictReader(handle))
    assert persisted["production_gate_state_count"] == "7"
    assert persisted["worker_direct_preference_variance_logit_max_abs"] == "0.8"
    assert persisted["future_installation_admission_masked_action_count"] == "6"
