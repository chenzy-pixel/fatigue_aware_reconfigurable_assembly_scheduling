from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agent.baselines import HeuristicPolicy
from agent.ppo import (
    PPOAgent,
    build_actor_critic,
    summarize_policy_decision_diagnostics,
)
from configs import load_config, project_path
from data import load_instance_pickle
from environment import (
    AssemblySchedulingEnv,
    CANONICAL_PREFERENCE,
    DecisionType,
    PreferenceVector,
    simplex_lattice,
)
from environment.env import (
    ReconfigurationRuntime,
    ResourceFeasibilitySnapshot,
    WorkerTaskSnapshot,
)
from environment.types import ReconfigurationStage
from result import (
    aggregate_evaluation_rows,
    aggregate_preference_diagnostics,
    result_schema_version,
)
from train import (
    TrainingPhaseController,
    _checkpoint_eligible_validation_event,
    _pareto_snapshot,
    _preference_key,
    _validation_log_row,
)


def _e2_3_network():
    config = load_config("configs/v7/e2_3_safe_production_preference.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=(0.0, 1.0, 0.0))
    network = build_actor_critic(observation, config["network"])
    return config, environment, observation, network


def test_e2_3_config_is_independent_and_production_only() -> None:
    e2_1 = load_config("configs/v7/e2_1_preference_pareto.json")
    e2_3 = load_config("configs/v7/e2_3_safe_production_preference.json")

    for config in (e2_1, e2_3):
        assert config["training"]["episodes"] == 2000
        assert config["training"]["parallel_envs"] == 20
        assert config["training"]["validation_parallel_envs"] == 20
        assert config["training"]["validation_interval_episodes"] == 20
        assert (
            config["training"]["episodes"]
            // config["training"]["parallel_envs"]
            == 100
        )
    assert e2_3["training"]["smoke_episodes"] == 10
    assert e2_3["training"]["smoke_parallel_envs"] == 10
    assert e2_3["experiment_name"] == "v7_e2_3_safe_production_preference"
    assert e2_3["experiment_suite_version"] == "v7_e2_3_pareto_protocol_v1"
    assert e2_3["network"]["production_action_semantics"] == "pair_plus_defer_v1"
    assert e2_3["network"]["production_commit_set_scorer"] is False
    assert e2_3["network"]["preference_action_score"]["scope"] == (
        "production_only"
    )
    assert e2_1["network"]["preference_action_score"].get("scope", "all") == (
        "all"
    )
    assert e2_3["environment"]["worker_resource_control"]["mode"] == (
        "matching_admission_recovery_v2"
    )
    assert e2_3["training"]["two_stage"]["quality_checkpoint_promotion"] == (
        "pareto_guarded_e2_3_v1"
    )
    assert result_schema_version(e2_3) == "4.4.0"


def test_e2_3_checkpoint_round_trip_and_old_scope_rejection(tmp_path) -> None:
    config, _, observation, network = _e2_3_network()
    assert network.network_spec()["preference_action_score"]["scope"] == (
        "production_only"
    )
    checkpoint = tmp_path / "e2_3.pt"
    PPOAgent(network, config["ppo"], device="cpu").save(checkpoint)

    clone = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device="cpu",
    )
    clone.load(checkpoint)

    for config_path, error in (
        (
            "configs/v7/e2_preference_conditioned.json",
            "preference_action_score",
        ),
        (
            "configs/v7/e2_1_preference_pareto.json",
            "preference_action_score",
        ),
        (
            "configs/v7/e2_2_hierarchical_preference.json",
            "production action semantics",
        ),
    ):
        old_config = load_config(config_path)
        old_agent = PPOAgent(
            build_actor_critic(observation, old_config["network"]),
            old_config["ppo"],
            device="cpu",
        )
        with pytest.raises(ValueError, match=error):
            old_agent.load(checkpoint)


def test_e2_3_flat_ppo_distribution_and_preference_gradient_are_finite() -> None:
    config, environment, observation, network = _e2_3_network()
    action_mask = environment.get_action_mask()
    legal_actions = np.flatnonzero(~action_mask)
    assert len(legal_actions) >= 2

    logits, value = network(observation, action_mask, device="cpu")
    diagnostic = network.consume_policy_decision_diagnostics()[-1]
    distribution = Categorical(logits=logits)
    action = torch.tensor(int(legal_actions[0]))
    log_probability = distribution.log_prob(action)
    ratio = torch.exp(log_probability - log_probability.detach())
    (-log_probability + 0.0 * value).backward()

    assert torch.isfinite(logits[~torch.as_tensor(action_mask)]).all()
    assert torch.isfinite(log_probability)
    assert torch.isfinite(distribution.entropy())
    assert float(ratio.detach()) == pytest.approx(1.0)
    assert network.preference_action_scale_raw.grad is not None
    assert torch.isfinite(network.preference_action_scale_raw.grad)
    summary = summarize_policy_decision_diagnostics([diagnostic])
    assert summary["production_ranker_top_decision_count"] == 1
    assert np.isfinite(summary["production_mean_preference_logit_std"])
    assert config["ppo"]["clip_epsilon"] > 0.0


def test_e2_3_production_direct_preference_changes_candidate_top() -> None:
    _, _, _, network = _e2_3_network()
    objectives = torch.tensor(
        [[1.0, 5.0, 9.0], [5.0, 1.0, 5.0], [9.0, 9.0, 1.0]]
    )
    feasible = torch.ones(3, dtype=torch.bool)
    selected = []
    for preference in torch.eye(3):
        logits = network._direct_preference_logits(
            objectives,
            feasible,
            preference,
        )
        selected.append(int(torch.argmax(logits)))
    assert selected == [0, 1, 2]


def test_worker_forward_has_no_direct_preference_override() -> None:
    _, environment, observation, network = _e2_3_network()
    heuristic = HeuristicPolicy()
    for _ in range(200):
        if observation.decision_type == DecisionType.WORKER:
            action_mask = environment.get_action_mask()
            if np.count_nonzero(~action_mask[:-1]) >= 2:
                network.consume_policy_decision_diagnostics()
                network(observation, action_mask, device="cpu")
                diagnostic = network.consume_policy_decision_diagnostics()[-1]
                assert diagnostic["decision_type"] == "WORKER"
                assert diagnostic["preference_logit_std"] == 0.0
                assert not diagnostic["preference_overrode_relative_top"]
                summary = summarize_policy_decision_diagnostics([diagnostic])
                assert summary["worker_ranker_top_decision_count"] == 1
                assert summary["worker_preference_override_count"] == 0
                return
        action = heuristic.select_action(environment)
        observation, _, terminated, truncated, _ = environment.step(action)
        assert not (terminated or truncated)
    pytest.fail("fixed instance did not expose a multi-pair worker decision")


def test_production_only_scope_has_zero_worker_direct_diagnostics() -> None:
    rows = [
        {
            "decision_type": "production",
            "legal_pair_count": 3,
            "terminal_legal": True,
            "preference_overrode_relative_top": True,
            "preference_logit_std": 0.4,
        },
        {
            "decision_type": "worker",
            "legal_pair_count": 2,
            "terminal_legal": False,
            "preference_overrode_relative_top": False,
            "preference_logit_std": 0.0,
        },
    ]
    diagnostics = summarize_policy_decision_diagnostics(rows)
    assert diagnostics["production_preference_override_count"] == 1
    assert diagnostics["production_preference_override_rate"] == 1.0
    assert diagnostics["worker_preference_override_count"] == 0
    assert diagnostics["worker_preference_override_rate"] == 0.0
    assert diagnostics["worker_mean_preference_logit_std"] == 0.0

    aggregate = aggregate_preference_diagnostics([diagnostics])
    assert aggregate["worker_preference_override_count"] == 0
    assert aggregate["worker_preference_override_rate"] == 0.0


def test_e2_3_result_schema_accepts_split_diagnostics() -> None:
    row = {
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
        "production_ranker_top_decision_count": 4,
        "production_preference_override_count": 2,
        "production_preference_override_rate": 0.5,
        "production_mean_preference_logit_std": 0.3,
        "worker_ranker_top_decision_count": 3,
        "worker_preference_override_count": 0,
        "worker_preference_override_rate": 0.0,
        "worker_mean_preference_logit_std": 0.0,
        "current_worker_matching_deficit": 0,
        "maximum_worker_matching_deficit": 2,
        "deficit_reducing_worker_action_candidate_count": 4,
        "deficit_reducing_worker_action_count": 3,
        "matching_deficit_recovery_advance_count": 1,
        "current_matching_admission_masked_action_count": 5,
        "future_installation_admission_candidate_count": 10,
        "future_installation_admission_masked_action_count": 2,
        "future_installation_admission_masked_action_ratio": 0.2,
        "maximum_projected_installation_deficit": 1,
    }
    aggregate = aggregate_evaluation_rows(
        [row],
        dataset="validation",
        policy="ppo",
        manifest="manifest.json",
        schema_version="4.4.0",
    )
    assert aggregate["evaluation_schema_version"] == "4.4.0"
    assert aggregate["production_preference_override_rate"] == 0.5
    assert aggregate["worker_preference_override_count"] == 0
    assert aggregate["future_installation_admission_masked_action_ratio"] == 0.2
    assert aggregate["future_installation_matching_deficit_after_commit"] == 1
    validation_row = _validation_log_row(
        aggregate,
        completed_episodes=20,
    )
    assert validation_row["maximum_worker_matching_deficit"] == 2
    assert validation_row[
        "deficit_reducing_worker_action_candidate_count"
    ] == 4
    assert validation_row[
        "future_installation_admission_masked_action_ratio"
    ] == 0.2


def _task(task_id: str, worker_edges: tuple[int, ...]) -> tuple[
    WorkerTaskSnapshot, tuple[int, ...]
]:
    return (
        WorkerTaskSnapshot(
            task_id=task_id,
            machine_index=0,
            stage=ReconfigurationStage.WAIT_DIS,
            module="A1",
        ),
        worker_edges,
    )


def _reconfiguration(
    task_id: str,
    machine_id: str = "M1",
) -> ReconfigurationRuntime:
    return ReconfigurationRuntime(
        id=task_id,
        machine_id=machine_id,
        operation_id="operation",
        source_module="A1",
        target_module="A2",
        lock_tick=0,
    )


def test_matching_deficit_lexicographic_rules(monkeypatch) -> None:
    _, environment, _, _ = _e2_3_network()
    zero_tasks = (_task("r0", (0, 1)), _task("r1", (1,)))
    zero_snapshot = ResourceFeasibilitySnapshot(
        tasks=tuple(value[0] for value in zero_tasks),
        safe_edges=tuple(value[1] for value in zero_tasks),
        matching_size=2,
        safe_idle_workers=tuple(range(len(environment.workers))),
        minimum_worker_alternatives=1,
    )
    monkeypatch.setattr(
        environment, "_resource_feasibility_snapshot", lambda: zero_snapshot
    )
    assert environment._worker_action_matching_deficits(
        _reconfiguration("r0"), 0
    ) == (0, 0)
    assert environment._worker_action_matching_deficits(
        _reconfiguration("r0"), 1
    ) == (0, 1)

    deficient_tasks = (
        _task("r0", (0,)),
        _task("r1", (0,)),
        _task("r2", (1,)),
    )
    deficient_snapshot = ResourceFeasibilitySnapshot(
        tasks=tuple(value[0] for value in deficient_tasks),
        safe_edges=tuple(value[1] for value in deficient_tasks),
        matching_size=2,
        safe_idle_workers=tuple(range(len(environment.workers))),
        minimum_worker_alternatives=1,
    )
    monkeypatch.setattr(
        environment,
        "_resource_feasibility_snapshot",
        lambda: deficient_snapshot,
    )
    before, after = environment._worker_action_matching_deficits(
        _reconfiguration("r0"), 0
    )
    assert before == 1
    assert after == 0


def test_matching_recovery_mask_prioritizes_pair_or_advances(
    monkeypatch,
) -> None:
    _, environment, _, _ = _e2_3_network()
    machine_ids = [machine.spec.id for machine in environment.machines[:3]]

    def install_snapshot(edge_map: dict[str, tuple[int, ...]]) -> None:
        environment.reconfigurations = {
            task_id: _reconfiguration(task_id, machine_ids[index])
            for index, task_id in enumerate(edge_map)
        }
        environment._machine_reconfiguration = {
            machine_ids[index]: task_id
            for index, task_id in enumerate(edge_map)
        }
        tasks = tuple(
            _task(task_id, edges)[0]
            for task_id, edges in edge_map.items()
        )
        safe_edges = tuple(edge_map.values())
        snapshot = ResourceFeasibilitySnapshot(
            tasks=tasks,
            safe_edges=safe_edges,
            matching_size=2 if len(edge_map) == 3 else 1,
            safe_idle_workers=tuple(range(len(environment.workers))),
            minimum_worker_alternatives=0,
        )
        monkeypatch.setattr(
            environment,
            "_resource_feasibility_snapshot",
            lambda: snapshot,
        )
        worker_indices = {
            worker.spec.id: index
            for index, worker in enumerate(environment.workers)
        }
        monkeypatch.setattr(
            environment,
            "_worker_can_start",
            lambda reconfiguration, worker: (
                worker_indices[worker.spec.id] in edge_map[reconfiguration.id]
            ),
        )
        monkeypatch.setattr(environment, "_has_strict_future", lambda: True)
        environment.decision_type = DecisionType.WORKER

    install_snapshot({"r0": (0,), "r1": (0,), "r2": (1,)})
    recovery_mask = environment.get_action_mask()
    assert np.count_nonzero(~recovery_mask[:-1]) == 2
    assert recovery_mask[-1]

    install_snapshot({"r0": (), "r1": (0,)})
    advance_mask = environment.get_action_mask()
    assert np.count_nonzero(~advance_mask[:-1]) == 0
    assert not advance_mask[-1]


def test_future_installation_matching_is_joint_and_v2_only(monkeypatch) -> None:
    _, environment, _, _ = _e2_3_network()
    machine_ids = [machine.spec.id for machine in environment.machines[:3]]
    environment.reconfigurations = {
        f"r{index}": ReconfigurationRuntime(
            id=f"r{index}",
            machine_id=machine_id,
            operation_id=f"o{index}",
            source_module="A1",
            target_module="A2",
            lock_tick=0,
            stage=ReconfigurationStage.WAIT_INS,
        )
        for index, machine_id in enumerate(machine_ids)
    }

    def projection(
        machine_index,
        worker_index,
        module,
        *,
        installation,
        earliest_tick,
    ):
        if installation and module == "A2" and worker_index < 3:
            return earliest_tick, 1
        return None

    monkeypatch.setattr(
        environment, "_earliest_safe_stage_projection", projection
    )
    deficit, candidate_workers = (
        environment._projected_future_installation_matching(
            candidate_machine_index=3,
            candidate_target_module="A2",
            candidate_installation_ready_tick=0,
        )
    )
    assert candidate_workers == 3
    assert deficit == 1

    environment.reconfigurations.pop("r2")
    deficit, _ = environment._projected_future_installation_matching(
        candidate_machine_index=3,
        candidate_target_module="A2",
        candidate_installation_ready_tick=0,
    )
    assert deficit == 0

    old_config = load_config("configs/v7/e2_1_preference_pareto.json")
    old_environment = AssemblySchedulingEnv(old_config)
    assert not old_environment.matching_recovery_enabled


def _full_grid_rows(
    *,
    trace_count: int = 8,
    objective_count: int = 22,
    dominated: bool = False,
) -> tuple[list[dict], tuple[str, ...], tuple[str, ...]]:
    preferences = tuple(simplex_lattice(5, include=(CANONICAL_PREFERENCE,)))
    instance_ids = tuple(f"validation-{index}" for index in range(20))
    rows: list[dict] = []
    for instance_id in instance_ids:
        for index, preference in enumerate(preferences):
            objective_index = index % objective_count
            objectives = (
                (100.0 + objective_index,) * 3
                if dominated
                else (
                    100.0 + objective_index,
                    200.0 - objective_index,
                    10.0,
                )
            )
            rows.append(
                {
                    "instance_id": instance_id,
                    "terminated": True,
                    "truncated": False,
                    "schedule_violation_count": 0,
                    "maximum_worker_fatigue": 0.5,
                    "safe_fatigue_limit": 0.75,
                    "flow_time_objective": objectives[0],
                    "reconfiguration_cost": objectives[1],
                    "worker_load_variance": objectives[2],
                    "preference_quality_score": 0.25,
                    "preference_key": _preference_key(preference),
                    "w_flow": preference.flow,
                    "w_cost": preference.cost,
                    "w_variance": preference.variance,
                    "action_trace_sha256": f"trace-{index % trace_count}",
                }
            )
    return (
        rows,
        instance_ids,
        tuple(_preference_key(preference) for preference in preferences),
    )


def _snapshot(rows, instance_ids, preference_keys):
    config = load_config("configs/v7/e2_3_safe_production_preference.json")
    return _pareto_snapshot(
        rows,
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=instance_ids,
        expected_preference_keys=preference_keys,
    )


def test_e2_3_full_grid_coverage_controllability_and_acceptance() -> None:
    rows, instance_ids, preference_keys = _full_grid_rows()
    snapshot = _snapshot(rows, instance_ids, preference_keys)
    assert snapshot["candidate_count"] == 440
    assert snapshot["completion_rate"] == 1.0
    assert snapshot["coverage_pass"] is True
    assert snapshot["controllability_pass"] is True
    assert snapshot["worker_direct_preference_pass"] is True
    assert snapshot["mean_unique_action_trace_count"] == 8.0
    assert snapshot["mean_unique_objective_count"] == 22.0
    assert snapshot["mean_nondominated_count"] == 22.0
    for name in (
        "preference_response_spearman_flow",
        "preference_response_spearman_cost",
        "preference_response_spearman_variance",
        "future_installation_admission_masked_action_ratio",
        "future_installation_matching_deficit_after_commit",
    ):
        assert np.isfinite(snapshot[name])

    config = load_config("configs/v7/e2_3_safe_production_preference.json")
    controller = TrainingPhaseController.from_config(config)
    controller.phase = "quality"
    assert controller.observe_pareto_snapshot(
        snapshot, completed_episodes=200
    ) == "accepted"

    anchor_snapshot = deepcopy(snapshot)
    anchor_snapshot["scope"] = "anchors_5"
    fresh = TrainingPhaseController.from_config(config)
    fresh.phase = "quality"
    assert fresh.observe_pareto_snapshot(
        anchor_snapshot, completed_episodes=200
    ) == "rejected"
    assert fresh.last_promotion_diagnostics["promotion_coverage_pass"] is False

    assert not _checkpoint_eligible_validation_event(
        "transition", "pareto_guarded_e2_3_v1"
    )
    assert _checkpoint_eligible_validation_event(
        "accepted", "pareto_guarded_e2_3_v1"
    )


def test_e2_3_controllability_threshold_boundaries_pass() -> None:
    rows, instance_ids, preference_keys = _full_grid_rows(objective_count=8)
    snapshot = _snapshot(rows, instance_ids, preference_keys)
    assert snapshot["mean_unique_action_trace_count"] == 8.0
    assert snapshot["mean_unique_objective_count"] == 8.0
    assert snapshot["controllability_pass"] is True

    rows, instance_ids, preference_keys = _full_grid_rows()
    for offset, row in enumerate(rows):
        preference_index = offset % 22
        if preference_index < 4:
            objectives = (
                100.0 + preference_index,
                200.0 - preference_index,
                10.0,
            )
        else:
            objectives = (
                300.0 + preference_index,
                300.0 + preference_index,
                30.0,
            )
        row["flow_time_objective"] = objectives[0]
        row["reconfiguration_cost"] = objectives[1]
        row["worker_load_variance"] = objectives[2]
    snapshot = _snapshot(rows, instance_ids, preference_keys)
    assert snapshot["mean_nondominated_count"] == 4.0
    assert snapshot["controllability_pass"] is True


def test_e2_3_rejects_nonfinite_canonical_quality() -> None:
    rows, instance_ids, preference_keys = _full_grid_rows()
    canonical_key = _preference_key(PreferenceVector(*CANONICAL_PREFERENCE))
    for row in rows:
        if row["preference_key"] == canonical_key:
            row["preference_quality_score"] = math.inf
    snapshot = _snapshot(rows, instance_ids, preference_keys)
    assert math.isinf(snapshot["canonical_quality"])

    config = load_config("configs/v7/e2_3_safe_production_preference.json")
    controller = TrainingPhaseController.from_config(config)
    controller.phase = "quality"
    assert controller.observe_pareto_snapshot(
        snapshot, completed_episodes=200
    ) == "not_promoted"


@pytest.mark.parametrize(
    ("mutation", "failed_field"),
    (
        ("missing", "coverage_pass"),
        ("duplicate", "coverage_pass"),
        ("trace", "unique_action_trace_pass"),
        ("objective", "unique_objective_pass"),
        ("nondominated", "nondominated_pass"),
        ("worker_direct", "worker_direct_preference_pass"),
        ("unsafe", "all_safe"),
    ),
)
def test_e2_3_full_grid_rejects_each_failed_gate(mutation, failed_field) -> None:
    if mutation == "trace":
        rows, instance_ids, preference_keys = _full_grid_rows(trace_count=7)
    elif mutation == "objective":
        rows, instance_ids, preference_keys = _full_grid_rows(objective_count=7)
    elif mutation == "nondominated":
        rows, instance_ids, preference_keys = _full_grid_rows(dominated=True)
    else:
        rows, instance_ids, preference_keys = _full_grid_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[-1]))
    elif mutation == "unsafe":
        rows[0]["truncated"] = True
        rows[0]["terminated"] = False
    elif mutation == "worker_direct":
        rows[0]["worker_ranker_top_decision_count"] = 1
        rows[0]["worker_preference_override_count"] = 1
        rows[0]["worker_preference_override_rate"] = 1.0
        rows[0]["worker_mean_preference_logit_std"] = 0.2

    snapshot = _snapshot(rows, instance_ids, preference_keys)
    assert snapshot[failed_field] is False
    config = load_config("configs/v7/e2_3_safe_production_preference.json")
    controller = TrainingPhaseController.from_config(config)
    controller.phase = "quality"
    assert controller.observe_pareto_snapshot(
        snapshot, completed_episodes=200
    ) == "rejected"
