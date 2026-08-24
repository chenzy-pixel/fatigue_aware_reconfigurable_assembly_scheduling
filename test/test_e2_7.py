from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from environment import AssemblySchedulingEnv, PreferenceVector, RewardVector
from result import aggregate_evaluation_rows, result_schema_version
from train import (
    TrainingPhaseController,
    E2_7PreferenceStageController,
    _accepted_checkpoint_path,
    _e2_3_failure_replay_cells,
    _pareto_anchor_preferences,
    _pareto_promotion_settings,
    _pareto_snapshot,
    _restore_e2_7_resume_provenance,
)


CONFIG_PATH = "configs/v7/e2_7_e1_warmstart_safe_gate_v1.json"
E1_CHECKPOINT = "result/runs/v7_2000_e1_seed11/accepted_checkpoint.pt"


def _fixture():
    config = load_config(CONFIG_PATH)
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=(0.5, 0.3, 0.2))
    network = build_actor_critic(observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device="cpu")
    report = agent.warm_start_from_e1(
        project_path(E1_CHECKPOINT), expected_shared_parameter_count=116
    )
    report = agent.verify_warm_start_canonical_identity(
        observation,
        environment.get_action_mask(),
        source_checkpoint=project_path(E1_CHECKPOINT),
    )
    return config, environment, observation, network, agent, report


def _single_transition_buffer(agent, observation, mask) -> RolloutBuffer:
    action, log_probability, value = agent.act(
        observation, mask, deterministic=True
    )
    buffer = RolloutBuffer(preserve_graph=True)
    buffer.add(
        observation,
        mask,
        action,
        log_probability,
        value,
        0.0,
        True,
    )
    buffer.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    return buffer


def _force_counterfactual_constraint_satisfied(network) -> None:
    with torch.no_grad():
        network.centered_gate_coefficients[-1].weight.zero_()
        network.centered_gate_coefficients[-1].bias.fill_(1.0)


def test_e2_7_warm_start_is_complete_fresh_and_canonically_identical() -> None:
    config, environment, observation, network, agent, report = _fixture()
    checkpoint = torch.load(
        project_path(E1_CHECKPOINT), map_location="cpu", weights_only=False
    )
    old_network = build_actor_critic(
        observation, checkpoint["metadata"]["effective_config"]["network"]
    )
    old_network.load_state_dict(checkpoint["network"], strict=True)
    old_network.eval()
    network.eval()
    mask = environment.get_action_mask()
    old_logits, old_value = old_network.forward_batch(
        [observation], [mask], device="cpu"
    )
    new_logits, new_value = network.forward_batch(
        [observation], [mask], device="cpu"
    )
    action_count = len(mask)
    assert torch.equal(old_logits[0, :action_count], new_logits[0, :action_count])
    assert torch.equal(
        torch.softmax(old_logits[0, :action_count], dim=0),
        torch.softmax(new_logits[0, :action_count], dim=0),
    )
    assert torch.equal(old_value, new_value)
    assert report["loaded_shared_parameter_count"] == 116
    assert report["new_parameter_count"] == 10
    assert len(report["source_checkpoint_sha256"]) == 64
    assert report["optimizer_restored"] is False
    assert report["optimizer_state_entry_count"] == 0
    assert len(agent.optimizer.state) == 0
    assert not any("preference_encoder" in key for key in network.state_dict())
    assert report["canonical_identity_result"]["pass"]
    assert report["canonical_identity_result"]["raw_logits_max_abs_error"] == 0.0


def test_e2_7_logsumexp_gate_is_flat_e1_equivalent_and_monotone() -> None:
    _, environment, observation, network, _, _ = _fixture()
    mask = torch.as_tensor(environment.get_action_mask(), dtype=torch.bool)
    pair_logits = torch.linspace(-1.5, 1.5, mask.numel() - 1)
    defer_logit = torch.tensor(0.25)
    canonical = torch.tensor((0.5, 0.3, 0.2))
    canonical_logits, diagnostic = network._centered_production_gate(
        observation,
        pair_logits,
        defer_logit,
        canonical,
        mask,
        device="cpu",
    )
    assert torch.equal(
        canonical_logits, torch.cat((pair_logits, defer_logit.reshape(1)))
    )
    legal = ~mask[:-1]
    assert diagnostic["base_commit_logit"] == pytest.approx(
        torch.logsumexp(pair_logits[legal], dim=0).item()
    )
    with torch.no_grad():
        network.centered_gate_coefficients[-1].bias.fill_(1.0)
    margins = {}
    for name, preference in {
        "flow": (1.0, 0.0, 0.0),
        "canonical": (0.5, 0.3, 0.2),
        "cost": (0.0, 1.0, 0.0),
        "variance": (0.0, 0.0, 1.0),
    }.items():
        _, values = network._centered_production_gate(
            observation,
            pair_logits,
            defer_logit,
            torch.tensor(preference),
            mask,
            device="cpu",
        )
        margins[name] = float(values["final_margin"])
        assert abs(float(values["residual"])) <= 3.0 + 1e-7
    assert margins["flow"] >= margins["canonical"]
    assert margins["cost"] <= margins["canonical"]
    assert margins["variance"] <= margins["canonical"]


def test_e2_7_stage_freezing_keeps_gnn_frozen() -> None:
    config, environment, observation, network, _, _ = _fixture()
    controller = E2_7PreferenceStageController.from_config(config)
    assert controller is not None
    assert controller.stage == "gate"
    network.set_centered_preference_stage("gate")
    gnn_prefixes = (
        "node_projectors.",
        "global_encoder.",
        "message_layers.",
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in network.named_parameters()
        if name.startswith(gnn_prefixes)
    )
    assert all(
        not parameter.requires_grad
        for parameter in network.production_scorer.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in network.centered_gate_coefficients.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in network.centered_value_adapter.parameters()
    )
    network.zero_grad(set_to_none=True)
    logits, values = network.forward_batch(
        [observation], [environment.get_action_mask()], device="cpu"
    )
    (logits[torch.isfinite(logits)].sum() + values.sum()).backward()
    assert all(
        parameter.grad is None
        for name, parameter in network.named_parameters()
        if name.startswith(gnn_prefixes)
    )
    assert all(
        parameter.grad is None
        for parameter in network.production_scorer.parameters()
    )
    network.set_centered_preference_stage("production_pair")
    assert all(
        not parameter.requires_grad
        for parameter in network.production_scorer.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in network.centered_value_adapter.parameters()
    )
    assert network.centered_production_pair_scale_raw.requires_grad
    assert all(
        not parameter.requires_grad for parameter in network.worker_scorer.parameters()
    )
    network.set_centered_preference_stage("worker_variance")
    assert all(
        not parameter.requires_grad for parameter in network.worker_scorer.parameters()
    )
    assert network.centered_worker_variance_scale_raw.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in network.named_parameters()
        if name.startswith(gnn_prefixes)
    )


def test_e2_7_pair_and_worker_scales_have_live_zero_boundary_gradients(
    monkeypatch,
) -> None:
    _, environment, observation, network, agent, _ = _fixture()
    mask = environment.get_action_mask()
    pool = [(observation, mask) for _ in range(64)]
    eligible = torch.ones(64, dtype=torch.bool)
    incorrect = torch.zeros(64, dtype=torch.bool)

    def pair_diagnostics(observations, *_args, **_kwargs):
        scale = network.centered_production_pair_scale().expand(len(observations))
        return {
            "flow_gain": scale,
            "cost_gain": scale,
            "flow_margin": scale,
            "cost_margin": scale,
            "eligible": eligible[: len(observations)],
            "flow_correct": incorrect[: len(observations)],
            "cost_correct": incorrect[: len(observations)],
        }

    def worker_diagnostics(observations, *_args, **_kwargs):
        scale = network.centered_worker_variance_scale().expand(len(observations))
        return {
            "variance_gain": scale,
            "variance_margin": scale,
            "eligible": eligible[: len(observations)],
            "variance_correct": incorrect[: len(observations)],
        }

    monkeypatch.setattr(
        network,
        "centered_production_pair_counterfactual_batch",
        pair_diagnostics,
    )
    monkeypatch.setattr(
        network,
        "centered_worker_variance_counterfactual_batch",
        worker_diagnostics,
    )

    network.set_centered_preference_stage("production_pair")
    agent.centered_state_pools["production_pair"] = pool
    pair_loss, _ = agent._centered_pool_objective("production_pair")
    pair_loss.backward()
    assert torch.isfinite(network.centered_production_pair_scale_raw.grad)
    assert network.centered_production_pair_scale_raw.grad.abs().item() > 0.0

    network.zero_grad(set_to_none=True)
    network.set_centered_preference_stage("worker_variance")
    agent.centered_state_pools["worker_variance"] = pool
    worker_loss, _ = agent._centered_pool_objective("worker_variance")
    worker_loss.backward()
    assert torch.isfinite(network.centered_worker_variance_scale_raw.grad)
    assert network.centered_worker_variance_scale_raw.grad.abs().item() > 0.0


def test_e2_7_metric_stage_controller_transitions_and_fails_fast() -> None:
    config = load_config(CONFIG_PATH)
    controller = E2_7PreferenceStageController.from_config(config)
    assert controller is not None
    gate_pass = {
        "counterfactual_constraint_status": "constraint_satisfied",
        "counterfactual_eligible_count": 64.0,
        "counterfactual_flow_cost_flip_count": 4.0,
        "counterfactual_flow_variance_flip_count": 4.0,
    }
    for update_id in range(1, 11):
        controller.observe(gate_pass, update_id=update_id)
    assert controller.stage == "production_pair"
    assert controller.transition_history[-1]["update_id"] == 10

    pair_pass = {
        "production_pair_constraint_status": "constraint_satisfied",
        "production_pair_correct_rate": 0.05,
    }
    for update_id in range(11, 31):
        controller.observe(pair_pass, update_id=update_id)
    assert controller.stage == "worker_variance"
    saved = controller.as_dict()
    restored = E2_7PreferenceStageController.from_config(config)
    assert restored is not None
    restored.restore(saved)
    assert restored.as_dict() == saved

    failing = E2_7PreferenceStageController.from_config(config)
    assert failing is not None
    with pytest.raises(RuntimeError, match="preference_stage_failed"):
        for update_id in range(1, 41):
            failing.observe({}, update_id=update_id)


def test_e2_7_fixed_safe_pool_and_shield_boundaries() -> None:
    config, environment, observation, _network, agent, _report = _fixture()
    mask = environment.get_action_mask()
    pool = [(observation, mask) for _ in range(64)]
    report = agent.set_safe_dual_legal_state_pool(
        pool, provenance={"test": True}
    )
    assert report["state_count"] == 64
    assert report["fixed_sampling_count_per_auxiliary_update"] == 64
    assert len(report["state_pool_sha256"]) == 64
    with pytest.raises(RuntimeError, match="too small"):
        agent.set_safe_dual_legal_state_pool(
            pool[:-1], provenance={"test": "insufficient"}
        )

    baseline_shield = environment.metrics()
    environment._state_version += 1
    no_progress = environment._production_defer_safety_certificate(1, None)
    assert not no_progress["allowed"]
    assert no_progress["reason"] == "no_state_progress"
    shield_metrics = environment.metrics()
    assert shield_metrics["production_defer_shield_candidate_count"] == (
        baseline_shield["production_defer_shield_candidate_count"] + 1
    )
    assert shield_metrics["production_defer_shield_masked_count"] == (
        baseline_shield["production_defer_shield_masked_count"] + 1
    )
    assert shield_metrics["production_defer_shield_reason_counts"] == {
        "no_state_progress": 1
    }
    assert shield_metrics["production_defer_shield_max_risk"] == 1.0
    assert (
        shield_metrics["production_defer_shield_max_work_lower_bound_ticks"]
        == no_progress["remaining_work_lower_bound_ticks"]
    )
    handoff = environment._production_defer_safety_certificate(
        1, (environment.current_tick, "worker_phase_handoff")
    )
    assert handoff["allowed"]
    assert handoff["reason"] == "zero_time_worker_handoff"
    only_defer = environment._production_defer_safety_certificate(
        0, (environment.current_tick + 1, "external_event")
    )
    assert only_defer["allowed"]
    assert only_defer["reason"] == "certified_progress_with_budget"
    terminal_only_defer = environment._production_defer_safety_certificate(0, None)
    assert not terminal_only_defer["allowed"]
    assert terminal_only_defer["reason"] == "unrecoverable_deadlock"
    deadline = environment._production_defer_safety_certificate(
        1, (environment.horizon_tick, "external_event")
    )
    assert not deadline["allowed"]
    assert deadline["reason"] == "completion_viability_exceeded"
    before = len(environment._production_defer_shield_candidates)
    environment._production_defer_safety_certificate(
        1, (environment.horizon_tick, "external_event")
    )
    assert len(environment._production_defer_shield_candidates) == before

    reward = RewardVector(
        flow=-1.0,
        cost=-2.0,
        variance=-3.0,
        completion_progress=0.25,
        completion_bonus=0.0,
        quality=-0.5,
        truncation=0.0,
        unfinished=0.0,
        feasibility_shaping=0.125,
        defer_risk_shaping=-0.03125,
    )
    total = reward.scalarize(config=config["reward"], phase="quality")
    base = reward.base_scalarize(config=config["reward"], phase="quality")
    assert total - reward.feasibility_shaping - reward.defer_risk_shaping == pytest.approx(
        base, abs=1e-8
    )


def test_e2_7_v1_shield_keeps_legacy_only_defer_behavior() -> None:
    config = load_config(CONFIG_PATH)
    config["environment"]["production_defer"]["shield"]["version"] = (
        "deadline_progress_shield_v1"
    )
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    certificate = environment._production_defer_safety_certificate(
        0,
        (environment.current_tick + 1, "external_event"),
    )
    assert certificate["allowed"]
    assert certificate["reason"] == "only_defer_legal"


def test_e2_7_viability_shield_masks_direct_pair_with_infeasible_suffix(
    monkeypatch,
) -> None:
    config = load_config(CONFIG_PATH)
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    direct_pair = next(
        (operation_index, machine_index)
        for operation_index, operation in enumerate(environment.operations)
        if operation.state.value == "READY"
        for machine_index, machine in enumerate(environment.machines)
        if (
            machine.current_module == operation.spec.required_module
            and machine.state.value == "IDLE"
        )
    )
    profile = environment._production_candidate_profile(*direct_pair)
    assert profile.predicted_finish_tick <= environment.horizon_tick
    monkeypatch.setattr(
        environment,
        "_candidate_completion_lower_bound_ticks",
        lambda *_args, **_kwargs: environment.horizon_tick + 1,
    )
    environment._invalidate_resource_snapshot()
    mask = environment.get_action_mask()
    assert mask[environment.encode_production_action(*direct_pair)]


def test_e2_7_fixed_pool_has_nonzero_loss_and_gate_gradient_before_ppo() -> None:
    _, environment, observation, network, agent, _ = _fixture()
    mask = environment.get_action_mask()
    shared_before = {
        name: value.detach().clone()
        for name, value in network.state_dict().items()
        if not name.startswith("centered_")
    }
    agent.set_safe_dual_legal_state_pool(
        [(observation, mask) for _ in range(64)],
        provenance={"test": True},
    )
    buffer = _single_transition_buffer(agent, observation, mask)
    metrics = agent.update(buffer, reward_phase="feasibility")
    assert metrics["counterfactual_eligible_count"] >= 64.0
    assert metrics["counterfactual_loss"] > 0.0
    assert metrics["counterfactual_constraint_status"] == "constraint_active"
    assert metrics["counterfactual_monotonicity_violation_count"] == 0.0
    assert agent._safe_pool_gradient_preflight_complete
    assert all(
        torch.equal(value, network.state_dict()[name])
        for name, value in shared_before.items()
    )


def test_e2_7_zero_loss_is_valid_on_first_preflight() -> None:
    _, environment, observation, network, agent, _ = _fixture()
    mask = environment.get_action_mask()
    agent.set_safe_dual_legal_state_pool(
        [(observation, mask) for _ in range(64)],
        provenance={"test": True},
    )
    _force_counterfactual_constraint_satisfied(network)

    metrics = agent.update(
        _single_transition_buffer(agent, observation, mask),
        reward_phase="feasibility",
    )

    assert metrics["counterfactual_loss"] == 0.0
    assert (
        metrics["counterfactual_constraint_status"]
        == "constraint_satisfied"
    )
    assert agent._safe_pool_gradient_preflight_complete


def test_e2_7_preflight_runs_once_and_later_zero_loss_is_valid(
    monkeypatch,
) -> None:
    _, environment, observation, network, agent, _ = _fixture()
    mask = environment.get_action_mask()
    agent.set_safe_dual_legal_state_pool(
        [(observation, mask) for _ in range(64)],
        provenance={"test": True},
    )
    preflight_count = 0
    original_preflight = agent._run_centered_pool_gradient_preflight

    def counting_preflight(name: str) -> None:
        nonlocal preflight_count
        preflight_count += 1
        original_preflight(name)

    monkeypatch.setattr(
        agent,
        "_run_centered_pool_gradient_preflight",
        counting_preflight,
    )
    first_metrics = agent.update(
        _single_transition_buffer(agent, observation, mask),
        reward_phase="feasibility",
    )
    assert first_metrics["counterfactual_constraint_status"] == "constraint_active"

    _force_counterfactual_constraint_satisfied(network)
    second_metrics = agent.update(
        _single_transition_buffer(agent, observation, mask),
        reward_phase="feasibility",
    )

    assert preflight_count == 1
    assert second_metrics["counterfactual_loss"] == 0.0
    assert (
        second_metrics["counterfactual_constraint_status"]
        == "constraint_satisfied"
    )

    agent.set_safe_dual_legal_state_pool(
        [(observation, mask) for _ in range(64)],
        provenance={"test": "replacement"},
    )
    assert not agent._safe_pool_gradient_preflight_complete
    third_metrics = agent.update(
        _single_transition_buffer(agent, observation, mask),
        reward_phase="feasibility",
    )
    assert preflight_count == 2
    assert (
        third_metrics["counterfactual_constraint_status"]
        == "constraint_satisfied"
    )


def test_e2_7_preflight_preserves_fail_fast_contracts(monkeypatch) -> None:
    _, environment, observation, _, agent, _ = _fixture()
    mask = environment.get_action_mask()
    buffer = _single_transition_buffer(agent, observation, mask)
    with pytest.raises(RuntimeError, match="not initialized"):
        agent.update(buffer, reward_phase="feasibility")

    agent.set_safe_dual_legal_state_pool(
        [(observation, mask) for _ in range(64)],
        provenance={"test": True},
    )

    def non_finite_objective(*args, **kwargs):
        return torch.full(
            (),
            float("nan"),
            device=agent.device,
            requires_grad=True,
        ), {}

    monkeypatch.setattr(
        agent,
        "_centered_pool_objective",
        non_finite_objective,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        agent.update(buffer, reward_phase="feasibility")
    assert not agent._safe_pool_gradient_preflight_complete

    def detached_positive_objective(*args, **kwargs):
        return torch.ones((), device=agent.device, requires_grad=True), {}

    monkeypatch.setattr(
        agent,
        "_centered_pool_objective",
        detached_positive_objective,
    )
    with pytest.raises(RuntimeError, match="adapter gradient"):
        agent.update(buffer, reward_phase="feasibility")
    assert not agent._safe_pool_gradient_preflight_complete


def test_e2_7_ungrouped_training_has_valid_pareto_anchors() -> None:
    config = load_config(CONFIG_PATH)
    assert config["training"]["preference_grouping"]["enabled"] is False
    anchors = _pareto_anchor_preferences(config)
    assert tuple(anchor.as_tuple() for anchor in anchors) == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.5, 0.3, 0.2),
    )
    invalid = deepcopy(config)
    invalid["training"]["two_stage"]["pareto_promotion"][
        "maximum_canonical_heuristic_relative_gap"
    ] = -1.01
    with pytest.raises(
        ValueError,
        match="maximum_canonical_heuristic_relative_gap",
    ):
        _pareto_promotion_settings(invalid)


def test_e2_7_gate_flip_requires_both_extreme_contrasts() -> None:
    config = load_config(CONFIG_PATH)
    settings = config["training"]["two_stage"]["pareto_promotion"]
    settings["required_full_grid_instance_count"] = 1
    settings["required_full_grid_preference_count"] = 1
    settings["required_full_grid_candidate_count"] = 1
    settings["minimum_mean_unique_action_trace_count"] = 1.0
    settings["minimum_mean_unique_objective_count"] = 1.0
    settings["minimum_mean_nondominated_count"] = 1.0
    row = {
        "instance_id": "unit",
        "preference_key": "0.5_0.3_0.2",
        "w_flow": 0.5,
        "w_cost": 0.3,
        "w_variance": 0.2,
        "terminated": True,
        "truncated": False,
        "schedule_violation_count": 0,
        "maximum_worker_fatigue": 0.1,
        "safe_fatigue_limit": 0.75,
        "flow_time_objective": 1.0,
        "reconfiguration_cost": 1.0,
        "worker_load_variance": 1.0,
        "preference_quality_score": 0.9,
        "heuristic_quality_score": 1.0,
        "action_trace_sha256": "trace",
        "counterfactual_constraint_status": "constraint_satisfied",
        "centered_gate_dual_legal_state_count": 100,
        "centered_gate_flow_cost_flip_count": 0,
        "centered_gate_flow_variance_flip_count": 0,
        "centered_gate_monotonicity_violation_count": 0,
    }
    no_flip = _pareto_snapshot(
        [row],
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=("unit",),
        expected_preference_keys=("0.5_0.3_0.2",),
    )
    assert no_flip["centered_gate_extreme_flip_rate"] == 0.0
    assert not no_flip["centered_gate_pass"]

    row["centered_gate_flow_cost_flip_count"] = 5
    row["centered_gate_flow_variance_flip_count"] = 4
    failed = _pareto_snapshot(
        [row],
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=("unit",),
        expected_preference_keys=("0.5_0.3_0.2",),
    )
    assert failed["centered_gate_flow_cost_flip_rate"] == pytest.approx(0.05)
    assert failed["centered_gate_flow_variance_flip_rate"] == pytest.approx(0.04)
    assert failed["centered_gate_extreme_flip_rate"] == pytest.approx(0.04)
    assert not failed["centered_gate_pass"]
    row["centered_gate_flow_variance_flip_count"] = 5
    passed = _pareto_snapshot(
        [row],
        config=config,
        scope="full_grid_22",
        update_id=20,
        completed_episodes=200,
        fatigue_tolerance=1e-9,
        expected_instance_ids=("unit",),
        expected_preference_keys=("0.5_0.3_0.2",),
    )
    assert passed["centered_gate_extreme_flip_rate"] == pytest.approx(0.05)
    assert passed["centered_gate_pass"]


def _passing_development_snapshot() -> dict[str, object]:
    return {
        "scope": "full_grid_22",
        "mean_hypervolume": 0.5,
        "canonical_quality": 0.32,
        "all_safe": True,
        "coverage_pass": True,
        "controllability_pass": True,
        "worker_direct_preference_pass": True,
        "preference_response_pass": True,
        "low_flow_safety_pass": True,
        "centered_gate_pass": True,
        "e2_3_failure_replay_pass": True,
        "canonical_development_quality_pass": True,
        "heldout_hv_pass": True,
    }


def test_e2_7_development_acceptance_requires_two_full_grids_and_new_name(tmp_path) -> None:
    config = load_config(CONFIG_PATH)
    controller = TrainingPhaseController.from_config(config)
    first = controller.observe_pareto_snapshot(
        _passing_development_snapshot(), completed_episodes=1800
    )
    assert first == "not_promoted"
    assert (
        controller.last_promotion_diagnostics["promotion_decision_reason"]
        == "awaiting_second_consecutive_full_grid"
    )
    second = controller.observe_pareto_snapshot(
        _passing_development_snapshot(), completed_episodes=2000
    )
    assert second == "promoted"
    assert controller.development_consecutive_full_grid_passes == 2
    assert _accepted_checkpoint_path(tmp_path, config).name == (
        "development_accepted_pareto_checkpoint.pt"
    )
    assert len(_e2_3_failure_replay_cells(config)) == 10


def test_e2_7_schema_and_old_checkpoint_loading_regression() -> None:
    config, _, _, _, _, _ = _fixture()
    assert result_schema_version(config) == "4.8.0"
    aggregate = aggregate_evaluation_rows(
        [],
        dataset="validation",
        policy="ppo",
        manifest="validation/manifest.json",
        schema_version="4.8.0",
    )
    assert aggregate["evaluation_schema_version"] == "4.8.0"
    e2_3_checkpoint = torch.load(
        project_path(
            "result/runs/v7_2000_e2_3_safe_production_seed11/last_checkpoint.pt"
        ),
        map_location="cpu",
        weights_only=False,
    )
    old_config = deepcopy(e2_3_checkpoint["metadata"]["effective_config"])
    instance = load_instance_pickle(project_path(old_config["paths"]["instance_cache"]))
    observation = AssemblySchedulingEnv(old_config).reset(instance)
    old_network = build_actor_critic(observation, old_config["network"])
    old_network.load_state_dict(e2_3_checkpoint["network"], strict=True)


def test_e2_7_v1_checkpoint_is_rejected_by_v2_strict_resume(tmp_path) -> None:
    _, environment, observation, _, agent, _ = _fixture()
    checkpoint_path = tmp_path / "simulated_e2_7_v1.pt"
    agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["network_spec"]["centered_preference_adapter"]["version"] = (
        "centered_parallel_adapter_v1"
    )
    torch.save(checkpoint, checkpoint_path)

    resumed_network = build_actor_critic(
        observation,
        load_config(CONFIG_PATH)["network"],
    )
    resumed_agent = PPOAgent(resumed_network, load_config(CONFIG_PATH)["ppo"], device="cpu")
    with pytest.raises(ValueError, match="centered_preference_adapter"):
        resumed_agent.load(checkpoint_path, load_optimizer=True)
    assert environment.get_action_mask().shape[0] > 1


def test_e2_7_strict_resume_restores_warm_provenance_and_teacher(tmp_path) -> None:
    config, environment, observation, _, agent, report = _fixture()
    checkpoint = tmp_path / "stage_checkpoint.pt"
    agent.save(checkpoint, metadata={"checkpoint_role": "stage"})
    resumed_network = build_actor_critic(observation, config["network"])
    resumed_agent = PPOAgent(resumed_network, config["ppo"], device="cpu")
    metadata = resumed_agent.load(checkpoint, load_optimizer=True)
    restored = _restore_e2_7_resume_provenance(
        resumed_agent, metadata, config
    )
    assert restored == report
    assert resumed_agent.canonical_teacher is not None
    assert all(
        not parameter.requires_grad
        for parameter in resumed_agent.canonical_teacher.parameters()
    )
    assert environment.get_action_mask().shape[0] > 1
