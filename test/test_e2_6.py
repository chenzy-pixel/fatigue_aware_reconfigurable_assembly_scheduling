import csv
import json

import pytest
import torch

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from configs import load_config, project_path
from data import load_instance_pickle
from e2_6_calibration_audit import audit_manifest
from environment import AssemblySchedulingEnv
from result import aggregate_evaluation_rows, result_schema_version
from train import TrainingPhaseController


def _network():
    config = load_config("configs/v7/e2_6_counterfactual_preference_consistency.json")
    instance = load_instance_pickle(project_path(config["paths"]["instance_cache"]))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance, preference=(0.2, 0.4, 0.4))
    return config, environment, observation, build_actor_critic(observation, config["network"])


def _force_base_defer(network) -> None:
    output = network.production_state_gate[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(torch.tensor((0.0, 1.0), dtype=output.bias.dtype))


def test_e2_6_gate_contract_and_checkpoint_recovery(tmp_path) -> None:
    config, environment, observation, network = _network()
    assert result_schema_version(config) == "4.7.0"
    assert network.production_gate_version == "state_only_counterfactual_monotone_flow_commit_gate_v3"
    _force_base_defer(network)
    network.eval()
    network.set_production_state_gate_frozen(True)
    network.set_production_flow_commit_residual_enabled(True)
    base = network._state_only_production_gate_logits(
        observation, dtype=torch.float32, device="cpu"
    )
    scale = network._production_flow_commit_residual_scale(
        observation, dtype=torch.float32, device="cpu"
    )
    low, _ = network._final_production_gate_logits(
        base, torch.tensor((0.2, 0.4, 0.4)), state_scale=scale
    )
    high, _ = network._final_production_gate_logits(
        base, torch.tensor((1.0, 0.0, 0.0)), state_scale=scale
    )
    assert torch.equal(low, base)
    assert float(scale) == pytest.approx(2.0)
    assert 0.0 < float(scale) < 8.0
    assert float(high[0] - high[1]) > 0.0

    diagnostics = network.counterfactual_production_gate_batch(
        [observation], [environment.get_action_mask()], device="cpu"
    )
    assert bool(diagnostics["eligible"][0])
    assert bool(diagnostics["high_flow_flip"][0])
    assert not bool(diagnostics["low_flow_identity_violation"][0])
    assert not bool(diagnostics["monotonicity_violation"][0])

    checkpoint = tmp_path / "e2_6.pt"
    agent = PPOAgent(network, config["ppo"], device="cpu")
    agent.save(checkpoint)
    clone_network = build_actor_critic(observation, config["network"])
    clone = PPOAgent(clone_network, config["ppo"], device="cpu")
    clone.load(checkpoint)
    assert clone_network.production_state_gate_frozen
    assert clone_network.production_flow_commit_residual_active
    assert all(
        not parameter.requires_grad
        for parameter in clone_network.production_state_gate.parameters()
    )


def test_e2_6_auxiliary_loss_only_updates_state_residual() -> None:
    config, environment, observation, network = _network()
    _force_base_defer(network)
    with torch.no_grad():
        network.production_state_gate[-1].bias.copy_(
            torch.tensor((0.0, 3.0), dtype=network.production_state_gate[-1].bias.dtype)
        )
    network.set_production_state_gate_frozen(True)
    network.set_production_flow_commit_residual_enabled(True)
    agent = PPOAgent(network, config["ppo"], device="cpu")
    mask = environment.get_action_mask()
    buffer = RolloutBuffer(preserve_graph=True)
    buffer.add(
        observation,
        mask,
        int(torch.nonzero(torch.as_tensor(~mask[:-1]))[0]),
        0.0,
        0.0,
        0.0,
        False,
    )
    loss, diagnostics = agent._counterfactual_loss(
        buffer.transitions, reward_phase="quality"
    )
    assert float(loss) > 0.0
    assert diagnostics["counterfactual_eligible_count"] == 1.0
    assert diagnostics["counterfactual_high_flow_commit_flip_count"] == 0.0
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in network.production_flow_commit_residual_state.parameters()
    )
    assert all(
        parameter.grad is None for parameter in network.production_state_gate.parameters()
    )
    assert all(parameter.grad is None for parameter in network.critic.parameters())
    network.zero_grad(set_to_none=True)
    inactive, inactive_diagnostics = agent._counterfactual_loss(
        buffer.transitions, reward_phase="feasibility"
    )
    assert float(inactive) == 0.0
    assert inactive_diagnostics["counterfactual_eligible_count"] == 0.0
    buffer.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    metrics = agent.update(buffer, reward_phase="quality")
    assert metrics["counterfactual_loss"] > 0.0
    assert metrics["counterfactual_eligible_count"] == 1.0


def test_e2_6_schema_is_accepted_by_aggregation() -> None:
    aggregate = aggregate_evaluation_rows(
        [],
        dataset="validation",
        policy="ppo",
        manifest="validation/manifest.json",
        schema_version="4.7.0",
    )
    assert aggregate["evaluation_schema_version"] == "4.7.0"


def test_e2_6_pareto_promotion_requires_counterfactual_gate() -> None:
    config, _, _, _ = _network()
    controller = TrainingPhaseController.from_config(config)
    snapshot = {
        "scope": "full_grid_22",
        "mean_hypervolume": 1.0,
        "canonical_quality": 1.0,
        "all_safe": True,
        "coverage_pass": True,
        "controllability_pass": True,
        "worker_direct_preference_pass": True,
        "preference_response_pass": True,
        "low_flow_safety_pass": True,
        "counterfactual_gate_pass": False,
    }
    assert controller.observe_pareto_snapshot(snapshot, completed_episodes=100) == "rejected"
    assert controller.last_promotion_diagnostics["promotion_decision_reason"] == "counterfactual_gate_failed"


def _audit_row(*, quality: float = 1.0, flip_rate: float = 0.5) -> dict[str, object]:
    return {
        "scope": "full_grid_22",
        "update_id": 1,
        "instance_count": 20,
        "preference_count": 22,
        "candidate_count": 440,
        "schedule_violation_count": 0,
        "completion_rate": 1.0,
        "low_flow_candidate_count": 220,
        "low_flow_completion_rate": 1.0,
        "all_safe": True,
        "coverage_pass": True,
        "controllability_pass": True,
        "worker_direct_preference_pass": True,
        "preference_response_pass": True,
        "low_flow_safety_pass": True,
        "counterfactual_gate_pass": True,
        "mean_unique_action_trace_count": 8.0,
        "mean_unique_objective_count": 8.0,
        "mean_nondominated_count": 4.0,
        "preference_response_spearman_flow": -0.05,
        "preference_response_spearman_cost": -0.05,
        "preference_response_spearman_variance": -0.05,
        "counterfactual_instance_coverage": 20,
        "counterfactual_high_flow_commit_flip_rate": flip_rate,
        "counterfactual_low_flow_identity_violation_count": 0,
        "counterfactual_monotonicity_violation_count": 0,
        "canonical_quality": quality,
    }


def _write_audit_run(tmp_path, name, *, coefficient=None, quality=1.0):
    run = tmp_path / name
    run.mkdir()
    is_control = coefficient is None
    config = {
        "seed": 101,
        "algorithm_seeds": [101],
        "experiment_name": (
            "v7_e2_4_calibration_control_seed101"
            if is_control
            else f"v7_e2_6_calibration_{coefficient}"
        ),
        "training": {"episodes": 500, "validation_split": "validation"},
    }
    if not is_control:
        config.update(
            {
                "evaluation": {"result_schema_version": "4.7.0"},
                "ppo": {
                    "counterfactual_preference_consistency": {
                        "enabled": True,
                        "loss_coefficient": coefficient,
                    }
                },
            }
        )
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": 500,
                "provenance": {
                    "source_state_sha256": "source",
                    "effective_config_sha256": "config",
                    "dataset_manifest_sha256": "manifest",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [_audit_row(quality=quality), _audit_row(quality=quality)]
    with (run / "pareto_validation_log.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run


def test_e2_6_auditor_selects_smallest_and_waits_for_control(tmp_path) -> None:
    control = _write_audit_run(tmp_path, "control", quality=1.0)
    candidates = {
        str(value): _write_audit_run(
            tmp_path, f"lambda_{value}", coefficient=value, quality=1.005
        )
        for value in (0.05, 0.10, 0.20)
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"e2_4_control_run": str(control), "candidates": {key: str(value) for key, value in candidates.items()}}),
        encoding="utf-8",
    )
    result = audit_manifest(manifest)
    assert result["status"] == "selected"
    assert result["selected_loss_coefficient"] == pytest.approx(0.05)
    manifest.write_text(
        json.dumps({"e2_4_control_run": None, "candidates": {key: str(value) for key, value in candidates.items()}}),
        encoding="utf-8",
    )
    pending = audit_manifest(manifest)
    assert pending["status"] == "guard_pending"
    assert not pending["formal_seed_training_authorized"]

    manifest.write_text(
        json.dumps(
            {
                "e2_4_control_run": str(tmp_path / "not_synced"),
                "candidates": {
                    key: str(value) for key, value in candidates.items()
                },
            }
        ),
        encoding="utf-8",
    )
    missing = audit_manifest(manifest)
    assert missing["status"] == "guard_pending"
    assert not missing["control_available"]


def test_e2_6_auditor_stops_on_guard_failure_and_rejects_hidden_data(tmp_path) -> None:
    control = _write_audit_run(tmp_path, "control", quality=1.0)
    candidates = {
        str(value): _write_audit_run(
            tmp_path, f"lambda_{value}", coefficient=value, quality=1.02
        )
        for value in (0.05, 0.10, 0.20)
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"e2_4_control_run": str(control), "candidates": {key: str(value) for key, value in candidates.items()}}),
        encoding="utf-8",
    )
    stopped = audit_manifest(manifest)
    assert stopped["status"] == "stopped"
    assert stopped["canonical_guard_failures"]

    config_path = candidates["0.05"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["validation_split"] = "test"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="validation only"):
        audit_manifest(manifest)
