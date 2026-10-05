from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from configs import load_config, project_path
from configs.ablation_protocol import assert_paired_configs, load_ablation_manifest, training_contract, validate_training_run
from configs.config import public_config
from environment import AssemblySchedulingEnv
from agent.baselines import HeuristicPolicy
from result.io import write_config


def test_saved_effective_config_round_trip_and_runtime_validation(tmp_path):
    config = load_config("configs/v8/universal.json")
    write_config(tmp_path, config)
    reloaded = load_config(tmp_path / "config.json")
    assert public_config(reloaded) == public_config(config)
    saved = json.loads((tmp_path / "config.json").read_text())
    saved["runtime_manifest"]["observation_schema"] = 5
    (tmp_path / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 10"):
        load_config(tmp_path / "config.json")


def test_snapshot_relocates_pinned_manifest_from_missing_checkout(tmp_path):
    config = public_config(load_config("configs/v8/universal.json"))
    manifest = Path(config["objective_scalarizer"]["normalization_manifest"])
    assert not manifest.is_absolute()
    config["objective_scalarizer"]["normalization_manifest"] = str(tmp_path / "other_checkout" / manifest)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["objective_scalarizer"]["normalization_manifest"] == manifest.as_posix()
    config["objective_scalarizer"]["normalization_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_config(path)


def test_fixed_instance_fixture_works_with_absent_pickle(config, tmp_path):
    settings = deepcopy(config)
    settings["paths"]["instance_cache"] = str(tmp_path / "missing.pkl")
    namespace = runpy.run_path(str(project_path("test/conftest.py")))
    instance = namespace["fixed_instance"].__wrapped__(settings)
    assert len(instance.operations) == 60
    assert not Path(settings["paths"]["instance_cache"]).exists()


def test_completion_on_last_allowed_decision_is_success(config, fixed_instance):
    settings = deepcopy(config)
    policy = HeuristicPolicy()
    reference = AssemblySchedulingEnv(settings)
    reference.reset(fixed_instance, build_observation=False)
    while not reference.task_done:
        reference.step(policy.select_action(reference), build_observation=False)
    assert reference.task_succeeded
    settings["environment"]["max_decisions"] = reference._decision_count
    bounded = AssemblySchedulingEnv(settings)
    bounded.reset(fixed_instance, build_observation=False)
    while not bounded.task_done:
        _, reward, _, _, _ = bounded.step(policy.select_action(bounded), build_observation=False)
    assert bounded.task_succeeded
    assert bounded.terminal_reason == "completed"
    assert reward.failure == 0.0
    assert bounded.metrics()["training_cumulative_reward"] == pytest.approx(reference.metrics()["training_cumulative_reward"])
    settings["environment"]["max_decisions"] -= 1
    incomplete = AssemblySchedulingEnv(settings)
    incomplete.reset(fixed_instance, build_observation=False)
    while not incomplete.task_done:
        incomplete.step(policy.select_action(incomplete), build_observation=False)
    assert incomplete.sampling_truncated and not incomplete.task_failed
    assert incomplete.terminal_reason == "decision_limit"


def test_ablation_manifest_has_matched_current_full_baselines():
    manifest = load_ablation_manifest()
    assert len(manifest["runs"]) == 9
    for objective in ("flow", "cost", "variance"):
        full, neutral = [load_config(manifest["runs"][f"{mode}_{objective}"]["config"]) for mode in ("full", "neutral")]
        assert_paired_configs(full, neutral, "fatigue_mode")
        assert full["training"]["episodes"] == neutral["training"]["episodes"] == 1000
        assert full["reward"]["terminal_failure_penalty"] == 2.0
        assert "rerun_20260928" not in manifest["runs"][f"full_{objective}"]["run_name"]


@pytest.mark.parametrize("field", ("penalty", "scale", "budget", "input", "validation"))
def test_ablation_control_check_rejects_confounded_pairs(field):
    full = load_config("configs/ablations/full_flow.json")
    neutral = load_config("configs/ablations/neutral_flow.json")
    if field == "penalty":
        full["reward"]["terminal_failure_penalty"] = 1.0
    elif field == "scale":
        full["objective_scalarizer"]["scales"]["flow"] = 1200.0
    elif field == "budget":
        full["training"]["episodes_per_update"] += 20
    elif field == "input":
        full["network"]["worker_flow_time_normalization"] = "absolute_v1"
    else:
        full["training"]["validation_interval_episodes"] = 80
    with pytest.raises(ValueError, match="unmatched ablation"):
        assert_paired_configs(full, neutral, "fatigue_mode")


def test_direct_smoke_script_imports_from_any_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(project_path("scripts/run_00_smoke.py")), run_name="audit_import")
    assert callable(namespace["main"])


def test_fatigue_training_launcher_dispatches_all_six_arms(monkeypatch):
    from scripts.ablation_protocol import ROLE_CONFIGS
    namespace = runpy.run_path(str(project_path("scripts/run_12_train_neutral.py")))
    calls = []
    monkeypatch.setitem(namespace["main"].__globals__, "train_group", lambda roles: calls.extend(roles))
    namespace["main"]()
    assert set(calls) == {f"{mode}_{objective}" for mode in ("full", "neutral") for objective in ("flow", "cost", "variance")}
    for role in calls:
        config = load_config(ROLE_CONFIGS[role])
        assert config["training"]["episodes"] == 1000
        assert config["training"]["episodes_per_update"] == 20
        assert config["objective_scalarizer"]["normalization_manifest"].endswith("v2_selected_scales_20261002.json")


@pytest.mark.parametrize("corruption", (None, "partial", "checkpoint_config", "weights", "validation_manifest"))
def test_training_run_preflight_checks_actual_checkpoint_and_completed_budget(fixed_instance, tmp_path, corruption):
    from agent.ppo import PPOAgent, build_actor_critic
    from train import _checkpoint_metadata
    from result.io import write_json

    config = load_config("configs/ablations/full_cost.json")
    config["network"]["hidden_dim"] = 16
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    metadata = _checkpoint_metadata(config, role="best", episode=400, validation_split="validation",
                                    validation_instance_limit=50)
    write_config(tmp_path, config)
    summary = {"episodes": 1000, "checkpoint_selection": {"has_best": True, "best_episode": 400}}
    if corruption == "partial":
        summary["episodes"] = 2
    elif corruption == "checkpoint_config":
        metadata["effective_config"]["reward"]["terminal_failure_penalty"] = 1.0
    elif corruption == "validation_manifest":
        metadata["validation_dataset_manifest"]["sha256"] = "wrong"
    agent.save(tmp_path / "best_checkpoint.pt", metadata=metadata)
    if corruption == "weights":
        payload = torch.load(tmp_path / "best_checkpoint.pt", weights_only=False)
        payload["network"][next(iter(payload["network"]))] = payload["network"][next(iter(payload["network"]))] + 1
        torch.save(payload, tmp_path / "best_checkpoint.pt")
    write_json(tmp_path / "summary.json", summary)
    if corruption is None:
        assert validate_training_run(tmp_path, config)["checkpoint_episode"] == 400
    else:
        with pytest.raises(ValueError):
            validate_training_run(tmp_path, config)


@pytest.mark.parametrize("saved_schema", range(5, 10))
@pytest.mark.parametrize("allow_migration", (False, True))
def test_saved_older_observation_contract_requires_retraining(tmp_path, saved_schema, allow_migration):
    config = public_config(load_config("configs/default.json"))
    config["runtime_manifest"]["observation_schema"] = saved_schema
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 10.*retraining"):
        load_config(path, allow_observation_migration=allow_migration)
