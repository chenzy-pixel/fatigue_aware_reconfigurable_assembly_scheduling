"""Workload identity must be preserved at every checkpoint/evaluation boundary."""
from copy import deepcopy
import json

import pytest
import torch

from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from agent.ppo.network import WORKLOAD_IDENTITY_FIELDS, infer_checkpoint_network_spec
from configs import load_config, project_path
from configs.config import public_config
from configs.normalization import apply_normalization_manifest, file_sha256
from configs.runtime import runtime_manifest
from environment import AssemblySchedulingEnv
from environment.observation_schema import GLOBAL_FEATURE_NAMES, LEGACY_GLOBAL_FEATURE_NAMES
from environment.types import flow_reward_version
from environment.workload import workload_config
from result.provenance import network_weights_sha256
from test_evaluation_agent_contract import ENTRIES, _call_entry
from test_module_entrypoints import tiny_config
import eval as evaluation


def _legacy_config():
    # Build a self-contained saved schema-10 fixture from the current common
    # physics/architecture settings and the unchanged legacy scale manifest.
    config = json.loads(project_path("configs/default.json").read_text(encoding="utf-8"))
    config.pop("workload_objective", None)
    network = config["network"]
    for field in (*WORKLOAD_IDENTITY_FIELDS, "reward_version", "flow_mode", "normalization_manifest_sha256", "global_objective_encoding"):
        network.pop(field, None)
    network.update(observation_schema_version=10, hidden_dim=16, actor_head_variant="shared_preference")
    manifest = project_path("configs/manifests/v2_selected_scales_20261002.json")
    config["objective_scalarizer"].update(
        flow_mode="raw_v1", normalization_manifest=str(manifest),
        normalization_manifest_sha256=file_sha256(manifest),
    )
    config["experiment_suite_version"] = "single_stage_progress_quality_failure_v3"
    config["reward"]["mode"] = config["experiment_suite_version"]
    config["device"] = "cpu"
    apply_normalization_manifest(config, project_root=project_path("."))
    config["runtime_manifest"] = runtime_manifest(config)
    return config


def _modern_config(mode="geometric_owa", q=0.5):
    config = _legacy_config()
    config.pop("runtime_manifest")
    config["workload_objective"] = {"mode": mode, **({"q": q} if mode == "geometric_owa" else {})}
    config["network"]["observation_schema_version"] = 12
    identity = workload_config(config)
    scales = config["objective_scalarizer"]["scales"]
    scales["workload"] = scales.pop("variance")
    config["network"].update(
        workload_mode=identity["mode"], workload_q=identity["q"],
        workload_version=identity["version"], workload_order=identity["order"],
        workload_weight_rule=identity["weights"], workload_scale=scales["workload"],
        objective_scales=tuple(scales[name] for name in ("flow", "cost", "workload")),
    )
    config["experiment_suite_version"] = flow_reward_version(config)
    config["reward"]["mode"] = config["experiment_suite_version"]
    config["network"]["reward_version"] = config["experiment_suite_version"]
    config["runtime_manifest"] = runtime_manifest(config)
    return config


def test_saved_schema10_config_missing_mode_and_schema_keeps_legacy_contract(tmp_path, fixed_instance):
    saved = _legacy_config()
    saved["network"].pop("observation_schema_version")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    restored = load_config(path)
    assert "workload_objective" not in restored
    assert restored["runtime_manifest"] == saved["runtime_manifest"]
    assert restored["network"]["observation_schema_version"] == 10
    observation = AssemblySchedulingEnv(restored).reset(fixed_instance)
    assert observation.global_feature_names == LEGACY_GLOBAL_FEATURE_NAMES
    network = build_actor_critic(observation, restored["network"])
    assert network.objectives == ("flow", "cost", "variance")
    assert "variance" in network.production_shared.schema
    assert not any("workload" in name for name in network.state_dict())
    assert not set(WORKLOAD_IDENTITY_FIELDS).intersection(network.network_spec())


@pytest.mark.parametrize("schema", [7, 9, 10, 11])
def test_missing_mode_requires_verified_saved_schema10(tmp_path, schema):
    config = _legacy_config()
    config["network"]["observation_schema_version"] = schema
    if schema == 10:
        config.pop("runtime_manifest")
    else:
        config["runtime_manifest"]["observation_schema"] = schema
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="schema|workload_objective"):
        load_config(path, allow_observation_migration=True)


def test_legacy_saved_manifest_is_verified_before_accepting_missing_mode(tmp_path):
    config = _legacy_config()
    config["runtime_manifest"]["message_function"] = "linear_v1"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_manifest"):
        load_config(path)


def test_saved_schema10_extending_modern_default_keeps_legacy_identity(tmp_path):
    saved = _legacy_config()
    saved["extends"] = str(project_path("configs/default.json"))
    saved["network"].pop("observation_schema_version")
    path = tmp_path / "legacy_child.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    restored = load_config(path)
    assert "workload_objective" not in restored
    assert restored["network"]["observation_schema_version"] == 10
    assert not set(WORKLOAD_IDENTITY_FIELDS).intersection(restored["network"])
    assert restored["runtime_manifest"] == saved["runtime_manifest"]


@pytest.mark.parametrize("mode", ["variance", "geometric_owa"])
def test_modern_modes_share_neutral_schema_and_store_full_identity(mode, fixed_instance):
    config = _modern_config(mode)
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    network = build_actor_critic(observation, config["network"])
    spec = infer_checkpoint_network_spec({"network_spec": network.network_spec()})
    assert observation.global_feature_names == GLOBAL_FEATURE_NAMES
    assert network.objectives == ("flow", "cost", "workload")
    assert spec["observation_schema_version"] == 12
    assert spec["objective_names"] == network.objectives
    assert spec["workload_mode"] == mode
    assert spec["workload_q"] == (0.5 if mode == "geometric_owa" else None)
    assert spec["objective_scales"][2] == spec["workload_scale"]
    assert "workload" in network.production_shared.schema
    assert not any("variance" in name for name in network.state_dict())
    logits, value = network(observation, env.get_action_mask(), device="cpu")
    assert torch.isfinite(value)
    assert torch.isfinite(logits).all()


def _change_workload(config, field):
    if field == "workload_mode":
        config["workload_objective"] = {"mode": "variance"}
        config["network"].update(workload_mode="variance", workload_q=None)
        config["network"]["workload_weight_rule"] = workload_config({"workload_objective": {"mode": "variance"}})["weights"]
    elif field == "workload_q":
        config["workload_objective"]["q"] = 0.7
        config["network"]["workload_q"] = 0.7
    else:
        config["objective_scalarizer"]["scales"]["workload"] *= 2
        config["network"]["workload_scale"] *= 2
        scales = list(config["network"]["objective_scales"])
        scales[2] *= 2
        config["network"]["objective_scales"] = tuple(scales)


@pytest.fixture
def workload_evaluation_case(tmp_path, fixed_instance):
    config, instance = tiny_config(tmp_path, fixed_instance)
    assert config["workload_objective"]["mode"] == "geometric_owa"
    observation = AssemblySchedulingEnv(config).reset(instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"])
    return config, instance, observation, agent


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("field", ["workload_mode", "workload_q", "workload_scale"])
def test_evaluation_paths_reject_workload_mismatch_before_inference(workload_evaluation_case, entry, field, monkeypatch):
    config, instance, observation, agent = workload_evaluation_case
    wrong = deepcopy(config)
    _change_workload(wrong, field)
    weights = network_weights_sha256(agent.network.state_dict())
    with pytest.raises(ValueError, match="workload|objective_scales"):
        _call_entry(entry, wrong, instance, observation, agent, monkeypatch)
    assert network_weights_sha256(agent.network.state_dict()) == weights
    assert agent.network.training


@pytest.mark.parametrize("field", ["workload_mode", "workload_q", "workload_scale"])
def test_prepared_policy_rechecks_workload_identity(workload_evaluation_case, field, monkeypatch):
    config, instance, observation, agent = workload_evaluation_case
    policy = evaluation.EvaluationPolicy(config, policy_name="ppo", bootstrap_observation=observation,
                                         ppo_agent=agent, decode_mode="greedy")
    _change_workload(config, field)
    monkeypatch.setattr(policy, "begin_episode", lambda *args: pytest.fail("incompatible prepared policy began an episode"))
    with pytest.raises(ValueError, match="workload|objective_scales"):
        evaluation.evaluate_instance(config, instance=instance, policy_name="ppo", prepared_policy=policy)


@pytest.mark.parametrize("mode,q", [("variance", None), ("geometric_owa", 0.7)])
def test_checkpoint_resume_rejects_cross_mode_or_q_before_state_load(tmp_path, fixed_instance, mode, q):
    source = _modern_config()
    destination = _modern_config(mode, q)
    first = PPOAgent(build_actor_critic(AssemblySchedulingEnv(source).reset(fixed_instance), source["network"]), source["ppo"])
    second = PPOAgent(build_actor_critic(AssemblySchedulingEnv(destination).reset(fixed_instance), destination["network"]), destination["ppo"])
    path = tmp_path / "checkpoint.pt"
    first.save(path)
    weights = network_weights_sha256(second.network.state_dict())
    with pytest.raises(ValueError, match="workload"):
        second.load(path, load_optimizer=True)
    assert network_weights_sha256(second.network.state_dict()) == weights


def test_checkpoint_resume_rejects_cross_scale_before_state_load(tmp_path, fixed_instance):
    source = _modern_config()
    destination = deepcopy(source)
    _change_workload(destination, "workload_scale")
    first = PPOAgent(build_actor_critic(AssemblySchedulingEnv(source).reset(fixed_instance), source["network"]), source["ppo"])
    second = PPOAgent(build_actor_critic(AssemblySchedulingEnv(destination).reset(fixed_instance), destination["network"]), destination["ppo"])
    path = tmp_path / "checkpoint.pt"
    first.save(path)
    weights = network_weights_sha256(second.network.state_dict())
    with pytest.raises(ValueError, match="workload_scale|objective_scales"):
        second.load(path, load_optimizer=True)
    assert network_weights_sha256(second.network.state_dict()) == weights


def test_schema10_checkpoint_is_rejected_by_modern_variance_control(tmp_path, fixed_instance):
    legacy, modern = _legacy_config(), _modern_config("variance")
    first = PPOAgent(build_actor_critic(AssemblySchedulingEnv(legacy).reset(fixed_instance), legacy["network"]), legacy["ppo"])
    second = PPOAgent(build_actor_critic(AssemblySchedulingEnv(modern).reset(fixed_instance), modern["network"]), modern["ppo"])
    path = tmp_path / "legacy.pt"
    first.save(path)
    with pytest.raises(ValueError, match="observation schema"):
        second.load(path, allow_observation_migration=True, load_optimizer=True)


@pytest.mark.parametrize("field", WORKLOAD_IDENTITY_FIELDS)
def test_modern_checkpoint_requires_complete_workload_identity(field, fixed_instance):
    config = _modern_config()
    network = build_actor_critic(AssemblySchedulingEnv(config).reset(fixed_instance), config["network"])
    spec = network.network_spec()
    spec.pop(field)
    with pytest.raises(ValueError, match="workload|objective"):
        infer_checkpoint_network_spec({"network_spec": spec})


def test_rollout_buffer_rejects_mixed_legacy_and_modern_observations(fixed_instance):
    legacy_env = AssemblySchedulingEnv(_legacy_config())
    modern_env = AssemblySchedulingEnv(_modern_config())
    first = legacy_env.reset(fixed_instance)
    second = modern_env.reset(fixed_instance)
    buffer = RolloutBuffer()
    buffer.add(first, legacy_env.get_action_mask(), 0, 0, 0, 0, False)
    with pytest.raises(ValueError, match="global feature"):
        buffer.add(second, modern_env.get_action_mask(), 0, 0, 0, 0, False)
    assert len(buffer) == 1


def test_ablation_profile_preserves_workload_identity_after_json_roundtrip(config):
    from scripts.ablation_protocol import protocol_profile

    saved = json.loads(json.dumps(public_config(config)))
    assert protocol_profile(saved) == protocol_profile(config)
    for field in ("workload_mode", "workload_q", "workload_scale"):
        changed = deepcopy(config)
        _change_workload(changed, field)
        assert protocol_profile(changed) != protocol_profile(config)


def test_ablation_pairing_compares_selected_workload_scale_and_legacy_variance():
    from analysis.ablation_analysis import CONTROL_FIELDS, paired_rows

    modern = {**dict.fromkeys(CONTROL_FIELDS, "same"), "instance_id": "i", "sampling_repeat": "0",
              "preference_key": "p", "task_succeeded": "True", "physical_safety_pass": "True"}
    changed = {**modern, "objective_scale_workload": "changed"}
    with pytest.raises(ValueError, match="objective_scale_workload"):
        paired_rows([modern], [changed], "workload")
    legacy = {key: value for key, value in modern.items() if key != "objective_scale_workload"}
    legacy["objective_scale_variance"] = "same"
    assert paired_rows([legacy], [deepcopy(legacy)], "legacy")[1]["common_success_count"] == 1
    with pytest.raises(ValueError, match="objective_scale_workload"):
        paired_rows([modern], [legacy], "cross_schema")
