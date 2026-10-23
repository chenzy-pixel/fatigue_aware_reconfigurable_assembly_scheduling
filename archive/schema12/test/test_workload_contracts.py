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
from environment.observation_schema import GLOBAL_FEATURE_NAMES
from environment.types import flow_reward_version
from environment.workload import workload_config
from result.provenance import network_weights_sha256
from test_evaluation_agent_contract import ENTRIES, _call_entry
from test_module_entrypoints import tiny_config
import eval as evaluation


def _modern_config(mode="geometric_owa", q=0.5):
    config = load_config("configs/controls/variance.json" if mode == "variance" else "configs/default.json")
    config["device"] = "cpu"
    config["network"]["hidden_dim"] = 16
    if mode == "geometric_owa":
        config["workload_objective"]["q"] = q
        config["network"]["workload_q"] = q
    config["runtime_manifest"] = runtime_manifest(config)
    return config


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
    with pytest.raises(ValueError, match="workload|normalization_manifest_sha256"):
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


@pytest.mark.parametrize("field", WORKLOAD_IDENTITY_FIELDS)
def test_modern_checkpoint_requires_complete_workload_identity(field, fixed_instance):
    config = _modern_config()
    network = build_actor_critic(AssemblySchedulingEnv(config).reset(fixed_instance), config["network"])
    spec = network.network_spec()
    spec.pop(field)
    with pytest.raises(ValueError, match="workload|objective"):
        infer_checkpoint_network_spec({"network_spec": spec})


def test_ablation_profile_preserves_workload_identity_after_json_roundtrip(config):
    from scripts.ablation_protocol import protocol_profile

    saved = json.loads(json.dumps(public_config(config)))
    assert protocol_profile(saved) == protocol_profile(config)
    for field in ("workload_mode", "workload_q", "workload_scale"):
        changed = deepcopy(config)
        _change_workload(changed, field)
        assert protocol_profile(changed) != protocol_profile(config)


def test_ablation_pairing_requires_selected_workload_scale():
    from analysis.ablation_analysis import CONTROL_FIELDS, paired_rows

    modern = {**dict.fromkeys(CONTROL_FIELDS, "same"), "instance_id": "i", "sampling_repeat": "0",
              "preference_key": "p", "task_succeeded": "True", "physical_safety_pass": "True"}
    changed = {**modern, "objective_scale_workload": "changed"}
    with pytest.raises(ValueError, match="objective_scale_workload"):
        paired_rows([modern], [changed], "workload")
    missing_scale = {key: value for key, value in modern.items() if key != "objective_scale_workload"}
    with pytest.raises(ValueError, match="objective_scale_workload"):
        paired_rows([modern], [missing_scale], "missing_scale")

    missing_scale["objective_scale_variance"] = "same"
    with pytest.raises(ValueError, match="objective_scale_workload"):
        paired_rows([missing_scale], [deepcopy(missing_scale)], "retired_scale_field")
