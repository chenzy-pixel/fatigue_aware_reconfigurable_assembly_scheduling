"""Retired schemas must fail before inheritance, inference or state mutation."""
from copy import deepcopy
import json

import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.network import normalize_network_config
from configs import load_config, project_path
from configs.config import public_config
from environment import AssemblySchedulingEnv, normalize_preference
from environment.observation_schema import global_feature_names, infer_global_observation_contract
from result.provenance import network_weights_sha256


@pytest.mark.parametrize("schema", [10, 11])
@pytest.mark.parametrize("saved", [False, True])
def test_retired_config_cannot_inherit_current_defaults(config, tmp_path, schema, saved):
    raw = {"extends": str(project_path("configs/default.json"))}
    if saved:
        raw["runtime_manifest"] = {"observation_schema": schema}
    else:
        raw["network"] = {"observation_schema_version": schema}
    path = tmp_path / "retired.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="only observation schema 12"):
        load_config(path)


def test_saved_config_without_workload_cannot_acquire_parent_contract(config, tmp_path):
    raw = public_config(config)
    raw.pop("workload_objective")
    path = tmp_path / "missing_workload.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="workload_objective is required"):
        load_config(path)


@pytest.mark.parametrize("schema", [10, 11])
def test_retired_network_and_global_feature_contracts_are_rejected(config, schema):
    network = deepcopy(config["network"])
    network["observation_schema_version"] = schema
    with pytest.raises(ValueError, match="schema 12"):
        normalize_network_config(network)
    with pytest.raises(ValueError, match="schema 12"):
        global_feature_names(schema)
    old_names = (
        "current_time_norm", "pending_reconfiguration_ratio", "completed_operation_ratio",
        "production_decision", "worker_matching_deficit_norm", "minimum_worker_alternative_ratio",
        "objective_flow_norm", "objective_cost_norm",
        "objective_committed_variance_norm" if schema == 10 else "objective_committed_workload_norm",
    )
    with pytest.raises(ValueError, match="global feature"):
        infer_global_observation_contract(old_names)


@pytest.mark.parametrize("schema", [10, 11])
@pytest.mark.parametrize("load_optimizer", [False, True])
def test_retired_checkpoint_rejection_preserves_model_and_optimizer(
    config, fixed_instance, tmp_path, schema, load_optimizer,
):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"])
    path = tmp_path / "checkpoint.pt"
    agent.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["network_spec"]["observation_schema_version"] = schema
    torch.save(payload, path)
    before = network_weights_sha256(agent.network.state_dict())
    optimizer = deepcopy(agent.optimizer.state_dict())
    with pytest.raises(ValueError, match="schema 12"):
        agent.load(path, load_optimizer=load_optimizer)
    assert network_weights_sha256(agent.network.state_dict()) == before
    assert agent.optimizer.state_dict() == optimizer


def test_preference_mapping_uses_current_objective_names():
    assert normalize_preference({"flow": 1, "cost": 0, "workload": 0}).as_tuple() == (1, 0, 0)
    with pytest.raises(ValueError, match="exactly flow/cost/workload"):
        normalize_preference({"flow": 1, "cost": 0, "variance": 0})
