"""Evaluation reuse must preserve the identity of the executing PPO model."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs.runtime import runtime_manifest
from environment import AssemblySchedulingEnv
import eval as evaluation
from result.provenance import network_weights_sha256
from test_module_entrypoints import tiny_config
from utils import capture_global_rng_state


@pytest.fixture
def evaluation_case(tmp_path, fixed_instance):
    config, instance = tiny_config(tmp_path, fixed_instance)
    observation = AssemblySchedulingEnv(config).reset(instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    # Initialize Adam to also detect changes to the optimizer on rejected reuse.
    for parameter in agent.network.parameters():
        parameter.grad = torch.zeros_like(parameter)
    agent.optimizer.step()
    agent.optimizer.zero_grad(set_to_none=True)
    return config, instance, observation, agent


def _assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_equal(left, right)
    elif hasattr(first, "shape"):
        assert (first == second).all()
    else:
        assert first == second


def _call_entry(entry, config, instance, observation, agent, monkeypatch):
    def unexpected_rollout(*args, **kwargs):
        pytest.fail("incompatible evaluation reached inference or workers")

    monkeypatch.setattr(agent, "act", unexpected_rollout)
    monkeypatch.setattr(agent, "act_batch", unexpected_rollout)
    runner = SimpleNamespace(config=config, worker_count=1, evaluate_records=unexpected_rollout)
    if entry == "policy":
        return evaluation.EvaluationPolicy(config, policy_name="ppo", bootstrap_observation=observation,
                                           ppo_agent=agent, decode_mode="sampled", sampling_seed=100011)
    if entry == "dataset":
        return evaluation.evaluate_dataset(config, dataset_name="test", policy_name="ppo", ppo_agent=agent,
                                           decode_mode="sampled", sampling_seed=100011)
    if entry == "diagnostic":
        return evaluation.evaluate_representative_diagnostic(config, dataset_name="test", ppo_agent=agent)
    if entry == "parallel_dataset":
        return evaluation.evaluate_dataset_parallel(config, dataset_name="test", ppo_agent=agent,
                                                     runner=runner, sampling_seed=100011)
    if entry == "parallel_grid":
        return evaluation.evaluate_preference_grid_parallel(config, dataset_name="test", ppo_agent=agent,
                                                             runner=runner, sampling_seed=100011)
    if entry == "worker_runner":
        # The direct worker API must reject before accessing the process pool.
        return ParallelEpisodeRunner.evaluate_records(runner, agent, [], sampling_seed=100011)
    raise AssertionError(entry)


ENTRIES = ("policy", "dataset", "diagnostic", "parallel_dataset", "parallel_grid", "worker_runner")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("field,value", [
    ("encoder_variant", "node_mlp_pool"),
    ("actor_head_variant", "shared_preference"),
    ("hidden_dim", 32),
    ("worker_flow_time_normalization", "absolute_v1"),
    ("normalization_manifest_sha256", "a" * 64),
])
def test_memory_agent_rejects_wrong_network_before_inference(evaluation_case, entry, field, value, monkeypatch):
    config, instance, observation, agent = evaluation_case
    wrong = deepcopy(config)
    wrong["network"][field] = value
    weights = network_weights_sha256(agent.network.state_dict())
    optimizer = deepcopy(agent.optimizer.state_dict())
    rng = capture_global_rng_state()
    with pytest.raises(ValueError, match=field):
        _call_entry(entry, wrong, instance, observation, agent, monkeypatch)
    assert agent.network.training
    assert network_weights_sha256(agent.network.state_dict()) == weights
    _assert_equal(optimizer, agent.optimizer.state_dict())
    _assert_equal(rng, capture_global_rng_state())


@pytest.mark.parametrize("field,value", [("encoder_variant", "node_mlp_pool"), ("actor_head_variant", "shared_preference")])
def test_actual_variant_cannot_be_reported_as_implicit_default(evaluation_case, field, value, monkeypatch):
    config, instance, observation, _ = evaluation_case
    network_config = {**config["network"], field: value}
    agent = PPOAgent(build_actor_critic(observation, network_config), config["ppo"], device="cpu")
    assert field not in config["network"]
    with pytest.raises(ValueError, match=field):
        _call_entry("dataset", config, instance, observation, agent, monkeypatch)


@pytest.mark.parametrize("identity", [None, "linear_v1"])
def test_memory_agent_requires_current_message_identity(evaluation_case, identity, monkeypatch):
    config, instance, observation, agent = evaluation_case
    spec = agent.network.network_spec()
    if identity is None:
        spec.pop("message_function")
    else:
        spec["message_function"] = identity
    monkeypatch.setattr(agent.network, "network_spec", lambda: spec)
    with pytest.raises(ValueError, match="message_function"):
        _call_entry("policy", config, instance, observation, agent, monkeypatch)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("source,target", [("neutral", "full"), ("full", "neutral"), (None, "neutral")])
def test_loaded_agent_checks_fatigue_metadata(evaluation_case, entry, source, target, tmp_path, monkeypatch):
    config, instance, observation, agent = evaluation_case
    source_config = deepcopy(config)
    if source is not None:
        source_config["environment"]["fatigue_mode"] = source
    path = tmp_path / "checkpoint.pt"
    metadata = {} if source is None else {"runtime_manifest": runtime_manifest(source_config)}
    agent.save(path, metadata)
    agent.load(path, load_optimizer=True)
    config["environment"]["fatigue_mode"] = target
    optimizer = deepcopy(agent.optimizer.state_dict())
    weights = network_weights_sha256(agent.network.state_dict())
    rng = capture_global_rng_state()
    with pytest.raises(ValueError, match="fatigue mode"):
        _call_entry(entry, config, instance, observation, agent, monkeypatch)
    assert agent.network.training
    assert network_weights_sha256(agent.network.state_dict()) == weights
    _assert_equal(optimizer, agent.optimizer.state_dict())
    _assert_equal(rng, capture_global_rng_state())


@pytest.mark.parametrize("change", ["encoder_variant", "actor_head_variant", "fatigue_mode"])
def test_prepared_policy_cannot_be_reused_with_changed_config(evaluation_case, change, monkeypatch):
    config, instance, observation, agent = evaluation_case
    policy = evaluation.EvaluationPolicy(config, policy_name="ppo", bootstrap_observation=observation,
                                         ppo_agent=agent, decode_mode="sampled", sampling_seed=100011)
    # Mutate the original config to ensure the policy did not keep its mode by reference.
    if change == "fatigue_mode":
        config["environment"][change] = "neutral"
    else:
        config["network"][change] = "node_mlp_pool" if change == "encoder_variant" else "shared_preference"
    monkeypatch.setattr(policy, "begin_episode", lambda *args: pytest.fail("rejected policy began an episode"))
    rng = capture_global_rng_state()
    with pytest.raises(ValueError, match=change.replace("_", " ") if change == "fatigue_mode" else change):
        evaluation.evaluate_instance(config, instance=instance, policy_name="ppo", prepared_policy=policy)
    assert agent.network.training
    _assert_equal(rng, capture_global_rng_state())


def test_prepared_policy_rechecks_checkpoint_loaded_after_preparation(evaluation_case, tmp_path, monkeypatch):
    config, instance, observation, agent = evaluation_case
    policy = evaluation.EvaluationPolicy(config, policy_name="ppo", bootstrap_observation=observation,
                                         ppo_agent=agent, decode_mode="greedy")
    source = deepcopy(config)
    source["environment"]["fatigue_mode"] = "neutral"
    path = tmp_path / "later.pt"
    agent.save(path, {"runtime_manifest": runtime_manifest(source)})
    agent.load(path)
    monkeypatch.setattr(policy, "begin_episode", lambda *args: pytest.fail("rejected policy began an episode"))
    with pytest.raises(ValueError, match="fatigue mode"):
        evaluation.evaluate_instance(config, instance=instance, policy_name="ppo", prepared_policy=policy)
    assert agent.network.training


@pytest.mark.parametrize("encoder,head,mode,loaded", [
    ("hetero_gnn", "objective_experts", "full", False),
    ("node_mlp_pool", "objective_experts", "full", False),
    ("hetero_gnn", "shared_preference", "full", False),
    ("hetero_gnn", "objective_experts", "neutral", False),
    ("node_mlp_pool", "shared_preference", "neutral", True),
])
def test_matching_memory_agent_runs_and_reports_actual_identity(evaluation_case, encoder, head, mode, loaded, tmp_path):
    config, instance, _, _ = evaluation_case
    config["network"].update(encoder_variant=encoder, actor_head_variant=head)
    config["environment"]["fatigue_mode"] = mode
    config["runtime_manifest"] = runtime_manifest(config)
    observation = AssemblySchedulingEnv(config).reset(instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    if loaded:
        path = tmp_path / "matching.pt"
        agent.save(path, {"runtime_manifest": config["runtime_manifest"]})
        agent.load(path)
    rows, _, _, aggregate = evaluation.evaluate_dataset(config, dataset_name="test", policy_name="ppo",
                                                        ppo_agent=agent, decode_mode="greedy")
    assert rows[0]["task_succeeded"]
    assert rows[0]["encoder_variant"] == encoder == agent.network.encoder_variant
    assert rows[0]["actor_head_variant"] == head == agent.network.actor_head_variant
    assert rows[0]["fatigue_mode"] == mode
    assert aggregate["completion_rate"] == 1.0
    assert agent.network.training


@pytest.mark.parametrize("encoder,head,mode", [
    ("node_mlp_pool", "objective_experts", "full"),
    ("hetero_gnn", "shared_preference", "neutral"),
])
def test_matching_loaded_agent_runs_in_actual_parallel_workers(evaluation_case, encoder, head, mode, tmp_path):
    config, instance, _, _ = evaluation_case
    config["network"].update(encoder_variant=encoder, actor_head_variant=head)
    config["environment"]["fatigue_mode"] = mode
    config["runtime_manifest"] = runtime_manifest(config)
    observation = AssemblySchedulingEnv(config).reset(instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    path = tmp_path / "parallel.pt"
    agent.save(path, {"runtime_manifest": config["runtime_manifest"]})
    agent.load(path)
    with ParallelEpisodeRunner(config=config, template=instance, episode_count=1, worker_count=1) as runner:
        ordinary, _ = evaluation.evaluate_dataset_parallel(config, dataset_name="test", ppo_agent=agent,
                                                            runner=runner, decode_mode="greedy")
        grid, _ = evaluation.evaluate_preference_grid_parallel(config, dataset_name="test", ppo_agent=agent,
                                                               runner=runner, decode_mode="greedy",
                                                               preferences=((1.0, 0.0, 0.0),))
    for rows in (ordinary, grid):
        assert len(rows) == 1
        assert rows[0]["task_succeeded"]
        assert rows[0]["encoder_variant"] == encoder
        assert rows[0]["actor_head_variant"] == head
        assert rows[0]["fatigue_mode"] == mode
    assert agent.network.training
