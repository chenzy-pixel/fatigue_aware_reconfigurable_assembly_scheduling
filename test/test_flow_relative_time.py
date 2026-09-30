from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import build_actor_critic
from agent.ppo.network import assert_network_config_matches_spec, normalize_network_config
from configs import load_config
from environment import AssemblySchedulingEnv, DecisionType


def networks(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance, preference=(1, 0, 0))
    baseline = build_actor_critic(observation, dict(config["network"], worker_flow_time_normalization="absolute_v1"))
    settings = dict(config["network"], worker_flow_time_normalization="candidate_zscore_v1")
    experiment = build_actor_critic(observation, settings)
    experiment.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    experiment.eval()
    return env, observation, baseline, experiment


def test_masked_duration_does_not_affect_ranking_and_gradients(config, fixed_instance):
    _, _, _, network = networks(config, fixed_instance)
    durations = torch.tensor([0.01, 0.02, 900.0], requires_grad=True)
    legal = torch.tensor([True, True, False])
    relative = network._relative_worker_flow_time(durations, legal)
    torch.testing.assert_close(relative, torch.tensor([-1., 1., 0.]))
    scores = network.worker_experts.experts["flow"].direct_ranker(relative[:, None])
    assert scores[0] > scores[1]
    scores.square().sum().backward()
    assert torch.isfinite(durations.grad).all()
    assert durations.grad[2] == 0


@pytest.mark.parametrize("legal", [[False, False], [True, False], [True, True]])
def test_ties_empty_and_single_candidate_are_neutral(config, fixed_instance, legal):
    _, _, _, network = networks(config, fixed_instance)
    actual = network._relative_worker_flow_time(torch.tensor([0.02, 0.02]), torch.tensor(legal))
    assert torch.equal(actual, torch.zeros(2))


def test_floor_prevents_amplification_of_tiny_differences(config, fixed_instance):
    _, _, _, network = networks(config, fixed_instance)
    result = network._relative_worker_flow_time(torch.tensor([0.02, 0.020001]), torch.tensor([True, True]))
    assert result.abs().max() < 0.001


def test_worker_integration_preserves_other_experts_and_wait(config, fixed_instance):
    env, observation, baseline, experiment = networks(config, fixed_instance)
    mask = env.get_action_mask()
    with torch.no_grad():
        old_logits, old_value = baseline(observation, mask, device="cpu")
        new_logits, new_value = experiment(observation, mask, device="cpu")
    torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
    torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)
    policy = HeuristicPolicy()
    for _ in range(500):
        mask = env.get_action_mask()
        if env.decision_type == DecisionType.WORKER and (~mask[:-1]).sum() >= 2:
            break
        observation, _, done, truncated, _ = env.step(policy.select_action(env))
        assert not (done or truncated), "fixture needs a multi-candidate worker state"
    else:
        pytest.fail("no worker state found")
    captures = {}
    handles = []
    for label, network in [("old", baseline), ("new", experiment)]:
        for kind in ("worker_experts", "worker_wait_experts"):
            def capture(module, args, key=(label, kind)):
                captures[key] = (args[0].detach().clone(), {k:v.detach().clone() for k,v in args[1].items()})
            handles.append(getattr(network, kind).register_forward_pre_hook(capture))
        with torch.no_grad():
            network(observation, mask, device="cpu")
    for kind in ("worker_experts", "worker_wait_experts"):
        old, new = captures["old", kind], captures["new", kind]
        torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
        for objective in ("cost", "variance"):
            torch.testing.assert_close(old[1][objective], new[1][objective], rtol=0, atol=0)
    torch.testing.assert_close(captures["old", "worker_wait_experts"][1]["flow"], captures["new", "worker_wait_experts"][1]["flow"], rtol=0, atol=0)
    assert not torch.equal(captures["old", "worker_experts"][1]["flow"], captures["new", "worker_experts"][1]["flow"])
    for handle in handles:
        handle.remove()
    for preference in ((0,1,0), (0,0,1)):
        obs = replace(observation, preference=np.asarray(preference, dtype=np.float32))
        with torch.no_grad():
            old, _ = baseline(obs, mask, device="cpu")
            new, _ = experiment(obs, mask, device="cpu")
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_checkpoint_compatibility_is_explicit(config, fixed_instance):
    _, _, baseline, experiment = networks(config, fixed_instance)
    legacy = deepcopy(baseline.network_spec())
    for name in ("worker_flow_time_normalization", "worker_flow_time_std_floor"):
        legacy.pop(name)
    with pytest.raises(ValueError, match="missing worker_flow_time"):
        assert_network_config_matches_spec(baseline.network_spec(), legacy)
    assert_network_config_matches_spec(experiment.network_spec(), experiment.network_spec())
    with pytest.raises(ValueError, match="worker_flow_time_normalization"):
        assert_network_config_matches_spec(experiment.network_spec(), legacy)
    changed = dict(experiment.network_spec(), worker_flow_time_std_floor=0.002)
    with pytest.raises(ValueError, match="worker_flow_time_std_floor"):
        assert_network_config_matches_spec(experiment.network_spec(), changed)


@pytest.mark.parametrize("floor", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_floor_is_rejected(floor):
    with pytest.raises(ValueError, match="worker_flow_time_std_floor"):
        normalize_network_config({"worker_flow_time_std_floor": floor})


def test_experiment_config():
    cfg = load_config("configs/e1/single_flow.json")
    assert cfg["network"]["worker_flow_time_normalization"] == "candidate_zscore_v1"
    assert cfg["training"]["episodes"] == 1000
    assert cfg["preference"]["quality"]["fixed"] == [1,0,0]
