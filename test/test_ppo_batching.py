from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, RolloutBuffer, build_actor_critic
from data.dataset import load_dataset_split
from environment import (
    AssemblySchedulingEnv,
    DecisionType,
    PolicyObservation,
)


def _find_worker_observation(config, instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(instance)
    policy = HeuristicPolicy()
    for _ in range(100):
        if observation.decision_type == DecisionType.WORKER:
            return observation, environment.get_action_mask()
        action = policy.select_action(environment)
        observation, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            break
    raise AssertionError("test instance did not reach a worker decision")


def test_mixed_variable_size_batch_matches_individual_forward(
    config,
    fixed_instance,
):
    validation_record = load_dataset_split(config, "validation")[0]
    first_environment = AssemblySchedulingEnv(config)
    first_observation = first_environment.reset(fixed_instance)
    first_mask = first_environment.get_action_mask()
    second_environment = AssemblySchedulingEnv(config)
    second_observation = second_environment.reset(
        validation_record.instance
    )
    second_mask = second_environment.get_action_mask()
    worker_observation, worker_mask = _find_worker_observation(
        config,
        fixed_instance,
    )
    observations = [
        first_observation,
        second_observation,
        worker_observation,
    ]
    masks = [first_mask, second_mask, worker_mask]
    network = build_actor_critic(first_observation, config["network"])
    network.eval()
    individual = [
        network(observation, mask, device="cpu")
        for observation, mask in zip(observations, masks)
    ]
    batch_logits, batch_values = network.forward_batch(
        observations,
        masks,
        device="cpu",
    )
    for index, ((logits, value), mask) in enumerate(
        zip(individual, masks)
    ):
        assert torch.allclose(
            logits,
            batch_logits[index, : len(mask)],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            value,
            batch_values[index],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.all(
            batch_logits[index, len(mask) :]
            == torch.finfo(batch_logits.dtype).min
        )
    agent = PPOAgent(network, config["ppo"], device="cpu")
    actions, _, values = agent.act_batch(
        observations,
        masks,
        deterministic=True,
    )
    assert all(not masks[index][action] for index, action in enumerate(actions))
    assert all(math.isfinite(value) for value in values)
    compact = PolicyObservation.from_observation(first_observation)
    assert not hasattr(compact, "relations")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("variant", ["hetero_gnn", "node_mlp_pool", "shared_preference"])
def test_phase_batched_head_matches_reference_values_and_gradients(
    config, fixed_instance, device, variant
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    effective_config = deepcopy(config)
    effective_config["network"]["worker_flow_time_normalization"] = "candidate_zscore_v1"
    if variant == "node_mlp_pool":
        effective_config["network"]["encoder_variant"] = variant
    elif variant == "shared_preference":
        effective_config["network"]["actor_head_variant"] = variant
    validation_instance = load_dataset_split(effective_config, "validation")[0].instance
    observations, masks = [], []
    for instance in (fixed_instance, validation_instance):
        environment = AssemblySchedulingEnv(effective_config)
        observations.append(environment.reset(instance))
        masks.append(environment.get_action_mask())
    worker_observation, worker_mask = _find_worker_observation(effective_config, fixed_instance)
    observations.append(worker_observation)
    masks.append(worker_mask)
    wait_only = np.ones_like(worker_mask, dtype=bool)
    wait_only[-1] = False
    observations.append(worker_observation)
    masks.append(wait_only)
    production_wait_only = np.ones_like(masks[0], dtype=bool)
    production_wait_only[-1] = False
    observations.append(observations[0])
    masks.append(production_wait_only)
    pair_only = np.ones_like(worker_mask, dtype=bool)
    pair_only[np.flatnonzero(~worker_mask[:-1])[0]] = False
    observations.append(worker_observation)
    masks.append(pair_only)
    network = build_actor_critic(observations[0], effective_config["network"]).to(device)
    network.eval()
    # Exercise learned context contributions rather than only their zero initialization.
    if variant != "shared_preference":
        with torch.no_grad():
            for prefix in ("production_experts", "worker_experts",
                           "production_wait_experts", "worker_wait_experts"):
                for expert in getattr(network, prefix).experts.values():
                    expert.context[-1].weight.uniform_(-0.05, 0.05)
                    expert.context[-1].bias.uniform_(-0.05, 0.05)

    def evaluate(mode):
        network.zero_grad(set_to_none=True)
        network.execution_mode = mode
        logits, values = network.forward_batch(observations, masks, device=device)
        loss = values.square().sum() + sum(
            logits[index, :len(mask)][~torch.as_tensor(mask, device=device)].square().sum()
            for index, mask in enumerate(masks)
        )
        loss.backward()
        gradients = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in network.parameters()
        ]
        return logits.detach().clone(), values.detach().clone(), gradients

    reference_logits, reference_values, reference_gradients = evaluate("reference_v8")
    batched_logits, batched_values, batched_gradients = evaluate("phase_batched_v1")
    for index, mask in enumerate(masks):
        torch.testing.assert_close(
            batched_logits[index, :len(mask)], reference_logits[index, :len(mask)],
            atol=1e-5, rtol=1e-4,
        )
        torch.testing.assert_close(
            torch.softmax(batched_logits[index, :len(mask)], dim=-1),
            torch.softmax(reference_logits[index, :len(mask)], dim=-1),
            atol=1e-5, rtol=1e-4,
        )
    torch.testing.assert_close(batched_values, reference_values, atol=1e-5, rtol=1e-4)
    for batched, reference in zip(batched_gradients, reference_gradients):
        assert (batched is None) == (reference is None)
        if batched is not None:
            torch.testing.assert_close(batched, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("wait_only", [False, True])
def test_candidate_heads_receive_only_legal_pair_rows(config, fixed_instance, wait_only):
    env = AssemblySchedulingEnv(config)
    production = env.reset(fixed_instance)
    worker, worker_mask = _find_worker_observation(config, fixed_instance)
    observations = [production, worker, production, worker]
    masks = [env.get_action_mask(), worker_mask, env.get_action_mask(), worker_mask]
    if wait_only:
        masks = [np.ones_like(mask, dtype=bool) for mask in masks]
        for mask in masks:
            mask[-1] = False
    network = build_actor_critic(production, config["network"])
    sizes = {}
    handles = []
    for name in ("production_edge_encoder", "worker_edge_encoder",
                 "production_action_encoder", "worker_action_encoder",
                 "production_residual", "worker_residual"):
        handles.append(getattr(network, name).register_forward_pre_hook(
            lambda module, args, name=name: sizes.__setitem__(name, args[0].shape[0])
        ))
    try:
        with torch.no_grad():
            # Tensor masks use the same candidate selection as NumPy masks.
            logits, _ = network.forward_batch(
                observations, [torch.as_tensor(mask) for mask in masks], device="cpu"
            )
        for phase, graph_indices in (("production", [0, 2]), ("worker", [1, 3])):
            expected = sum(int((~masks[i][:-1]).sum()) for i in graph_indices)
            assert sizes[phase + "_edge_encoder"] == expected
            assert sizes[phase + "_action_encoder"] == expected
            assert sizes[phase + "_residual"] == expected + len(graph_indices)
        for i, mask in enumerate(masks):
            assert torch.isfinite(logits[i, :len(mask)][~torch.as_tensor(mask)]).all()
            assert (logits[i, :len(mask)][torch.as_tensor(mask)] == torch.finfo(logits.dtype).min).all()
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("variant", ["hetero_gnn", "node_mlp_pool", "shared_preference"])
def test_value_path_matches_actor_critic_and_skips_actor(
    config, fixed_instance, monkeypatch, variant
):
    settings = deepcopy(config["network"])
    if variant == "node_mlp_pool":
        settings["encoder_variant"] = variant
    elif variant == "shared_preference":
        settings["actor_head_variant"] = variant
    env = AssemblySchedulingEnv(config)
    production = env.reset(fixed_instance)
    worker, worker_mask = _find_worker_observation(config, fixed_instance)
    validation = AssemblySchedulingEnv(config)
    other = validation.reset(load_dataset_split(config, "validation")[0].instance)
    observations = [production, worker, other]
    masks = [env.get_action_mask(), worker_mask, validation.get_action_mask()]
    network = build_actor_critic(production, settings)
    network.eval()
    network.zero_grad(set_to_none=True)
    _, expected = network.forward_batch(observations, masks, device="cpu")
    expected.square().sum().backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None and name.startswith(
            ("node_projectors.", "message_layers.", "node_mlp_layers.",
             "global_encoder.", "preference_encoder.", "critic.")
        )
    }
    network.zero_grad(set_to_none=True)

    def actor_was_called(*args, **kwargs):
        raise AssertionError("value-only inference must skip the actor")

    monkeypatch.setattr(network, "forward_batch", actor_was_called)
    for name in ("graph_context_projector", "production_edge_encoder", "worker_edge_encoder",
                 "production_action_encoder", "worker_action_encoder", "wait_feature_encoder",
                 "wait_action_encoder", "action_preference_projector",
                 "production_residual", "worker_residual"):
        monkeypatch.setattr(getattr(network, name), "forward", actor_was_called)
    actual = network.value_batch(observations, masks, device="cpu")
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for name, parameter in network.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name])
        else:
            assert parameter.grad is None
    agent = PPOAgent(network, config["ppo"], device="cpu")
    assert agent.value_batch(observations, masks) == pytest.approx(expected.detach().tolist())
    assert network.consume_policy_decision_diagnostics() == []


@pytest.mark.parametrize("entrypoint", ["forward_batch", "value_batch"])
def test_inference_paths_reject_invalid_masks(config, fixed_instance, entrypoint):
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    mask = env.get_action_mask()
    network = build_actor_critic(observation, config["network"])
    evaluate = getattr(network, entrypoint)
    with pytest.raises(ValueError, match="non-empty and aligned"):
        evaluate([observation], [], device="cpu")
    with pytest.raises(ValueError, match="one legal action"):
        evaluate([observation], [np.ones_like(mask)], device="cpu")
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate([observation], [mask.reshape(1, -1)], device="cpu")
    with pytest.raises(ValueError, match="mask width"):
        evaluate([observation], [np.zeros(len(mask) + 1, dtype=bool)], device="cpu")


def test_batched_policy_diagnostics_are_plain_values(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    network = build_actor_critic(observation, config["network"])
    with torch.no_grad():
        network.forward_batch([observation, observation], [mask, mask], device="cpu")
    rows = network.consume_policy_decision_diagnostics()
    assert len(rows) == 2
    for name, value in rows[0].items():
        if isinstance(value, float):
            assert rows[1][name] == pytest.approx(value, abs=1e-6)
        else:
            assert rows[1][name] == value
    assert rows[0]["legal_pair_count"] == int((~mask[:-1]).sum())
    assert all(
        isinstance(value, (str, int, float, bool))
        for row in rows
        for value in row.values()
    )
    assert network.consume_policy_decision_diagnostics() == []


def test_sparse_diagnostics_preserve_original_action_indices(config, fixed_instance):
    worker, mask = _find_worker_observation(config, fixed_instance)
    selected = np.flatnonzero(~mask[:-1])[-1]
    pair_only = np.ones_like(mask)
    pair_only[selected] = False
    wait_only = np.ones_like(mask)
    wait_only[-1] = False
    masks = [mask, pair_only, wait_only]
    network = build_actor_critic(worker, config["network"])
    network.eval()
    rows = {}
    with torch.no_grad():
        for mode in ("reference_v8", "phase_batched_v1"):
            network.execution_mode = mode
            network.forward_batch([worker] * len(masks), masks, device="cpu")
            rows[mode] = network.consume_policy_decision_diagnostics()
    for actual, expected in zip(rows["phase_batched_v1"], rows["reference_v8"], strict=True):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            if isinstance(value, float):
                assert actual[name] == pytest.approx(value, abs=1e-6)
            else:
                assert actual[name] == value
    assert rows["phase_batched_v1"][1]["relative_top_action"] == selected
    assert rows["phase_batched_v1"][1]["final_pair_top_action"] == selected


def test_sampled_batch_uses_independent_reproducible_generator(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    network = build_actor_critic(observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device="cpu")
    observations = [observation] * 32
    masks = [mask] * 32
    global_state = torch.random.get_rng_state().clone()
    first_generator = torch.Generator(device="cpu").manual_seed(12345)
    second_generator = torch.Generator(device="cpu").manual_seed(12345)

    first = agent.act_batch(
        observations,
        masks,
        generator=first_generator,
    )
    second = agent.act_batch(
        observations,
        masks,
        generator=second_generator,
    )

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_ppo_update_uses_one_batched_forward_per_minibatch(
    config,
    fixed_instance,
    monkeypatch,
):
    effective_config = deepcopy(config)
    effective_config["ppo"]["batch_size"] = 4
    effective_config["ppo"]["epochs"] = 2
    environment = AssemblySchedulingEnv(effective_config)
    observation = environment.reset(fixed_instance)
    network = build_actor_critic(observation, effective_config["network"])
    agent = PPOAgent(network, effective_config["ppo"], device="cpu")
    buffer = RolloutBuffer(preserve_graph=True)
    for _ in range(10):
        mask = environment.get_action_mask()
        action, log_probability, value = agent.act(observation, mask)
        next_observation, reward, terminated, truncated, _ = (
            environment.step(action)
        )
        buffer.add(
            observation,
            mask,
            action,
            log_probability,
            value,
            reward.scalarize(effective_config["reward"]),
            terminated or truncated,
        )
        observation = next_observation
        if terminated or truncated:
            break
    buffer.compute_gae(
        last_value=0.0,
        gamma=float(effective_config["ppo"]["gamma"]),
        gae_lambda=float(effective_config["ppo"]["gae_lambda"]),
    )
    call_count = 0
    original_forward_batch = network.forward_batch

    def counting_forward_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward_batch(*args, **kwargs)

    monkeypatch.setattr(
        network,
        "forward_batch",
        counting_forward_batch,
    )
    metrics = agent.update(buffer)
    expected = (
        math.ceil(len(buffer) / effective_config["ppo"]["batch_size"])
        * effective_config["ppo"]["epochs"]
    )
    assert call_count == expected
    assert all(math.isfinite(value) for value in metrics.values())


def test_gae_is_computed_before_parallel_buffers_are_merged():
    first = RolloutBuffer()
    second = RolloutBuffer()
    observation = PolicyObservation(
        operations=np.zeros((1, 1), dtype=np.float32),
        machines=np.zeros((1, 1), dtype=np.float32),
        workers=np.zeros((1, 1), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        decision_type=DecisionType.PRODUCTION,
    )
    mask = np.array([False, True], dtype=np.bool_)
    first.add(observation, mask, 0, 0.0, 1.0, 1.0, True)
    second.add(observation, mask, 0, 0.0, 10.0, 5.0, True)
    first.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    second.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=0.95)
    combined = RolloutBuffer()
    combined.extend(first)
    combined.extend(second)
    assert [value.advantage for value in combined.transitions] == [
        pytest.approx(0.0),
        pytest.approx(-5.0),
    ]
