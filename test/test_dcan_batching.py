"""Dense-reference parity for DCAN's packed graph and neighbor execution."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.comparators.dcan import ConditionalAttention, _neighbor_rows
from agent.comparators.learning import build_agent
from configs import project_path
from configs.experiment_suite import arm_config, load_suite
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, DecisionType, PreferenceVector


@pytest.fixture(scope="module")
def samples():
    torch.set_num_threads(1)
    config = arm_config(load_suite(), "dcan", 11, "cpu")
    rows = []
    for index, cell in enumerate(("C01", "C14", "C27")):
        instance = load_instance_yaml(project_path(f"data/instances/v3/validation/{cell}.json"))
        target = instance.operations[0].required_module
        machines = tuple(replace(m, initial_module=next(a for a in m.module_parameters if a != target))
                         for m in instance.machines)
        instance = replace(instance, machines=machines, orders=instance.orders[:index+1])
        env = AssemblySchedulingEnv(config)
        obs = env.reset(instance, preference=PreferenceVector(*((1., 0., 0.), (.2, .3, .5), (0., 0., 1.))[index]))
        rows.append((obs.copy(), env.get_action_mask().copy()))
        for _ in range(30):
            obs, *_ = env.step(int(np.flatnonzero(~env.get_action_mask())[0]))
            if obs.decision_type == DecisionType.WORKER:
                rows.append((obs.copy(), env.get_action_mask().copy()))
                break
        else:
            pytest.fail("fixture did not reach a worker decision")
    return config, rows


@pytest.mark.parametrize("n", [0, 1, 9])
@pytest.mark.parametrize("intensity_enabled", [False, True])
def test_neighbor_attention_outputs_and_gradients(n, intensity_enabled):
    torch.manual_seed(48)
    reference = ConditionalAttention(16, 4).double()
    with torch.no_grad():
        reference.edge_score.uniform_(-.2, .2)
    optimized = deepcopy(reference)
    # Non-chain neighborhoods catch accidental hard-coding of three neighbors.
    adjacency = torch.rand(n, n) > .65
    adjacency.fill_diagonal_(True)
    intensity = torch.rand(n, n, dtype=torch.float64) if intensity_enabled else None
    indices, valid = _neighbor_rows(n, *np.nonzero(adjacency.numpy()))
    neighbors = torch.as_tensor(indices)
    valid = torch.as_tensor(np.c_[valid, np.ones(n, dtype=np.bool_)])
    packed_intensity = None
    if intensity_enabled:
        packed_intensity = torch.cat((intensity.gather(1, neighbors), intensity.mean(1, keepdim=True)), 1) if n else torch.empty(0, 2)
    x = torch.randn(n, 16, dtype=torch.float64, requires_grad=True)
    p = torch.randn(n, 16, dtype=torch.float64, requires_grad=True)
    xx, pp = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
    a = reference(x, p, adjacency, intensity)
    b = optimized.forward_neighbors(xx, pp, neighbors, valid, packed_intensity)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    weights = [torch.randn_like(t) for t in a]
    sum((t*w).sum() for t, w in zip(a, weights)).backward()
    sum((t*w).sum() for t, w in zip(b, weights)).backward()
    for expected, actual in ((x.grad, xx.grad), (p.grad, pp.grad)):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    for (name, expected), (_, actual) in zip(reference.named_parameters(), optimized.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-11, msg=name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_graph_outputs_gradients_and_batch_composition(samples, dtype):
    config, rows = samples
    torch.manual_seed(31)
    reference = build_agent(config, rows[0][0], "dcan").network.to(dtype=dtype)
    with torch.no_grad():
        for blocks in reference.blocks:
            for block in blocks.values():
                block.edge_score.uniform_(-.2, .2)
    optimized = deepcopy(reference)
    reference.execution_mode = "reference_dcan_v1"
    observations, masks = map(list, zip(*rows))
    a = reference.forward_batch(observations, masks, device="cpu")
    b = optimized.forward_batch(observations, masks, device="cpu")
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-11)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(actual, expected, **tolerance)
    mask = torch.ones_like(a[0], dtype=torch.bool)
    for i, m in enumerate(masks):
        mask[i, :len(m)] = torch.as_tensor(m)
    torch.manual_seed(19)
    weights = torch.randn_like(a[0]).masked_fill(mask, 0)
    ((a[0].masked_fill(mask, 0)*weights).sum()+a[1].square().sum()).backward()
    ((b[0].masked_fill(mask, 0)*weights).sum()+b[1].square().sum()).backward()
    for (name, expected), (_, actual) in zip(reference.named_parameters(), optimized.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, **tolerance, msg=name)
    with torch.no_grad():
        for i, (obs, m) in enumerate(rows):
            logits, value = optimized.forward_batch([obs], [m], device="cpu")
            torch.testing.assert_close(logits[0], b[0][i, :len(m)], **tolerance)
            torch.testing.assert_close(value[0], b[1][i], **tolerance)
        order = list(reversed(range(len(rows))))
        logits, values = optimized.forward_batch([observations[i] for i in order], [masks[i] for i in order], device="cpu")
        torch.testing.assert_close(logits, b[0][order], **tolerance)
        torch.testing.assert_close(values, b[1][order], **tolerance)
    assert optimized.network_spec() == reference.network_spec()
    assert list(optimized.state_dict()) == list(reference.state_dict())


def test_full_size_graph_and_empty_candidates(samples):
    config, rows = samples
    env = AssemblySchedulingEnv(config)
    obs = env.reset(load_instance_yaml(project_path("data/instances/v3/validation/C27.json")))
    masks = [env.get_action_mask()]
    agent = build_agent(config, obs, "dcan")
    with torch.no_grad():
        actual = agent.network.forward_batch([obs], masks, device="cpu")
        expected = agent.network._forward_batch_reference([obs], masks, device="cpu")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    # An empty production candidate relation still has a legal WAIT action.
    from environment import CAPABLE_EDGE
    empty = obs.copy()
    edge = empty.relations[CAPABLE_EDGE]
    empty.relations[CAPABLE_EDGE] = replace(edge, edge_index=edge.edge_index[:, :0], edge_features=edge.edge_features[:0])
    wait_only = np.ones_like(masks[0])
    wait_only[-1] = False
    with torch.no_grad():
        a = agent.network.forward_batch([empty, rows[1][0]], [wait_only, rows[1][1]], device="cpu")
        b = agent.network._forward_batch_reference([empty, rows[1][0]], [wait_only, rows[1][1]], device="cpu")
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_batch_validation_and_checkpoint_compatibility(samples, tmp_path):
    config, rows = samples
    agent = build_agent(config, rows[0][0], "dcan")
    for observations, masks, message in (([], [], "non-empty"), ([rows[0][0]], [], "aligned"),
                                         ([rows[0][0]], [np.zeros(2, dtype=bool)], "width"),
                                         ([rows[0][0]], [np.ones_like(rows[0][1])], "no legal")):
        with pytest.raises(ValueError, match=message):
            agent.network.forward_batch(observations, masks, device="cpu")
    agent.network.execution_mode = "reference_dcan_v1"
    agent.save(tmp_path/"reference.pt", {"algorithm": "dcan"})
    loaded = build_agent(config, rows[0][0], "dcan")
    loaded.load(tmp_path/"reference.pt", load_optimizer=True)
    assert loaded.network.execution_mode == "packed_neighbors_v1"
    assert all(torch.equal(v, loaded.network.state_dict()[k]) for k, v in agent.network.state_dict().items())


def test_empty_auxiliary_node_types(samples):
    config, rows = samples
    obs = rows[0][0].copy()
    for kind in ("order", "module"):
        obs.node_features[kind] = obs.node_features[kind][:0]
        for relation, edge in tuple(obs.relations.items()):
            if kind in (relation[0], relation[2]):
                obs.relations[relation] = replace(edge, edge_index=edge.edge_index[:, :0], edge_features=edge.edge_features[:0])
    network = build_agent(config, rows[0][0], "dcan").network
    with torch.no_grad():
        a = network.forward_batch([obs, rows[1][0]], [rows[0][1], rows[1][1]], device="cpu")
        b = network._forward_batch_reference([obs, rows[1][0]], [rows[0][1], rows[1][1]], device="cpu")
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_outputs_and_gradients(samples):
    config, rows = samples
    torch.manual_seed(31)
    reference = build_agent(config, rows[0][0], "dcan").network.cuda()
    with torch.no_grad():
        for blocks in reference.blocks:
            for block in blocks.values():
                block.edge_score.uniform_(-.2, .2)
    optimized = deepcopy(reference)
    reference.execution_mode = "reference_dcan_v1"
    obs, masks = map(list, zip(*rows))
    a = reference.forward_batch(obs, masks, device="cuda")
    b = optimized.forward_batch(obs, masks, device="cuda")
    for actual, expected in zip(b, a):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for logits, values in (a, b):
        legal = logits > torch.finfo(logits.dtype).min
        (logits[legal].square().sum()+values.square().sum()).backward()
    for (name, expected), (_, actual) in zip(reference.named_parameters(), optimized.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=1e-5, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_formal_minibatch_update(samples):
    from agent.ppo.buffer import RolloutBuffer

    config = deepcopy(samples[0])
    config["device"] = "cuda"
    env = AssemblySchedulingEnv(config)
    obs = env.reset(load_instance_yaml(project_path("data/instances/v3/validation/C27.json")))
    mask = env.get_action_mask()
    agent = build_agent(config, obs, "dcan")
    count = config["ppo"]["batch_size"]
    actions, logps, values = agent.act_batch([obs]*count, [mask]*count)
    buffer = RolloutBuffer()
    for i, (action, logp, value) in enumerate(zip(actions, logps, values)):
        buffer.add(obs, mask, action, logp, value, 0., False)
        buffer.transitions[-1].advantage = float(np.sin(i+.5))
        buffer.transitions[-1].return_value = value+.1*float(np.cos(i+.5))
    before = deepcopy(agent.network.state_dict())
    loss = agent.update(buffer)
    assert loss["optimizer_updates"] == config["ppo"]["epochs"] == 2
    assert loss["sample_presentations"] == 2*count == 128
    assert all(np.isfinite(v) for v in loss.values())
    assert any(not torch.equal(v, before[k]) for k, v in agent.network.state_dict().items())
