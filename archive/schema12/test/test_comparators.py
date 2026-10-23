from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent.comparators.common import method_config, FOUR, pooled_state, atomic_json
from agent.comparators.collection import collect
from agent.comparators.dcan import DCANActorCritic, ConditionalAttention
from agent.comparators.ddqn import RuleDDQN, rule_actions, double_q_targets
from agent.comparators.evolution import EvolutionSolver, crossover, mutate, weight_vectors, fronts, crowding
from agent.comparators.learning import build_agent, evaluate_request
from agent.mo_alns import SearchBudget, decode_solution
from agent.mo_alns.solver import _rule_solution
from environment import AssemblySchedulingEnv, PreferenceVector, DecisionType
from analysis.comparator_study import snapshot, plateau, clustered_interval


@pytest.fixture
def tiny_instance(fixed_instance):
    order = replace(fixed_instance.orders[0], release_time=0.)
    return replace(fixed_instance, instance_id="comparator_test", orders=(order,),
                   waves={k: {**v, "order_ids": [order.id] if k == order.wave else [], "release_interval": [0., 0.]} for k, v in fixed_instance.waves.items()})


@pytest.mark.parametrize("method", ["nsga2_ls", "moead"])
def test_evolution_paid_native_and_deterministic(config, tiny_instance, method):
    c = method_config(config, method)
    c["comparison"]["evolution"] = {"population_size": 8}
    solver = EvolutionSolver(c, method)
    first = solver.solve(tiny_instance, PreferenceVector(*FOUR[-1]), budget=SearchBudget(max_evaluations=40))
    second = solver.solve(tiny_instance, PreferenceVector(*FOUR[-1]), budget=SearchBudget(max_evaluations=40))
    assert first.selected is not None
    assert first.selected.actions == second.selected.actions
    assert first.environment_evaluations <= 40
    assert first.selected.tchebycheff <= first.initial_best_tchebycheff+1e-12
    assert first.operator_statistics["completed_generations" if method == "nsga2_ls" else "completed_scans"] >= 1
    if method == "nsga2_ls":
        assert first.operator_statistics["local_search_evaluations"] > 0
    replay = decode_solution(c, tiny_instance, first.selected.solution, PreferenceVector(*FOUR[-1]))
    assert replay.action_trace_sha256 == first.selected.action_trace_sha256


def test_crossovers_and_mutations_clear_trace(config, tiny_instance):
    rng = random.Random(11)
    first = _rule_solution(tiny_instance, "earliest_finish", rng)
    second = _rule_solution(tiny_instance, "configuration_reuse", rng)
    first.replay_actions = (0,)
    child = crossover(first, second, tiny_instance, rng)
    assert child.replay_actions is None
    child.validate(tiny_instance)
    for kind in range(6):
        mutated = mutate(first, tiny_instance, rng, kind)
        mutated.validate(tiny_instance)
        assert mutated.replay_actions is None
    assert first.replay_actions == (0,)


def test_weights_include_endpoints_request_and_simplex():
    weights = weight_vectors(16, PreferenceVector(.2, .3, .5))
    assert np.allclose(weights.sum(1), 1)
    for point in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1/3,)*3, (.2, .3, .5)]:
        assert any(np.allclose(point, w) for w in weights)
    assert len({tuple(w) for w in weights}) == 16


def test_fronts_and_crowding_known_answer():
    points = [(0, 2, 2), (1, 1, 1), (2, 0, 2), (3, 3, 3)]
    pool = [SimpleNamespace(feasible=True, objectives=p, normalized_objectives=p, action_trace_sha256=str(i)) for i, p in enumerate(points)]
    assert fronts(pool) == [[0, 1, 2], [3]]
    assert np.isinf(crowding(pool, [0, 1, 2])[0])


def test_conditional_attention_formula_and_permutation():
    torch.manual_seed(11)
    layer = ConditionalAttention(8, 2)
    h, p = torch.randn(3, 8), torch.randn(3, 8)
    adjacency = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.bool)
    out, _ = layer(h, p, adjacency)
    gamma, beta = layer.modulation(p).chunk(2, -1)
    nodes = layer.transform(h*gamma+beta).reshape(3, 2, 4)
    pref = layer.transform(p).reshape(3, 2, 4)
    manual = []
    for i in range(3):
        keys = torch.cat((nodes[adjacency[i]], pref[i:i+1]), 0)
        score = torch.nn.functional.leaky_relu((nodes[i]*layer.score[:, :4]).sum(-1)[None]+(keys*layer.score[:, 4:]).sum(-1), .2)
        manual.append(torch.nn.functional.elu((score.softmax(0)[..., None]*keys).sum(0).flatten()))
    assert torch.allclose(out, torch.stack(manual), atol=1e-6)
    order = torch.tensor([2, 0, 1])
    reordered, _ = layer(h[order], p[order], adjacency[order][:, order])
    assert torch.allclose(out[order], reordered, atol=1e-6)
    assert not torch.allclose(out, layer(h, p+1, adjacency)[0])
    out.sum().backward()
    assert all(torch.isfinite(param.grad).all() for param in layer.parameters() if param.grad is not None)


@pytest.mark.parametrize("method", ["ppo", "dcan", "rule_ddqn"])
def test_learning_deliver_checkpoint_contract(config, tiny_instance, method, tmp_path):
    torch.set_num_threads(1)
    c = method_config(config, method)
    env = AssemblySchedulingEnv(c)
    obs = env.reset(tiny_instance)
    agent = build_agent(c, obs, method)
    result = evaluate_request(c, agent, tiny_instance, FOUR[-1], method, mode="greedy")
    assert result["summary"]["candidate_count"] == 1
    assert all(not r["violations"] for r in result["raw"])
    path = tmp_path/f"{method}.pt"
    agent.save(path, {"algorithm": method})
    restored = build_agent(c, obs, method)
    restored.load(path)
    again = evaluate_request(c, restored, tiny_instance, FOUR[-1], method, mode="greedy")
    assert [r["trace"] for r in result["raw"]] == [r["trace"] for r in again["raw"]]
    if method == "dcan":
        wrong = build_agent(method_config(config, "ppo"), obs, "ppo")
        with pytest.raises(ValueError):
            wrong.load(path)


def test_rules_legal_unique_across_phases(config, tiny_instance):
    target = tiny_instance.operations[0].required_module
    tiny_instance = replace(tiny_instance, machines=tuple(replace(m, initial_module=next((module for module in m.module_parameters if module != target), m.initial_module)) for m in tiny_instance.machines))
    env = AssemblySchedulingEnv(config)
    obs = env.reset(tiny_instance)
    phases = set()
    for step in range(300):
        if env.terminated or env.truncated:
            break
        mask = env.get_action_mask()
        rules = rule_actions(obs, mask)
        actions = rules[rules >= 0]
        assert len(set(actions)) == len(actions)
        assert not mask[actions].any()
        phases.add(obs.decision_type)
        # Earliest available non-WAIT rule normally triggers both phases.
        obs, _, _, _, _ = env.step(int(actions[0]))
    assert DecisionType.PRODUCTION in phases and DecisionType.WORKER in phases


def test_double_q_and_truncation_bootstrap():
    online = torch.tensor([[1., 9., 3.], [8., 2., 4.], [9., 8., 7.]])
    target = torch.tensor([[20., 10., 5.], [40., 30., 50.], [5., 5., 5.]])
    mask = torch.tensor([[False, True, False], [False, False, False], [True, True, True]])
    result = double_q_targets(torch.tensor([1., 2., 3.]), torch.tensor([False, True, False]), online, target, mask)
    assert result.tolist() == [6., 2., 3.]
    # An external truncation has terminated=False and bootstraps normally.
    result = double_q_targets(torch.tensor([1.]), torch.tensor([False]), online[:1], target[:1], mask[:1])
    assert result.item() == 6.


def test_ddqn_update_and_target_sync(config, tiny_instance):
    c = method_config(config, "rule_ddqn")
    c["comparison"]["ddqn"] = {"warmup": 2, "batch_size": 2, "update_every": 1, "sync_every": 1, "capacity": 3}
    env = AssemblySchedulingEnv(c)
    obs = env.reset(tiny_instance)
    mask = env.get_action_mask()
    agent = RuleDDQN(obs, c)
    original = [p.detach().clone() for p in agent.online.parameters()]
    rule = int(np.flatnonzero(rule_actions(obs, mask) >= 0)[0])
    for _ in range(4):
        agent.remember(obs, rule, -.1, obs, mask, False)
    assert agent.updates == 3 and len(agent.replay) == 3
    assert any(not torch.equal(a, b) for a, b in zip(original, agent.online.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))


def test_exact_global_allowance_and_bootstrap(config, tiny_instance):
    from agent.ppo.parallel import ParallelEpisodeRunner
    c = method_config(config, "dcan")
    env = AssemblySchedulingEnv(c)
    agent = build_agent(c, env.reset(tiny_instance), "dcan")
    with ParallelEpisodeRunner(config=c, template=tiny_instance, episode_count=20, worker_count=2) as runner:
        batch = collect(runner, agent, c, 0, 3)
    assert batch.steps == 3 and len(batch.buffer) == 3
    assert sum(r["steps"] for r in batch.episodes) == 3
    assert any(r["budget_truncated"] for r in batch.episodes)
    assert any(t.return_value != t.reward for t in batch.buffer.transitions if not t.done)


def test_analysis_cutoff_and_cluster_grain():
    job = {"candidates": [{"completed_seconds": 1., "feasible": True, "quality": .5, "objectives": [1, 2, 3], "action_trace_sha256": "a"},
                           {"completed_seconds": 3., "feasible": True, "quality": .1, "objectives": [0, 1, 1], "action_trace_sha256": "b"}], "events": []}
    assert snapshot(job, 2)["quality"] == .5
    assert not snapshot(job, .5)["success"]
    rows = [{"instance_id": "a", "value": 1.}]*10+[{"instance_id": "b", "value": 3.}]
    assert clustered_interval(rows, lambda r: r["value"], samples=100)["mean"] == 2.
    assert not plateau([{"pressure_type": "a", "quality": 1., "success": False}], [{"pressure_type": "a", "quality": .999, "success": True}])


def test_dcan_worker_and_wait_only(config, tiny_instance):
    target = tiny_instance.operations[0].required_module
    instance = replace(tiny_instance, machines=tuple(replace(m, initial_module=next((module for module in m.module_parameters if module != target), m.initial_module)) for m in tiny_instance.machines))
    env = AssemblySchedulingEnv(config)
    obs = env.reset(instance)
    network = DCANActorCritic(obs, config)
    for _ in range(30):
        if obs.decision_type == DecisionType.WORKER:
            break
        actions = rule_actions(obs, env.get_action_mask())
        obs, *_ = env.step(int(actions[actions >= 0][0]))
    assert obs.decision_type == DecisionType.WORKER
    mask = env.get_action_mask()
    logits, values = network.forward_batch([obs, obs], [mask, mask], device="cpu")
    assert torch.isfinite(values).all()
    assert not mask[int(logits[0].argmax())]
    forced = np.ones_like(mask)
    forced[-1] = False
    wait_logits, _ = network.forward_batch([obs], [forced], device="cpu")
    assert int(wait_logits[0].argmax()) == len(mask)-1


def test_dcan_native_operation_relabeling(config, tiny_instance):
    env = AssemblySchedulingEnv(config)
    obs = env.reset(tiny_instance)
    network = DCANActorCritic(obs, config)
    mask = env.get_action_mask()
    permuted = obs.copy()
    order = np.arange(len(obs.node_features["operation"]))[::-1].copy()
    inverse = np.argsort(order)
    permuted.node_features["operation"] = permuted.node_features["operation"][order]
    permuted.node_ids["operation"] = tuple(permuted.node_ids["operation"][i] for i in order)
    for kind, edge in permuted.relations.items():
        ix = edge.edge_index.copy()
        if kind[0] == "operation":
            ix[0] = inverse[ix[0]]
        if kind[2] == "operation":
            ix[1] = inverse[ix[1]]
        permuted.relations[kind] = replace(edge, edge_index=ix)
    width = len(obs.node_features["machine"])
    permuted_mask = np.concatenate((mask[:-1].reshape(-1, width)[order].flatten(), mask[-1:]))
    before, value = network.forward_batch([obs], [mask], device="cpu")
    after, next_value = network.forward_batch([permuted], [permuted_mask], device="cpu")
    expected = torch.cat((before[0, :-1].reshape(-1, width)[order.copy()].flatten(), before[0, -1:]))
    assert torch.allclose(after[0], expected, atol=1e-6)
    assert torch.allclose(value, next_value, atol=1e-6)


def test_subsets_disjoint_and_legacy_hash(config, tmp_path, monkeypatch):
    from configs.runtime import runtime_manifest
    from scripts.comparison_suite import prepare
    raw = deepcopy(config)
    before = runtime_manifest(raw)
    assert "comparison" not in before
    current = method_config(raw, "dcan")
    assert current["runtime_manifest"]["comparison"]["method"] == "dcan"
    assert before == runtime_manifest(raw)
    _, subsets, _ = prepare(current, tmp_path/"study")
    assert [len(subsets[k]) for k in ("tuning", "selection", "confirmation")] == [21, 21, 7]
    assert not set(subsets["tuning"]) & set(subsets["selection"])
    assert not set(subsets["tuning"]) & set(subsets["confirmation"])
    assert not set(subsets["selection"]) & set(subsets["confirmation"])


def test_complete_synthetic_report(config, tmp_path):
    from analysis.comparator_study import report
    from configs.config import public_config
    arms = ["ppo", "ppo_greedy", "dcan", "dcan_greedy", "rule_ddqn", "nsga2_ls", "moead", "mo_alns",
            "nsga2_ls_ppo_deadline", "moead_ppo_deadline", "mo_alns_ppo_deadline"]
    atomic_json(tmp_path/"identity.json", {"config": public_config(config)})
    for arm in arms:
        for i in range(7):
            for j in range(13):
                for seed in (11, 23, 37):
                    row = {"method": arm, "instance_id": f"synthetic_{i}", "preference": [j/12, 1-j/12, 0.], "seed": seed,
                           "success": True, "quality": .1, "objectives": [1., 2., 3.], "normalized_objectives": [.1, .2, .3],
                           "search_seconds": 1., "initial_quality": .2}
                    atomic_json(tmp_path/"confirmation"/arm/f"{i}_{j}_{seed}.json", {"summary": row,
                        "raw": [{"success": True, "objectives": [1., 2., 3.]}]})
    summary = report(tmp_path, arms)
    assert len(summary["methods"]) == 11
    assert (tmp_path/"paid_archive_hv.csv").exists()
    assert (tmp_path/"comparison.svg").exists()
    assert summary["independent_search_improvement_confirmed"]["nsga2_ls"]
    (tmp_path/"confirmation"/"ppo"/"0_0_11.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        report(tmp_path, arms)


def test_search_budget_selection_uses_mechanisms_and_prefixes(tmp_path):
    from data.distribution import PRESSURE_TYPES
    from scripts.comparison_suite import select_search_budget
    for i, p in enumerate(PRESSURE_TYPES):
        row = {"instance_id": f"synthetic_{i}", "pressure_type": p, "seed": 11, "preference": FOUR[-1],
               "initial_quality": .5, "quality": .4}
        candidates = [{"completed_seconds": 1., "feasible": True, "quality": .5, "objectives": [1, 2, 3], "action_trace_sha256": "a"},
                      {"completed_seconds": 30., "feasible": True, "quality": .4, "objectives": [1, 2, 2], "action_trace_sha256": "b"}]
        atomic_json(tmp_path/"curves_600"/"nsga2_ls"/f"{i}.json", {"summary": row, "candidates": candidates,
            "events": [{"kind": "iteration", "completed_seconds": 40., "completed_generations": 5, "local_search_evaluations": 2}]})
    selected = select_search_budget(tmp_path, "nsga2_ls", expected=7)
    assert selected["selected_seconds"] == 60
    assert not selected["needs_extension"]
