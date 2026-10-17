"""Population baselines using the project's paid native scheduling decoder."""
from dataclasses import replace
import hashlib
import math
import random

import numpy as np

from agent.baselines import HeuristicPolicy
from agent.mo_alns import SearchBudget, BudgetExhausted
from agent.mo_alns.archive import ParetoArchive
from agent.mo_alns.search import CandidateEvaluator
from agent.mo_alns.solver import INITIAL_RULES, _rule_solution, _topological_order
from agent.mo_alns.types import SearchResult, GridSearchResult, preference_key
from environment import normalize_preference, simplex_lattice


def dominates(a, b):
    if a.feasible != b.feasible:
        return a.feasible
    if not a.feasible:
        return a.completion_rank < b.completion_rank
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(
        x < y for x, y in zip(a.objectives, b.objectives))


def fronts(population):
    remaining = list(range(len(population)))
    result = []
    while remaining:
        front = [i for i in remaining if not any(dominates(population[j], population[i]) for j in remaining if i != j)]
        result.append(front)
        chosen = set(front)
        remaining = [i for i in remaining if i not in chosen]
    return result


def crowding(population, indices):
    distances = {i: 0.0 for i in indices}
    for objective in range(3):
        ordered = sorted(indices, key=lambda i: (population[i].normalized_objectives[objective], population[i].action_trace_sha256))
        if not ordered:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = math.inf
        span = population[ordered[-1]].normalized_objectives[objective] - population[ordered[0]].normalized_objectives[objective]
        if span > 1e-15:
            for j in range(1, len(ordered)-1):
                distances[ordered[j]] += (population[ordered[j+1]].normalized_objectives[objective]
                                          - population[ordered[j-1]].normalized_objectives[objective]) / span
    return distances


def ranked(population):
    return [(i, rank, distance) for rank, front in enumerate(fronts(population))
            for i, distance in crowding(population, front).items()]


def survive(population, size):
    # Prefer distinct trajectories. Fill remaining slots only if the search has
    # not yet discovered enough distinct traces; duplicate slots are explicit.
    unique = {c.action_trace_sha256: c for c in reversed(population)}
    pool = list(unique.values())
    order = sorted(ranked(pool), key=lambda t: (t[1], -t[2], pool[t[0]].action_trace_sha256))
    selected = [pool[i] for i, _, _ in order[:size]]
    if selected:
        distinct = list(selected)
        while len(selected) < size:
            selected.append(distinct[(len(selected)-len(distinct)) % len(distinct)])
    return selected


def topology(instance, order):
    priorities = {opid: (i,) for i, opid in enumerate(order)}
    return _topological_order(instance, priorities)


def crossover(first, second, instance, rng):
    child = first.clone(origin="pox")
    child.replay_actions = None
    jobs = [o.id for o in instance.orders]
    chosen = set(rng.sample(jobs, rng.randint(1, len(jobs))))
    byid = {op.id: op for op in instance.operations}
    retained = {opid for opid in first.operation_order if byid[opid].order_id in chosen}
    rest = iter(x for x in second.operation_order if x not in retained)
    child.operation_order = topology(instance, tuple(x if x in retained else next(rest) for x in first.operation_order))
    for opid in child.operation_order:
        donor = first if rng.random() < .5 else second
        for field in ("machine_rankings", "disassembly_worker_rankings", "installation_worker_rankings",
                      "production_wait", "disassembly_wait", "installation_wait"):
            getattr(child, field)[opid] = getattr(donor, field)[opid]
        if opid in donor.resource_modes:
            child.resource_modes[opid] = donor.resource_modes[opid]
        else:
            child.resource_modes.pop(opid, None)
    return child


def mutate(solution, instance, rng, kind=None, opid=None):
    kind = rng.randrange(6) if kind is None else kind
    child = solution.clone(origin=f"mutation_{kind}")
    child.replay_actions = None
    opid = rng.choice(child.operation_order) if opid is None else opid
    if kind == 0:
        order = list(child.operation_order)
        order.remove(opid)
        order.insert(rng.randrange(len(order)+1), opid)
        child.operation_order = topology(instance, order)
    elif kind in (1, 2, 3):
        field = ("machine_rankings", "disassembly_worker_rankings", "installation_worker_rankings")[kind-1]
        values = list(getattr(child, field)[opid])
        if len(values) >= 2:
            a, b = rng.sample(range(len(values)), 2)
            values[a], values[b] = values[b], values[a]
        getattr(child, field)[opid] = tuple(values)
        child.resource_modes.pop(opid, None)
    elif kind == 4:
        field = rng.choice(("production_wait", "disassembly_wait", "installation_wait"))
        getattr(child, field)[opid] = not getattr(child, field)[opid]
    else:
        modes = [None, "earliest_finish", "minimum_reconfiguration_cost", "minimum_workload_delta", "lowest_predicted_fatigue"]
        modes.remove(child.resource_modes.get(opid))
        mode = rng.choice(modes)
        if mode is None:
            child.resource_modes.pop(opid, None)
        else:
            child.resource_modes[opid] = mode
    child.validate(instance)
    return child


def weight_vectors(size, preference):
    chosen = [np.eye(3)[i] for i in range(3)] + [np.full(3, 1/3)]
    p = np.asarray(preference.as_tuple())
    if not any(np.allclose(p, w, atol=1e-12, rtol=0) for w in chosen):
        chosen.append(p)
    grid = sorted((np.asarray(p.as_tuple() if hasattr(p, "as_tuple") else p) for p in simplex_lattice(12, include=())), key=lambda p: tuple(p))
    while len(chosen) < size:
        distances = [min(np.linalg.norm(p-w) for w in chosen) for p in grid]
        chosen.append(grid.pop(int(np.argmax(distances))))
    return np.stack(chosen[:size])


def decomposition(candidate, weight, ideal):
    return float(np.max(np.maximum(weight, 1e-6) * np.abs(np.asarray(candidate.normalized_objectives)-ideal)))


class EvolutionSolver:
    def __init__(self, config, method, *, algorithm_seed=11, dataset_name=""):
        if method not in ("nsga2_ls", "moead"):
            raise ValueError("unknown population method")
        self.config, self.method, self.seed, self.dataset = config, method, algorithm_seed, dataset_name
        self.settings = config.get("comparison", {}).get("evolution", {})

    def solve(self, instance, preference, *, budget=None):
        preference = normalize_preference(preference)
        budget = budget or SearchBudget(seconds=self.settings.get("seconds", 600))
        token = f"{self.method}|{self.seed}|{self.dataset}|{instance.instance_id}|{preference_key(preference)}"
        rng = random.Random(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big"))
        initial_token = f"population_initial|{self.seed}|{self.dataset}|{instance.instance_id}|{preference_key(preference)}"
        initial_rng = random.Random(int.from_bytes(hashlib.sha256(initial_token.encode()).digest()[:8], "big"))
        size = int(self.settings.get("population_size", 32))
        if size < 5:
            raise ValueError("population_size must be at least 5")
        if size > 64:
            raise ValueError("comparison v1 population_size must not exceed 64")
        if int(self.settings.get("neighbors", 10)) < 2 or int(self.settings.get("maximum_replacements", 2)) < 1:
            raise ValueError("neighbors must be >=2 and maximum_replacements >=1")
        for name, default in (("crossover_probability", .9), ("mutation_probability", .2), ("neighbor_mating_probability", .9)):
            if not 0 <= float(self.settings.get(name, default)) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        evaluator = CandidateEvaluator(self.config, instance, preference, budget=budget)
        initial, population, events = [], [], []
        stats = {"completed_generations": 0, "completed_scans": 0, "local_search_evaluations": 0,
                 "replacements": 0, "proposals": 0, "mutation_uses": [0]*6,
                 "initialization_complete": False, "population_target": size}
        weights = weight_vectors(size, preference)
        neighbors = np.argsort(np.linalg.norm(weights[:, None]-weights[None, :], axis=-1), axis=1)[:, :min(int(self.settings.get("neighbors", 10)), size)]
        ideal = np.full(3, math.inf)
        reason, initial_seconds = "evaluation_limit", 0.0
        def evaluate(solution, policy=None):
            candidate = evaluator.evaluate(solution, policy=policy)
            if candidate is None:
                raise BudgetExhausted("evaluation_limit")
            if candidate.feasible:
                np.minimum(ideal, candidate.normalized_objectives, out=ideal)
            return candidate
        def breed(a, b):
            child = crossover(a.solution, b.solution, instance, rng) if rng.random() < float(self.settings.get("crossover_probability", .9)) else a.solution.clone(origin="reproduction")
            child.replay_actions = None
            if rng.random() < float(self.settings.get("mutation_probability", .2)):
                kind = rng.randrange(6)
                stats["mutation_uses"][kind] += 1
                child = mutate(child, instance, rng, kind)
            stats["proposals"] += 1
            return child
        try:
            for rule in INITIAL_RULES:
                solution = _rule_solution(instance, "earliest_finish" if rule == "existing_heuristic" else rule, initial_rng)
                solution.origin = rule
                initial.append(evaluate(solution, HeuristicPolicy() if rule == "existing_heuristic" else None))
            population = list(initial)
            while len(population) < size:
                solution = _rule_solution(instance, "same_wave_randomized", rng)
                solution.operation_order = topology(instance, rng.sample(list(solution.operation_order), len(solution.operation_order)))
                for field in ("machine_rankings", "disassembly_worker_rankings", "installation_worker_rankings"):
                    for opid, ranking in getattr(solution, field).items():
                        getattr(solution, field)[opid] = tuple(rng.sample(list(ranking), len(ranking)))
                candidate = evaluate(solution)
                population.append(candidate)
                initial.append(candidate)
            population = population[:size]
            initial = list(population)
            stats["initialization_complete"] = True
            initial_seconds = budget.elapsed
            stalled = 0
            while population:
                budget.check_time()
                before = evaluator.evaluation_count
                if self.method == "nsga2_ls":
                    ranks = {i: (rank, -distance) for i, rank, distance in ranked(population)}
                    def tournament():
                        a, b = rng.sample(range(len(population)), 2) if len(population)>1 else (0, 0)
                        return population[min((a, b), key=lambda i: (ranks[i], population[i].action_trace_sha256))]
                    offspring = [evaluate(breed(tournament(), tournament())) for _ in range(size)]
                    # Diagnostic guided local moves: expensive orders, costly
                    # reconfigurations and busy workers, with all six move types.
                    elite = survive(population+offspring, size)[:math.ceil(size/8)]
                    for candidate in elite:
                        scores = candidate.diagnostics.get("operation_reconfiguration_cost", {})
                        flow = candidate.diagnostics.get("order_flow", {})
                        loads = candidate.diagnostics.get("worker_load", {})
                        op_order = {op.id: op.order_id for op in instance.operations}
                        workers = candidate.diagnostics.get("worker_tasks", {})
                        busy = set(workers.get(max(loads, key=loads.get), ())) if loads else set()
                        order = sorted(candidate.solution.operation_order,
                            key=lambda op: (-scores.get(op, 0), -flow.get(op_order[op], 0), op not in busy, op))
                        for offset in range(2):
                            move = (stats["completed_generations"]*2+offset) % 6
                            count = evaluator.evaluation_count
                            offspring.append(evaluate(mutate(candidate.solution, instance, rng, move, order[offset % len(order)])))
                            stats["local_search_evaluations"] += evaluator.evaluation_count-count
                    population = survive(population+offspring, size)
                    stats["completed_generations"] += 1
                else:
                    for i in range(size):
                        pool = list(neighbors[i]) if rng.random() < float(self.settings.get("neighbor_mating_probability", .9)) else list(range(size))
                        a, b = rng.sample(pool, 2)
                        candidate = evaluate(breed(population[a], population[b]))
                        replaced = 0
                        rng.shuffle(pool)
                        for j in pool:
                            other = population[j]
                            better = (candidate.feasible and not other.feasible) or (
                                candidate.feasible and other.feasible and decomposition(candidate, weights[j], ideal) < decomposition(other, weights[j], ideal)-1e-12) or (
                                not candidate.feasible and not other.feasible and candidate.completion_rank < other.completion_rank)
                            if better:
                                population[j] = candidate
                                stats["replacements"] += 1
                                replaced += 1
                                if replaced == int(self.settings.get("maximum_replacements", 2)):
                                    break
                    stats["completed_scans"] += 1
                events.append({"kind": "iteration", "completed_seconds": budget.elapsed, **stats, "mutation_uses": list(stats["mutation_uses"]),
                               "population_distinct": len({c.action_trace_sha256 for c in population})})
                stalled = stalled+1 if before == evaluator.evaluation_count else 0
                if stalled >= 100:
                    reason = "no_new_evaluation_proposals"
                    break
        except BudgetExhausted as error:
            reason = str(error)
        initial_seconds = initial_seconds or budget.elapsed
        best_initial = min((c.tchebycheff for c in initial if c.feasible), default=None)
        selected = evaluator.archive.best(preference) if evaluator.archive.entries else None
        return SearchResult(preference, selected, evaluator.archive.snapshots(), best_initial,
            evaluator.evaluation_count, evaluator.cache_hits, stats["proposals"], budget.elapsed, stats,
            tuple(events), status="completed" if selected else "no_feasible_solution", stop_reason=reason,
            accounting=budget.counters(), candidate_history=tuple(evaluator.history), initial_seconds=initial_seconds,
            initial_candidate_count=len(initial))

    def solve_grid(self, instance, preferences):
        points = tuple(normalize_preference(p) for p in preferences)
        requests = tuple(self.solve(instance, p) for p in points)
        archive = ParetoArchive(self.config)
        for result in requests:
            archive.extend(result.archive)
        return GridSearchResult(points, {preference_key(r.preference): r.selected for r in requests}, archive.snapshots(), requests)
