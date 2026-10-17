"""Bounded ALNS search; every exact simulation passes one evaluator."""
from dataclasses import dataclass
import math
import random

from agent.baselines import HeuristicPolicy
from .archive import ParetoArchive, augmented_tchebycheff, candidate_better
from .budget import SearchBudget, BudgetExhausted
from .projection import SerialPredictor, MODES
from .types import SearchResult


class CandidateEvaluator:
    def __init__(self, config, instance, preference, maximum_evaluations=None, *, budget=None):
        self.config, self.instance, self.preference = config, instance, preference
        self.budget = budget or SearchBudget(max_evaluations=maximum_evaluations)
        self.maximum_evaluations = self.budget.max_evaluations or math.inf
        self.cache, self.history = {}, []
        self.cache_hits = 0
        self.archive = ParetoArchive(dict(config))
        self.best = None
        self.predictor = None

    @property
    def evaluation_count(self):
        return self.budget.completed_evaluations

    def evaluate(self, solution, *, policy=None):
        from .solver import decode_solution
        self.budget.check_time()
        digest = solution.digest()
        if policy is None and digest in self.cache:
            self.cache_hits += 1
            return self.cache[digest]
        try:
            self.budget.begin_evaluation()
        except BudgetExhausted as exc:
            if str(exc) == "evaluation_limit":
                return None
            raise
        try:
            result = decode_solution(self.config, self.instance, solution, self.preference,
                                     capture_logs=True, budget=self.budget, policy=policy)
            self.budget.check_time()
        except BudgetExhausted:
            self.budget.aborted_evaluations += 1
            raise
        self.budget.completed_evaluations += 1
        if policy is not None:
            # A trace-backed initial candidate keeps the actual dispatching policy.
            result.solution.replay_actions = result.actions
            ordered = tuple(result.realized["machines"])
            result.solution.operation_order = ordered + tuple(x for x in solution.operation_order if x not in ordered)
            try:
                result.solution.validate(self.instance)
            except ValueError:
                # Incomplete heuristic: its replay is still exact, but topology for
                # subsequent proposals must use a valid complete ordering.
                result.solution.operation_order = solution.operation_order
        result.completed_seconds = self.budget.elapsed
        self.archive.update(result)
        if self.best is None or candidate_better(result, self.best):
            self.best = result
        self.history.append({"kind": "candidate", "evaluation": self.evaluation_count,
                             "completed_seconds": result.completed_seconds,
                             "origin": solution.origin, "solution_digest": result.solution_digest,
                             "action_trace_sha256": result.action_trace_sha256,
                             "feasible": result.feasible, "objectives": result.objectives,
                             "quality": result.tchebycheff, "decisions": len(result.actions)})
        if len(self.cache) >= 512:
            self.cache.pop(next(iter(self.cache)))
        if policy is None:
            self.cache[digest] = result
        else:
            self.cache[result.solution_digest] = result
        return result

    def predict(self, solution):
        if self.predictor is None:
            self.predictor = SerialPredictor(self.config, self.instance, self.budget)
        return self.predictor.predict(solution)


def regret_value(costs, rank):
    """Classical regret-k over distinct insertion alternatives."""
    return sum(value - costs[0] for value in costs[1:rank]) if costs else 0.0


def insertion_signature(solution, prediction):
    """Collapse modes that predict exactly the same insertion decisions."""
    return (solution.operation_order, prediction.objectives,
            tuple((op, tuple(sorted(assignment.items()))) for op, assignment in sorted(prediction.assignments.items())))


@dataclass
class RepairOutcome:
    solution: object
    probes: tuple = ()


def repair_solution(name, base, removed, instance, candidate, evaluator, settings):
    from .solver import _valid_positions, _sample_positions, _insert, _complete_partial_order
    original = base.operation_order
    working = base.clone(origin=name)
    working.replay_actions = None
    remaining = list(removed)
    working.operation_order = tuple(x for x in original if x not in set(remaining))
    probe_limit = int(settings.get("max_proxy_evaluations_per_neighbourhood", 24))
    proxy_attempts = 0
    while remaining:
        evaluator.budget.check_time()
        choices = []
        for op in remaining:
            evaluator.budget.check_time()
            lo, hi = _valid_positions(working.operation_order, op, instance)
            positions = _sample_positions(lo, hi, int(settings["regret_position_candidates"]))
            modes = MODES[:int(settings["regret_resource_variants"])] if name.startswith("regret") else (name,)
            options, seen, effective = [], set(), set()
            for pos in positions:
                for mode in modes:
                    if proxy_attempts >= probe_limit:
                        break
                    proxy_attempts += 1
                    partial = working.clone(origin=name)
                    partial.operation_order = _insert(working.operation_order, op, pos)
                    partial.resource_modes[op] = mode
                    partial.production_wait[op] = False
                    partial.disassembly_wait[op] = False
                    partial.installation_wait[op] = False
                    complete = partial.clone()
                    complete.operation_order = _complete_partial_order(partial.operation_order, original, instance)
                    digest = complete.digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    predicted = evaluator.predict(complete)
                    signature = insertion_signature(complete, predicted)
                    if signature in effective:
                        continue
                    effective.add(signature)
                    quality = augmented_tchebycheff(predicted.objectives, evaluator.preference, evaluator.config)
                    options.append(((not predicted.feasible, quality, digest), partial))
                if proxy_attempts >= probe_limit:
                    break
            if options:
                options.sort(key=lambda x: x[0])
                costs = [float(key[0]) + key[1] for key, _ in options]
                regret = regret_value(costs, 2 if name == "regret_2" else 3) if name.startswith("regret") else 0
                choices.append((regret, -costs[0], op, options[0][1]))
            if proxy_attempts >= probe_limit:
                break
        if not choices:
            # No more proxy work: complete every remaining insertion, without
            # spending the exact evaluation reserved for this main repair.
            for op in remaining:
                lo, _ = _valid_positions(working.operation_order, op, instance)
                working.operation_order = _insert(working.operation_order, op, lo)
                working.resource_modes[op] = "earliest_finish" if name.startswith("regret") else name
                working.production_wait[op] = False
                working.disassembly_wait[op] = False
                working.installation_wait[op] = False
            break
        _, _, op, working = max(choices, key=lambda x: x[:3])
        remaining.remove(op)
    working.operation_order = _complete_partial_order(working.operation_order, original, instance)
    working.validate(instance)
    return RepairOutcome(working)


def alternatives(solution, removed, evaluator, rng):
    """A bounded set of genuinely distinct complete resource/wait moves."""
    choices = []
    for _ in range(4):
        value = solution.clone(origin="local_probe")
        op = rng.choice(list(removed))
        variant = rng.randrange(4)
        if variant == 0:
            values = value.machine_rankings[op]
            value.machine_rankings[op] = values[1:] + values[:1]
            value.resource_modes.pop(op, None)
        elif variant == 1:
            rankings = rng.choice([value.disassembly_worker_rankings, value.installation_worker_rankings])
            values = rankings[op]
            rankings[op] = values[1:] + values[:1]
            value.resource_modes.pop(op, None)
        elif variant == 2:
            waits = rng.choice([value.production_wait, value.disassembly_wait, value.installation_wait])
            waits[op] = not waits[op]
        else:
            value.resource_modes[op] = rng.choice(MODES)
        choices.append(value)
    ranked, seen = [], {solution.digest()}
    for value in choices:
        evaluator.budget.check_time()
        digest = value.digest()
        if digest in seen:
            continue
        seen.add(digest)
        pred = evaluator.predict(value)
        ranked.append(((not pred.feasible, augmented_tchebycheff(pred.objectives, evaluator.preference, evaluator.config), digest), value))
    return [value for _, value in sorted(ranked, key=lambda x: x[0])]


def solve(solver, instance, preference, *, budget=None):
    from .solver import (INITIAL_RULES, DESTROY_OPERATORS, REPAIR_OPERATORS, _rule_solution,
                         derive_mo_alns_seed, _roulette, _update_weights,
                         destroy_operations, matching_safe_repair)
    settings = solver.settings
    budget = budget or SearchBudget(settings["max_evaluations_per_preference"], settings.get("max_seconds_per_preference"))
    rng = random.Random(derive_mo_alns_seed(solver.algorithm_seed, instance.instance_id, preference, solver.dataset_name))
    evaluator = CandidateEvaluator(solver.config, instance, preference, budget=budget)
    initial, logs, initial_seconds = [], [], 0.0
    current = None
    dw, rw = {x: 1.0 for x in DESTROY_OPERATORS}, {x: 1.0 for x in REPAIR_OPERATORS}
    ds, rs, du, ru = {}, {}, {}, {}
    totals = {"destroy_uses": {}, "repair_uses": {}, "repair_evaluations": {}, "repair_scores": {}}
    proposals = completed = updates = no_new = stale = reheats = accepted_worse = 0
    temperature, initial_temperature = 0.01, 0.01
    reason = "evaluation_limit"
    local_limit = int(settings["max_evaluations_per_neighbourhood"])
    try:
        for rule in INITIAL_RULES:
            budget.check_time()
            solution = _rule_solution(instance, "earliest_finish" if rule == "existing_heuristic" else rule, rng)
            solution.origin = rule
            candidate = evaluator.evaluate(solution, policy=HeuristicPolicy() if rule == "existing_heuristic" else None)
            if candidate is None:
                break
            initial.append(candidate)
            logs.append({"kind": "initial", "origin": rule, "evaluation": evaluator.evaluation_count,
                         "completed_seconds": budget.elapsed, "feasible": candidate.feasible,
                         "tchebycheff": candidate.tchebycheff})
        initial_seconds = budget.elapsed
        current = evaluator.best
        feasible = [x for x in initial if x.feasible]
        if feasible:
            initial_temperature = max(1e-6, 0.02 * max(min(x.tchebycheff for x in feasible), 1e-3) / math.log(2))
        temperature = initial_temperature
        while current is not None:
            budget.check_time()
            if budget.max_evaluations is not None and budget.started_evaluations >= budget.max_evaluations:
                break
            if no_new >= int(settings["no_new_evaluation_proposals"]):
                reason = "no_new_evaluation_proposals"
                break
            proposals += 1
            dn, rn = _roulette(dw, rng), _roulette(rw, rng)
            before = budget.started_evaluations
            previous_best = evaluator.best
            archive_before = {tuple(x.objectives) for x in evaluator.archive.entries}
            removed = destroy_operations(dn, current.solution, current, instance, settings, rng)
            outcome = repair_solution(rn, current.solution, removed, instance, current, evaluator, settings)
            candidate = evaluator.evaluate(outcome.solution)
            if candidate is None:
                break
            neighbours = [candidate]
            variants = (alternatives(outcome.solution, removed, evaluator, rng)
                        if local_limit > 1 and (budget.max_evaluations is None or budget.started_evaluations < budget.max_evaluations) else [])
            for alternative in variants:
                if budget.started_evaluations - before >= min(5, local_limit):
                    break
                probe = evaluator.evaluate(alternative)
                if probe is None:
                    break
                neighbours.append(probe)
            candidate = min(neighbours, key=lambda x: (not x.feasible, x.tchebycheff if x.feasible else x.completion_rank))
            if settings["matching_repair"] and budget.started_evaluations - before < local_limit:
                repaired = matching_safe_repair(candidate.solution, candidate)
                if repaired.digest() != candidate.solution.digest():
                    probe = evaluator.evaluate(repaired)
                    if probe is not None:
                        neighbours.append(probe)
                        if candidate_better(probe, candidate):
                            candidate = probe
            completed += 1
            used = budget.started_evaluations - before
            no_new = no_new + 1 if used == 0 else 0
            improved = candidate_better(candidate, current)
            delta = candidate.tchebycheff - current.tchebycheff
            accepted = improved
            worse = False
            if not improved and candidate.feasible and current.feasible:
                accepted = rng.random() < math.exp(-max(0.0, delta) / max(temperature, 1e-12))
                worse = accepted and delta > 1e-12
            if accepted:
                first_feasible = candidate.feasible and not current.feasible
                current = candidate
                if first_feasible:
                    initial_temperature = max(1e-6, 0.02 * max(candidate.tchebycheff, 1e-3) / math.log(2))
                    temperature = initial_temperature
            accepted_worse += int(worse)
            added = bool({tuple(x.objectives) for x in evaluator.archive.entries} - archive_before)
            best_improvement = candidate_better(evaluator.best, previous_best)
            scores = settings["operator_scores"]
            score = max(scores["archive_addition"] if added else 0,
                        scores["scalar_best"] if best_improvement else 0,
                        scores["current_improvement"] if improved else 0,
                        scores["accepted"] if accepted else 0)
            for name, uses, gains in ((dn, du, ds), (rn, ru, rs)):
                uses[name] = uses.get(name, 0) + 1
                gains[name] = gains.get(name, 0) + score
            for key, name, amount in (("destroy_uses", dn, 1), ("repair_uses", rn, 1),
                                      ("repair_evaluations", rn, used), ("repair_scores", rn, score)):
                totals[key][name] = totals[key].get(name, 0) + amount
            temperature *= 0.99
            stale = 0 if added or best_improvement else stale + 1
            if stale >= 50 and reheats < int(settings["maximum_reheats"]):
                temperature = max(temperature, 0.25 * initial_temperature)
                if evaluator.archive.entries:
                    current = rng.choice(evaluator.archive.entries)
                reheats += 1
                stale = 0
            if completed % int(settings["operator_segment_length"]) == 0 and settings["adaptive_weights"]:
                for weights, gains, uses in ((dw, ds, du), (rw, rs, ru)):
                    _update_weights(weights, gains, uses, settings["operator_reaction"], settings["operator_minimum_probability"])
                ds, rs, du, ru = {}, {}, {}, {}
                updates += 1
            logs.append({"kind": "neighbour", "proposal": proposals, "completed_neighbourhoods": completed,
                         "completed_seconds": budget.elapsed, "evaluation": evaluator.evaluation_count,
                         "neighbour_evaluations": used, "destroy_operator": dn, "repair_operator": rn,
                         "removed_operation_count": len(removed), "accepted": accepted, "accepted_worse": worse,
                         "tchebycheff": candidate.tchebycheff, "best_quality": evaluator.best.tchebycheff,
                         "temperature": temperature, "weight_updates": updates, "archive_size": len(evaluator.archive.entries)})
    except BudgetExhausted as exc:
        reason = str(exc)
    selected = evaluator.archive.best(preference) if evaluator.archive.entries else None
    elapsed = budget.elapsed
    initial_best = min((x.tchebycheff for x in initial if x.feasible), default=None)
    stats = {**totals, "destroy_weights": dw, "repair_weights": rw, "reheats": reheats,
             "temperature": temperature, "initial_temperature": initial_temperature,
             "completed_neighbourhoods": completed, "weight_updates": updates, "accepted_worse": accepted_worse}
    accounting = {**budget.counters(), "deadline_overshoot_seconds": max(0, elapsed - budget.seconds) if budget.seconds else 0,
                  "max_evaluations": budget.max_evaluations, "seconds": budget.seconds}
    return SearchResult(preference, selected, evaluator.archive.snapshots(), initial_best,
                        evaluator.evaluation_count, evaluator.cache_hits, proposals, elapsed, stats, tuple(logs),
                        "completed" if selected else "no_feasible_solution", reason, accounting,
                        tuple(evaluator.history), initial_seconds or elapsed, len(initial))
