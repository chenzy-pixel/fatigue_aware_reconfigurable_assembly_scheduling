"""Budgeted exact simulation with trajectory caching and a Pareto archive."""
import math
from .archive import ParetoArchive, candidate_better
from .budget import SearchBudget, BudgetExhausted

class CandidateEvaluator:
    def __init__(self, config, instance, preference, maximum_evaluations=None, *, budget=None):
        self.config, self.instance, self.preference = config, instance, preference
        self.budget = budget or SearchBudget(max_evaluations=maximum_evaluations)
        self.maximum_evaluations = self.budget.max_evaluations or math.inf
        self.cache, self.history = {}, []
        self.cache_hits = 0
        self.archive = ParetoArchive(dict(config))
        self.best = None

    @property
    def evaluation_count(self):
        return self.budget.completed_evaluations

    def evaluate(self, solution, *, policy=None):
        from .decoder import decode_solution
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
