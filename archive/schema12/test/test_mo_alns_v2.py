from copy import deepcopy
from dataclasses import replace
import random

import pytest

from agent.baselines import HeuristicPolicy
from agent.mo_alns import MOALNSSolver, SearchBudget, BudgetExhausted, decode_solution
from agent.mo_alns.archive import ParetoArchive
from agent.mo_alns.search import CandidateEvaluator, regret_value, insertion_signature
from agent.mo_alns.projection import PredictedSchedule
from agent.mo_alns.solver import _rule_solution
from agent.mo_alns.projection import SerialPredictor, route_key
from environment import AssemblySchedulingEnv, PreferenceVector


PREF = PreferenceVector(1/3, 1/3, 1/3)


def tiny(fixed_instance):
    order = replace(fixed_instance.orders[0], release_time=0.0)
    waves = {wid: {**wave, "order_ids": [order.id] if wid == order.wave else [],
                   "release_interval": [0.0, 0.0]} for wid, wave in fixed_instance.waves.items()}
    return replace(fixed_instance, instance_id="alns_v2_tiny", orders=(order,), waves=waves)


def test_semantic_cache_and_origin(config, fixed_instance):
    instance = tiny(fixed_instance)
    solution = _rule_solution(instance, "earliest_finish", random.Random(1))
    assert solution.digest() == solution.clone(origin="other").digest()
    evaluator = CandidateEvaluator(config, instance, PREF, 3)
    first = evaluator.evaluate(solution)
    assert evaluator.evaluate(solution.clone(origin="other")) is first
    assert evaluator.evaluation_count == 1 and evaluator.cache_hits == 1
    other = solution.clone()
    other.resource_modes[other.operation_order[0]] = "minimum_workload_delta"
    assert other.digest() != solution.digest()


def test_regret_unique_cost_formula():
    assert regret_value([2, 3, 6], 3) == 5
    assert regret_value([2, 3, 6], 2) == 1
    assert regret_value([2], 3) == 0


def test_distinct_modes_with_same_predicted_decisions_are_one_insertion(config, fixed_instance):
    instance = tiny(fixed_instance)
    solution = _rule_solution(instance, "earliest_finish", random.Random(1))
    other = solution.clone()
    other.resource_modes[other.operation_order[0]] = "minimum_workload_delta"
    prediction = PredictedSchedule((1., 2., 3.), True, {solution.operation_order[0]: {"machine": "M1", "dis": None, "ins": "W1"}})
    assert solution.digest() != other.digest()
    assert insertion_signature(solution, prediction) == insertion_signature(other, prediction)


def test_no_solution_before_initialization_deadline(config, fixed_instance):
    clock = iter([0.0, 2.0, 2.0, 2.0])
    budget = SearchBudget(seconds=1, clock=lambda: next(clock))
    result = MOALNSSolver(config).solve(tiny(fixed_instance), PREF, budget=budget)
    assert result.selected is None
    assert result.status == "no_feasible_solution" and result.stop_reason == "time_limit"
    assert result.accounting["started_evaluations"] == 0


def test_original_heuristic_trace_is_exact_and_counted_once(config, fixed_instance):
    instance = tiny(fixed_instance)
    solution = _rule_solution(instance, "earliest_finish", random.Random(1))
    evaluator = CandidateEvaluator(config, instance, PREF, 1)
    direct = evaluator.evaluate(solution, policy=HeuristicPolicy())
    replay = decode_solution(config, instance, direct.solution, PREF)
    assert direct.actions == replay.actions
    assert direct.objectives == pytest.approx(replay.objectives)
    assert evaluator.evaluation_count == 1
    assert evaluator.evaluate(direct.solution) is direct


def test_archive_equal_objectives_do_not_reward_duplicates(config, fixed_instance):
    instance = tiny(fixed_instance)
    candidate = decode_solution(config, instance, _rule_solution(instance, "earliest_finish", random.Random(1)), PREF)
    archive = ParetoArchive(config)
    assert archive.update(candidate)
    duplicate = replace(candidate, action_trace_sha256="different")
    assert not archive.update(duplicate)
    assert len(archive.entries) == 1


def test_proxy_is_read_only_and_uses_workload_modes(config, fixed_instance):
    instance = tiny(fixed_instance)
    budget = SearchBudget(max_evaluations=8)
    predictor = SerialPredictor(config, instance, budget)
    solution = _rule_solution(instance, "earliest_finish", random.Random(1))
    before = (predictor.env.current_tick, tuple(w.fatigue for w in predictor.env.workers),
              tuple(predictor.env._committed_worker_loads))
    prediction = predictor.predict(solution)
    assert prediction.objectives[2] >= 0
    assert before == (predictor.env.current_tick, tuple(w.fatigue for w in predictor.env.workers),
                      tuple(predictor.env._committed_worker_loads))
    mi = 0
    module = instance.machines[mi].initial_module
    state = predictor.projector.initial_resources()
    route = predictor.routes(mi, module, module, 10, state)[0]
    assert predictor.projector.path_costs(mi, route) == (0, 0, 0, 0, 0)
    assert route_key("minimum_workload_delta", 10, (0, 0, 10, 1, 2), .3) < route_key("minimum_workload_delta", 5, (0, 0, 1, 0, 3), .2)


def test_budgeted_search_activates_loop_and_keeps_incumbent(config, fixed_instance):
    config = deepcopy(config)
    config["mo_alns"] = {"max_evaluations_per_preference": 32,
                          "max_evaluations_per_neighbourhood": 2,
                          "max_proxy_evaluations_per_neighbourhood": 2,
                          "regret_position_candidates": 1, "regret_resource_variants": 1,
                          "operator_segment_length": 2}
    result = MOALNSSolver(config, algorithm_seed=11).solve(tiny(fixed_instance), PREF)
    assert result.selected.feasible
    assert result.selected.tchebycheff <= result.initial_best_tchebycheff + 1e-12
    assert result.accounting["started_evaluations"] <= 32
    assert result.operator_statistics["weight_updates"] >= 1
    assert result.operator_statistics["initial_temperature"] > 0
    assert all(row["neighbour_evaluations"] <= 2 for row in result.search_log if row["kind"] == "neighbour")


def test_legacy_protocol_requires_explicit_migration(config):
    config = deepcopy(config)
    config["mo_alns"] = {"protocol": "mo_alns_solver_budget_v2"}
    with pytest.raises(ValueError, match="migration"):
        MOALNSSolver(config)


def test_larger_budget_keeps_candidate_prefix(config, fixed_instance):
    config = deepcopy(config)
    config["mo_alns"] = {"max_proxy_evaluations_per_neighbourhood": 2,
                          "regret_position_candidates": 1, "regret_resource_variants": 1}
    instance = tiny(fixed_instance)
    short = MOALNSSolver(config, algorithm_seed=11).solve(instance, PREF, budget=SearchBudget(max_evaluations=12))
    long = MOALNSSolver(config, algorithm_seed=11).solve(instance, PREF, budget=SearchBudget(max_evaluations=20))
    assert [r["action_trace_sha256"] for r in short.candidate_history] == [r["action_trace_sha256"] for r in long.candidate_history[:12]]


def test_deadline_inside_decode_discards_partial_candidate(config, fixed_instance):
    calls = [0]

    def clock():
        calls[0] += 1
        return 2.0 if calls[0] >= 6 else 0.0

    budget = SearchBudget(seconds=1, clock=clock)
    result = MOALNSSolver(config).solve(tiny(fixed_instance), PREF, budget=budget)
    assert result.selected is None and result.candidate_history == ()
    assert result.accounting["started_evaluations"] == 1
    assert result.accounting["aborted_evaluations"] == 1
    assert result.accounting["completed_evaluations"] == 0
    assert result.accounting["environment_steps"] == 1
