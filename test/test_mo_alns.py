from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random

import pytest

from agent.mo_alns import MOALNSSolver, ParetoArchive, augmented_tchebycheff, decode_solution
from agent.mo_alns.archive import normalized_objectives
from agent.mo_alns.solver import (
    DESTROY_OPERATORS,
    REPAIR_OPERATORS,
    _CandidateEvaluator,
    _rule_solution,
    _update_weights,
    _valid_positions,
    derive_mo_alns_seed,
    destroy_operations,
    repair_solution,
)
from environment import PreferenceVector


def _tiny_instance(fixed_instance):
    order = replace(fixed_instance.orders[0], release_time=0.0)
    waves = {}
    for wave_id, wave in fixed_instance.waves.items():
        updated = dict(wave)
        if wave_id == order.wave:
            updated["order_ids"] = [order.id]
            updated["release_interval"] = [0.0, 0.0]
        else:
            updated["order_ids"] = []
            updated["release_interval"] = [0.0, 0.0]
        waves[wave_id] = updated
    return replace(
        fixed_instance,
        instance_id="mo_alns_tiny",
        instance_type="test",
        orders=(order,),
        waves=waves,
    )


def _settings(config, *, budget=10):
    result = deepcopy(config)
    result["mo_alns"] = {
        "max_evaluations_per_preference": budget,
        "max_proposals_multiplier": 3,
        "temperature_calibration_samples": 1,
        "regret_position_candidates": 1,
        "regret_resource_variants": 1,
        "operator_segment_length": 1,
    }
    return result


def test_baseline_record_pipeline_uses_current_config_and_schema(config, fixed_instance):
    from data import load_dataset_split
    from scripts.mo_alns import _run_record
    from result.metrics import EVALUATION_SCHEMA_VERSION
    record = load_dataset_split(config, "test")[0]
    tiny_record = replace(record, instance=_tiny_instance(fixed_instance))
    result = _run_record(_settings(config, budget=8), tiny_record, "test", 11, (PreferenceVector(1.0, 0.0, 0.0),))
    row = result["rows"][0]
    assert row["result_schema_version"] == EVALUATION_SCHEMA_VERSION
    assert row["objective_scale_flow"] == config["objective_scalarizer"]["scales"]["flow"]
    assert row["normalization_manifest_sha256"] == config["objective_scalarizer"]["normalization_manifest_sha256"]
    assert row["arm"] == "mo_alns"
    assert row["replay_verified"]
    assert row["single_stage_proxy_return"] == pytest.approx(row["training_cumulative_reward"])


def test_encoding_decoder_and_destroy_rules_keep_environment_semantics(config, fixed_instance):
    instance = _tiny_instance(fixed_instance)
    solution = _rule_solution(instance, "earliest_finish", random.Random(3))
    solution.validate(instance)
    candidate = decode_solution(config, instance, solution, PreferenceVector(0.5, 0.3, 0.2))
    assert candidate.feasible
    assert candidate.metrics["schedule_violations"] == []
    assert candidate.metrics["maximum_worker_fatigue"] <= candidate.metrics["safe_fatigue_limit"] + 1e-9
    assert candidate.metrics["current_worker_matching_deficit"] == 0
    for name in DESTROY_OPERATORS:
        removed = destroy_operations(name, solution, candidate, instance, {"destroy_fraction": [0.5, 0.5], "minimum_removed_operations": 2}, random.Random(7))
        assert len(removed) == 2
        assert len(set(removed)) == 2
        assert set(removed).issubset(set(solution.operation_order))


def test_solver_budget_grid_archive_and_replay_are_deterministic(config, fixed_instance):
    instance = _tiny_instance(fixed_instance)
    settings = _settings(config, budget=10)
    solver = MOALNSSolver(settings, algorithm_seed=11)
    preference = PreferenceVector(0.5, 0.3, 0.2)
    first = solver.solve(instance, preference)
    second = MOALNSSolver(settings, algorithm_seed=11).solve(instance, preference)
    assert first.environment_evaluations <= 10
    assert first.selected.feasible
    assert first.selected.tchebycheff <= first.initial_best_tchebycheff + 1e-12
    assert first.selected.action_trace_sha256 == second.selected.action_trace_sha256
    assert first.selected.objectives == pytest.approx(second.selected.objectives)
    grid = MOALNSSolver(_settings(config, budget=8), algorithm_seed=11).solve_grid(instance)
    assert len(grid.endpoints) == 66
    assert all(value.feasible for value in grid.endpoints.values())
    for target, endpoint in ((PreferenceVector(1.0, 0.0, 0.0), grid.endpoints["1_0_0"]), (preference, grid.endpoints["0.5_0.3_0.2"])):
        replay = decode_solution(settings, instance, endpoint.solution, target)
        assert replay.objectives == pytest.approx(endpoint.objectives, abs=1e-9)
        assert replay.action_trace_sha256 == endpoint.action_trace_sha256


def test_tchebycheff_and_archive_deduplicate_decoded_trace(config, fixed_instance):
    instance = _tiny_instance(fixed_instance)
    preference = PreferenceVector(0.5, 0.3, 0.2)
    solution = _rule_solution(instance, "short_flow", random.Random(9))
    candidate = decode_solution(config, instance, solution, preference)
    archive = ParetoArchive(config)
    assert archive.update(candidate)
    assert not archive.update(candidate)
    assert archive.best(preference).action_trace_sha256 == candidate.action_trace_sha256
    assert augmented_tchebycheff(candidate.objectives, preference, config) == pytest.approx(candidate.tchebycheff)
    assert candidate.normalized_objectives == pytest.approx(normalized_objectives(candidate.objectives, config))


def test_operator_probability_floor_and_dataset_seed_are_stable():
    weights = {name: 1.0 for name in DESTROY_OPERATORS}
    _update_weights(
        weights,
        {DESTROY_OPERATORS[0]: 8.0},
        {DESTROY_OPERATORS[0]: 1},
        reaction=0.2,
        floor=0.05,
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.05
    preference = PreferenceVector(0.5, 0.3, 0.2)
    assert derive_mo_alns_seed(11, "instance", preference, "test") == derive_mo_alns_seed(
        11, "instance", preference, "test"
    )
    assert derive_mo_alns_seed(11, "instance", preference, "test") != derive_mo_alns_seed(
        11, "instance", preference, "ood"
    )


def test_all_repairs_preserve_topology_for_adjacent_removed_operations(config, fixed_instance):
    instance = _tiny_instance(fixed_instance)
    preference = PreferenceVector(0.5, 0.3, 0.2)
    base = _rule_solution(instance, "earliest_finish", random.Random(5))
    candidate = decode_solution(config, instance, base, preference)
    removed = ("O_J1_2", "O_J1_3")
    # Inserting operation 3 while operation 2 is absent must still stay after
    # the present transitive predecessor operation 1.
    lower, upper = _valid_positions(("O_J1_1", "O_J1_4"), "O_J1_3", instance)
    assert (lower, upper) == (1, 1)
    for repair_name in REPAIR_OPERATORS:
        evaluator = _CandidateEvaluator(config, instance, preference, maximum_evaluations=50)
        outcome = repair_solution(
            repair_name,
            base,
            removed,
            instance,
            candidate,
            evaluator,
            {"regret_position_candidates": 1, "regret_resource_variants": 1},
        )
        outcome.solution.validate(instance)
        repaired = decode_solution(config, instance, outcome.solution, preference)
        assert repaired.feasible
