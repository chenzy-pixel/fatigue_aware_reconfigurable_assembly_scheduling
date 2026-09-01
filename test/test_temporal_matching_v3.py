from __future__ import annotations

import random
from dataclasses import replace

import pytest

from configs import load_config
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from environment.env import TemporalFeasibilityResult, TemporalWorkerTask
from environment.types import ReconfigurationStage


def _single_worker_temporal_env(fixed_instance):
    config = load_config("configs/v7/e1_single_flow.json")
    orders = (
        replace(fixed_instance.orders[5], release_time=0.0),
        replace(fixed_instance.orders[10], release_time=0.0),
    )
    worker = replace(
        fixed_instance.workers[0],
        qualified_modules=fixed_instance.modules,
        initial_fatigue=0.1,
    )
    instance = replace(
        fixed_instance,
        instance_id="temporal_v3_serial_reuse",
        orders=orders,
        workers=(worker,),
    )
    environment = AssemblySchedulingEnv(config)
    environment.reset(instance)
    return environment


def test_temporal_v3_rescues_serial_worker_reuse(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    first = environment.encode_production_action(0, 0)
    second = environment.encode_production_action(4, 1)
    environment.step(first)
    mask = environment.get_action_mask()
    assert not mask[second]
    metrics = environment.metrics()
    assert metrics["temporal_oracle_feasible_count"] > 0
    assert metrics["temporal_future_installation_rescued_count"] > 0


def test_temporal_v3_precedence_and_cache_are_deterministic(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    module = next(iter(environment.workers[0].spec.qualified_modules))
    tasks = (
        TemporalWorkerTask(
            "dis:first", 0, ReconfigurationStage.WAIT_DIS, module, 0
        ),
        TemporalWorkerTask(
            "ins:first", 0, ReconfigurationStage.WAIT_INS, module, 0,
            predecessor_id="dis:first",
        ),
    )
    first = environment._run_temporal_feasibility_search(tasks)
    second = environment._run_temporal_feasibility_search(tasks)
    assert first.status == second.status == "feasible"
    assert first.candidate_completion_tick is None
    assert second.searched_nodes == first.searched_nodes
    assert environment.metrics()["temporal_oracle_cache_hit_count"] >= 1


def test_temporal_v3_node_budget_returns_unknown_and_allows_action(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    environment.config["environment"]["worker_resource_control"][
        "temporal_feasibility"
    ]["max_search_nodes"] = 1
    module = next(iter(environment.workers[0].spec.qualified_modules))
    tasks = (
        TemporalWorkerTask(
            "dis:first", 0, ReconfigurationStage.WAIT_DIS, module, 0
        ),
        TemporalWorkerTask(
            "ins:first", 0, ReconfigurationStage.WAIT_INS, module, 0,
            predecessor_id="dis:first",
        ),
    )
    result = environment._run_temporal_feasibility_search(tasks)
    assert result.status == "unknown"
    assert environment.metrics()["temporal_oracle_unknown_count"] == 1


def test_temporal_assignment_vectorization_preserves_tick_candidates(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    machine_index = 0
    module = environment.machines[machine_index].current_module
    task = TemporalWorkerTask(
        "candidate-dis:vectorized",
        machine_index,
        ReconfigurationStage.WAIT_DIS,
        module,
        environment.current_tick,
        candidate=True,
    )
    states = environment._temporal_initial_worker_states()
    reconfiguration = environment._temporal_task_reconfiguration(task)
    safe_limit = environment.instance.fatigue.maximum_safe_fatigue
    recovery_rate = environment.instance.fatigue.idle_recovery_rate_per_minute
    expected = []
    for worker_index, state in enumerate(states):
        worker = environment.workers[worker_index]
        earliest = max(task.ready_tick, state.available_tick)
        for start_tick in range(earliest, environment.horizon_tick + 1):
            recovered = max(
                0.0,
                state.fatigue
                - recovery_rate
                * (start_tick - state.available_tick)
                * environment.resolution,
            )
            duration_ticks = environment._stage_duration_ticks(
                reconfiguration,
                worker,
                fatigue_override=recovered,
            )
            end_tick = start_tick + duration_ticks
            end_fatigue = (
                recovered
                + environment._stage_accumulation_rate(reconfiguration)
                * duration_ticks
                * environment.resolution
            )
            if (
                end_tick <= environment.horizon_tick
                and end_fatigue <= safe_limit + 1e-9
            ):
                expected.append(
                    (worker_index, start_tick, end_tick, end_fatigue)
                )
    expected.sort(key=lambda value: (value[2], value[1], value[0]))

    actual = environment._temporal_assignment_options(
        task,
        states,
        minimum_start_tick=environment.current_tick,
    )
    assert [value[:3] for value in actual] == [
        value[:3] for value in expected
    ]
    assert [value[3] for value in actual] == pytest.approx(
        [value[3] for value in expected],
        abs=1e-12,
    )


def test_temporal_candidate_completion_drives_processing_start(
    fixed_instance,
    monkeypatch,
):
    environment = _single_worker_temporal_env(fixed_instance)
    environment.step(environment.encode_production_action(0, 0))
    environment._invalidate_resource_snapshot()
    candidate_completion_tick = environment.current_tick + 400
    calls = 0

    def temporal_result(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return TemporalFeasibilityResult(
            "feasible",
            1,
            candidate_completion_tick=candidate_completion_tick,
        )

    monkeypatch.setattr(
        environment,
        "_temporal_production_result",
        temporal_result,
    )
    profile = environment._production_resource_profile(
        1,
        environment.operations[4].spec.required_module,
    )
    assert calls == 1
    assert profile.processing_start_tick >= candidate_completion_tick


def test_strict_frontier_keeps_equal_duration_recovery_counterexample():
    early_high_fatigue = (0, 0, 10, 0.90)
    late_low_fatigue = (0, 5, 15, 0.10)
    frontier, witnesses = AssemblySchedulingEnv._temporal_strict_frontier(
        [early_high_fatigue, late_low_fatigue],
        recovery_rate=0.01,
        resolution=1.0,
    )
    assert early_high_fatigue in frontier
    assert late_low_fatigue in frontier
    assert late_low_fatigue not in witnesses
    assert not AssemblySchedulingEnv._temporal_option_dominates(
        early_high_fatigue,
        late_low_fatigue,
        recovery_rate=0.01,
        resolution=1.0,
    )


def test_strict_frontier_matches_quadratic_reference_and_has_witnesses():
    rng = random.Random(20260901)
    options = [
        (
            rng.randrange(3),
            start := rng.randrange(40),
            start + rng.randrange(1, 12),
            rng.random(),
        )
        for _ in range(300)
    ]
    recovery_rate = 0.013
    resolution = 0.1
    actual, witnesses = AssemblySchedulingEnv._temporal_strict_frontier(
        options,
        recovery_rate=recovery_rate,
        resolution=resolution,
    )
    expected = []
    for worker_index in sorted({option[0] for option in options}):
        ordered = sorted(
            (option for option in options if option[0] == worker_index),
            key=lambda value: (value[2], value[3], value[1], value[0]),
        )
        retained = []
        for option in ordered:
            if any(
                AssemblySchedulingEnv._temporal_option_dominates(
                    witness,
                    option,
                    recovery_rate=recovery_rate,
                    resolution=resolution,
                )
                for witness in retained
            ):
                continue
            retained.append(option)
        expected.extend(retained)
    expected.sort(key=lambda value: (value[2], value[1], value[0]))
    assert actual == expected
    removed = set(options) - set(actual)
    assert removed == set(witnesses)
    for option in removed:
        assert AssemblySchedulingEnv._temporal_option_dominates(
            witnesses[option],
            option,
            recovery_rate=recovery_rate,
            resolution=resolution,
        )


def test_temporal_option_budget_is_unknown_and_never_cached(fixed_instance):
    environment = _single_worker_temporal_env(fixed_instance)
    temporal = environment.config["environment"]["worker_resource_control"][
        "temporal_feasibility"
    ]
    temporal["max_option_evaluations_per_call"] = 1
    module = next(iter(environment.workers[0].spec.qualified_modules))
    tasks = (
        TemporalWorkerTask(
            "dis:first", 0, ReconfigurationStage.WAIT_DIS, module, 0
        ),
        TemporalWorkerTask(
            "ins:first",
            0,
            ReconfigurationStage.WAIT_INS,
            module,
            0,
            predecessor_id="dis:first",
        ),
    )
    first = environment._run_temporal_feasibility_search(tasks)
    second = environment._run_temporal_feasibility_search(tasks)
    assert first.status == second.status == "unknown"
    assert first.termination_reason == "call_option_budget_exhausted"
    assert environment.metrics()["temporal_oracle_cache_hit_count"] == 0


def test_strict_oracle_matches_full_tick_reference_completion(
    fixed_instance,
    monkeypatch,
):
    strict_environment = _single_worker_temporal_env(fixed_instance)
    module = strict_environment.workers[0].spec.qualified_modules[0]
    tasks = (
        TemporalWorkerTask(
            "ordinary-dis",
            0,
            ReconfigurationStage.WAIT_DIS,
            module,
            0,
        ),
        TemporalWorkerTask(
            "candidate-ins",
            1,
            ReconfigurationStage.WAIT_INS,
            module,
            5,
            candidate=True,
        ),
    )
    strict = strict_environment._run_temporal_feasibility_search(tasks)

    def full_tick_frontier(cls, options, **_kwargs):
        return (
            sorted(options, key=lambda value: (value[2], value[1], value[0])),
            {},
        )

    monkeypatch.setattr(
        AssemblySchedulingEnv,
        "_temporal_strict_frontier",
        classmethod(full_tick_frontier),
    )
    reference_environment = _single_worker_temporal_env(fixed_instance)
    reference = reference_environment._run_temporal_feasibility_search(tasks)
    assert strict.status == reference.status == "feasible"
    assert strict.candidate_completion_tick == reference.candidate_completion_tick
