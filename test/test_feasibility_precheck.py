from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from data.feasibility import (
    PRECHECK_VERSION,
    cheap_feasibility_precheck,
)
from data.generate_orders import GenerationError, InstanceGenerator
from data.models import OperationSpec, OrderSpec


def _codes(instance) -> set[str]:
    return set(cheap_feasibility_precheck(instance).reason_codes)


def _two_mandatory_module_instance(fixed_instance, *, horizon: float):
    operations = (
        OperationSpec("P_A2", "P2", 1, "A2", 1.0),
        OperationSpec("P_A3", "P3", 1, "A3", 1.0),
    )
    orders = (
        OrderSpec("P2", "W1", 0.0, (operations[0],)),
        OrderSpec("P3", "W1", 0.0, (operations[1],)),
    )
    machines = tuple(
        replace(machine, initial_module="A1")
        for machine in fixed_instance.machines
    )
    workers = (
        replace(
            fixed_instance.workers[0],
            qualified_modules=("A2", "A3"),
        ),
        *(
            replace(worker, qualified_modules=("A1",))
            for worker in fixed_instance.workers[1:]
        ),
    )
    return replace(
        fixed_instance,
        instance_id=f"mandatory_pair_h{horizon}",
        horizon=horizon,
        machines=machines,
        workers=tuple(workers),
        orders=orders,
    )


def test_precheck_positive_and_stable_version(fixed_instance):
    report = cheap_feasibility_precheck(fixed_instance)
    assert report.passed
    assert report.version == PRECHECK_VERSION == "necessary_conditions_v1"
    assert report.reason_codes == ()


def test_precheck_reports_machine_worker_path_and_capacity_reasons(
    fixed_instance,
):
    module = fixed_instance.operations[0].required_module
    machines = tuple(
        replace(
            machine,
            module_parameters={
                name: parameters
                for name, parameters in machine.module_parameters.items()
                if name != module
            },
        )
        for machine in fixed_instance.machines
    )
    assert "no_machine_edge" in _codes(
        replace(fixed_instance, machines=machines)
    )

    workers = tuple(
        replace(
            worker,
            qualified_modules=tuple(
                value for value in worker.qualified_modules if value != module
            ),
        )
        for worker in fixed_instance.workers
    )
    assert "no_worker_edge" in _codes(
        replace(fixed_instance, workers=workers)
    )

    short = replace(fixed_instance, horizon=100.0)
    short_codes = _codes(short)
    assert "critical_path_over_horizon" in short_codes
    assert "machine_capacity_over_horizon" in short_codes
    assert "module_capacity_over_horizon" in short_codes


def test_precheck_mandatory_safety_and_only_compulsory_matching(
    fixed_instance,
):
    compulsory = _two_mandatory_module_instance(
        fixed_instance, horizon=4.8
    )
    assert "mandatory_matching_deficit" in _codes(compulsory)

    staggerable = _two_mandatory_module_instance(
        fixed_instance, horizon=20.0
    )
    staggerable_codes = _codes(staggerable)
    assert "mandatory_matching_deficit" not in staggerable_codes
    assert "mandatory_task_no_safe_edge" not in staggerable_codes

    unsafe = replace(
        staggerable,
        fatigue=replace(
            staggerable.fatigue,
            maximum_safe_fatigue=0.0,
        ),
    )
    assert "mandatory_task_no_safe_edge" in _codes(unsafe)


def test_generator_precheck_failure_skips_rollout(
    config,
    fixed_instance,
    monkeypatch,
):
    first_order = fixed_instance.orders[0]
    first_operation = replace(
        first_order.operations[0], base_processing_time=10_000.0
    )
    candidate = replace(
        fixed_instance,
        orders=(
            replace(
                first_order,
                operations=(first_operation, *first_order.operations[1:]),
            ),
            *fixed_instance.orders[1:],
        ),
    )
    settings = deepcopy(config["generator"])
    settings["max_generation_attempts"] = 1
    generator = InstanceGenerator(candidate, settings, config=config)
    monkeypatch.setattr(
        generator,
        "_build_candidate",
        lambda **_kwargs: candidate,
    )
    monkeypatch.setattr(
        generator,
        "_static_rejection_reasons",
        lambda *_args, **_kwargs: [],
    )
    rollout_calls = 0

    def fail_if_called(*_args, **_kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        raise AssertionError("rollout must not run after precheck failure")

    monkeypatch.setattr("data.generate_orders._rollout_metrics", fail_if_called)
    with pytest.raises(GenerationError) as failure:
        generator.generate(
            seed=1_000_000,
            split="train",
            pressure_type="balanced",
        )
    assert rollout_calls == 0
    assert "critical_path_over_horizon" in failure.value.failure_reasons
