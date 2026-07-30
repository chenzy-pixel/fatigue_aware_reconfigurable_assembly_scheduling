from __future__ import annotations

import math

import pytest

from data.generate_orders import PRESSURE_TYPES
from data.models import validate_instance


def _assert_between(value, lower, upper, *, tolerance=1e-9):
    assert lower - tolerance <= value <= upper + tolerance


def _assert_record_contract(record, config):
    instance = record.instance
    metadata = record.metadata
    sparse = (
        metadata.get("ood_factor") == "worker_qualification_sparsity"
    )
    validate_instance(
        instance,
        minimum_qualified_workers=1 if sparse else 3,
    )
    heuristic = metadata["heuristic_metrics"]
    assert heuristic["schedule_violations"] == []
    assert heuristic["ready_operation_count"] > 0
    assert heuristic["ready_configuration_gap_count"] >= 1
    assert heuristic["ready_configuration_gap_ratio"] > 0
    if metadata["split"] != "stress":
        assert heuristic["heuristic_completed"]
        assert not heuristic["heuristic_truncated"]
        assert heuristic["heuristic_makespan"] <= config["generator"][
            "acceptance"
        ]["max_heuristic_makespan_minutes"]


def _assert_generation_ranges(record, config, template):
    instance = record.instance
    metadata = record.metadata
    settings = config["generator"]
    pressure_type = metadata["pressure_type"]
    profile = settings["pressure_profiles"][pressure_type]
    ood_factor = metadata.get("ood_factor")

    _assert_between(
        len(instance.orders),
        int(profile["order_count"][0]),
        int(profile["order_count"][1]),
    )
    for order in instance.orders:
        _assert_between(
            len(order.operations),
            int(profile["operations_per_order"][0]),
            int(profile["operations_per_order"][1]),
        )
        for operation in order.operations:
            _assert_between(
                operation.base_processing_time,
                float(settings["processing_time"][0]),
                float(settings["processing_time"][1]),
            )

    release_windows = profile["release_windows"]
    if ood_factor == "arrival_overlap":
        release_windows = ((0.0, 0.03), (0.05, 0.08), (0.10, 0.14))
    for wave_index, wave_id in enumerate(instance.waves):
        low, high = release_windows[wave_index]
        for order in instance.orders:
            if order.wave == wave_id:
                _assert_between(
                    order.release_time,
                    float(low) * instance.horizon,
                    float(high) * instance.horizon
                    + instance.resolution,
                )

    if ood_factor == "initial_fatigue":
        fatigue_bounds = (0.25, 0.40)
    elif pressure_type == "fatigue_bottleneck":
        fatigue_bounds = (0.10, 0.25)
    elif pressure_type == "easy":
        fatigue_bounds = (0.05, 0.10)
    else:
        fatigue_bounds = tuple(settings["initial_fatigue"])
    for worker in instance.workers:
        _assert_between(worker.initial_fatigue, *fatigue_bounds)

    template_machines = {
        machine.id: machine for machine in template.machines
    }
    speed_bounds = (
        (0.80, 1.20)
        if ood_factor == "processing_time_heterogeneity"
        else tuple(settings["processing_factor_local"])
    )
    reconfiguration_scale = (
        (1.50, 2.00)
        if ood_factor == "reconfiguration_time_scale"
        else tuple(profile["reconfiguration_scale"])
    )
    reconfiguration_local = tuple(
        settings["reconfiguration_factor_local"]
    )
    reconfiguration_low = (
        reconfiguration_scale[0] * reconfiguration_local[0]
    )
    reconfiguration_high = (
        reconfiguration_scale[1] * reconfiguration_local[1]
    )
    for machine in instance.machines:
        template_machine = template_machines[machine.id]
        for module, parameters in machine.module_parameters.items():
            baseline = template_machine.module_parameters[module]
            _assert_between(
                parameters.processing_speed_factor
                / baseline.processing_speed_factor,
                *speed_bounds,
            )
            for actual, original in (
                (
                    parameters.installation_base_time,
                    baseline.installation_base_time,
                ),
                (
                    parameters.disassembly_base_time,
                    baseline.disassembly_base_time,
                ),
            ):
                _assert_between(
                    actual,
                    original * reconfiguration_low,
                    original * reconfiguration_high
                    + instance.resolution,
                )

    cost_profile = metadata["cost_profile"]
    cost_local = tuple(settings["cost_factor_local"])

    def cost_scale(kind):
        dominant = f"{kind}_dominant"
        if cost_profile == dominant:
            return 1.5
        return 1.0 if cost_profile == "balanced_cost" else 0.75

    for machine in instance.machines:
        baseline = template_machines[machine.id].downtime_cost_per_minute
        _assert_between(
            machine.downtime_cost_per_minute / baseline,
            cost_scale("downtime") * cost_local[0],
            cost_scale("downtime") * cost_local[1],
        )
    template_workers = {
        worker.id: worker for worker in template.workers
    }
    for worker in instance.workers:
        baseline = template_workers[worker.id].labor_cost_per_minute
        _assert_between(
            worker.labor_cost_per_minute / baseline,
            cost_scale("labor") * cost_local[0],
            cost_scale("labor") * cost_local[1],
        )
    for module, costs in instance.module_costs.items():
        baseline = template.module_costs[module]
        for actual, original in (
            (
                costs.fixed_disassembly_cost,
                baseline.fixed_disassembly_cost,
            ),
            (
                costs.fixed_installation_cost,
                baseline.fixed_installation_cost,
            ),
        ):
            _assert_between(
                actual / original,
                cost_scale("fixed_cost") * cost_local[0],
                cost_scale("fixed_cost") * cost_local[1],
            )


def test_pressure_records_are_structurally_valid(pressure_records, config):
    for pressure_type, record in pressure_records.items():
        instance = record.instance
        _assert_record_contract(record, config)
        assert record.metadata["pressure_type"] == pressure_type
        assert all(
            len({operation.required_module for operation in order.operations}) >= 2
            for order in instance.orders
        )
        previous = None
        for wave_id, wave in instance.waves.items():
            dominant = wave["dominant_module"]
            operations = [
                operation
                for order in instance.orders
                if order.wave == wave_id
                for operation in order.operations
            ]
            share = sum(
                operation.required_module == dominant for operation in operations
            ) / len(operations)
            assert share >= 0.60
            assert dominant != previous
            previous = dominant
        assert all(
            math.isfinite(value) and value > 0
            for machine in instance.machines
            for parameters in machine.module_parameters.values()
            for value in (
                parameters.processing_speed_factor,
                parameters.installation_base_time,
                parameters.disassembly_base_time,
            )
        )


def test_pressure_records_stay_within_configured_ranges(
    pressure_records,
    config,
    fixed_instance,
):
    for record in pressure_records.values():
        _assert_generation_ranges(record, config, fixed_instance)


@pytest.mark.parametrize(
    "ood_factor",
    (
        "processing_time_heterogeneity",
        "reconfiguration_time_scale",
        "arrival_overlap",
        "worker_qualification_sparsity",
        "initial_fatigue",
        "module_demand_imbalance",
    ),
)
def test_ood_factor_uses_its_own_valid_range(
    instance_generator,
    config,
    fixed_instance,
    ood_factor,
):
    factor_index = config["generator"]["ood"]["factors"].index(ood_factor)
    record = instance_generator.generate(
        seed=4_000_100 + factor_index,
        split="ood",
        pressure_type="balanced",
        ood_factor=ood_factor,
    )
    _assert_record_contract(record, config)
    _assert_generation_ranges(record, config, fixed_instance)


@pytest.mark.slow
def test_slow_105_instance_distribution_audit(
    instance_generator,
    config,
    fixed_instance,
):
    records = []
    for profile_index, pressure_type in enumerate(PRESSURE_TYPES):
        for offset in range(15):
            record = instance_generator.generate(
                seed=1_100_000 + profile_index * 1_000 + offset,
                split="train",
                pressure_type=pressure_type,
            )
            _assert_record_contract(record, config)
            _assert_generation_ranges(record, config, fixed_instance)
            records.append(record)
    assert len(records) == 105


def test_pressure_profiles_create_expected_dynamic_pressure(pressure_records):
    def metrics(name):
        record = pressure_records[name]
        return {
            **record.metadata["pressure_metrics"],
            **record.metadata["heuristic_metrics"],
        }

    easy = metrics("easy")
    balanced = metrics("balanced")
    machine = metrics("machine_bottleneck")
    reconfiguration = metrics("reconfiguration_bottleneck")
    worker = metrics("worker_bottleneck")
    fatigue = metrics("fatigue_bottleneck")
    arrival = metrics("high_arrival_pressure")

    assert machine["max_module_load"] > easy["max_module_load"]
    assert (
        reconfiguration["ready_configuration_gap_ratio"]
        > easy["ready_configuration_gap_ratio"]
    )
    assert (
        reconfiguration["heuristic_reconfiguration_ratio"]
        > easy["heuristic_reconfiguration_ratio"]
    )
    assert worker["worker_competition_event_count"] >= 1
    assert worker["machine_waiting_for_worker_time"] > 0
    assert fatigue["fatigue_masked_action_count"] >= 1
    assert (
        fatigue["maximum_worker_fatigue"]
        > balanced["maximum_worker_fatigue"]
    )
    assert arrival["mean_wave_overlap_ratio"] > easy["mean_wave_overlap_ratio"]
