from __future__ import annotations

from dataclasses import replace

import pytest

from data.generate_orders import generate_random_instance
from data.models import validate_instance


def test_fixed_instance_contract(fixed_instance):
    assert len(fixed_instance.machines) == 8
    assert len(fixed_instance.workers) == 6
    assert len(fixed_instance.orders) == 15
    assert len(fixed_instance.operations) == 60
    assert all(
        len(machine.module_parameters) == 2
        for machine in fixed_instance.machines
    )
    for wave_id, wave in fixed_instance.waves.items():
        order_ids = set(wave["order_ids"])
        dominant = wave["dominant_module"]
        count = sum(
            operation.required_module == dominant
            for order in fixed_instance.orders
            if order.id in order_ids
            for operation in order.operations
        )
        assert count == 15, wave_id


def test_random_instance_is_reproducible_and_feasible(fixed_instance):
    first = generate_random_instance(
        fixed_instance, num_orders=6, seed=12345
    )
    second = generate_random_instance(
        fixed_instance, num_orders=6, seed=12345
    )
    assert first == second
    assert first.instance_type == "random"
    assert all(
        len({operation.required_module for operation in order.operations}) >= 2
        for order in first.orders
    )
    for wave_id, wave in first.waves.items():
        assert set(wave["order_ids"]) == {
            order.id for order in first.orders if order.wave == wave_id
        }
        wave_orders = [
            order for order in first.orders if order.wave == wave_id
        ]
        if wave_orders:
            assert wave["release_interval"] == [
                min(order.release_time for order in wave_orders),
                max(order.release_time for order in wave_orders),
            ]
    validate_instance(first)

    stale_waves = replace(first, waves=fixed_instance.waves)
    with pytest.raises(ValueError, match="order_ids do not match"):
        validate_instance(stale_waves)
