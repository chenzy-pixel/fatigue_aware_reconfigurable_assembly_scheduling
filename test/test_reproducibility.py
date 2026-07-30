from __future__ import annotations

from data.dataset import canonical_json_bytes
from data.models import instance_to_dict


def test_generated_record_is_reproducible_and_template_is_unchanged(
    instance_generator,
    fixed_instance,
    pressure_records,
):
    template_before = canonical_json_bytes(instance_to_dict(fixed_instance))
    first = pressure_records["balanced"]
    second = instance_generator.generate(
        seed=1000001,
        split="train",
        pressure_type="balanced",
    )
    assert canonical_json_bytes(first.to_dict()) == canonical_json_bytes(
        second.to_dict()
    )
    assert canonical_json_bytes(instance_to_dict(fixed_instance)) == template_before


def test_different_seed_changes_generated_instance(instance_generator):
    first = instance_generator.generate(
        seed=1010000,
        split="train",
        pressure_type="balanced",
    )
    second = instance_generator.generate(
        seed=1010001,
        split="train",
        pressure_type="balanced",
    )
    assert canonical_json_bytes(first.to_dict()) != canonical_json_bytes(
        second.to_dict()
    )
