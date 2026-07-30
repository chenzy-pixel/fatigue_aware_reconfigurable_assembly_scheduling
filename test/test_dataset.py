from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from itertools import combinations

import pytest

from configs import project_path
from data.dataset import (
    ALL_SPLITS,
    MANIFEST_REQUIRED_KEYS,
    PERSISTED_SPLITS,
    GeneratedInstanceRecord,
    InstanceDataset,
    build_dataset_split,
    canonical_json_bytes,
    sha256_bytes,
    split_seed_range,
)
from data.models import validate_instance


def _build_validation(config, fixed_instance, tmp_path, **kwargs):
    instances_root = tmp_path / "instances"
    manifests_root = tmp_path / "manifests"
    manifest = build_dataset_split(
        config=config,
        template=fixed_instance,
        split="validation",
        count=kwargs.pop("count", 1),
        instances_root=instances_root,
        manifests_root=manifests_root,
        **kwargs,
    )
    return manifest, instances_root


def test_manifest_build_is_reproducible_and_loader_verifies_hash(
    config,
    fixed_instance,
    tmp_path,
):
    first_manifest, instances_root = _build_validation(
        config, fixed_instance, tmp_path
    )
    first_bytes = first_manifest.read_bytes()
    first = InstanceDataset(
        first_manifest,
        instances_root=instances_root,
        expected_split="validation",
        expected_seed_range=split_seed_range(config, "validation"),
    )
    assert len(first) == 1
    first_record = first[0]
    first_payload = canonical_json_bytes(first_record.to_dict())

    second_manifest, _ = _build_validation(
        config,
        fixed_instance,
        tmp_path,
        overwrite=True,
    )
    assert second_manifest.read_bytes() == first_bytes
    second = InstanceDataset(
        second_manifest,
        instances_root=instances_root,
    )
    assert second[0] == first_record
    second_payload = canonical_json_bytes(second[0].to_dict())
    assert second_payload == first_payload
    assert sha256_bytes(second_payload) == sha256_bytes(first_payload)

    manifest = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert set(manifest) == MANIFEST_REQUIRED_KEYS
    assert manifest["seed_start"] == 2_000_000
    assert manifest["instance_count"] == 1
    assert manifest["template_instance"] == "fixed_15x4_v1"
    assert manifest["files"][0]["path"] == "instance_2000000.json"
    instance_path = (
        instances_root
        / "validation"
        / manifest["files"][0]["path"]
    )
    instance_path.write_text(
        instance_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _ = InstanceDataset(
            second_manifest,
            instances_root=instances_root,
        )[0]


def test_manifest_rejects_unsafe_paths_and_metadata_mismatch(
    config,
    fixed_instance,
    tmp_path,
):
    manifest_path, instances_root = _build_validation(
        config, fixed_instance, tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    unsafe = deepcopy(manifest)
    unsafe["files"][0]["path"] = "../instance.json"
    manifest_path.write_bytes(canonical_json_bytes(unsafe))
    with pytest.raises(ValueError, match="unsafe instance path"):
        InstanceDataset(manifest_path, instances_root=instances_root)

    duplicate = deepcopy(manifest)
    duplicate["instance_count"] = 2
    duplicate["files"].append(
        {
            **duplicate["files"][0],
            "seed": duplicate["seed_start"] + 1,
        }
    )
    manifest_path.write_bytes(canonical_json_bytes(duplicate))
    with pytest.raises(ValueError, match="duplicate instance path"):
        InstanceDataset(manifest_path, instances_root=instances_root)

    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="outside"):
        InstanceDataset(
            manifest_path,
            instances_root=instances_root,
            expected_seed_range=(2_000_001, 3_000_000),
        )
    with pytest.raises(ValueError, match="generator_version mismatch"):
        InstanceDataset(
            manifest_path,
            instances_root=instances_root,
            expected_generator_version="9.9.9",
        )

    manifest_path.write_bytes(canonical_json_bytes(manifest))
    instance_path = (
        instances_root
        / "validation"
        / manifest["files"][0]["path"]
    )
    raw = json.loads(instance_path.read_text(encoding="utf-8"))
    raw["metadata"]["seed"] += 1
    payload = canonical_json_bytes(raw)
    instance_path.write_bytes(payload)
    manifest["files"][0]["sha256"] = sha256_bytes(payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    dataset = InstanceDataset(manifest_path, instances_root=instances_root)
    with pytest.raises(ValueError, match="metadata seed mismatch"):
        _ = dataset[0]


def test_interrupted_build_resumes_and_published_split_refuses_overwrite(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    from data.generate_orders import InstanceGenerator

    original_generate = InstanceGenerator.generate
    failed_once = False

    def flaky_generate(self, **kwargs):
        nonlocal failed_once
        if kwargs["seed"] == 2_000_001 and not failed_once:
            failed_once = True
            raise RuntimeError("simulated interruption")
        return original_generate(self, **kwargs)

    monkeypatch.setattr(InstanceGenerator, "generate", flaky_generate)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build_validation(
            config,
            fixed_instance,
            tmp_path,
            count=2,
        )
    partials = list(
        (tmp_path / "instances" / ".build").rglob(
            "instance_2000000.json"
        )
    )
    assert len(partials) == 1
    partial_bytes = partials[0].read_bytes()

    manifest_path, instances_root = _build_validation(
        config,
        fixed_instance,
        tmp_path,
        count=2,
    )
    assert (
        instances_root
        / "validation"
        / "instance_2000000.json"
    ).read_bytes() == partial_bytes
    dataset = InstanceDataset(
        manifest_path,
        instances_root=instances_root,
    )
    assert len(dataset) == 2
    with pytest.raises(FileExistsError):
        _build_validation(
            config,
            fixed_instance,
            tmp_path,
            count=2,
        )


def _replace_first_operation(instance, **changes):
    orders = list(instance.orders)
    operations = list(orders[0].operations)
    operations[0] = replace(operations[0], **changes)
    orders[0] = replace(orders[0], operations=tuple(operations))
    return replace(instance, orders=tuple(orders))


def test_validate_instance_rejects_inconsistent_wave_contract(
    fixed_instance,
):
    wave_id = next(iter(fixed_instance.waves))

    waves = deepcopy(fixed_instance.waves)
    waves[wave_id]["order_ids"] = waves[wave_id]["order_ids"][1:]
    with pytest.raises(ValueError, match="order_ids do not match"):
        validate_instance(replace(fixed_instance, waves=waves))

    waves = deepcopy(fixed_instance.waves)
    waves[wave_id]["release_interval"][1] += fixed_instance.resolution
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_instance(replace(fixed_instance, waves=waves))


def test_validate_instance_rejects_resource_and_dominance_failures(
    fixed_instance,
):
    demanded_module = fixed_instance.operations[0].required_module
    workers = tuple(
        replace(
            worker,
            qualified_modules=tuple(
                module
                for module in worker.qualified_modules
                if module != demanded_module
            ),
        )
        for worker in fixed_instance.workers
    )
    with pytest.raises(ValueError, match="qualified workers"):
        validate_instance(
            replace(fixed_instance, workers=workers),
            minimum_compatible_machines=1,
            minimum_qualified_workers=1,
        )

    machines = []
    for machine in fixed_instance.machines:
        parameters = dict(machine.module_parameters)
        initial_module = machine.initial_module
        if demanded_module in parameters:
            replacement_module = next(
                module
                for module in fixed_instance.modules
                if module not in parameters
            )
            replacement_parameters = parameters.pop(demanded_module)
            parameters[replacement_module] = replacement_parameters
            if initial_module == demanded_module:
                initial_module = replacement_module
        machines.append(
            replace(
                machine,
                initial_module=initial_module,
                module_parameters=parameters,
            )
        )
    with pytest.raises(ValueError, match="compatible machines"):
        validate_instance(
            replace(fixed_instance, machines=tuple(machines)),
            minimum_compatible_machines=1,
            minimum_qualified_workers=1,
        )

    wave_id = next(iter(fixed_instance.waves))
    dominant = str(fixed_instance.waves[wave_id]["dominant_module"])
    replacement_module = next(
        module for module in fixed_instance.modules if module != dominant
    )
    orders = tuple(
        replace(
            order,
            operations=tuple(
                replace(operation, required_module=replacement_module)
                if order.wave == wave_id
                and operation.required_module == dominant
                else operation
                for operation in order.operations
            ),
        )
        for order in fixed_instance.orders
    )
    with pytest.raises(ValueError, match="dominant module share"):
        validate_instance(replace(fixed_instance, orders=orders))


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("invalid_sequence", "sequence is not contiguous"),
        ("nan_cost", "finite non-negative"),
        ("zero_processing_time", "finite positive"),
        ("zero_speed", "finite positive"),
        ("fatigue_above_limit", "exceeds maximum_safe_fatigue"),
        ("negative_release", "finite non-negative"),
    ),
)
def test_validate_instance_rejects_invalid_numbers_and_precedence(
    fixed_instance,
    case,
    message,
):
    if case == "invalid_sequence":
        invalid = _replace_first_operation(fixed_instance, sequence=2)
    elif case == "nan_cost":
        machines = list(fixed_instance.machines)
        machines[0] = replace(
            machines[0],
            downtime_cost_per_minute=float("nan"),
        )
        invalid = replace(fixed_instance, machines=tuple(machines))
    elif case == "zero_processing_time":
        invalid = _replace_first_operation(
            fixed_instance,
            base_processing_time=0.0,
        )
    elif case == "zero_speed":
        machines = list(fixed_instance.machines)
        parameters = dict(machines[0].module_parameters)
        module = next(iter(parameters))
        parameters[module] = replace(
            parameters[module],
            processing_speed_factor=0.0,
        )
        machines[0] = replace(
            machines[0],
            module_parameters=parameters,
        )
        invalid = replace(fixed_instance, machines=tuple(machines))
    elif case == "fatigue_above_limit":
        workers = list(fixed_instance.workers)
        workers[0] = replace(
            workers[0],
            initial_fatigue=(
                fixed_instance.fatigue.maximum_safe_fatigue + 0.01
            ),
        )
        invalid = replace(fixed_instance, workers=tuple(workers))
    else:
        orders = list(fixed_instance.orders)
        orders[0] = replace(orders[0], release_time=-0.1)
        invalid = replace(fixed_instance, orders=tuple(orders))
    with pytest.raises(ValueError, match=message):
        validate_instance(invalid)


def test_configured_and_published_seed_sets_are_disjoint(config):
    ranges = {
        split: split_seed_range(config, split) for split in ALL_SPLITS
    }
    for first, second in combinations(ALL_SPLITS, 2):
        first_start, first_end = ranges[first]
        second_start, second_end = ranges[second]
        assert first_end <= second_start or second_end <= first_start
    for seed in config["algorithm_seeds"]:
        assert all(
            not start <= seed < end for start, end in ranges.values()
        )

    actual_seed_sets = {}
    manifest_root = project_path(config["paths"]["manifests_root"])
    for split in PERSISTED_SPLITS:
        manifest = json.loads(
            (manifest_root / split / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        actual_seed_sets[split] = {
            int(entry["seed"]) for entry in manifest["files"]
        }
    for first, second in combinations(PERSISTED_SPLITS, 2):
        assert actual_seed_sets[first].isdisjoint(actual_seed_sets[second])


def _fake_generated_record(
    generator,
    fixed_instance,
    *,
    seed,
    split,
    truncated,
):
    return GeneratedInstanceRecord(
        instance=fixed_instance,
        metadata={
            "generator_version": generator.version,
            "template_instance": generator.template_instance,
            "template_sha256": generator.template_hash,
            "seed": seed,
            "split": split,
            "distribution": split,
            "pressure_type": "balanced",
            "cost_profile": "balanced_cost",
            "ood_factor": "arrival_overlap",
            "heuristic_metrics": {
                "heuristic_truncated": truncated,
            },
        },
    )


def test_stress_split_enforces_dataset_truncation_quota(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    from data.generate_orders import InstanceGenerator

    def too_many_truncated(self, *, seed, split, **kwargs):
        return _fake_generated_record(
            self,
            fixed_instance,
            seed=seed,
            split=split,
            truncated=seed < 5_000_002,
        )

    monkeypatch.setattr(
        InstanceGenerator,
        "generate",
        too_many_truncated,
    )
    with pytest.raises(RuntimeError, match="stress truncated fraction"):
        build_dataset_split(
            config=config,
            template=fixed_instance,
            split="stress",
            count=5,
            instances_root=tmp_path / "instances",
            manifests_root=tmp_path / "manifests",
        )

    def within_quota(self, *, seed, split, **kwargs):
        return _fake_generated_record(
            self,
            fixed_instance,
            seed=seed,
            split=split,
            truncated=seed == 5_000_000,
        )

    monkeypatch.setattr(InstanceGenerator, "generate", within_quota)
    manifest_path = build_dataset_split(
        config=config,
        template=fixed_instance,
        split="stress",
        count=5,
        instances_root=tmp_path / "instances_allowed",
        manifests_root=tmp_path / "manifests_allowed",
    )
    dataset = InstanceDataset(
        manifest_path,
        instances_root=tmp_path / "instances_allowed",
        expected_split="stress",
        expected_seed_range=split_seed_range(config, "stress"),
    )
    assert len(dataset) == 5
    assert sum(
        record.metadata["heuristic_metrics"]["heuristic_truncated"]
        for record in dataset
    ) == 1


def test_strict_ood_rejects_truncation_but_stress_still_rejects_violations(
    instance_generator,
    pressure_records,
):
    instance = pressure_records["balanced"].instance
    metrics = deepcopy(
        pressure_records["balanced"].metadata["heuristic_metrics"]
    )
    metrics.update(
        {
            "heuristic_completed": False,
            "heuristic_truncated": True,
            "heuristic_makespan": instance.horizon,
        }
    )
    ood_reasons = instance_generator._dynamic_rejection_reasons(
        instance=instance,
        pressure_type="balanced",
        split="ood",
        metrics=metrics,
    )
    assert "heuristic_truncated" in ood_reasons

    stress_reasons = instance_generator._dynamic_rejection_reasons(
        instance=instance,
        pressure_type="balanced",
        split="stress",
        metrics=metrics,
    )
    assert "heuristic_truncated" not in stress_reasons
    metrics["schedule_violations"] = ["simulated violation"]
    stress_reasons = instance_generator._dynamic_rejection_reasons(
        instance=instance,
        pressure_type="balanced",
        split="stress",
        metrics=metrics,
    )
    assert "schedule_infeasible" in stress_reasons
