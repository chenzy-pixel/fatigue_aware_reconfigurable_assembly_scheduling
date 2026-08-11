from __future__ import annotations

from copy import deepcopy

import pytest

from data.dataset import (
    GeneratedInstanceRecord,
    OnlineInstanceDataset,
    dataset_profile_counts,
    resolve_dataset_count,
    split_seed_range,
    validate_algorithm_seed,
    validate_instance_seed,
    validate_seed_configuration,
)
from data.generate_orders import InstanceGenerator


def test_seed_ranges_profiles_and_algorithm_seeds(config):
    validate_seed_configuration(config)
    assert split_seed_range(config, "train") == (1_000_000, 2_000_000)
    assert split_seed_range(config, "validation") == (
        2_000_000,
        3_000_000,
    )
    assert split_seed_range(config, "test") == (3_000_000, 4_000_000)
    assert split_seed_range(config, "ood") == (4_000_000, 5_000_000)
    assert split_seed_range(config, "stress") == (
        5_000_000,
        6_000_000,
    )
    assert dataset_profile_counts(config, "dev") == {
        "validation": 20,
        "test": 20,
        "ood": 20,
        "stress": 20,
    }
    assert dataset_profile_counts(config, "publication") == {
        "validation": 500,
        "test": 1000,
        "ood": 500,
        "stress": 500,
    }
    assert resolve_dataset_count(
        config,
        split="test",
        profile="publication",
        count=3,
    ) == 3
    for seed in (11, 23, 37, 53, 71):
        assert validate_algorithm_seed(config, seed) == seed
    with pytest.raises(ValueError, match="not one of"):
        validate_algorithm_seed(config, 12)
    validate_instance_seed(config, "train", 1_000_000)
    validate_instance_seed(config, "train", 1_999_999)
    with pytest.raises(ValueError, match="outside"):
        validate_instance_seed(config, "train", 2_000_000)


def test_overlapping_seed_configuration_is_rejected(config):
    invalid = deepcopy(config)
    invalid["dataset"]["splits"]["validation"]["seed_start"] = 1_999_999
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_configuration(invalid)


def test_online_dataset_uses_episode_seeds_independent_of_algorithm_seed(
    config,
    fixed_instance,
    monkeypatch,
):
    calls: list[tuple[int, str, str, bool]] = []

    def fake_generate(
        self,
        *,
        seed,
        split,
        pressure_type,
        ood_factor=None,
        classify_reconfiguration_value=True,
    ):
        calls.append(
            (
                seed,
                split,
                pressure_type,
                classify_reconfiguration_value,
            )
        )
        return GeneratedInstanceRecord(
            instance=fixed_instance,
            metadata={
                "seed": seed,
                "split": split,
                "pressure_type": pressure_type,
            },
        )

    monkeypatch.setattr(InstanceGenerator, "generate", fake_generate)
    first = OnlineInstanceDataset(
        config=config,
        template=fixed_instance,
        episode_count=3,
    )
    first_records = [first[index] for index in range(3)]

    other_seed_config = deepcopy(config)
    other_seed_config["seed"] = 71
    second = OnlineInstanceDataset(
        config=other_seed_config,
        template=fixed_instance,
        episode_count=3,
    )
    second_records = [second[index] for index in range(3)]

    assert [record.metadata["seed"] for record in first_records] == [
        1_000_000,
        1_000_001,
        1_000_002,
    ]
    assert [
        record.metadata["pressure_type"] for record in first_records
    ] == [
        record.metadata["pressure_type"] for record in second_records
    ]
    assert all(
        split == "train" and classify_reconfiguration_value is False
        for _, split, _, classify_reconfiguration_value in calls
    )
