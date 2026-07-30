from __future__ import annotations

import pytest

from data.dataset import load_dataset_split, split_seed_range


@pytest.mark.parametrize("split", ("validation", "test", "ood", "stress"))
def test_published_development_split_contract(config, split):
    dataset = load_dataset_split(config, split)
    start, _ = split_seed_range(config, split)
    assert len(dataset) == 20
    records = list(dataset)
    assert [record.metadata["seed"] for record in records] == list(
        range(start, start + 20)
    )
    assert len({record.instance.instance_id for record in records}) == 20
    assert dataset.manifest["schema_version"] == "1.1.0"
    assert dataset.manifest["generator_version"] == "1.2.0"
    assert dataset.manifest["template_instance"] == "fixed_15x4_v1"
