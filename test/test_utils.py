from __future__ import annotations

import os

import torch

from configs import load_config
from utils import (
    configured_formal_evaluation_sampling_seeds,
    derive_evaluation_sampling_seed,
    set_seed,
)


def test_set_seed_configures_deterministic_cuda_libraries(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    set_seed(11)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_set_seed_preserves_supported_explicit_cublas_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    set_seed(11)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_formal_sampling_namespaces_are_disjoint_and_algorithm_seed_relative():
    config = load_config("configs/default.json")
    assert configured_formal_evaluation_sampling_seeds(
        config, "validation"
    ) == [100011, 100012, 100013]
    assert configured_formal_evaluation_sampling_seeds(
        config, "audit"
    ) == [200011]
    assert configured_formal_evaluation_sampling_seeds(
        config, "final_test"
    ) == [300011, 300012, 300013]

    config["seed"] = 47
    namespaces = [
        set(configured_formal_evaluation_sampling_seeds(config, name))
        for name in ("validation", "audit", "final_test")
    ]
    assert namespaces[0] == {100047, 100048, 100049}
    assert not (namespaces[0] & namespaces[1])
    assert not (namespaces[0] & namespaces[2])
    assert not (namespaces[1] & namespaces[2])


def test_universal_sampling_seed_is_preference_specific_and_pair_reproducible():
    first = derive_evaluation_sampling_seed(100011, "instance-1", "pref-a")
    repeated = derive_evaluation_sampling_seed(100011, "instance-1", "pref-a")
    second = derive_evaluation_sampling_seed(100011, "instance-1", "pref-b")
    specialist = derive_evaluation_sampling_seed(100011, "instance-1")
    assert first == repeated
    assert len({first, second, specialist}) == 3
