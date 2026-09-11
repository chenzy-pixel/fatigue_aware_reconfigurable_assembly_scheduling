from __future__ import annotations

import os

import torch

from utils import set_seed


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
