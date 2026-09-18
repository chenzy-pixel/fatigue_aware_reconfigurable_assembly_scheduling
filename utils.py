from __future__ import annotations

import os
import random
import hashlib
import json
from typing import Any

import numpy as np


SAMPLED_EVALUATION_RNG_VERSION = "evaluation_unit_sha256_v2"


def derive_evaluation_sampling_seed(
    sampling_seed: int,
    instance_id: str,
    evaluation_key: str | None = None,
) -> int:
    """Derive an independent Torch seed for one fixed evaluation unit.

    ``evaluation_key`` is omitted for specialist evaluation and set to the
    preference key for Universal V8.  This gives every instance-preference
    pair its own reproducible stream while preserving common random numbers
    when candidate and incumbent checkpoints are compared.
    """
    stable_id = str(instance_id)
    if not stable_id:
        raise ValueError("instance_id must not be empty")
    stable_key = "" if evaluation_key is None else str(evaluation_key)
    payload = (
        f"{SAMPLED_EVALUATION_RNG_VERSION}\0{int(sampling_seed)}\0"
        f"{stable_id}\0{stable_key}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def formal_evaluation_sampling_seeds(
    algorithm_seed: int,
    *,
    seed_offset: int,
    repeats: int,
) -> list[int]:
    """Return one reproducible, positive seed namespace for formal evaluation."""

    base = int(algorithm_seed)
    offset = int(seed_offset)
    count = int(repeats)
    if offset < 1 or count < 1:
        raise ValueError("formal evaluation seed offset/repeats must be positive")
    return [base + offset + repeat for repeat in range(count)]


def configured_formal_evaluation_sampling_seeds(
    config: dict[str, Any],
    namespace: str,
) -> list[int]:
    """Resolve the disjoint validation/audit/final-test seed namespace."""

    fields = {
        "validation": ("validation_seed_offset", "validation_repeats"),
        "audit": ("audit_seed_offset", "audit_repeats"),
        "final_test": ("final_test_seed_offset", "final_test_repeats"),
    }
    if namespace not in fields:
        raise ValueError(f"unknown formal evaluation namespace {namespace!r}")
    settings = config["training"]["formal_evaluation"]
    if str(settings.get("decode_mode")) != "sampled":
        raise ValueError("formal PPO evaluation must use sampled decoding")
    if float(settings.get("temperature", 1.0)) != 1.0:
        raise ValueError("formal PPO evaluation requires temperature 1.0")
    offset_field, repeats_field = fields[namespace]
    return formal_evaluation_sampling_seeds(
        int(config["seed"]),
        seed_offset=int(settings[offset_field]),
        repeats=int(settings[repeats_field]),
    )


def action_trace_sha256(actions: list[int]) -> str:
    payload = (
        json.dumps(
            [int(action) for action in actions],
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def capture_global_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUDA >= 10.2 requires a CuBLAS workspace policy for deterministic GEMM.
    # Set it before the first CUDA operation while preserving an explicit
    # caller choice between the two supported deterministic configurations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
