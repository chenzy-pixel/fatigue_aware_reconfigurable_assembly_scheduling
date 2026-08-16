from __future__ import annotations

import os
import random
import hashlib
import json
from typing import Any

import numpy as np


SAMPLED_EVALUATION_RNG_VERSION = "per_instance_sha256_v1"


def derive_evaluation_sampling_seed(
    sampling_seed: int,
    instance_id: str,
) -> int:
    """Derive an independent Torch seed without touching global RNG state."""
    stable_id = str(instance_id)
    if not stable_id:
        raise ValueError("instance_id must not be empty")
    payload = (
        f"{SAMPLED_EVALUATION_RNG_VERSION}\0{int(sampling_seed)}\0"
        f"{stable_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
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
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
