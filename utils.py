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






def action_trace_sha256(actions: list[int]) -> str:
    payload = (
        json.dumps(
            [int(action) for action in actions],
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()






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
