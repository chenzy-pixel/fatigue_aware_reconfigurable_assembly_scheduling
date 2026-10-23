"""Ordered global observation contract shared by environment and networks."""

from collections.abc import Mapping, Sequence

import numpy as np

OBSERVATION_SCHEMA_VERSION = 13
SUPPORTED_OBSERVATION_SCHEMAS = (OBSERVATION_SCHEMA_VERSION,)
GLOBAL_OBJECTIVE_ENCODINGS = ("log2_v1", "bounded_v1")
STATE_GLOBAL_FEATURE_NAMES = (
    "current_time_norm", "pending_reconfiguration_ratio", "completed_operation_ratio",
    "production_decision", "worker_matching_deficit_norm", "minimum_worker_alternative_ratio",
)


def normalize_global_objective_encoding(version: int, encoding: str | None = None) -> str:
    if isinstance(version, bool) or version not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ValueError("only observation schema 13 is supported; older models require retraining")
    selected = "log2_v1" if encoding is None else encoding
    if selected not in GLOBAL_OBJECTIVE_ENCODINGS:
        raise ValueError("global_objective_encoding must be log2_v1 or bounded_v1")
    return selected


def configured_global_objective_encoding(config: Mapping, version: int) -> str:
    network = config.get("network", config)
    return normalize_global_objective_encoding(version, network.get("global_objective_encoding"))


def global_feature_names(version: int, encoding: str | None = None) -> tuple[str, ...]:
    selected = normalize_global_objective_encoding(version, encoding)
    suffix = selected.removesuffix("_v1")
    return STATE_GLOBAL_FEATURE_NAMES + (
        f"objective_flow_{suffix}_norm", f"objective_cost_{suffix}_norm",
        f"objective_committed_workload_{suffix}_norm", "maximum_unfinished_order_remaining_norm", "unprojectable_order_ratio",
    )


GLOBAL_FEATURE_NAMES = global_feature_names(OBSERVATION_SCHEMA_VERSION)


def infer_global_observation_contract(names: Sequence[str]) -> tuple[int, str]:
    for encoding in GLOBAL_OBJECTIVE_ENCODINGS:
        if tuple(names) == global_feature_names(OBSERVATION_SCHEMA_VERSION, encoding):
            return OBSERVATION_SCHEMA_VERSION, encoding
    raise ValueError("global feature names/order are incompatible")


def encode_global_objectives(objectives: Sequence[float], scales: Sequence[float],
                             *, version: int, encoding: str | None = None) -> np.ndarray:
    """Encode cumulative objectives only; reward scalarization is independent."""
    values = np.asarray(objectives, dtype=np.float64)
    units = np.asarray(scales, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("global objectives must be three finite nonnegative values")
    if units.shape != (3,) or not np.isfinite(units).all() or (units <= 0).any():
        raise ValueError("global objective scales must be three finite positive values")
    selected = normalize_global_objective_encoding(version, encoding)
    if selected == "log2_v1":
        with np.errstate(divide="ignore"):
            log_ratio = np.log(values) - np.log(units)
        result = np.logaddexp(0.0, log_ratio) / np.log(2.0)
    else:
        # Equivalent to x/(1+x), without overflowing x or (scale + objective).
        small = values <= units
        result = np.empty(3, dtype=np.float64)
        ratio = values[small] / units[small]
        result[small] = ratio / (1.0 + ratio)
        result[~small] = 1.0 / (1.0 + units[~small] / values[~small])
    if not np.isfinite(result).all():
        raise ValueError("encoded global objectives must be finite")
    return result
