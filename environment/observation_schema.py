"""Ordered global observation contract shared by environment and networks."""

from collections.abc import Mapping, Sequence

import numpy as np

OBSERVATION_SCHEMA_VERSION = 12
SUPPORTED_OBSERVATION_SCHEMAS = (10, 11, 12)
GLOBAL_OBJECTIVE_ENCODINGS = ("log2_v1", "bounded_v1")
LEGACY_OBSERVATION_SCHEMA_VERSION = 10
LEGACY_GLOBAL_FEATURE_NAMES = (
    "current_time_norm", "pending_reconfiguration_ratio", "completed_operation_ratio",
    "production_decision", "worker_matching_deficit_norm", "minimum_worker_alternative_ratio",
    "objective_flow_norm", "objective_cost_norm", "objective_committed_variance_norm",
)
SCHEMA11_GLOBAL_FEATURE_NAMES = LEGACY_GLOBAL_FEATURE_NAMES[:-1] + ("objective_committed_workload_norm",)


def normalize_global_objective_encoding(version: int, encoding: str | None = None) -> str:
    if version not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ValueError("unsupported observation schema version")
    if version < 12:
        if encoding is not None and encoding != "linear_v1":
            raise ValueError("schema 10/11 require linear global objective encoding")
        return "linear_v1"
    selected = "log2_v1" if encoding is None else encoding
    if selected not in GLOBAL_OBJECTIVE_ENCODINGS:
        raise ValueError("global_objective_encoding must be log2_v1 or bounded_v1")
    return selected


def configured_global_objective_encoding(config: Mapping, version: int) -> str:
    network = config.get("network", config)
    return normalize_global_objective_encoding(version, network.get("global_objective_encoding"))


def global_feature_names(version: int, encoding: str | None = None) -> tuple[str, ...]:
    selected = normalize_global_objective_encoding(version, encoding)
    if version == 10:
        return LEGACY_GLOBAL_FEATURE_NAMES
    if version == 11:
        return SCHEMA11_GLOBAL_FEATURE_NAMES
    suffix = selected.removesuffix("_v1")
    return SCHEMA11_GLOBAL_FEATURE_NAMES[:6] + (
        f"objective_flow_{suffix}_norm", f"objective_cost_{suffix}_norm",
        f"objective_committed_workload_{suffix}_norm", "minimum_unfinished_order_slack_norm",
    )


GLOBAL_FEATURE_NAMES = global_feature_names(OBSERVATION_SCHEMA_VERSION)


def infer_global_observation_contract(names: Sequence[str]) -> tuple[int, str]:
    for version, encoding in ((10, "linear_v1"), (11, "linear_v1"),
                              *((12, item) for item in GLOBAL_OBJECTIVE_ENCODINGS)):
        if tuple(names) == global_feature_names(version, encoding):
            return version, encoding
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
    if selected == "linear_v1":
        result = values / units
    elif selected == "log2_v1":
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
