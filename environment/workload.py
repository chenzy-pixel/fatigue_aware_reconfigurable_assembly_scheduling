"""Versioned objectives for actual DIS/INS labor minutes, including idle workers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import math

import numpy as np
from .observation_schema import OBSERVATION_SCHEMA_VERSION, SUPPORTED_OBSERVATION_SCHEMAS

WORKLOAD_VERSION = "workload_objective_v1"
GEOMETRIC_WEIGHT_RULE = "normalized_geometric_by_worker_count_v1"
VARIANCE_WEIGHT_RULE = "population_variance_v1"
MODERN_OBJECTIVE_NAMES = ("flow", "cost", "workload")


def _network(config: Mapping) -> Mapping:
    value = config.get("network", config)
    if not isinstance(value, Mapping):
        raise TypeError("network must be an object")
    return value


def workload_config(config: Mapping) -> dict:
    """Canonicalize the full or flattened workload identity without inference."""
    network = _network(config)
    explicit = config.get("workload_objective")
    if "workload_objective" in config:
        if not isinstance(explicit, Mapping):
            raise TypeError("workload_objective must be an object")
        if "mode" not in explicit:
            raise ValueError("workload_objective.mode is required")
        raw = explicit
    elif "workload_mode" in network:
        raw = {
            "mode": network["workload_mode"],
            "q": network.get("workload_q"),
            "version": network.get("workload_version", WORKLOAD_VERSION),
            "order": network.get("workload_order", "descending"),
            "weights": network.get("workload_weight_rule", network.get("workload_rule")),
        }
    else:
        raise ValueError("an explicit workload_objective or network workload_mode is required")
    mode = str(raw["mode"])
    if mode not in {"variance", "geometric_owa"}:
        raise ValueError("workload_objective.mode must be variance or geometric_owa")
    if mode == "geometric_owa":
        q_value = raw.get("q", 0.5)
        if isinstance(q_value, bool) or q_value is None:
            raise ValueError("geometric_owa q must be finite and in (0, 1)")
        q = float(q_value)
        if not math.isfinite(q) or not 0.0 < q < 1.0:
            raise ValueError("geometric_owa q must be finite and in (0, 1)")
        weight_rule = GEOMETRIC_WEIGHT_RULE
    else:
        if raw.get("q") is not None:
            raise ValueError("variance workload objective requires q=None")
        q = None
        weight_rule = VARIANCE_WEIGHT_RULE
    canonical = {"mode": mode, "q": q, "version": WORKLOAD_VERSION,
                 "order": "descending", "weights": weight_rule}
    for field in ("version", "order", "weights"):
        if raw.get(field) is not None and raw[field] != canonical[field]:
            raise ValueError(f"workload_objective.{field} is incompatible")
    # A full config must agree with any identity stamped into its network.
    if explicit is not None and "workload_mode" in network:
        flattened = workload_config({key: value for key, value in network.items() if key != "network"})
        if flattened != canonical:
            raise ValueError("network workload identity differs from workload_objective")
    return canonical


def observation_schema_version(config: Mapping) -> int:
    """Validate the current observation and explicit workload contracts."""
    network = _network(config)
    value = network.get("observation_schema_version", OBSERVATION_SCHEMA_VERSION)
    if isinstance(value, bool) or value not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ValueError("only observation schema 12 is supported; older models require retraining")
    workload_config(config)
    return OBSERVATION_SCHEMA_VERSION


def objective_names(config: Mapping) -> tuple[str, str, str]:
    observation_schema_version(config)
    return MODERN_OBJECTIVE_NAMES


@lru_cache(maxsize=256)
def geometric_owa_weights(worker_count: int, q: float) -> np.ndarray:
    """Return immutable positive normalized geometric weights for descending ranks."""
    if isinstance(worker_count, bool) or not isinstance(worker_count, (int, np.integer)) or worker_count < 1:
        raise ValueError("worker count must be a positive integer")
    if isinstance(q, bool) or not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("geometric_owa q must be finite and in (0, 1)")
    powers = np.power(float(q), np.arange(worker_count, dtype=np.float64))
    weights = powers / math.fsum(powers.tolist())
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("geometric_owa weights underflow or are not finite and strictly positive")
    weights.setflags(write=False)
    return weights


def workload_value(loads: Sequence[float] | np.ndarray, config: Mapping) -> float:
    """Aggregate every worker's nonnegative actual or committed labor minutes."""
    values = np.asarray(loads, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("worker loads must be a finite nonnegative one-dimensional vector")
    definition = workload_config(config)
    if not values.size:
        return 0.0
    if definition["mode"] == "variance":
        return float(np.var(values))
    weights = geometric_owa_weights(len(values), definition["q"])
    return float(np.dot(weights, np.sort(values)[::-1]))


def workload_delta(before: Sequence[float], after: Sequence[float], config: Mapping) -> float:
    if len(before) != len(after):
        raise ValueError("workload delta requires the same worker count")
    return workload_value(after, config) - workload_value(before, config)


def workload_scale(config: Mapping) -> float:
    scalarizer = config.get("objective_scalarizer", config)
    scales = scalarizer.get("scales", {})
    value = scales.get("workload")
    if value is None:
        value = scalarizer.get("workload_scale")
    if value is None:
        raise ValueError("a workload objective scale is required")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("workload objective scale must be finite and positive")
    return scale


def metrics_workload_objective(metrics: Mapping, config: Mapping) -> float:
    if "workload_objective_value" in metrics:
        return float(metrics["workload_objective_value"])
    if workload_config(config)["mode"] == "variance":
        return float(metrics["worker_load_variance"])
    raise KeyError("geometric_owa metrics require workload_objective_value")
