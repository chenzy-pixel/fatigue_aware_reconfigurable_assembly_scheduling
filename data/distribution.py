"""Versioned sampling rules shared by online and fixed instance datasets."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping

PRESSURE_TYPES = (
    "easy", "balanced", "machine_bottleneck", "reconfiguration_bottleneck",
    "worker_bottleneck", "fatigue_bottleneck", "high_arrival_pressure",
)


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")


def benchmark_environment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the physical benchmark settings shared by fatigue ablations."""
    environment = dict(config["environment"])
    environment.pop("fatigue_mode", None)
    return environment


def protocol_hashes(config: Mapping[str, Any], settings: Mapping[str, Any] | None = None) -> dict[str, str]:
    # Historical snapshots retain these training-only fields. Fixed benchmark
    # hashes exclude them; the online cache fingerprints the full generator.
    generator = {key: value for key, value in (config["generator"] if settings is None else settings).items()
                 if key not in {"curriculum", "severity_curriculum", "sampling_window_episodes"}}
    payloads = {
        "generator_config_sha256": generator,
        "environment_config_sha256": benchmark_environment_config(config),
        "distribution_contract_sha256": {
            "version": "2.0.0", "pressure_types": PRESSURE_TYPES,
            "dataset": config["dataset"], "generator": generator,
            "precheck_version": "necessary_conditions_v1",
        },
    }
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for name, value in payloads.items()}


def validate_training_distribution(config: Mapping[str, Any]) -> None:
    """Reject retired training instructions before creating workers or output."""
    generator = config["generator"]
    if {"curriculum", "severity_curriculum"}.intersection(generator):
        raise ValueError(
            "historical curriculum configurations support evaluation and analysis only; "
            "reproduce training with the original code version"
        )
    weights = generator["dataset_pressure_weights"]
    if set(weights) != set(PRESSURE_TYPES):
        raise ValueError("training weights must define all seven pressure types")
    weighted_labels(1, weights)
    if int(generator["sampling_window_episodes"]) < 1:
        raise ValueError("sampling_window_episodes must be positive")


def severity_bounds(bounds: Any, severity: float, *, integer: bool = False) -> tuple[Any, Any]:
    low, high = bounds
    upper = float(low) + severity * (float(high) - float(low))
    return (int(low), int(math.floor(upper + 1e-9))) if integer else (float(low), upper)


def weighted_labels(count: int, weights: Mapping[str, float]) -> list[str]:
    if count < 1 or not weights or any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
        raise ValueError("invalid sampling count or weights")
    total = sum(map(float, weights.values()))
    if total <= 0:
        raise ValueError("weights must have positive total")
    names = [name for name in PRESSURE_TYPES if name in weights]
    names += sorted(set(weights) - set(names))
    exact = {name: count * float(weights[name]) / total for name in names}
    quotas = {name: math.floor(exact[name]) for name in names}
    ranked = sorted(names, key=lambda name: (-round(exact[name] - quotas[name], 12), names.index(name)))
    for name in ranked[:count - sum(quotas.values())]:
        quotas[name] += 1
    labels = []
    while len(labels) < count:
        for name in names:
            if quotas[name]:
                labels.append(name)
                quotas[name] -= 1
    return labels


def training_sampling_plan(config: Mapping[str, Any], episode_count: int) -> list[tuple[str, float]]:
    """Allocate fixed scenario quotas and shuffle each window deterministically."""
    validate_training_distribution(config)
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    window = int(config["generator"]["sampling_window_episodes"])
    contract = protocol_hashes(config)["distribution_contract_sha256"]
    result = []
    for start in range(0, episode_count, window):
        end = min(start + window, episode_count)
        labels = weighted_labels(end - start, config["generator"]["dataset_pressure_weights"])
        random.Random(stable_seed("training-window-v2", contract, config["dataset"]["splits"]["train"]["seed_start"], episode_count, start)).shuffle(labels)
        result.extend((label, 1.0) for label in labels)
    return result
