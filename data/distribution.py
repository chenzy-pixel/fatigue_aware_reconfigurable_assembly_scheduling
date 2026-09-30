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


def protocol_hashes(config: Mapping[str, Any], settings: Mapping[str, Any] | None = None) -> dict[str, str]:
    # A fixed evaluation pool is shared across curriculum ablations. The online
    # cache fingerprints the full generator separately, including the course.
    generator = {key: value for key, value in (config["generator"] if settings is None else settings).items()
                 if key not in {"curriculum", "severity_curriculum", "sampling_window_episodes"}}
    payloads = {
        "generator_config_sha256": generator,
        "environment_config_sha256": config["environment"],
        "distribution_contract_sha256": {
            "version": "2.0.0", "pressure_types": PRESSURE_TYPES,
            "dataset": config["dataset"], "generator": generator,
            "precheck_version": "necessary_conditions_v1",
        },
    }
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for name, value in payloads.items()}


def severity_at(config: Mapping[str, Any], progress: float) -> float:
    anchors = config["generator"]["severity_curriculum"]["anchors"]
    if not 0.0 <= progress <= 1.0:
        raise ValueError("severity progress must be in [0, 1]")
    if len(anchors) < 2 or anchors[0]["at_fraction"] != 0 or anchors[-1]["at_fraction"] != 1:
        raise ValueError("severity anchors must cover [0, 1]")
    previous = -1.0
    for anchor in anchors:
        at, value = float(anchor["at_fraction"]), float(anchor["value"])
        if at <= previous or not 0 <= at <= 1 or not 0 < value <= 1:
            raise ValueError("invalid severity curriculum anchors")
        previous = at
    for left, right in zip(anchors, anchors[1:]):
        if progress <= right["at_fraction"]:
            alpha = (progress - left["at_fraction"]) / (right["at_fraction"] - left["at_fraction"])
            return float(left["value"] + alpha * (right["value"] - left["value"]))
    return float(anchors[-1]["value"])


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
    # Import lazily: dataset also uses the shared protocol helpers.
    from data.dataset import curriculum_weights_at

    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    window = int(config["generator"]["sampling_window_episodes"])
    if window < 1:
        raise ValueError("sampling_window_episodes must be positive")
    contract = protocol_hashes(config)["distribution_contract_sha256"]
    result = []
    for start in range(0, episode_count, window):
        end = min(start + window, episode_count)
        progress = [(index + 0.5) / episode_count for index in range(start, end)]
        averages = {name: 0.0 for name in PRESSURE_TYPES}
        for fraction in progress:
            weights = curriculum_weights_at(config["generator"]["curriculum"], fraction)
            if set(weights) != set(PRESSURE_TYPES):
                raise ValueError("curriculum must define all seven pressure types")
            for name in PRESSURE_TYPES:
                averages[name] += weights[name] / len(progress)
        labels = weighted_labels(end - start, averages)
        random.Random(stable_seed("training-window-v2", contract, config["dataset"]["splits"]["train"]["seed_start"], episode_count, start)).shuffle(labels)
        result.extend((label, severity_at(config, fraction)) for label, fraction in zip(labels, progress, strict=True))
    return result
