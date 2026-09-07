from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PREFERENCE_NAMES = ("flow", "cost", "variance")
CANONICAL_PREFERENCE = (0.5, 0.3, 0.2)
PREFERENCE_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PreferenceVector:
    """Validated flow/cost/variance weights on the probability simplex."""

    flow: float
    cost: float
    variance: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("preference weights must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("preference weights must be non-negative")
        if not math.isclose(
            sum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=PREFERENCE_SUM_TOLERANCE,
        ):
            raise ValueError("preference weights must sum to 1")

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.flow), float(self.cost), float(self.variance))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.as_tuple(), dtype=np.float32)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(PREFERENCE_NAMES, self.as_tuple(), strict=True))


PreferenceInput = PreferenceVector | Mapping[str, float] | Sequence[float]


def normalize_preference(value: PreferenceInput) -> PreferenceVector:
    """Return one strict preference vector without silently renormalizing it."""

    if isinstance(value, PreferenceVector):
        return value
    if isinstance(value, Mapping):
        if set(value) != set(PREFERENCE_NAMES):
            raise ValueError(
                "preference mapping must contain exactly flow/cost/variance"
            )
        values = tuple(float(value[name]) for name in PREFERENCE_NAMES)
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError("preference must be a numeric sequence or mapping")
        values = tuple(float(item) for item in value)
        if len(values) != len(PREFERENCE_NAMES):
            raise ValueError("preference must contain exactly three weights")
    return PreferenceVector(*values)


def preference_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the optional repository-level preference config."""

    raw = config.get("preference")
    if raw is None:
        reward = config.get("reward", {})
        weights = reward.get("quality_weights", CANONICAL_PREFERENCE)
        if isinstance(weights, Mapping):
            default = normalize_preference(weights)
        else:
            default = normalize_preference(weights)
        return {
            "enabled": False,
            "names": PREFERENCE_NAMES,
            "default": default,
            "sampler": None,
        }
    if not isinstance(raw, Mapping):
        raise TypeError("config.preference must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("preference.enabled must be boolean")
    network = config.get("network", {})
    if not isinstance(network, Mapping):
        raise TypeError("config.network must be an object")
    conditioning = str(network.get("preference_conditioning", "none"))
    centered_adapter = bool(
        network.get("production_action_semantics")
        == "hierarchical_e1_logsumexp_gate_then_pair_v4"
        and isinstance(network.get("centered_preference_adapter"), Mapping)
        and network["centered_preference_adapter"].get("enabled", False)
    )
    if enabled and conditioning != "separate_encoder_v1" and not centered_adapter:
        raise ValueError(
            "enabled preferences require "
            "network.preference_conditioning='separate_encoder_v1' or the "
            "E1-centered parallel adapter"
        )
    if not enabled and conditioning != "none":
        raise ValueError(
            "preference conditioning requires preference.enabled=true"
        )
    names = tuple(raw.get("names", PREFERENCE_NAMES))
    if names != PREFERENCE_NAMES:
        raise ValueError("preference.names must be flow/cost/variance in order")
    default = normalize_preference(raw.get("default", CANONICAL_PREFERENCE))
    sampler_raw = raw.get("sampler")
    sampler = None
    if enabled:
        if not isinstance(sampler_raw, Mapping):
            raise TypeError("preference.sampler must be an object when enabled")
        version = str(sampler_raw.get("version", ""))
        if version != "dirichlet_anchor_mixture_v1":
            raise ValueError(
                "preference.sampler.version must be "
                "'dirichlet_anchor_mixture_v1'"
            )
        probability = float(sampler_raw.get("dirichlet_probability", 0.7))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("dirichlet_probability must be in [0, 1]")
        concentration = tuple(
            float(value)
            for value in sampler_raw.get("concentration", (1.0, 1.0, 1.0))
        )
        if len(concentration) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in concentration
        ):
            raise ValueError(
                "preference sampler concentration must contain three "
                "finite positive values"
            )
        anchors_raw = sampler_raw.get("anchors")
        if not isinstance(anchors_raw, Sequence) or isinstance(
            anchors_raw, (str, bytes)
        ):
            raise TypeError("preference sampler anchors must be a sequence")
        anchors = tuple(normalize_preference(value) for value in anchors_raw)
        if not anchors:
            raise ValueError("preference sampler must contain at least one anchor")
        sampler = {
            "version": version,
            "dirichlet_probability": probability,
            "concentration": concentration,
            "anchors": anchors,
            "seed_derivation": "algorithm_seed_episode_sha256_v1",
        }
    elif sampler_raw is not None:
        raise ValueError("disabled preference conditioning cannot define a sampler")
    return {
        "enabled": enabled,
        "names": names,
        "default": default,
        "sampler": sampler,
    }


def default_preference(config: Mapping[str, Any]) -> PreferenceVector:
    return preference_config(config)["default"]


def preference_enabled(config: Mapping[str, Any]) -> bool:
    return bool(preference_config(config)["enabled"])


def derive_preference_sampling_seed(
    algorithm_seed: int,
    episode_index: int,
) -> int:
    return _derive_episode_seed(
        "e2_preference_v1",
        algorithm_seed,
        episode_index,
    )


def derive_episode_action_seed(
    algorithm_seed: int,
    episode_index: int,
) -> int:
    """Derive an E2 policy-sampling stream independent of batch scheduling."""

    return _derive_episode_seed(
        "e2_policy_action_v1",
        algorithm_seed,
        episode_index,
    )


def _derive_episode_seed(
    domain: str,
    algorithm_seed: int,
    episode_index: int,
) -> int:
    if int(algorithm_seed) < 0:
        raise ValueError("algorithm_seed must be non-negative")
    if int(episode_index) < 0:
        raise ValueError("episode_index must be non-negative")
    payload = f"{domain}|{int(algorithm_seed)}|{int(episode_index)}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_episode_preference(
    config: Mapping[str, Any],
    *,
    algorithm_seed: int,
    episode_index: int,
) -> tuple[PreferenceVector, str]:
    """Sample an episode preference without touching any global RNG."""

    normalized = preference_config(config)
    if not normalized["enabled"]:
        return normalized["default"], "fixed_default"
    sampler = normalized["sampler"]
    if sampler is None:
        raise RuntimeError("enabled preference conditioning has no sampler")
    rng = np.random.default_rng(
        derive_preference_sampling_seed(algorithm_seed, episode_index)
    )
    if float(rng.random()) < sampler["dirichlet_probability"]:
        values = rng.dirichlet(sampler["concentration"])
        return normalize_preference(values), "dirichlet"
    anchor_index = int(rng.integers(0, len(sampler["anchors"])))
    return sampler["anchors"][anchor_index], f"anchor_{anchor_index}"


def simplex_lattice(
    denominator: int = 5,
    *,
    include: Sequence[PreferenceInput] = (CANONICAL_PREFERENCE,),
) -> tuple[PreferenceVector, ...]:
    """Return the deterministic 3-objective simplex lattice plus extra points."""

    denominator = int(denominator)
    if denominator < 1:
        raise ValueError("simplex lattice denominator must be positive")
    points = [
        PreferenceVector(
            first / denominator,
            second / denominator,
            (denominator - first - second) / denominator,
        )
        for first in range(denominator + 1)
        for second in range(denominator - first + 1)
    ]
    points.extend(normalize_preference(value) for value in include)
    unique: list[PreferenceVector] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        key = tuple(round(value, 12) for value in point.as_tuple())
        if key not in seen:
            unique.append(point)
            seen.add(key)
    return tuple(unique)
