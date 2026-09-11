from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


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


@dataclass(frozen=True)
class PreferenceContext:
    """One normalized objective preference shared by an entire episode."""

    preference: PreferenceVector
    source: str = "fixed"
    sample_index: int | None = None

    @classmethod
    def from_input(
        cls,
        value: "PreferenceContext | PreferenceInput",
        *,
        source: str = "fixed",
        sample_index: int | None = None,
    ) -> "PreferenceContext":
        if isinstance(value, cls):
            return cls(
                value.preference,
                source=value.source,
                sample_index=value.sample_index,
            )
        return cls(
            normalize_preference(value),
            source=str(source),
            sample_index=(
                None if sample_index is None else int(sample_index)
            ),
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return self.preference.as_tuple()

    def as_array(self) -> np.ndarray:
        return self.preference.as_array()

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self.preference.as_dict(),
            "source": self.source,
            "sample_index": self.sample_index,
            "key": self.key,
        }

    @property
    def key(self) -> str:
        rendered = ",".join(f"{value:.9f}" for value in self.as_tuple())
        return hashlib.sha256(rendered.encode("ascii")).hexdigest()[:16]


PreferenceContextInput = PreferenceContext | PreferenceInput


def normalize_preference(value: PreferenceInput) -> PreferenceVector:
    """Normalize non-negative finite weights onto the probability simplex."""

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
    if any(not math.isfinite(item) for item in values):
        raise ValueError("preference weights must be finite")
    if any(item < 0.0 for item in values):
        raise ValueError("preference weights must be non-negative")
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("preference weights must have a finite positive sum")
    normalized = tuple(item / total for item in values)
    # Pin the final coordinate to the simplex after floating-point division.
    normalized = (
        normalized[0],
        normalized[1],
        1.0 - normalized[0] - normalized[1],
    )
    return PreferenceVector(*normalized)


def default_preference(config: Mapping[str, object]) -> PreferenceVector:
    """Return the configured fixed quality preference."""

    preference = config.get("preference", {})
    if preference:
        if not isinstance(preference, Mapping):
            raise TypeError("config.preference must be an object")
        quality = preference.get("quality", {})
        if not isinstance(quality, Mapping):
            raise TypeError("config.preference.quality must be an object")
        return normalize_preference(
            quality.get("fixed", CANONICAL_PREFERENCE)
        )
    # Historical result/config readers may still construct an environment from
    # an archived V7 mapping. New V8 configs are validated separately.
    reward = config.get("reward", {})
    if not isinstance(reward, Mapping):
        raise TypeError("config.reward must be an object")
    return normalize_preference(reward.get("quality_weights", CANONICAL_PREFERENCE))


def default_preference_context(
    config: Mapping[str, object],
) -> PreferenceContext:
    return PreferenceContext(default_preference(config), source="fixed")


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


def quality_preference_for_episode(
    config: Mapping[str, object],
    *,
    algorithm_seed: int,
    quality_episode_index: int,
) -> PreferenceContext:
    """Return the deterministic V8 endpoint/Sobol quality preference."""

    if quality_episode_index < 0:
        raise ValueError("quality_episode_index must be non-negative")
    preference = config.get("preference", {})
    if not isinstance(preference, Mapping):
        raise TypeError("config.preference must be an object")
    quality = preference.get("quality", {})
    if not isinstance(quality, Mapping):
        raise TypeError("config.preference.quality must be an object")
    mode = str(quality.get("mode", "fixed"))
    if mode == "fixed":
        return PreferenceContext(
            normalize_preference(quality.get("fixed", CANONICAL_PREFERENCE)),
            source="quality_fixed",
            sample_index=quality_episode_index,
        )
    if mode != "universal_sobol_v1":
        raise ValueError(f"unknown preference quality mode {mode!r}")
    block_size = int(quality.get("block_size", 20))
    endpoint_repeats = int(quality.get("endpoint_repeats", 2))
    sobol_count = int(quality.get("sobol_count", 14))
    if block_size != 20 or endpoint_repeats != 2 or sobol_count != 14:
        raise ValueError("V8 universal blocks require 20/2/14 preference quotas")
    offset = quality_episode_index % block_size
    endpoint_slots = endpoint_repeats * len(PREFERENCE_NAMES)
    if offset < endpoint_slots:
        objective = offset // endpoint_repeats
        weights = [0.0, 0.0, 0.0]
        weights[objective] = 1.0
        return PreferenceContext(
            normalize_preference(weights),
            source=f"endpoint_{PREFERENCE_NAMES[objective]}",
            sample_index=quality_episode_index,
        )
    sobol_index = (quality_episode_index // block_size) * sobol_count + (
        offset - endpoint_slots
    )
    engine = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=int(algorithm_seed),
    )
    if sobol_index:
        engine.fast_forward(sobol_index)
    uniform = engine.draw(1).squeeze(0).to(dtype=torch.float64)
    exponential = -torch.log(uniform.clamp(min=1e-12, max=1.0 - 1e-12))
    weights = (exponential / exponential.sum()).cpu().numpy()
    return PreferenceContext(
        normalize_preference(weights),
        source="scrambled_sobol_exponential_simplex",
        sample_index=sobol_index,
    )


def feasibility_preference_context(
    config: Mapping[str, object],
) -> PreferenceContext:
    preference = config.get("preference", {})
    if not isinstance(preference, Mapping):
        raise TypeError("config.preference must be an object")
    return PreferenceContext(
        normalize_preference(
            preference.get(
                "feasibility", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            )
        ),
        source="feasibility_balanced",
    )
