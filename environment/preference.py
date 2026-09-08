from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


def default_preference(config: Mapping[str, object]) -> PreferenceVector:
    """Use the run's fixed objective weights; policy conditioning is unsupported."""

    reward = config.get("reward", {})
    if not isinstance(reward, Mapping):
        raise TypeError("config.reward must be an object")
    return normalize_preference(
        reward.get("quality_weights", CANONICAL_PREFERENCE)
    )


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
