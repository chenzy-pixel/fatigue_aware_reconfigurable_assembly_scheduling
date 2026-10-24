"""Numerical Pareto comparisons, normalization and exact three-objective HV."""
from __future__ import annotations
import math
from typing import Iterable, Sequence

RELATIVE_TOLERANCE = 1e-9


REFERENCE_POINT = (1.0, 1.0, 1.0)


def _relative_slack(first: float, second: float, tolerance: float) -> float:
    return tolerance * max(1.0, abs(first), abs(second))


def vectors_equal(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    """Return whether two objective vectors are equal within relative tolerance."""

    if len(first) != len(second):
        return False
    return all(
        abs(left - right) <= _relative_slack(left, right, tolerance)
        for left, right in zip(first, second, strict=True)
    )


def dominates(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    """Return whether ``first`` Pareto-dominates ``second`` for minimization."""

    if len(first) != len(second):
        raise ValueError("objective vectors must have the same dimension")
    weakly_better = True
    strictly_better = False
    for left, right in zip(first, second, strict=True):
        slack = _relative_slack(left, right, tolerance)
        if left > right + slack:
            weakly_better = False
            break
        if left < right - slack:
            strictly_better = True
    return weakly_better and strictly_better


def nondominated_indices(
    points: Sequence[Sequence[float]],
    *,
    tolerance: float = RELATIVE_TOLERANCE,
) -> list[int]:
    """Return stable indices of all nondominated points, retaining equal points."""

    return [
        index
        for index, point in enumerate(points)
        if not any(
            other_index != index
            and dominates(other, point, tolerance=tolerance)
            for other_index, other in enumerate(points)
        )
    ]


def normalize_objectives(
    objectives: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, float, float]:
    """Apply the canonical monotone bounded transform ``x / (scale + x)``."""

    if len(objectives) != 3 or len(scales) != 3:
        raise ValueError("normalization requires three objectives and three scales")
    normalized: list[float] = []
    for objective, scale in zip(objectives, scales, strict=True):
        value = float(objective)
        denominator_scale = float(scale)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("objectives must be finite and nonnegative")
        if not math.isfinite(denominator_scale) or denominator_scale <= 0.0:
            raise ValueError("objective scales must be finite and positive")
        normalized.append(value / (denominator_scale + value))
    return tuple(normalized)  # type: ignore[return-value]


def hypervolume_3d(points: Iterable[Sequence[float]], *, reference: Sequence[float] = REFERENCE_POINT) -> float:
    """Compute the exact union of dominated boxes by an x sweep."""
    bounds = tuple(float(value) for value in reference)
    if len(bounds) != 3 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("the hypervolume reference point must be finite and three-dimensional")
    unique = set()
    for raw in points:
        point = tuple(float(value) for value in raw)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("hypervolume points must be finite and three-dimensional")
        if any(value > bound + _relative_slack(value, bound, RELATIVE_TOLERANCE) for value, bound in zip(point, bounds)):
            raise ValueError("a hypervolume point lies beyond the reference point")
        unique.add(tuple(min(value, bound) for value, bound in zip(point, bounds)))
    ordered = sorted(unique)
    xs = sorted({point[0] for point in ordered} | {bounds[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = sorted((y, z) for x, y, z in ordered if x <= left)
        area = 0.0
        minimum_z = bounds[2]
        for index, (y, z) in enumerate(active):
            minimum_z = min(minimum_z, z)
            next_y = active[index + 1][0] if index + 1 < len(active) else bounds[1]
            area += (next_y - y) * (bounds[2] - minimum_z)
        volume += (right - left) * area
    return volume
