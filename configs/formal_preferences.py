"""Ordered preference sets used by formal validation and final evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from environment import PreferenceContext, simplex_lattice


def formal_preferences(
    config: Mapping[str, Any], stage: str
) -> tuple[PreferenceContext, ...]:
    if stage not in {"validation", "final_test"}:
        raise ValueError(f"unknown formal evaluation stage: {stage}")
    quality = config["preference"]["quality"]
    if quality["mode"] != "universal_sobol_v1":
        return (PreferenceContext.from_input(quality["fixed"]),)
    formal = config["training"]["formal_evaluation"]
    if stage == "validation" and "validation_preferences" in formal:
        raw = formal["validation_preferences"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("validation_preferences must be a nonempty list")
        points = tuple(PreferenceContext.from_input(point) for point in raw)
    else:
        denominator = int(formal.get("final_test_lattice_denominator", 10))
        if denominator < 1:
            raise ValueError("final_test_lattice_denominator must be positive")
        points = tuple(
            PreferenceContext.from_input(point)
            for point in simplex_lattice(denominator, include=())
        )
    keys = [point.key for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{stage} preferences contain duplicate points")
    return points
