"""Pareto and scalarisation helpers for MO-ALNS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from environment import PreferenceVector
from pareto_analysis import dominates, normalize_objectives, vectors_equal

from .types import CandidateEvaluation, OBJECTIVE_SCALES


def normalized_objectives(objectives: tuple[float, float, float]) -> tuple[float, float, float]:
    return normalize_objectives(objectives, OBJECTIVE_SCALES)


def augmented_tchebycheff(
    objectives: tuple[float, float, float],
    preference: PreferenceVector,
    *,
    epsilon: float = 1e-6,
    augmentation: float = 1e-4,
) -> float:
    """Scalarise a minimisation vector against the theoretical zero ideal point."""

    normalized = normalized_objectives(objectives)
    weights = tuple(max(float(weight), epsilon) for weight in preference.as_tuple())
    return float(
        max(weight * value for weight, value in zip(weights, normalized, strict=True))
        + augmentation
        * sum(weight * value for weight, value in zip(weights, normalized, strict=True))
    )


def candidate_better(first: CandidateEvaluation, second: CandidateEvaluation) -> bool:
    """Return whether first is preferable as the search's current solution."""

    if first.feasible != second.feasible:
        return first.feasible
    if first.feasible:
        return first.tchebycheff < second.tchebycheff - 1e-12
    return first.completion_rank < second.completion_rank


@dataclass
class ParetoArchive:
    """Feasible non-dominated candidates, deduplicated by decoded action trace."""

    entries: list[CandidateEvaluation] = field(default_factory=list)

    def update(self, candidate: CandidateEvaluation) -> bool:
        if not candidate.feasible:
            return False
        for index, existing in enumerate(self.entries):
            if existing.action_trace_sha256 == candidate.action_trace_sha256:
                if candidate.tchebycheff < existing.tchebycheff - 1e-12:
                    self.entries[index] = candidate
                    return True
                return False
        if any(dominates(existing.objectives, candidate.objectives) for existing in self.entries):
            return False
        retained = [
            existing
            for existing in self.entries
            if not dominates(candidate.objectives, existing.objectives)
        ]
        retained.append(candidate)
        self.entries = retained
        return True

    def extend(self, candidates: Iterable[CandidateEvaluation]) -> None:
        for candidate in candidates:
            self.update(candidate)

    def best(self, preference: PreferenceVector) -> CandidateEvaluation:
        if not self.entries:
            raise RuntimeError("cannot select a candidate from an empty Pareto archive")
        return min(
            self.entries,
            key=lambda candidate: (
                augmented_tchebycheff(candidate.objectives, preference),
                candidate.action_trace_sha256,
            ),
        )

    def snapshots(self) -> tuple[CandidateEvaluation, ...]:
        return tuple(
            sorted(
                self.entries,
                key=lambda candidate: (
                    candidate.objectives,
                    candidate.action_trace_sha256,
                ),
            )
        )
