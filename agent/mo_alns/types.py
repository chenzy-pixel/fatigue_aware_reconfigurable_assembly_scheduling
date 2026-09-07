"""Data structures shared by the multi-objective ALNS baseline.

The solver deliberately keeps an encoded solution separate from an evaluated
schedule.  The encoding is only a set of preferences; the environment remains
the authority that decides whether a dynamic action is safe and feasible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from data.models import AssemblyInstance
from environment import PreferenceVector


OBJECTIVE_FIELDS = (
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
)
OBJECTIVE_SCALES = (1200.0, 1000.0, 50.0)
ENCODING_SCHEMA_VERSION = "mo_alns_encoding_v1"


def preference_key(preference: PreferenceVector) -> str:
    """Return the repository's stable, human-readable preference key."""

    return "_".join(f"{value:.12g}" for value in preference.as_tuple())


@dataclass
class MOALNSSolution:
    """Priority and resource-preference encoding used by the ALNS search.

    Rankings are complete permutations.  This makes a candidate robust to a
    resource becoming temporarily unavailable: the decoder can select the next
    legal choice without inventing a schedule outside ``AssemblySchedulingEnv``.
    """

    operation_order: tuple[str, ...]
    machine_rankings: dict[str, tuple[str, ...]]
    disassembly_worker_rankings: dict[str, tuple[str, ...]]
    installation_worker_rankings: dict[str, tuple[str, ...]]
    production_wait: dict[str, bool]
    disassembly_wait: dict[str, bool]
    installation_wait: dict[str, bool]
    origin: str = "generated"

    def clone(self, *, origin: str | None = None) -> "MOALNSSolution":
        return MOALNSSolution(
            operation_order=tuple(self.operation_order),
            machine_rankings={key: tuple(value) for key, value in self.machine_rankings.items()},
            disassembly_worker_rankings={
                key: tuple(value)
                for key, value in self.disassembly_worker_rankings.items()
            },
            installation_worker_rankings={
                key: tuple(value)
                for key, value in self.installation_worker_rankings.items()
            },
            production_wait=dict(self.production_wait),
            disassembly_wait=dict(self.disassembly_wait),
            installation_wait=dict(self.installation_wait),
            origin=self.origin if origin is None else origin,
        )

    def validate(self, instance: AssemblyInstance, *, allow_partial: bool = False) -> None:
        operation_ids = [operation.id for operation in instance.operations]
        known_operations = set(operation_ids)
        encoded = list(self.operation_order)
        if len(encoded) != len(set(encoded)):
            raise ValueError("operation_order contains duplicate operation ids")
        if not set(encoded).issubset(known_operations):
            raise ValueError("operation_order contains an unknown operation id")
        if not allow_partial and set(encoded) != known_operations:
            raise ValueError("operation_order must contain every operation exactly once")

        positions = {operation_id: index for index, operation_id in enumerate(encoded)}
        for order in instance.orders:
            for first, second in zip(order.operations, order.operations[1:], strict=False):
                if first.id in positions and second.id in positions and positions[first.id] > positions[second.id]:
                    raise ValueError("operation_order violates within-order precedence")

        machine_ids = {machine.id for machine in instance.machines}
        worker_ids = {worker.id for worker in instance.workers}
        for operation_id in operation_ids:
            if operation_id not in self.machine_rankings:
                raise ValueError(f"missing machine ranking for {operation_id}")
            if set(self.machine_rankings[operation_id]) != machine_ids:
                raise ValueError(f"machine ranking for {operation_id} is not complete")
            for rankings, label in (
                (self.disassembly_worker_rankings, "disassembly"),
                (self.installation_worker_rankings, "installation"),
            ):
                if operation_id not in rankings:
                    raise ValueError(f"missing {label} worker ranking for {operation_id}")
                if set(rankings[operation_id]) != worker_ids:
                    raise ValueError(
                        f"{label} worker ranking for {operation_id} is not complete"
                    )
            for waits, label in (
                (self.production_wait, "production"),
                (self.disassembly_wait, "disassembly"),
                (self.installation_wait, "installation"),
            ):
                if operation_id not in waits or not isinstance(waits[operation_id], bool):
                    raise ValueError(f"missing boolean {label} wait gene for {operation_id}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": ENCODING_SCHEMA_VERSION,
            "operation_order": list(self.operation_order),
            "machine_rankings": {
                key: list(self.machine_rankings[key]) for key in sorted(self.machine_rankings)
            },
            "disassembly_worker_rankings": {
                key: list(self.disassembly_worker_rankings[key])
                for key in sorted(self.disassembly_worker_rankings)
            },
            "installation_worker_rankings": {
                key: list(self.installation_worker_rankings[key])
                for key in sorted(self.installation_worker_rankings)
            },
            "production_wait": {
                key: bool(self.production_wait[key]) for key in sorted(self.production_wait)
            },
            "disassembly_wait": {
                key: bool(self.disassembly_wait[key]) for key in sorted(self.disassembly_wait)
            },
            "installation_wait": {
                key: bool(self.installation_wait[key])
                for key in sorted(self.installation_wait)
            },
            "origin": self.origin,
        }

    def digest(self) -> str:
        raw = json.dumps(
            self.payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class CandidateEvaluation:
    """One full environment rollout of a ``MOALNSSolution``."""

    solution: MOALNSSolution
    preference: PreferenceVector
    metrics: dict[str, Any]
    objectives: tuple[float, float, float]
    normalized_objectives: tuple[float, float, float]
    tchebycheff: float
    feasible: bool
    action_trace_sha256: str
    realized: dict[str, Any]
    diagnostics: dict[str, Any]
    schedule_log: tuple[dict[str, Any], ...] = ()
    reconfiguration_log: tuple[dict[str, Any], ...] = ()

    @property
    def solution_digest(self) -> str:
        return self.solution.digest()

    @property
    def completion_rank(self) -> tuple[float, float, float, float]:
        """A total order used when the search temporarily has infeasible states."""

        violations = float(len(self.metrics.get("schedule_violations", ())))
        fatigue_excess = max(
            0.0,
            float(self.metrics.get("maximum_worker_fatigue", math.inf))
            - float(self.metrics.get("safe_fatigue_limit", -math.inf)),
        )
        return (
            float(self.metrics.get("unfinished_orders", math.inf)),
            violations + fatigue_excess,
            float(self.objectives[0]),
            float(self.tchebycheff),
        )

    def without_logs(self) -> "CandidateEvaluation":
        return CandidateEvaluation(
            solution=self.solution,
            preference=self.preference,
            metrics=dict(self.metrics),
            objectives=self.objectives,
            normalized_objectives=self.normalized_objectives,
            tchebycheff=self.tchebycheff,
            feasible=self.feasible,
            action_trace_sha256=self.action_trace_sha256,
            realized={key: value for key, value in self.realized.items()},
            diagnostics={key: value for key, value in self.diagnostics.items()},
        )


@dataclass
class SearchResult:
    preference: PreferenceVector
    selected: CandidateEvaluation
    archive: tuple[CandidateEvaluation, ...]
    initial_best_tchebycheff: float | None
    environment_evaluations: int
    cache_hits: int
    proposal_count: int
    search_time_seconds: float
    operator_statistics: dict[str, Any]
    search_log: tuple[dict[str, Any], ...]


@dataclass
class GridSearchResult:
    preferences: tuple[PreferenceVector, ...]
    endpoints: dict[str, CandidateEvaluation]
    archive: tuple[CandidateEvaluation, ...]
    searches: tuple[SearchResult, ...]


def metrics_objectives(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    values = tuple(float(metrics[field]) for field in OBJECTIVE_FIELDS)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("candidate objectives must be finite non-negative values")
    return values  # type: ignore[return-value]
