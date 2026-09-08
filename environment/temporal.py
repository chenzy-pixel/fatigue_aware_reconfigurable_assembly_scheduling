"""Data contracts for deterministic temporal feasibility search v3."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ReconfigurationStage


@dataclass(frozen=True)
class WorkerTaskSnapshot:
    task_id: str
    machine_index: int
    stage: ReconfigurationStage
    module: str


@dataclass(frozen=True)
class TemporalWorkerTask:
    task_id: str
    machine_index: int
    stage: ReconfigurationStage
    module: str
    ready_tick: int
    predecessor_id: str | None = None
    candidate: bool = False


@dataclass(frozen=True)
class TemporalWorkerState:
    available_tick: int
    fatigue: float


@dataclass(frozen=True)
class TemporalFeasibilityResult:
    status: str
    searched_nodes: int
    candidate_completion_tick: int | None = None
    termination_reason: str | None = None
    option_evaluations: int = 0
    frontier_options_before: int = 0
    frontier_options_after: int = 0


@dataclass
class TemporalSearchBudget:
    searched_nodes: int = 0
    option_evaluations: int = 0
    frontier_options_before: int = 0
    frontier_options_after: int = 0
    dominated_options: int = 0
    last_options_before: int = 0
    last_options_after: int = 0


class TemporalBudgetExhausted(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
