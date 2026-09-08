"""Mutable runtime state records for the scheduling kernel."""

from __future__ import annotations

from dataclasses import dataclass

from data.models import MachineSpec, OperationSpec, WorkerSpec

from .types import MachineState, OperationState, ReconfigurationStage, WorkerState


@dataclass
class OperationRuntime:
    spec: OperationSpec
    state: OperationState
    machine_id: str | None = None
    start_tick: int | None = None
    end_tick: int | None = None


@dataclass
class MachineRuntime:
    spec: MachineSpec
    state: MachineState
    current_module: str
    busy_until_tick: int | None = None
    locked_operation_id: str | None = None
    source_module: str | None = None
    target_module: str | None = None


@dataclass
class WorkerRuntime:
    spec: WorkerSpec
    state: WorkerState
    fatigue: float
    peak_fatigue: float = 0.0
    load: float = 0.0
    busy_until_tick: int | None = None


@dataclass
class ReconfigurationRuntime:
    id: str
    machine_id: str
    operation_id: str
    source_module: str
    target_module: str
    lock_tick: int
    stage: ReconfigurationStage = ReconfigurationStage.WAIT_DIS
    disassembly_worker_id: str | None = None
    installation_worker_id: str | None = None
    disassembly_start_tick: int | None = None
    disassembly_end_tick: int | None = None
    installation_start_tick: int | None = None
    installation_end_tick: int | None = None


@dataclass(frozen=True)
class WorkerTaskSnapshot:
    """Lightweight pending worker task used by observations and diagnostics."""

    task_id: str
    machine_index: int
    stage: ReconfigurationStage
    module: str
