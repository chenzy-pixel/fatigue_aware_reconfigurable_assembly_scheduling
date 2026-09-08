"""Stable pair-plus-defer action encoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCodec:
    operation_count: int
    machine_count: int
    worker_count: int

    @property
    def production_size(self) -> int:
        return self.operation_count * self.machine_count + 1

    @property
    def worker_size(self) -> int:
        return self.machine_count * self.worker_count + 1

    @property
    def production_defer(self) -> int:
        return self.production_size - 1

    @property
    def worker_advance(self) -> int:
        return self.worker_size - 1

    def encode_production(self, operation_index: int, machine_index: int) -> int:
        if not 0 <= operation_index < self.operation_count:
            raise ValueError("operation index is out of range")
        if not 0 <= machine_index < self.machine_count:
            raise ValueError("machine index is out of range")
        return operation_index * self.machine_count + machine_index

    def decode_production(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.production_defer:
            raise ValueError("not a production pair action")
        return divmod(action, self.machine_count)

    def encode_worker(self, machine_index: int, worker_index: int) -> int:
        if not 0 <= machine_index < self.machine_count:
            raise ValueError("machine index is out of range")
        if not 0 <= worker_index < self.worker_count:
            raise ValueError("worker index is out of range")
        return machine_index * self.worker_count + worker_index

    def decode_worker(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.worker_advance:
            raise ValueError("not a worker pair action")
        return divmod(action, self.worker_count)
