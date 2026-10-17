"""One accounting boundary shared by construction, proposals, and decoding."""
from dataclasses import dataclass, field
import math
import time
from typing import Callable


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class SearchBudget:
    max_evaluations: int | None = None
    seconds: float | None = None
    clock: Callable[[], float] = time.perf_counter
    started: float = field(init=False)
    started_evaluations: int = 0
    completed_evaluations: int = 0
    aborted_evaluations: int = 0
    environment_steps: int = 0
    proxy_calls: int = 0

    def __post_init__(self):
        if self.max_evaluations is None and self.seconds is None:
            raise ValueError("at least one search budget is required")
        if self.max_evaluations is not None and (type(self.max_evaluations) is not int or self.max_evaluations < 1):
            raise ValueError("max_evaluations must be a positive integer or null")
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("seconds must be finite and positive")
        self.started = self.clock()

    @property
    def elapsed(self):
        return self.clock() - self.started

    def check_time(self):
        if self.seconds is not None and self.elapsed >= self.seconds:
            raise BudgetExhausted("time_limit")

    def begin_evaluation(self):
        self.check_time()
        if self.max_evaluations is not None and self.started_evaluations >= self.max_evaluations:
            raise BudgetExhausted("evaluation_limit")
        self.started_evaluations += 1

    def counters(self):
        return {name: getattr(self, name) for name in (
            "started_evaluations", "completed_evaluations", "aborted_evaluations",
            "environment_steps", "proxy_calls")}
