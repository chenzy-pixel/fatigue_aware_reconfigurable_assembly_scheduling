"""Time-grid primitives shared by processing, reconfiguration, and fatigue logic."""

from __future__ import annotations

import math


EPSILON = 1e-9


def quantize_to_ticks(minutes: float, resolution: float) -> int:
    if minutes < 0 or not math.isfinite(minutes):
        raise ValueError("duration must be finite and non-negative")
    return int(math.ceil((minutes - EPSILON) / resolution))


def ticks_to_minutes(ticks: int, resolution: float) -> float:
    return float(ticks) * resolution
