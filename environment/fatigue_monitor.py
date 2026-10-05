"""Original-instance fatigue exposure on a realized worker timeline."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from data.models import AssemblyInstance


def _segment(
    start: float,
    end: float,
    fatigue: float,
    rate: float,
    limit: float,
    worker_id: str,
    stage: str,
) -> tuple[float, dict[str, Any]]:
    duration = end - start
    cuts = [0.0, duration]
    if rate:
        for level in (0.0, limit, 1.0):
            crossing = (level - fatigue) / rate
            if 0.0 < crossing < duration:
                cuts.append(crossing)
    cuts = sorted(set(cuts))
    area = exposure = 0.0
    for left, right in zip(cuts, cuts[1:]):
        first = min(1.0, max(0.0, fatigue + rate * left))
        last = min(1.0, max(0.0, fatigue + rate * right))
        over_first = max(0.0, first - limit)
        over_last = max(0.0, last - limit)
        area += (over_first + over_last) * (right - left) / 2.0
        if over_first > 0.0 or over_last > 0.0:
            exposure += right - left
    final = min(1.0, max(0.0, fatigue + rate * duration))
    return final, {
        "worker_id": worker_id,
        "stage": stage,
        "start": start,
        "end": end,
        "fatigue_start": fatigue,
        "fatigue_end": final,
        "over_limit_minutes": exposure,
        "over_limit_area": area,
    }


def audit_fatigue(
    instance: AssemblyInstance,
    records: Sequence[Mapping[str, Any]],
    end_time: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Integrate physical fatigue on actual intervals without altering dynamics."""
    end_time = float(end_time)
    if not math.isfinite(end_time) or end_time < 0.0:
        raise ValueError("fatigue audit end_time must be finite and non-negative")
    limit = instance.fatigue.maximum_safe_fatigue
    worker_ids = {worker.id for worker in instance.workers}
    by_worker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        worker_id = str(record["worker_id"])
        if worker_id not in worker_ids:
            raise ValueError(f"unknown worker in fatigue audit: {worker_id}")
        if record["stage"] not in {"DIS", "INS"}:
            raise ValueError(f"unknown stage in fatigue audit: {record['stage']}")
        start, end = float(record["start"]), float(record["end"])
        if not all(math.isfinite(value) for value in (start, end)) or not 0.0 <= start <= end:
            raise ValueError("fatigue audit intervals must be finite and ordered")
        by_worker[worker_id].append(record)

    segments: list[dict[str, Any]] = []
    total_area = total_minutes = busy_minutes = 0.0
    peak = 0.0
    affected = 0
    idle_gaps: list[float] = []
    max_consecutive = 0
    busy_by_worker: dict[str, float] = {}
    for worker in instance.workers:
        fatigue = worker.initial_fatigue
        worker_peak = fatigue
        worker_area = worker_busy = 0.0
        time = 0.0
        consecutive = 0
        previous_end: float | None = None
        intervals = sorted(
            by_worker[worker.id],
            key=lambda item: (float(item["start"]), float(item["end"])),
        )
        for record in intervals:
            start = min(end_time, float(record["start"]))
            end = min(end_time, float(record["end"]))
            if end <= start:
                continue
            if start < time - 1e-9:
                raise ValueError(f"overlapping worker audit intervals: {worker.id}")
            start = max(start, time)
            if start > time:
                fatigue, row = _segment(
                    time, start, fatigue,
                    -instance.fatigue.idle_recovery_rate_per_minute,
                    limit, worker.id, "IDLE",
                )
                segments.append(row)
                worker_area += row["over_limit_area"]
                total_minutes += row["over_limit_minutes"]
            if previous_end is not None:
                idle_gaps.append(start - previous_end)
            consecutive = (
                consecutive + 1
                if previous_end is not None and abs(start - previous_end) < 1e-9
                else 1
            )
            max_consecutive = max(max_consecutive, consecutive)
            rate = (
                instance.fatigue.disassembly_accumulation_rate_per_minute
                if record["stage"] == "DIS"
                else instance.fatigue.installation_accumulation_rate_per_minute
            )
            fatigue, row = _segment(
                start, end, fatigue, rate, limit, worker.id, str(record["stage"])
            )
            segments.append(row)
            worker_peak = max(worker_peak, fatigue)
            worker_area += row["over_limit_area"]
            total_minutes += row["over_limit_minutes"]
            worker_busy += end - start
            time = end
            previous_end = end
        if time < end_time:
            fatigue, row = _segment(
                time, end_time, fatigue,
                -instance.fatigue.idle_recovery_rate_per_minute,
                limit, worker.id, "IDLE",
            )
            segments.append(row)
            worker_area += row["over_limit_area"]
            total_minutes += row["over_limit_minutes"]
        total_area += worker_area
        busy_minutes += worker_busy
        busy_by_worker[worker.id] = worker_busy
        peak = max(peak, worker_peak)
        affected += int(worker_peak > limit + 1e-9)

    denominator = len(instance.workers) * end_time
    metrics = {
        "fatigue_monitor_peak": peak,
        "fatigue_monitor_over_limit_worker_ratio": affected / max(1, len(instance.workers)),
        "fatigue_monitor_over_limit_minutes": total_minutes,
        "fatigue_monitor_over_limit_area": total_area,
        "fatigue_monitor_over_limit_time_ratio": total_minutes / denominator if denominator else None,
        "fatigue_monitor_over_limit_area_ratio": total_area / denominator if denominator else None,
        "worker_reconfiguration_busy_minutes": busy_minutes,
        "worker_reconfiguration_idle_minutes": denominator - busy_minutes,
        "worker_reconfiguration_duty_ratio": busy_minutes / denominator if denominator else None,
        "worker_reconfiguration_busy_minutes_by_worker": busy_by_worker,
        "worker_reconfiguration_idle_minutes_by_worker": {
            worker_id: end_time - busy for worker_id, busy in busy_by_worker.items()
        },
        "worker_reconfiguration_duty_ratio_by_worker": {
            worker_id: busy / end_time if end_time else None
            for worker_id, busy in busy_by_worker.items()
        },
        "mean_interstage_idle_minutes": sum(idle_gaps) / len(idle_gaps) if idle_gaps else None,
        "max_consecutive_worker_stages": max_consecutive,
    }
    return metrics, segments
