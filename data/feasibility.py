"""Structural checks and processing bounds without a scheduling deadline."""
import math
from dataclasses import dataclass
from data.models import validate_instance, instance_time_scale
PRECHECK_VERSION = 'structural_batch_v1'

def maximum_matching_size(edges, worker_count):
    assigned = [-1] * worker_count

    def augment(task, seen):
        for worker in edges[task]:
            if worker in seen:
                continue
            seen.add(worker)
            if assigned[worker] < 0 or augment(assigned[worker], seen):
                assigned[worker] = task
                return True
        return False
    return sum((augment(t, set()) for t in range(len(edges))))

@dataclass(frozen=True)
class FeasibilityPrecheckReport:
    passed: bool
    reason_codes: tuple
    metrics: dict
    version: str = PRECHECK_VERSION

    def to_dict(self):
        return vars(self)

def analyze_static_feasibility(instance):
    validate_instance(instance)
    return {'time_scale': instance_time_scale(instance), 'operations': len(instance.operations)}

def cheap_feasibility_precheck(instance, static_analysis=None):
    validate_instance(instance)
    reasons = []
    f = instance.fatigue
    r = instance.resolution
    for m in instance.machines:
        for a, p in m.module_parameters.items():
            for phase, base, rate in [('DIS', p.disassembly_base_time, f.disassembly_accumulation_rate_per_minute), ('INS', p.installation_base_time, f.installation_accumulation_rate_per_minute)]:
                duration = max(1, math.ceil((base - 1e-09) / r)) * r
                if rate * duration > f.maximum_safe_fatigue + 1e-09:
                    reasons.append(f'zero_fatigue_unsafe:{m.id}:{a}:{phase}')
    return FeasibilityPrecheckReport(not reasons, tuple(reasons), analyze_static_feasibility(instance))
