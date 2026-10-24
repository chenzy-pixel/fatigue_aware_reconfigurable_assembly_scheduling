"""Resource scores used by the native decoder."""
import math
from environment.resource_projection import ResourceProjector
from environment.workload import workload_value

def route_key(mode, finish, costs, peak, fallback=()):
    cost = sum(costs[:4])
    if mode == "minimum_reconfiguration_cost":
        return (cost, finish, peak, *fallback)
    if mode == "minimum_workload_delta":
        return (costs[4], finish, cost, *fallback)
    if mode == "lowest_predicted_fatigue":
        return (peak, finish, cost, *fallback)
    return (finish, cost, peak, *fallback)


def production_score(env, oi, mi, mode):
    projector = ResourceProjector(env)
    path = projector.candidate(oi, mi)
    costs = projector.path_costs(mi, path.path)
    finish = path.finish_tick if path.finish_tick is not None else math.inf
    peak = max((s.end_fatigue for s in path.path.stages), default=0) if path.path else 0
    return route_key(mode, finish, costs, peak)


def worker_score(env, mi, wi, mode):
    rec = env._pending_reconfiguration(env.machines[mi].spec.id)
    duration = env._stage_duration_ticks(rec, env.workers[wi]) * env.resolution
    finish = env.current_time + duration
    peak = env.projected_worker_fatigue(mi, wi)
    cost = duration * (env.workers[wi].spec.labor_cost_per_minute + env.machines[mi].spec.downtime_cost_per_minute)
    delta = env.projected_worker_workload_objective(mi, wi) - workload_value(env._committed_worker_loads, env.config)
    return route_key(mode, finish, (0, 0, cost, 0, delta), peak)
