"""Pure approximate insertion schedules and real-state resource ranking.

The serial predictor orders proposals only. Physical feasibility and objectives
always come from the event-driven environment decoder.
"""
from dataclasses import dataclass
import math

from environment import AssemblySchedulingEnv
from environment.dynamics import quantize_to_ticks
from environment.resource_projection import ProjectedResources, ReconfigurationProjection, ResourceProjector
from environment.types import FLOW_EXCESS, flow_mode
from environment.workload import workload_value


MODES = ("earliest_finish", "minimum_reconfiguration_cost", "minimum_workload_delta")


def route_key(mode, finish, costs, peak, fallback=()):
    cost = sum(costs[:4])
    if mode == "minimum_reconfiguration_cost":
        return (cost, finish, peak, *fallback)
    if mode == "minimum_workload_delta":
        return (costs[4], finish, cost, *fallback)
    if mode == "lowest_predicted_fatigue":
        return (peak, finish, cost, *fallback)
    return (finish, cost, peak, *fallback)


@dataclass
class PredictedSchedule:
    objectives: tuple[float, float, float]
    feasible: bool
    assignments: dict


class SerialPredictor:
    def __init__(self, config, instance, budget):
        self.config, self.instance, self.budget = config, instance, budget
        self.env = AssemblySchedulingEnv(dict(config))
        self.env.reset(instance, build_observation=False)
        self.projector = ResourceProjector(self.env)
        self.operations = {op.id: (i, op) for i, op in enumerate(instance.operations)}
        self.orders = {order.id: order for order in instance.orders}
        self.cache = {}

    def routes(self, mi, source, target, tick, resources):
        if source == target:
            return [ReconfigurationProjection(tick, tick, (), resources.copy(), resources.loads.copy())]
        prefixes = [((), resources)]
        if source != self.instance.no_module_state:
            prefixes = [((stage,), state) for stage, state in
                        self.projector.stage_options(mi, source, False, tick, resources)]
        paths = []
        for prefix, state in prefixes:
            self.budget.check_time()
            start = prefix[-1].end_tick if prefix else tick
            for stage, after in self.projector.stage_options(mi, target, True, start, state):
                paths.append(ReconfigurationProjection(tick, stage.end_tick, prefix + (stage,),
                                                       after, resources.loads.copy()))
        return paths

    def predict(self, solution):
        self.budget.check_time()
        key = solution.digest()
        if key in self.cache:
            return self.cache[key]
        self.budget.proxy_calls += 1
        ins, env = self.instance, self.env
        resources = self.projector.initial_resources()
        machine_times = [0] * len(ins.machines)
        modules = [m.initial_module for m in ins.machines]
        order_times = {o.id: quantize_to_ticks(o.release_time, ins.resolution) for o in ins.orders}
        cost, assignments = 0.0, {}
        for opid in solution.operation_order:
            self.budget.check_time()
            oi, op = self.operations[opid]
            options = []
            mode = solution.resource_modes.get(opid, "ranking")
            for rank, mid in enumerate(solution.machine_rankings[opid]):
                mi = ins.machine_index[mid]
                if op.required_module not in ins.machines[mi].module_parameters:
                    continue
                tick = max(machine_times[mi], order_times[op.order_id])
                for path in self.routes(mi, modules[mi], op.required_module, tick, resources):
                    finish = path.end_tick + env.estimate_processing_ticks(oi, mi)
                    costs = self.projector.path_costs(mi, path)
                    peak = max((s.end_fatigue for s in path.stages), default=0)
                    fallback = (rank, tuple(s.worker_index for s in path.stages))
                    key = ((rank, finish, *fallback) if mode == "ranking"
                           else route_key(mode, finish, costs, peak, fallback))
                    options.append((key, mi, path, finish, costs))
            if not options:
                result = PredictedSchedule((1e12, cost, workload_value(resources.loads, self.config)), False, assignments)
                break
            _, mi, path, finish, costs = min(options, key=lambda x: x[0])
            machine_times[mi] = finish
            modules[mi] = op.required_module
            resources = path.resources
            order_times[op.order_id] = finish
            cost += sum(costs[:4])
            assignments[opid] = {"machine": ins.machines[mi].id,
                                 "dis": next((ins.workers[s.worker_index].id for s in path.stages if not s.installation), None),
                                 "ins": next((ins.workers[s.worker_index].id for s in path.stages if s.installation), None)}
        else:
            flow = sum(max(0, order_times[o.id] * ins.resolution - o.release_time) for o in ins.orders)
            if flow_mode(self.config) == FLOW_EXCESS:
                flow = max(0, flow - env._total_processing_lower_bound_ticks * ins.resolution)
            result = PredictedSchedule((flow, cost, workload_value(resources.loads, self.config)),
                                       all(t <= env.horizon_tick for t in order_times.values()), assignments)
        if len(self.cache) >= 256:
            self.cache.pop(next(iter(self.cache)))
        self.cache[solution.digest()] = result
        return result


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
