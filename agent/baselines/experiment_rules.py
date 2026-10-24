"""Frozen two-phase dispatching rules for the factorial-27 experiment."""
import math
import time
import numpy as np

from agent.baselines import HeuristicPolicy
from agent.comparators.common import candidates, feature_column, digest
from environment import DecisionType, OPERATION_ORDER_EDGE
from environment.types import metrics_flow_objective
from environment.workload import metrics_workload_objective

RULES = ('eft_lf', 'srw_lf', 'mc_mc', 'mwl_mwl', 'existing_heuristic')


class FixedRulePolicy:
    def __init__(self, name):
        if name not in RULES:
            raise ValueError(f'unknown fixed rule: {name}')
        self.name = name

    def select_action(self, env, observation=None):
        if self.name == 'existing_heuristic':
            return HeuristicPolicy().select_action(env)
        mask = env.get_action_mask()
        legal = np.flatnonzero(~mask)
        if not len(legal):
            raise ValueError('fixed rule has no legal action')
        pairs = legal[legal != env.wait_action]
        if not len(pairs):
            if mask[env.wait_action]:
                raise ValueError('WAIT is masked with no legal pair')
            return env.wait_action
        obs = observation if observation is not None else env.observe()
        edge, _, _, ids = candidates(obs)
        col = lambda name: feature_column(edge, name)
        indices = [i for i, action in enumerate(ids) if not mask[action]]
        if len(indices) != len(pairs):
            raise ValueError('legal pairs and sparse candidates disagree')
        production = obs.decision_type == DecisionType.PRODUCTION
        if production:
            valid = col('projection_valid') > .5
            finish = col('predicted_finish_time_norm')
            if self.name == 'eft_lf':
                scores = (finish,)
            elif self.name == 'srw_lf':
                remain = np.zeros(len(env.operations))
                links = obs.relations[OPERATION_ORDER_EDGE].edge_index
                remain[links[0]] = obs.node_features['order'][links[1], obs.node_feature_names['order'].index('remaining_workload_norm')]
                scores = (remain[edge.edge_index[0]], finish)
            elif self.name == 'mc_mc':
                scores = (col('fixed_disassembly_cost_norm') + col('fixed_installation_cost_norm') + col('estimated_labor_cost_norm') + col('estimated_downtime_cost_norm'),)
            else:
                scores = (col('estimated_worker_load_objective_delta_norm'),)
            stable = lambda i: (env.operations[edge.edge_index[0, i]].spec.id, env.machines[edge.edge_index[1, i]].spec.id)
        else:
            valid = np.ones(len(ids), dtype=bool)
            if self.name in ('eft_lf', 'srw_lf'):
                scores = (col('projected_fatigue_ratio'),)
            elif self.name == 'mc_mc':
                scores = (col('fixed_cost_norm') + col('incremental_labor_cost_norm') + col('incremental_downtime_cost_norm'),)
            else:
                scores = (col('incremental_workload_objective_norm'),)
            stable = lambda i: (env.machines[edge.edge_index[0, i]].spec.id, env.workers[edge.edge_index[1, i]].spec.id)
        good = [i for i in indices if valid[i] and all(math.isfinite(float(s[i])) for s in scores)]
        winner = min(good, key=lambda i: (*[float(s[i]) for s in scores], *stable(i))) if good else min(indices, key=stable)
        return int(ids[winner])


def evaluate_rule(config, instance, name):
    from environment import AssemblySchedulingEnv
    began = time.perf_counter()
    env = AssemblySchedulingEnv(config)
    observation = env.reset(instance)
    policy, actions = FixedRulePolicy(name), []
    while not (env.terminated or env.truncated):
        action = policy.select_action(env, observation)
        actions.append(action)
        observation, _, _, _, _ = env.step(action)
    metrics, violations = env.metrics(), env.validate_schedule()
    return {'success': bool(metrics['task_succeeded'] and not violations),
            'objectives': (metrics_flow_objective(metrics, config), metrics['reconfiguration_cost'], metrics_workload_objective(metrics, config)),
            'actions': actions, 'trace': digest(actions), 'metrics': metrics, 'violations': violations,
            'seconds': time.perf_counter()-began, 'sampling_repeat': 0,
            'schedule': env.schedule_log, 'reconfigurations': env.reconfiguration_log}
