from __future__ import annotations

import random

import numpy as np

from environment import AssemblySchedulingEnv, DecisionType


class HeuristicPolicy:
    """Deterministic earliest-finish / lowest-fatigue dispatching policy."""

    def select_action(self, env: AssemblySchedulingEnv) -> int:
        mask = env.get_action_mask()
        feasible = np.flatnonzero(~mask)
        terminal_action = (
            env.production_defer_action
            if env.decision_type == DecisionType.PRODUCTION
            else env.worker_advance_action
        )
        pair_actions = [
            int(value) for value in feasible if value != terminal_action
        ]
        if not pair_actions:
            return terminal_action
        if env.decision_type == DecisionType.PRODUCTION:
            scored = []
            for action in pair_actions:
                operation_index, machine_index = env.decode_production_action(action)
                processing = env.estimate_processing_ticks(
                    operation_index, machine_index
                )
                reconfiguration = env.estimate_reconfiguration_ticks(
                    operation_index, machine_index
                )
                operation = env.operations[operation_index]
                same_configuration = (
                    env.machines[machine_index].current_module
                    == operation.spec.required_module
                )
                scored.append(
                    (
                        processing + reconfiguration,
                        0 if same_configuration else 1,
                        operation.spec.sequence,
                        operation_index,
                        machine_index,
                        action,
                    )
                )
            return min(scored)[-1]
        scored = []
        for action in pair_actions:
            machine_index, worker_index = env.decode_worker_action(action)
            scored.append(
                (
                    env.projected_worker_fatigue(machine_index, worker_index),
                    env.workers[worker_index].load,
                    worker_index,
                    machine_index,
                    action,
                )
            )
        return min(scored)[-1]


class RandomPolicy:
    """Uniform random policy over the environment's feasible action set."""

    def __init__(self, seed: int):
        self._random = random.Random(seed)

    def select_action(self, env: AssemblySchedulingEnv) -> int:
        feasible = np.flatnonzero(~env.get_action_mask()).tolist()
        return int(self._random.choice(feasible))
