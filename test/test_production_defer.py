from __future__ import annotations

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from environment import AssemblySchedulingEnv, DecisionType


def _step_requested_pair(environment, *, reconfiguration: bool) -> dict:
    policy = HeuristicPolicy()
    for _ in range(500):
        if environment.decision_type == DecisionType.PRODUCTION:
            for action in np.flatnonzero(~environment.get_action_mask()):
                if int(action) == environment.production_defer_action:
                    continue
                operation_index, machine_index = environment.decode_production_action(
                    int(action)
                )
                operation = environment.operations[operation_index]
                machine = environment.machines[machine_index]
                mismatch = machine.current_module != operation.spec.required_module
                if mismatch == reconfiguration:
                    return environment.step(int(action))[-1]
        environment.step(policy.select_action(environment))
        if environment.terminated or environment.truncated:
            break
    raise AssertionError("trajectory lacks the requested pair-action kind")


@pytest.mark.parametrize(
    ("reconfiguration", "expected"),
    [(False, "DIRECT_PROCESS"), (True, "COMMIT_RECONFIG")],
)
def test_pair_action_has_stable_direct_or_reconfiguration_semantics(
    config, fixed_instance, reconfiguration, expected
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    info = _step_requested_pair(environment, reconfiguration=reconfiguration)
    assert info["action_type"] == expected


def test_latest_production_action_space_is_pairs_plus_one_defer(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    pair_count = len(environment.operations) * len(environment.machines)
    assert environment.production_defer_action == pair_count
    assert len(environment.get_action_mask()) == pair_count + 1


def test_initial_defer_waits_without_reconfiguration_cost(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    assert not environment.get_action_mask()[environment.production_defer_action]
    before_tick = environment.current_tick
    _, reward, _, _, info = environment.step(environment.production_defer_action)
    assert environment.current_tick > before_tick
    assert environment.metrics()["reconfiguration_cost"] == pytest.approx(0.0)
    assert reward.cost == pytest.approx(0.0)
    assert info["action_type"] == "DEFER_PRODUCTION"
