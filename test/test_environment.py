from __future__ import annotations

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy, RandomPolicy
from environment import AssemblySchedulingEnv, DecisionType


def _run(environment, policy):
    rewards = []
    actions = []
    while not (environment.terminated or environment.truncated):
        action = policy.select_action(environment)
        actions.append(action)
        _, reward, _, _, _ = environment.step(action)
        rewards.append(reward)
    return actions, rewards


def test_initial_pair_wait_action_contract(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(fixed_instance)
    mask = environment.get_action_mask()
    assert observation.decision_type == DecisionType.PRODUCTION
    assert len(mask) == len(environment.operations) * len(environment.machines) + 1
    assert environment.wait_action == len(mask) - 1
    assert np.count_nonzero(~mask) > 0


def test_heuristic_schedule_is_safe_and_reward_components_telescope(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance)
    _, rewards = _run(environment, HeuristicPolicy())
    metrics = environment.metrics()
    assert environment.terminated and not environment.truncated
    assert environment.validate_schedule() == []
    assert sum(reward.flow for reward in rewards) == pytest.approx(
        -metrics["flow_time_objective"]
    )
    assert sum(reward.cost for reward in rewards) == pytest.approx(
        -metrics["reconfiguration_cost"]
    )
    assert sum(reward.variance for reward in rewards) == pytest.approx(
        -metrics["worker_load_variance"]
    )


def test_random_policy_trajectory_is_reproducible(config, fixed_instance):
    traces = []
    for _ in range(2):
        environment = AssemblySchedulingEnv(config)
        environment.reset(fixed_instance)
        actions, _ = _run(environment, RandomPolicy(17))
        traces.append((actions, environment.metrics()["terminal_reason"]))
    assert traces[0] == traces[1]


def test_observe_and_public_metrics_are_available(config, fixed_instance):
    environment = AssemblySchedulingEnv(config)
    initial = environment.reset(fixed_instance)
    observed = environment.observe()
    assert observed.feature_dimensions == initial.feature_dimensions
    assert config["runtime_manifest"]["observation_schema"] == 5
