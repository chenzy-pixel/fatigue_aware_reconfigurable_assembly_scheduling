from __future__ import annotations

import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agent.ppo.parallel import forced_action_from_mask
from environment import DecisionType, PolicyObservation, RewardVector
from train import _collect_serial_batch


def _observation(step: int) -> PolicyObservation:
    return PolicyObservation(
        operations=np.asarray([[float(step)]], dtype=np.float32),
        machines=np.zeros((1, 1), dtype=np.float32),
        workers=np.zeros((1, 1), dtype=np.float32),
        global_features=np.asarray([float(step)], dtype=np.float32),
        decision_type=DecisionType.PRODUCTION,
    )


class _ScriptedEnvironment:
    def __init__(self, masks: list[list[bool]], rewards: list[float]):
        self.masks = [np.asarray(mask, dtype=np.bool_) for mask in masks]
        self.rewards = list(rewards)
        self.index = 0
        self.terminated = False
        self.truncated = False

    def reset(self, instance):
        self.index = 0
        self.terminated = False
        self.truncated = False
        return _observation(0)

    def get_action_mask(self):
        return self.masks[self.index].copy()

    def step(self, action: int):
        mask = self.get_action_mask()
        if action < 0 or action >= len(mask) or mask[action]:
            raise ValueError(f"illegal scripted action {action}")
        reward = RewardVector(
            flow=self.rewards[self.index],
            cost=0.0,
            variance=0.0,
        )
        self.index += 1
        self.terminated = self.index == len(self.rewards)
        observation = None if self.terminated else _observation(self.index)
        return observation, reward, self.terminated, False, {}

    def metrics(self):
        return {
            "flow_time_objective": -sum(self.rewards[: self.index]),
            "reconfiguration_cost": 0.0,
            "worker_load_variance": 0.0,
            "completed_operations": 0,
            "time": float(self.index),
        }

    def validate_schedule(self):
        return []


class _CountingAgent:
    requires_graph_observation = False

    def __init__(self):
        self.act_calls = 0
        self.value_calls = 0

    def act(self, observation, action_mask):
        self.act_calls += 1
        legal = np.flatnonzero(~action_mask)
        return int(legal[0]), -0.25, 0.25

    def value(self, observation, action_mask):
        self.value_calls += 1
        return 0.5


def _config(config, *, compression: bool) -> dict:
    effective = deepcopy(config)
    effective["training"]["forced_action_compression"] = compression
    effective["ppo"]["gamma"] = 1.0
    effective["ppo"]["gae_lambda"] = 0.995
    effective["reward"].update(
        {
            "mode": "legacy_weighted_sum",
            "flow_weight": 1.0,
            "cost_weight": 0.0,
            "variance_weight": 0.0,
            "flow_scale": 1.0,
            "cost_scale": 1.0,
            "variance_scale": 1.0,
        }
    )
    return effective


def _collect(config, environment, agent, *, step_limit=None):
    return _collect_serial_batch(
        config=config,
        agent=agent,
        environment=environment,
        instance=SimpleNamespace(instance_id="scripted"),
        record=None,
        episode_index=0,
        sampling_start=time.perf_counter(),
        generation_time_seconds=0.0,
        step_limit=step_limit,
        reward_phase="legacy",
    )


def test_forced_action_detection_uses_single_legal_action():
    assert forced_action_from_mask(
        np.asarray([True, False, True], dtype=np.bool_)
    ) == 1
    assert forced_action_from_mask(
        np.asarray([False, False, True], dtype=np.bool_)
    ) is None
    with pytest.raises(ValueError, match="no legal actions"):
        forced_action_from_mask(np.ones(3, dtype=np.bool_))


def test_compression_merges_initial_and_terminal_forced_rewards(config):
    masks = [[False, True], [False, False], [True, False]]
    rewards = [-1.0, -2.0, -3.0]
    compressed_agent = _CountingAgent()
    compressed = _collect(
        _config(config, compression=True),
        _ScriptedEnvironment(masks, rewards),
        compressed_agent,
    ).episodes[0]
    regular_agent = _CountingAgent()
    regular = _collect(
        _config(config, compression=False),
        _ScriptedEnvironment(masks, rewards),
        regular_agent,
    ).episodes[0]

    assert compressed.step_count == regular.step_count == 3
    assert compressed.policy_step_count == 1
    assert compressed.forced_action_count == 2
    assert compressed.forced_action_ratio == pytest.approx(2.0 / 3.0)
    assert compressed_agent.act_calls == 1
    assert regular_agent.act_calls == 3
    assert compressed.reward_sum == regular.reward_sum == -6.0
    assert compressed.reward_components == regular.reward_components
    assert len(compressed.buffer) == 1
    assert compressed.buffer.transitions[0].reward == -6.0
    assert compressed.buffer.transitions[0].done is True
    assert compressed.unattributed_forced_reward == 0.0


def test_all_forced_episode_has_no_policy_transition(config):
    agent = _CountingAgent()
    episode = _collect(
        _config(config, compression=True),
        _ScriptedEnvironment(
            [[False, True], [True, False]],
            [-1.0, -2.0],
        ),
        agent,
    ).episodes[0]

    assert agent.act_calls == 0
    assert episode.policy_step_count == 0
    assert episode.forced_action_count == 2
    assert episode.reward_sum == -3.0
    assert episode.unattributed_forced_reward == -3.0


def test_cutoff_commits_pending_reward_and_bootstraps(config):
    agent = _CountingAgent()
    episode = _collect(
        _config(config, compression=True),
        _ScriptedEnvironment(
            [[False, False], [True, False], [False, False]],
            [-1.0, -2.0, -3.0],
        ),
        agent,
        step_limit=2,
    ).episodes[0]

    assert episode.step_count == 2
    assert episode.policy_step_count == 1
    assert episode.forced_action_count == 1
    transition = episode.buffer.transitions[0]
    assert transition.reward == -3.0
    assert transition.done is False
    assert transition.return_value == pytest.approx(-2.5)
    assert agent.value_calls == 1


def test_compression_rejects_discounted_objective(config):
    effective = _config(config, compression=True)
    effective["ppo"]["gamma"] = 0.99
    with pytest.raises(
        ValueError,
        match="forced action compression requires ppo.gamma = 1.0",
    ):
        _collect(
            effective,
            _ScriptedEnvironment([[False, False]], [-1.0]),
            _CountingAgent(),
        )
