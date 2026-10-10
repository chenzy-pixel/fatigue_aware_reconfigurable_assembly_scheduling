from copy import deepcopy
import pickle

import numpy as np
import pytest

from data.distribution import protocol_hashes
from environment import AssemblySchedulingEnv
from environment.resource_projection import ResourceProjector, projection_cache_scope
from environment.time_context import project_wait_state
from scripts.audit_observation_reward import compare_observations
from scripts.audit_state_sufficiency import worker_alias_instance, worker_histories


def projector(config, instance, entries=1024):
    settings = deepcopy(config)
    settings['training']['resource_projection_cache_entries'] = entries
    env = AssemblySchedulingEnv(settings)
    env.reset(instance)
    return env, ResourceProjector(env)


def test_scope_cache_is_bounded_and_isolates_returned_branches(config, fixed_instance, monkeypatch):
    env, p = projector(config, fixed_instance, 2)
    resources = p.initial_resources()
    original = ResourceProjector._transition_uncached
    calls = []

    def tracked(self, *args):
        calls.append(args[3])
        return original(self, *args)

    monkeypatch.setattr(ResourceProjector, '_transition_uncached', tracked)
    before = pickle.dumps(env)
    with projection_cache_scope(env):
        first = p.transition(5, 'A3', 'A2', 0, resources)
        first.resources.loads[:] = -999
        first.resources.workers[0] = (999999, 1.)
        first.start_tick = -999
        second = p.transition(5, 'A3', 'A2', 0, resources)
        assert len(calls) == 1
        assert second.start_tick == 0 and np.all(second.resources.loads >= 0)
        for earliest in (1, 2, 3):
            p.transition(5, 'A3', 'A2', earliest, resources)
            assert len(env._projection_transition_cache.entries) <= 2
    assert env._projection_transition_cache is None
    assert pickle.dumps(env) == before


def test_cheap_paths_bypass_cache_and_resource_key_is_exact(config, fixed_instance):
    env, p = projector(config, fixed_instance)
    resources = p.initial_resources()
    with projection_cache_scope(env):
        p.transition(0, 'A2', 'A2', 0, resources)
        p.transition(0, 'A0', 'A2', 0, resources)
        assert not env._projection_transition_cache.entries
        p.transition(5, 'A3', 'A2', 0, resources)
        different = resources.copy()
        different.loads[0] += 1e-12
        p.transition(5, 'A3', 'A2', 0, different)
        different.workers[0] = (different.workers[0][0], different.workers[0][1] + 1e-12)
        p.transition(5, 'A3', 'A2', 0, different)
        assert len(env._projection_transition_cache.entries) == 3


def test_wait_projection_and_state_change_do_not_share_cache(config, fixed_instance):
    env, p = projector(config, fixed_instance)
    with projection_cache_scope(env):
        p.transition(5, 'A3', 'A2', 0, p.initial_resources())
        current = env._projection_transition_cache
        projected = project_wait_state(env, 1)
        assert projected._projection_transition_cache is None
        assert env._projection_transition_cache is current
        env._invalidate_resource_snapshot()
        assert env._projection_transition_cache is None
    assert env._projection_transition_cache is None


def test_time_cache_is_bounded_and_independent_of_full_load_state(config, fixed_instance):
    env, p = projector(config, fixed_instance, 2)
    resources = p.initial_resources()
    with projection_cache_scope(env):
        timed = p.transition_time(5, 'A3', 'A2', 0, resources.workers)
        full = p.transition(5, 'A3', 'A2', 0, resources)
        assert timed.stages == full.stages
        assert timed.workers == tuple(full.resources.workers)
        assert len(env._projection_transition_cache.time_entries) == 1
        assert len(env._projection_transition_cache.entries) == 1
        # Full projection must apply the cached timing route to this caller's
        # loads; a time cache hit cannot carry another branch's load array.
        changed = resources.copy()
        changed.loads += 123
        another = p.transition(5, 'A3', 'A2', 0, changed)
        np.testing.assert_array_equal(another.resources.loads, full.resources.loads + 123)
        np.testing.assert_array_equal(another.baseline_loads, changed.loads)
        full.resources.workers[0] = (999999, 1.)
        full.resources.loads[:] = -999
        assert p.transition_time(5, 'A3', 'A2', 0, resources.workers) is timed
        for tick in (1, 2, 3):
            p.transition_time(5, 'A3', 'A2', tick, resources.workers)
            assert len(env._projection_transition_cache.time_entries) <= 2
    assert env._projection_transition_cache is None


@pytest.mark.parametrize('prefix', [0, 9, 11, 13, 14])
def test_cached_observations_equal_reference_for_real_commitments(config, fixed_instance, prefix):
    instance = worker_alias_instance(fixed_instance)
    histories = worker_histories(config, instance)[1][0]
    env, _ = projector(config, instance, 0)
    for action in histories[:prefix]:
        env.step(action, build_observation=False)
    reference = env.observe()
    reference_mask = env.get_action_mask()
    env.config['training']['resource_projection_cache_entries'] = 1024
    env._invalidate_resource_snapshot()
    observed = env.observe()
    assert all(compare_observations(reference, observed).values())
    np.testing.assert_array_equal(reference_mask, env.get_action_mask())
    assert env._projection_transition_cache is None


def test_cache_setting_does_not_change_physical_dataset_identity(config):
    a, b = deepcopy(config), deepcopy(config)
    a['training']['resource_projection_cache_entries'] = 0
    b['training']['resource_projection_cache_entries'] = 1024
    assert protocol_hashes(a) == protocol_hashes(b)


@pytest.mark.parametrize('capacity', [-1, True, 1.5])
def test_invalid_cache_capacity_is_rejected(config, capacity):
    settings = deepcopy(config)
    settings['training']['resource_projection_cache_entries'] = capacity
    with pytest.raises(ValueError, match='resource_projection_cache_entries'):
        AssemblySchedulingEnv(settings)
