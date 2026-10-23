from copy import deepcopy
from dataclasses import replace
from contextvars import copy_context
import pickle

import pytest

from data.models import AssemblyInstance
from environment import AssemblySchedulingEnv
from environment.observation_index import observation_index_scope, observation_operation_index
from environment.resource_projection import ResourceProjector
from environment.time_context import project_wait_state


def counted_index(monkeypatch):
    original = AssemblyInstance.operation_index
    calls = []

    def getter(instance):
        calls.append(instance)
        return original.fget(instance)

    monkeypatch.setattr(AssemblyInstance, "operation_index", property(getter))
    return calls


@pytest.mark.parametrize("fail", [False, True])
def test_scope_returns_independent_dicts_and_releases_after_success_or_error(fixed_instance, monkeypatch, fail):
    reference = fixed_instance.operation_index
    calls = counted_index(monkeypatch)

    def query():
        with observation_index_scope(fixed_instance):
            result = observation_operation_index(fixed_instance)
            result.clear()
            assert observation_operation_index(fixed_instance) == reference
            if fail:
                raise RuntimeError("abort observation")

    if fail:
        with pytest.raises(RuntimeError, match="abort observation"):
            query()
    else:
        query()
    assert len(calls) == 1
    assert observation_operation_index(fixed_instance) == reference
    assert observation_operation_index(fixed_instance) == reference
    assert len(calls) == 3  # next queries are uncached


def test_copied_context_cannot_extend_observation_cache_lifetime(fixed_instance, monkeypatch):
    calls = counted_index(monkeypatch)
    with observation_index_scope(fixed_instance):
        inherited = copy_context()
        inherited.run(observation_operation_index, fixed_instance)
    assert len(calls) == 1
    inherited.run(observation_operation_index, fixed_instance)
    assert len(calls) == 2


def test_nested_scopes_use_instance_identity_and_restore_outer_index(fixed_instance, monkeypatch):
    # Same instance_id, different operation order: value/name lookup is insufficient.
    other = replace(fixed_instance, orders=tuple(reversed(fixed_instance.orders)))
    expected = {id(instance): instance.operation_index for instance in (fixed_instance, other)}
    calls = counted_index(monkeypatch)
    with observation_index_scope(fixed_instance):
        with observation_index_scope(fixed_instance):
            assert observation_operation_index(fixed_instance) == expected[id(fixed_instance)]
        with observation_index_scope(other):
            assert observation_operation_index(other) == expected[id(other)]
        assert observation_operation_index(fixed_instance) == expected[id(fixed_instance)]
    assert [id(instance) for instance in calls] == [id(fixed_instance), id(other)]


def test_wait_query_shares_only_static_index_and_cannot_keep_it_after_scope(config, fixed_instance, monkeypatch):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    calls = counted_index(monkeypatch)
    before = pickle.dumps(env)
    with observation_index_scope(env.instance):
        projected = project_wait_state(env, 1)
        observation_operation_index(projected.instance)
        observation_operation_index(env.instance)
    assert pickle.dumps(env) == before
    assert len(calls) == 1
    observation_operation_index(projected.instance)
    assert len(calls) == 2


@pytest.mark.parametrize("projection_entries", [0, 1024])
def test_cold_observe_builds_one_index_and_new_instance_reset_refreshes_it(
    config, fixed_instance, monkeypatch, projection_entries,
):
    settings = deepcopy(config)
    settings["training"]["resource_projection_cache_entries"] = projection_entries
    env = AssemblySchedulingEnv(settings)
    env.reset(fixed_instance, build_observation=False)
    calls = counted_index(monkeypatch)
    env.observe()
    assert len(calls) == 1
    env.observe()  # cached graph copy does not build an index
    assert len(calls) == 1
    env._invalidate_resource_snapshot()
    env.observe()
    assert len(calls) == 2
    other = replace(fixed_instance, orders=tuple(reversed(fixed_instance.orders)))
    env.reset(other, build_observation=False)
    calls.clear()
    env.observe()
    assert len(calls) == 1 and calls[0] is env.instance
    assert observation_operation_index(env.instance) == other.operation_index


def test_projector_outside_observe_does_not_mutate_env_or_persist_index(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    projector = ResourceProjector(env)
    before = pickle.dumps(env)
    projector.transition(5, "A3", "A2", 0, projector.initial_resources())
    assert pickle.dumps(env) == before
