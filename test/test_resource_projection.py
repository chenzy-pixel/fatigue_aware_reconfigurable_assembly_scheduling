"""Sequential observation paths agree with real commitments and fatigue rules."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from environment import AssemblySchedulingEnv, CAPABLE_EDGE
from environment.resource_projection import ResourceProjector, ProjectedResources
from environment.time_context import OrderTimeEstimator, _Resources
from scripts.audit_state_sufficiency import worker_alias_instance, worker_histories
from scripts.audit_schema9_followup import edge_times


def sequential_instance(template):
    instance = worker_alias_instance(template)
    workers = tuple(replace(w, initial_fatigue=.5 if w.id == 'H5' else .74) for w in template.workers)
    orders = tuple(replace(o, release_time=0 if o.id == 'A' else o.release_time) for o in instance.orders)
    waves = deepcopy(instance.waves)
    waves['W1']['release_interval'][0] = 0
    return replace(instance, workers=workers, orders=orders, waves=waves)


def test_standard_busy_machine_prediction_honors_actual_end(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    env.step(0)
    env.step(env.wait_action)
    values = edge_times(env, 'O_J2_1', 'M1')
    assert env.current_time == 3
    assert values['earliest_start_time_norm'] == pytest.approx(9.9, abs=1e-5)
    assert values['resource_ready_time_norm'] == pytest.approx(9.9, abs=1e-5)
    assert values['predicted_finish_time_norm'] == pytest.approx(18.9, abs=1e-5)
    oi = fixed_instance.operation_index['O_J1_1']
    own = env._candidate_resource_projection(oi, 0)
    assert own.processing_start_tick == 0 and own.finish_tick == 99


@pytest.mark.parametrize('mode', ['full', 'neutral'])
def test_running_installation_uses_fixed_finish(config, fixed_instance, mode):
    settings = deepcopy(config)
    settings['environment']['fatigue_mode'] = mode
    env = worker_histories(settings, worker_alias_instance(fixed_instance))[0][0]
    values = edge_times(env, 'A_1', 'M6')
    assert values['earliest_start_time_norm'] == pytest.approx(38, abs=1e-5)
    assert values['predicted_finish_time_norm'] == pytest.approx(46.4, abs=1e-5)
    oi = env.instance.operation_index['E0_1']
    future = env._candidate_resource_projection(oi, 5)
    assert future.processing_start_tick >= 464
    assert future.path is not None and not future.path.stages
    assert future.resources.loads.tolist() == env._committed_worker_loads.tolist()


def test_sequential_path_replays_time_fatigue_cost_and_load(config, fixed_instance):
    instance = sequential_instance(fixed_instance)
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    projector = ResourceProjector(env)
    route = env._candidate_resource_projection(0, 5).path
    assert route.end_tick == 136
    assert len(route.stages) == 2
    assert [s.worker_index for s in route.stages] == [4, 4]
    assert route.stages[0].end_fatigue == pytest.approx(.6505)
    assert route.stages[1].start_tick > route.stages[0].end_tick
    estimator = OrderTimeEstimator(env)
    state = _Resources(estimator.machines.copy(), estimator.workers.copy(), estimator.loads.copy())
    assert estimator._transition_finish(5, 'A3', 'A2', 0, state) == route.end_tick
    dis, ins, labor, downtime, variance = projector.path_costs(5, route)
    assert dis == 4 and ins == 6
    assert labor == pytest.approx(sum(s.duration_ticks for s in route.stages)*env.resolution)
    assert downtime == pytest.approx(13.6*4.3)
    assert variance == pytest.approx(np.var(route.resources.loads))
    env.step(env.encode_production_action(0, 5), build_observation=False)
    env.step(env.wait_action, build_observation=False)
    for stage in route.stages:
        while env.current_tick < stage.start_tick:
            env.step(env.wait_action, build_observation=False)
        if env.decision_type.value == 'PRODUCTION':
            env.step(env.wait_action, build_observation=False)
        action = env.encode_worker_action(5, stage.worker_index)
        assert not env.get_action_mask()[action]
        env.step(action, build_observation=False)
        while env.current_tick < stage.end_tick:
            env.step(env.wait_action, build_observation=False)
        assert env.workers[stage.worker_index].fatigue == pytest.approx(stage.end_fatigue)
    assert env.current_tick == route.end_tick
    assert env._reconfiguration_cost == pytest.approx(dis+ins+labor+downtime)
    np.testing.assert_allclose(env._committed_worker_loads, route.resources.loads)


def test_time_only_route_matches_full_route_without_load_materialization(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(sequential_instance(fixed_instance), build_observation=False)
    projector = ResourceProjector(env)
    resources = projector.initial_resources()

    full = projector.transition(5, 'A3', 'A2', 0, resources)
    timed = projector.transition_time(5, 'A3', 'A2', 0, resources.workers)

    assert full is not None and timed is not None
    assert timed.end_tick == full.end_tick
    assert timed.stages == full.stages
    assert timed.safe_disassembly_workers == full.safe_disassembly_workers
    assert timed.safe_installation_workers == full.safe_installation_workers
    assert list(timed.workers) == full.resources.workers
    assert not hasattr(timed, 'loads')


def test_route_selection_is_independent_of_loads(config, fixed_instance):
    # Protect the invariant that lets both result types share one route kernel.
    env = AssemblySchedulingEnv(config)
    env.reset(sequential_instance(fixed_instance), build_observation=False)
    projector = ResourceProjector(env)
    state = projector.initial_resources()
    reference = projector.transition(5, 'A3', 'A2', 0, state)
    state.loads[:] = np.arange(len(state.loads)) * 1e9
    changed = projector.transition(5, 'A3', 'A2', 0, state)
    timed = projector.transition_time(5, 'A3', 'A2', 0, state.workers)
    assert reference.stages == changed.stages == timed.stages
    assert reference.end_tick == changed.end_tick == timed.end_tick
    assert reference.resources.workers == changed.resources.workers == list(timed.workers)


def test_order_chain_does_not_materialize_full_resource_state(config, fixed_instance, monkeypatch):
    env = AssemblySchedulingEnv(config)
    env.reset(sequential_instance(fixed_instance), build_observation=False)
    expected = OrderTimeEstimator(env).finish_ticks()

    def forbidden(*args, **kwargs):
        raise AssertionError('time-only order chain materialized full resources')

    monkeypatch.setattr(ResourceProjector, 'initial_resources', forbidden)
    monkeypatch.setattr(ResourceProjector, 'transition', forbidden)
    monkeypatch.setattr(ProjectedResources, 'copy', forbidden)
    assert OrderTimeEstimator(env).finish_ticks() == expected


def test_legacy_estimator_load_access_remains_isolated(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    estimator = OrderTimeEstimator(env)
    estimator.finish_ticks()
    before = env._committed_worker_loads.copy()
    estimator.loads[:] = -999
    np.testing.assert_array_equal(env._committed_worker_loads, before)


@pytest.mark.parametrize('source, target', [('A0', 'A2'), ('A3', 'A3')])
def test_time_only_route_matches_single_stage_and_noop(config, fixed_instance, source, target):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    projector = ResourceProjector(env)
    resources = projector.initial_resources()

    full = projector.transition(0, source, target, 0, resources)
    timed = projector.transition_time(0, source, target, 0, resources.workers)
    assert (full is None) == (timed is None)
    if full is not None:
        assert timed is not None
        assert timed.end_tick == full.end_tick
        assert timed.stages == full.stages
        assert list(timed.workers) == full.resources.workers


def test_all_worker_pairs_are_compared_before_selecting_route(config, fixed_instance):
    # The quickest DIS by H5 delays its own INS; slightly slower H6 lets H5 stay rested.
    instance = worker_alias_instance(fixed_instance)
    workers = tuple(replace(w, initial_fatigue=0 if w.id == 'H5' else .05 if w.id == 'H6' else .74)
                    for w in instance.workers)
    parameters = dict(instance.machines[5].module_parameters)
    parameters['A2'] = replace(parameters['A2'], installation_base_time=8)
    machines = (*instance.machines[:5], replace(instance.machines[5], module_parameters=parameters), *instance.machines[6:])
    env = AssemblySchedulingEnv(config)
    env.reset(replace(instance, workers=workers, machines=machines), build_observation=False)
    # Use t=0 initial snapshot so recovery before the first release is irrelevant.
    projector = ResourceProjector(env)
    state = ProjectedResources([(0, w.initial_fatigue) for w in workers], np.zeros(6))
    options = projector.stage_options(5, 'A3', False, 0, state)
    fastest = min(options, key=lambda item: (item[0].end_tick, item[0].worker_index))[0]
    route = projector.transition(5, 'A3', 'A2', 0, state)
    assert fastest.worker_index == 4
    assert route.stages[0].worker_index == 5
    assert route.stages[1].worker_index == 4


def test_prefix_commitment_load_is_not_double_counted(config, fixed_instance):
    env = worker_histories(config, worker_alias_instance(fixed_instance))[0][0]
    oi = env.instance.operation_index['E0_1']
    projection = env._candidate_resource_projection(oi, 5)
    np.testing.assert_array_equal(projection.baseline_loads, env._committed_worker_loads)
    assert projection.path.stages == ()
    assert ResourceProjector(env).path_costs(5, projection.path) == (0, 0, 0, 0, 0)


def test_pending_prefix_is_materialized_once_and_excluded_from_new_cost(config, fixed_instance):
    instance = worker_alias_instance(fixed_instance)
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    for action in worker_histories(config, instance)[1][0][:3]:
        env.step(action, build_observation=False)
    projector = ResourceProjector(env)
    prefix = projector.active(5, projector.initial_resources())
    future = env._candidate_resource_projection(instance.operation_index['E0_1'], 5)
    assert len(prefix.stages) == 2
    assert future.processing_start_tick >= prefix.end_tick+env.estimate_processing_ticks(0, 5)
    np.testing.assert_array_equal(future.baseline_loads, prefix.resources.loads)
    np.testing.assert_array_equal(future.resources.loads, prefix.resources.loads)
    assert projector.path_costs(5, future.path) == (0, 0, 0, 0, 0)


def test_projection_purity_branch_isolation_and_finite_cross_horizon(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(sequential_instance(fixed_instance))
    projector = ResourceProjector(env)
    before = pickle.dumps(env)
    state = projector.initial_resources()
    copy = pickle.dumps(state)
    first = projector.transition(5, 'A3', 'A2', env.horizon_tick+10, state)
    second = projector.transition(5, 'A3', 'A2', env.horizon_tick+10, state)
    assert first.end_tick == second.end_tick > env.horizon_tick
    assert pickle.dumps(env) == before and pickle.dumps(state) == copy
    for stage in first.stages:
        assert stage.end_fatigue <= env.instance.fatigue.maximum_safe_fatigue + 1e-9
    first.resources.loads[0] += 100
    assert second.resources.loads[0] != first.resources.loads[0]


def test_a0_installation_and_unreachable_stage(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    projector = ResourceProjector(env)
    route = projector.transition(0, 'A0', 'A2', 0, projector.initial_resources())
    assert len(route.stages) == 1 and route.stages[0].installation
    env.instance = replace(env.instance, fatigue=replace(env.instance.fatigue,
        idle_recovery_rate_per_minute=0, installation_accumulation_rate_per_minute=1))
    assert ResourceProjector(env).transition(0, 'A0', 'A2', 0, projector.initial_resources()) is None


def test_unreachable_candidate_uses_finite_sentinel_and_zero_alternatives(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    env.instance = replace(env.instance, fatigue=replace(env.instance.fatigue,
        idle_recovery_rate_per_minute=0, installation_accumulation_rate_per_minute=1))
    env._invalidate_resource_snapshot()
    oi = next(i for i, op in enumerate(env.operations) if op.spec.required_module == 'A2')
    candidate = env._candidate_resource_projection(oi, 0)
    profile = env._production_resource_profile(0, 'A2')
    assert candidate.processing_start_tick is None and candidate.finish_tick is None
    assert candidate.resource_ready_tick == profile.resource_ready_tick == env.horizon_tick+1
    obs = env.observe()
    store = obs.relations[CAPABLE_EDGE]
    pos = np.flatnonzero((store.edge_index[0] == oi) & (store.edge_index[1] == 0))[0]
    values = dict(zip(store.feature_names, store.edge_features[pos]))
    assert values['safe_disassembly_worker_ratio'] == values['safe_installation_worker_ratio'] == 0
    assert values['predicted_finish_time_norm'] == pytest.approx((env.horizon_tick+1)/env.horizon_tick)
    assert np.all(np.isfinite(store.edge_features))


def test_projector_safety_threshold_waits_on_the_same_grid(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    projector = ResourceProjector(env)
    boundary = .75-.03*2.2
    results = []
    for offset in (-1e-7, 1e-7):
        state = projector.initial_resources()
        state.workers[0] = (0, boundary+offset)
        stage = next(stage for stage, _ in projector.stage_options(0, 'A2', True, 0, state)
                     if stage.worker_index == 0)
        results.append(stage)
    assert results[0].start_tick == 0 and results[1].start_tick == 1
    assert all(s.duration_ticks == 22 and s.end_fatigue <= .75+1e-9 for s in results)


@pytest.mark.parametrize('offset,expected', [(5e-10, 10), (2e-9, 11)])
def test_projector_ceil_duration_matches_kernel_boundary(config, fixed_instance, offset, expected):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance)
    parameters = dict(env.machines[0].spec.module_parameters)
    parameters['A2'] = replace(parameters['A2'], installation_base_time=1+offset)
    env.machines[0].spec = replace(env.machines[0].spec, module_parameters=parameters)
    state = ResourceProjector(env).initial_resources()
    state.workers[0] = (0, 0)
    stage = next(stage for stage, _ in ResourceProjector(env).stage_options(0, 'A2', True, 0, state)
                 if stage.worker_index == 0)
    assert stage.duration_ticks == expected


@pytest.mark.parametrize('stage_prefix', [9, 11, 13, 14])
def test_active_reconfiguration_stages_share_candidate_and_order_projector(config, fixed_instance, stage_prefix):
    instance = worker_alias_instance(fixed_instance)
    path = worker_histories(config, instance)[1][0]
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    for action in path[:stage_prefix]:
        env.step(action, build_observation=False)
    oi, mi = instance.operation_index['A_1'], 5
    route = env._candidate_resource_projection(oi, mi)
    projector = ResourceProjector(env)
    own = projector.active(mi, projector.initial_resources())
    assert route.processing_start_tick == own.end_tick
    assert route.finish_tick == own.end_tick+env.estimate_processing_ticks(oi, mi)
    assert np.all(np.isfinite(env.observe().relations[CAPABLE_EDGE].edge_features))
