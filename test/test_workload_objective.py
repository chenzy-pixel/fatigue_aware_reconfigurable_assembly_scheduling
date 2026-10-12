"""Mathematical and physical regressions for the versioned workload objective."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, DecisionType, bounded_quality_score, proxy_return_from_metrics
from environment.observation_schema import global_feature_names
from environment.resource_projection import ResourceProjector
from environment.types import CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE, flow_reward_version
from environment.workload import (
    geometric_owa_weights, workload_config, workload_value, workload_delta,
    observation_schema_version, objective_names, metrics_workload_objective,
)


@pytest.fixture(scope="module")
def config():
    # Test the environment contract independently of config loading/calibration.
    return json.loads((Path(__file__).parents[1] / "configs/default.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixed_instance(config):
    return load_instance_yaml(Path(__file__).parents[1] / config["paths"]["fixed_instance"])


def _modern(config, mode="geometric_owa", q=0.5):
    effective = deepcopy(config)
    definition = {"mode": mode, "q": q if mode == "geometric_owa" else None}
    effective["workload_objective"] = definition
    effective["network"]["observation_schema_version"] = 11
    for key in tuple(effective["network"]):
        if key.startswith("workload_"):
            effective["network"].pop(key)
    scalarizer = effective["objective_scalarizer"]
    scalarizer["scales"] = {"flow": 1000.0, "cost": 300.0, "workload": 20.0}
    effective["reward"]["mode"] = flow_reward_version(effective)
    quality = dict(effective["evaluation"].get("quality_metric") or {})
    effective["evaluation"]["quality_metric"] = quality
    quality.pop("variance_scale", None)
    quality.update(version="canonical_bounded_workload_quality_v1", workload_scale=20.0,
                   flow_scale=1000.0, cost_scale=300.0,
                   workload_mode=mode, workload_q=definition["q"],
                   quality_weights={"flow": 0.5, "cost": 0.3, "workload": 0.2})
    return effective


def _legacy(config):
    effective = deepcopy(config)
    effective.pop("workload_objective", None)
    for key in tuple(effective["network"]):
        if key.startswith("workload_"):
            effective["network"].pop(key)
    effective["network"]["observation_schema_version"] = 10
    effective["objective_scalarizer"]["scales"] = {"flow": 1000.0, "cost": 300.0, "variance": 2.0}
    effective["reward"]["mode"] = flow_reward_version(effective)
    quality = dict(effective["evaluation"].get("quality_metric") or {})
    effective["evaluation"]["quality_metric"] = quality
    for key in tuple(quality):
        if key.startswith("workload_"):
            quality.pop(key)
    quality.update(version="canonical_bounded_quality_v1", variance_scale=2.0,
                   flow_scale=1000.0, cost_scale=300.0,
                   quality_weights={"flow": 0.5, "cost": 0.3, "variance": 0.2})
    return effective


def _single_reconfiguration(instance):
    source = instance.orders[0]
    operation = replace(source.operations[0], required_module=instance.modules[1])
    return replace(instance, orders=(replace(source, release_time=0.0, operations=(operation,)),))


def _enter_worker_phase(env):
    for action in np.flatnonzero(~env.get_action_mask()[:-1]):
        oi, mi = env.decode_production_action(int(action))
        if env.operations[oi].spec.required_module != env.machines[mi].current_module:
            env.step(int(action), build_observation=False)
            env.step(env.wait_action, build_observation=False)
            assert env.decision_type == DecisionType.WORKER
            return
    raise AssertionError("expected a reconfiguration candidate")


def test_geometric_weights_closed_form_examples_and_zero_workers_count():
    cfg = {"workload_objective": {"mode": "geometric_owa", "q": 0.5}}
    weights = geometric_owa_weights(6, 0.5)
    np.testing.assert_allclose(weights, np.array([32, 16, 8, 4, 2, 1]) / 63, rtol=0, atol=0)
    assert geometric_owa_weights(6, 0.5) is weights
    with pytest.raises(ValueError):
        weights[0] = 0.0
    assert workload_value([10, 10, 10, 10, 10, 0], cfg) == pytest.approx(620 / 63)
    assert workload_value([10] * 6, cfg) == pytest.approx(10)
    assert workload_delta([10, 10, 10, 10, 10, 0], [10] * 6, cfg) == pytest.approx(10 / 63)
    assert workload_value([10, 0], cfg) == pytest.approx(20 / 3)
    assert workload_value([10], cfg) == 10.0
    assert workload_value([], cfg) == 0.0


@pytest.mark.parametrize("q", [None, True, 0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_geometric_q_rejected(q):
    with pytest.raises(ValueError, match="q"):
        workload_config({"workload_objective": {"mode": "geometric_owa", "q": q}})


def test_full_geometric_default_and_flattened_identity_requirements():
    assert workload_config({"workload_objective": {"mode": "geometric_owa"}})["q"] == 0.5
    with pytest.raises(ValueError, match="q"):
        workload_config({"workload_mode": "geometric_owa", "observation_schema_version": 11})


def test_underflow_and_invalid_loads_fail_instead_of_zero_weight_approximation():
    with pytest.raises(ValueError, match="underflow"):
        geometric_owa_weights(1100, 0.5)
    for loads in ([float("inf")], [float("nan")], [-1], [[1, 2]]):
        with pytest.raises(ValueError, match="loads"):
            workload_value(loads, {"workload_objective": {"mode": "geometric_owa", "q": 0.5}})


def test_permutation_monotonicity_and_balance_transfers_include_rank_changes():
    cfg = {"workload_objective": {"mode": "geometric_owa", "q": 0.5}}
    rng = np.random.default_rng(41)
    weights = geometric_owa_weights(6, 0.5)
    for _ in range(40):
        before = rng.uniform(0, 40, 6)
        increments = rng.uniform(0.1, 8, 6)
        after = before + increments
        assert workload_value(rng.permutation(before), cfg) == pytest.approx(workload_value(before, cfg))
        delta = workload_delta(before, after, cfg)
        assert weights[-1] * increments.sum() <= delta <= weights[0] * increments.sum()
        assert delta > 0
    assert workload_value([14, 6, 6, 0], cfg) > workload_value([10, 10, 6, 0], cfg)
    assert workload_value([14, 6, 6, 0], cfg) > workload_value([14, 4, 4, 4], cfg)
    assert workload_value([14, 6, 6, 0], cfg) > workload_value([9, 9, 8, 0], cfg)


def test_variance_mode_is_population_variance_and_never_an_owa_fallback():
    cfg = {"workload_objective": {"mode": "variance", "q": None}}
    loads = [0, 0, 3, 4, 8, 10]
    assert workload_value(loads, cfg) == np.var(np.asarray(loads, dtype=np.float64))
    assert metrics_workload_objective({"worker_load_variance": 3.0}, cfg) == 3.0
    with pytest.raises(KeyError, match="workload_objective_value"):
        metrics_workload_objective({"worker_load_variance": 3.0},
                                   {"workload_objective": {"mode": "geometric_owa", "q": 0.5}})
    assert observation_schema_version(cfg) == 11
    assert objective_names(cfg) == ("flow", "cost", "workload")
    legacy = {**cfg, "network": {"observation_schema_version": 10}}
    assert objective_names(legacy) == ("flow", "cost", "variance")
    with pytest.raises(ValueError, match="schema 10"):
        observation_schema_version({"workload_objective": {"mode": "geometric_owa", "q": 0.5},
                                    "network": {"observation_schema_version": 10}})


@pytest.mark.parametrize("q", [.25, .5, .75])
def test_owa_decomposes_into_mean_labor_and_nonnegative_concentration(q):
    cfg = {"workload_objective": {"mode": "geometric_owa", "q": q}}
    weights = geometric_owa_weights(6, q)
    rng = np.random.default_rng(73)
    for loads in (np.zeros(6), np.full(6, 10.), *rng.uniform(0, 40, (20, 6))):
        ordered = np.sort(loads)[::-1]
        mean = float(loads.mean())
        gaps = np.cumsum(ordered)[:-1] - np.arange(1, 6) * mean
        concentration = float(np.dot(weights[:-1] - weights[1:], gaps))
        assert concentration >= -1e-12
        assert workload_value(loads, cfg) == pytest.approx(mean + concentration)


@pytest.mark.parametrize("mode", ["geometric_owa", "variance"])
def test_worker_production_global_and_wait_features_share_active_objective(config, fixed_instance, mode):
    effective = _modern(config, mode)
    env = AssemblySchedulingEnv(effective)
    env.reset(_single_reconfiguration(fixed_instance), preference=[0, 0, 1])
    observation = env.observe()
    assert observation.global_feature_names == global_feature_names(11)
    relation = observation.relations[CAPABLE_EDGE]
    position = relation.feature_names.index("estimated_worker_load_objective_delta_norm")
    oi, mi = map(int, relation.edge_index[:, 0])
    projection = ResourceProjector(env).candidate(oi, mi)
    expected = workload_delta(projection.path.baseline_loads, projection.path.resources.loads, effective)
    assert relation.edge_features[0, position] == pytest.approx(expected / 20, abs=1e-6)
    _enter_worker_phase(env)
    observation = env.observe()
    relation = observation.relations[SERVICE_CANDIDATE_EDGE]
    position = relation.feature_names.index("incremental_workload_objective_norm")
    for row, (mi, wi) in enumerate(relation.edge_index.T):
        expected = env.projected_worker_workload_objective(int(mi), int(wi)) - env._committed_workload_objective()
        assert relation.edge_features[row, position] == pytest.approx(expected / 20, abs=1e-6)
    action = int(np.flatnonzero(~env.get_action_mask()[:-1])[0])
    _, reward, _, _, _ = env.step(action, build_observation=False)
    metrics = env.metrics()
    assert metrics["workload_objective_value"] == 0.0
    assert metrics["committed_workload_objective_value"] > 0.0
    assert metrics["worker_load_variance"] == 0.0
    assert reward.workload == pytest.approx(-metrics["committed_workload_objective_value"])
    assert reward.variance == pytest.approx(-metrics["committed_worker_load_variance"])
    observed = env.observe()
    assert observed.global_features[-1] == pytest.approx(metrics["committed_workload_objective_value"] / 20, abs=1e-6)
    assert "estimated_workload_objective_delta_if_wait" in observed.action_set_feature_names
    assert metrics["training_cumulative_reward"] == pytest.approx(proxy_return_from_metrics(metrics, effective, preference=[0, 0, 1]))
    derived = dict(metrics)
    for name in ("reward_preference_quality_score", "actual_preference_quality_score", "raw_preference_quality_score"):
        derived.pop(name)
    assert proxy_return_from_metrics(derived, effective, preference=[0, 0, 1]) == pytest.approx(metrics["training_cumulative_reward"])
    derived.pop("reward_objective_workload_value")
    assert proxy_return_from_metrics(derived, effective, preference=[0, 0, 1]) == pytest.approx(metrics["training_cumulative_reward"])


def test_success_telescope_actual_logs_and_legacy_physics(config, fixed_instance):
    configs = [_legacy(config), _modern(config)]
    envs = [AssemblySchedulingEnv(effective) for effective in configs]
    for env in envs:
        env.reset(fixed_instance, preference=[0, 0, 1], build_observation=False)
    policy = HeuristicPolicy()
    modern_rewards = []
    while not envs[0].task_done:
        np.testing.assert_array_equal(envs[0].get_action_mask(), envs[1].get_action_mask())
        action = policy.select_action(envs[0])
        envs[0].step(action, build_observation=False)
        _, reward, _, _, _ = envs[1].step(action, build_observation=False)
        modern_rewards.append(reward)
    assert envs[1].task_succeeded
    assert envs[0].schedule_log == envs[1].schedule_log
    assert envs[0].reconfiguration_log == envs[1].reconfiguration_log
    assert envs[1].validate_schedule() == []
    metrics = envs[1].metrics()
    loads = {worker.spec.id: 0.0 for worker in envs[1].workers}
    for record in envs[1].reconfiguration_log:
        if record["stage"] in {"DIS", "INS"}:
            loads[record["worker_id"]] += record["end"] - record["start"]
    assert metrics["worker_loads"] == pytest.approx(loads)
    assert metrics["worker_load_total"] == pytest.approx(sum(loads.values()))
    assert metrics["worker_load_max"] == pytest.approx(max(loads.values()))
    assert metrics["worker_load_variance"] == pytest.approx(np.var(list(loads.values())))
    assert metrics["workload_objective_value"] == pytest.approx(workload_value(list(loads.values()), configs[1]))
    assert metrics["committed_workload_objective_value"] == pytest.approx(metrics["workload_objective_value"])
    assert sum(reward.workload for reward in modern_rewards) == pytest.approx(-metrics["workload_objective_value"])
    assert sum(reward.variance for reward in modern_rewards) == pytest.approx(-metrics["worker_load_variance"])
    assert metrics["training_cumulative_reward"] == pytest.approx(proxy_return_from_metrics(metrics, configs[1], preference=[0, 0, 1]))


def test_failure_settles_unexecuted_commitments_and_charges_one_penalty(config, fixed_instance):
    effective = _modern(config)
    env = AssemblySchedulingEnv(effective)
    env.reset(replace(_single_reconfiguration(fixed_instance), horizon=1.0),
              preference=[0, 0, 1], build_observation=False)
    _enter_worker_phase(env)
    action = int(np.flatnonzero(~env.get_action_mask()[:-1])[0])
    mi, wi = env.decode_worker_action(action)
    planned_workload = env.projected_worker_workload_objective(mi, wi)
    _, reward, _, _, _ = env.step(action, build_observation=False)
    assert env.task_failed
    metrics = env.metrics()
    assert 0 < metrics["workload_objective_value"] < planned_workload
    assert not env._active_committed_worker_tasks
    assert metrics["committed_workload_objective_value"] == pytest.approx(metrics["workload_objective_value"])
    assert metrics["terminal_failure_penalty_applied"] == effective["reward"]["terminal_failure_penalty"]
    assert reward.failure == -effective["reward"]["terminal_failure_penalty"]
    assert metrics["training_cumulative_reward"] == pytest.approx(proxy_return_from_metrics(metrics, effective, preference=[0, 0, 1]))


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("flow", ["raw_v1", "excess_proportional_lb_v1"])
def test_wait_terminal_projection_matches_commitment_rollback(config, fixed_instance, q, flow):
    effective = _modern(config, q=q)
    effective["objective_scalarizer"]["flow_mode"] = flow
    effective["reward"]["mode"] = flow_reward_version(effective)
    first = _single_reconfiguration(fixed_instance)
    late = replace(fixed_instance.orders[1], release_time=1.0,
                   operations=(fixed_instance.orders[1].operations[0],))
    instance = replace(first, horizon=1.0, orders=(*first.orders, late))
    env = AssemblySchedulingEnv(effective)
    env.reset(instance, preference=[0, 0, 1], build_observation=False)
    _enter_worker_phase(env)
    action = int(np.flatnonzero(~env.get_action_mask()[:-1])[0])
    env.step(action, build_observation=False)
    assert not env.task_done
    before = env._committed_workload_objective()
    observation = env.observe()
    column = observation.action_set_feature_names.index("estimated_workload_objective_delta_if_wait")
    predicted = float(observation.action_set_features[column]) * 20
    assert predicted < 0
    _, reward, _, _, _ = env.step(env.wait_action, build_observation=False)
    assert env.task_failed
    assert env._committed_workload_objective() - before == pytest.approx(predicted, abs=1e-6)
    assert reward.workload == pytest.approx(-predicted, abs=1e-6)
    assert env._committed_workload_objective() == pytest.approx(env._workload_objective())


def test_direct_modern_quality_metric_and_legacy_variance_names(config):
    effective = _modern(config)
    metric = effective["evaluation"]["quality_metric"]
    assert bounded_quality_score(0, 0, 20, metric, preference=[0, 0, 1]) == 0.5
    old = _legacy(config)
    assert objective_names(old) == ("flow", "cost", "variance")
    assert flow_reward_version(old).endswith("v3")
    assert flow_reward_version(effective).endswith("v5")


def test_schema11_variance_renames_contract_without_changing_initial_numbers(config, fixed_instance):
    legacy = _legacy(config)
    modern = _modern(config, mode="variance")
    modern["objective_scalarizer"]["scales"]["workload"] = legacy["objective_scalarizer"]["scales"]["variance"]
    envs = [AssemblySchedulingEnv(effective) for effective in (legacy, modern)]
    observations = [env.reset(_single_reconfiguration(fixed_instance)) for env in envs]
    np.testing.assert_array_equal(observations[0].global_features, observations[1].global_features)
    np.testing.assert_array_equal(observations[0].action_set_features, observations[1].action_set_features)
    for node_type in observations[0].node_features:
        np.testing.assert_array_equal(observations[0].node_features[node_type], observations[1].node_features[node_type])
    for edge_type in observations[0].relations:
        np.testing.assert_array_equal(observations[0].relations[edge_type].edge_features,
                                      observations[1].relations[edge_type].edge_features)
    assert observations[0].global_feature_names[-1] == "objective_committed_variance_norm"
    assert observations[1].global_feature_names[-1] == "objective_committed_workload_norm"


def test_modern_raw_wait_reuses_projection_and_preserves_order_slack(config, fixed_instance, monkeypatch):
    import environment.env as env_module
    from environment.time_context import project_wait_state

    env = AssemblySchedulingEnv(_modern(config))
    env.reset(fixed_instance, build_observation=False)
    rule = HeuristicPolicy()
    tested = 0
    for _ in range(50):
        certificate = env._wait_certificate()
        ticks = int(certificate.get("wait_ticks", 0))
        if certificate.get("allowed") and ticks > 0:
            reference = project_wait_state(env, ticks)
            settled = project_wait_state(env, ticks, settle_terminal=True, certificate=certificate)
            if not settled.task_done:
                expected = reference.minimum_active_order_slack_norm()
                assert settled.minimum_active_order_slack_norm() == expected
                calls = []

                def counted(*args, **kwargs):
                    calls.append(kwargs.get("settle_terminal", False))
                    return project_wait_state(*args, **kwargs)

                with monkeypatch.context() as scoped:
                    scoped.setattr(env_module, "project_wait_state", counted)
                    values, names = env._build_action_set_features({})
                assert values[names.index("minimum_order_slack_after_wait_norm")] == pytest.approx(expected, abs=1e-6)
                assert calls == [True]
                tested += 1
                if tested == 8:
                    break
        if env.task_done:
            break
        env.step(rule.select_action(env), build_observation=False)
    assert tested == 8
