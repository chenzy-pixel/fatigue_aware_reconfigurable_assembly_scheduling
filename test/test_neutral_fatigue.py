from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from agent.baselines import HeuristicPolicy
from data.dataset import canonical_json_bytes
from data.distribution import PRESSURE_TYPES
from data.generate_orders import InstanceGenerator
from data.models import instance_to_dict
from environment import AssemblySchedulingEnv, DecisionType, proxy_return_from_metrics
from environment.fatigue_monitor import audit_fatigue
from environment.types import WorkerState


def _config(config, mode):
    effective = deepcopy(config)
    effective["environment"]["fatigue_mode"] = mode
    return effective


def _assert_observations_equal(first, second):
    assert first.decision_type == second.decision_type
    assert first.node_feature_names == second.node_feature_names
    assert first.global_feature_names == second.global_feature_names
    assert first.action_set_feature_names == second.action_set_feature_names
    assert first.node_ids == second.node_ids
    np.testing.assert_array_equal(first.preference, second.preference)
    np.testing.assert_array_equal(first.global_features, second.global_features)
    np.testing.assert_array_equal(first.action_set_features, second.action_set_features)
    assert first.node_features.keys() == second.node_features.keys()
    for node_type in first.node_features:
        np.testing.assert_array_equal(first.node_features[node_type], second.node_features[node_type])
    assert first.relations.keys() == second.relations.keys()
    for relation, edges in first.relations.items():
        other = second.relations[relation]
        assert edges.feature_names == other.feature_names
        np.testing.assert_array_equal(edges.edge_index, other.edge_index)
        np.testing.assert_array_equal(edges.edge_features, other.edge_features)


def _first_reconfiguration_action(env):
    for action in np.flatnonzero(~env.get_action_mask()[:-1]):
        operation, machine = env.decode_production_action(int(action))
        if env.operations[operation].spec.required_module != env.machines[machine].current_module:
            return int(action)
    raise AssertionError("expected an initial reconfiguration candidate")


def _initial_reconfiguration_instance(instance):
    source = instance.orders[0]
    operation = replace(source.operations[0], required_module=instance.modules[1])
    return replace(
        instance,
        orders=(replace(source, release_time=0.0, operations=(operation,)),),
    )


def _enter_worker_phase(env):
    action = _first_reconfiguration_action(env)
    _, machine = env.decode_production_action(action)
    env.step(action, build_observation=False)
    env.step(env.wait_action, build_observation=False)
    assert env.decision_type == DecisionType.WORKER
    return machine


def test_neutral_fatigue_source_parameters_cannot_change_policy_or_dynamics(config, fixed_instance):
    source = replace(
        fixed_instance,
        fatigue=replace(
            fixed_instance.fatigue,
            maximum_safe_fatigue=0.9,
            disassembly_time_coefficient=25.0,
            installation_time_coefficient=31.0,
            disassembly_accumulation_rate_per_minute=0.8,
            installation_accumulation_rate_per_minute=0.7,
            idle_recovery_rate_per_minute=0.3,
        ),
        workers=tuple(replace(worker, initial_fatigue=0.8) for worker in fixed_instance.workers),
    )
    original = canonical_json_bytes(instance_to_dict(source))
    first, second = [AssemblySchedulingEnv(_config(config, "neutral")) for _ in range(2)]
    first.reset(fixed_instance)
    second.reset(source)
    assert first.original_instance is fixed_instance
    assert second.original_instance is source
    assert canonical_json_bytes(instance_to_dict(source)) == original
    assert second.instance.workers != source.workers
    assert all(worker.fatigue == 0.0 for worker in second.workers)
    policy = HeuristicPolicy()
    decisions = 0
    while not first.task_done:
        _assert_observations_equal(first.observe(), second.observe())
        np.testing.assert_array_equal(first.get_action_mask(), second.get_action_mask())
        action = policy.select_action(first)
        first_result = first.step(action)
        second_result = second.step(action)
        assert first_result[1:] == second_result[1:]
        assert first.current_tick == second.current_tick
        assert [worker.fatigue for worker in first.workers] == [0.0] * len(first.workers)
        assert [worker.fatigue for worker in second.workers] == [0.0] * len(second.workers)
        decisions += 1
        assert decisions < 1000
    assert second.task_done
    _assert_observations_equal(first.observe(), second.observe())
    assert first.schedule_log == second.schedule_log
    assert first.reconfiguration_log == second.reconfiguration_log
    metrics = second.metrics()
    assert metrics["fatigue_mode"] == "neutral"
    assert metrics["maximum_worker_fatigue"] == 0.0
    assert metrics["fatigue_masked_action_count"] == 0
    assert metrics["fatigue_monitor_peak"] > metrics["safe_fatigue_limit"]
    assert metrics["fatigue_monitor_over_limit_area"] > 0.0
    assert second.validate_schedule() == []
    assert canonical_json_bytes(instance_to_dict(source)) == original


def test_explicit_full_mode_matches_default_trajectory_and_original_monitor(config, fixed_instance):
    default = deepcopy(config)
    default["environment"].pop("fatigue_mode", None)
    first = AssemblySchedulingEnv(default)
    second = AssemblySchedulingEnv(_config(config, "full"))
    first.reset(fixed_instance)
    second.reset(fixed_instance)
    assert second.instance is fixed_instance
    policy = HeuristicPolicy()
    while not first.task_done:
        _assert_observations_equal(first.observe(), second.observe())
        np.testing.assert_array_equal(first.get_action_mask(), second.get_action_mask())
        action = policy.select_action(first)
        assert first.step(action)[1:] == second.step(action)[1:]
    assert first.metrics() == second.metrics()
    assert first.schedule_log == second.schedule_log
    assert first.reconfiguration_log == second.reconfiguration_log
    metrics = second.metrics()
    assert metrics["fatigue_mode"] == "full"
    assert metrics["fatigue_monitor_peak"] == pytest.approx(metrics["maximum_worker_fatigue"])
    assert metrics["fatigue_monitor_over_limit_area"] == pytest.approx(0.0, abs=1e-10)


def test_neutral_keeps_worker_qualification_and_exclusive_resources(config, fixed_instance):
    env = AssemblySchedulingEnv(_config(config, "neutral"))
    env.reset(_initial_reconfiguration_instance(fixed_instance))
    machine = _enter_worker_phase(env)
    task = env._pending_reconfiguration(env.machines[machine].spec.id)
    module = task.source_module
    mask = env.get_action_mask()
    eligible = []
    ineligible = []
    for index, worker in enumerate(env.workers):
        action = env.encode_worker_action(machine, index)
        qualified = module in worker.spec.qualified_modules
        assert bool(not mask[action]) == qualified
        (eligible if qualified else ineligible).append(action)
    assert eligible and ineligible
    with pytest.raises(ValueError, match="illegal"):
        env.step(ineligible[0])
    env.step(eligible[0])
    _, worker_index = env.decode_worker_action(eligible[0])
    worker = env.workers[worker_index]
    assert worker.state == WorkerState.DIS
    assert worker.fatigue == 0.0
    assert env.reconfiguration_log[-1]["duration"] == pytest.approx(
        env.machines[machine].spec.module_parameters[module].disassembly_base_time
    )
    assert not env._worker_can_start(task, worker)


def test_original_fatigue_monitor_integrates_threshold_crossings_and_saturation(fixed_instance):
    worker = replace(fixed_instance.workers[0], initial_fatigue=0.4)
    instance = replace(
        fixed_instance,
        workers=(worker,),
        fatigue=replace(
            fixed_instance.fatigue,
            maximum_safe_fatigue=0.6,
            disassembly_accumulation_rate_per_minute=0.2,
            idle_recovery_rate_per_minute=0.1,
        ),
    )
    record = {"worker_id": worker.id, "stage": "DIS", "start": 0.0, "end": 4.0,
              "reconfiguration_id": "R1"}
    metrics, segments = audit_fatigue(instance, [record], 10.0)
    # DIS: threshold crossing at 1, saturation at 3. IDLE: recovery crosses at 8.
    assert metrics["fatigue_monitor_peak"] == 1.0
    assert metrics["fatigue_monitor_over_limit_minutes"] == pytest.approx(7.0)
    assert metrics["fatigue_monitor_over_limit_area"] == pytest.approx(1.6)
    assert metrics["fatigue_monitor_over_limit_time_ratio"] == pytest.approx(0.7)
    assert metrics["fatigue_monitor_over_limit_area_ratio"] == pytest.approx(0.16)
    assert metrics["fatigue_monitor_over_limit_worker_ratio"] == 1.0
    assert metrics["worker_reconfiguration_busy_minutes"] == 4.0
    assert metrics["worker_reconfiguration_idle_minutes"] == 6.0
    assert metrics["worker_reconfiguration_duty_ratio"] == pytest.approx(0.4)
    assert [row["stage"] for row in segments] == ["DIS", "IDLE"]
    assert segments[-1]["fatigue_end"] == pytest.approx(0.4)
    assert sum(row["over_limit_area"] for row in segments) == pytest.approx(1.6)


def test_monitor_clips_active_stages_and_counts_completed_installations(fixed_instance):
    worker = replace(fixed_instance.workers[0], initial_fatigue=0.0)
    instance = replace(fixed_instance, workers=(worker,))
    records = [
        {"worker_id": worker.id, "stage": "DIS", "start": 0.0, "end": 2.0,
         "reconfiguration_id": "R1"},
        {"worker_id": worker.id, "stage": "INS", "start": 2.0, "end": 9.0,
         "reconfiguration_id": "R1"},
        {"worker_id": worker.id, "stage": "DIS", "start": 10.0, "end": 12.0,
         "reconfiguration_id": "R2"},
    ]
    metrics, segments = audit_fatigue(instance, records, 5.0)
    assert metrics["worker_reconfiguration_busy_minutes"] == 5.0
    assert metrics["worker_reconfiguration_idle_minutes"] == 0.0
    assert metrics["max_consecutive_worker_stages"] == 2
    assert metrics["mean_interstage_idle_minutes"] == 0.0
    assert metrics["completed_reconfigurations_per_minute"] == 0.0
    assert segments[-1]["end"] == 5.0
    records[1] = {**records[1], "end": 5.0, "truncated": True}
    assert audit_fatigue(instance, records, 5.0)[0]["completed_reconfigurations_per_minute"] == 0.0
    records[1].pop("truncated")
    assert audit_fatigue(instance, records, 5.0)[0]["completed_reconfigurations_per_minute"] == 0.2
    with pytest.raises(ValueError, match="overlapping"):
        audit_fatigue(instance, records + [{**records[0], "start": 1.0}], 5.0)


@pytest.mark.parametrize("pressure_type", PRESSURE_TYPES)
def test_generator_full_and_neutral_preserve_v2_instance_identity(config, fixed_instance, pressure_type):
    generators = [InstanceGenerator(fixed_instance, config["generator"], config=_config(config, mode))
                  for mode in ("full", "neutral")]
    records = [generator.generate(seed=1_001_050, split="train", pressure_type=pressure_type,
                                   severity=0.7, run_diagnostics=False)
               for generator in generators]
    assert canonical_json_bytes(records[0].to_dict()) == canonical_json_bytes(records[1].to_dict())


def test_generator_diagnostics_and_counterfactuals_use_full_physics(config, fixed_instance, monkeypatch):
    import environment
    original_env = environment.AssemblySchedulingEnv
    modes = []
    def track_env(effective):
        modes.append(effective["environment"]["fatigue_mode"])
        return original_env(effective)
    monkeypatch.setattr(environment, "AssemblySchedulingEnv", track_env)
    generators = [InstanceGenerator(fixed_instance, config["generator"], config=_config(config, mode))
                  for mode in ("full", "neutral")]
    records = [generator.generate(seed=2_001_050, split="validation", pressure_type="easy",
                                   run_diagnostics=True, classify_reconfiguration_value=True)
               for generator in generators]
    assert canonical_json_bytes(records[0].to_dict()) == canonical_json_bytes(records[1].to_dict())
    assert modes == ["full"] * 4
    assert generators[1].config["environment"]["fatigue_mode"] == "neutral"


@pytest.mark.parametrize("mode", ("full", "neutral"))
def test_partial_worker_commitment_reward_reconstructs_exactly(config, fixed_instance, mode):
    effective = _config(config, mode)
    env = AssemblySchedulingEnv(effective)
    env.reset(_initial_reconfiguration_instance(fixed_instance), preference=[0.0, 0.0, 1.0])
    _enter_worker_phase(env)
    action = int(np.flatnonzero(~env.get_action_mask()[:-1])[0])
    _, reward, _, _, _ = env.step(action)
    metrics = env.metrics()
    assert metrics["time"] == 0.0
    assert metrics["worker_load_variance"] == 0.0
    assert metrics["reward_objective_worker_load_variance"] > 0.0
    assert reward.variance < 0.0
    assert metrics["reward_preference_quality_score"] > metrics["actual_preference_quality_score"]
    assert metrics["training_cumulative_reward"] == pytest.approx(
        proxy_return_from_metrics(metrics, effective, preference=env.preference), abs=1e-12
    )
    derived = dict(metrics)
    for name in ("reward_preference_quality_score", "actual_preference_quality_score", "raw_preference_quality_score"):
        derived.pop(name)
    assert proxy_return_from_metrics(derived, effective, preference=env.preference) == pytest.approx(
        metrics["training_cumulative_reward"], abs=1e-12
    )
    policy = HeuristicPolicy()
    while not env.task_done:
        env.step(policy.select_action(env), build_observation=False)
        current = env.metrics()
        assert current["training_cumulative_reward"] == pytest.approx(
            proxy_return_from_metrics(current, effective, preference=env.preference), abs=1e-10
        )
    assert env.metrics()["reward_objective_worker_load_variance"] == pytest.approx(
        env.metrics()["worker_load_variance"], abs=1e-10
    )
