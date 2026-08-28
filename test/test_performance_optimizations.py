from __future__ import annotations

import math
import random

import numpy as np

from agent.baselines import HeuristicPolicy
from data.dataset import OnlineInstanceDataset, build_dataset_split
from data.generate_orders import InstanceGenerator
from environment import AssemblySchedulingEnv, CAPABLE_EDGE, DecisionType
from environment.env import ReconfigurationRuntime
from environment.types import ReconfigurationStage


def _metadata_without_counterfactual(metadata):
    return {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "reconfiguration_value_class",
            "counterfactual_candidate_count",
        }
    }


def _scalar_capability_features(environment):
    values = []
    cost_scale = float(environment.config["reward"]["cost_scale"])
    horizon_tick = environment.horizon_tick
    for operation_index, machine_index in environment._static_edge_indices[
        CAPABLE_EDGE
    ].T:
        operation_index = int(operation_index)
        machine_index = int(machine_index)
        operation = environment.operations[operation_index]
        machine = environment.machines[machine_index]
        profile = environment._production_candidate_profile(
            operation_index, machine_index
        )
        configuration_match = (
            machine.current_module == operation.spec.required_module
        )
        source_cost = environment.instance.module_costs.get(
            machine.current_module
        )
        fixed_disassembly_cost = (
            0.0
            if configuration_match or source_cost is None
            else source_cost.fixed_disassembly_cost
        )
        fixed_installation_cost = (
            0.0
            if configuration_match
            else environment.instance.module_costs[
                operation.spec.required_module
            ].fixed_installation_cost
        )
        labor_cost, downtime_cost = (
            environment._estimate_candidate_reconfiguration_costs(
                machine, operation.spec.required_module
            )
        )
        values.append(
            [
                min(
                    2.0,
                    max(
                        0.0,
                        environment.estimate_processing_ticks(
                            operation_index, machine_index
                        )
                        / horizon_tick,
                    ),
                ),
                float(configuration_match),
                min(
                    2.0,
                    max(
                        0.0,
                        environment.estimate_earliest_start_tick(
                            operation_index, machine_index
                        )
                        / horizon_tick,
                    ),
                ),
                min(2.0, max(0.0, profile.resource_ready_tick / horizon_tick)),
                min(
                    2.0,
                    max(0.0, profile.predicted_finish_tick / horizon_tick),
                ),
                profile.safe_disassembly_workers
                / max(1, len(environment.workers)),
                profile.safe_installation_workers
                / max(1, len(environment.workers)),
                profile.matching_deficit_after_commit
                / max(1, len(environment.workers)),
                max(
                    -1.0,
                    min(1.0, profile.horizon_slack_ticks / horizon_tick),
                ),
                environment.estimate_reconfiguration_ticks(
                    operation_index, machine_index
                )
                / horizon_tick,
                fixed_disassembly_cost / cost_scale,
                fixed_installation_cost / cost_scale,
                labor_cost / cost_scale,
                downtime_cost / cost_scale,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def test_online_generation_skips_counterfactual_without_changing_instance(
    config,
    fixed_instance,
    monkeypatch,
):
    calls = []

    def classify(self, instance):
        calls.append(instance.instance_id)
        return "positive", 7

    monkeypatch.setattr(
        InstanceGenerator, "_classify_reconfiguration_value", classify
    )
    generator = InstanceGenerator(
        fixed_instance, config["generator"], config=config
    )
    arguments = {
        "seed": 1_000_000,
        "split": "train",
        "pressure_type": "balanced",
    }
    complete = generator.generate(**arguments)
    fast = generator.generate(
        **arguments, classify_reconfiguration_value=False
    )

    assert complete.instance == fast.instance
    assert _metadata_without_counterfactual(
        complete.metadata
    ) == _metadata_without_counterfactual(fast.metadata)
    assert complete.metadata["reconfiguration_value_class"] == "positive"
    assert complete.metadata["counterfactual_candidate_count"] == 7
    assert fast.metadata["reconfiguration_value_class"] is None
    assert fast.metadata["counterfactual_candidate_count"] == 0
    assert len(calls) == 1

    online = OnlineInstanceDataset(
        config=config,
        template=fixed_instance,
        episode_count=1,
    )[0]
    assert len(calls) == 1
    assert online.metadata["reconfiguration_value_class"] is None
    assert online.metadata["counterfactual_candidate_count"] == 0


def test_persisted_dataset_generation_keeps_counterfactual_classification(
    config,
    fixed_instance,
    tmp_path,
    monkeypatch,
):
    calls = []

    def classify(self, instance):
        calls.append(instance.instance_id)
        return "mixed", 3

    monkeypatch.setattr(
        InstanceGenerator, "_classify_reconfiguration_value", classify
    )
    build_dataset_split(
        config=config,
        template=fixed_instance,
        split="validation",
        count=1,
        instances_root=tmp_path / "instances",
        manifests_root=tmp_path / "manifests",
    )
    assert len(calls) == 1


def test_build_observation_false_preserves_environment_trajectory(
    config,
    fixed_instance,
):
    observed = AssemblySchedulingEnv(config)
    state_only = AssemblySchedulingEnv(config)
    assert observed.reset(fixed_instance) is not None
    assert state_only.reset(fixed_instance, build_observation=False) is None
    policy = HeuristicPolicy()

    while not (observed.terminated or observed.truncated):
        assert observed.decision_type == state_only.decision_type
        assert np.array_equal(
            observed.get_action_mask(), state_only.get_action_mask()
        )
        action = policy.select_action(observed)
        _, observed_reward, observed_terminated, observed_truncated, observed_info = (
            observed.step(action)
        )
        state_observation, state_reward, state_terminated, state_truncated, state_info = (
            state_only.step(action, build_observation=False)
        )
        assert state_observation is None
        assert state_reward == observed_reward
        assert state_terminated == observed_terminated
        assert state_truncated == observed_truncated
        assert state_info == observed_info

    assert state_only.schedule_log == observed.schedule_log
    assert state_only.reconfiguration_log == observed.reconfiguration_log
    assert state_only.metrics() == observed.metrics()


def test_observation_cache_is_versioned_and_returns_isolated_arrays(
    config,
    fixed_instance,
    monkeypatch,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    relation_builds = 0
    resource_profile_calls = 0
    original_build = environment._build_graph_relations
    original_profile = environment._compute_production_resource_profile

    def counted_build():
        nonlocal relation_builds
        relation_builds += 1
        return original_build()

    def counted_profile(machine_index, target_module):
        nonlocal resource_profile_calls
        resource_profile_calls += 1
        return original_profile(machine_index, target_module)

    monkeypatch.setattr(environment, "_build_graph_relations", counted_build)
    monkeypatch.setattr(
        environment, "_compute_production_resource_profile", counted_profile
    )
    first = environment.observe()
    second = environment.observe()
    assert relation_builds == 1
    assert resource_profile_calls == len(
        environment._capability_unique_group_ids
    )
    assert resource_profile_calls <= (
        len(environment.machines) * len(environment.instance.modules)
    )
    assert not np.shares_memory(first.operations, second.operations)
    assert not np.shares_memory(
        first.relations[CAPABLE_EDGE].edge_features,
        second.relations[CAPABLE_EDGE].edge_features,
    )
    expected_operation_value = float(second.operations[0, 0])
    expected_edge_value = float(
        second.relations[CAPABLE_EDGE].edge_features[0, 0]
    )
    first.operations[0, 0] = 123.0
    first.relations[CAPABLE_EDGE].edge_features[0, 0] = 456.0
    third = environment.observe()
    assert third.operations[0, 0] == expected_operation_value
    assert (
        third.relations[CAPABLE_EDGE].edge_features[0, 0]
        == expected_edge_value
    )
    assert relation_builds == 1

    action = HeuristicPolicy().select_action(environment)
    environment.step(action, build_observation=False)
    environment.observe()
    assert relation_builds == 2


def test_action_mask_cache_is_versioned_and_returns_isolated_arrays(
    config,
    fixed_instance,
    monkeypatch,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    environment._invalidate_resource_snapshot()
    opportunity_calls = 0
    original_opportunity = environment._production_defer_opportunity

    def counted_opportunity():
        nonlocal opportunity_calls
        opportunity_calls += 1
        return original_opportunity()

    monkeypatch.setattr(
        environment,
        "_production_defer_opportunity",
        counted_opportunity,
    )
    first = environment.get_action_mask()
    first_call_count = opportunity_calls
    assert first_call_count > 0
    second = environment.get_action_mask()
    assert opportunity_calls == first_call_count
    assert np.array_equal(first, second)

    first[:] = ~first
    third = environment.get_action_mask()
    assert np.array_equal(third, second)

    environment._invalidate_resource_snapshot()
    environment.get_action_mask()
    assert opportunity_calls > first_call_count


def test_grouped_capability_features_match_scalar_reference_across_states(
    config,
    fixed_instance,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    policy = HeuristicPolicy()
    seen = {"reset"}

    def compare():
        observation = environment.observe()
        np.testing.assert_array_equal(
            observation.relations[CAPABLE_EDGE].edge_features,
            _scalar_capability_features(environment),
        )

    compare()
    for _ in range(2_000):
        if environment.terminated or environment.truncated:
            break
        phase = environment.decision_type
        before_tick = environment.current_tick
        action = policy.select_action(environment)
        environment.step(action, build_observation=False)
        tag = None
        if phase == DecisionType.PRODUCTION and action != (
            len(environment.operations) * len(environment.machines)
        ):
            tag = "production"
        elif phase == DecisionType.WORKER and action != (
            len(environment.machines) * len(environment.workers)
        ):
            tag = "worker"
        elif environment.current_tick > before_tick:
            tag = "event"
        if tag is not None and tag not in seen:
            compare()
            seen.add(tag)
    assert environment.terminated or environment.truncated
    compare()
    assert {"reset", "production", "worker", "event"} <= seen


def test_safe_stage_binary_search_matches_brute_force_and_is_logarithmic(
    config,
    fixed_instance,
    monkeypatch,
):
    environment = AssemblySchedulingEnv(config)
    environment.reset(fixed_instance, build_observation=False)
    rng = random.Random(20260808)
    eligible = [
        (machine_index, worker_index, module, installation)
        for machine_index, machine in enumerate(environment.machines)
        for worker_index, worker in enumerate(environment.workers)
        for module in environment.instance.modules
        if module in machine.spec.module_parameters
        and module in worker.spec.qualified_modules
        for installation in (False, True)
    ]

    def brute_force(
        machine_index,
        worker_index,
        module,
        installation,
        earliest_tick,
    ):
        worker = environment.workers[worker_index]
        available_tick, available_fatigue = (
            environment._worker_fatigue_at_availability(worker_index)
        )
        stage = (
            ReconfigurationStage.WAIT_INS
            if installation
            else ReconfigurationStage.WAIT_DIS
        )
        reconfiguration = ReconfigurationRuntime(
            id="brute-force",
            machine_id=environment.machines[machine_index].spec.id,
            operation_id="",
            source_module=(
                environment.instance.no_module_state
                if installation
                else module
            ),
            target_module=(
                module
                if installation
                else environment.instance.no_module_state
            ),
            lock_tick=environment.current_tick,
            stage=stage,
        )
        recovery_rate = (
            environment.instance.fatigue.idle_recovery_rate_per_minute
        )
        for tick in range(
            max(earliest_tick, available_tick),
            environment.horizon_tick + 1,
        ):
            safe, duration_ticks = environment._safe_stage_projection_at_tick(
                reconfiguration,
                worker,
                available_tick=available_tick,
                available_fatigue=available_fatigue,
                recovery_rate=recovery_rate,
                tick=tick,
            )
            if safe and tick + duration_ticks <= environment.horizon_tick:
                return tick, duration_ticks
        return None

    for _ in range(64):
        machine_index, worker_index, module, installation = rng.choice(eligible)
        environment.current_tick = rng.randrange(
            0, max(1, environment.horizon_tick - 20)
        )
        environment.workers[worker_index].fatigue = rng.random()
        earliest_tick = min(
            environment.horizon_tick,
            environment.current_tick + rng.randrange(0, 20),
        )
        environment._invalidate_resource_snapshot()
        expected = brute_force(
            machine_index,
            worker_index,
            module,
            installation,
            earliest_tick,
        )
        actual = environment._earliest_safe_stage_projection(
            machine_index,
            worker_index,
            module,
            installation=installation,
            earliest_tick=earliest_tick,
        )
        assert actual == expected

    environment.reset(fixed_instance, build_observation=False)
    machine_index, worker_index, module, installation = eligible[0]
    environment.workers[worker_index].fatigue = 1.0
    environment._invalidate_resource_snapshot()
    predicate_calls = 0
    original_predicate = environment._safe_stage_projection_at_tick

    def counted_predicate(*args, **kwargs):
        nonlocal predicate_calls
        predicate_calls += 1
        return original_predicate(*args, **kwargs)

    monkeypatch.setattr(
        environment, "_safe_stage_projection_at_tick", counted_predicate
    )
    environment._earliest_safe_stage_projection(
        machine_index,
        worker_index,
        module,
        installation=installation,
        earliest_tick=0,
    )
    assert predicate_calls <= math.ceil(
        math.log2(environment.horizon_tick + 1)
    ) + 3
