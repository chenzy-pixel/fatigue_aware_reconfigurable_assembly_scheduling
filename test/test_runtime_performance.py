from dataclasses import replace

import numpy as np
import pytest
import torch

from agent.ppo import build_actor_critic
from environment import AssemblySchedulingEnv, DecisionType
from environment.state import ReconfigurationRuntime
from environment.time_context import project_wait_state
from environment.types import ReconfigurationStage, WorkerState


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("variant", ["objective_experts", "shared_preference"])
def test_grouped_diagnostics_match_reference_statistics_and_pair_ties(
    config, fixed_instance, device, variant
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    settings = {**config["network"], "actor_head_variant": variant}
    network = build_actor_critic(observation, settings).to(device)
    indices = [0, 2, 4, 5, 6]
    # Masked WAIT, WAIT only, pair only, mixed actions, and no pair candidates.
    masks = [
        [False, False, False, True], [False], [False, True],
        [False, False, False], [False],
    ]
    actions = [[2, 7, 11, 12], [20], [9, 20], [3, 13, 20], [20]]
    widths = [len(mask) for mask in masks]
    graph_ids = torch.tensor(np.repeat(indices, widths), device=device)
    mask = torch.tensor([v for row in masks for v in row], device=device)
    action_indices = torch.tensor([v for row in actions for v in row], device=device)
    waits = torch.tensor(np.cumsum(widths) - 1, device=device)
    phase_indices = torch.tensor(indices, device=device)
    preference = torch.tensor([
        [0.6, 0.3, 0.1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.2, 0.5, 0.3], [1 / 3, 1 / 3, 1 / 3], [0, 0, 1],
    ], device=device)
    size = mask.numel()
    direct = torch.stack((
        torch.linspace(-1, 1, size, device=device),
        torch.ones(size, device=device),
        torch.linspace(0, 1e-6, size, device=device),
    ), dim=-1)
    context = direct * 0.75
    experts = direct + context
    base = torch.tensor([0.8, 0.8, 0.2, float("nan"), 0, 0, 0.1, 0.3, 0.2, 0.4, 0], device=device)
    residual = torch.tensor([0, 0, 0.7, float("nan"), 0, 1e-12, 0, 0.2, 0, 0, 0], device=device)
    final = base + residual
    for phase in (DecisionType.PRODUCTION, DecisionType.WORKER):
        with torch.no_grad():
            network._latest_policy_decision_diagnostics.clear()
            start = 0
            for index, width in zip(indices, widths, strict=True):
                segment = slice(start, start + width)
                network._record_components(
                    phase, mask[segment], direct[segment], context[segment], experts[segment],
                    base[segment], residual[segment], final[segment], preference[index],
                    pair_action_indices=action_indices[segment][:-1],
                )
                start += width
            expected = network.consume_policy_decision_diagnostics()
            network._record_components_grouped(
                phase, indices, graph_ids, mask, direct, context, experts,
                base, residual, final, preference, action_indices=action_indices,
                wait_positions=waits, phase_indices=phase_indices,
            )
            actual = network.consume_policy_decision_diagnostics()
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert got.keys() == want.keys()
            for name, value in want.items():
                assert type(got[name]) is type(value)
                if isinstance(value, float):
                    assert got[name] == pytest.approx(value, abs=1e-6, rel=1e-6)
                else:
                    assert got[name] == value
        assert actual[0]["relative_top_action"] == 2
        assert actual[0]["final_pair_top_action"] == 11
        assert actual[0]["context_overrode_top"] is True
        assert actual[2]["relative_top_action"] == 9
        assert "relative_top_action" not in actual[1]
        assert "relative_top_action" not in actual[4]


def _recovery_environment(config, fixed_instance, *, stage=ReconfigurationStage.WAIT_INS):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    module = next(iter(env.machines[0].spec.module_parameters))
    parameters = env.machines[0].spec.module_parameters[module]
    env.machines[0].spec = replace(env.machines[0].spec, module_parameters={
        **env.machines[0].spec.module_parameters,
        module: replace(parameters, installation_base_time=1.0, disassembly_base_time=1.0),
    })
    env.instance = replace(env.instance, resolution=0.1, fatigue=replace(
        env.instance.fatigue, maximum_safe_fatigue=0.5,
        disassembly_time_coefficient=0.5, installation_time_coefficient=0.5,
        disassembly_accumulation_rate_per_minute=0.02,
        installation_accumulation_rate_per_minute=0.02,
        idle_recovery_rate_per_minute=0.1,
    ))
    env.current_tick = 10
    env.horizon_tick = 100
    for worker in env.workers:
        worker.state = WorkerState.DIS
    env.workers[0].state = WorkerState.IDLE
    env.workers[0].spec = replace(env.workers[0].spec, qualified_modules=(module,))
    env.workers[0].fatigue = 0.9
    env.reconfigurations = {
        "recovery_test": ReconfigurationRuntime(
            id="recovery_test", machine_id=env.machines[0].spec.id,
            operation_id=env.operations[0].spec.id, source_module=module,
            target_module=module, lock_tick=0, stage=stage,
        )
    }
    return env


def _linear_recovery_tick(env):
    # Exhaustive physical predicate oracle, independent of the search strategy.
    for tick in range(env.current_tick + 1, env.horizon_tick + 1):
        for recon in env.reconfigurations.values():
            if recon.stage not in {ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS}:
                continue
            module = recon.source_module if recon.stage == ReconfigurationStage.WAIT_DIS else recon.target_module
            for worker in env.workers:
                if (worker.state != WorkerState.IDLE or module not in worker.spec.qualified_modules
                        or env._worker_can_start(recon, worker)):
                    continue
                if env._safe_stage_projection_at_tick(
                    recon, worker, available_tick=env.current_tick,
                    available_fatigue=worker.fatigue,
                    recovery_rate=env.instance.fatigue.idle_recovery_rate_per_minute, tick=tick,
                )[0]:
                    return tick
    return None


@pytest.mark.parametrize("stage", [ReconfigurationStage.WAIT_DIS, ReconfigurationStage.WAIT_INS])
def test_binary_recovery_matches_exhaustive_tick_search(config, fixed_instance, stage):
    env = _recovery_environment(config, fixed_instance, stage=stage)
    rng = np.random.default_rng(117)
    for _ in range(32):
        env.workers[0].fatigue = float(rng.uniform(0, 1))
        env.horizon_tick = int(rng.integers(env.current_tick, 130))
        env.instance = replace(env.instance, fatigue=replace(
            env.instance.fatigue,
            idle_recovery_rate_per_minute=float(rng.choice([0.0, 0.01, 0.1, 0.5])),
            installation_time_coefficient=float(rng.choice([0.0, 0.5, 2.0])),
            disassembly_time_coefficient=float(rng.choice([0.0, 0.5, 2.0])),
        ))
        assert env._earliest_worker_pair_recovery_tick() == _linear_recovery_tick(env)


def test_recovery_horizon_boundary_eligibility_and_impossible_stage(config, fixed_instance):
    env = _recovery_environment(config, fixed_instance)
    expected = _linear_recovery_tick(env)
    assert expected is not None
    env.horizon_tick = expected - 1
    assert env._earliest_worker_pair_recovery_tick() is None
    env.horizon_tick = expected
    assert env._earliest_worker_pair_recovery_tick() == expected
    env.workers[0].state = WorkerState.INS
    assert env._earliest_worker_pair_recovery_tick() is None
    env.workers[0].state = WorkerState.IDLE
    env.workers[0].spec = replace(env.workers[0].spec, qualified_modules=())
    assert env._earliest_worker_pair_recovery_tick() is None
    env = _recovery_environment(config, fixed_instance)
    env.instance = replace(env.instance, fatigue=replace(
        env.instance.fatigue, installation_accumulation_rate_per_minute=1.0,
    ))
    assert env._earliest_worker_pair_recovery_tick() is None
    env.reconfigurations.clear()
    assert env._earliest_worker_pair_recovery_tick() is None


def test_recovery_selects_earliest_across_workers_and_waiting_stages(config, fixed_instance):
    env = _recovery_environment(config, fixed_instance)
    first = next(iter(env.reconfigurations.values()))
    env.reconfigurations["second"] = replace(
        first, id="second", stage=ReconfigurationStage.WAIT_DIS,
    )
    env.workers[1].state = WorkerState.IDLE
    env.workers[1].spec = replace(env.workers[1].spec, qualified_modules=(first.target_module,))
    env.horizon_tick = 150
    for first_fatigue, second_fatigue in ((0.9, 0.8), (0.7, 0.95), (1.0, 0.5), (0.5, 1.0)):
        env.workers[0].fatigue = first_fatigue
        env.workers[1].fatigue = second_fatigue
        for coefficient in (0.0, 1.0, 2.0):
            env.instance = replace(env.instance, fatigue=replace(
                env.instance.fatigue, disassembly_time_coefficient=coefficient,
            ))
            assert env._earliest_worker_pair_recovery_tick() == _linear_recovery_tick(env)


def test_recovery_search_uses_logarithmic_projection_count(config, fixed_instance, monkeypatch):
    env = _recovery_environment(config, fixed_instance)
    env.horizon_tick = env.current_tick + 100_000
    expected = _linear_recovery_tick(env)
    calls = []
    original = env._safe_stage_projection_at_tick

    def counted(*args, **kwargs):
        calls.append(kwargs["tick"])
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "_safe_stage_projection_at_tick", counted)
    assert env._earliest_worker_pair_recovery_tick() == expected
    assert len(calls) <= 19


def test_wait_certificate_cache_is_isolated_and_invalidated(config, fixed_instance, monkeypatch):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    env._invalidate_resource_snapshot()
    calls = []
    original = env._compute_wait_certificate

    def counted():
        calls.append(env.current_tick)
        return original()

    monkeypatch.setattr(env, "_compute_wait_certificate", counted)
    expected = env._wait_certificate()
    first = env._wait_certificate()
    first["reason"] = "caller_mutation"
    env._last_wait_certificate["allowed"] = not expected["allowed"]
    assert env._wait_certificate() == expected
    assert env._last_wait_certificate == expected
    assert len(calls) == 1
    for mutate in (
        env._invalidate_resource_snapshot,
        lambda: setattr(env, "current_tick", env.current_tick + 1),
        lambda: setattr(env, "horizon_tick", env.horizon_tick - 1),
        lambda: setattr(env, "decision_type", DecisionType.WORKER),
    ):
        mutate()
        actual = env._wait_certificate()
        assert actual == original()
        env._wait_certificate()
    assert len(calls) == 5
    env.reset(fixed_instance, build_observation=False)
    assert env._wait_certificate() == original()


def test_wait_projection_keeps_original_certificate_cache(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(fixed_instance, build_observation=False)
    certificate = env._wait_certificate()
    assert certificate["wait_ticks"] > 0
    key = env._wait_certificate_cache_key
    projected = project_wait_state(env, certificate["wait_ticks"])
    assert projected._wait_certificate() == projected._compute_wait_certificate()
    assert env._wait_certificate_cache_key == key
    assert env._wait_certificate() == certificate
    assert projected._wait_certificate_cache is not env._wait_certificate_cache
