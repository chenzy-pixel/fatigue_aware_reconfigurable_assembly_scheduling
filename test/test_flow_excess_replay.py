from dataclasses import replace

import numpy as np
import pytest

from data.models import OperationSpec, OrderSpec
from environment import AssemblySchedulingEnv
from environment.time_context import project_wait_state
from scripts.replay_flow_excess import ProcessingInterval, Replay, evaluate_grid


def sample_replay(*, completed=True, resolution=1.0):
    # Fastest duration 10, chosen machine 20, truncated after 5 if failed.
    end = 20 if completed else 5
    op = ProcessingInterval("O", "R", 0, end, 20, 10, completed, 1.0)
    return Replay(np.array([0]), np.array([end]), [op], end, resolution,
                  0 if completed else 100, not completed)


def test_closed_form_is_monotone_and_completion_conversion_is_continuous():
    replay = sample_replay()
    state = replay.state(np.arange(21))
    assert np.diff(state["excess_proportional"]).min() >= 0
    np.testing.assert_allclose(state["excess_proportional"], np.arange(21)*0.5)
    assert state["credit_proportional"][-1] == state["credit_completed_only"][-1] == 10
    assert replay.exact_proportional_credit_at(20) == 10
    assert state["excess_completed_only"][-1] < state["excess_completed_only"][-2]


def test_repeated_zero_time_snapshots_do_not_change_excess():
    state = sample_replay().state(np.array([0, 2, 2, 5, 5, 20]))
    assert state["excess_proportional"][1] == state["excess_proportional"][2]
    assert state["excess_proportional"][3] == state["excess_proportional"][4]


def test_fastest_processing_adds_no_excess_but_waiting_does():
    op = ProcessingInterval("O", "R", 4, 9, 5, 5, True, 1.0)
    replay = Replay(np.array([0]), np.array([9]), [op], 9, 1.0, 0, False)
    state = replay.state(np.arange(10))
    np.testing.assert_allclose(state["excess_proportional"], np.minimum(np.arange(10), 4))


@pytest.mark.parametrize("scale", [1089.15, 368.31428571428575])
def test_success_returns_equal_and_reward_identity_holds(scale):
    frame, rows = evaluate_grid(sample_replay(), np.arange(21), {"test": scale}, {})
    np.testing.assert_allclose([row["return"] for row in rows], rows[0]["return"], atol=1e-12)
    assert max(row["identity_error"] for row in rows) < 1e-12
    assert frame["quality_reward_test_completed_only"].max() > 0
    assert frame["quality_reward_test_proportional"].max() == 0


def test_failure_credit_uses_planned_duration_and_has_terminal_jump():
    replay = sample_replay(completed=False)
    frame, rows = evaluate_grid(replay, np.arange(6), {"test": 10.0}, {})
    assert frame["credit_proportional"].iloc[-1] == 2.5  # 10 * 5 / 20, not 10 * 5 / 5.
    assert frame["credit_front_loaded"].iloc[-1] == 5
    assert frame["excess_proportional"].iloc[-1] == 102.5
    assert frame.time.iloc[-1] == frame.time.iloc[-2]
    assert frame.excess_proportional.iloc[-1]-frame.excess_proportional.iloc[-2] == 100
    assert rows[0]["return"] != rows[1]["return"]
    assert max(row["identity_error"] for row in rows) < 1e-12


def terminal_wait_instance(template):
    parameter = template.machines[0].module_parameters["A1"]
    machine = replace(template.machines[0], initial_module="A1",
                      module_parameters={"A1": replace(parameter, processing_speed_factor=1.0)})
    orders = tuple(OrderSpec(f"R{i}", "W1", float(release),
                            (OperationSpec(f"O{i}", f"R{i}", 1, "A1", 8.0),))
                   for i, release in enumerate((0, 10)))
    return replace(template, instance_id="terminal_wait_replay", horizon=10.0,
                   machines=(machine,), orders=orders,
                   waves={"W1": {"dominant_module": "A1", "order_ids": [order.id for order in orders]}})


def native_excess(env):
    credit = 0.0
    for oi, op in enumerate(env.operations):
        minimum = min(env.estimate_processing_ticks(oi,mi) for mi,machine in enumerate(env.machines)
                      if op.spec.required_module in machine.spec.module_parameters)
        if op.state.value == "DONE":
            credit += minimum
        elif op.state.value == "PROCESSING":
            planned = env.estimate_processing_ticks(oi,env.instance.machine_index[op.machine_id])
            credit += minimum*(env.current_tick-op.start_tick)/planned
    return env._flow_integral+env._flow_penalty-credit*env.resolution


def test_wait_projection_needs_terminal_resolution_for_exact_penalty(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    env.reset(terminal_wait_instance(fixed_instance), build_observation=False)
    before_start = native_excess(env)
    env.step(env.encode_production_action(0, 0), build_observation=False)
    assert native_excess(env) == before_start == 0  # Zero-time processing start.
    certificate = env._wait_certificate()
    projected = project_wait_state(env, certificate["wait_ticks"])
    predicted_delta = native_excess(projected)-native_excess(env)
    before_wait = native_excess(env)
    env.step(env.wait_action, build_observation=False)
    assert predicted_delta == native_excess(env)-before_wait == 0
    assert projected._flow_integral == env._flow_integral  # Ordinary WAIT agrees.
    assert not env.task_done and env.current_time == 8.0
    certificate = env._wait_certificate()
    projected = project_wait_state(env, certificate["wait_ticks"])
    assert projected._flow_penalty == 0  # Current helper does not settle terminal failure.
    raw_projected_delta = native_excess(projected)-native_excess(env)
    projected._resolve_terminal_or_deadlock()
    predicted_delta = native_excess(projected)-native_excess(env)
    before_wait = native_excess(env)
    env.step(env.wait_action, build_observation=False)
    assert env.task_failed and projected.task_failed
    assert projected._flow_integral == env._flow_integral
    assert projected._flow_penalty == env._flow_penalty > 0
    assert predicted_delta == native_excess(env)-before_wait
    assert raw_projected_delta != predicted_delta
