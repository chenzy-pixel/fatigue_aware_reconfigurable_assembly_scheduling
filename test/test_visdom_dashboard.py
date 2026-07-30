from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from result.visdom_dashboard import (
    DisabledTrainingDashboard,
    TrainingDashboard,
    build_schedule_gantt_svg,
    create_training_dashboard,
    resolve_visdom_settings,
    visdom_environment_name,
)


class FakeVisdom:
    connected = True
    instances: list["FakeVisdom"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple, dict]] = []
        self.__class__.instances.append(self)

    def check_connection(self, timeout_seconds=0):
        self.calls.append(
            (
                "check_connection",
                (),
                {"timeout_seconds": timeout_seconds},
            )
        )
        return self.__class__.connected

    def _record(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return kwargs.get("win", True)

    def text(self, *args, **kwargs):
        return self._record("text", *args, **kwargs)

    def line(self, *args, **kwargs):
        return self._record("line", *args, **kwargs)

    def svg(self, *args, **kwargs):
        return self._record("svg", *args, **kwargs)

    def bar(self, *args, **kwargs):
        return self._record("bar", *args, **kwargs)

    def save(self, *args, **kwargs):
        return self._record("save", *args, **kwargs)


def _enabled_config(config):
    effective = deepcopy(config)
    effective["logging"]["visdom"]["enabled"] = True
    effective["logging"]["visdom"]["connection_timeout_seconds"] = 0
    return effective


def _phase_state():
    return {
        "phase": "feasibility",
        "consecutive_validation_successes": 0,
        "formal_training_status": "feasibility_not_reached",
    }


def _episode_row(reward=1.0):
    return {
        "reward": reward,
        "quality_score": 0.2,
        "completed_order_ratio": 0.5,
        "completed_operation_ratio": 0.4,
        "terminated": False,
        "truncated": False,
        "reward_flow": -1.0,
        "reward_cost": -2.0,
        "reward_variance": -3.0,
        "reward_completion_progress": 0.5,
        "reward_completion_bonus": 0.0,
        "reward_quality": -0.1,
        "flow_time_objective": 100.0,
        "reconfiguration_cost": 10.0,
        "worker_load_variance": 2.0,
        "maximum_worker_fatigue": 0.7,
        "mean_peak_worker_fatigue": 0.5,
        "fatigue_masked_action_ratio": 0.1,
        "safe_fatigue_limit": 0.9,
        "completed_reconfigurations": 2,
        "worker_switch_ratio": 0.5,
        "worker_competition_event_count": 1,
        "machine_waiting_for_worker_time": 3.0,
        "pressure_type": "balanced",
    }


def _update_row(update_id=1):
    return {
        "update_id": update_id,
        "episode_start": update_id - 1,
        "episode_end": update_id - 1,
        "transition_count": 8,
        "sampling_wall_time_seconds": 0.1,
        "policy_inference_time_seconds": 0.02,
        "generation_time_seconds": 0.01,
        "environment_step_time_seconds": 0.03,
        "ppo_update_time_seconds": 0.04,
        "transitions_per_second": 50.0,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "loss": 0.3,
        "entropy": 0.4,
        "approx_kl": 0.01,
        "clip_fraction": 0.1,
        "ratio_mean": 1.0,
        "gradient_norm": 0.4,
        "gradient_norm_max": 0.6,
        "gradient_clipped_fraction": 0.2,
        "pre_update_explained_variance": 0.3,
        "return_mean": 1.0,
        "return_std": 0.2,
        "advantage_mean": 0.1,
        "advantage_std": 0.3,
        "value_prediction_mean": 0.9,
        "value_prediction_std": 0.1,
        "learning_rate": 3e-4,
    }


def _trace():
    return {
        "instance_id": "sample<&>",
        "dataset": "validation",
        "instance_index": 0,
        "metrics": {},
        "worker_ids": ["W1"],
        "safe_fatigue_limit": 0.9,
        "worker_peak_fatigue": {"W1": 0.7},
        "fatigue_trace": [
            {"time": 0.0, "workers": {"W1": 0.1}},
            {"time": 5.0, "workers": {"W1": 0.7}},
        ],
        "schedule": [
            {
                "order_id": "O1",
                "operation_id": "OP<&>",
                "required_module": "A1",
                "machine_id": "M1",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
            }
        ],
        "reconfigurations": [
            {
                "reconfiguration_id": "R1",
                "operation_id": "OP2",
                "machine_id": "M1",
                "worker_id": "W1",
                "stage": "DIS",
                "source_module": "A1",
                "target_module": "A2",
                "start": 5.0,
                "end": 6.0,
                "duration": 1.0,
                "fixed_cost": 1.0,
            },
            {
                "reconfiguration_id": "R1",
                "operation_id": "OP2",
                "machine_id": "M1",
                "worker_id": "W1",
                "stage": "INS",
                "source_module": "A1",
                "target_module": "A2",
                "start": 6.0,
                "end": 7.0,
                "duration": 1.0,
                "fixed_cost": 1.0,
            },
        ],
    }


def test_disabled_dashboard_is_a_noop(config, tmp_path):
    dashboard = create_training_dashboard(
        config=config,
        run_directory=tmp_path,
        total_episodes=2,
    )
    assert isinstance(dashboard, DisabledTrainingDashboard)
    dashboard.log_update({}, [], {})
    dashboard.close()


def test_environment_names_are_isolated_and_legacy_flag_is_supported(
    config,
    tmp_path,
):
    first = tmp_path / "first run"
    second = tmp_path / "second run"
    assert visdom_environment_name(config, first) != (
        visdom_environment_name(config, second)
    )
    legacy = deepcopy(config)
    legacy["logging"] = {"visdom_enabled": True}
    assert resolve_visdom_settings(legacy)["enabled"] is True


def test_dashboard_creates_then_appends_stable_windows(config, tmp_path):
    FakeVisdom.instances.clear()
    FakeVisdom.connected = True
    dashboard = TrainingDashboard(
        config=_enabled_config(config),
        run_directory=tmp_path,
        total_episodes=2,
        visdom_class=FakeVisdom,
    )
    dashboard.log_update(
        _update_row(1),
        [_episode_row(1.0)],
        _phase_state(),
    )
    dashboard.log_update(
        _update_row(2),
        [_episode_row(2.0)],
        _phase_state(),
    )
    client = FakeVisdom.instances[-1]
    reward_calls = [
        call
        for call in client.calls
        if call[0] == "line"
        and call[2].get("win") == "10_training_effect"
    ]
    assert reward_calls
    assert reward_calls[0][2]["update"] is None
    assert any(call[2]["update"] == "append" for call in reward_calls[1:])
    assert client.kwargs["env"] == dashboard.environment
    assert client.kwargs["log_to_filename"].endswith(
        "visdom_events.log"
    )


def test_unreachable_server_switches_to_offline_events(config, tmp_path):
    FakeVisdom.instances.clear()
    FakeVisdom.connected = False
    with pytest.warns(RuntimeWarning, match="offline events"):
        dashboard = TrainingDashboard(
            config=_enabled_config(config),
            run_directory=tmp_path,
            total_episodes=2,
            visdom_class=FakeVisdom,
        )
    assert dashboard.connected is False
    assert FakeVisdom.instances[-1].kwargs["offline"] is True
    dashboard.log_update(
        _update_row(1),
        [_episode_row()],
        _phase_state(),
    )


def test_real_visdom_client_records_replayable_offline_events(
    config,
    tmp_path,
):
    pytest.importorskip("visdom")
    effective = _enabled_config(config)
    effective["logging"]["visdom"]["port"] = 65534
    with pytest.warns(RuntimeWarning, match="offline events"):
        dashboard = TrainingDashboard(
            config=effective,
            run_directory=tmp_path,
            total_episodes=2,
        )
    dashboard.log_update(
        _update_row(1),
        [_episode_row()],
        _phase_state(),
    )
    dashboard.log_diagnostic(_trace(), completed_episodes=1)
    dashboard.close()
    event_log = tmp_path / "visdom_events.log"
    assert event_log.exists()
    assert event_log.stat().st_size > 0
    assert (
        tmp_path / "diagnostics" / "validation_0000001.json"
    ).exists()


def test_gantt_and_diagnostic_include_resources_and_fatigue(
    config,
    tmp_path,
):
    svg = build_schedule_gantt_svg(_trace())
    assert "machine:M1" in svg
    assert "worker:W1" in svg
    assert "DIS:OP2" in svg
    assert "INS:OP2" in svg
    assert "OP&lt;&amp;&gt;" in svg
    assert "Time (minutes)" in svg

    FakeVisdom.instances.clear()
    FakeVisdom.connected = True
    dashboard = TrainingDashboard(
        config=_enabled_config(config),
        run_directory=tmp_path,
        total_episodes=2,
        visdom_class=FakeVisdom,
    )
    dashboard.log_diagnostic(_trace(), completed_episodes=2)
    saved = json.loads(
        (
            tmp_path / "diagnostics" / "validation_0000002.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["safe_fatigue_limit"] == pytest.approx(0.9)
    client = FakeVisdom.instances[-1]
    fatigue_calls = [
        call
        for call in client.calls
        if call[0] == "line"
        and call[2].get("win")
        == "71_representative_fatigue_trace"
    ]
    assert fatigue_calls
    values = fatigue_calls[-1][2]["Y"]
    assert np.all(values[:, -1] == pytest.approx(0.9))
