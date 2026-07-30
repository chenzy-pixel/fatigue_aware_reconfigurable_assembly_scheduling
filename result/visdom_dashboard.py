from __future__ import annotations

import html
import json
import math
import re
import socket
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from result.io import write_json


DEFAULT_VISDOM_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "server": "http://localhost",
    "port": 8097,
    "base_url": "/",
    "env_prefix": "fatigue_assembly",
    "connection_timeout_seconds": 2.0,
    "fail_fast": False,
    "update_every": 1,
    "rolling_window_updates": 20,
    "diagnostic_every_validations": 5,
    "representative_instance_index": 0,
    "save_environment": True,
}


def resolve_visdom_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the nested configuration and the legacy enabled flag."""
    settings = dict(DEFAULT_VISDOM_SETTINGS)
    logging_config = config.get("logging", {})
    if not isinstance(logging_config, Mapping):
        raise TypeError("logging configuration must be a mapping")
    nested = logging_config.get("visdom")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("logging.visdom must be a mapping")
        settings.update(nested)
    elif "visdom_enabled" in logging_config:
        settings["enabled"] = bool(logging_config["visdom_enabled"])
    if int(settings["port"]) < 1:
        raise ValueError("logging.visdom.port must be positive")
    if int(settings["update_every"]) < 1:
        raise ValueError("logging.visdom.update_every must be positive")
    if int(settings["rolling_window_updates"]) < 1:
        raise ValueError(
            "logging.visdom.rolling_window_updates must be positive"
        )
    if int(settings["diagnostic_every_validations"]) < 1:
        raise ValueError(
            "logging.visdom.diagnostic_every_validations must be positive"
        )
    if int(settings["representative_instance_index"]) < 0:
        raise ValueError(
            "logging.visdom.representative_instance_index cannot be negative"
        )
    return settings


def override_visdom_enabled(
    config: dict[str, Any],
    enabled: bool | None,
) -> None:
    """Apply a CLI override while preserving the resolved configuration."""
    if enabled is None:
        return
    logging_config = config.setdefault("logging", {})
    nested = logging_config.setdefault("visdom", {})
    nested["enabled"] = bool(enabled)


def _safe_environment_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "run"


def visdom_environment_name(
    config: Mapping[str, Any],
    run_directory: str | Path,
) -> str:
    settings = resolve_visdom_settings(config)
    prefix = _safe_environment_component(str(settings["env_prefix"]))
    run_name = _safe_environment_component(Path(run_directory).name)
    seed = int(config["seed"])
    return f"{prefix}_{run_name}_seed{seed}"


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    return float(np.mean(values)) if values else None


def _std(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    return float(np.std(values)) if values else None


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if field in row]
    return float(np.mean(values)) if values else None


class DisabledTrainingDashboard:
    enabled = False
    connected = False
    environment = None
    validation_count = 0

    def log_update(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_validation(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_capture_diagnostic(self, **kwargs: Any) -> bool:
        return False

    def log_diagnostic(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class TrainingDashboard:
    """Main-process Visdom writer with live/offline event logging."""

    enabled = True

    def __init__(
        self,
        *,
        config: dict[str, Any],
        run_directory: str | Path,
        total_episodes: int,
        visdom_class: type | None = None,
    ) -> None:
        self.config = config
        self.run_directory = Path(run_directory)
        self.total_episodes = int(total_episodes)
        self.settings = resolve_visdom_settings(config)
        self.environment = visdom_environment_name(
            config,
            self.run_directory,
        )
        self.event_log = self.run_directory / "visdom_events.log"
        self.fail_fast = bool(self.settings["fail_fast"])
        self.validation_count = 0
        self._event_messages: list[str] = []
        self._known_windows: set[str] = set()
        self._reward_history: deque[float] = deque(
            maxlen=int(self.settings["rolling_window_updates"])
        )
        self._injected_visdom_class = visdom_class is not None
        self._visdom_class = visdom_class or self._load_visdom_class()
        self.connected = self._probe_connection()
        if not self.connected and self.fail_fast:
            raise ConnectionError(
                "Visdom server is unavailable at "
                f"{self.settings['server']}:{self.settings['port']}"
            )
        if not self.connected:
            warnings.warn(
                "Visdom server is unavailable; recording offline events to "
                f"{self.event_log}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.client = self._new_client(offline=not self.connected)
        self._write_metadata()
        self.log_event(
            "dashboard connected"
            if self.connected
            else "dashboard is in offline replay mode"
        )

    @staticmethod
    def _load_visdom_class() -> type:
        try:
            from visdom import Visdom
        except ImportError as error:
            raise RuntimeError(
                "Visdom logging is enabled but visdom is not installed; "
                "install requirements.txt or run `pip install visdom==0.2.4`"
            ) from error
        return Visdom

    def _client_arguments(self) -> dict[str, Any]:
        return {
            "server": str(self.settings["server"]),
            "port": int(self.settings["port"]),
            "base_url": str(self.settings["base_url"]),
            "env": self.environment,
            "use_incoming_socket": False,
        }

    def _probe_connection(self) -> bool:
        if not self._injected_visdom_class:
            server = str(self.settings["server"])
            parsed = urlparse(
                server if "://" in server else f"http://{server}"
            )
            host = parsed.hostname or "localhost"
            try:
                with socket.create_connection(
                    (host, int(self.settings["port"])),
                    timeout=max(
                        0.05,
                        float(
                            self.settings[
                                "connection_timeout_seconds"
                            ]
                        ),
                    ),
                ):
                    return True
            except OSError:
                return False
        probe = self._visdom_class(
            **self._client_arguments(),
            raise_exceptions=False,
        )
        try:
            return bool(
                probe.check_connection(
                    timeout_seconds=float(
                        self.settings["connection_timeout_seconds"]
                    )
                )
            )
        except Exception:
            return False

    def _new_client(self, *, offline: bool) -> Any:
        return self._visdom_class(
            **self._client_arguments(),
            raise_exceptions=True,
            log_to_filename=str(self.event_log),
            offline=offline,
        )

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except Exception as error:
            if self.fail_fast:
                raise
            if self.connected:
                warnings.warn(
                    "Visdom connection failed during training; subsequent "
                    f"events will be written offline ({error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.connected = False
                self.client = self._new_client(offline=True)
                return getattr(self.client, method)(*args, **kwargs)
            warnings.warn(
                f"Visdom offline event logging failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _write_metadata(self) -> None:
        ppo = self.config["ppo"]
        network = self.config["network"]
        reward = self.config["reward"]
        training = self.config["training"]
        summary = {
            "run_directory": str(self.run_directory),
            "environment": self.environment,
            "seed": self.config["seed"],
            "device": self.config["device"],
            "encoder_type": network.get("encoder_type", "typed_mlp"),
            "hidden_dim": network["hidden_dim"],
            "message_passing_layers": network.get(
                "message_passing_layers"
            ),
            "learning_rate": ppo["learning_rate"],
            "gamma": ppo["gamma"],
            "gae_lambda": ppo["gae_lambda"],
            "clip_epsilon": ppo["clip_epsilon"],
            "ppo_epochs": ppo["epochs"],
            "batch_size": ppo["batch_size"],
            "entropy_coefficient": ppo["entropy_coefficient"],
            "value_coefficient": ppo["value_coefficient"],
            "max_grad_norm": ppo["max_grad_norm"],
            "reward_mode": reward.get("mode", "legacy_weighted_sum"),
            "quality_weights": reward.get("quality_weights"),
            "quality_budget": reward.get("quality_budget"),
            "parallel_envs": training.get("parallel_envs"),
            "validation_split": training.get("validation_split"),
            "validation_interval_episodes": training.get(
                "validation_interval_episodes"
            ),
        }
        metadata_html = (
            "<h3>Run metadata</h3><pre>"
            + html.escape(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )
            + "</pre>"
            + "<h3>Read-only tuning map</h3>"
            + "<ul>"
            + "<li>KL / clip fraction → learning rate, clip epsilon, epochs</li>"
            + "<li>Entropy → entropy coefficient</li>"
            + "<li>Value loss / explained variance → critic and value coefficient</li>"
            + "<li>Gradient clipping → learning rate, batch size, reward scale</li>"
            + "<li>Fatigue / reconfiguration → curriculum, quality weights, policy structure</li>"
            + "</ul>"
        )
        self._invoke(
            "text",
            metadata_html,
            win="00_run_metadata",
            opts={"title": "00 Run metadata and tuning map"},
        )

    def _line(
        self,
        *,
        win: str,
        title: str,
        x: int | float,
        series: Mapping[str, Any],
        xlabel: str = "Completed episodes",
        ylabel: str | None = None,
    ) -> None:
        finite_series = [
            (name, value)
            for name, raw in series.items()
            if (value := _finite_number(raw)) is not None
        ]
        if not finite_series:
            return
        exists = win in self._known_windows
        for index, (name, value) in enumerate(finite_series):
            update = "append" if exists or index > 0 else None
            options = None
            if update is None:
                options = {
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel or title,
                    "showlegend": True,
                }
            self._invoke(
                "line",
                X=np.asarray([float(x)], dtype=np.float64),
                Y=np.asarray([value], dtype=np.float64),
                win=win,
                name=name,
                update=update,
                opts=options,
            )
        self._known_windows.add(win)

    def _status_text(
        self,
        phase_state: Mapping[str, Any],
        *,
        completed_episodes: int,
    ) -> None:
        status = {
            "completed_episodes": int(completed_episodes),
            "total_episodes": self.total_episodes,
            **dict(phase_state),
        }
        self._invoke(
            "text",
            "<h3>Current training state</h3><pre>"
            + html.escape(json.dumps(status, ensure_ascii=False, indent=2))
            + "</pre>",
            win="01_training_state",
            opts={"title": "01 Current training state"},
        )

    def log_update(
        self,
        update_row: dict[str, Any],
        episode_rows: list[dict[str, Any]],
        phase_state: Mapping[str, Any],
    ) -> None:
        update_id = int(update_row["update_id"])
        if update_id % int(self.settings["update_every"]) != 0:
            return
        completed_episodes = int(update_row["episode_end"]) + 1
        reward_mean = _mean(episode_rows, "reward")
        if reward_mean is not None:
            self._reward_history.append(reward_mean)
        rolling_reward = (
            float(np.mean(self._reward_history))
            if self._reward_history
            else None
        )
        self._status_text(
            phase_state,
            completed_episodes=completed_episodes,
        )
        self._line(
            win="10_training_effect",
            title="10 Training reward and quality",
            x=completed_episodes,
            series={
                "reward_mean": reward_mean,
                "reward_std": _std(episode_rows, "reward"),
                "reward_rolling_mean": rolling_reward,
                "quality_score": _mean(episode_rows, "quality_score"),
            },
        )
        self._line(
            win="11_training_completion",
            title="11 Training completion",
            x=completed_episodes,
            series={
                "completed_order_ratio": _mean(
                    episode_rows, "completed_order_ratio"
                ),
                "completed_operation_ratio": _mean(
                    episode_rows, "completed_operation_ratio"
                ),
                "terminated_rate": _rate(episode_rows, "terminated"),
                "truncated_rate": _rate(episode_rows, "truncated"),
            },
        )
        self._line(
            win="12_reward_components",
            title="12 Reward components",
            x=completed_episodes,
            series={
                name: _mean(episode_rows, f"reward_{name}")
                for name in (
                    "flow",
                    "cost",
                    "variance",
                    "completion_progress",
                    "completion_bonus",
                    "quality",
                )
            },
        )
        self._line(
            win="13_training_objectives",
            title="13 Training objectives",
            x=completed_episodes,
            series={
                "flow_time_objective": _mean(
                    episode_rows, "flow_time_objective"
                ),
                "reconfiguration_cost": _mean(
                    episode_rows, "reconfiguration_cost"
                ),
                "worker_load_variance": _mean(
                    episode_rows, "worker_load_variance"
                ),
            },
        )
        self._line(
            win="20_ppo_losses",
            title="20 PPO losses",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "policy_loss",
                    "value_loss",
                    "loss",
                    "entropy",
                )
            },
        )
        self._line(
            win="21_ppo_policy_health",
            title="21 PPO policy update health",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "approx_kl",
                    "clip_fraction",
                    "ratio_mean",
                    "learning_rate",
                )
            },
        )
        self._line(
            win="22_ppo_gradient_health",
            title="22 PPO gradient and critic health",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "gradient_norm",
                    "gradient_norm_max",
                    "gradient_clipped_fraction",
                    "pre_update_explained_variance",
                )
            },
        )
        self._line(
            win="23_ppo_value_scales",
            title="23 PPO return, advantage and value scales",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "return_mean",
                    "return_std",
                    "advantage_mean",
                    "advantage_std",
                    "value_prediction_mean",
                    "value_prediction_std",
                )
            },
        )
        self._line(
            win="30_fatigue_training",
            title="30 Training fatigue",
            x=completed_episodes,
            series={
                "maximum_worker_fatigue": _mean(
                    episode_rows, "maximum_worker_fatigue"
                ),
                "mean_peak_worker_fatigue": _mean(
                    episode_rows, "mean_peak_worker_fatigue"
                ),
                "fatigue_masked_action_ratio": _mean(
                    episode_rows, "fatigue_masked_action_ratio"
                ),
                "safe_fatigue_limit": _mean(
                    episode_rows, "safe_fatigue_limit"
                ),
            },
        )
        self._line(
            win="31_reconfiguration_training",
            title="31 Training reconfiguration and worker pressure",
            x=completed_episodes,
            series={
                "completed_reconfigurations": _mean(
                    episode_rows, "completed_reconfigurations"
                ),
                "worker_switch_ratio": _mean(
                    episode_rows, "worker_switch_ratio"
                ),
                "worker_competition_event_count": _mean(
                    episode_rows, "worker_competition_event_count"
                ),
                "machine_waiting_for_worker_time": _mean(
                    episode_rows, "machine_waiting_for_worker_time"
                ),
            },
        )
        self._line(
            win="40_training_throughput",
            title="40 Training throughput",
            x=completed_episodes,
            series={
                "transitions_per_second": update_row.get(
                    "transitions_per_second"
                ),
                "transition_count": update_row.get("transition_count"),
            },
        )
        self._line(
            win="41_training_times",
            title="41 Training wall-clock components",
            x=completed_episodes,
            series={
                name: update_row.get(name)
                for name in (
                    "sampling_wall_time_seconds",
                    "policy_inference_time_seconds",
                    "ppo_update_time_seconds",
                    "generation_time_seconds",
                    "environment_step_time_seconds",
                )
            },
        )
        profiles = sorted(
            {
                str(row["pressure_type"])
                for row in episode_rows
                if row.get("pressure_type") is not None
            }
        )
        self._line(
            win="50_pressure_completion",
            title="50 Completion by pressure profile",
            x=completed_episodes,
            series={
                profile: _mean(
                    [
                        row
                        for row in episode_rows
                        if str(row.get("pressure_type")) == profile
                    ],
                    "completed_order_ratio",
                )
                for profile in profiles
            },
        )
        self._line(
            win="51_pressure_quality",
            title="51 Quality by pressure profile",
            x=completed_episodes,
            series={
                profile: _mean(
                    [
                        row
                        for row in episode_rows
                        if str(row.get("pressure_type")) == profile
                    ],
                    "quality_score",
                )
                for profile in profiles
            },
        )

    def log_validation(
        self,
        validation_row: dict[str, Any],
        *,
        best_validation: Mapping[str, Any] | None,
        phase_state: Mapping[str, Any],
    ) -> None:
        self.validation_count += 1
        completed_episodes = int(validation_row["episode"])
        self._status_text(
            phase_state,
            completed_episodes=completed_episodes,
        )
        self._line(
            win="60_validation_feasibility",
            title="60 Validation feasibility",
            x=completed_episodes,
            series={
                "completion_rate": validation_row.get("completion_rate"),
                "truncated_count": validation_row.get("truncated_count"),
                "schedule_violation_count": validation_row.get(
                    "schedule_violation_count"
                ),
            },
        )
        self._line(
            win="61_validation_flow",
            title="61 Validation flow objectives",
            x=completed_episodes,
            series={
                "mean_makespan": validation_row.get("mean_makespan"),
                "mean_total_flow_time": validation_row.get(
                    "mean_total_flow_time"
                ),
                "mean_flow_time_objective": validation_row.get(
                    "mean_flow_time_objective"
                ),
                "best_makespan": (
                    best_validation.get("mean_makespan")
                    if best_validation
                    else None
                ),
                "best_total_flow_time": (
                    best_validation.get("mean_total_flow_time")
                    if best_validation
                    else None
                ),
                "best_flow_time_objective": (
                    best_validation.get("mean_flow_time_objective")
                    if best_validation
                    else None
                ),
            },
        )
        self._line(
            win="62_validation_quality_components",
            title="62 Validation quality components",
            x=completed_episodes,
            series={
                "mean_reconfiguration_cost": validation_row.get(
                    "mean_reconfiguration_cost"
                ),
                "mean_worker_load_variance": validation_row.get(
                    "mean_worker_load_variance"
                ),
                "best_reconfiguration_cost": (
                    best_validation.get("mean_reconfiguration_cost")
                    if best_validation
                    else None
                ),
                "best_worker_load_variance": (
                    best_validation.get("mean_worker_load_variance")
                    if best_validation
                    else None
                ),
            },
        )
        self._line(
            win="63_validation_gaps",
            title="63 Validation gaps to heuristic (%)",
            x=completed_episodes,
            series={
                "flow_gap": validation_row.get(
                    "mean_relative_heuristic_gap_percent"
                ),
                "makespan_gap": validation_row.get(
                    "mean_makespan_heuristic_gap_percent"
                ),
                "reconfiguration_cost_gap": validation_row.get(
                    "mean_reconfiguration_cost_heuristic_gap_percent"
                ),
                "worker_load_variance_gap": validation_row.get(
                    "mean_worker_load_variance_heuristic_gap_percent"
                ),
            },
            ylabel="Gap to heuristic (%)",
        )
        self._line(
            win="64_validation_fatigue",
            title="64 Validation fatigue",
            x=completed_episodes,
            series={
                name: validation_row.get(f"mean_{name}")
                for name in (
                    "maximum_worker_fatigue",
                    "mean_peak_worker_fatigue",
                    "safe_fatigue_limit",
                    "fatigue_masked_action_ratio",
                )
            },
        )
        self._line(
            win="65_validation_reconfiguration",
            title="65 Validation reconfiguration and worker pressure",
            x=completed_episodes,
            series={
                name: validation_row.get(f"mean_{name}")
                for name in (
                    "worker_competition_event_count",
                    "machine_waiting_for_worker_time",
                    "completed_reconfigurations",
                    "worker_switch_ratio",
                )
            },
        )

    def log_event(self, message: str) -> None:
        self._event_messages.append(str(message))
        rendered = "<br>".join(
            html.escape(value) for value in self._event_messages[-200:]
        )
        self._invoke(
            "text",
            rendered,
            win="02_training_events",
            opts={"title": "02 Training events"},
        )

    def should_capture_diagnostic(
        self,
        *,
        validation_event: str,
        is_new_best: bool,
    ) -> bool:
        interval = int(
            self.settings["diagnostic_every_validations"]
        )
        return (
            self.validation_count % interval == 0
            or validation_event in {"transition", "rejected"}
            or bool(is_new_best)
        )

    def log_diagnostic(
        self,
        trace: dict[str, Any],
        *,
        completed_episodes: int,
    ) -> None:
        output_directory = self.run_directory / "diagnostics"
        output_directory.mkdir(parents=True, exist_ok=True)
        output = (
            output_directory
            / f"validation_{int(completed_episodes):07d}.json"
        )
        write_json(output, trace)
        svg = build_schedule_gantt_svg(trace)
        self._invoke(
            "svg",
            svg,
            win="70_representative_schedule",
            opts={"title": "70 Representative schedule"},
        )
        fatigue_trace = trace.get("fatigue_trace", [])
        worker_names = list(trace.get("worker_ids", []))
        if fatigue_trace and worker_names:
            times = np.asarray(
                [row["time"] for row in fatigue_trace],
                dtype=np.float64,
            )
            values = np.asarray(
                [
                    [row["workers"][name] for name in worker_names]
                    for row in fatigue_trace
                ],
                dtype=np.float64,
            )
            limit = float(trace["safe_fatigue_limit"])
            values = np.column_stack(
                [values, np.full(len(times), limit, dtype=np.float64)]
            )
            names = [*worker_names, "safe_limit"]
            self._invoke(
                "line",
                X=np.tile(times[:, None], (1, len(names))),
                Y=values,
                win="71_representative_fatigue_trace",
                opts={
                    "title": "71 Representative worker fatigue trace",
                    "xlabel": "Schedule time (minutes)",
                    "ylabel": "Fatigue",
                    "legend": names,
                    "showlegend": True,
                },
            )
            self._known_windows.add(
                "71_representative_fatigue_trace"
            )
        peaks = trace.get("worker_peak_fatigue", {})
        if peaks:
            names = list(peaks)
            values = [float(peaks[name]) for name in names]
            if len(values) == 1:
                # Visdom 0.2.4 squeezes a single bar to a scalar.
                names.append("")
                values.append(0.0)
            self._invoke(
                "bar",
                X=np.asarray(values, dtype=np.float64),
                win="72_representative_peak_fatigue",
                opts={
                    "title": "72 Representative peak worker fatigue",
                    "rownames": names,
                    "ylabel": "Peak fatigue",
                },
            )
        self.log_event(
            "representative diagnostic saved at episode "
            f"{completed_episodes}: {output.name}"
        )

    def close(self) -> None:
        if bool(self.settings["save_environment"]):
            self._invoke("save", [self.environment])


def create_training_dashboard(
    *,
    config: dict[str, Any],
    run_directory: str | Path,
    total_episodes: int,
    visdom_class: type | None = None,
) -> TrainingDashboard | DisabledTrainingDashboard:
    settings = resolve_visdom_settings(config)
    if not bool(settings["enabled"]):
        return DisabledTrainingDashboard()
    return TrainingDashboard(
        config=config,
        run_directory=run_directory,
        total_episodes=total_episodes,
        visdom_class=visdom_class,
    )


def build_schedule_gantt_svg(trace: Mapping[str, Any]) -> str:
    """Render machine and worker use without adding a plotting dependency."""
    schedule = list(trace.get("schedule", []))
    reconfigurations = list(trace.get("reconfigurations", []))
    machine_ids = sorted(
        {
            str(row["machine_id"])
            for row in [*schedule, *reconfigurations]
        }
    )
    worker_ids = sorted(
        {str(row["worker_id"]) for row in reconfigurations}
        | {str(value) for value in trace.get("worker_ids", [])}
    )
    rows = [
        *(("machine", value) for value in machine_ids),
        *(("worker", value) for value in worker_ids),
    ]
    width = 1200
    left = 150
    right = 30
    top = 54
    row_height = 30
    axis_height = 52
    height = top + max(1, len(rows)) * row_height + axis_height
    plot_width = width - left - right
    maximum_time = max(
        [
            _finite_number(row.get("end")) or 0.0
            for row in [*schedule, *reconfigurations]
        ]
        or [1.0]
    )
    maximum_time = max(maximum_time, 1.0)
    row_index = {value: index for index, value in enumerate(rows)}
    module_palette = (
        "#4C78A8",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#EDC948",
    )
    modules = sorted(
        {str(row.get("required_module", "")) for row in schedule}
    )
    module_color = {
        module: module_palette[index % len(module_palette)]
        for index, module in enumerate(modules)
    }

    def x_position(value: Any) -> float:
        number = _finite_number(value) or 0.0
        return left + plot_width * number / maximum_time

    def rect(
        *,
        row_key: tuple[str, str],
        start: Any,
        end: Any,
        color: str,
        label: str,
        opacity: float = 1.0,
    ) -> str:
        y = top + row_index[row_key] * row_height + 5
        x = x_position(start)
        rectangle_width = max(1.0, x_position(end) - x)
        safe_label = html.escape(label)
        return (
            f'<rect x="{x:.2f}" y="{y}" width="{rectangle_width:.2f}" '
            f'height="20" rx="2" fill="{color}" opacity="{opacity}">'
            f"<title>{safe_label}</title></rect>"
            f'<text x="{x + 3:.2f}" y="{y + 14}" font-size="10" '
            f'fill="white">{safe_label}</text>'
        )

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="18" font-weight="bold">'
        + html.escape(
            f"Representative schedule: {trace.get('instance_id', '')}"
        )
        + "</text>",
    ]
    for index, (kind, identifier) in enumerate(rows):
        y = top + index * row_height
        fill = "#F7F7F7" if index % 2 == 0 else "#FFFFFF"
        content.append(
            f'<rect x="0" y="{y}" width="{width}" '
            f'height="{row_height}" fill="{fill}"/>'
        )
        content.append(
            f'<text x="8" y="{y + 20}" font-size="12">'
            f"{html.escape(kind)}:{html.escape(identifier)}</text>"
        )
    for row in schedule:
        module = str(row.get("required_module", ""))
        content.append(
            rect(
                row_key=("machine", str(row["machine_id"])),
                start=row["start"],
                end=row["end"],
                color=module_color.get(module, "#4C78A8"),
                label=f"{row['operation_id']} ({module})",
            )
        )
    stage_colors = {"DIS": "#B279A2", "INS": "#FF9DA7"}
    for row in reconfigurations:
        stage = str(row["stage"])
        color = stage_colors.get(stage, "#9D755D")
        label = f"{stage}:{row['operation_id']}"
        content.append(
            rect(
                row_key=("machine", str(row["machine_id"])),
                start=row["start"],
                end=row["end"],
                color=color,
                label=label,
                opacity=0.9,
            )
        )
        content.append(
            rect(
                row_key=("worker", str(row["worker_id"])),
                start=row["start"],
                end=row["end"],
                color=color,
                label=label,
                opacity=0.9,
            )
        )
    axis_y = top + max(1, len(rows)) * row_height
    content.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" '
        f'y2="{axis_y}" stroke="#333"/>'
    )
    for index in range(6):
        value = maximum_time * index / 5
        x = x_position(value)
        content.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" '
            f'y2="{axis_y + 6}" stroke="#333"/>'
        )
        content.append(
            f'<text x="{x:.2f}" y="{axis_y + 22}" font-size="11" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    content.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{axis_y + 42}" '
        'font-size="12" text-anchor="middle">Time (minutes)</text>'
    )
    legend_x = 540
    for index, (label, color) in enumerate(
        [*(module_color.items()), *stage_colors.items()]
    ):
        x = legend_x + index * 105
        content.append(
            f'<rect x="{x}" y="15" width="14" height="14" fill="{color}"/>'
            f'<text x="{x + 18}" y="27" font-size="11">'
            f"{html.escape(label)}</text>"
        )
    content.append("</svg>")
    return "".join(content)


__all__ = [
    "DisabledTrainingDashboard",
    "TrainingDashboard",
    "build_schedule_gantt_svg",
    "create_training_dashboard",
    "override_visdom_enabled",
    "resolve_visdom_settings",
    "visdom_environment_name",
]
