from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import time
import traceback
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from agent.ppo.agent import summarize_policy_decision_diagnostics
from agent.ppo.network import network_requires_graph_observation
from data.dataset import GeneratedInstanceRecord, OnlineInstanceDataset
from data.distribution import stable_seed
from data.models import AssemblyInstance
from environment import (
    AssemblySchedulingEnv,
    Observation,
    PreferenceContext,
    PreferenceContextInput,
    RewardVector,
)
from utils import action_trace_sha256, derive_evaluation_sampling_seed

if TYPE_CHECKING:
    from agent.ppo.agent import PPOAgent

EVALUATION_WORKER_TIMING_FIELDS = (
    "observation_time_seconds", "environment_step_time_seconds",
    "terminal_metrics_time_seconds", "worker_service_time_seconds", "reset_time_seconds",
)


@dataclass
class WorkerResponse:
    lane_id: int
    observation: Observation | None = None
    action_mask: np.ndarray | None = None
    reward_vector: RewardVector | None = None
    terminated: bool = False
    truncated: bool = False
    metrics: dict[str, Any] | None = None
    instance_id: str | None = None
    metadata: dict[str, Any] | None = None
    generation_time_seconds: float = 0.0
    environment_step_time_seconds: float = 0.0
    observation_time_seconds: float = 0.0
    terminal_metrics_time_seconds: float = 0.0
    worker_service_time_seconds: float = 0.0
    reset_time_seconds: float = 0.0
    environment_step_count: int = 0
    local_physical_forced_action_count: int = 0
    cache_hit: bool = False
    record_sha256: str | None = None


@dataclass(frozen=True)
class WorkerProgress:
    lane_id: int
    command: str
    timestamp: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class _WorkerResetRequest:
    value: int | AssemblyInstance
    drain_physical_forced_actions: bool = False
    max_environment_steps: int | None = None
    preference: PreferenceContextInput | None = None
    capture_trajectory: bool = False


@dataclass(frozen=True)
class _WorkerStepRequest:
    action: int
    drain_physical_forced_actions: bool = False
    max_environment_steps: int | None = None


@dataclass
class WorkerFailure:
    lane_id: int
    command: str
    message: str
    traceback: str








def forced_action_from_mask(action_mask: np.ndarray) -> int | None:
    """Return the only legal action, or ``None`` for a policy decision."""

    mask = np.asarray(action_mask, dtype=np.bool_)
    legal_actions = np.flatnonzero(~mask)
    if legal_actions.size == 0:
        raise ValueError("action mask has no legal actions")
    if legal_actions.size == 1:
        return int(legal_actions[0])
    return None


def physical_forced_action_from_mask(
    environment: AssemblySchedulingEnv,
    action_mask: np.ndarray,
) -> int | None:
    """Return a physically forced pair or WAIT action for local chaining."""

    action = forced_action_from_mask(action_mask)
    if action is None:
        return None
    diagnostic = environment.forced_action_diagnostic(action_mask)
    if diagnostic is None:
        raise RuntimeError(
            "singleton action mask has no forced-action diagnostic"
        )
    if not (
        bool(diagnostic["wait_physically_unavailable"])
        or bool(diagnostic["pair_physically_unavailable"])
    ):
        return None
    return action


def _aggregate_reward_vectors(
    reward_vectors: Sequence[RewardVector],
) -> RewardVector | None:
    if not reward_vectors:
        return None
    totals = {
        name: 0.0
        for name in RewardVector.__dataclass_fields__
        if name != "preference_key" and (name != "workload" or reward_vectors[0].workload is not None)
    }
    preference_keys = {value.preference_key for value in reward_vectors}
    if len(preference_keys) != 1:
        raise RuntimeError("an aggregated transition changed episode preference")
    for reward_vector in reward_vectors:
        if (reward_vector.workload is None) != (reward_vectors[0].workload is None):
            raise RuntimeError("an aggregated transition changed workload reward identity")
        for name in totals:
            totals[name] += float(getattr(reward_vector, name))
    return RewardVector(
        **totals, preference_key=next(iter(preference_keys))
    )


def _worker_roll_forward(
    lane_id: int,
    environment: AssemblySchedulingEnv,
    observation: Observation | None,
    *,
    preserve_graph: bool,
    requested_action: int | None,
    drain_physical_forced_actions: bool,
    max_environment_steps: int | None,
    **kwargs,
) -> WorkerResponse:
    """Execute one requested action plus its physically forced suffix."""
    service_started = time.perf_counter()
    observation_started = (0.0 if requested_action is None else
                           getattr(environment, "_observation_time_seconds", 0.0))
    metrics_seconds = 0.0

    def stamp(response):
        response.observation_time_seconds = max(
            0.0, getattr(environment, "_observation_time_seconds", 0.0) - observation_started)
        response.terminal_metrics_time_seconds = metrics_seconds
        response.worker_service_time_seconds = (
            time.perf_counter() - service_started + float(kwargs.get("reset_time_seconds", 0.0)))
        return response

    if max_environment_steps is not None and max_environment_steps < 0:
        raise ValueError("max_environment_steps cannot be negative")
    if requested_action is not None and max_environment_steps == 0:
        raise ValueError("a step request requires a positive step budget")

    rewards: list[RewardVector] = []
    environment_step_count = 0
    local_forced_action_count = 0
    environment_step_time_seconds = 0.0
    terminated = environment.terminated
    truncated = environment.truncated

    def execute(action: int, *, local_forced: bool) -> None:
        nonlocal observation
        nonlocal environment_step_count
        nonlocal local_forced_action_count
        nonlocal environment_step_time_seconds
        nonlocal terminated
        nonlocal truncated
        step_start = time.perf_counter()
        observation, reward, terminated, truncated, _ = environment.step(
            int(action),
            build_observation=False,
        )
        environment_step_time_seconds += time.perf_counter() - step_start
        rewards.append(reward)
        environment_step_count += 1
        if local_forced:
            local_forced_action_count += 1

    if requested_action is not None:
        execute(requested_action, local_forced=False)

    while (
        drain_physical_forced_actions
        and not (terminated or truncated)
        and (
            max_environment_steps is None
            or environment_step_count < max_environment_steps
        )
    ):
        action_mask = environment.get_action_mask()
        forced_action = physical_forced_action_from_mask(
            environment,
            action_mask,
        )
        if forced_action is None:
            break
        execute(forced_action, local_forced=True)

    response_kwargs = {
        **kwargs,
        "reward_vector": _aggregate_reward_vectors(rewards),
        "environment_step_time_seconds": environment_step_time_seconds,
        "environment_step_count": environment_step_count,
        "local_physical_forced_action_count": (
            local_forced_action_count
        ),
    }
    if terminated:
        metrics_started = time.perf_counter()
        terminal_metrics = _terminal_metrics(environment)
        metrics_seconds += time.perf_counter() - metrics_started
        return stamp(WorkerResponse(
            lane_id=lane_id,
            terminated=terminated,
            truncated=truncated,
            metrics=terminal_metrics,
            **response_kwargs,
        ))
    if environment_step_count > 0:
        observation = environment.observe()
    if observation is None:
        raise RuntimeError("active worker has no observation")
    response = _worker_state(
        lane_id,
        environment,
        observation,
        preserve_graph=preserve_graph,
        **response_kwargs,
    )
    if truncated:
        response.truncated = True
        metrics_started = time.perf_counter()
        response.metrics = _terminal_metrics(environment)
        metrics_seconds += time.perf_counter() - metrics_started
    return stamp(response)




@dataclass
class FixedEvaluationRollout:
    record_index: int
    metrics: dict[str, Any]
    decisions: int
    inference_time_seconds: float
    solve_time_seconds: float
    action_trace_sha256: str
    sampling_seed: int | None = None
    derived_sampling_seed: int | None = None
    sampling_evaluation_key: str | None = None
    observation_time_seconds: float = 0.0
    environment_step_time_seconds: float = 0.0
    terminal_metrics_time_seconds: float = 0.0
    worker_service_time_seconds: float = 0.0
    reset_time_seconds: float = 0.0
    actions: tuple[int, ...] = ()
    schedule: list[dict[str, Any]] = field(default_factory=list)
    reconfigurations: list[dict[str, Any]] = field(default_factory=list)


class ParallelWorkerError(RuntimeError):
    pass


class ParallelWorkerTimeout(TimeoutError):
    pass


def _worker_state(
    lane_id: int,
    environment: AssemblySchedulingEnv,
    observation,
    *,
    preserve_graph: bool,
    **kwargs,
) -> WorkerResponse:
    if not preserve_graph:
        raise ValueError("workers require graph observations")
    policy_observation = observation.copy()
    return WorkerResponse(
        lane_id=lane_id,
        observation=policy_observation,
        action_mask=environment.get_action_mask().copy(),
        **kwargs,
    )


def _terminal_metrics(environment: AssemblySchedulingEnv) -> dict[str, Any]:
    metrics = environment.metrics()
    metrics["schedule_violations"] = environment.validate_schedule()
    if getattr(environment, "_capture_evaluation_trajectory", False):
        metrics["_evaluation_trajectory"] = {
            "schedule": environment.schedule_log,
            "reconfigurations": environment.reconfiguration_log,
        }
    return metrics


def _worker_main(
    lane_id: int,
    connection: Connection,
    config: dict[str, Any],
    template: AssemblyInstance,
    episode_count: int,
) -> None:
    command = "startup"
    try:
        dataset = None
        environment = AssemblySchedulingEnv(config)
        progress_context: dict[str, Any] = {}

        def emit_progress(payload: dict[str, Any]) -> None:
            connection.send(
                WorkerProgress(
                    lane_id=lane_id,
                    command=command,
                    timestamp=time.time(),
                    payload={**progress_context, **payload},
                )
            )

        preserve_graph = network_requires_graph_observation(
            config["network"]
        )
        connection.send(WorkerResponse(lane_id=lane_id))
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(WorkerResponse(lane_id=lane_id))
                return
            if command in {"reset_online", "generate_online"} and dataset is None:
                dataset = OnlineInstanceDataset(
                    config=config,
                    template=template,
                    episode_count=episode_count,
                )
                dataset.generator.progress_callback = emit_progress
            if command == "reset_online":
                request = (
                    payload
                    if isinstance(payload, _WorkerResetRequest)
                    else _WorkerResetRequest(value=int(payload))
                )
                episode_index = int(request.value)
                progress_context = {
                    "episode": episode_index,
                    "seed": stable_seed('factorial27_v1','train',episode_index),
                    "phase": "instance_generation",
                }
                generation_start = time.perf_counter()
                record, cache_hit, digest, _ = dataset.get_with_cache_info(
                    episode_index
                )
                generation_time = time.perf_counter() - generation_start
                reset_started = time.perf_counter()
                observation = environment.reset(
                    record.instance, preference=request.preference
                )
                environment._capture_evaluation_trajectory = False
                reset_seconds = time.perf_counter() - reset_started
                metadata = {
                    key: record.metadata.get(key)
                    for key in (
                        "seed",
                        "cell_id", "N", "K", "M", "W", "A",
                        "scale",
                        "feasibility_status",
                        "diagnostic_status",
                        "sampled_parameters",
                        "generator_config_sha256",
                        "environment_config_sha256",
                        "distribution_contract_sha256",
                        "training_cache_fingerprint",
                        "generation_attempt",
                    )
                }
                connection.send(
                    _worker_roll_forward(
                        lane_id,
                        environment,
                        observation,
                        preserve_graph=preserve_graph,
                        requested_action=None,
                        drain_physical_forced_actions=(
                            request.drain_physical_forced_actions
                        ),
                        max_environment_steps=(
                            request.max_environment_steps
                        ),
                        instance_id=record.instance.instance_id,
                        metadata=metadata,
                        generation_time_seconds=generation_time,
                        reset_time_seconds=reset_seconds,
                        cache_hit=cache_hit,
                        record_sha256=digest,
                    )
                )
                continue
            if command == "generate_online":
                episode_index = int(payload)
                progress_context = {
                    "episode": episode_index,
                    "seed": stable_seed('factorial27_v1','train',episode_index),
                    "phase": "instance_generation",
                }
                generation_start = time.perf_counter()
                record, cache_hit, digest, _ = dataset.get_with_cache_info(
                    episode_index
                )
                generation_time = time.perf_counter() - generation_start
                connection.send(
                    WorkerResponse(
                        lane_id=lane_id,
                        instance_id=record.instance.instance_id,
                        metadata=dict(record.metadata),
                        generation_time_seconds=generation_time,
                        cache_hit=cache_hit,
                        record_sha256=digest,
                    )
                )
                continue
            if command == "reset_instance":
                request = (
                    payload
                    if isinstance(payload, _WorkerResetRequest)
                    else _WorkerResetRequest(value=payload)
                )
                if not isinstance(request.value, AssemblyInstance):
                    raise TypeError(
                        "reset_instance requires an AssemblyInstance"
                    )
                reset_started = time.perf_counter()
                observation = environment.reset(
                    request.value, preference=request.preference
                )
                environment._capture_evaluation_trajectory = request.capture_trajectory
                reset_seconds = time.perf_counter() - reset_started
                connection.send(
                    _worker_roll_forward(
                        lane_id,
                        environment,
                        observation,
                        preserve_graph=preserve_graph,
                        requested_action=None,
                        drain_physical_forced_actions=(
                            request.drain_physical_forced_actions
                        ),
                        max_environment_steps=(
                            request.max_environment_steps
                        ),
                        instance_id=request.value.instance_id,
                        reset_time_seconds=reset_seconds,
                    )
                )
                continue
            if command == "step":
                request = (
                    payload
                    if isinstance(payload, _WorkerStepRequest)
                    else _WorkerStepRequest(action=int(payload))
                )
                connection.send(
                    _worker_roll_forward(
                        lane_id,
                        environment,
                        None,
                        preserve_graph=preserve_graph,
                        requested_action=request.action,
                        drain_physical_forced_actions=(
                            request.drain_physical_forced_actions
                        ),
                        max_environment_steps=(
                            request.max_environment_steps
                        ),
                    )
                )
                continue
            if command == "snapshot":
                metrics_started = time.perf_counter()
                observation_started = environment._observation_time_seconds
                metrics = _terminal_metrics(environment)
                metrics_seconds = time.perf_counter() - metrics_started
                connection.send(
                    WorkerResponse(
                        lane_id=lane_id,
                        metrics=metrics,
                        observation_time_seconds=environment._observation_time_seconds - observation_started,
                        terminal_metrics_time_seconds=metrics_seconds,
                        worker_service_time_seconds=metrics_seconds,
                    )
                )
                continue
            raise ValueError(f"unknown worker command {command!r}")
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(
                WorkerFailure(
                    lane_id=lane_id,
                    command=command,
                    message=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ParallelEpisodeRunner:
    """Synchronous complete-episode runner with central batched inference."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        template: AssemblyInstance,
        episode_count: int,
        worker_count: int,
        diagnostic_directory: str | Path | None = None,
    ):
        if worker_count < 1:
            raise ValueError("episode runner requires at least one worker")
        if episode_count < 1:
            raise ValueError("episode_count must be positive")
        training = config["training"]
        start_method = str(
            training["multiprocessing_start_method"]
        )
        if start_method != "spawn":
            raise ValueError(
                "multiprocessing_start_method must be 'spawn'"
            )
        self.config = config
        self.worker_count = int(worker_count)
        self.episode_count = int(episode_count)
        self.timeout_seconds = float(
            training["worker_timeout_seconds"]
        )
        if self.timeout_seconds <= 0:
            raise ValueError("worker_timeout_seconds must be positive")
        self.stall_timeout_seconds = float(
            training.get(
                "worker_stall_timeout_seconds",
                min(60.0, self.timeout_seconds),
            )
        )
        if self.stall_timeout_seconds <= 0:
            raise ValueError("worker_stall_timeout_seconds must be positive")
        self.slow_instance_seconds = float(
            training.get("slow_instance_seconds", 30.0)
        )
        self.debug_worker_steps = config.get("logging", {}).get(
            "worker_progress", {}
        ).get("debug_steps", False)
        if not isinstance(self.debug_worker_steps, bool):
            raise ValueError("logging.worker_progress.debug_steps must be boolean")
        self.diagnostic_directory = (
            None
            if diagnostic_directory is None
            else Path(diagnostic_directory)
        )
        if self.diagnostic_directory is not None:
            self.diagnostic_directory.mkdir(parents=True, exist_ok=True)
        self._command_serial = 0
        self._lane_command_started: dict[int, float] = {}
        self._lane_command_serial: dict[int, int] = {}
        self._lane_command_name: dict[int, str] = {}
        self._latest_progress: dict[int, dict[str, Any]] = {}
        self._slow_command_serial: dict[int, int] = {}
        self._lane_instances: dict[int, dict[str, Any]] = {}
        self._last_logged_error: BaseException | None = None
        context = multiprocessing.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[Any] = []
        self._closed = False
        try:
            for lane_id in range(self.worker_count):
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=_worker_main,
                    args=(
                        lane_id,
                        child_connection,
                        config,
                        template,
                        episode_count,
                    ),
                    name=f"assembly-rollout-{lane_id}",
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)
                self._lane_command_started[lane_id] = time.monotonic()
                self._lane_command_serial[lane_id] = 0
                self._lane_command_name[lane_id] = "startup"
            self._receive_responses(range(self.worker_count))
        except BaseException:
            self.close(force=True)
            raise

    def __enter__(self) -> "ParallelEpisodeRunner":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_value is not None and exc_value is not self._last_logged_error:
            for lane_id in list(self._lane_instances) or [None]:
                self._record_worker_error(lane_id, exc_value)
        self.close(force=exc_type is not None)

    def _send_commands(
        self,
        commands: dict[int, tuple[str, Any]],
    ) -> None:
        if self._closed:
            raise RuntimeError("parallel runner is closed")
        timing = getattr(self, "_evaluation_runtime_timing", None)
        sending_started = time.perf_counter() if timing is not None else 0.0
        command_started = time.monotonic()
        for lane_id, message in commands.items():
            starts_instance = message[0] in {"reset_online", "reset_instance", "generate_online"}
            if starts_instance and lane_id in self._lane_instances:
                self._finish_instance(lane_id, reason="replaced")
            self._command_serial += 1
            self._lane_command_started[lane_id] = command_started
            self._lane_command_serial[lane_id] = self._command_serial
            self._lane_command_name[lane_id] = str(message[0])
            self._latest_progress.pop(lane_id, None)
            self._slow_command_serial.pop(lane_id, None)
            if starts_instance:
                self._start_instance(lane_id, message, command_started)
            try:
                process = self._processes[lane_id]
                if not process.is_alive():
                    raise ParallelWorkerError(
                        f"worker {lane_id} exited with code "
                        f"{process.exitcode}"
                    )
                self._connections[lane_id].send(message)
            except (ParallelWorkerError, OSError) as error:
                self._record_worker_error(lane_id, error)
                raise
        if timing is not None:
            timing["main_send_seconds"] += time.perf_counter() - sending_started

    def _exchange(self, commands: dict[int, tuple[str, Any]]) -> dict[int, WorkerResponse]:
        self._send_commands(commands)
        timing = getattr(self, "_evaluation_runtime_timing", None)
        if timing is None:
            return self._receive_responses(commands)
        receiving_started = time.perf_counter()
        timing["exchange_count"] += 1
        try:
            return self._receive_responses(commands)
        finally:
            # Includes worker computation, transport and response handling; not pure IPC.
            timing["main_receive_seconds"] += time.perf_counter() - receiving_started

    def _append_diagnostic_jsonl(
        self, filename: str, payload: dict[str, Any]
    ) -> None:
        if self.diagnostic_directory is None:
            return
        destination = self.diagnostic_directory / filename
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()

    def _instance_log_context(self, lane_id: int | None) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._lane_instances.get(lane_id, {}).items()
            if not key.startswith("_")
        }

    def _start_instance(
        self, lane_id: int, message: tuple[str, Any], started: float
    ) -> None:
        if lane_id in self._lane_instances:
            self._finish_instance(lane_id, reason="replaced")
        command, payload = message
        value = payload.value if isinstance(payload, _WorkerResetRequest) else payload
        online = command != "reset_instance"
        self._lane_instances[lane_id] = {
            "instance_task_id": self._lane_command_serial[lane_id],
            "task_kind": "generation" if command == "generate_online" else "rollout",
            "instance_id": None if online else value.instance_id,
            "episode_index": int(value) if online else None,
            "seed": (
                stable_seed('factorial27_v1','train',int(value)) if online else None
            ),
            "environment_step_count": 0,
            "_started_at": started,
            "_slow_recorded": False,
        }
        self._append_diagnostic_jsonl("worker_progress.jsonl", {
            **self._instance_log_context(lane_id),
            "event": "instance_start",
            "timestamp": time.time(),
            "lane": lane_id,
            "command": command,
            "command_serial": self._lane_command_serial[lane_id],
        })

    def _record_slow_task(self, lane_id: int, record: dict[str, Any], classification: str) -> None:
        serial = self._lane_command_serial.get(lane_id, -1)
        context = self._lane_instances.get(lane_id)
        if self._slow_command_serial.get(lane_id) == serial or (
            context is not None and context["_slow_recorded"]
        ):
            return
        self._slow_command_serial[lane_id] = serial
        if context is not None:
            context["_slow_recorded"] = True
        payload = {
            **self._instance_log_context(lane_id),
            **record,
            "event": "slow_task",
            "classification": classification,
        }
        self._append_diagnostic_jsonl("worker_progress.jsonl", payload)
        self._append_diagnostic_jsonl("slow_instances.jsonl", payload)

    def _finish_instance(
        self, lane_id: int, *, reason: str, response: WorkerResponse | None = None
    ) -> None:
        context = self._lane_instances.get(lane_id)
        if context is None:
            return
        record = {
            **self._instance_log_context(lane_id),
            "event": "instance_end",
            "timestamp": time.time(),
            "lane": lane_id,
            "command": self._lane_command_name.get(lane_id),
            "command_serial": self._lane_command_serial.get(lane_id),
            "elapsed_seconds": time.monotonic() - context["_started_at"],
            "terminal_reason": reason,
            "terminated": bool(response and response.terminated),
            "truncated": bool(response and response.truncated),
        }
        if response is not None and response.metrics is not None:
            record["task_failed"] = response.metrics.get("task_failed")
        if record["elapsed_seconds"] >= self.slow_instance_seconds:
            self._record_slow_task(lane_id, record, "completed_slow_instance")
        self._append_diagnostic_jsonl("worker_progress.jsonl", record)
        del self._lane_instances[lane_id]

    def _record_worker_error(self, lane_id: int | None, error: BaseException) -> None:
        self._last_logged_error = error
        self._append_diagnostic_jsonl("worker_progress.jsonl", {
            **self._instance_log_context(lane_id),
            "event": "error",
            "timestamp": time.time(),
            "lane": lane_id,
            "command": self._lane_command_name.get(lane_id),
            "command_serial": self._lane_command_serial.get(lane_id),
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "latest_progress": self._latest_progress.get(lane_id),
        })
        if lane_id is not None:
            self._finish_instance(lane_id, reason="error")

    def _record_worker_progress(self, progress: WorkerProgress) -> None:
        lane_id = int(progress.lane_id)
        elapsed = time.monotonic() - self._lane_command_started.get(
            lane_id, time.monotonic()
        )
        record = {
            "event": "heartbeat",
            "timestamp": progress.timestamp,
            "lane": lane_id,
            "command": progress.command,
            "command_serial": self._lane_command_serial.get(lane_id),
            "elapsed_seconds": elapsed,
            **progress.payload,
        }
        self._latest_progress[lane_id] = record
        if elapsed >= self.slow_instance_seconds:
            self._record_slow_task(lane_id, record, "active_slow_search")

    def _record_completed_response(self, response: WorkerResponse) -> None:
        lane_id = int(response.lane_id)
        elapsed = time.monotonic() - self._lane_command_started.get(
            lane_id, time.monotonic()
        )
        command = self._lane_command_name.get(lane_id)
        context = self._lane_instances.get(lane_id)
        if context is not None:
            context["environment_step_count"] += response.environment_step_count
            if response.instance_id is not None:
                context["instance_id"] = response.instance_id
            if response.metadata is not None:
                for key in ("seed", "generation_attempt", "cell_id", "scale"):
                    context[key] = response.metadata.get(key)
            if command in {"reset_online", "reset_instance", "generate_online"}:
                context["generation_time_seconds"] = response.generation_time_seconds
                context["cache_hit"] = response.cache_hit
                context["record_sha256"] = response.record_sha256
        debug_step = self.debug_worker_steps and command == "step"
        slow_command = (
            max(elapsed, float(response.generation_time_seconds)) >= self.slow_instance_seconds
        )
        if debug_step or slow_command:
            record = {
                **self._instance_log_context(lane_id),
                "event": "response",
                "timestamp": time.time(),
                "lane": lane_id,
                "command": command,
                "command_serial": self._lane_command_serial.get(lane_id),
                "elapsed_seconds": elapsed,
                "generation_time_seconds": response.generation_time_seconds,
                "environment_step_count": response.environment_step_count,
                "cache_hit": response.cache_hit,
            }
            if debug_step:
                self._append_diagnostic_jsonl("worker_progress.jsonl", record)
            if slow_command:
                self._record_slow_task(lane_id, record, "completed_slow_search")
        if command == "generate_online":
            self._finish_instance(lane_id, reason="generated", response=response)
        elif command == "snapshot":
            self._finish_instance(lane_id, reason="step_limit", response=response)
        elif response.terminated or response.truncated:
            self._finish_instance(
                lane_id,
                reason=str((response.metrics or {}).get("terminal_reason") or (
                    "terminated" if response.terminated else "truncated"
                )),
                response=response,
            )

    def _receive_responses(
        self,
        lane_ids,
    ) -> dict[int, WorkerResponse]:
        pending = {
            self._connections[lane_id]: lane_id
            for lane_id in lane_ids
        }
        try:
            return self._wait_for_responses(pending)
        except (ParallelWorkerError, ParallelWorkerTimeout, OSError) as error:
            for lane_id in pending.values():
                self._record_worker_error(lane_id, error)
            raise

    def _wait_for_responses(
        self, pending: dict[Connection, int]
    ) -> dict[int, WorkerResponse]:
        responses: dict[int, WorkerResponse] = {}
        timing = getattr(self, "_evaluation_runtime_timing", None)
        arrivals = []
        started = time.monotonic()
        hard_deadline = started + self.timeout_seconds
        last_heartbeat = {lane_id: started for lane_id in pending.values()}
        while pending:
            now = time.monotonic()
            if now >= hard_deadline:
                lanes = sorted(pending.values())
                raise ParallelWorkerTimeout(
                    "hard_timeout: active slow search exceeded total command "
                    f"timeout {self.timeout_seconds:.1f}s; workers={lanes}; "
                    f"latest_progress={self._latest_progress}"
                )
            stalled = [
                lane_id
                for lane_id in pending.values()
                if now - last_heartbeat[lane_id]
                >= self.stall_timeout_seconds
            ]
            if stalled:
                details = {
                    lane_id: self._latest_progress.get(lane_id)
                    for lane_id in stalled
                }
                raise ParallelWorkerTimeout(
                    "stall_timeout: no worker heartbeat within "
                    f"{self.stall_timeout_seconds:.1f}s; workers={stalled}; "
                    f"latest_progress={details}"
                )
            next_stall = min(
                last_heartbeat[lane_id] + self.stall_timeout_seconds
                for lane_id in pending.values()
            )
            remaining = max(
                0.0, min(hard_deadline, next_stall) - time.monotonic()
            )
            ready = wait(list(pending), timeout=remaining)
            if not ready:
                continue
            for connection in ready:
                lane_id = pending[connection]
                try:
                    response = connection.recv()
                except EOFError as error:
                    process = self._processes[lane_id]
                    raise ParallelWorkerError(
                        f"worker {lane_id} closed its pipe; exit code "
                        f"{process.exitcode}"
                    ) from error
                if isinstance(response, WorkerProgress):
                    last_heartbeat[lane_id] = time.monotonic()
                    self._record_worker_progress(response)
                    continue
                if isinstance(response, WorkerFailure):
                    raise ParallelWorkerError(
                        f"worker {lane_id} failed during "
                        f"{response.command}: {response.message}\n"
                        f"{response.traceback}"
                    )
                if not isinstance(response, WorkerResponse):
                    raise ParallelWorkerError(
                        f"worker {lane_id} returned an invalid response"
                    )
                pending.pop(connection)
                responses[lane_id] = response
                if timing is not None:
                    arrivals.append(time.perf_counter())
                self._record_completed_response(response)
        if timing is not None and len(arrivals) > 1:
            timing["response_tail_seconds"] += arrivals[-1] - arrivals[0]
        return responses




    @contextmanager
    def _track_evaluation_runtime(self):
        previous = getattr(self, "_evaluation_runtime_timing", None)
        timing = dict(main_send_seconds=0.0, main_receive_seconds=0.0,
                      response_tail_seconds=0.0, policy_inference_seconds=0.0,
                      exchange_count=0, policy_batch_count=0, policy_observation_count=0)
        self._evaluation_runtime_timing = timing
        started = time.perf_counter()
        try:
            yield
        finally:
            timing["wall_seconds"] = time.perf_counter() - started
            timing["mean_policy_batch_size"] = (
                timing["policy_observation_count"] / max(1, timing["policy_batch_count"]))
            self.last_evaluation_runtime = dict(timing)
            self._evaluation_runtime_timing = previous

    def evaluate_records(
        self, agent: "PPOAgent", records: Sequence[GeneratedInstanceRecord], *,
        max_parallelism: int | None = None, deterministic: bool = True,
        sampling_seed: int | None = None,
        preferences: Sequence[PreferenceContextInput] | None = None,
        capture_trajectories: bool = False,
        unit_sampling_seeds: Sequence[int] | None = None,
        capture_actions: bool = False,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> list[FixedEvaluationRollout]:
        # Reject incompatible agents before starting any evaluation bookkeeping.
        agent.assert_evaluation_config(self.config)
        with self._track_evaluation_runtime():
            return self._evaluate_records(
                agent, records, max_parallelism=max_parallelism,
                deterministic=deterministic, sampling_seed=sampling_seed, preferences=preferences,
                capture_trajectories=capture_trajectories,
                unit_sampling_seeds=unit_sampling_seeds, capture_actions=capture_actions,
                progress_callback=progress_callback)

    def _evaluate_records(
        self,
        agent: "PPOAgent",
        records: Sequence[GeneratedInstanceRecord],
        *,
        max_parallelism: int | None = None,
        deterministic: bool = True,
        sampling_seed: int | None = None,
        preferences: Sequence[PreferenceContextInput] | None = None,
        capture_trajectories: bool = False,
        unit_sampling_seeds: Sequence[int] | None = None,
        capture_actions: bool = False,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> list[FixedEvaluationRollout]:
        parallelism = (
            self.worker_count
            if max_parallelism is None
            else int(max_parallelism)
        )
        if parallelism < 1 or parallelism > self.worker_count:
            raise ValueError(
                "max_parallelism must be within the worker pool size"
            )
        if not deterministic and sampling_seed is None and unit_sampling_seeds is None:
            raise ValueError(
                "sampling_seed is required for sampled fixed evaluation"
            )
        if preferences is not None and len(preferences) != len(records):
            raise ValueError("evaluation preferences must align with records")
        if unit_sampling_seeds is not None and len(unit_sampling_seeds) != len(records):
            raise ValueError("unit sampling seeds must align with records")
        results: list[FixedEvaluationRollout] = []
        states: dict[int, WorkerResponse] = {}
        active: set[int] = set()
        record_indices: dict[int, int] = {}
        started_at: dict[int, float] = {}
        decisions: dict[int, int] = {}
        inference_times: dict[int, float] = {}
        action_traces: dict[int, list[int]] = {}
        policy_diagnostics: dict[int, list[dict[str, Any]]] = {}
        evaluation_keys: dict[int, str | None] = {}
        derived_sampling_seeds: dict[int, int] = {}
        generators: dict[int, torch.Generator] = {}
        next_record = 0
        worker_times: dict[int, dict[str, float]] = {}
        environment_steps = 0

        def progress():
            if progress_callback is not None:
                progress_callback(dict(completed=len(results), total=len(records),
                                       environment_steps=environment_steps, active=len(active)))

        def fill_available(lanes: Sequence[int]) -> None:
            nonlocal next_record
            requests = {}
            reset_start = time.perf_counter()
            for lane in lanes:
                if next_record >= len(records):
                    break
                index = next_record
                next_record += 1
                record_indices[lane] = index
                started_at[lane] = reset_start
                decisions[lane] = 0
                inference_times[lane] = 0.0
                action_traces[lane] = []
                policy_diagnostics[lane] = []
                worker_times[lane] = dict.fromkeys(EVALUATION_WORKER_TIMING_FIELDS, 0.0)
                preference = None if preferences is None else preferences[index]
                evaluation_keys[lane] = (
                    None if preference is None else PreferenceContext.from_input(preference).key
                )
                if not deterministic:
                    derived_sampling_seeds[lane] = int(unit_sampling_seeds[index]) if unit_sampling_seeds is not None else derive_evaluation_sampling_seed(
                        int(sampling_seed), records[index].instance.instance_id,
                        evaluation_keys[lane],
                    )
                    generators[lane] = torch.Generator(device=agent.device).manual_seed(
                        derived_sampling_seeds[lane]
                    )
                requests[lane] = (
                    "reset_instance",
                    _WorkerResetRequest(value=records[index].instance, preference=preference,
                                        capture_trajectory=capture_trajectories),
                )
            if requests:
                responses = self._exchange(requests)
                states.update(responses)
                for lane, response in responses.items():
                    for name in EVALUATION_WORKER_TIMING_FIELDS:
                        worker_times[lane][name] += float(getattr(response, name, 0.0))
                active.update(requests)

        # A completed lane immediately takes the next fixed evaluation unit.
        # Sampling state belongs to the instance/preference unit, not its lane.
        fill_available(range(parallelism))
        progress()
        while active:
            lanes = sorted(active)
            observations = [
                states[lane].observation for lane in lanes
            ]
            masks = [
                states[lane].action_mask for lane in lanes
            ]
            if deterministic:
                inference_start = time.perf_counter()
                actions, _, _ = agent.act_batch(
                    observations,
                    masks,
                    deterministic=True,
                )
                elapsed = time.perf_counter() - inference_start
                share = elapsed / len(lanes)
                for lane in lanes:
                    inference_times[lane] += share
                diagnostic_rows = (
                    agent.consume_policy_decision_diagnostics()
                )
                if diagnostic_rows and len(diagnostic_rows) != len(lanes):
                    raise RuntimeError(
                        "batched policy diagnostics do not match lanes"
                    )
                for lane, action, diagnostic in zip(
                    lanes, actions, diagnostic_rows
                ):
                    diagnostic["selected_action"] = int(action)
                    diagnostic["ranker_top_selected"] = bool(
                        int(action)
                        == int(diagnostic.get("relative_top_action", -1))
                    )
                    policy_diagnostics[lane].append(diagnostic)
            else:
                inference_start = time.perf_counter()
                if getattr(agent.network, "execution_mode", "") == "reference_v8":
                    actions = []
                    diagnostic_rows = []
                    for lane, observation, mask in zip(lanes, observations, masks):
                        actions.append(agent.act(
                            observation, mask, generator=generators[lane]
                        )[0])
                        diagnostic_rows.extend(agent.consume_policy_decision_diagnostics())
                else:
                    actions, _, _ = agent.act_batch(
                        observations, masks,
                        generators=[generators[lane] for lane in lanes],
                    )
                    diagnostic_rows = agent.consume_policy_decision_diagnostics()
                elapsed = time.perf_counter() - inference_start
                for lane in lanes:
                    inference_times[lane] += elapsed / len(lanes)
                if diagnostic_rows and len(diagnostic_rows) != len(lanes):
                    raise RuntimeError("batched policy diagnostics do not match lanes")
                for lane, action, diagnostic in zip(lanes, actions, diagnostic_rows):
                    diagnostic["selected_action"] = int(action)
                    diagnostic["ranker_top_selected"] = bool(
                        int(action) == int(diagnostic.get("relative_top_action", -1))
                    )
                    policy_diagnostics[lane].append(diagnostic)
            timing = self._evaluation_runtime_timing
            timing["policy_inference_seconds"] += elapsed
            timing["policy_batch_count"] += 1
            timing["policy_observation_count"] += len(lanes)
            for lane, action in zip(lanes, actions):
                action_traces[lane].append(int(action))
            step_responses = self._exchange(
                {
                    lane: ("step", action)
                    for lane, action in zip(lanes, actions)
                }
            )
            environment_steps += sum(response.environment_step_count for response in step_responses.values())
            for lane in lanes:
                decisions[lane] += 1
                response = step_responses[lane]
                for name in EVALUATION_WORKER_TIMING_FIELDS:
                    worker_times[lane][name] += float(getattr(response, name, 0.0))
                if response.terminated or response.truncated:
                    if response.metrics is None:
                        raise ParallelWorkerError(
                            f"worker {lane} returned no metrics"
                        )
                    response.metrics.update(
                        summarize_policy_decision_diagnostics(
                            policy_diagnostics[lane]
                        )
                    )
                    trajectory = response.metrics.pop("_evaluation_trajectory", {})
                    if capture_trajectories and not trajectory:
                        raise ParallelWorkerError("worker returned no requested trajectory")
                    results.append(
                        FixedEvaluationRollout(
                            record_index=record_indices[lane],
                            metrics=response.metrics,
                            decisions=decisions[lane],
                            inference_time_seconds=(
                                inference_times[lane]
                            ),
                            solve_time_seconds=(
                                time.perf_counter() - started_at[lane]
                            ),
                            action_trace_sha256=action_trace_sha256(
                                action_traces[lane]
                            ),
                            sampling_seed=(
                                None if deterministic or sampling_seed is None else int(sampling_seed)
                            ),
                            derived_sampling_seed=(
                                None
                                if deterministic
                                else derived_sampling_seeds[lane]
                            ),
                            sampling_evaluation_key=evaluation_keys[lane],
                            actions=tuple(action_traces[lane]) if capture_trajectories or capture_actions else (),
                            schedule=trajectory.get("schedule", []),
                            reconfigurations=trajectory.get("reconfigurations", []),
                            **worker_times[lane],
                        )
                    )
                    active.remove(lane)
                else:
                    states[lane] = response
            fill_available([lane for lane in range(parallelism) if lane not in active])
            progress()
        results.sort(key=lambda value: value.record_index)
        return results

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        for lane_id in list(self._lane_instances):
            self._finish_instance(lane_id, reason="aborted" if force else "runner_closed")
        if not force:
            for lane_id, process in enumerate(self._processes):
                if process.is_alive():
                    try:
                        self._connections[lane_id].send(
                            ("close", None)
                        )
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            deadline = time.monotonic() + 5.0
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(5.0)
        for connection in self._connections:
            connection.close()
        self._last_logged_error = None
