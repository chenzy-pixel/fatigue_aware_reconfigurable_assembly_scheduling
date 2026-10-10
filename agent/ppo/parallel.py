from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import time
import traceback
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from agent.ppo.buffer import RolloutBuffer
from agent.ppo.agent import summarize_policy_decision_diagnostics
from agent.ppo.network import network_requires_graph_observation
from data.dataset import GeneratedInstanceRecord, OnlineInstanceDataset
from data.models import AssemblyInstance
from environment import (
    AssemblySchedulingEnv,
    Observation,
    PreferenceContext,
    PreferenceContextInput,
    RewardVector,
    proxy_return_from_metrics,
    quality_preference_for_episode,
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


@dataclass
class EpisodeRollout:
    episode_index: int
    instance_id: str
    metadata: dict[str, Any]
    buffer: RolloutBuffer
    reward_sum: float
    step_count: int
    metrics: dict[str, Any]
    generation_time_seconds: float
    environment_step_time_seconds: float
    reward_components: dict[str, float] = field(default_factory=dict)
    expected_reward: float = 0.0
    unattributed_forced_reward: float = 0.0
    worker_step_command_count: int = 0
    worker_local_physical_forced_action_count: int = 0
    policy_diagnostics: dict[str, float | int] = field(default_factory=dict)
    observation_time_seconds: float = 0.0
    terminal_metrics_time_seconds: float = 0.0
    worker_service_time_seconds: float = 0.0
    reset_time_seconds: float = 0.0

    @property
    def base_reward_sum(self) -> float:
        return float(self.reward_components.get("operation_progress", 0.0)) + float(
            self.reward_components.get("quality", 0.0)
        )

    @property
    def unshaped_reward_sum(self) -> float:
        return self.base_reward_sum + float(
            self.reward_components.get("failure", 0.0)
        )

    @property
    def policy_step_count(self) -> int:
        return len(self.buffer)

    @property
    def forced_action_count(self) -> int:
        return self.step_count - self.policy_step_count

    @property
    def forced_action_ratio(self) -> float:
        return (
            self.forced_action_count / self.step_count
            if self.step_count > 0
            else 0.0
        )

    @property
    def worker_local_physical_forced_share(self) -> float:
        return (
            self.worker_local_physical_forced_action_count
            / self.forced_action_count
            if self.forced_action_count > 0
            else 0.0
        )


@dataclass
class TrainingRolloutBatch:
    episodes: list[EpisodeRollout]
    buffer: RolloutBuffer
    sampling_wall_time_seconds: float
    policy_inference_time_seconds: float

    @property
    def transition_count(self) -> int:
        return len(self.buffer)

    @property
    def environment_step_count(self) -> int:
        return sum(episode.step_count for episode in self.episodes)

    @property
    def forced_action_count(self) -> int:
        return sum(
            episode.forced_action_count for episode in self.episodes
        )

    @property
    def forced_action_ratio(self) -> float:
        return (
            self.forced_action_count / self.environment_step_count
            if self.environment_step_count > 0
            else 0.0
        )

    @property
    def worker_step_command_count(self) -> int:
        return sum(
            episode.worker_step_command_count
            for episode in self.episodes
        )

    @property
    def worker_local_physical_forced_action_count(self) -> int:
        return sum(
            episode.worker_local_physical_forced_action_count
            for episode in self.episodes
        )

    @property
    def worker_local_physical_forced_share(self) -> float:
        return (
            self.worker_local_physical_forced_action_count
            / self.forced_action_count
            if self.forced_action_count > 0
            else 0.0
        )


@dataclass
class _PendingTransition:
    observation: Observation
    action_mask: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float = 0.0


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
        if name != "preference_key"
    }
    preference_keys = {value.preference_key for value in reward_vectors}
    if len(preference_keys) != 1:
        raise RuntimeError("an aggregated transition changed episode preference")
    for reward_vector in reward_vectors:
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


def _commit_pending_transition(
    context: dict[str, Any],
    *,
    done: bool,
) -> None:
    pending = context["pending_transition"]
    if pending is None:
        return
    context["buffer"].add(
        pending.observation,
        pending.action_mask,
        pending.action,
        pending.log_probability,
        pending.value,
        pending.reward,
        done,
    )
    context["pending_transition"] = None


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
        dataset = OnlineInstanceDataset(
            config=config,
            template=template,
            episode_count=episode_count,
        )
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

        dataset.generator.progress_callback = emit_progress
        preserve_graph = network_requires_graph_observation(
            config["network"]
        )
        connection.send(WorkerResponse(lane_id=lane_id))
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(WorkerResponse(lane_id=lane_id))
                return
            if command == "reset_online":
                request = (
                    payload
                    if isinstance(payload, _WorkerResetRequest)
                    else _WorkerResetRequest(value=int(payload))
                )
                episode_index = int(request.value)
                progress_context = {
                    "episode": episode_index,
                    "seed": dataset.seed_start + episode_index,
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
                reset_seconds = time.perf_counter() - reset_started
                metadata = {
                    key: record.metadata.get(key)
                    for key in (
                        "seed",
                        "pressure_type",
                        "cost_profile",
                        "severity",
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
                    "seed": dataset.seed_start + episode_index,
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
                int(self.config["dataset"]["splits"]["train"]["seed_start"])
                + int(value) if online else None
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
                for key in ("seed", "generation_attempt", "pressure_type", "cost_profile"):
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

    def pre_generate_training_instances(
        self,
        episode_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Fill and verify the deterministic training cache before rollout."""
        requested_indices = (
            list(range(self.episode_count))
            if episode_indices is None
            else [int(value) for value in episode_indices]
        )
        if (
            not requested_indices
            or len(set(requested_indices)) != len(requested_indices)
            or min(requested_indices) < 0
            or max(requested_indices) >= self.episode_count
        ):
            raise ValueError("invalid training cache episode indices")
        requested_count = len(requested_indices)
        entries: list[dict[str, Any]] = []
        rejection_reasons: dict[str, int] = {}
        generation_times: list[float] = []
        cache_hits = 0
        for batch_start in range(0, requested_count, self.worker_count):
            indices = requested_indices[
                batch_start : batch_start + self.worker_count
            ]
            responses = self._exchange(
                {
                    lane_id: ("generate_online", episode_index)
                    for lane_id, episode_index in enumerate(indices)
                }
            )
            for lane_id, episode_index in enumerate(indices):
                response = responses[lane_id]
                metadata = response.metadata or {}
                duration = float(response.generation_time_seconds)
                generation_times.append(duration)
                cache_hits += int(response.cache_hit)
                rejected = metadata.get("generation_rejection_reasons", {})
                if isinstance(rejected, dict):
                    for reason, count in rejected.items():
                        rejection_reasons[str(reason)] = (
                            rejection_reasons.get(str(reason), 0) + int(count)
                        )
                entries.append(
                    {
                        "train_index": episode_index,
                        "seed": metadata.get("seed"),
                        "instance_id": response.instance_id,
                        "sha256": response.record_sha256,
                        "cache_hit": bool(response.cache_hit),
                        "generation_time_seconds": duration,
                        "pressure_type": metadata.get("pressure_type"),
                        "severity": metadata.get("severity"),
                        "diagnostic_status": metadata.get("diagnostic_status"),
                        "feasibility_status": metadata.get("feasibility_status"),
                        "generation_attempt": metadata.get(
                            "generation_attempt"
                        ),
                        "feasibility_precheck": metadata.get(
                            "feasibility_precheck"
                        ),
                    }
                )
            print(
                "[training-cache] generated_or_verified="
                f"{len(entries)}/{requested_count}",
                flush=True,
            )
        values = np.asarray(generation_times, dtype=np.float64)
        p99 = float(np.percentile(values, 99)) if values.size else 0.0
        slow_seeds = [
            {
                "seed": entry["seed"],
                "train_index": entry["train_index"],
                "generation_time_seconds": entry["generation_time_seconds"],
            }
            for entry in entries
            if float(entry["generation_time_seconds"]) >= p99
        ]
        first_metadata = next(
            (response.metadata for response in responses.values()), {}
        )
        summary = {
            "version": "training_instance_cache_manifest_v2",
            **{name: (first_metadata or {}).get(name) for name in ("generator_config_sha256", "environment_config_sha256", "distribution_contract_sha256")},
            "instance_count": requested_count,
            "total_training_episode_count": self.episode_count,
            "generator_version": (
                first_metadata or {}
            ).get("generator_version"),
            "template_sha256": (first_metadata or {}).get(
                "template_sha256"
            ),
            "cache_fingerprint": (first_metadata or {}).get(
                "training_cache_fingerprint"
            ),
            "generator_environment_precheck_config_hash": (
                first_metadata or {}
            ).get("generator_environment_precheck_config_hash"),
            "cache_hit_count": cache_hits,
            "cache_hit_rate": cache_hits / max(1, requested_count),
            "generation_time_seconds": {
                "p50": float(np.percentile(values, 50)) if values.size else 0.0,
                "p95": float(np.percentile(values, 95)) if values.size else 0.0,
                "p99": p99,
                "max": float(np.max(values)) if values.size else 0.0,
            },
            "slow_seeds": slow_seeds,
            "generation_rejection_reasons": dict(
                sorted(rejection_reasons.items())
            ),
            "files": sorted(entries, key=lambda value: value["train_index"]),
        }
        if self.diagnostic_directory is not None:
            for filename, payload in (
                ("training_instance_manifest.json", summary),
                (
                    "training_instance_generation_summary.json",
                    {key: value for key, value in summary.items() if key != "files"},
                ),
            ):
                destination = self.diagnostic_directory / filename
                temporary = destination.with_name(f".{destination.name}.tmp")
                rendered = (
                    json.dumps(
                        payload, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n"
                )
                temporary.write_text(rendered, encoding="utf-8")
                os.replace(temporary, destination)
                if filename == "training_instance_manifest.json":
                    digest = hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest()
                    digest_path = self.diagnostic_directory / (
                        "training_instance_manifest.sha256"
                    )
                    digest_temporary = digest_path.with_name(
                        f".{digest_path.name}.tmp"
                    )
                    digest_temporary.write_text(
                        digest + "\n", encoding="utf-8"
                    )
                    os.replace(digest_temporary, digest_path)
        return summary

    def collect_training_batch(
        self,
        agent: "PPOAgent",
        episode_indices: Sequence[int],
        *,
        gamma: float,
        gae_lambda: float,
        step_limit: int | None = None,
        preferences: Sequence[PreferenceContextInput] | None = None,
        max_parallelism: int | None = None,
    ) -> TrainingRolloutBatch:
        if not episode_indices:
            raise ValueError("episode_indices cannot be empty")
        parallelism = self.worker_count if max_parallelism is None else int(max_parallelism)
        if parallelism < 1 or parallelism > self.worker_count:
            raise ValueError("max_parallelism must be within the worker pool size")
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode indices must be unique")
        forced_action_compression = bool(
            self.config["training"].get(
                "forced_action_compression", False
            )
        )
        local_physical_setting = self.config["training"].get(
            "worker_local_physical_forced_actions",
            True,
        )
        if not isinstance(local_physical_setting, bool):
            raise ValueError(
                "training.worker_local_physical_forced_actions must be "
                "boolean"
            )
        worker_local_physical_forced_actions = bool(
            forced_action_compression and local_physical_setting
        )
        if forced_action_compression and float(gamma) != 1.0:
            raise ValueError(
                "forced action compression requires ppo.gamma = 1.0"
            )
        if preferences is not None and len(preferences) != len(episode_indices):
            raise ValueError("preferences must align with episode_indices")
        if preferences is None:
            preferences = [
                quality_preference_for_episode(
                    self.config,
                    algorithm_seed=int(self.config["seed"]),
                    quality_episode_index=int(index),
                )
                for index in episode_indices
            ]
        sampling_start = time.perf_counter()
        states: dict[int, WorkerResponse] = {}
        contexts: dict[int, dict[str, Any]] = {}
        active: set[int] = set()
        completed: list[EpisodeRollout] = []
        reset_cutoff_lanes: list[int] = []

        def initialize_lane(lane_id: int, episode_index: int, response: WorkerResponse) -> None:
            if (
                response.instance_id is None
                or response.metadata is None
            ):
                raise ParallelWorkerError(
                    f"worker {lane_id} returned an incomplete reset"
                )
            context = {
                "episode_index": int(episode_index),
                "instance_id": response.instance_id,
                "metadata": response.metadata,
                "buffer": RolloutBuffer(
                    preserve_graph=agent.requires_graph_observation
                ),
                "reward_sum": 0.0,
                "reward_components": {
                    "flow": 0.0,
                    "cost": 0.0,
                    "variance": 0.0,
                    "operation_progress": 0.0,
                    "quality": 0.0,
                    "failure": 0.0,
                    "feasibility_shaping": 0.0,
                },
                "step_count": response.environment_step_count,
                "policy_step_count": 0,
                "forced_action_count": (
                    response.local_physical_forced_action_count
                ),
                "worker_step_command_count": 0,
                "worker_local_physical_forced_action_count": (
                    response.local_physical_forced_action_count
                ),
                "pending_transition": None,
                "policy_diagnostic_rows": [],
                "unattributed_forced_reward": 0.0,
                "generation_time_seconds": (
                    response.generation_time_seconds
                ),
                "environment_step_time_seconds": (
                    response.environment_step_time_seconds
                ),
                **{name: float(getattr(response, name, 0.0))
                   for name in EVALUATION_WORKER_TIMING_FIELDS if name != "environment_step_time_seconds"},
            }
            context["metadata"] = {
                **context["metadata"],
                "preference": response.observation.preference.tolist()
                if response.observation is not None
                else (response.metrics or {}).get("preference"),
                "preference_key": (
                    response.reward_vector.preference_key
                    if response.reward_vector is not None
                    else (response.metrics or {}).get("preference_key")
                    or (
                        PreferenceContext.from_input(
                            response.observation.preference
                        ).key
                        if response.observation is not None
                        else None
                    )
                ),
            }
            contexts[lane_id] = context
            if response.environment_step_count != (
                response.local_physical_forced_action_count
            ):
                raise ParallelWorkerError(
                    "reset worker reported non-forced local steps"
                )
            if response.reward_vector is not None:
                scalar_reward = response.reward_vector.scalarize(
                    self.config["reward"],
                )
                context["reward_sum"] += scalar_reward
                context["unattributed_forced_reward"] += scalar_reward
                for name in context["reward_components"]:
                    context["reward_components"][name] += float(
                        getattr(response.reward_vector, name)
                    )
            elif response.environment_step_count:
                raise ParallelWorkerError(
                    "reset worker returned steps without rewards"
                )
            if response.terminated:
                if response.metrics is None:
                    raise ParallelWorkerError(
                        "terminal reset worker returned no metrics"
                    )
                context["buffer"].compute_gae(
                    last_value=0.0,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                completed.append(
                    self._episode_result(context, response.metrics)
                )
                return
            if response.observation is None or response.action_mask is None:
                raise ParallelWorkerError(
                    f"worker {lane_id} returned no active reset state"
                )
            if (
                response.truncated
                or (step_limit is not None and context["step_count"] >= step_limit)
            ):
                reset_cutoff_lanes.append(lane_id)
            else:
                active.add(lane_id)

        def finish_reset_cutoffs() -> None:
            if reset_cutoff_lanes:
                snapshots = self._exchange(
                    {
                        lane: ("snapshot", None)
                        for lane in reset_cutoff_lanes
                    }
                )
                for lane in reset_cutoff_lanes:
                    context = contexts[lane]
                    for name in EVALUATION_WORKER_TIMING_FIELDS:
                        context[name] += float(getattr(snapshots[lane], name, 0.0))
                    context["buffer"].compute_gae(
                        last_value=0.0,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                    )
                    metrics = snapshots[lane].metrics
                    if metrics is None:
                        raise ParallelWorkerError(
                            "reset cutoff worker returned no snapshot metrics"
                        )
                    completed.append(self._episode_result(context, metrics))
            reset_cutoff_lanes.clear()

        next_position = 0

        def fill_available(lanes: Sequence[int]) -> None:
            nonlocal next_position
            available = sorted(lanes)
            while available and next_position < len(episode_indices):
                assignments = []
                for lane in available:
                    if next_position >= len(episode_indices):
                        break
                    assignments.append((lane, next_position))
                    next_position += 1
                responses = self._exchange({
                    lane: (
                        "reset_online",
                        _WorkerResetRequest(
                            value=int(episode_indices[position]),
                            drain_physical_forced_actions=worker_local_physical_forced_actions,
                            max_environment_steps=step_limit,
                            preference=preferences[position],
                        ),
                    )
                    for lane, position in assignments
                })
                states.update(responses)
                for lane, position in assignments:
                    initialize_lane(lane, episode_indices[position], responses[lane])
                finish_reset_cutoffs()
                available = [lane for lane, _ in assignments if lane not in active]

        fill_available(range(min(parallelism, len(episode_indices))))
        inference_time = 0.0
        while active:
            lanes = sorted(active)
            policy_lanes: list[int] = []
            policy_observations = []
            policy_masks: list[np.ndarray] = []
            selected_actions: dict[int, int] = {}
            sampled_transitions: dict[int, _PendingTransition] = {}
            for lane in lanes:
                observation = states[lane].observation
                action_mask = states[lane].action_mask
                if observation is None or action_mask is None:
                    raise ParallelWorkerError(
                        "active worker returned an incomplete policy state"
                    )
                forced_action = (
                    forced_action_from_mask(action_mask)
                    if forced_action_compression
                    else None
                )
                if forced_action is not None:
                    selected_actions[lane] = forced_action
                    contexts[lane]["forced_action_count"] += 1
                    continue
                _commit_pending_transition(
                    contexts[lane],
                    done=False,
                )
                policy_lanes.append(lane)
                policy_observations.append(observation)
                policy_masks.append(action_mask)
            if policy_lanes:
                inference_start = time.perf_counter()
                actions, log_probabilities, values = agent.act_batch(
                    policy_observations,
                    policy_masks,
                )
                diagnostic_rows = agent.consume_policy_decision_diagnostics()
                if diagnostic_rows and len(diagnostic_rows) != len(policy_lanes):
                    raise RuntimeError("training policy diagnostics do not align with lanes")
                inference_time += time.perf_counter() - inference_start
                for local_index, lane in enumerate(policy_lanes):
                    context = contexts[lane]
                    action = actions[local_index]
                    selected_actions[lane] = action
                    context["policy_step_count"] += 1
                    sampled_transitions[lane] = _PendingTransition(
                        observation=policy_observations[local_index],
                        action_mask=policy_masks[local_index],
                        action=action,
                        log_probability=log_probabilities[local_index],
                        value=values[local_index],
                        reward=context["unattributed_forced_reward"],
                    )
                    context["unattributed_forced_reward"] = 0.0
                    if diagnostic_rows:
                        diagnostic = diagnostic_rows[local_index]
                        diagnostic["selected_action"] = int(action)
                        diagnostic["ranker_top_selected"] = bool(
                            int(action)
                            == int(diagnostic.get("relative_top_action", -1))
                        )
                        context["policy_diagnostic_rows"].append(diagnostic)
            step_responses = self._exchange(
                {
                    lane: (
                        "step",
                        _WorkerStepRequest(
                            action=selected_actions[lane],
                            drain_physical_forced_actions=(
                                worker_local_physical_forced_actions
                            ),
                            max_environment_steps=(
                                None
                                if step_limit is None
                                else step_limit
                                - contexts[lane]["step_count"]
                            ),
                        ),
                    )
                    for lane in lanes
                }
            )
            cutoff_lanes: list[int] = []
            for lane in lanes:
                response = step_responses[lane]
                context = contexts[lane]
                if response.reward_vector is None:
                    raise ParallelWorkerError(
                        f"worker {lane} returned no reward"
                    )
                if response.environment_step_count < 1:
                    raise ParallelWorkerError(
                        f"worker {lane} returned no environment steps"
                    )
                if response.local_physical_forced_action_count > (
                    response.environment_step_count - 1
                ):
                    raise ParallelWorkerError(
                        f"worker {lane} returned invalid local step counts"
                    )
                scalar_reward = response.reward_vector.scalarize(
                    self.config["reward"],
                )
                if lane in sampled_transitions:
                    pending = sampled_transitions[lane]
                    pending.reward += scalar_reward
                    context["pending_transition"] = pending
                elif context["pending_transition"] is not None:
                    context["pending_transition"].reward += scalar_reward
                else:
                    context["unattributed_forced_reward"] += scalar_reward
                context["reward_sum"] += scalar_reward
                for name in context["reward_components"]:
                    context["reward_components"][name] += float(
                        getattr(response.reward_vector, name)
                    )
                context["step_count"] += response.environment_step_count
                context["forced_action_count"] += (
                    response.local_physical_forced_action_count
                )
                context["worker_step_command_count"] += 1
                context[
                    "worker_local_physical_forced_action_count"
                ] += response.local_physical_forced_action_count
                for name in EVALUATION_WORKER_TIMING_FIELDS:
                    context[name] += float(getattr(response, name, 0.0))
                if response.terminated:
                    _commit_pending_transition(context, done=True)
                    if response.metrics is None:
                        raise ParallelWorkerError(
                            f"worker {lane} returned no terminal metrics"
                        )
                    context["buffer"].compute_gae(
                        last_value=0.0,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                    )
                    completed.append(
                        self._episode_result(
                            context,
                            response.metrics,
                        )
                    )
                    active.remove(lane)
                    continue
                if (
                    response.truncated
                    or (step_limit is not None and context["step_count"] >= step_limit)
                ):
                    _commit_pending_transition(context, done=False)
                    cutoff_lanes.append(lane)
                else:
                    states[lane] = response
            if cutoff_lanes:
                value_lanes = [
                    lane
                    for lane in cutoff_lanes
                    if len(contexts[lane]["buffer"]) > 0
                ]
                last_value_by_lane = {
                    lane: 0.0 for lane in cutoff_lanes
                }
                if value_lanes:
                    cutoff_observations = [
                        step_responses[lane].observation
                        for lane in value_lanes
                    ]
                    cutoff_masks = [
                        step_responses[lane].action_mask
                        for lane in value_lanes
                    ]
                    if any(
                        value is None
                        for value in cutoff_observations + cutoff_masks
                    ):
                        raise ParallelWorkerError(
                            "cutoff worker returned an incomplete value state"
                        )
                    inference_start = time.perf_counter()
                    cutoff_values = agent.value_batch(
                        cutoff_observations,
                        cutoff_masks,
                    )
                    inference_time += time.perf_counter() - inference_start
                    last_value_by_lane.update(
                        zip(value_lanes, cutoff_values)
                    )
                snapshots = self._exchange(
                    {
                        lane: ("snapshot", None)
                        for lane in cutoff_lanes
                    }
                )
                for lane in cutoff_lanes:
                    context = contexts[lane]
                    for name in EVALUATION_WORKER_TIMING_FIELDS:
                        context[name] += float(getattr(snapshots[lane], name, 0.0))
                    context["buffer"].compute_gae(
                        last_value=last_value_by_lane[lane],
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                    )
                    metrics = snapshots[lane].metrics
                    if metrics is None:
                        raise ParallelWorkerError(
                            f"worker {lane} returned no snapshot metrics"
                        )
                    completed.append(
                        self._episode_result(context, metrics)
                    )
                    active.remove(lane)
            fill_available([lane for lane in lanes if lane not in active])
        completed.sort(key=lambda value: value.episode_index)
        combined = RolloutBuffer(
            preserve_graph=agent.requires_graph_observation
        )
        for episode in completed:
            combined.extend(episode.buffer)
        return TrainingRolloutBatch(
            episodes=completed,
            buffer=combined,
            sampling_wall_time_seconds=(
                time.perf_counter() - sampling_start
            ),
            policy_inference_time_seconds=inference_time,
        )

    def _episode_result(
        self,
        context: dict[str, Any],
        metrics: dict[str, Any],
    ) -> EpisodeRollout:
        if context["pending_transition"] is not None:
            raise RuntimeError("episode ended with an uncommitted transition")
        if (
            context["step_count"]
            != context["policy_step_count"]
            + context["forced_action_count"]
        ):
            raise RuntimeError("compressed rollout step accounting diverged")
        if len(context["buffer"]) != context["policy_step_count"]:
            raise RuntimeError("compressed rollout buffer accounting diverged")
        if (
            context["worker_step_command_count"]
            + context["worker_local_physical_forced_action_count"]
            != context["step_count"]
        ):
            raise RuntimeError(
                "worker-local forced rollout accounting diverged"
            )
        attributed_reward = sum(
            transition.reward
            for transition in context["buffer"].transitions
        )
        reward_error = (
            attributed_reward
            + context["unattributed_forced_reward"]
            - context["reward_sum"]
        )
        if abs(reward_error) > 1e-8:
            raise RuntimeError(
                "compressed rollout reward attribution diverged by "
                f"{reward_error}"
            )
        episode = EpisodeRollout(
            episode_index=context["episode_index"],
            instance_id=context["instance_id"],
            metadata=context["metadata"],
            buffer=context["buffer"],
            reward_sum=context["reward_sum"],
            step_count=context["step_count"],
            metrics=metrics,
            generation_time_seconds=context[
                "generation_time_seconds"
            ],
            environment_step_time_seconds=context[
                "environment_step_time_seconds"
            ],
            **{name: context.get(name, 0.0)
               for name in EVALUATION_WORKER_TIMING_FIELDS if name != "environment_step_time_seconds"},
            reward_components=dict(context["reward_components"]),
            expected_reward=proxy_return_from_metrics(
                metrics,
                self.config,
                preference=metrics.get("preference"),
            ),
            unattributed_forced_reward=context[
                "unattributed_forced_reward"
            ],
            worker_step_command_count=context[
                "worker_step_command_count"
            ],
            worker_local_physical_forced_action_count=context[
                "worker_local_physical_forced_action_count"
            ],
            policy_diagnostics=summarize_policy_decision_diagnostics(
                context["policy_diagnostic_rows"]
            ),
        )
        reward_identity_tolerance = 1e-8
        reward_identity_error = (
            episode.unshaped_reward_sum - episode.expected_reward
        )
        if abs(reward_identity_error) > reward_identity_tolerance:
            raise RuntimeError(
                "trajectory reward identity diverged by "
                f"{reward_identity_error}"
            )
        return episode

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
    ) -> list[FixedEvaluationRollout]:
        # Reject incompatible agents before starting any evaluation bookkeeping.
        agent.assert_evaluation_config(self.config)
        with self._track_evaluation_runtime():
            return self._evaluate_records(
                agent, records, max_parallelism=max_parallelism,
                deterministic=deterministic, sampling_seed=sampling_seed, preferences=preferences)

    def _evaluate_records(
        self,
        agent: "PPOAgent",
        records: Sequence[GeneratedInstanceRecord],
        *,
        max_parallelism: int | None = None,
        deterministic: bool = True,
        sampling_seed: int | None = None,
        preferences: Sequence[PreferenceContextInput] | None = None,
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
        if not deterministic and sampling_seed is None:
            raise ValueError(
                "sampling_seed is required for sampled fixed evaluation"
            )
        if preferences is not None and len(preferences) != len(records):
            raise ValueError("evaluation preferences must align with records")
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
                    derived_sampling_seeds[lane] = derive_evaluation_sampling_seed(
                        int(sampling_seed), records[index].instance.instance_id,
                        evaluation_keys[lane],
                    )
                    generators[lane] = torch.Generator(device=agent.device).manual_seed(
                        derived_sampling_seeds[lane]
                    )
                requests[lane] = (
                    "reset_instance",
                    _WorkerResetRequest(value=records[index].instance, preference=preference),
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
                                None if deterministic else int(sampling_seed)
                            ),
                            derived_sampling_seed=(
                                None
                                if deterministic
                                else derived_sampling_seeds[lane]
                            ),
                            sampling_evaluation_key=evaluation_keys[lane],
                            **worker_times[lane],
                        )
                    )
                    active.remove(lane)
                else:
                    states[lane] = response
            fill_available([lane for lane in range(parallelism) if lane not in active])
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
