from __future__ import annotations

import multiprocessing
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, TYPE_CHECKING

import numpy as np

from agent.ppo.buffer import RolloutBuffer
from agent.ppo.network import network_requires_graph_observation
from data.dataset import GeneratedInstanceRecord, OnlineInstanceDataset
from data.models import AssemblyInstance
from environment import (
    AssemblySchedulingEnv,
    Observation,
    PolicyObservation,
    RewardVector,
    proxy_return_from_metrics,
)

if TYPE_CHECKING:
    from agent.ppo.agent import PPOAgent


@dataclass
class WorkerResponse:
    lane_id: int
    observation: Observation | PolicyObservation | None = None
    action_mask: np.ndarray | None = None
    reward_vector: RewardVector | None = None
    terminated: bool = False
    truncated: bool = False
    metrics: dict[str, Any] | None = None
    instance_id: str | None = None
    metadata: dict[str, Any] | None = None
    generation_time_seconds: float = 0.0
    environment_step_time_seconds: float = 0.0


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
    reward_phase: str = "legacy"
    reward_components: dict[str, float] = field(default_factory=dict)
    expected_reward: float = 0.0


@dataclass
class TrainingRolloutBatch:
    episodes: list[EpisodeRollout]
    buffer: RolloutBuffer
    sampling_wall_time_seconds: float
    policy_inference_time_seconds: float

    @property
    def transition_count(self) -> int:
        return len(self.buffer)


@dataclass
class FixedEvaluationRollout:
    record_index: int
    metrics: dict[str, Any]
    decisions: int
    inference_time_seconds: float
    solve_time_seconds: float


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
    policy_observation = (
        observation.copy()
        if preserve_graph
        else PolicyObservation.from_observation(observation)
    )
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
                generation_start = time.perf_counter()
                record = dataset[int(payload)]
                generation_time = time.perf_counter() - generation_start
                observation = environment.reset(record.instance)
                metadata = {
                    key: record.metadata.get(key)
                    for key in (
                        "seed",
                        "pressure_type",
                        "cost_profile",
                    )
                }
                connection.send(
                    _worker_state(
                        lane_id,
                        environment,
                        observation,
                        preserve_graph=preserve_graph,
                        instance_id=record.instance.instance_id,
                        metadata=metadata,
                        generation_time_seconds=generation_time,
                    )
                )
                continue
            if command == "reset_instance":
                observation = environment.reset(payload)
                connection.send(
                    _worker_state(
                        lane_id,
                        environment,
                        observation,
                        preserve_graph=preserve_graph,
                        instance_id=payload.instance_id,
                    )
                )
                continue
            if command == "step":
                step_start = time.perf_counter()
                observation, reward, terminated, truncated, _ = (
                    environment.step(int(payload))
                )
                step_time = time.perf_counter() - step_start
                if terminated or truncated:
                    connection.send(
                        WorkerResponse(
                            lane_id=lane_id,
                            reward_vector=reward,
                            terminated=terminated,
                            truncated=truncated,
                            metrics=_terminal_metrics(environment),
                            environment_step_time_seconds=step_time,
                        )
                    )
                else:
                    connection.send(
                        _worker_state(
                            lane_id,
                            environment,
                            observation,
                            preserve_graph=preserve_graph,
                            reward_vector=reward,
                            environment_step_time_seconds=step_time,
                        )
                    )
                continue
            if command == "snapshot":
                connection.send(
                    WorkerResponse(
                        lane_id=lane_id,
                        metrics=_terminal_metrics(environment),
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
    ):
        if worker_count < 2:
            raise ValueError("parallel runner requires at least two workers")
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
        self.timeout_seconds = float(
            training["worker_timeout_seconds"]
        )
        if self.timeout_seconds <= 0:
            raise ValueError("worker_timeout_seconds must be positive")
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
            self._receive_responses(range(self.worker_count))
        except BaseException:
            self.close(force=True)
            raise

    def __enter__(self) -> "ParallelEpisodeRunner":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close(force=exc_type is not None)

    def _exchange(
        self,
        commands: dict[int, tuple[str, Any]],
    ) -> dict[int, WorkerResponse]:
        if self._closed:
            raise RuntimeError("parallel runner is closed")
        for lane_id, message in commands.items():
            process = self._processes[lane_id]
            if not process.is_alive():
                raise ParallelWorkerError(
                    f"worker {lane_id} exited with code "
                    f"{process.exitcode}"
                )
            self._connections[lane_id].send(message)
        return self._receive_responses(commands)

    def _receive_responses(
        self,
        lane_ids,
    ) -> dict[int, WorkerResponse]:
        pending = {
            self._connections[lane_id]: lane_id
            for lane_id in lane_ids
        }
        responses: dict[int, WorkerResponse] = {}
        deadline = time.monotonic() + self.timeout_seconds
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                lanes = sorted(pending.values())
                raise ParallelWorkerTimeout(
                    f"workers {lanes} did not respond within "
                    f"{self.timeout_seconds:.1f} seconds"
                )
            ready = wait(list(pending), timeout=remaining)
            if not ready:
                continue
            for connection in ready:
                lane_id = pending.pop(connection)
                try:
                    response = connection.recv()
                except EOFError as error:
                    process = self._processes[lane_id]
                    raise ParallelWorkerError(
                        f"worker {lane_id} closed its pipe; exit code "
                        f"{process.exitcode}"
                    ) from error
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
                responses[lane_id] = response
        return responses

    def collect_training_batch(
        self,
        agent: "PPOAgent",
        episode_indices: Sequence[int],
        *,
        gamma: float,
        gae_lambda: float,
        step_limit: int | None = None,
        reward_phase: str | None = None,
    ) -> TrainingRolloutBatch:
        if not episode_indices:
            raise ValueError("episode_indices cannot be empty")
        if len(episode_indices) > self.worker_count:
            raise ValueError("episode batch exceeds worker count")
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode indices must be unique")
        effective_reward_phase = (
            "feasibility"
            if reward_phase is None
            and str(
                self.config["reward"].get(
                    "mode",
                    "legacy_weighted_sum",
                )
            )
            == "hierarchical_constrained_v1"
            else "legacy"
            if reward_phase is None
            else str(reward_phase)
        )
        sampling_start = time.perf_counter()
        reset_responses = self._exchange(
            {
                lane_id: ("reset_online", int(episode_index))
                for lane_id, episode_index in enumerate(episode_indices)
            }
        )
        states: dict[int, WorkerResponse] = dict(reset_responses)
        contexts: dict[int, dict[str, Any]] = {}
        for lane_id, episode_index in enumerate(episode_indices):
            response = reset_responses[lane_id]
            if (
                response.observation is None
                or response.action_mask is None
                or response.instance_id is None
                or response.metadata is None
            ):
                raise ParallelWorkerError(
                    f"worker {lane_id} returned an incomplete reset"
                )
            contexts[lane_id] = {
                "episode_index": int(episode_index),
                "instance_id": response.instance_id,
                "metadata": response.metadata,
                "buffer": RolloutBuffer(
                    preserve_graph=agent.requires_graph_observation
                ),
                "reward_sum": 0.0,
                "reward_phase": effective_reward_phase,
                "reward_components": {
                    "flow": 0.0,
                    "cost": 0.0,
                    "variance": 0.0,
                    "completion_progress": 0.0,
                    "completion_bonus": 0.0,
                    "quality": 0.0,
                },
                "step_count": 0,
                "generation_time_seconds": (
                    response.generation_time_seconds
                ),
                "environment_step_time_seconds": 0.0,
            }
        active = set(contexts)
        completed: list[EpisodeRollout] = []
        inference_time = 0.0
        while active:
            lanes = sorted(active)
            observations = [states[lane].observation for lane in lanes]
            masks = [states[lane].action_mask for lane in lanes]
            if any(value is None for value in observations + masks):
                raise ParallelWorkerError(
                    "active worker returned an incomplete policy state"
                )
            inference_start = time.perf_counter()
            actions, log_probabilities, values = agent.act_batch(
                observations,
                masks,
            )
            inference_time += time.perf_counter() - inference_start
            step_responses = self._exchange(
                {
                    lane: ("step", action)
                    for lane, action in zip(lanes, actions)
                }
            )
            cutoff_lanes: list[int] = []
            for local_index, lane in enumerate(lanes):
                previous = states[lane]
                response = step_responses[lane]
                context = contexts[lane]
                if response.reward_vector is None:
                    raise ParallelWorkerError(
                        f"worker {lane} returned no reward"
                    )
                scalar_reward = response.reward_vector.scalarize(
                    self.config["reward"],
                    effective_reward_phase,
                )
                done = response.terminated or response.truncated
                context["buffer"].add(
                    previous.observation,
                    previous.action_mask,
                    actions[local_index],
                    log_probabilities[local_index],
                    values[local_index],
                    scalar_reward,
                    done,
                )
                context["reward_sum"] += scalar_reward
                for name, value in response.reward_vector.as_dict().items():
                    context["reward_components"][name] += float(value)
                context["step_count"] += 1
                context["environment_step_time_seconds"] += (
                    response.environment_step_time_seconds
                )
                if done:
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
                    step_limit is not None
                    and context["step_count"] >= step_limit
                ):
                    cutoff_lanes.append(lane)
                else:
                    states[lane] = response
            if cutoff_lanes:
                cutoff_observations = [
                    step_responses[lane].observation
                    for lane in cutoff_lanes
                ]
                cutoff_masks = [
                    step_responses[lane].action_mask
                    for lane in cutoff_lanes
                ]
                inference_start = time.perf_counter()
                last_values = agent.value_batch(
                    cutoff_observations,
                    cutoff_masks,
                )
                inference_time += time.perf_counter() - inference_start
                snapshots = self._exchange(
                    {
                        lane: ("snapshot", None)
                        for lane in cutoff_lanes
                    }
                )
                for lane, last_value in zip(
                    cutoff_lanes,
                    last_values,
                ):
                    context = contexts[lane]
                    context["buffer"].compute_gae(
                        last_value=last_value,
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
        return EpisodeRollout(
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
            reward_phase=context["reward_phase"],
            reward_components=dict(context["reward_components"]),
            expected_reward=proxy_return_from_metrics(
                metrics,
                self.config["reward"],
                context["reward_phase"],
            ),
        )

    def evaluate_records(
        self,
        agent: "PPOAgent",
        records: Sequence[GeneratedInstanceRecord],
        *,
        max_parallelism: int | None = None,
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
        results: list[FixedEvaluationRollout] = []
        for start in range(0, len(records), parallelism):
            chunk = records[start : start + parallelism]
            chunk_start = time.perf_counter()
            reset_responses = self._exchange(
                {
                    lane_id: ("reset_instance", record.instance)
                    for lane_id, record in enumerate(chunk)
                }
            )
            states = dict(reset_responses)
            active = set(range(len(chunk)))
            decisions = {lane: 0 for lane in active}
            inference_times = {lane: 0.0 for lane in active}
            while active:
                lanes = sorted(active)
                observations = [
                    states[lane].observation for lane in lanes
                ]
                masks = [
                    states[lane].action_mask for lane in lanes
                ]
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
                step_responses = self._exchange(
                    {
                        lane: ("step", action)
                        for lane, action in zip(lanes, actions)
                    }
                )
                for lane in lanes:
                    decisions[lane] += 1
                    response = step_responses[lane]
                    if response.terminated or response.truncated:
                        if response.metrics is None:
                            raise ParallelWorkerError(
                                f"worker {lane} returned no metrics"
                            )
                        results.append(
                            FixedEvaluationRollout(
                                record_index=start + lane,
                                metrics=response.metrics,
                                decisions=decisions[lane],
                                inference_time_seconds=(
                                    inference_times[lane]
                                ),
                                solve_time_seconds=(
                                    time.perf_counter() - chunk_start
                                ),
                            )
                        )
                        active.remove(lane)
                    else:
                        states[lane] = response
        results.sort(key=lambda value: value.record_index)
        return results

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
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
