from __future__ import annotations

import multiprocessing
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
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
    PolicyObservation,
    PreferenceInput,
    PreferenceVector,
    RewardVector,
    derive_episode_action_seed,
    normalize_preference,
    preference_enabled,
    proxy_return_from_metrics,
    sample_episode_preference,
)
from utils import action_trace_sha256, derive_evaluation_sampling_seed

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
    environment_step_count: int = 0
    local_physical_forced_action_count: int = 0


@dataclass(frozen=True)
class _WorkerResetRequest:
    value: int | AssemblyInstance
    preference: PreferenceVector | None = None
    drain_physical_forced_actions: bool = False
    max_environment_steps: int | None = None


@dataclass(frozen=True)
class _WorkerStepRequest:
    action: int
    drain_physical_forced_actions: bool = False
    max_environment_steps: int | None = None


@dataclass(frozen=True)
class TrainingEpisodeAssignment:
    trajectory_index: int
    base_instance_index: int
    preference_slot: int
    preference_group_id: int
    preference: PreferenceVector
    preference_source: str


def training_preference_group(config: Mapping[str, Any]) -> dict[str, Any] | None:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("config.training must be an object")
    raw = training.get("preference_grouping")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("training.preference_grouping must be an object")
    if not bool(raw.get("enabled", False)):
        return None
    version = str(raw.get("version", "fixed_anchor_group_v1"))
    if version != "fixed_anchor_group_v1":
        raise ValueError(
            "training.preference_grouping.version must be "
            "'fixed_anchor_group_v1'"
        )
    anchors_raw = raw.get("anchors")
    if not isinstance(anchors_raw, Sequence) or isinstance(
        anchors_raw, (str, bytes)
    ):
        raise TypeError("training.preference_grouping.anchors must be a sequence")
    anchors = tuple(normalize_preference(value) for value in anchors_raw)
    if len(anchors) < 2:
        raise ValueError("grouped preference training requires at least two anchors")
    return {"version": version, "anchors": anchors, "group_size": len(anchors)}


def training_base_instance_count(
    config: Mapping[str, Any], trajectory_count: int
) -> int:
    count = int(trajectory_count)
    if count < 1:
        raise ValueError("trajectory_count must be positive")
    grouping = training_preference_group(config)
    if grouping is None:
        return count
    group_size = int(grouping["group_size"])
    if count % group_size:
        raise ValueError(
            "grouped preference trajectory count must be divisible by group size"
        )
    return count // group_size


def training_episode_assignment(
    config: Mapping[str, Any], trajectory_index: int
) -> TrainingEpisodeAssignment:
    index = int(trajectory_index)
    if index < 0:
        raise ValueError("trajectory_index must be non-negative")
    grouping = training_preference_group(config)
    if grouping is None:
        preference, source = sample_episode_preference(
            config,
            algorithm_seed=int(config["seed"]),
            episode_index=index,
        )
        return TrainingEpisodeAssignment(
            trajectory_index=index,
            base_instance_index=index,
            preference_slot=-1,
            preference_group_id=index,
            preference=preference,
            preference_source=source,
        )
    anchors: tuple[PreferenceVector, ...] = grouping["anchors"]
    group_size = int(grouping["group_size"])
    base_instance_index, preference_slot = divmod(index, group_size)
    return TrainingEpisodeAssignment(
        trajectory_index=index,
        base_instance_index=base_instance_index,
        preference_slot=preference_slot,
        preference_group_id=base_instance_index,
        preference=anchors[preference_slot],
        preference_source=f"group_anchor_{preference_slot}",
    )


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
    preference: PreferenceVector = field(
        default_factory=lambda: PreferenceVector(0.5, 0.3, 0.2)
    )
    preference_source: str = "fixed_default"
    reward_phase: str = "legacy"
    reward_components: dict[str, float] = field(default_factory=dict)
    expected_reward: float = 0.0
    unattributed_forced_reward: float = 0.0
    worker_step_command_count: int = 0
    worker_local_physical_forced_action_count: int = 0
    base_instance_index: int | None = None
    preference_slot: int = -1
    preference_group_id: int | None = None

    @property
    def base_reward_sum(self) -> float:
        return self.reward_sum - float(
            self.reward_components.get("feasibility_shaping", 0.0)
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
    observation: Observation | PolicyObservation
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
    """Return a physically forced action that is safe to execute locally.

    A singleton created by the non-delay worker-dispatch rule remains visible
    to the parent process.  It is still compressed there, but is deliberately
    excluded from worker-local chaining so policy masking is not mistaken for
    physical determinism.
    """

    action = forced_action_from_mask(action_mask)
    if action is None:
        return None
    diagnostic = environment.forced_action_diagnostic(action_mask)
    if diagnostic is None:
        raise RuntimeError(
            "singleton action mask has no forced-action diagnostic"
        )
    if bool(diagnostic["non_delay_blocked_advance"]):
        return None
    if not (
        bool(diagnostic["advance_physically_unavailable"])
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
    }
    for reward_vector in reward_vectors:
        for name, value in reward_vector.as_dict().items():
            totals[name] += float(value)
    return RewardVector(**totals)


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
    if terminated or truncated:
        return WorkerResponse(
            lane_id=lane_id,
            terminated=terminated,
            truncated=truncated,
            metrics=_terminal_metrics(environment),
            **response_kwargs,
        )
    if environment_step_count > 0:
        observation = environment.observe()
    if observation is None:
        raise RuntimeError("active worker has no observation")
    return _worker_state(
        lane_id,
        environment,
        observation,
        preserve_graph=preserve_graph,
        **response_kwargs,
    )


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
                request = (
                    payload
                    if isinstance(payload, _WorkerResetRequest)
                    else _WorkerResetRequest(value=int(payload))
                )
                generation_start = time.perf_counter()
                record = dataset[int(request.value)]
                generation_time = time.perf_counter() - generation_start
                observation = environment.reset(
                    record.instance,
                    preference=request.preference,
                )
                metadata = {
                    key: record.metadata.get(key)
                    for key in (
                        "seed",
                        "pressure_type",
                        "cost_profile",
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
                observation = environment.reset(
                    request.value,
                    preference=request.preference,
                )
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
        assignments = {
            int(episode_index): training_episode_assignment(
                self.config, int(episode_index)
            )
            for episode_index in episode_indices
        }
        reset_responses = self._exchange(
            {
                lane_id: (
                    "reset_online",
                    _WorkerResetRequest(
                        value=assignments[
                            int(episode_index)
                        ].base_instance_index,
                        preference=assignments[int(episode_index)].preference,
                        drain_physical_forced_actions=(
                            worker_local_physical_forced_actions
                        ),
                        max_environment_steps=step_limit,
                    ),
                )
                for lane_id, episode_index in enumerate(episode_indices)
            }
        )
        states: dict[int, WorkerResponse] = dict(reset_responses)
        contexts: dict[int, dict[str, Any]] = {}
        active: set[int] = set()
        completed: list[EpisodeRollout] = []
        reset_cutoff_lanes: list[int] = []
        for lane_id, episode_index in enumerate(episode_indices):
            assignment = assignments[int(episode_index)]
            response = reset_responses[lane_id]
            if (
                response.instance_id is None
                or response.metadata is None
            ):
                raise ParallelWorkerError(
                    f"worker {lane_id} returned an incomplete reset"
                )
            context = {
                "episode_index": int(episode_index),
                "base_instance_index": assignment.base_instance_index,
                "preference_slot": assignment.preference_slot,
                "preference_group_id": assignment.preference_group_id,
                "instance_id": response.instance_id,
                "metadata": response.metadata,
                "preference": assignment.preference,
                "preference_source": assignment.preference_source,
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
                    "truncation": 0.0,
                    "unfinished": 0.0,
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
                "unattributed_forced_reward": 0.0,
                "generation_time_seconds": (
                    response.generation_time_seconds
                ),
                "environment_step_time_seconds": (
                    response.environment_step_time_seconds
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
                    effective_reward_phase,
                )
                context["reward_sum"] += scalar_reward
                context["unattributed_forced_reward"] += scalar_reward
                for name, value in response.reward_vector.as_dict().items():
                    context["reward_components"][name] += float(value)
            elif response.environment_step_count:
                raise ParallelWorkerError(
                    "reset worker returned steps without rewards"
                )
            done = response.terminated or response.truncated
            if done:
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
                continue
            if response.observation is None or response.action_mask is None:
                raise ParallelWorkerError(
                    f"worker {lane_id} returned no active reset state"
                )
            if (
                step_limit is not None
                and context["step_count"] >= step_limit
            ):
                reset_cutoff_lanes.append(lane_id)
            else:
                active.add(lane_id)
        if reset_cutoff_lanes:
            snapshots = self._exchange(
                {
                    lane: ("snapshot", None)
                    for lane in reset_cutoff_lanes
                }
            )
            for lane in reset_cutoff_lanes:
                context = contexts[lane]
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
        inference_time = 0.0
        action_generators = (
            {
                lane: torch.Generator(device=agent.device).manual_seed(
                    derive_episode_action_seed(
                        int(self.config["seed"]),
                        int(contexts[lane]["episode_index"]),
                    )
                )
                for lane in contexts
            }
            if preference_enabled(self.config)
            else {}
        )
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
                if action_generators:
                    sampled = [
                        agent.act(
                            observation,
                            mask,
                            generator=action_generators[lane],
                        )
                        for lane, observation, mask in zip(
                            policy_lanes,
                            policy_observations,
                            policy_masks,
                            strict=True,
                        )
                    ]
                    actions = [item[0] for item in sampled]
                    log_probabilities = [item[1] for item in sampled]
                    values = [item[2] for item in sampled]
                else:
                    actions, log_probabilities, values = agent.act_batch(
                        policy_observations,
                        policy_masks,
                    )
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
                    effective_reward_phase,
                )
                done = response.terminated or response.truncated
                if lane in sampled_transitions:
                    pending = sampled_transitions[lane]
                    pending.reward += scalar_reward
                    context["pending_transition"] = pending
                elif context["pending_transition"] is not None:
                    context["pending_transition"].reward += scalar_reward
                else:
                    context["unattributed_forced_reward"] += scalar_reward
                context["reward_sum"] += scalar_reward
                for name, value in response.reward_vector.as_dict().items():
                    context["reward_components"][name] += float(value)
                context["step_count"] += response.environment_step_count
                context["forced_action_count"] += (
                    response.local_physical_forced_action_count
                )
                context["worker_step_command_count"] += 1
                context[
                    "worker_local_physical_forced_action_count"
                ] += response.local_physical_forced_action_count
                context["environment_step_time_seconds"] += (
                    response.environment_step_time_seconds
                )
                if done:
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
                    step_limit is not None
                    and context["step_count"] >= step_limit
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
            base_instance_index=context["base_instance_index"],
            preference_slot=context["preference_slot"],
            preference_group_id=context["preference_group_id"],
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
            preference=context["preference"],
            preference_source=context["preference_source"],
            reward_phase=context["reward_phase"],
            reward_components=dict(context["reward_components"]),
            expected_reward=proxy_return_from_metrics(
                metrics,
                self.config["reward"],
                context["reward_phase"],
                preference=context["preference"],
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
        )
        reward_identity_tolerance = float(
            self.config["training"]
            .get("ablation_gate", {})
            .get("reward_identity_tolerance", 1e-8)
        )
        reward_identity_error = (
            episode.base_reward_sum - episode.expected_reward
        )
        if abs(reward_identity_error) > reward_identity_tolerance:
            raise RuntimeError(
                "trajectory reward identity diverged by "
                f"{reward_identity_error}"
            )
        return episode

    def evaluate_records(
        self,
        agent: "PPOAgent",
        records: Sequence[GeneratedInstanceRecord],
        *,
        max_parallelism: int | None = None,
        deterministic: bool = True,
        sampling_seed: int | None = None,
        preference: PreferenceInput | None = None,
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
        evaluation_preference = (
            None
            if preference is None
            else normalize_preference(preference)
        )
        results: list[FixedEvaluationRollout] = []
        for start in range(0, len(records), parallelism):
            chunk = records[start : start + parallelism]
            chunk_start = time.perf_counter()
            reset_responses = self._exchange(
                {
                    lane_id: (
                        "reset_instance",
                        _WorkerResetRequest(
                            value=record.instance,
                            preference=evaluation_preference,
                        ),
                    )
                    for lane_id, record in enumerate(chunk)
                }
            )
            states = dict(reset_responses)
            active = set(range(len(chunk)))
            decisions = {lane: 0 for lane in active}
            inference_times = {lane: 0.0 for lane in active}
            action_traces: dict[int, list[int]] = {
                lane: [] for lane in active
            }
            policy_diagnostics: dict[int, list[dict[str, Any]]] = {
                lane: [] for lane in active
            }
            generators = (
                {
                    lane: torch.Generator(device=agent.device).manual_seed(
                        derive_evaluation_sampling_seed(
                            int(sampling_seed),
                            chunk[lane].instance.instance_id,
                        )
                    )
                    for lane in active
                }
                if not deterministic
                else {}
            )
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
                    for lane, action, diagnostic, mask in zip(
                        lanes, actions, diagnostic_rows, masks
                    ):
                        diagnostic["selected_action"] = int(action)
                        diagnostic["ranker_top_selected"] = bool(
                            int(action)
                            == int(diagnostic.get("relative_top_action", -1))
                        )
                        diagnostic["unsafe_worker_preference_selected"] = bool(
                            str(diagnostic.get("decision_type", ""))
                            == "WORKER"
                            and int(action) < len(mask)
                            and bool(mask[int(action)])
                        )
                        policy_diagnostics[lane].append(diagnostic)
                else:
                    actions = []
                    for lane, observation, mask in zip(
                        lanes, observations, masks
                    ):
                        inference_start = time.perf_counter()
                        action, _, _ = agent.act(
                            observation,
                            mask,
                            deterministic=False,
                            generator=generators[lane],
                        )
                        inference_times[lane] += (
                            time.perf_counter() - inference_start
                        )
                        actions.append(action)
                        diagnostic_rows = (
                            agent.consume_policy_decision_diagnostics()
                        )
                        for diagnostic in diagnostic_rows:
                            diagnostic["selected_action"] = int(action)
                            diagnostic["ranker_top_selected"] = bool(
                                int(action)
                                == int(
                                    diagnostic.get("relative_top_action", -1)
                                )
                            )
                            diagnostic[
                                "unsafe_worker_preference_selected"
                            ] = bool(
                                str(diagnostic.get("decision_type", ""))
                                == "WORKER"
                                and int(action) < len(mask)
                                and bool(mask[int(action)])
                            )
                            policy_diagnostics[lane].append(diagnostic)
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
                                record_index=start + lane,
                                metrics=response.metrics,
                                decisions=decisions[lane],
                                inference_time_seconds=(
                                    inference_times[lane]
                                ),
                                solve_time_seconds=(
                                    time.perf_counter() - chunk_start
                                ),
                                action_trace_sha256=action_trace_sha256(
                                    action_traces[lane]
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
