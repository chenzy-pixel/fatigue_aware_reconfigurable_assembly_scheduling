from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.baselines import HeuristicPolicy
from configs import load_config, project_path
from data import load_dataset_split
from environment import (
    AssemblySchedulingEnv,
    DecisionType,
    FAILURE_PENALTY_REWARD,
    LEGACY_PROGRESS_QUALITY_REWARD,
    proxy_return_from_metrics,
    terminal_failure_penalty,
)
from eval import EvaluationPolicy
from result.io import write_json
from utils import action_trace_sha256, set_seed


def _load_effective_config(path: str | Path) -> dict[str, Any]:
    """Load either a source config or the fully resolved config saved by a run."""

    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "extends" in raw:
        return load_config(config_path)
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be an object")
    return raw


def _describe_action(
    environment: AssemblySchedulingEnv,
    action: int,
    *,
    phase: DecisionType | None = None,
) -> dict[str, Any]:
    phase = environment.decision_type if phase is None else phase
    description: dict[str, Any] = {
        "index": int(action),
        "phase": phase.value,
    }
    if action == environment.wait_action:
        description["kind"] = "WAIT"
        description["wait_certificate"] = dict(
            environment._last_wait_certificate or {}
        )
        return description
    if phase == DecisionType.PRODUCTION:
        operation_index, machine_index = environment.decode_production_action(action)
        operation = environment.operations[operation_index]
        machine = environment.machines[machine_index]
        description.update(
            {
                "kind": environment._action_type(phase, action),
                "operation_index": operation_index,
                "operation_id": operation.spec.id,
                "order_id": operation.spec.order_id,
                "required_module": operation.spec.required_module,
                "machine_index": machine_index,
                "machine_id": machine.spec.id,
                "machine_module": machine.current_module,
            }
        )
        return description
    machine_index, worker_index = environment.decode_worker_action(action)
    machine = environment.machines[machine_index]
    worker = environment.workers[worker_index]
    reconfiguration = environment._pending_reconfiguration(machine.spec.id)
    description.update(
        {
            "kind": "WORKER_ASSIGN",
            "machine_index": machine_index,
            "machine_id": machine.spec.id,
            "worker_index": worker_index,
            "worker_id": worker.spec.id,
            "reconfiguration_id": (
                reconfiguration.id if reconfiguration is not None else None
            ),
            "reconfiguration_stage": (
                reconfiguration.stage.value
                if reconfiguration is not None
                else None
            ),
        }
    )
    return description


def _worker_stage_check(
    environment: AssemblySchedulingEnv,
    reconfiguration,
    worker,
) -> dict[str, Any]:
    module = (
        reconfiguration.source_module
        if reconfiguration.stage.value == "WAIT_DIS"
        else reconfiguration.target_module
    )
    qualified = module in worker.spec.qualified_modules
    idle = worker.state.value == "IDLE"
    duration_ticks = None
    predicted_fatigue = None
    safe = False
    if idle and qualified:
        duration_ticks = environment._stage_duration_ticks(reconfiguration, worker)
        predicted_fatigue = worker.fatigue + environment._stage_accumulation_rate(
            reconfiguration
        ) * (duration_ticks * environment.resolution)
        safe = environment._worker_can_start(reconfiguration, worker)
    blockers = []
    if not idle:
        blockers.append("worker_busy")
    if not qualified:
        blockers.append("unqualified")
    if idle and qualified and not safe:
        blockers.append("unsafe_predicted_fatigue")
    return {
        "worker_id": worker.spec.id,
        "state": worker.state.value,
        "fatigue": float(worker.fatigue),
        "busy_until_tick": worker.busy_until_tick,
        "qualified": qualified,
        "duration_ticks": duration_ticks,
        "predicted_fatigue": predicted_fatigue,
        "safe": safe,
        "blockers": blockers,
    }


def _unfinished_operation_checks(
    environment: AssemblySchedulingEnv,
) -> list[dict[str, Any]]:
    checks = []
    for operation in environment.operations:
        if operation.state.value == "DONE":
            continue
        order = environment._order_by_id(operation.spec.order_id)
        order_operations = list(order.operations)
        position = next(
            index
            for index, candidate in enumerate(order_operations)
            if candidate.id == operation.spec.id
        )
        predecessor = (
            None
            if position == 0
            else environment._operation_by_id(order_operations[position - 1].id)
        )
        machine_checks = []
        for machine in environment.machines:
            blockers = []
            if operation.state.value != "READY":
                blockers.append("operation_not_ready")
            if machine.state.value != "IDLE":
                blockers.append("machine_not_idle")
            if machine.current_module == environment.instance.no_module_state:
                blockers.append("machine_has_no_module")
            if operation.spec.required_module not in machine.spec.module_parameters:
                blockers.append("required_module_unsupported")
            machine_checks.append(
                {
                    "machine_id": machine.spec.id,
                    "state": machine.state.value,
                    "current_module": machine.current_module,
                    "busy_until_tick": machine.busy_until_tick,
                    "locked_operation_id": machine.locked_operation_id,
                    "direct_process": bool(
                        not blockers
                        and machine.current_module == operation.spec.required_module
                    ),
                    "reconfiguration_candidate": bool(
                        not blockers
                        and machine.current_module != operation.spec.required_module
                    ),
                    "blockers": blockers,
                }
            )
        checks.append(
            {
                "operation_id": operation.spec.id,
                "order_id": operation.spec.order_id,
                "sequence": operation.spec.sequence,
                "state": operation.state.value,
                "required_module": operation.spec.required_module,
                "assigned_machine_id": operation.machine_id,
                "predecessor": (
                    None
                    if predecessor is None
                    else {
                        "operation_id": predecessor.spec.id,
                        "state": predecessor.state.value,
                        "done": predecessor.state.value == "DONE",
                    }
                ),
                "qualified_workers": [
                    worker.spec.id
                    for worker in environment.workers
                    if operation.spec.required_module
                    in worker.spec.qualified_modules
                ],
                "machine_checks": machine_checks,
            }
        )
    return checks


def snapshot_environment(
    environment: AssemblySchedulingEnv,
    *,
    include_mask: bool = True,
) -> dict[str, Any]:
    action_mask = None
    legal_actions: list[dict[str, Any]] = []
    if include_mask and environment.decision_type != DecisionType.TERMINAL:
        action_mask = environment.get_action_mask()
        legal_actions = [
            _describe_action(environment, int(action))
            for action in np.flatnonzero(~action_mask)
        ]
    events = [
        {
            "tick": int(tick),
            "time": float(tick * environment.resolution),
            "priority": int(priority),
            "serial": int(serial),
            "event_type": event_type.value,
            "payload": dict(payload),
            "within_horizon": bool(tick <= environment.horizon_tick),
        }
        for tick, priority, serial, event_type, payload in sorted(
            environment._events
        )
    ]
    pending_reconfigurations = []
    for reconfiguration in sorted(
        environment.reconfigurations.values(), key=lambda value: value.id
    ):
        if reconfiguration.stage.value == "DONE":
            continue
        pending_reconfigurations.append(
            {
                "id": reconfiguration.id,
                "machine_id": reconfiguration.machine_id,
                "operation_id": reconfiguration.operation_id,
                "source_module": reconfiguration.source_module,
                "target_module": reconfiguration.target_module,
                "stage": reconfiguration.stage.value,
                "lock_tick": reconfiguration.lock_tick,
                "disassembly_worker_id": reconfiguration.disassembly_worker_id,
                "installation_worker_id": reconfiguration.installation_worker_id,
                "disassembly_end_tick": reconfiguration.disassembly_end_tick,
                "installation_end_tick": reconfiguration.installation_end_tick,
                "worker_checks": [
                    _worker_stage_check(environment, reconfiguration, worker)
                    for worker in environment.workers
                ]
                if reconfiguration.stage.value in {"WAIT_DIS", "WAIT_INS"}
                else [],
            }
        )
    return {
        "state_version": int(environment._state_version),
        "tick": int(environment.current_tick),
        "time": float(environment.current_time),
        "horizon_tick": int(environment.horizon_tick),
        "remaining_horizon_ticks": int(
            environment.horizon_tick - environment.current_tick
        ),
        "decision_type": environment.decision_type.value,
        "terminated": bool(environment.terminated),
        "truncated": bool(environment.truncated),
        "terminal_reason": environment.terminal_reason,
        "operation_progress": float(environment.operation_progress()),
        "action_mask": (
            None if action_mask is None else action_mask.astype(int).tolist()
        ),
        "action_mask_analysis": dict(
            environment._last_action_mask_analysis or {}
        ),
        "legal_actions": legal_actions,
        "operations": [
            {
                "id": operation.spec.id,
                "order_id": operation.spec.order_id,
                "sequence": operation.spec.sequence,
                "required_module": operation.spec.required_module,
                "state": operation.state.value,
                "machine_id": operation.machine_id,
                "start_tick": operation.start_tick,
                "end_tick": operation.end_tick,
            }
            for operation in environment.operations
        ],
        "machines": [
            {
                "id": machine.spec.id,
                "state": machine.state.value,
                "current_module": machine.current_module,
                "busy_until_tick": machine.busy_until_tick,
                "locked_operation_id": machine.locked_operation_id,
                "source_module": machine.source_module,
                "target_module": machine.target_module,
                "supported_modules": sorted(machine.spec.module_parameters),
            }
            for machine in environment.machines
        ],
        "workers": [
            {
                "id": worker.spec.id,
                "state": worker.state.value,
                "fatigue": float(worker.fatigue),
                "peak_fatigue": float(worker.peak_fatigue),
                "load": float(worker.load),
                "busy_until_tick": worker.busy_until_tick,
                "qualified_modules": list(worker.spec.qualified_modules),
            }
            for worker in environment.workers
        ],
        "pending_reconfigurations": pending_reconfigurations,
        "future_events": events,
        "unfinished_operation_checks": _unfinished_operation_checks(environment),
        "completion_estimate_ticks": int(
            environment._remaining_completion_estimate_ticks()
        ),
    }


def _unfinished_order_timing(
    environment: AssemblySchedulingEnv,
    *,
    observation_tick: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize elapsed and remaining work for orders unfinished at a tick.

    Worker waiting is derived from the exact reconfiguration runtime timestamps.
    Processing estimates are machine-minimum lower bounds and are labelled as
    such; they do not claim that a globally feasible continuation exists.
    """

    tick = int(
        environment.current_tick
        if observation_tick is None
        else observation_tick
    )

    def to_minutes(value: int) -> float:
        return float(value * environment.resolution)

    operation_by_id = {
        operation.spec.id: operation for operation in environment.operations
    }
    reconfigurations_by_order: dict[str, list[Any]] = {}
    for reconfiguration in environment.reconfigurations.values():
        operation = operation_by_id[reconfiguration.operation_id]
        reconfigurations_by_order.setdefault(operation.spec.order_id, []).append(
            reconfiguration
        )

    summaries: list[dict[str, Any]] = []
    for order in environment.instance.orders:
        operations = [operation_by_id[spec.id] for spec in order.operations]
        if all(operation.state.value == "DONE" for operation in operations):
            continue

        processing_records = {
            str(record["operation_id"]): record
            for record in environment.schedule_log
            if record["order_id"] == order.id
        }
        processing_elapsed_time = float(
            sum(float(record["duration"]) for record in processing_records.values())
        )
        processing_planned_time = float(
            sum(
                float(record.get("planned_end", record["end"]))
                - float(record["start"])
                for record in processing_records.values()
            )
        )
        remaining_processing_lb_ticks = 0
        remaining_operations = []
        for operation in operations:
            if operation.state.value == "DONE":
                continue
            compatible_ticks = [
                environment.estimate_processing_ticks(
                    environment.operations.index(operation), machine_index
                )
                for machine_index, machine in enumerate(environment.machines)
                if operation.spec.required_module in machine.spec.module_parameters
            ]
            processing_record = processing_records.get(operation.spec.id)
            if processing_record is not None:
                planned_end = float(
                    processing_record.get("planned_end", processing_record["end"])
                )
                remaining_ticks = max(
                    0,
                    int(
                        round(
                            (planned_end - tick * environment.resolution)
                            / environment.resolution
                        )
                    ),
                )
            else:
                remaining_ticks = min(compatible_ticks) if compatible_ticks else None
            if remaining_ticks is not None:
                remaining_processing_lb_ticks += int(remaining_ticks)
            remaining_operations.append(
                {
                    "operation_id": operation.spec.id,
                    "state": operation.state.value,
                    "required_module": operation.spec.required_module,
                    "machine_id": operation.machine_id,
                    "remaining_processing_lower_bound_ticks": remaining_ticks,
                }
            )

        worker_wait_dis_ticks = 0
        worker_wait_ins_ticks = 0
        disassembly_active_ticks = 0
        installation_active_ticks = 0
        reconfiguration_details = []
        for reconfiguration in sorted(
            reconfigurations_by_order.get(order.id, []),
            key=lambda value: value.lock_tick,
        ):
            cap_tick = tick
            dis_start = (
                reconfiguration.disassembly_start_tick
                if reconfiguration.disassembly_start_tick is not None
                else cap_tick
            )
            worker_wait_dis_ticks += max(
                0, min(dis_start, cap_tick) - reconfiguration.lock_tick
            )
            dis_end = reconfiguration.disassembly_end_tick
            if reconfiguration.disassembly_start_tick is not None:
                disassembly_active_ticks += max(
                    0,
                    min(dis_end if dis_end is not None else cap_tick, cap_tick)
                    - reconfiguration.disassembly_start_tick,
                )
            if dis_end is not None and dis_end <= cap_tick:
                ins_start = (
                    reconfiguration.installation_start_tick
                    if reconfiguration.installation_start_tick is not None
                    else cap_tick
                )
                worker_wait_ins_ticks += max(0, min(ins_start, cap_tick) - dis_end)
            if reconfiguration.installation_start_tick is not None:
                ins_end = reconfiguration.installation_end_tick
                installation_active_ticks += max(
                    0,
                    min(ins_end if ins_end is not None else cap_tick, cap_tick)
                    - reconfiguration.installation_start_tick,
                )
            reconfiguration_details.append(
                {
                    "id": reconfiguration.id,
                    "operation_id": reconfiguration.operation_id,
                    "machine_id": reconfiguration.machine_id,
                    "source_module": reconfiguration.source_module,
                    "target_module": reconfiguration.target_module,
                    "stage": reconfiguration.stage.value,
                    "lock_tick": reconfiguration.lock_tick,
                    "disassembly_start_tick": reconfiguration.disassembly_start_tick,
                    "disassembly_end_tick": reconfiguration.disassembly_end_tick,
                    "installation_start_tick": reconfiguration.installation_start_tick,
                    "installation_end_tick": reconfiguration.installation_end_tick,
                }
            )

        summaries.append(
            {
                "order_id": order.id,
                "release_time": float(order.release_time),
                "unfinished_operation_count": len(remaining_operations),
                "remaining_operations": remaining_operations,
                "processing_elapsed_time": processing_elapsed_time,
                "processing_planned_time": processing_planned_time,
                "remaining_processing_lower_bound_time": to_minutes(
                    remaining_processing_lb_ticks
                ),
                "worker_wait_before_disassembly_time": to_minutes(
                    worker_wait_dis_ticks
                ),
                "worker_wait_before_installation_time": to_minutes(
                    worker_wait_ins_ticks
                ),
                "worker_wait_time": to_minutes(
                    worker_wait_dis_ticks + worker_wait_ins_ticks
                ),
                "reconfiguration_active_time": to_minutes(
                    disassembly_active_ticks + installation_active_ticks
                ),
                "reconfiguration_count": len(reconfiguration_details),
                "reconfigurations": reconfiguration_details,
            }
        )
    return summaries


def _failure_evidence(environment: AssemblySchedulingEnv) -> dict[str, Any]:
    events = sorted(environment._events)
    within = [event for event in events if event[0] <= environment.horizon_tick]
    after = [event for event in events if event[0] > environment.horizon_tick]
    next_event = events[0] if events else None
    return {
        "first_inability_tick": (
            None
            if environment.deadlock_detection_snapshot is None
            else environment.deadlock_detection_snapshot["tick"]
        ),
        "future_event_count": len(events),
        "within_horizon_event_count": len(within),
        "post_horizon_event_count": len(after),
        "next_future_event": (
            None
            if next_event is None
            else {
                "tick": int(next_event[0]),
                "time": float(next_event[0] * environment.resolution),
                "event_type": next_event[3].value,
                "payload": dict(next_event[4]),
            }
        ),
        "horizon_overrun_evidence": bool(after and not within),
        "structural_recoverability": (
            "not_assessed_beyond_horizon"
            if after
            else "not_assessed_no_future_event"
        ),
        "interpretation": (
            "A post-horizon event proves that the current schedule cannot make "
            "its next deterministic progress before the horizon. It does not "
            "prove that every remaining operation is structurally recoverable."
        ),
    }


def _reward_audit(
    environment: AssemblySchedulingEnv,
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = environment.metrics()
    sums = {
        name: float(sum(float(item["reward"].get(name, 0.0)) for item in transitions))
        for name in (
            "flow",
            "cost",
            "variance",
            "operation_progress",
            "quality",
            "failure",
            "feasibility_shaping",
        )
    }
    initial_progress = float(metrics["initial_progress"])
    final_progress = float(metrics["operation_progress"])
    initial_quality = float(metrics["initial_preference_quality_score"])
    formal_final_quality = float(metrics["preference_quality_score"])
    actual_final_quality = float(metrics["raw_preference_quality_score"])
    task_failed = bool(metrics["task_failed"])
    penalty = terminal_failure_penalty(environment.config) if task_failed else 0.0
    legacy_identity = (
        final_progress
        - initial_progress
        - formal_final_quality
        + initial_quality
    )
    failure_v2_base_identity = (
        final_progress
        - initial_progress
        - actual_final_quality
        + initial_quality
    )
    failure_v2_identity = failure_v2_base_identity - penalty
    recorded_base_return = sums["operation_progress"] + sums["quality"]
    recorded_training_return = recorded_base_return + sums["failure"]
    configured_mode = str(environment.config["reward"]["mode"])
    return {
        "recorded_reward_version": configured_mode,
        "component_sums": sums,
        "base_cumulative_reward": recorded_base_return,
        "scalar_training_return": recorded_training_return,
        "configured_proxy_return": proxy_return_from_metrics(
            metrics,
            environment.config,
            preference=metrics.get("preference"),
        ),
        "configured_identity_residual": float(
            recorded_training_return
            - proxy_return_from_metrics(
                metrics,
                environment.config,
                preference=metrics.get("preference"),
            )
        ),
        "recomputed_versions": {
            LEGACY_PROGRESS_QUALITY_REWARD: {
                "terminal_quality": formal_final_quality,
                "failure_penalty": 0.0,
                "base_cumulative_reward": legacy_identity,
                "training_cumulative_reward": legacy_identity,
            },
            FAILURE_PENALTY_REWARD: {
                "terminal_quality": actual_final_quality,
                "failure_penalty": penalty,
                "base_cumulative_reward": failure_v2_base_identity,
                "training_cumulative_reward": failure_v2_identity,
            },
        },
        "initial_progress": initial_progress,
        "final_progress": final_progress,
        "initial_quality": initial_quality,
        "formal_final_quality": formal_final_quality,
        "actual_final_quality": actual_final_quality,
        "flow_objective": float(metrics["flow_time_objective"]),
        "unfinished_order_penalty_component": float(environment._flow_penalty),
        "flow_integral_component": float(environment._flow_integral),
        "failed_trajectory_quality_overwrite": bool(
            task_failed and configured_mode == LEGACY_PROGRESS_QUALITY_REWARD
        ),
    }


class ReplayEnvironment(AssemblySchedulingEnv):
    """Capture the state before the normal deadlock handler advances to horizon."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.deadlock_detection_snapshot: dict[str, Any] | None = None

    def _record_unrecoverable_deadlock_diagnostic(
        self,
        *,
        classified_terminal_reason: str = "unrecoverable_deadlock",
    ) -> None:
        if self.deadlock_detection_snapshot is None:
            self.deadlock_detection_snapshot = snapshot_environment(self)
        super()._record_unrecoverable_deadlock_diagnostic(
            classified_terminal_reason=classified_terminal_reason
        )


def _rollout_heuristic(environment: AssemblySchedulingEnv) -> dict[str, Any]:
    policy = HeuristicPolicy()
    actions = []
    while not environment.task_done:
        action = policy.select_action(environment)
        actions.append(_describe_action(environment, action))
        environment.step(action, build_observation=False)
    metrics = environment.metrics()
    return {
        "task_succeeded": bool(metrics["task_succeeded"]),
        "terminal_reason": metrics["terminal_reason"],
        "operation_progress": float(metrics["operation_progress"]),
        "decisions": len(actions),
        "actions": actions,
    }


def _rollout_sampled_ppo(
    environment: AssemblySchedulingEnv,
    agent,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=agent.device).manual_seed(int(seed))
    actions: list[int] = []
    while not environment.task_done:
        observation = environment.observe()
        action, _, _ = agent.act(
            observation,
            environment.get_action_mask(),
            deterministic=False,
            generator=generator,
        )
        agent.consume_policy_decision_diagnostics()
        actions.append(int(action))
        environment.step(int(action), build_observation=False)
    metrics = environment.metrics()
    return {
        "seed": int(seed),
        "task_succeeded": bool(metrics["task_succeeded"]),
        "terminal_reason": metrics["terminal_reason"],
        "operation_progress": float(metrics["operation_progress"]),
        "decisions": len(actions),
        "action_trace_sha256": action_trace_sha256(actions),
        "actions": actions if metrics["task_succeeded"] else None,
        "timing": {
            "makespan": float(metrics["time"]),
            "flow_time_objective": float(metrics["flow_time_objective"]),
            "wait_total_time": float(metrics["wait_total_time"]),
            "worker_wait_time": float(metrics["worker_wait_time"]),
            "machine_waiting_for_worker_time": float(
                metrics["machine_waiting_for_worker_time"]
            ),
        },
        "unfinished_order_timing": _unfinished_order_timing(environment),
    }


def paired_continuation_comparison(
    config: dict[str, Any],
    instance,
    actions: list[int],
    *,
    decision_index: int | None,
    alternative_action: int | None,
    ppo_agent,
    seeds: list[int],
) -> dict[str, Any]:
    """Compare original and replacement actions with common continuation seeds."""

    if decision_index is None or alternative_action is None or not seeds:
        return {"status": "not_requested", "pairs": []}
    if decision_index < 0 or decision_index >= len(actions):
        raise ValueError(f"paired decision index out of range: {decision_index}")

    environment = AssemblySchedulingEnv(config)
    environment.reset(instance, build_observation=False)
    for prefix_action in actions[:decision_index]:
        mask = environment.get_action_mask()
        if prefix_action >= len(mask) or bool(mask[prefix_action]):
            raise RuntimeError("recorded prefix is illegal during paired replay")
        environment.step(prefix_action, build_observation=False)
    if environment.task_done:
        raise RuntimeError("paired branch point is already terminal")

    original_action = int(actions[decision_index])
    mask = environment.get_action_mask()
    for label, action in (
        ("original", original_action),
        ("replacement", int(alternative_action)),
    ):
        if action < 0 or action >= len(mask) or bool(mask[action]):
            raise ValueError(f"{label} paired action {action} is not legal")

    pairs = []
    for seed in seeds:
        outcomes = {}
        for label, action in (
            ("original", original_action),
            ("replacement", int(alternative_action)),
        ):
            branch = deepcopy(environment)
            branch.step(action, build_observation=False)
            outcomes[label] = _rollout_sampled_ppo(
                branch,
                ppo_agent,
                seed=seed,
            )
        pairs.append({"seed": int(seed), **outcomes})

    def aggregate(label: str) -> dict[str, Any]:
        outcomes = [pair[label] for pair in pairs]
        successes = [item for item in outcomes if item["task_succeeded"]]
        return {
            "count": len(outcomes),
            "success_count": len(successes),
            "completion_rate": len(successes) / len(outcomes),
            "mean_operation_progress": float(
                np.mean([item["operation_progress"] for item in outcomes])
            ),
            "mean_wait_total_time": float(
                np.mean([item["timing"]["wait_total_time"] for item in outcomes])
            ),
            "mean_machine_waiting_for_worker_time": float(
                np.mean(
                    [
                        item["timing"]["machine_waiting_for_worker_time"]
                        for item in outcomes
                    ]
                )
            ),
            "mean_success_flow_time_objective": (
                None
                if not successes
                else float(
                    np.mean(
                        [item["timing"]["flow_time_objective"] for item in successes]
                    )
                )
            ),
        }

    discordant = {
        "original_only_success": sum(
            pair["original"]["task_succeeded"]
            and not pair["replacement"]["task_succeeded"]
            for pair in pairs
        ),
        "replacement_only_success": sum(
            pair["replacement"]["task_succeeded"]
            and not pair["original"]["task_succeeded"]
            for pair in pairs
        ),
    }
    return {
        "status": "completed",
        "decision_index": int(decision_index),
        "tick": int(environment.current_tick),
        "progress": float(environment.operation_progress()),
        "original_action": _describe_action(environment, original_action),
        "replacement_action": _describe_action(
            environment, int(alternative_action)
        ),
        "continuation_seeds": [int(seed) for seed in seeds],
        "original": aggregate("original"),
        "replacement": aggregate("replacement"),
        "discordant_pairs": discordant,
        "pairs": pairs,
        "evidence_note": (
            "Common random-number seeds reduce avoidable sampling variation, "
            "but divergent states induce different action distributions. This "
            "is an exploratory paired comparison, not proof of causality."
        ),
    }


def search_late_alternatives(
    config: dict[str, Any],
    instance,
    actions: list[int],
    *,
    lookback: int,
    max_alternatives: int,
    decision_indices: list[int] | None = None,
    ppo_agent=None,
    ppo_samples: int = 0,
    ppo_seed_start: int = 910_000,
) -> dict[str, Any]:
    """Try deterministic heuristic continuations after late legal alternatives.

    One successful continuation proves recoverability at that decision. Failure
    of this bounded search is deliberately reported only as "not found".
    """

    explicit_indices = list(decision_indices or [])
    if lookback <= 0 and not explicit_indices:
        return {"attempted": 0, "successful_continuation": None}
    if explicit_indices:
        invalid = [
            index
            for index in explicit_indices
            if index < 0 or index >= len(actions)
        ]
        if invalid:
            raise ValueError(f"branch decision indices out of range: {invalid}")
        search_indices = sorted(set(explicit_indices), reverse=True)
    else:
        start = max(0, len(actions) - lookback)
        search_indices = list(range(len(actions) - 1, start - 1, -1))
    attempts = []
    for decision_index in search_indices:
        environment = AssemblySchedulingEnv(config)
        environment.reset(instance, build_observation=False)
        prefix_valid = True
        for prefix_action in actions[:decision_index]:
            mask = environment.get_action_mask()
            if prefix_action >= len(mask) or bool(mask[prefix_action]):
                prefix_valid = False
                break
            environment.step(prefix_action, build_observation=False)
        if not prefix_valid or environment.task_done:
            continue
        mask = environment.get_action_mask()
        original_action = actions[decision_index]
        alternatives = [
            int(action)
            for action in np.flatnonzero(~mask)
            if int(action) != original_action
        ][:max_alternatives]
        for alternative in alternatives:
            branch = deepcopy(environment)
            branch_description = _describe_action(branch, alternative)
            branch.step(alternative, build_observation=False)
            continuation = _rollout_heuristic(deepcopy(branch))
            sampled_continuations = []
            if not continuation["task_succeeded"] and ppo_agent is not None:
                for sample_index in range(ppo_samples):
                    sampled = _rollout_sampled_ppo(
                        deepcopy(branch),
                        ppo_agent,
                        seed=(
                            ppo_seed_start
                            + decision_index * 10_000
                            + alternative * 100
                            + sample_index
                        ),
                    )
                    sampled_continuations.append(sampled)
                    if sampled["task_succeeded"]:
                        break
            attempt = {
                "decision_index": decision_index,
                "tick": int(environment.current_tick),
                "progress": float(environment.operation_progress()),
                "original_action": _describe_action(environment, original_action),
                "alternative_action": branch_description,
                "continuation": continuation,
                "sampled_continuations": sampled_continuations,
            }
            attempts.append(attempt)
            if continuation["task_succeeded"] or any(
                item["task_succeeded"] for item in sampled_continuations
            ):
                return {
                    "attempted": len(attempts),
                    "search_status": "successful_continuation_found",
                    "successful_continuation": attempt,
                    "attempts": attempts,
                }
    return {
        "attempted": len(attempts),
        "search_status": "no_success_found_in_bounded_search",
        "successful_continuation": None,
        "attempts": attempts,
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    paired_fields_present = (
        args.paired_decision_index is not None,
        args.paired_alternative_action is not None,
    )
    if any(paired_fields_present) != all(paired_fields_present):
        raise ValueError(
            "paired decision index and alternative action must be provided together"
        )
    if args.paired_seed_count < 0:
        raise ValueError("paired seed count cannot be negative")
    if all(paired_fields_present) and args.paired_seed_count <= 0:
        raise ValueError("paired comparison requires a positive seed count")

    config = _load_effective_config(args.config)
    config["device"] = args.device
    set_seed(int(config["seed"]))
    dataset = load_dataset_split(config, args.dataset)
    record = next(
        (
            candidate
            for candidate in dataset
            if candidate.instance.instance_id == args.instance_id
        ),
        None,
    )
    if record is None:
        raise ValueError(
            f"instance {args.instance_id!r} not found in {args.dataset!r}"
        )
    environment = ReplayEnvironment(config)
    observation = environment.reset(record.instance)
    policy = EvaluationPolicy(
        config,
        policy_name="ppo",
        bootstrap_observation=observation,
        checkpoint=args.checkpoint,
        decode_mode="sampled",
        sampling_seed=args.sampling_seed,
    )
    policy.begin_episode(record.instance.instance_id)
    was_training = policy.enter_evaluation_mode()
    actions: list[int] = []
    transitions = []
    try:
        while not environment.task_done:
            action = int(policy.select_action(observation, environment))
            before = snapshot_environment(environment)
            action_description = _describe_action(environment, action)
            observation, reward, terminated, truncated, info = environment.step(
                action
            )
            actions.append(action)
            transitions.append(
                {
                    "decision_index": len(actions) - 1,
                    "before": before,
                    "action": action_description,
                    "reward": reward.as_dict(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )
    finally:
        policy.restore_mode(was_training)
    trace_hash = action_trace_sha256(actions)
    if args.expected_action_trace_sha256 is not None:
        expected = args.expected_action_trace_sha256.lower()
        if trace_hash != expected:
            raise RuntimeError(
                "replayed action trace does not match expected hash: "
                f"expected {expected}, got {trace_hash}"
            )
    metrics = environment.metrics()
    branch_was_training = policy.enter_evaluation_mode()
    try:
        branch_search = search_late_alternatives(
            config,
            record.instance,
            actions,
            lookback=args.branch_lookback,
            max_alternatives=args.branch_max_alternatives,
            decision_indices=args.branch_decision_index,
            ppo_agent=policy.ppo_agent,
            ppo_samples=args.branch_ppo_samples,
            ppo_seed_start=args.branch_ppo_seed_start,
        )
        paired_seeds = [
            args.paired_seed_start + index
            for index in range(args.paired_seed_count)
        ]
        paired_comparison = paired_continuation_comparison(
            config,
            record.instance,
            actions,
            decision_index=args.paired_decision_index,
            alternative_action=args.paired_alternative_action,
            ppo_agent=policy.ppo_agent,
            seeds=paired_seeds,
        )
    finally:
        policy.restore_mode(branch_was_training)
    return {
        "replay": {
            "config": str(project_path(args.config)),
            "checkpoint": str(project_path(args.checkpoint)),
            "dataset": args.dataset,
            "instance_id": args.instance_id,
            "sampling_seed": args.sampling_seed,
            "derived_sampling_seed": policy.derived_sampling_seed,
            "device": args.device,
            "action_trace_sha256": trace_hash,
            "expected_action_trace_sha256": args.expected_action_trace_sha256,
            "action_count": len(actions),
        },
        "metrics": metrics,
        "actions": actions,
        "transitions": transitions,
        "deadlock_detection_snapshot": environment.deadlock_detection_snapshot,
        "terminal_snapshot": snapshot_environment(
            environment, include_mask=False
        ),
        "failure_evidence": _failure_evidence(environment),
        "unfinished_order_timing": _unfinished_order_timing(environment),
        "reward_audit": _reward_audit(environment, transitions),
        "schedule_log": list(environment.schedule_log),
        "reconfiguration_log": list(environment.reconfiguration_log),
        "late_alternative_search": branch_search,
        "paired_continuation_comparison": paired_comparison,
        "evidence_note": (
            "A successful continuation proves recoverability at that snapshot. "
            "No success in a bounded search does not prove infeasibility. A "
            "post-horizon event proves delayed deterministic progress, not "
            "structural recoverability of all remaining work."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one sampled PPO trajectory with deadlock snapshots."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--sampling-seed", required=True, type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--expected-action-trace-sha256")
    parser.add_argument("--branch-lookback", type=int, default=0)
    parser.add_argument("--branch-max-alternatives", type=int, default=12)
    parser.add_argument(
        "--branch-decision-index",
        type=int,
        action="append",
        default=[],
        help="Replay and branch at this zero-based decision; may be repeated.",
    )
    parser.add_argument("--branch-ppo-samples", type=int, default=0)
    parser.add_argument("--branch-ppo-seed-start", type=int, default=910_000)
    parser.add_argument("--paired-decision-index", type=int)
    parser.add_argument("--paired-alternative-action", type=int)
    parser.add_argument("--paired-seed-count", type=int, default=0)
    parser.add_argument("--paired-seed-start", type=int, default=960_000)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = replay(args)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                **result["replay"],
                "terminal_reason": result["metrics"]["terminal_reason"],
                "operation_progress": result["metrics"]["operation_progress"],
                "branch_search_status": result["late_alternative_search"].get(
                    "search_status"
                ),
                "paired_comparison_status": result[
                    "paired_continuation_comparison"
                ].get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
