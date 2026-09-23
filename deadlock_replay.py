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
from environment import AssemblySchedulingEnv, DecisionType
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
        "late_alternative_search": branch_search,
        "evidence_note": (
            "A successful continuation proves recoverability. No success in "
            "the bounded heuristic or sampled search does not prove infeasibility."
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
