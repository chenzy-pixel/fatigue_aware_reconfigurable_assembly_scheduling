from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.ppo import PPOAgent, build_actor_critic
from configs import project_path
from data import load_dataset_split
from environment import AssemblySchedulingEnv, DecisionType, normalize_preference
from utils import action_trace_sha256, set_seed


def _load_effective_config(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("effective config root must be an object")
    return config


def _find_instance(config: dict[str, Any], instance_id: str):
    dataset = load_dataset_split(config, "validation")
    for record in dataset:
        if record.instance.instance_id == instance_id:
            return record.instance
    raise ValueError(f"validation instance not found: {instance_id}")


def _action_description(
    environment: AssemblySchedulingEnv,
    phase: DecisionType,
    action: int,
) -> dict[str, Any]:
    if phase == DecisionType.PRODUCTION:
        if action == environment.production_defer_action:
            return {"kind": "defer"}
        operation_index, machine_index = environment.decode_production_action(
            action
        )
        operation = environment.operations[operation_index]
        machine = environment.machines[machine_index]
        profile = environment._production_candidate_profile(
            operation_index,
            machine_index,
        )
        return {
            "kind": "production_pair",
            "operation_id": operation.spec.id,
            "machine_id": machine.spec.id,
            "source_module": machine.current_module,
            "target_module": operation.spec.required_module,
            "resource_ready_tick": profile.resource_ready_tick,
            "predicted_finish_tick": profile.predicted_finish_tick,
            "matching_deficit_after_commit": (
                profile.matching_deficit_after_commit
            ),
            "safe_disassembly_workers": profile.safe_disassembly_workers,
            "safe_installation_workers": profile.safe_installation_workers,
            "horizon_slack_ticks": profile.horizon_slack_ticks,
            "admissible": profile.admissible,
        }
    if action == environment.worker_advance_action:
        return {"kind": "advance"}
    machine_index, worker_index = environment.decode_worker_action(action)
    machine = environment.machines[machine_index]
    worker = environment.workers[worker_index]
    reconfiguration = environment._pending_reconfiguration(machine.spec.id)
    return {
        "kind": "worker_pair",
        "machine_id": machine.spec.id,
        "worker_id": worker.spec.id,
        "worker_fatigue": worker.fatigue,
        "reconfiguration_id": (
            reconfiguration.id if reconfiguration is not None else None
        ),
        "stage": (
            reconfiguration.stage.value
            if reconfiguration is not None
            else None
        ),
        "source_module": (
            reconfiguration.source_module
            if reconfiguration is not None
            else None
        ),
        "target_module": (
            reconfiguration.target_module
            if reconfiguration is not None
            else None
        ),
    }


def _select_action(
    agent: PPOAgent,
    observation,
    action_mask: np.ndarray,
    decoder: str,
) -> tuple[int, torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        logits, _ = agent.network.forward(
            observation,
            action_mask,
            device=agent.device,
        )
        if decoder == "gate_first":
            action = int(
                agent._deterministic_actions(
                    [observation],
                    [action_mask],
                    logits.unsqueeze(0),
                )[0].detach().cpu()
            )
        elif decoder == "joint_argmax":
            action = int(torch.argmax(logits).detach().cpu())
        else:
            raise ValueError(f"unknown decoder: {decoder}")
    diagnostics = agent.network.consume_policy_decision_diagnostics()
    diagnostic = diagnostics[0] if diagnostics else {}
    return action, logits, diagnostic


def replay(
    base_config: dict[str, Any],
    *,
    checkpoint: str | Path,
    instance,
    preference: list[float],
    decoder: str,
    require_full_matching: bool,
    preserve_matching_on_worker_action: bool,
    direct_preference_scope: str,
    max_decisions: int,
    stall_chain_limit: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["environment"]["worker_resource_control"][
        "require_full_matching"
    ] = require_full_matching
    config["environment"]["worker_resource_control"][
        "preserve_matching_on_worker_action"
    ] = preserve_matching_on_worker_action
    set_seed(int(config["seed"]))
    environment = AssemblySchedulingEnv(config)
    effective_preference = normalize_preference(preference)
    observation = environment.reset(
        instance,
        preference=effective_preference,
    )
    network = build_actor_critic(observation, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    agent.load(project_path(checkpoint))
    agent.network.eval()
    if direct_preference_scope not in {
        "all",
        "none",
        "production_only",
        "worker_only",
    }:
        raise ValueError(
            f"unknown direct preference scope: {direct_preference_scope}"
        )
    if (
        direct_preference_scope != "all"
        and agent.network.preference_action_score_enabled
    ):
        original_direct_preference_logits = (
            agent.network._direct_preference_logits
        )

        def scoped_direct_preference_logits(
            objectives: torch.Tensor,
            feasible: torch.Tensor,
            preference_tensor: torch.Tensor,
        ) -> torch.Tensor:
            phase = "worker" if objectives.shape[1] == 3 else "production"
            enabled = direct_preference_scope in {
                phase,
                f"{phase}_only",
            }
            if enabled:
                return original_direct_preference_logits(
                    objectives,
                    feasible,
                    preference_tensor,
                )
            return torch.zeros(
                objectives.shape[0],
                dtype=objectives.dtype,
                device=objectives.device,
            )

        agent.network._direct_preference_logits = (
            scoped_direct_preference_logits
        )

    actions: list[int] = []
    trace: list[dict[str, Any]] = []
    first_stall_decision: int | None = None
    stopped_for_stall = False
    while not (environment.terminated or environment.truncated):
        if len(actions) >= max_decisions:
            break
        mask = environment.get_action_mask()
        phase = environment.decision_type
        time_before = environment.current_time
        action, logits, policy_diagnostic = _select_action(
            agent,
            observation,
            mask,
            decoder,
        )
        probabilities = torch.softmax(logits, dim=0)
        legal_pair_indices = np.flatnonzero(~mask[:-1])
        terminal_legal = not bool(mask[-1])
        if legal_pair_indices.size:
            pair_tensor = torch.as_tensor(
                legal_pair_indices,
                dtype=torch.long,
                device=probabilities.device,
            )
            pair_probabilities = probabilities.index_select(0, pair_tensor)
            commit_probability = float(pair_probabilities.sum().cpu())
            top_offset = int(torch.argmax(pair_probabilities).cpu())
            top_pair_action = int(legal_pair_indices[top_offset])
            top_pair_probability = float(pair_probabilities[top_offset].cpu())
        else:
            commit_probability = 0.0
            top_pair_action = None
            top_pair_probability = 0.0
        terminal_probability = (
            float(probabilities[-1].cpu()) if terminal_legal else 0.0
        )
        description = _action_description(environment, phase, action)
        top_pair_description = (
            _action_description(environment, phase, top_pair_action)
            if top_pair_action is not None
            else None
        )
        observation, _, _, _, info = environment.step(action)
        actions.append(action)
        forced_chain = int(environment._current_forced_action_chain)
        if forced_chain >= stall_chain_limit and first_stall_decision is None:
            first_stall_decision = len(actions)

        row = {
            "decision": len(actions),
            "time_before": time_before,
            "time_after": environment.current_time,
            "phase": phase.value,
            "legal_action_count": int(np.count_nonzero(~mask)),
            "legal_pair_count": int(legal_pair_indices.size),
            "terminal_legal": terminal_legal,
            "selected_action": action,
            "selected": description,
            "top_pair_action": top_pair_action,
            "top_pair": top_pair_description,
            "commit_probability": commit_probability,
            "terminal_probability": terminal_probability,
            "top_pair_probability": top_pair_probability,
            "preference_overrode_relative_top": bool(
                policy_diagnostic.get(
                    "preference_overrode_relative_top", False
                )
            ),
            "preference_logit_std": float(
                policy_diagnostic.get("preference_logit_std", 0.0)
            ),
            "commit_set_logit": float(
                policy_diagnostic.get("commit_set_logit", 0.0)
            ),
            "defer_reason": info.get("defer_reason"),
            "wait_time": info.get("wait_time"),
            "forced_chain": forced_chain,
            "matching_deficit_events": int(
                len(environment._worker_matching_deficit_ticks)
            ),
            "resource_admission_masks": int(
                len(environment._resource_admission_masked)
            ),
        }
        if (
            row["legal_action_count"] > 1
            or description["kind"] in {"production_pair", "worker_pair"}
            or len(trace) < 12
            or forced_chain in {1, stall_chain_limit}
        ):
            trace.append(row)
        if forced_chain >= stall_chain_limit:
            stopped_for_stall = True
            break

    metrics = environment.metrics()
    return {
        "preference": list(effective_preference.as_tuple()),
        "decoder": decoder,
        "require_full_matching": require_full_matching,
        "preserve_matching_on_worker_action": (
            preserve_matching_on_worker_action
        ),
        "direct_preference_scope": direct_preference_scope,
        "decisions_observed": len(actions),
        "stopped_for_stall": stopped_for_stall,
        "first_stall_decision": first_stall_decision,
        "partial_action_trace_sha256": action_trace_sha256(actions),
        "action_sequence": actions,
        "terminated": bool(environment.terminated),
        "truncated": bool(environment.truncated),
        "terminal_reason": environment.terminal_reason,
        "time": environment.current_time,
        "completed_orders": int(metrics["completed_orders"]),
        "unfinished_orders": int(metrics["unfinished_orders"]),
        "direct_process_action_count": int(
            metrics["direct_process_action_count"]
        ),
        "commit_reconfig_action_count": int(
            metrics["commit_reconfig_action_count"]
        ),
        "defer_production_action_count": int(
            metrics["defer_production_action_count"]
        ),
        "worker_assign_action_count": int(
            metrics["worker_assign_action_count"]
        ),
        "advance_event_action_count": int(
            metrics["advance_event_action_count"]
        ),
        "worker_matching_deficit_event_count": int(
            metrics["worker_matching_deficit_event_count"]
        ),
        "resource_admission_masked_action_count": int(
            metrics["resource_admission_masked_action_count"]
        ),
        "machine_waiting_for_worker_time": float(
            metrics["machine_waiting_for_worker_time"]
        ),
        "maximum_worker_fatigue": float(metrics["maximum_worker_fatigue"]),
        "schedule_violation_count": len(environment.validate_schedule()),
        "trace": trace,
    }


def _first_divergence(left: dict[str, Any], right: dict[str, Any]):
    left_actions = left["action_sequence"]
    right_actions = right["action_sequence"]
    left_rows = {row["decision"]: row for row in left["trace"]}
    right_rows = {row["decision"]: row for row in right["trace"]}
    for offset, (left_action, right_action) in enumerate(
        zip(left_actions, right_actions)
    ):
        if left_action != right_action:
            decision = offset + 1
            return {
                "decision": decision,
                "left": left_rows.get(
                    decision,
                    {"decision": decision, "selected_action": left_action},
                ),
                "right": right_rows.get(
                    decision,
                    {"decision": decision, "selected_action": right_action},
                ),
            }
    if len(left_actions) != len(right_actions):
        decision = min(len(left_actions), len(right_actions)) + 1
        return {
            "decision": decision,
            "left": left_rows.get(decision),
            "right": right_rows.get(decision),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay E2.2 worker-matching deadlock counterfactuals"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--instance-id",
        default="validation_worker_bottleneck_2000003",
    )
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--stall-chain-limit", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = _load_effective_config(args.config)
    instance = _find_instance(config, args.instance_id)
    scenarios = [
        ("cost_failure", [0.0, 0.2, 0.8], "gate_first", True, True, "all"),
        ("zero_cost_success", [0.0, 0.0, 1.0], "gate_first", True, True, "all"),
        ("joint_argmax", [0.0, 0.2, 0.8], "joint_argmax", True, True, "all"),
        ("partial_matching", [0.0, 0.2, 0.8], "gate_first", False, True, "all"),
        (
            "worker_recovery",
            [0.0, 0.2, 0.8],
            "gate_first",
            True,
            False,
            "all",
        ),
        (
            "production_direct_only",
            [0.0, 0.2, 0.8],
            "gate_first",
            True,
            True,
            "production_only",
        ),
        (
            "worker_direct_only",
            [0.0, 0.2, 0.8],
            "gate_first",
            True,
            True,
            "worker_only",
        ),
    ]
    results = {
        name: replay(
            config,
            checkpoint=args.checkpoint,
            instance=instance,
            preference=preference,
            decoder=decoder,
            require_full_matching=require_full_matching,
            preserve_matching_on_worker_action=(
                preserve_matching_on_worker_action
            ),
            direct_preference_scope=direct_preference_scope,
            max_decisions=args.max_decisions,
            stall_chain_limit=args.stall_chain_limit,
        )
        for (
            name,
            preference,
            decoder,
            require_full_matching,
            preserve_matching_on_worker_action,
            direct_preference_scope,
        ) in scenarios
    }
    output = {
        "instance_id": args.instance_id,
        "checkpoint": str(project_path(args.checkpoint)),
        "scenarios": results,
        "first_divergence_cost_vs_zero_cost": _first_divergence(
            results["cost_failure"], results["zero_cost_success"]
        ),
        "first_divergence_gate_vs_joint_argmax": _first_divergence(
            results["cost_failure"], results["joint_argmax"]
        ),
        "first_divergence_full_vs_partial_matching": _first_divergence(
            results["cost_failure"], results["partial_matching"]
        ),
        "first_divergence_full_vs_worker_recovery": _first_divergence(
            results["cost_failure"], results["worker_recovery"]
        ),
        "first_divergence_all_vs_production_direct_only": (
            _first_divergence(
                results["cost_failure"],
                results["production_direct_only"],
            )
        ),
        "first_divergence_all_vs_worker_direct_only": _first_divergence(
            results["cost_failure"], results["worker_direct_only"]
        ),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        path = project_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        console = {
            "output": str(path),
            "scenarios": {
                name: {
                    key: scenario[key]
                    for key in (
                        "decisions_observed",
                        "stopped_for_stall",
                        "terminated",
                        "truncated",
                        "time",
                        "completed_orders",
                        "unfinished_orders",
                        "maximum_worker_fatigue",
                        "schedule_violation_count",
                    )
                }
                for name, scenario in results.items()
            },
            "first_divergence_cost_vs_zero_cost": output[
                "first_divergence_cost_vs_zero_cost"
            ],
            "first_divergence_full_vs_worker_recovery": output[
                "first_divergence_full_vs_worker_recovery"
            ],
        }
        print(json.dumps(console, ensure_ascii=False, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
