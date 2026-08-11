from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional

from agent.baselines import HeuristicPolicy
from agent.ppo import (
    PPOAgent,
    RolloutBuffer,
    assert_network_config_matches_spec,
    build_actor_critic,
)
from configs import load_config, project_path
from configs.config import public_config
from data import (
    GeneratedInstanceRecord,
    canonical_json_bytes,
    curriculum_weights_at,
    load_generated_record,
    save_generated_record_atomic,
    sha256_bytes,
)
from data.generate_orders import InstanceGenerator
from environment import (
    CAPABLE_EDGE,
    SERVICE_CANDIDATE_EDGE,
    AssemblySchedulingEnv,
    DecisionType,
    HeterogeneousGraphObservation,
    bounded_quality_score,
)
from eval import EvaluationPolicy, evaluate_instance, load_configured_instance
from result.io import write_json
from result.metrics import relative_gap_percent
from train import (
    TrainingPhaseController,
    ValidationStabilityController,
    _collect_serial_batch,
    train,
)
from utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
M1_METHOD_VERSION = "M1_candidate_graph_v6"
M1_ROOT = PROJECT_ROOT / "result" / "m1" / M1_METHOD_VERSION
STAGE_DIRECTORIES = {
    "p1": M1_ROOT / "P1_state_machine",
    "p2": M1_ROOT / "P2_candidate_heads",
    "p3": M1_ROOT / "P3_observation_schema_v3",
    "p4": M1_ROOT / "P4_behavior_cloning",
    "p4_1": M1_ROOT / "P4_1_candidate_ranking_repair",
    "p5": M1_ROOT / "P5_small_instance_overfit",
    "p6": M1_ROOT / "P6_seed11",
}


@dataclass(frozen=True)
class RankingState:
    observation: HeterogeneousGraphObservation
    action_mask: np.ndarray
    legal_actions: tuple[int, ...]
    teacher_action: int
    teacher_keys: tuple[tuple[float, ...], ...]
    decision_type: str
    instance_seed: int
    step: int
    candidate_records: tuple[dict[str, Any], ...]
    pairwise_actions: tuple[tuple[int, int], ...]


# Keep torch-saved diagnostic caches importable whether this file is invoked
# as a script or imported as a module.
sys.modules.setdefault("m1_experiments", sys.modules[__name__])
RankingState.__module__ = "m1_experiments"


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _provenance(config: dict[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", "."],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    status = _git_output("status", "--short").encode("utf-8")
    return {
        "method_version": config.get("method_version"),
        "commit": _git_output("rev-parse", "HEAD").strip(),
        "workspace_sha256": hashlib.sha256(diff + status).hexdigest(),
        "config_path": config.get("_config_path"),
        "seeds": [int(seed) for seed in seeds],
        "created_unix_time": time.time(),
    }


def _prepare_stage(
    stage: str,
    config: dict[str, Any],
    seeds: Sequence[int],
) -> Path:
    configured_method = str(config.get("method_version", ""))
    if configured_method != M1_METHOD_VERSION:
        raise ValueError(
            "M1 artifact version mismatch: expected "
            f"{M1_METHOD_VERSION}, got {configured_method or '<missing>'}. "
            "older artifacts and caches cannot be loaded by the v6 pipeline."
        )
    configured_head = int(
        config.get("network", {}).get("policy_head_version", 6)
    )
    if configured_head != 6:
        raise ValueError(
            "M1 policy head version mismatch: expected 6, "
            f"got {configured_head}. Automatic weight conversion is disabled."
        )
    directory = STAGE_DIRECTORIES[stage]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "effective_config.json", public_config(config))
    write_json(directory / "provenance.json", _provenance(config, seeds))
    return directory


def _write_gate(
    directory: Path,
    *,
    stage: str,
    passed: bool,
    checks: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "stage": stage.upper(),
        "passed": bool(passed),
        "checks": checks,
        "diagnostics": diagnostics or {},
    }
    write_json(directory / "gate.json", payload)
    return payload


def _require_passed(*stages: str) -> None:
    for stage in stages:
        gate_path = STAGE_DIRECTORIES[stage] / "gate.json"
        if not gate_path.exists():
            raise RuntimeError(f"{stage.upper()} gate has not been run")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not bool(gate.get("passed")):
            raise RuntimeError(f"{stage.upper()} gate did not pass")


def _require_p4_expression_gate() -> None:
    repaired_gate = STAGE_DIRECTORIES["p4_1"] / "gate.json"
    if repaired_gate.exists():
        payload = json.loads(repaired_gate.read_text(encoding="utf-8"))
        if bool(payload.get("passed")):
            return
    _require_passed("p4")


def _run_pytest_gate(
    config: dict[str, Any],
    stage: str,
    node_ids: Sequence[str],
) -> dict[str, Any]:
    directory = _prepare_stage(stage, config, ())
    command = [sys.executable, "-m", "pytest", *node_ids, "-q"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    (directory / "pytest.txt").write_text(output, encoding="utf-8")
    return _write_gate(
        directory,
        stage=stage,
        passed=completed.returncode == 0,
        checks={
            "pytest_return_code": completed.returncode,
            "node_ids": list(node_ids),
        },
        diagnostics={"pytest_output": output[-12000:]},
    )


def run_p1(config: dict[str, Any]) -> dict[str, Any]:
    return _run_pytest_gate(
        config,
        "p1",
        (
            "test/test_hierarchical_reward.py::test_validation_stability_rollback_thresholds",
            "test/test_hierarchical_reward.py::test_validation_log_promotion_key_and_q12_score_are_identical",
            "test/test_training_dataset.py::test_three_not_promoted_updates_keep_network_and_optimizer_advancing",
            "test/test_training_dataset.py::test_catastrophic_regression_rolls_back_safe_and_keeps_decayed_lr",
            "test/test_training_dataset.py::test_training_uses_unique_episode_instances_and_writes_validation_artifacts",
        ),
    )


def run_p2(config: dict[str, Any]) -> dict[str, Any]:
    _require_passed("p1")
    return _run_pytest_gate(
        config,
        "p2",
        (
            "test/test_hetero_gnn.py::test_action_edge_rows_align_with_flat_actions_and_change_logits",
            "test/test_hetero_gnn.py::test_direct_edge_heads_handle_only_advance",
            "test/test_hetero_gnn.py::test_hetero_batch_matches_individual_and_masks",
            "test/test_hetero_gnn.py::test_worker_head_requires_unique_locked_operation",
        ),
    )


def run_p3(config: dict[str, Any]) -> dict[str, Any]:
    _require_passed("p1", "p2")
    return _run_pytest_gate(
        config,
        "p3",
        (
            "test/test_graph_observation.py::test_graph_observation_static_contract",
            "test/test_graph_observation.py::test_fixed_cost_counterfactual_changes_state_and_candidate_edges",
            "test/test_graph_observation.py::test_schema_v3_future_demand_order_progress_and_snapshot_stability",
            "test/test_graph_observation.py::test_future_module_counterfactual_is_observable_without_mask_aliasing",
            "test/test_graph_observation.py::test_graph_copy_buffer_and_terminal_observation",
        ),
    )


def _diagnostic_records(
    config: dict[str, Any],
    seeds: Sequence[int],
) -> list[GeneratedInstanceRecord]:
    template = load_configured_instance(config)
    generator = InstanceGenerator(
        template,
        config["generator"],
        config=config,
    )
    records: list[GeneratedInstanceRecord] = []
    diagnostic_start = int(
        config["m1_gates"].get("diagnostic_seed_start", min(seeds))
    )
    diagnostic_count = int(
        config["m1_gates"].get("diagnostic_instance_count", len(seeds))
    )
    for seed in seeds:
        weights = curriculum_weights_at(
            config["generator"]["curriculum"],
            (int(seed) - diagnostic_start) / max(1, diagnostic_count),
        )
        chooser = random.Random(int(seed))
        pressure_type = chooser.choices(
            list(weights),
            weights=[float(weights[name]) for name in weights],
            k=1,
        )[0]
        records.append(
            generator.generate(
                seed=int(seed),
                split="train",
                pressure_type=pressure_type,
            )
        )
    return records


def _p5_instance_snapshot(
    directory: Path,
    config: dict[str, Any],
    seeds: Sequence[int],
    records: Sequence[GeneratedInstanceRecord] | None = None,
    *,
    snapshot_role: str = "p5",
) -> list[GeneratedInstanceRecord]:
    """Persist or reload an immutable, hash-checked M1 instance batch."""
    instances_directory = directory / "instances"
    records_directory = instances_directory / "records"
    manifest_path = instances_directory / "manifest.json"
    expected_seeds = [int(seed) for seed in seeds]

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_version = {
            "snapshot_schema_version": "M1_INSTANCE_SNAPSHOT_V1",
            "snapshot_role": snapshot_role,
            "method_version": M1_METHOD_VERSION,
            "policy_head_version": 6,
            "seeds": expected_seeds,
        }
        for field, expected in expected_version.items():
            actual = manifest.get(field)
            if actual != expected:
                raise ValueError(
                    "P5 instance cache version mismatch for "
                    f"{field}: expected {expected!r}, got {actual!r}. "
                    "older caches cannot be reused by the v6 pipeline."
                )
        entries = manifest.get("records")
        if not isinstance(entries, list) or len(entries) != len(
            expected_seeds
        ):
            raise ValueError("P5 instance manifest has an invalid record list")
        loaded: list[GeneratedInstanceRecord] = []
        for expected_seed, entry in zip(expected_seeds, entries):
            if not isinstance(entry, dict):
                raise ValueError("P5 instance manifest record must be an object")
            relative_path = Path(str(entry.get("path", "")))
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parts[:1] != ("records",)
            ):
                raise ValueError(
                    f"P5 instance manifest has unsafe path {relative_path!s}"
                )
            if int(entry.get("seed", -1)) != expected_seed:
                raise ValueError("P5 instance manifest seed order mismatch")
            loaded.append(
                load_generated_record(
                    instances_directory / relative_path,
                    expected_sha256=str(entry.get("sha256", "")),
                )
            )
        if records is not None:
            supplied = list(records)
            if len(supplied) != len(loaded):
                raise ValueError(
                    "Supplied P5 records do not match the persisted snapshot"
                )
            for expected_seed, cached, candidate in zip(
                expected_seeds, loaded, supplied
            ):
                cached_hash = sha256_bytes(
                    canonical_json_bytes(cached.to_dict())
                )
                candidate_hash = sha256_bytes(
                    canonical_json_bytes(candidate.to_dict())
                )
                if cached_hash != candidate_hash:
                    raise ValueError(
                        "Supplied P5 record does not match persisted snapshot "
                        f"for seed {expected_seed}: expected {cached_hash}, "
                        f"got {candidate_hash}"
                    )
        return loaded

    if records is None:
        generated = []
        for seed in expected_seeds:
            filename = f"instance_{seed}.json"
            record_path = records_directory / filename
            if record_path.exists():
                record = load_generated_record(record_path)
            else:
                record = _diagnostic_records(config, [seed])[0]
                save_generated_record_atomic(record, record_path)
            generated.append(record)
    else:
        generated = list(records)
    if len(generated) != len(expected_seeds):
        raise ValueError(
            "P5 instance count mismatch: expected "
            f"{len(expected_seeds)}, got {len(generated)}"
        )
    entries: list[dict[str, Any]] = []
    for seed, record in zip(expected_seeds, generated):
        record_seed = int(record.metadata.get("seed", -1))
        if record_seed != seed:
            raise ValueError(
                f"P5 record seed mismatch: expected {seed}, got {record_seed}"
            )
        filename = f"instance_{seed}.json"
        digest = save_generated_record_atomic(
            record, records_directory / filename
        )
        entries.append(
            {
                "seed": seed,
                "path": f"records/{filename}",
                "sha256": digest,
            }
        )
    write_json(
        manifest_path,
        {
            "snapshot_schema_version": "M1_INSTANCE_SNAPSHOT_V1",
            "snapshot_role": snapshot_role,
            "method_version": M1_METHOD_VERSION,
            "policy_head_version": 6,
            "seeds": expected_seeds,
            "records": entries,
        },
    )
    return generated


def _edge_features_for_action(
    observation: HeterogeneousGraphObservation,
    edge_type,
    source: int,
    target: int,
) -> dict[str, float]:
    store = observation.relations[edge_type]
    matches = np.flatnonzero(
        (store.edge_index[0] == source)
        & (store.edge_index[1] == target)
    )
    if len(matches) != 1:
        return {}
    row = int(matches[0])
    return {
        name: float(store.edge_features[row, index])
        for index, name in enumerate(store.feature_names)
    }


def _pairwise_actions(
    legal_actions: Sequence[int],
    teacher_keys: Sequence[tuple[float, ...]],
    *,
    seed: int,
    maximum_pairs: int,
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for left in range(len(legal_actions)):
        for right in range(left + 1, len(legal_actions)):
            if teacher_keys[left] == teacher_keys[right]:
                continue
            better, worse = (
                (left, right)
                if teacher_keys[left] < teacher_keys[right]
                else (right, left)
            )
            pairs.append(
                (int(legal_actions[better]), int(legal_actions[worse]))
            )
    if len(pairs) <= maximum_pairs:
        return tuple(pairs)
    chooser = random.Random(seed)
    return tuple(chooser.sample(pairs, maximum_pairs))


def _ranking_state(
    environment: AssemblySchedulingEnv,
    observation: HeterogeneousGraphObservation,
    *,
    instance_seed: int,
    step: int,
    maximum_pairs: int,
) -> RankingState | None:
    mask = environment.get_action_mask()
    legal_actions = tuple(
        int(action)
        for action in np.flatnonzero(~mask)
        if int(action) != environment.advance_action
    )
    if len(legal_actions) < 2:
        return None
    teacher_keys: list[tuple[float, ...]] = []
    candidates: list[dict[str, Any]] = []
    if environment.decision_type == DecisionType.PRODUCTION:
        for action in legal_actions:
            operation_index, machine_index = (
                environment.decode_production_action(action)
            )
            operation = environment.operations[operation_index]
            machine = environment.machines[machine_index]
            duration = environment.estimate_processing_ticks(
                operation_index, machine_index
            ) + environment.estimate_reconfiguration_ticks(
                operation_index, machine_index
            )
            key = (
                float(duration),
                float(machine.current_module != operation.spec.required_module),
                float(operation.spec.sequence),
            )
            teacher_keys.append(key)
            candidates.append(
                {
                    "action": action,
                    "operation_id": operation.spec.id,
                    "machine_id": machine.spec.id,
                    "teacher_key": list(key),
                    "edge_features": _edge_features_for_action(
                        observation,
                        CAPABLE_EDGE,
                        operation_index,
                        machine_index,
                    ),
                }
            )
        decision_type = "production"
    else:
        for action in legal_actions:
            machine_index, worker_index = environment.decode_worker_action(
                action
            )
            worker = environment.workers[worker_index]
            projected_load_variance = (
                environment.projected_worker_load_variance(
                    machine_index,
                    worker_index,
                )
            )
            key = (
                float(
                    environment.projected_worker_fatigue(
                        machine_index, worker_index
                    )
                ),
                float(projected_load_variance),
            )
            teacher_keys.append(key)
            candidates.append(
                {
                    "action": action,
                    "machine_id": environment.machines[
                        machine_index
                    ].spec.id,
                    "worker_id": worker.spec.id,
                    "teacher_key": list(key),
                    "projected_load_variance": projected_load_variance,
                    "edge_features": _edge_features_for_action(
                        observation,
                        SERVICE_CANDIDATE_EDGE,
                        machine_index,
                        worker_index,
                    ),
                }
            )
        decision_type = "worker"
    teacher_action = int(legal_actions[teacher_keys.index(min(teacher_keys))])
    return RankingState(
        observation=observation.copy(),
        action_mask=mask.copy(),
        legal_actions=legal_actions,
        teacher_action=int(teacher_action),
        teacher_keys=tuple(teacher_keys),
        decision_type=decision_type,
        instance_seed=int(instance_seed),
        step=int(step),
        candidate_records=tuple(candidates),
        pairwise_actions=_pairwise_actions(
            legal_actions,
            teacher_keys,
            seed=int(instance_seed) * 100000 + int(step),
            maximum_pairs=maximum_pairs,
        ),
    )


def collect_ranking_states(
    config: dict[str, Any],
    records: Sequence[GeneratedInstanceRecord],
    *,
    maximum_pairs: int,
) -> list[RankingState]:
    states: list[RankingState] = []
    teacher = HeuristicPolicy()
    for record in records:
        environment = AssemblySchedulingEnv(config)
        observation = environment.reset(record.instance)
        step = 0
        while not (environment.terminated or environment.truncated):
            state = _ranking_state(
                environment,
                observation,
                instance_seed=int(record.metadata["seed"]),
                step=step,
                maximum_pairs=maximum_pairs,
            )
            if state is not None:
                states.append(state)
            action = teacher.select_action(environment)
            observation, _, _, _, _ = environment.step(action)
            step += 1
    return states


def _state_loss(
    logits: torch.Tensor,
    state: RankingState,
) -> torch.Tensor:
    action_count = int(state.action_mask.shape[0])
    log_probabilities = functional.log_softmax(
        logits[:action_count], dim=0
    )
    best_key = min(state.teacher_keys)
    best_actions = torch.as_tensor(
        [
            action
            for action, key in zip(
                state.legal_actions, state.teacher_keys
            )
            if key == best_key
        ],
        dtype=torch.long,
        device=logits.device,
    )
    cross_entropy = -torch.logsumexp(
        log_probabilities.index_select(0, best_actions), dim=0
    )
    if not state.pairwise_actions:
        return cross_entropy
    better = torch.as_tensor(
        [pair[0] for pair in state.pairwise_actions],
        dtype=torch.long,
        device=logits.device,
    )
    worse = torch.as_tensor(
        [pair[1] for pair in state.pairwise_actions],
        dtype=torch.long,
        device=logits.device,
    )
    ranking = functional.softplus(
        -(logits.index_select(0, better) - logits.index_select(0, worse))
    ).mean()
    return cross_entropy + ranking


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(model_logits: np.ndarray, teacher_keys) -> float:
    unique_keys = sorted(set(teacher_keys))
    teacher_utility = np.asarray(
        [-float(unique_keys.index(key)) for key in teacher_keys],
        dtype=np.float64,
    )
    model_ranks = _rank(model_logits)
    teacher_ranks = _rank(teacher_utility)
    if len(model_logits) < 2 or np.std(teacher_ranks) <= 1e-12:
        return math.nan
    if np.std(model_ranks) <= 1e-12:
        return 0.0
    return float(np.corrcoef(model_ranks, teacher_ranks)[0, 1])


def _required_minimization_gap_percent(
    value: float,
    reference: float,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Return a strict gap, including a deterministic zero-baseline rule."""
    gap = relative_gap_percent(value, reference, tolerance=tolerance)
    if gap is not None:
        return float(gap)
    actual = float(value)
    baseline = float(reference)
    if (
        math.isfinite(actual)
        and math.isfinite(baseline)
        and abs(baseline) <= tolerance
    ):
        return 0.0 if actual <= tolerance else math.inf
    return math.inf


def evaluate_ranking_states(
    network,
    states: Sequence[RankingState],
    *,
    device: str,
    batch_size: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    network.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            logits, _ = network.forward_batch(
                [state.observation for state in batch],
                [state.action_mask for state in batch],
                device=device,
            )
            for index, state in enumerate(batch):
                legal = np.asarray(state.legal_actions, dtype=np.int64)
                values = (
                    logits[index, legal].detach().cpu().numpy().astype(float)
                )
                selected_position = int(np.argmax(values))
                selected_action = int(legal[selected_position])
                teacher_position = state.legal_actions.index(
                    state.teacher_action
                )
                if state.decision_type == "production":
                    best = float(state.teacher_keys[teacher_position][0])
                    selected = float(
                        state.teacher_keys[selected_position][0]
                    )
                    regret = (selected - best) / max(best, 1e-12)
                    variance_regret_percent = None
                else:
                    best = float(state.teacher_keys[teacher_position][0])
                    selected = float(
                        state.teacher_keys[selected_position][0]
                    )
                    regret = selected - best
                    teacher_variance = float(
                        state.candidate_records[teacher_position][
                            "projected_load_variance"
                        ]
                    )
                    selected_variance = float(
                        state.candidate_records[selected_position][
                            "projected_load_variance"
                        ]
                    )
                    variance_regret_percent = (
                        _required_minimization_gap_percent(
                            selected_variance,
                            teacher_variance,
                        )
                    )
                candidates = []
                for candidate_index, candidate in enumerate(
                    state.candidate_records
                ):
                    enriched = dict(candidate)
                    enriched["model_logit"] = float(values[candidate_index])
                    candidates.append(enriched)
                rows.append(
                    {
                        "instance_seed": state.instance_seed,
                        "step": state.step,
                        "decision_type": state.decision_type,
                        "teacher_action": state.teacher_action,
                        "selected_action": selected_action,
                        "top1": selected_action == state.teacher_action,
                        "spearman": _spearman(
                            values, state.teacher_keys
                        ),
                        "regret": float(regret),
                        "variance_regret_percent": (
                            variance_regret_percent
                        ),
                        "candidates": candidates,
                    }
                )
    metrics: dict[str, Any] = {}
    for decision_type in ("production", "worker"):
        selected = [
            row for row in rows if row["decision_type"] == decision_type
        ]
        correlations = [
            row["spearman"]
            for row in selected
            if math.isfinite(float(row["spearman"]))
        ]
        metrics[decision_type] = {
            "state_count": len(selected),
            "rankable_state_count": len(correlations),
            "mean_spearman": float(
                np.mean(correlations)
            )
            if correlations
            else None,
            "top1_accuracy": float(
                np.mean([row["top1"] for row in selected])
            )
            if selected
            else None,
            "mean_regret": float(
                np.mean([row["regret"] for row in selected])
            )
            if selected
            else None,
            "mean_variance_regret_percent": (
                float(
                    np.mean(
                        [
                            row["variance_regret_percent"]
                            for row in selected
                        ]
                    )
                )
                if decision_type == "worker" and selected
                else None
            ),
        }
    return metrics, rows


def _ranking_acceptance_checks(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    def finite_value(
        decision_type: str,
        name: str,
        *,
        missing: float,
    ) -> float:
        value = metrics.get(decision_type, {}).get(name)
        return (
            float(value)
            if value is not None and math.isfinite(float(value))
            else missing
        )

    return {
        "production_spearman": finite_value(
            "production", "mean_spearman", missing=-math.inf
        )
        >= float(thresholds["minimum_production_spearman"]),
        "worker_spearman": finite_value(
            "worker", "mean_spearman", missing=-math.inf
        )
        >= float(thresholds["minimum_worker_spearman"]),
        "production_duration_regret": finite_value(
            "production", "mean_regret", missing=math.inf
        )
        <= float(thresholds["maximum_production_regret_percent"]) / 100.0,
        "worker_fatigue_regret": finite_value(
            "worker", "mean_regret", missing=math.inf
        )
        <= float(thresholds["maximum_worker_fatigue_regret"]),
        "worker_variance_regret": finite_value(
            "worker",
            "mean_variance_regret_percent",
            missing=math.inf,
        )
        <= float(thresholds["maximum_worker_variance_regret_percent"]),
    }


def _mean_supervised_loss(
    network,
    states: Sequence[RankingState],
    *,
    device: str,
    batch_size: int,
) -> float:
    network.eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            logits, _ = network.forward_batch(
                [state.observation for state in batch],
                [state.action_mask for state in batch],
                device=device,
            )
            values.extend(
                float(_state_loss(logits[index], state).item())
                for index, state in enumerate(batch)
            )
    return float(np.mean(values))


def _train_behavior_cloning_network(
    network,
    training_states: Sequence[RankingState],
    held_out_states: Sequence[RankingState],
    *,
    settings: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], int, float, list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    maximum_epochs = int(settings["maximum_epochs"])
    patience = int(settings["patience_epochs"])
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    shuffle = random.Random(int(settings.get("algorithm_seed", 190)))
    for epoch in range(1, maximum_epochs + 1):
        indices = list(range(len(training_states)))
        shuffle.shuffle(indices)
        network.train()
        training_losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = [training_states[index] for index in selected]
            logits, _ = network.forward_batch(
                [state.observation for state in batch],
                [state.action_mask for state in batch],
                device=device,
            )
            loss = torch.stack(
                [
                    _state_loss(logits[index], state)
                    for index, state in enumerate(batch)
                ]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu().item()))
        validation_loss = _mean_supervised_loss(
            network,
            held_out_states,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(training_losses)),
                "held_out_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("behavior cloning produced no checkpoint")
    return best_state, best_epoch, best_loss, history


def _feature_oracle_metrics(
    states: Sequence[RankingState],
) -> dict[str, dict[str, float | int | None]]:
    correlations: dict[str, list[float]] = {
        "production": [],
        "worker": [],
    }
    for state in states:
        predicted_keys: list[tuple[float, ...]] = []
        for candidate in state.candidate_records:
            edge = candidate["edge_features"]
            if state.decision_type == "production":
                operation_index = state.observation.node_ids[
                    "operation"
                ].index(candidate["operation_id"])
                operation_names = state.observation.node_feature_names[
                    "operation"
                ]
                sequence = float(
                    state.observation.node_features["operation"][
                        operation_index,
                        operation_names.index("sequence_norm"),
                    ]
                )
                predicted_keys.append(
                    (
                        float(edge["processing_time_norm"])
                        + float(edge["reconfiguration_time_norm"]),
                        1.0 - float(edge["configuration_match"]),
                        sequence,
                    )
                )
            else:
                predicted_keys.append(
                    (
                        float(edge["projected_fatigue_ratio"]),
                        float(candidate["projected_load_variance"]),
                    )
                )
        unique_keys = sorted(set(predicted_keys))
        utility = np.asarray(
            [-float(unique_keys.index(key)) for key in predicted_keys],
            dtype=np.float64,
        )
        correlation = _spearman(utility, state.teacher_keys)
        if math.isfinite(correlation):
            correlations[state.decision_type].append(correlation)
    return {
        decision_type: {
            "rankable_state_count": len(values),
            "mean_spearman": float(np.mean(values)) if values else None,
            "minimum_spearman": float(np.min(values)) if values else None,
        }
        for decision_type, values in correlations.items()
    }


def _candidate_count_slices(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for decision_type in ("production", "worker"):
        buckets: dict[str, list[float]] = {}
        for row in rows:
            if row["decision_type"] != decision_type:
                continue
            count = len(row["candidates"])
            bucket = (
                "2"
                if count == 2
                else "3-4"
                if count <= 4
                else "5-8"
                if count <= 8
                else "9+"
            )
            correlation = float(row["spearman"])
            if math.isfinite(correlation):
                buckets.setdefault(bucket, []).append(correlation)
        result[decision_type] = {
            bucket: {
                "rankable_state_count": len(values),
                "mean_spearman": float(np.mean(values)) if values else None,
            }
            for bucket, values in sorted(buckets.items())
        }
    return result


def run_p4(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[GeneratedInstanceRecord], list[RankingState]]:
    _require_passed("p1", "p2", "p3")
    settings = config["m1_gates"]["bc"]
    seed_start = int(config["m1_gates"]["diagnostic_seed_start"])
    seed_count = int(config["m1_gates"]["diagnostic_instance_count"])
    seeds = list(range(seed_start, seed_start + seed_count))
    directory = _prepare_stage("p4", config, seeds)
    records = _p5_instance_snapshot(
        directory,
        config,
        seeds,
        snapshot_role="p4",
    )
    training_count = int(config["m1_gates"]["bc_train_instance_count"])
    training_records = records[:training_count]
    held_out_records = records[training_count:]
    maximum_pairs = int(settings["maximum_ranking_pairs_per_state"])
    ranking_cache = directory / "ranking_states.pt"
    if ranking_cache.exists():
        cached_states = _load_ranking_cache(
            ranking_cache,
            seeds=seeds,
            training_count=training_count,
            maximum_pairs=maximum_pairs,
        )
        training_states = cached_states["training_states"]
        held_out_states = cached_states["held_out_states"]
    else:
        training_states = _collect_ranking_states_cached(
            directory,
            config,
            training_records,
            maximum_pairs=maximum_pairs,
        )
        held_out_states = _collect_ranking_states_cached(
            directory,
            config,
            held_out_records,
            maximum_pairs=maximum_pairs,
        )
        _save_ranking_cache(
            ranking_cache,
            seeds=seeds,
            training_count=training_count,
            maximum_pairs=maximum_pairs,
            training_states=training_states,
            held_out_states=held_out_states,
        )
    if not training_states or not held_out_states:
        gate = _write_gate(
            directory,
            stage="p4",
            passed=False,
            checks={"non_empty_train_and_holdout": False},
        )
        return gate, records, held_out_states

    set_seed(int(settings.get("algorithm_seed", 190)))
    bootstrap = training_states[0].observation
    network = build_actor_critic(bootstrap, config["network"]).to(
        config["device"]
    )
    preflight_metrics, _ = evaluate_ranking_states(
        network,
        held_out_states,
        device=config["device"],
    )
    thresholds = config["m1_gates"]["thresholds"]
    preflight_checks = _ranking_acceptance_checks(
        preflight_metrics,
        thresholds,
    )
    if all(preflight_checks.values()):
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in network.state_dict().items()
        }
        best_epoch = 0
        best_loss = _mean_supervised_loss(
            network,
            held_out_states,
            device=config["device"],
            batch_size=int(settings["batch_size"]),
        )
        history = [
            {
                "epoch": 0,
                "training_loss": None,
                "held_out_loss": best_loss,
                "selection": "structural_monotone_preflight",
            }
        ]
    else:
        best_state, best_epoch, best_loss, history = (
            _train_behavior_cloning_network(
                network,
                training_states,
                held_out_states,
                settings=settings,
                device=config["device"],
            )
        )
    network.load_state_dict(best_state)
    metrics, rows = evaluate_ranking_states(
        network,
        held_out_states,
        device=config["device"],
    )
    checks = _ranking_acceptance_checks(metrics, thresholds)
    passed = all(checks.values())
    torch.save(
        {
            "diagnostic_only": True,
            "network": best_state,
            "network_spec": network.network_spec(),
            "training_seeds": seeds[:training_count],
            "held_out_seeds": seeds[training_count:],
            "best_epoch": best_epoch,
        },
        directory / "bc_diagnostic_checkpoint.pt",
    )
    write_json(directory / "history.json", history)
    write_json(directory / "held_out_metrics.json", metrics)
    write_json(
        directory / "worst_50_states.json",
        sorted(
            rows,
            key=lambda row: (
                row["spearman"]
                if math.isfinite(float(row["spearman"]))
                else 1.0,
                -row["regret"],
            ),
        )[:50],
    )
    gate = _write_gate(
        directory,
        stage="p4",
        passed=passed,
        checks=checks,
        diagnostics={
            "training_state_count": len(training_states),
            "held_out_state_count": len(held_out_states),
            "best_epoch": best_epoch,
            "best_held_out_loss": best_loss,
            "preflight_metrics": preflight_metrics,
            "preflight_checks": preflight_checks,
            "metrics": metrics,
        },
    )
    return gate, records, held_out_states


def _load_ranking_cache(
    path: Path,
    *,
    seeds: Sequence[int],
    training_count: int,
    maximum_pairs: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "method_version": M1_METHOD_VERSION,
        "policy_head_version": 6,
        "seeds": [int(seed) for seed in seeds],
        "training_count": int(training_count),
        "maximum_pairs": int(maximum_pairs),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                "ranking cache version mismatch for "
                f"{field}: expected {value!r}, got {payload.get(field)!r}. "
                "older ranking caches cannot be loaded by the v6 pipeline."
            )
    if not isinstance(payload.get("training_states"), list) or not isinstance(
        payload.get("held_out_states"), list
    ):
        raise ValueError("ranking cache is missing serialized ranking states")
    return payload


def _save_ranking_cache(
    path: Path,
    *,
    seeds: Sequence[int],
    training_count: int,
    maximum_pairs: int,
    training_states: Sequence[RankingState],
    held_out_states: Sequence[RankingState],
) -> None:
    torch.save(
        {
            "method_version": M1_METHOD_VERSION,
            "policy_head_version": 6,
            "seeds": [int(seed) for seed in seeds],
            "training_count": int(training_count),
            "maximum_pairs": int(maximum_pairs),
            "training_states": list(training_states),
            "held_out_states": list(held_out_states),
        },
        path,
    )


def _collect_ranking_states_cached(
    directory: Path,
    config: dict[str, Any],
    records: Sequence[GeneratedInstanceRecord],
    *,
    maximum_pairs: int,
) -> list[RankingState]:
    """Collect ranking states with one atomic, versioned shard per instance."""
    shard_directory = directory / "ranking_state_shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    combined: list[RankingState] = []
    for record in records:
        seed = int(record.metadata["seed"])
        record_hash = sha256_bytes(canonical_json_bytes(record.to_dict()))
        shard_path = shard_directory / f"seed_{seed}.pt"
        if shard_path.exists():
            payload = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )
            expected = {
                "method_version": M1_METHOD_VERSION,
                "policy_head_version": 6,
                "seed": seed,
                "record_sha256": record_hash,
                "maximum_pairs": int(maximum_pairs),
            }
            for field, value in expected.items():
                if payload.get(field) != value:
                    raise ValueError(
                        "ranking cache shard version mismatch for "
                        f"{field}: expected {value!r}, "
                        f"got {payload.get(field)!r}. v3 ranking caches "
                        "cannot be loaded by the v6 pipeline."
                    )
            states = payload.get("states")
            if not isinstance(states, list):
                raise ValueError(
                    f"ranking cache shard {shard_path} has no state list"
                )
        else:
            states = collect_ranking_states(
                config,
                [record],
                maximum_pairs=maximum_pairs,
            )
            payload = {
                "method_version": M1_METHOD_VERSION,
                "policy_head_version": 6,
                "seed": seed,
                "record_sha256": record_hash,
                "maximum_pairs": int(maximum_pairs),
                "states": states,
            }
            temporary = shard_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(shard_path)
        combined.extend(states)
    return combined


def run_p4_1(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[GeneratedInstanceRecord], list[RankingState]]:
    """Diagnose and retest the candidate-relative ranking repair."""
    _require_passed("p1", "p2", "p3")
    original_gate_path = STAGE_DIRECTORIES["p4"] / "gate.json"
    original_checkpoint = (
        STAGE_DIRECTORIES["p4"] / "bc_diagnostic_checkpoint.pt"
    )
    if not original_gate_path.exists() or not original_checkpoint.exists():
        raise RuntimeError("P4 failure artifacts are required for P4.1")
    original_gate = json.loads(
        original_gate_path.read_text(encoding="utf-8")
    )
    if bool(original_gate.get("passed")):
        raise RuntimeError("P4.1 is only applicable after a failed P4 gate")

    settings = config["m1_gates"]["bc"]
    seed_start = int(config["m1_gates"]["diagnostic_seed_start"])
    seed_count = int(config["m1_gates"]["diagnostic_instance_count"])
    seeds = list(range(seed_start, seed_start + seed_count))
    directory = _prepare_stage("p4_1", config, seeds)
    records = _p5_instance_snapshot(
        directory,
        config,
        seeds,
        snapshot_role="p4_1",
    )
    training_count = int(config["m1_gates"]["bc_train_instance_count"])
    maximum_pairs = int(settings["maximum_ranking_pairs_per_state"])
    ranking_cache = directory / "ranking_states.pt"
    if ranking_cache.exists():
        cached_states = _load_ranking_cache(
            ranking_cache,
            seeds=seeds,
            training_count=training_count,
            maximum_pairs=maximum_pairs,
        )
        training_states = cached_states["training_states"]
        held_out_states = cached_states["held_out_states"]
    else:
        training_states = collect_ranking_states(
            config,
            records[:training_count],
            maximum_pairs=maximum_pairs,
        )
        held_out_states = collect_ranking_states(
            config,
            records[training_count:],
            maximum_pairs=maximum_pairs,
        )
        _save_ranking_cache(
            ranking_cache,
            seeds=seeds,
            training_count=training_count,
            maximum_pairs=maximum_pairs,
            training_states=training_states,
            held_out_states=held_out_states,
        )
    if not training_states or not held_out_states:
        gate = _write_gate(
            directory,
            stage="p4.1",
            passed=False,
            checks={"non_empty_train_and_holdout": False},
        )
        return gate, records, held_out_states

    baseline_network = build_actor_critic(
        training_states[0].observation,
        config["network"],
    ).to(config["device"])
    original_payload = torch.load(
        original_checkpoint,
        map_location=config["device"],
        weights_only=False,
    )
    assert_network_config_matches_spec(
        baseline_network.network_spec(),
        original_payload["network_spec"],
    )
    baseline_network.load_state_dict(original_payload["network"])
    baseline_training_metrics, _ = evaluate_ranking_states(
        baseline_network,
        training_states,
        device=config["device"],
    )
    baseline_held_metrics, baseline_rows = evaluate_ranking_states(
        baseline_network,
        held_out_states,
        device=config["device"],
    )
    feature_oracle = _feature_oracle_metrics(held_out_states)

    set_seed(int(settings.get("algorithm_seed", 190)))
    repaired_network = build_actor_critic(
        training_states[0].observation,
        config["network"],
    ).to(config["device"])
    best_state, best_epoch, best_loss, history = (
        _train_behavior_cloning_network(
            repaired_network,
            training_states,
            held_out_states,
            settings=settings,
            device=config["device"],
        )
    )
    repaired_network.load_state_dict(best_state)
    repaired_training_metrics, _ = evaluate_ranking_states(
        repaired_network,
        training_states,
        device=config["device"],
    )
    repaired_held_metrics, repaired_rows = evaluate_ranking_states(
        repaired_network,
        held_out_states,
        device=config["device"],
    )
    thresholds = config["m1_gates"]["thresholds"]
    checks = {
        **_ranking_acceptance_checks(repaired_held_metrics, thresholds),
        "feature_oracle_production_observable": feature_oracle[
            "production"
        ]["mean_spearman"]
        >= 0.99,
        "feature_oracle_worker_observable": feature_oracle["worker"][
            "mean_spearman"
        ]
        >= 0.99,
    }
    passed = all(checks.values())
    torch.save(
        {
            "diagnostic_only": True,
            "p4_1_repair": True,
            "network": best_state,
            "network_spec": repaired_network.network_spec(),
            "training_seeds": seeds[:training_count],
            "held_out_seeds": seeds[training_count:],
            "best_epoch": best_epoch,
        },
        directory / "bc_diagnostic_checkpoint.pt",
    )
    diagnosis = {
        "root_cause": (
            "Candidate features contain the teacher ordering, but the original "
            "contextual action heads overfit the 15 training instances and do "
            "not preserve the simple ordering on held-out instances."
        ),
        "training_state_count": len(training_states),
        "held_out_state_count": len(held_out_states),
        "baseline_training_metrics": baseline_training_metrics,
        "baseline_held_out_metrics": baseline_held_metrics,
        "feature_oracle_metrics": feature_oracle,
        "baseline_candidate_count_slices": _candidate_count_slices(
            baseline_rows
        ),
        "repair": {
            "candidate_relative_features": {
                "production": [
                    "processing_time_norm + reconfiguration_time_norm"
                ],
                "worker": [
                    "projected_fatigue_ratio",
                    "incremental_load_variance_norm",
                ],
            },
            "context": "small non-negative learned residual gate",
            "top1_ties": "probability mass over all teacher-optimal actions",
        },
        "repaired_training_metrics": repaired_training_metrics,
        "repaired_held_out_metrics": repaired_held_metrics,
        "repaired_candidate_count_slices": _candidate_count_slices(
            repaired_rows
        ),
    }
    write_json(directory / "diagnosis.json", diagnosis)
    write_json(directory / "history.json", history)
    write_json(
        directory / "before_after_metrics.json",
        {
            "before": baseline_held_metrics,
            "after": repaired_held_metrics,
            "feature_oracle": feature_oracle,
        },
    )
    write_json(directory / "held_out_rows.json", repaired_rows)
    write_json(
        directory / "worst_50_states.json",
        sorted(
            repaired_rows,
            key=lambda row: (
                row["spearman"]
                if math.isfinite(float(row["spearman"]))
                else 1.0,
                -row["regret"],
            ),
        )[:50],
    )
    gate = _write_gate(
        directory,
        stage="p4.1",
        passed=passed,
        checks=checks,
        diagnostics={
            "best_epoch": best_epoch,
            "best_held_out_loss": best_loss,
            "metrics": repaired_held_metrics,
            "feature_oracle": feature_oracle,
        },
    )
    decision_path = M1_ROOT / (
        "READY_FOR_P5.json" if passed else "STOPPED_AFTER_P4_1.json"
    )
    write_json(
        decision_path,
        {
            "last_completed_gate": "P4.1",
            "passed": passed,
            "failed_checks": [
                name for name, value in checks.items() if not value
            ],
            "p5_started": False,
            "p6_started": False,
            "seed11_600_started": False,
        },
    )
    if (
        not passed
        and bool(settings.get("select_context_pruned_diagnostic", False))
    ):
        gate = finalize_p4_1_context_pruning(config)
    return gate, records, held_out_states


def finalize_p4_1_context_pruning(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select the diagnostic-only raw anchor when context overfits P4.1."""
    _require_passed("p1", "p2", "p3")
    seed_start = int(config["m1_gates"]["diagnostic_seed_start"])
    seed_count = int(config["m1_gates"]["diagnostic_instance_count"])
    directory = _prepare_stage(
        "p4_1",
        config,
        tuple(range(seed_start, seed_start + seed_count)),
    )
    checkpoint_path = directory / "bc_diagnostic_checkpoint.pt"
    cache_path = directory / "ranking_states.pt"
    if not checkpoint_path.exists() or not cache_path.exists():
        raise RuntimeError("P4.1 checkpoint and ranking cache are required")
    training_count = int(config["m1_gates"]["bc_train_instance_count"])
    maximum_pairs = int(
        config["m1_gates"]["bc"]["maximum_ranking_pairs_per_state"]
    )
    cached_states = _load_ranking_cache(
        cache_path,
        seeds=tuple(range(seed_start, seed_start + seed_count)),
        training_count=training_count,
        maximum_pairs=maximum_pairs,
    )
    training_states = cached_states["training_states"]
    held_out_states = cached_states["held_out_states"]
    # Re-save with the stable module-qualified RankingState name.
    torch.save(cached_states, cache_path)

    payload = torch.load(
        checkpoint_path,
        map_location=config["device"],
        weights_only=False,
    )
    network = build_actor_critic(
        training_states[0].observation,
        config["network"],
    ).to(config["device"])
    assert_network_config_matches_spec(
        network.network_spec(),
        payload["network_spec"],
    )
    network.load_state_dict(payload["network"])
    already_pruned = bool(payload.get("context_pruned_after_bc", False))
    stored_contextual_metrics = payload.get(
        "contextual_metrics_before_pruning"
    )
    if already_pruned and stored_contextual_metrics is not None:
        contextual_metrics = stored_contextual_metrics
        contextual_loss = float(payload["contextual_held_out_loss"])
    else:
        contextual_metrics, _ = evaluate_ranking_states(
            network, held_out_states, device=config["device"]
        )
        contextual_loss = _mean_supervised_loss(
            network,
            held_out_states,
            device=config["device"],
            batch_size=int(config["m1_gates"]["bc"]["batch_size"]),
        )
    with torch.no_grad():
        network.production_context_gate.fill_(-20.0)
        network.worker_context_gate.fill_(-20.0)
    pruned_training_metrics, _ = evaluate_ranking_states(
        network, training_states, device=config["device"]
    )
    pruned_metrics, pruned_rows = evaluate_ranking_states(
        network, held_out_states, device=config["device"]
    )
    pruned_loss = _mean_supervised_loss(
        network,
        held_out_states,
        device=config["device"],
        batch_size=int(config["m1_gates"]["bc"]["batch_size"]),
    )
    feature_oracle = _feature_oracle_metrics(held_out_states)
    thresholds = config["m1_gates"]["thresholds"]
    checks = {
        **_ranking_acceptance_checks(pruned_metrics, thresholds),
        "feature_oracle_production_observable": feature_oracle[
            "production"
        ]["mean_spearman"]
        >= 0.99,
        "feature_oracle_worker_observable": feature_oracle["worker"][
            "mean_spearman"
        ]
        >= 0.99,
    }
    passed = all(checks.values())
    selected_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    payload.update(
        {
            "network": selected_state,
            "network_spec": network.network_spec(),
            "context_pruned_after_bc": True,
            "selection_target": "unchanged_P4_ranking_and_regret_gates",
            "contextual_metrics_before_pruning": contextual_metrics,
            "contextual_held_out_loss": contextual_loss,
            "selected_held_out_loss": pruned_loss,
        }
    )
    torch.save(payload, checkpoint_path)

    diagnosis_path = directory / "diagnosis.json"
    diagnosis = (
        json.loads(diagnosis_path.read_text(encoding="utf-8"))
        if diagnosis_path.exists()
        else {}
    )
    diagnosis.update(
        {
            "selection_mismatch": (
                "The lowest supervised-loss contextual checkpoint did not "
                "maximize the declared Spearman/regret acceptance gates."
            ),
            "contextual_candidate_metrics": contextual_metrics,
            "contextual_held_out_loss": contextual_loss,
            "context_pruned_training_metrics": pruned_training_metrics,
            "context_pruned_held_out_metrics": pruned_metrics,
            "context_pruned_held_out_loss": pruned_loss,
            "selected_diagnostic_variant": "primary_anchor_context_pruned",
            "formal_ppo_initialization": (
                "Fresh random network; this BC checkpoint is never loaded."
            ),
        }
    )
    write_json(diagnosis_path, diagnosis)
    previous_metrics_path = directory / "before_after_metrics.json"
    previous_metrics = (
        json.loads(previous_metrics_path.read_text(encoding="utf-8"))
        if previous_metrics_path.exists()
        else {}
    )
    write_json(
        previous_metrics_path,
        {
            "original_p4": previous_metrics.get(
                "original_p4", previous_metrics.get("before")
            ),
            "p4_1_contextual": contextual_metrics,
            "p4_1_context_pruned": pruned_metrics,
            "feature_oracle": feature_oracle,
        },
    )
    write_json(directory / "held_out_rows.json", pruned_rows)
    write_json(
        directory / "worst_50_states.json",
        sorted(
            pruned_rows,
            key=lambda row: (
                row["spearman"]
                if math.isfinite(float(row["spearman"]))
                else 1.0,
                -row["regret"],
            ),
        )[:50],
    )
    attempts_path = directory / "attempts.json"
    attempts_payload = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if attempts_path.exists()
        else {"stage": "P4.1", "attempts": []}
    )
    attempts = [
        attempt
        for attempt in attempts_payload["attempts"]
        if attempt.get("name")
        not in {"primary_anchor_contextual", "primary_anchor_context_pruned"}
    ]
    attempts.extend(
        [
            {
                "name": "primary_anchor_contextual",
                "passed": False,
                "production_mean_spearman": contextual_metrics["production"][
                    "mean_spearman"
                ],
                "worker_mean_spearman": contextual_metrics["worker"][
                    "mean_spearman"
                ],
                "held_out_loss": contextual_loss,
            },
            {
                "name": "primary_anchor_context_pruned",
                "passed": passed,
                "production_mean_spearman": pruned_metrics["production"][
                    "mean_spearman"
                ],
                "worker_mean_spearman": pruned_metrics["worker"][
                    "mean_spearman"
                ],
                "production_mean_regret": pruned_metrics["production"][
                    "mean_regret"
                ],
                "worker_mean_regret": pruned_metrics["worker"][
                    "mean_regret"
                ],
                "held_out_loss": pruned_loss,
            },
        ]
    )
    write_json(
        attempts_path,
        {"stage": "P4.1", "attempts": attempts},
    )
    gate = _write_gate(
        directory,
        stage="p4.1",
        passed=passed,
        checks=checks,
        diagnostics={
            "best_epoch": payload.get("best_epoch"),
            "contextual_held_out_loss": contextual_loss,
            "selected_held_out_loss": pruned_loss,
            "selection_target": "ranking_and_regret_hard_gates",
            "metrics": pruned_metrics,
            "feature_oracle": feature_oracle,
        },
    )
    write_json(
        M1_ROOT / "READY_FOR_P5.json",
        {
            "last_completed_gate": "P4.1",
            "passed": passed,
            "failed_checks": [
                name for name, value in checks.items() if not value
            ],
            "p5_started": False,
            "p6_started": False,
            "seed11_600_started": False,
            "diagnostic_checkpoint_only": True,
        },
    )
    stopped_path = M1_ROOT / "STOPPED_AFTER_P4_1.json"
    if stopped_path.exists():
        stopped = json.loads(stopped_path.read_text(encoding="utf-8"))
        stopped["superseded"] = passed
        stopped["superseded_by"] = (
            "result/m1/READY_FOR_P5.json" if passed else None
        )
        write_json(stopped_path, stopped)
    return gate


def _combine_rollouts(rollouts) -> RolloutBuffer:
    combined = RolloutBuffer(preserve_graph=True)
    for rollout in rollouts:
        combined.extend(rollout.buffer)
    return combined


def _overfit_validation(
    config: dict[str, Any],
    agent: PPOAgent,
    records: Sequence[GeneratedInstanceRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bootstrap_environment = AssemblySchedulingEnv(config)
    bootstrap = bootstrap_environment.reset(records[0].instance)
    policy = EvaluationPolicy(
        config,
        policy_name="ppo",
        bootstrap_observation=bootstrap,
        ppo_agent=agent,
        decode_mode="greedy",
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        _, metrics = evaluate_instance(
            config,
            instance=record.instance,
            policy_name="ppo",
            prepared_policy=policy,
        )
        heuristic = record.metadata["heuristic_metrics"]
        quality = bounded_quality_score(
            metrics["flow_time_objective"],
            metrics["reconfiguration_cost"],
            metrics["worker_load_variance"],
            config["reward"],
        )
        heuristic_quality = bounded_quality_score(
            heuristic["heuristic_flow_time"],
            heuristic["heuristic_reconfiguration_cost"],
            heuristic["worker_workload_variance"],
            config["reward"],
        )
        rows.append(
            {
                "instance_id": record.instance.instance_id,
                "seed": int(record.metadata["seed"]),
                "terminated": bool(metrics["terminated"]),
                "truncated": bool(metrics["truncated"]),
                "schedule_violation_count": len(
                    metrics["schedule_violations"]
                ),
                "flow_time_objective": metrics["flow_time_objective"],
                "heuristic_flow_time": heuristic["heuristic_flow_time"],
                "flow_gap_percent": relative_gap_percent(
                    metrics["flow_time_objective"],
                    heuristic["heuristic_flow_time"],
                ),
                "reconfiguration_cost": metrics["reconfiguration_cost"],
                "heuristic_reconfiguration_cost": heuristic[
                    "heuristic_reconfiguration_cost"
                ],
                "worker_load_variance": metrics["worker_load_variance"],
                "heuristic_worker_load_variance": heuristic[
                    "worker_workload_variance"
                ],
                "quality_score": quality,
                "heuristic_quality_score": heuristic_quality,
                "worker_assignment_count": int(
                    metrics.get("worker_assignment_count", 0)
                ),
                "worker_assignment_variance_reward_sum": float(
                    metrics.get(
                        "worker_assignment_variance_reward_sum", 0.0
                    )
                ),
                "worker_assignment_variance_reward_abs_sum": float(
                    metrics.get(
                        "worker_assignment_variance_reward_abs_sum", 0.0
                    )
                ),
                "worker_assignment_nonzero_variance_reward_count": int(
                    metrics.get(
                        "worker_assignment_nonzero_variance_reward_count",
                        0,
                    )
                ),
            }
        )

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))

    aggregate: dict[str, Any] = {
        "instance_count": len(rows),
        "completion_rate": float(
            np.mean(
                [
                    row["terminated"] and not row["truncated"]
                    for row in rows
                ]
            )
        ),
        "schedule_violation_count": sum(
            int(row["schedule_violation_count"]) for row in rows
        ),
        "mean_flow_gap_percent": mean("flow_gap_percent"),
        "mean_reconfiguration_cost": mean("reconfiguration_cost"),
        "mean_heuristic_reconfiguration_cost": mean(
            "heuristic_reconfiguration_cost"
        ),
        "mean_worker_load_variance": mean("worker_load_variance"),
        "mean_heuristic_worker_load_variance": mean(
            "heuristic_worker_load_variance"
        ),
        "mean_quality_score": mean("quality_score"),
        "mean_heuristic_quality_score": mean("heuristic_quality_score"),
        "worker_assignment_count": sum(
            int(row["worker_assignment_count"]) for row in rows
        ),
        "worker_assignment_variance_reward_sum": sum(
            float(row["worker_assignment_variance_reward_sum"])
            for row in rows
        ),
        "worker_assignment_variance_reward_abs_sum": sum(
            float(row["worker_assignment_variance_reward_abs_sum"])
            for row in rows
        ),
        "worker_assignment_nonzero_variance_reward_count": sum(
            int(row["worker_assignment_nonzero_variance_reward_count"])
            for row in rows
        ),
    }
    aggregate["mean_reconfiguration_cost_gap_percent"] = (
        _required_minimization_gap_percent(
            aggregate["mean_reconfiguration_cost"],
            aggregate["mean_heuristic_reconfiguration_cost"],
        )
    )
    aggregate["mean_worker_load_variance_gap_percent"] = (
        _required_minimization_gap_percent(
            aggregate["mean_worker_load_variance"],
            aggregate["mean_heuristic_worker_load_variance"],
        )
    )
    aggregate["mean_quality_gap_percent"] = (
        _required_minimization_gap_percent(
            aggregate["mean_quality_score"],
            aggregate["mean_heuristic_quality_score"],
        )
    )
    assignment_count = int(aggregate["worker_assignment_count"])
    aggregate["mean_absolute_worker_assignment_variance_reward"] = (
        float(aggregate["worker_assignment_variance_reward_abs_sum"])
        / assignment_count
        if assignment_count
        else 0.0
    )
    return aggregate, rows


def _overfit_checks(
    aggregate: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    tolerance = 1e-9
    return {
        "completion_rate": aggregate["completion_rate"] >= 1.0 - tolerance,
        "zero_schedule_violations": aggregate["schedule_violation_count"] == 0,
        "flow_gap": aggregate["mean_flow_gap_percent"]
        <= float(thresholds["maximum_flow_gap_percent"]) + tolerance,
        "reconfiguration_cost_gap": aggregate[
            "mean_reconfiguration_cost_gap_percent"
        ]
        <= float(thresholds["maximum_cost_gap_percent"]) + tolerance,
        "worker_load_variance_gap": aggregate[
            "mean_worker_load_variance_gap_percent"
        ]
        <= float(thresholds["maximum_variance_gap_percent"]) + tolerance,
        "quality_gap": aggregate["mean_quality_gap_percent"]
        <= float(thresholds["maximum_quality_gap_percent"]) + tolerance,
    }


def _p5_promotion_decision(
    aggregate: dict[str, Any],
    checks: dict[str, bool],
    accepted_quality_score: float | None,
) -> dict[str, Any]:
    guardrails = {
        name: bool(value)
        for name, value in checks.items()
        if name != "quality_gap"
    }
    quality = float(aggregate["mean_quality_score"])
    quality_is_finite = math.isfinite(quality)
    quality_improved = bool(
        quality_is_finite
        and (
            accepted_quality_score is None
            or quality < float(accepted_quality_score) - 1e-12
        )
    )
    promoted = bool(
        guardrails and all(guardrails.values()) and quality_improved
    )
    if promoted:
        reason = (
            "accepted_anchor"
            if accepted_quality_score is None
            else "quality_improved_with_guardrails"
        )
    elif not all(guardrails.values()):
        reason = "guardrail_failed"
    elif not quality_is_finite:
        reason = "non_finite_quality"
    else:
        reason = "quality_not_improved"
    return {
        "promoted": promoted,
        "reason": reason,
        "candidate_quality_score": quality if quality_is_finite else None,
        "accepted_quality_score_before": accepted_quality_score,
        "guardrails": guardrails,
    }


def _save_p5_accepted_checkpoint(
    agent: PPOAgent,
    *,
    accepted_checkpoint: Path,
    best_checkpoint: Path,
    root_directory: Path,
    promotion: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    if not bool(promotion.get("promoted")):
        return False
    agent.save(accepted_checkpoint, metadata=metadata)
    shutil.copyfile(accepted_checkpoint, best_checkpoint)
    shutil.copyfile(
        accepted_checkpoint,
        root_directory / "accepted_checkpoint.pt",
    )
    shutil.copyfile(
        accepted_checkpoint,
        root_directory / "best_checkpoint.pt",
    )
    return True


def _p5_failure_diagnostics(
    update_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not update_rows:
        raise ValueError("P5 failure diagnostics require update rows")

    check_names = list(update_rows[0]["checks"])
    pass_counts = {
        name: sum(bool(row["checks"].get(name)) for row in update_rows)
        for name in check_names
    }
    best_quality = min(
        update_rows,
        key=lambda row: float(row["aggregate"]["mean_quality_score"]),
    )
    best_variance = min(
        update_rows,
        key=lambda row: float(
            row["aggregate"]["mean_worker_load_variance"]
        ),
    )
    best_flow = min(
        update_rows,
        key=lambda row: float(row["aggregate"]["mean_flow_gap_percent"]),
    )
    best_cost = min(
        update_rows,
        key=lambda row: float(
            row["aggregate"]["mean_reconfiguration_cost"]
        ),
    )

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "stage_update": int(row["stage_update"]),
            "global_update": int(row["global_update"]),
            "reward_phase": row["reward_phase"],
            "aggregate": row["aggregate"],
            "checks": row["checks"],
            "stability": row.get("stability"),
            "checkpoint_promotion": row.get("checkpoint_promotion"),
            "effective_relative_cost_weights": row.get(
                "effective_relative_cost_weights"
            ),
            "losses": row["losses"],
            "instance_rows": row["instance_rows"],
        }

    variance_reference = float(
        best_variance["aggregate"][
            "mean_heuristic_worker_load_variance"
        ]
    )
    quality_reference = float(
        best_quality["aggregate"]["mean_heuristic_quality_score"]
    )
    last = update_rows[-1]
    catastrophic_updates = [
        int(row["stage_update"])
        for row in update_rows
        if float(row["aggregate"]["completion_rate"]) <= 0.90
    ]
    incomplete_updates = [
        int(row["stage_update"])
        for row in update_rows
        if float(row["aggregate"]["completion_rate"]) < 1.0
    ]
    rollback_updates = [
        int(row["stage_update"])
        for row in update_rows
        if bool((row.get("stability") or {}).get("rollback"))
    ]
    return {
        "updates": len(update_rows),
        "pass_counts": pass_counts,
        "never_passed_checks": [
            name for name, count in pass_counts.items() if count == 0
        ],
        "incomplete_updates": incomplete_updates,
        "catastrophic_completion_updates": catastrophic_updates,
        "recorded_rollback_updates": rollback_updates,
        "best_quality": compact(best_quality),
        "best_variance": compact(best_variance),
        "best_flow": compact(best_flow),
        "best_cost": compact(best_cost),
        "last": compact(last),
        "tradeoff": {
            "best_variance_to_heuristic_ratio": (
                float(
                    best_variance["aggregate"][
                        "mean_worker_load_variance"
                    ]
                )
                / variance_reference
                if variance_reference > 0.0
                else None
            ),
            "best_variance_flow_gap_percent": float(
                best_variance["aggregate"]["mean_flow_gap_percent"]
            ),
            "best_variance_completion_rate": float(
                best_variance["aggregate"]["completion_rate"]
            ),
            "best_quality_gap_percent": relative_gap_percent(
                float(best_quality["aggregate"]["mean_quality_score"]),
                quality_reference,
            ),
            "last_quality_gap_percent": relative_gap_percent(
                float(last["aggregate"]["mean_quality_score"]),
                float(
                    last["aggregate"]["mean_heuristic_quality_score"]
                ),
            ),
        },
    }


def run_p5(
    config: dict[str, Any],
    records: Sequence[GeneratedInstanceRecord] | None = None,
    held_out_states: Sequence[RankingState] | None = None,
) -> dict[str, Any]:
    _require_passed("p1", "p2", "p3")
    _require_p4_expression_gate()
    settings = config["m1_gates"]["overfit"]
    seed_start = int(config["m1_gates"]["diagnostic_seed_start"])
    seed_count = int(config["m1_gates"]["diagnostic_instance_count"])
    seeds = list(range(seed_start, seed_start + seed_count))
    directory = _prepare_stage("p5", config, seeds)
    readiness_path = M1_ROOT / "READY_FOR_P5.json"
    readiness = (
        json.loads(readiness_path.read_text(encoding="utf-8"))
        if readiness_path.exists()
        else {}
    )
    readiness.update(
        {
            "last_completed_gate": (
                "P4.1"
                if (
                    STAGE_DIRECTORIES["p4_1"] / "gate.json"
                ).exists()
                else "P4"
            ),
            "p5_started": True,
            "p5_completed": False,
            "p5_passed": None,
            "p6_started": False,
            "seed11_600_started": False,
        }
    )
    write_json(readiness_path, readiness)
    if (
        records is None
        and (
            STAGE_DIRECTORIES["p4"] / "instances" / "manifest.json"
        ).exists()
    ):
        records = _p5_instance_snapshot(
            STAGE_DIRECTORIES["p4"],
            config,
            seeds,
            snapshot_role="p4",
        )
    records = _p5_instance_snapshot(
        directory,
        config,
        seeds,
        records,
    )
    if held_out_states is None:
        training_count = int(
            config["m1_gates"]["bc_train_instance_count"]
        )
        maximum_pairs = int(
            config["m1_gates"]["bc"][
                "maximum_ranking_pairs_per_state"
            ]
        )
        p4_cache = STAGE_DIRECTORIES["p4"] / "ranking_states.pt"
        if p4_cache.exists():
            cached_states = _load_ranking_cache(
                p4_cache,
                seeds=seeds,
                training_count=training_count,
                maximum_pairs=maximum_pairs,
            )
            held_out_states = cached_states["held_out_states"]
        else:
            held_out_states = _collect_ranking_states_cached(
                directory,
                config,
                records[training_count:],
                maximum_pairs=maximum_pairs,
            )

    set_seed(int(settings["algorithm_seed"]))
    bootstrap_environment = AssemblySchedulingEnv(config)
    bootstrap = bootstrap_environment.reset(records[0].instance)
    agent = PPOAgent(
        build_actor_critic(bootstrap, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    phase_controller = TrainingPhaseController.from_config(config)
    stability_controller = ValidationStabilityController.from_config(config)
    thresholds = config["m1_gates"]["thresholds"]
    stages = [int(value) for value in settings["instance_stages"]]
    maximum_updates = int(settings["maximum_updates_per_stage"])
    required_successes = int(settings["consecutive_passes"])
    stage_results: list[dict[str, Any]] = []
    global_update = 0
    current_run_has_accepted = False
    for instance_count in stages:
        selected_records = records[:instance_count]
        stage_directory = directory / f"instances_{instance_count}"
        stage_directory.mkdir(parents=True, exist_ok=True)
        accepted_checkpoint = stage_directory / "accepted_checkpoint.pt"
        best_checkpoint = stage_directory / "best_checkpoint.pt"
        safe_checkpoint = stage_directory / "safe_checkpoint.pt"
        last_online_checkpoint = (
            stage_directory / "last_online_checkpoint.pt"
        )
        accepted_quality_score: float | None = None
        accepted_checkpoint_available = False
        safe_checkpoint_available = False
        if current_run_has_accepted:
            agent.save(
                safe_checkpoint,
                metadata={
                    "checkpoint_role": "stage_entry_safe",
                    "instance_count": instance_count,
                    "global_update": global_update,
                },
            )
            safe_checkpoint_available = True
        consecutive = 0
        update_rows: list[dict[str, Any]] = []
        stage_passed = False
        write_json(
            directory / "live_status.json",
            {
                "status": "running",
                "instance_count": instance_count,
                "stage_update": 0,
                "global_update": global_update,
                "maximum_updates": maximum_updates,
                "required_consecutive_passes": required_successes,
            },
        )
        for stage_update in range(1, maximum_updates + 1):
            rollouts = []
            for record_index, record in enumerate(selected_records):
                environment = AssemblySchedulingEnv(config)
                rollouts.append(
                    _collect_serial_batch(
                        config=config,
                        agent=agent,
                        environment=environment,
                        instance=record.instance,
                        record=record,
                        episode_index=record_index,
                        sampling_start=time.perf_counter(),
                        generation_time_seconds=0.0,
                        step_limit=None,
                        reward_phase=phase_controller.phase,
                    )
                )
            losses = agent.update(_combine_rollouts(rollouts))
            global_update += 1
            aggregate, instance_rows = _overfit_validation(
                config, agent, selected_records
            )
            effective_weights = (
                agent.network.effective_relative_cost_weights()
            )
            agent.save(
                last_online_checkpoint,
                metadata={
                    "checkpoint_role": "last_online",
                    "instance_count": instance_count,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "effective_relative_cost_weights": effective_weights,
                },
            )
            shutil.copyfile(
                last_online_checkpoint,
                directory / "last_online_checkpoint.pt",
            )
            quality_key = (
                -float(aggregate["completion_rate"]),
                float(aggregate["mean_quality_score"]),
                float(aggregate["mean_flow_gap_percent"]),
                float(
                    aggregate[
                        "mean_worker_load_variance_gap_percent"
                    ]
                ),
            )
            event = phase_controller.observe_validation(
                aggregate["completion_rate"],
                completed_episodes=global_update,
                score=quality_key,
                normalized_quality_score=aggregate["mean_quality_score"],
            )
            stability = stability_controller.observe_greedy(
                quality_key,
                aggregate["completion_rate"],
                completed_episodes=global_update,
                feasibility_phase=phase_controller.phase == "feasibility",
            )
            checks = _overfit_checks(aggregate, thresholds)
            if instance_count == 20:
                ranking, _ = evaluate_ranking_states(
                    agent.network,
                    held_out_states,
                    device=config["device"],
                )
                checks.update(
                    _ranking_acceptance_checks(ranking, thresholds)
                )
            else:
                ranking = None
            promotion = _p5_promotion_decision(
                aggregate,
                checks,
                accepted_quality_score,
            )
            if bool(promotion["promoted"]):
                accepted_quality_score = float(
                    aggregate["mean_quality_score"]
                )
                accepted_checkpoint_available = True
                checkpoint_metadata = {
                    "checkpoint_role": "accepted_best",
                    "diagnostic_only": True,
                    "instance_count": instance_count,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "accepted_quality_score": accepted_quality_score,
                    "promotion": promotion,
                    "effective_relative_cost_weights": effective_weights,
                }
                _save_p5_accepted_checkpoint(
                    agent,
                    accepted_checkpoint=accepted_checkpoint,
                    best_checkpoint=best_checkpoint,
                    root_directory=directory,
                    promotion=promotion,
                    metadata=checkpoint_metadata,
                )
            if (
                aggregate["completion_rate"] >= 1.0 - 1e-12
                and aggregate["schedule_violation_count"] == 0
            ):
                agent.save(
                    safe_checkpoint,
                    metadata={
                        "checkpoint_role": "latest_safe",
                        "instance_count": instance_count,
                        "stage_update": stage_update,
                        "global_update": global_update,
                        "effective_relative_cost_weights": effective_weights,
                    },
                )
                shutil.copyfile(
                    safe_checkpoint,
                    directory / "safe_checkpoint.pt",
                )
                safe_checkpoint_available = True
            stability = dict(stability)
            stability["rollback_applied"] = False
            if bool(stability["rollback"]):
                if (
                    not safe_checkpoint_available
                    or not safe_checkpoint.exists()
                ):
                    raise RuntimeError(
                        "P5 catastrophic rollback requested without safe state"
                    )
                agent.load(safe_checkpoint, load_optimizer=True)
                stability["rollback_applied"] = True
            agent.set_learning_rate(
                stability_controller.current_learning_rate
            )
            passed_now = all(checks.values())
            consecutive = consecutive + 1 if passed_now else 0
            update_rows.append(
                {
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "reward_phase": phase_controller.phase,
                    "validation_event": event,
                    "stability": stability,
                    "checkpoint_promotion": promotion,
                    "effective_relative_cost_weights": effective_weights,
                    "aggregate": aggregate,
                    "checks": checks,
                    "consecutive_passes": consecutive,
                    "ranking": ranking,
                    "losses": losses,
                    "instance_rows": instance_rows,
                }
            )
            live_status = {
                "status": "running",
                "instance_count": instance_count,
                "stage_update": stage_update,
                "global_update": global_update,
                "maximum_updates": maximum_updates,
                "required_consecutive_passes": required_successes,
                "consecutive_passes": consecutive,
                "checks": checks,
                "aggregate": aggregate,
                "ranking": ranking,
                "stability": stability,
                "checkpoint_promotion": promotion,
                "effective_relative_cost_weights": effective_weights,
            }
            write_json(directory / "live_status.json", live_status)
            print(
                "[P5] " + json.dumps(live_status, ensure_ascii=False),
                flush=True,
            )
            if consecutive >= required_successes:
                stage_passed = True
                break
        write_json(stage_directory / "updates.json", update_rows)
        accepted_evaluation: dict[str, Any] | None = None
        if accepted_checkpoint_available:
            accepted_metadata = agent.load(
                accepted_checkpoint,
                load_optimizer=True,
            )
            accepted_aggregate, accepted_instance_rows = (
                _overfit_validation(config, agent, selected_records)
            )
            accepted_checks = _overfit_checks(
                accepted_aggregate,
                thresholds,
            )
            if instance_count == 20:
                accepted_ranking, _ = evaluate_ranking_states(
                    agent.network,
                    held_out_states,
                    device=config["device"],
                )
                accepted_checks.update(
                    _ranking_acceptance_checks(
                        accepted_ranking,
                        thresholds,
                    )
                )
            else:
                accepted_ranking = None
            accepted_evaluation = {
                "metadata": accepted_metadata,
                "aggregate": accepted_aggregate,
                "checks": accepted_checks,
                "ranking": accepted_ranking,
                "instance_rows": accepted_instance_rows,
                "effective_relative_cost_weights": (
                    agent.network.effective_relative_cost_weights()
                ),
            }
            stage_passed = bool(
                stage_passed and all(accepted_checks.values())
            )
            agent.save(
                stage_directory / "checkpoint.pt",
                metadata={
                    "checkpoint_role": "accepted_stage_handoff",
                    "diagnostic_only": True,
                    "instance_count": instance_count,
                    "stage_passed": stage_passed,
                    "global_update": global_update,
                    "accepted_source_metadata": accepted_metadata,
                },
            )
            current_run_has_accepted = True
        else:
            stage_passed = False
        stage_result = {
            "instance_count": instance_count,
            "passed": stage_passed,
            "updates_used": len(update_rows),
            "last": update_rows[-1],
            "accepted": accepted_evaluation,
            "checkpoint_paths": {
                "accepted": (
                    str(accepted_checkpoint)
                    if accepted_checkpoint_available
                    else None
                ),
                "best": (
                    str(best_checkpoint)
                    if accepted_checkpoint_available
                    else None
                ),
                "safe": (
                    str(safe_checkpoint)
                    if safe_checkpoint_available
                    else None
                ),
                "last_online": str(last_online_checkpoint),
            },
        }
        stage_results.append(stage_result)
        write_json(
            directory / "live_status.json",
            {
                "status": "stage_passed" if stage_passed else "stage_failed",
                "instance_count": instance_count,
                "stage_update": len(update_rows),
                "global_update": global_update,
                "maximum_updates": maximum_updates,
                "required_consecutive_passes": required_successes,
                "consecutive_passes": update_rows[-1][
                    "consecutive_passes"
                ],
                "checks": update_rows[-1]["checks"],
                "aggregate": update_rows[-1]["aggregate"],
                "ranking": update_rows[-1]["ranking"],
                "accepted": accepted_evaluation,
            },
        )
        if not stage_passed:
            failure_diagnostics = _p5_failure_diagnostics(update_rows)
            diagnostics_path = (
                stage_directory / "failure_diagnostics.json"
            )
            write_json(diagnostics_path, failure_diagnostics)
            write_json(directory / "stage_results.json", stage_results)
            gate = _write_gate(
                directory,
                stage="p5",
                passed=False,
                checks={f"instances_{instance_count}": False},
                diagnostics={
                    "failed_instance_count": instance_count,
                    "updates_used": len(update_rows),
                    "last_checks": update_rows[-1]["checks"],
                    "last_aggregate": update_rows[-1]["aggregate"],
                    "failure_diagnostics": str(diagnostics_path),
                    "never_passed_checks": failure_diagnostics[
                        "never_passed_checks"
                    ],
                },
            )
            readiness.update(
                {
                    "last_completed_gate": "P5",
                    "p5_completed": True,
                    "p5_passed": False,
                    "p6_started": False,
                    "seed11_600_started": False,
                }
            )
            write_json(readiness_path, readiness)
            write_json(
                M1_ROOT / "READY_FOR_P6.json",
                {
                    "last_completed_gate": "P5",
                    "passed": False,
                    "blocked_by": f"instances_{instance_count}",
                    "p6_started": False,
                    "seed11_600_started": False,
                },
            )
            write_json(
                M1_ROOT / "STOPPED_AFTER_P5.json",
                {
                    "last_completed_gate": "P5",
                    "passed": False,
                    "failed_instance_count": instance_count,
                    "failed_checks": failure_diagnostics[
                        "never_passed_checks"
                    ],
                    "failure_diagnostics": str(diagnostics_path),
                    "p6_started": False,
                    "seed11_600_started": False,
                },
            )
            return gate
    write_json(directory / "stage_results.json", stage_results)
    gate = _write_gate(
        directory,
        stage="p5",
        passed=True,
        checks={f"instances_{count}": True for count in stages},
        diagnostics={
            "stage_results": stage_results,
            "accepted_checkpoint": str(
                directory / "accepted_checkpoint.pt"
            ),
            "best_checkpoint": str(directory / "best_checkpoint.pt"),
            "safe_checkpoint": str(directory / "safe_checkpoint.pt"),
            "last_online_checkpoint": str(
                directory / "last_online_checkpoint.pt"
            ),
        },
    )
    readiness.update(
        {
            "last_completed_gate": "P5",
            "p5_completed": True,
            "p5_passed": True,
            "p6_started": False,
            "seed11_600_started": False,
        }
    )
    write_json(readiness_path, readiness)
    write_json(
        M1_ROOT / "READY_FOR_P6.json",
        {
            "last_completed_gate": "P5",
            "passed": True,
            "accepted_checkpoint": str(
                directory / "accepted_checkpoint.pt"
            ),
            "p6_started": False,
            "seed11_600_started": False,
        },
    )
    return gate


def run_p6(config: dict[str, Any]) -> dict[str, Any]:
    _require_passed("p1", "p2", "p3", "p5")
    _require_p4_expression_gate()
    directory = _prepare_stage("p6", config, (11,))
    accepted_checkpoint = (
        STAGE_DIRECTORIES["p5"] / "accepted_checkpoint.pt"
    )
    if not accepted_checkpoint.exists():
        raise RuntimeError(
            "P6 requires the accepted P5 checkpoint; last_online and safe "
            "checkpoints are not valid initialization sources"
        )
    run_directory = train(
        config,
        smoke=False,
        online_instances=True,
        algorithm_seed=11,
        parallel_envs=int(config["training"]["parallel_envs"]),
        run_name="m1_seed11_600",
        initial_checkpoint=accepted_checkpoint,
    )
    summary = json.loads(
        (run_directory / "summary.json").read_text(encoding="utf-8")
    )
    write_json(directory / "training_summary.json", summary)
    return _write_gate(
        directory,
        stage="p6",
        passed=True,
        checks={"seed11_600_completed": True},
        diagnostics={
            "run_directory": str(run_directory),
            "initial_checkpoint_role": "p5_accepted",
            "initial_checkpoint": str(accepted_checkpoint),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "final_checkpoint_evaluation": summary.get(
                "final_checkpoint_evaluation"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M1 gates serially and stop at the first failure."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--stage",
        choices=(
            "p1",
            "p2",
            "p3",
            "p4",
            "p4_1",
            "p4_1_finalize",
            "p5",
            "p6",
            "auto",
        ),
        default="auto",
    )
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    torch.set_num_threads(
        int(config["training"].get("torch_num_threads", 1))
    )
    if args.stage == "p1":
        result = run_p1(config)
    elif args.stage == "p2":
        result = run_p2(config)
    elif args.stage == "p3":
        result = run_p3(config)
    elif args.stage == "p4":
        result, _, _ = run_p4(config)
    elif args.stage == "p4_1":
        result, _, _ = run_p4_1(config)
    elif args.stage == "p4_1_finalize":
        result = finalize_p4_1_context_pruning(config)
    elif args.stage == "p5":
        result = run_p5(config)
    elif args.stage == "p6":
        result = run_p6(config)
    else:
        records = None
        held_out_states = None
        result = {}
        for runner in (run_p1, run_p2, run_p3):
            result = runner(config)
            if not result["passed"]:
                break
        else:
            result, records, held_out_states = run_p4(config)
            if not result["passed"]:
                result, records, held_out_states = run_p4_1(config)
            if result["passed"]:
                result = run_p5(config, records, held_out_states)
            if result["passed"]:
                result = run_p6(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
