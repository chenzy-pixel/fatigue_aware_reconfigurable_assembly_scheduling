from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agent.ppo import read_checkpoint_network_spec
from configs import load_config, project_path
from configs.config import public_config


ROOT = Path(__file__).resolve().parent
PROTOCOL = "v7_e0_e5_protocol_v1"
CODE_VERSION = "policy-head-v7-code"
RESULT_SCHEMA_VERSION = "4.0.0"
SCREEN_SEEDS = (11, 37, 53)
FORMAL_SEEDS = (11, 23, 37, 53, 71)
SAMPLING_SEEDS = (100011, 100012, 100013)
EVALUATION_SPLITS = ("test", "ood", "stress")
ARMS = {
    "c0": "configs/v7/c0_v6_control.json",
    "e1": "configs/v7/e1_context_exception.json",
    "e2": "configs/v7/e2_commit_set.json",
    "e3": "configs/v7/e3_future_value.json",
    "e4": "configs/v7/e4_conditional_wait.json",
    "e5": "configs/v7/e5_variance_scale.json",
    "full": "configs/v7/full_v7.json",
}
METADATA_DIFFS = {"experiment_name", "method_version"}
ARM_DIFF_WHITELIST = {
    "c0": set(),
    "e1": {
        "network.policy_head_version",
        "network.candidate_context_mode",
        "network.residual_context_gate_initial_logit",
        "network.residual_scale_ratio",
        "network.production_commit_set_scorer",
        "network.future_value_features",
        "network.worker_common_context_enabled",
    },
    "e2": {
        "network.policy_head_version",
        "network.production_commit_set_scorer",
        "network.future_value_features",
        "network.worker_common_context_enabled",
    },
    "e3": {
        "network.policy_head_version",
        "network.production_commit_set_scorer",
        "network.future_value_features",
        "network.worker_common_context_enabled",
        "network.production_relative_feature_names",
        "network.worker_relative_feature_names",
        "network.production_relative_initial_weights",
        "network.worker_relative_initial_weights",
    },
    "e4": {
        "environment.worker_resource_control.conditional_wait.enabled",
        "environment.worker_resource_control.conditional_wait.max_wait_minutes",
        "environment.worker_resource_control.conditional_wait.minimum_fatigue_ratio_improvement",
        "environment.worker_resource_control.conditional_wait.minimum_duration_improvement_ticks",
        "environment.worker_resource_control.conditional_wait.max_consecutive_waits",
        "environment.worker_resource_control.conditional_wait.require_full_matching",
        "environment.worker_resource_control.conditional_wait.require_horizon_feasible",
    },
    "e5": {"reward.variance_scale"},
    "full": {
        "network.policy_head_version",
        "network.candidate_context_mode",
        "network.residual_context_gate_initial_logit",
        "network.residual_scale_ratio",
        "network.production_commit_set_scorer",
        "network.future_value_features",
        "network.worker_common_context_enabled",
        "network.production_relative_feature_names",
        "network.worker_relative_feature_names",
        "network.production_relative_initial_weights",
        "network.worker_relative_initial_weights",
        "environment.worker_resource_control.conditional_wait.enabled",
        "environment.worker_resource_control.conditional_wait.max_wait_minutes",
        "environment.worker_resource_control.conditional_wait.minimum_fatigue_ratio_improvement",
        "environment.worker_resource_control.conditional_wait.minimum_duration_improvement_ticks",
        "environment.worker_resource_control.conditional_wait.max_consecutive_waits",
        "environment.worker_resource_control.conditional_wait.require_full_matching",
        "environment.worker_resource_control.conditional_wait.require_horizon_feasible",
        "reward.variance_scale",
    },
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(child, path))
    return result


def config_differences(first: dict, second: dict) -> set[str]:
    left = _flatten(public_config(first))
    right = _flatten(public_config(second))
    return {
        key
        for key in set(left) | set(right)
        if left.get(key, object()) != right.get(key, object())
    }


def validate_arm_configs() -> dict[str, set[str]]:
    configs = {arm: load_config(path) for arm, path in ARMS.items()}
    control = configs["c0"]
    differences: dict[str, set[str]] = {}
    for arm, config in configs.items():
        if config.get("experiment_suite_version") != PROTOCOL:
            raise ValueError(f"{arm} has the wrong experiment protocol")
        if bool(config["training"].get("forced_action_compression", False)):
            raise ValueError(f"{arm} enables excluded P5/E6 behavior")
        if str(config["training"]["two_stage"].get(
            "quality_checkpoint_promotion"
        )) != "balanced_guarded_v7":
            raise ValueError(f"{arm} does not use balanced_guarded_v7")
        changed = config_differences(control, config) - METADATA_DIFFS
        allowed = ARM_DIFF_WHITELIST[arm]
        unexpected = changed - allowed
        if unexpected:
            raise ValueError(
                f"{arm} contains non-whitelisted experimental changes: "
                f"{sorted(unexpected)}"
            )
        if arm == "e5" and changed != {"reward.variance_scale"}:
            raise ValueError("E5 must change only reward.variance_scale")
        differences[arm] = changed
    return differences


def _hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(path.resolve() for path in paths)):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def suite_input_hash() -> str:
    paths = [
        path
        for path in ROOT.rglob("*.py")
        if not any(
            part in {".git", ".venv", "result", "__pycache__"}
            for part in path.relative_to(ROOT).parts
        )
    ]
    paths.extend((ROOT / "configs").rglob("*.json"))
    paths.extend((ROOT / "data" / "manifests").rglob("*"))
    files = [path for path in paths if path.is_file()]
    digest = hashlib.sha256()
    digest.update(PROTOCOL.encode("utf-8"))
    digest.update(CODE_VERSION.encode("utf-8"))
    digest.update(_hash_files(files).encode("ascii"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_valid(path: Path, required: str) -> bool:
    try:
        value = json.loads((path / required).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(value, dict)


class Orchestrator:
    def __init__(
        self,
        *,
        parallel_envs: int,
        dry_run: bool,
        max_retries: int,
    ) -> None:
        self.parallel_envs = int(parallel_envs)
        if self.parallel_envs < 1:
            raise ValueError("parallel_envs must be positive")
        self.dry_run = bool(dry_run)
        self.max_retries = int(max_retries)
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.input_hash = suite_input_hash()
        self.suite_dir = (
            ROOT / "result" / "experiments" /
            f"{PROTOCOL}_{self.input_hash[:12]}"
        )
        self.state_path = self.suite_dir / "state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("input_hash") != self.input_hash:
                raise ValueError("suite state input hash does not match")
            return state
        return {
            "protocol": PROTOCOL,
            "code_version": CODE_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "input_hash": self.input_hash,
            "cpu_only_serial_training": True,
            "tasks": {},
        }

    def _save_state(self) -> None:
        if not self.dry_run:
            atomic_json(self.state_path, self.state)

    def _execute(
        self,
        task_id: str,
        command_builder,
        *,
        artifact_builder,
        required: str,
    ) -> Path:
        previous = self.state["tasks"].get(task_id, {})
        previous_artifact = Path(previous.get("artifact", ""))
        if (
            previous.get("status") == "complete"
            and previous.get("input_hash") == self.input_hash
            and previous_artifact.exists()
            and _artifact_valid(previous_artifact, required)
        ):
            print(f"[skip] {task_id}")
            return previous_artifact
        attempts = int(previous.get("attempts", 0))
        local_attempt = 0
        while local_attempt <= self.max_retries:
            local_attempt += 1
            attempts += 1
            run_name = task_id if attempts == 1 else f"{task_id}_r{attempts}"
            artifact = artifact_builder(run_name)
            command = command_builder(run_name)
            print("[run]", " ".join(str(value) for value in command))
            if self.dry_run:
                return artifact
            self.suite_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.suite_dir / "logs" / f"{run_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.state["tasks"][task_id] = {
                "status": "running",
                "attempts": attempts,
                "input_hash": self.input_hash,
                "artifact": str(artifact),
                "command": command,
                "started_at": time.time(),
            }
            self._save_state()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            valid = completed.returncode == 0 and _artifact_valid(
                artifact, required
            )
            self.state["tasks"][task_id].update(
                {
                    "status": "complete" if valid else "failed",
                    "returncode": completed.returncode,
                    "log": str(log_path),
                    "finished_at": time.time(),
                }
            )
            self._save_state()
            if valid:
                return artifact
            if local_attempt > self.max_retries:
                raise RuntimeError(
                    f"task {task_id} failed; inspect {log_path}"
                )
        raise AssertionError("unreachable")

    def pytest_smoke(self) -> None:
        task_id = f"v7_{self.input_hash[:8]}_smoke_pytest"
        marker = self.suite_dir / "smoke_pytest"

        def command(_: str) -> list[str]:
            return [
                sys.executable,
                "-m",
                "pytest",
                "test/test_v7_experiments.py",
                "test/test_v7_policy.py",
                "-q",
            ]

        if self.dry_run:
            print("[run]", " ".join(command(task_id)))
            return
        previous = self.state["tasks"].get(task_id, {})
        if previous.get("status") == "complete" and marker.exists():
            print(f"[skip] {task_id}")
            return
        marker.mkdir(parents=True, exist_ok=True)
        log = marker / "pytest.log"
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command(task_id), cwd=ROOT, stdout=handle,
                stderr=subprocess.STDOUT, text=True, check=False
            )
        atomic_json(marker / "result.json", {"returncode": result.returncode})
        self.state["tasks"][task_id] = {
            "status": "complete" if result.returncode == 0 else "failed",
            "artifact": str(marker),
            "input_hash": self.input_hash,
            "attempts": 1,
            "log": str(log),
        }
        self._save_state()
        if result.returncode:
            raise RuntimeError(f"v7 pytest smoke failed; inspect {log}")

    def train(self, phase: str, arm: str, seed: int, episodes: int) -> Path:
        task_id = f"v7_{self.input_hash[:8]}_{phase}_{arm}_seed{seed}"

        def artifact(run_name: str) -> Path:
            return ROOT / "result" / "runs" / run_name

        def command(run_name: str) -> list[str]:
            values = [
                sys.executable,
                "train.py",
                "--config",
                ARMS[arm],
                "--algorithm-seed",
                str(seed),
                "--parallel-envs",
                str(self.parallel_envs),
                "--no-visdom",
                "--run-name",
                run_name,
            ]
            if phase == "smoke":
                values.append("--smoke")
            else:
                values.extend(("--episodes", str(episodes)))
            return values

        return self._execute(
            task_id, command, artifact_builder=artifact, required="summary.json"
        )

    def evaluate(
        self,
        *,
        namespace: str,
        arm: str,
        algorithm_seed: int,
        config_path: str,
        checkpoint: Path,
        split: str,
        decode_mode: str,
        sampling_seed: int | None,
    ) -> Path:
        suffix = (
            f"sample{sampling_seed}" if sampling_seed is not None else "greedy"
        )
        task_id = (
            f"v7_{self.input_hash[:8]}_{namespace}_{arm}_seed"
            f"{algorithm_seed}_{split}_{suffix}"
        )

        def artifact(run_name: str) -> Path:
            return ROOT / "result" / "runs" / run_name

        def command(run_name: str) -> list[str]:
            values = [
                sys.executable,
                "eval.py",
                "--config",
                config_path,
                "--policy",
                "ppo",
                "--checkpoint",
                str(checkpoint),
                "--algorithm-seed",
                str(algorithm_seed),
                "--dataset",
                split,
                "--decode-mode",
                decode_mode,
                "--run-name",
                run_name,
            ]
            if sampling_seed is not None:
                values.extend(("--sampling-seed", str(sampling_seed)))
            return values

        return self._execute(
            task_id, command, artifact_builder=artifact, required="metrics.json"
        )

    def smoke(self) -> None:
        self.pytest_smoke()
        for arm in ARMS:
            self.train("smoke", arm, 11, 2)

    def screen(self) -> dict[tuple[str, int], Path]:
        return {
            (arm, seed): self.train("screen", arm, seed, 600)
            for arm in ARMS
            for seed in SCREEN_SEEDS
        }

    def screen_gate(self, runs: dict[tuple[str, int], Path]) -> None:
        failures: list[str] = []
        for key, run in runs.items():
            summary_path = run / "summary.json"
            if self.dry_run:
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            last = summary.get("last_episode", {})
            if summary.get("formal_training_status") != "quality_constrained":
                failures.append(f"{key}: feasibility gate not reached")
            if not (run / "accepted_checkpoint.pt").exists():
                failures.append(f"{key}: accepted checkpoint missing")
            if last.get("schedule_violation_count", 0):
                failures.append(f"{key}: schedule violation")
            sampled = summary.get("final_accepted_sampled_validation") or {}
            if sampled and not bool(sampled.get("fatigue_safe_line_pass", False)):
                failures.append(f"{key}: sampled fatigue safety violation")
            numeric = [
                value
                for value in last.values()
                if isinstance(value, (float, int)) and not isinstance(value, bool)
            ]
            if any(not math.isfinite(float(value)) for value in numeric):
                failures.append(f"{key}: non-finite summary value")
        report = {
            "passed": not failures,
            "failures": failures,
            "run_count": len(runs),
            "expected_run_count": 21,
        }
        if not self.dry_run:
            atomic_json(self.suite_dir / "screen_gate.json", report)
        if failures:
            raise RuntimeError("screen safety gate failed: " + "; ".join(failures))

    def formal(self) -> dict[tuple[str, int], Path]:
        return {
            (arm, seed): self.train("formal", arm, seed, 2000)
            for arm in ARMS
            for seed in FORMAL_SEEDS
        }

    def _historical_v6_checkpoints(self) -> list[tuple[int, Path, Path]]:
        discovered_by_seed: dict[int, tuple[int, Path, Path]] = {}
        pattern = ROOT / "result" / "runs"
        for checkpoint in pattern.glob(
            "policy_head_v6_seed*/accepted_checkpoint.pt"
        ):
            match = re.fullmatch(
                r"policy_head_v6_seed(\d+)", checkpoint.parent.name
            )
            if match is None:
                continue
            seed = int(match.group(1))
            if seed not in FORMAL_SEEDS:
                continue
            run = checkpoint.parent
            config = run / "config.json"
            if not config.exists():
                raise FileNotFoundError(
                    f"missing historical v6 checkpoint/config for seed {seed}"
                )
            discovered_by_seed[seed] = (seed, checkpoint, config)
        missing = sorted(set(FORMAL_SEEDS) - set(discovered_by_seed))
        if missing:
            raise FileNotFoundError(
                f"missing historical v6 checkpoints for seeds {missing}"
            )
        return [discovered_by_seed[seed] for seed in FORMAL_SEEDS]

    def audit(self) -> list[Path]:
        records: list[dict[str, Any]] = []
        outputs: list[Path] = []
        for seed, checkpoint, config in self._historical_v6_checkpoints():
            spec = read_checkpoint_network_spec(checkpoint)
            if int(spec.get("policy_head_version", -1)) != 6:
                raise ValueError(f"seed {seed} is not policy-head v6")
            if int(spec.get("observation_schema_version", -1)) != 3:
                raise ValueError(f"seed {seed} has the wrong v6 schema")
            actual_hash = sha256_file(checkpoint)
            summary_path = checkpoint.parent / "summary.json"
            recorded_hash = None
            if summary_path.exists():
                recorded_hash = json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).get("checkpoint_sha256")
            if recorded_hash is not None and recorded_hash != actual_hash:
                raise ValueError(f"seed {seed} checkpoint hash mismatch")
            records.append(
                {
                    "algorithm_seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": actual_hash,
                    "recorded_sha256": recorded_hash,
                    "network_spec": spec,
                }
            )
            for split in EVALUATION_SPLITS:
                outputs.append(
                    self.evaluate(
                        namespace="e0",
                        arm="v6",
                        algorithm_seed=seed,
                        config_path=str(config),
                        checkpoint=checkpoint,
                        split=split,
                        decode_mode="greedy",
                        sampling_seed=None,
                    )
                )
                for sampling_seed in SAMPLING_SEEDS:
                    outputs.append(
                        self.evaluate(
                            namespace="e0",
                            arm="v6",
                            algorithm_seed=seed,
                            config_path=str(config),
                            checkpoint=checkpoint,
                            split=split,
                            decode_mode="sampled",
                            sampling_seed=sampling_seed,
                        )
                    )
        if not self.dry_run:
            atomic_json(self.suite_dir / "e0_checkpoint_audit.json", records)
            aggregate_e0_results(self.suite_dir, outputs)
        return outputs

    def formal_evaluations(
        self, runs: dict[tuple[str, int], Path]
    ) -> list[Path]:
        outputs: list[Path] = []
        for (arm, seed), run in runs.items():
            checkpoint = run / "accepted_checkpoint.pt"
            if not self.dry_run and not checkpoint.exists():
                raise FileNotFoundError(
                    f"formal accepted checkpoint is missing: {checkpoint}"
                )
            for split in EVALUATION_SPLITS:
                outputs.append(
                    self.evaluate(
                        namespace="formal_eval",
                        arm=arm,
                        algorithm_seed=seed,
                        config_path=ARMS[arm],
                        checkpoint=checkpoint,
                        split=split,
                        decode_mode="greedy",
                        sampling_seed=None,
                    )
                )
                for sampling_seed in SAMPLING_SEEDS:
                    outputs.append(
                        self.evaluate(
                            namespace="formal_eval",
                            arm=arm,
                            algorithm_seed=seed,
                            config_path=ARMS[arm],
                            checkpoint=checkpoint,
                            split=split,
                            decode_mode="sampled",
                            sampling_seed=sampling_seed,
                        )
                    )
        return outputs


def exact_wilcoxon_two_sided(differences: Iterable[float]) -> dict[str, Any]:
    nonzero = [float(value) for value in differences if abs(float(value)) > 1e-15]
    if not nonzero:
        return {"n": 0, "statistic": 0.0, "p_value": 1.0}
    ordered = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(nonzero)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and math.isclose(
                abs(ordered[end][1]), abs(ordered[start][1]),
                rel_tol=0.0, abs_tol=1e-15
            )
        ):
            end += 1
        rank = (start + 1 + end) / 2.0
        for original, _ in ordered[start:end]:
            ranks[original] = rank
        start = end
    observed = abs(sum(
        rank if value > 0 else -rank
        for rank, value in zip(ranks, nonzero)
    ))
    outcomes = [
        abs(sum(sign * rank for sign, rank in zip(signs, ranks)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(ranks))
    ]
    p_value = sum(value >= observed - 1e-12 for value in outcomes) / len(outcomes)
    return {"n": len(nonzero), "statistic": observed, "p_value": p_value}


def _metric_value(metrics: dict, name: str) -> float:
    if name == "completion_rate":
        return float(metrics[name])
    tail_mapping = {
        "worker_load_variance_p90": ("worker_load_variance", "quantile"),
        "worker_load_variance_cvar90": ("worker_load_variance", "cvar"),
        "maximum_worker_fatigue_p90": ("maximum_worker_fatigue", "quantile"),
        "maximum_worker_fatigue_cvar90": ("maximum_worker_fatigue", "cvar"),
        "forced_action_chain_p95": ("forced_action_chain", "quantile"),
        "forced_action_chain_max": ("forced_action_chain", "max"),
    }
    if name in tail_mapping:
        metric, statistic = tail_mapping[name]
        value = metrics["tail_metrics"][metric][statistic]
        return float(value) if value is not None else float("nan")
    value = metrics["all_instance_metrics"][name]["mean"]
    return float(value) if value is not None else float("nan")


def aggregate_formal_results(
    suite_dir: Path, evaluation_dirs: list[Path]
) -> dict[str, Any]:
    metric_names = (
        "completion_rate",
        "quality_score",
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
        "worker_load_variance_p90",
        "worker_load_variance_cvar90",
        "maximum_worker_fatigue",
        "maximum_worker_fatigue_p90",
        "maximum_worker_fatigue_cvar90",
        "ranker_top_selection_rate",
        "context_override_rate",
        "mean_commit_set_logit",
        "reconfiguration_reuse_count",
        "qualification_scarcity_regret",
        "longest_forced_action_chain",
        "forced_action_chain_p95",
        "forced_action_chain_max",
    )
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for directory in evaluation_dirs:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        name = directory.name
        tokens = name.split("_")
        try:
            arm = tokens[tokens.index("eval") + 1]
            seed_token = next(value for value in tokens if value.startswith("seed"))
            seed = int(seed_token[4:])
            split = next(value for value in tokens if value in EVALUATION_SPLITS)
        except (ValueError, StopIteration) as error:
            raise ValueError(f"cannot parse evaluation run name {name}") from error
        mode = str(metrics.get("decode_mode", "greedy"))
        grouped[(arm, seed, split, mode)].append(metrics)
    per_seed: list[dict[str, Any]] = []
    for (arm, seed, split, mode), metrics_list in sorted(grouped.items()):
        row: dict[str, Any] = {
            "arm": arm, "algorithm_seed": seed,
            "split": split, "decode_mode": mode,
            "repeat_count": len(metrics_list),
        }
        for metric in metric_names:
            values = [_metric_value(value, metric) for value in metrics_list]
            finite = [value for value in values if math.isfinite(value)]
            row[metric] = sum(finite) / len(finite) if finite else None
        per_seed.append(row)
    comparisons: list[dict[str, Any]] = []
    index = {
        (row["arm"], row["algorithm_seed"], row["split"], row["decode_mode"]): row
        for row in per_seed
    }
    for arm in ARMS:
        if arm == "c0":
            continue
        for split in EVALUATION_SPLITS:
            for mode in ("greedy", "sampled"):
                for metric in metric_names:
                    differences = []
                    for seed in FORMAL_SEEDS:
                        treatment = index[(arm, seed, split, mode)].get(metric)
                        control = index[("c0", seed, split, mode)].get(metric)
                        if treatment is not None and control is not None:
                            differences.append(float(treatment) - float(control))
                    test = exact_wilcoxon_two_sided(differences)
                    comparisons.append(
                        {
                            "arm": arm, "split": split, "decode_mode": mode,
                            "metric": metric,
                            "mean_treatment_minus_c0": (
                                sum(differences) / len(differences)
                                if differences else None
                            ),
                            **test,
                            "note": (
                                "algorithm seed is the independent unit; with five "
                                "nonzero pairs the minimum two-sided exact p-value is 0.0625"
                            ),
                        }
                    )
    result = {
        "protocol": PROTOCOL,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "formal_seed_count": len(FORMAL_SEEDS),
        "per_seed": per_seed,
        "paired_exact_wilcoxon": comparisons,
    }
    atomic_json(suite_dir / "formal_aggregate.json", result)
    _write_csv(suite_dir / "formal_per_seed.csv", per_seed)
    _write_csv(suite_dir / "formal_wilcoxon.csv", comparisons)
    _write_report_and_plot(suite_dir, result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_e0_results(suite_dir: Path, evaluation_dirs: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for directory in evaluation_dirs:
        metrics = json.loads(
            (directory / "metrics.json").read_text(encoding="utf-8")
        )
        tokens = directory.name.split("_")
        seed_token = next(value for value in tokens if value.startswith("seed"))
        split = next(value for value in tokens if value in EVALUATION_SPLITS)
        rows.append(
            {
                "algorithm_seed": int(seed_token[4:]),
                "split": split,
                "decode_mode": metrics.get("decode_mode", "greedy"),
                "completion_rate": metrics["completion_rate"],
                "quality_score": _metric_value(metrics, "quality_score"),
                "worker_load_variance_p90": _metric_value(
                    metrics, "worker_load_variance_p90"
                ),
                "maximum_worker_fatigue_cvar90": _metric_value(
                    metrics, "maximum_worker_fatigue_cvar90"
                ),
                "schedule_violation_count": metrics[
                    "schedule_violation_count"
                ],
                "instance_count": metrics["instance_count"],
                "run_directory": str(directory),
            }
        )
    result = {
        "protocol": PROTOCOL,
        "audit": "historical_policy_head_v6",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "rows": rows,
    }
    atomic_json(suite_dir / "e0_aggregate.json", result)
    _write_csv(suite_dir / "e0_per_seed.csv", rows)
    greedy = [
        row for row in rows
        if row["decode_mode"] == "greedy" and row["split"] == "test"
    ]
    maximum = max((float(row["quality_score"]) for row in greedy), default=1.0)
    bars = []
    for index, row in enumerate(greedy):
        x = 65 + index * 125
        height = 230 * float(row["quality_score"]) / maximum if maximum else 0
        bars.append(
            f'<rect x="{x}" y="{290-height:.2f}" width="70" height="{height:.2f}" fill="#527d3c"/>'
            f'<text x="{x+35}" y="315" text-anchor="middle">seed {row["algorithm_seed"]}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="380" y="24" text-anchor="middle" font-size="16">E0 historical v6: test greedy quality</text>'
        '<line x1="40" y1="290" x2="730" y2="290" stroke="black"/>'
        + "".join(bars) + "</svg>"
    )
    (suite_dir / "e0_comparison.svg").write_text(svg, encoding="utf-8")
    report = [
        "# E0 historical v6 checkpoint audit",
        "",
        "Five local `accepted_checkpoint.pt` files were checked for policy-head v6, observation schema 3, and recorded SHA-256 consistency, then evaluated on fixed test/OOD/stress manifests.",
        "",
        "![E0 comparison](e0_comparison.svg)",
        "",
        f"Evaluation run count: {len(rows)}.",
    ]
    (suite_dir / "E0_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def _write_report_and_plot(suite_dir: Path, result: dict[str, Any]) -> None:
    rows = [
        row for row in result["per_seed"]
        if row["split"] == "test" and row["decode_mode"] == "greedy"
    ]
    means = {
        arm: sum(float(row["quality_score"]) for row in rows if row["arm"] == arm) / 5
        for arm in ARMS
    }
    width, height = 760, 360
    maximum = max(means.values()) if means else 1.0
    bars = []
    for index, (arm, value) in enumerate(means.items()):
        x = 55 + index * 95
        bar_height = 250 * value / maximum if maximum else 0
        y = 300 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="55" height="{bar_height:.2f}" fill="#3569a8"/>'
            f'<text x="{x + 27.5}" y="325" text-anchor="middle">{arm.upper()}</text>'
            f'<text x="{x + 27.5}" y="{max(15, y - 6):.2f}" text-anchor="middle" font-size="11">{value:.4f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="380" y="24" text-anchor="middle" font-size="16">Test greedy normalized quality (lower is better)</text>'
        '<line x1="40" y1="300" x2="730" y2="300" stroke="black"/>'
        + "".join(bars) + "</svg>"
    )
    (suite_dir / "formal_comparison.svg").write_text(svg, encoding="utf-8")
    report = [
        f"# {PROTOCOL} formal report",
        "",
        "All formal comparisons use algorithm seed as the independent unit. ",
        "Treatment−C0 is paired within each of seeds 11/23/37/53/71 and tested ",
        "with an exact two-sided Wilcoxon signed-rank test. With five nonzero ",
        "pairs, the smallest attainable two-sided p-value is 0.0625. Instance-level ",
        "tests are exploratory only.",
        "",
        "![Formal comparison](formal_comparison.svg)",
        "",
        "| arm | test greedy quality |",
        "|---|---:|",
    ]
    report.extend(f"| {arm.upper()} | {value:.6f} |" for arm, value in means.items())
    (suite_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(stage: str, *, parallel_envs: int, dry_run: bool, max_retries: int) -> Path:
    validate_arm_configs()
    orchestrator = Orchestrator(
        parallel_envs=parallel_envs,
        dry_run=dry_run,
        max_retries=max_retries,
    )
    normalized = stage.strip().lower()
    if normalized not in {"smoke", "screen", "formal", "audit", "all"}:
        raise ValueError("stage must be Smoke, Screen, Formal, Audit, or All")
    if normalized in {"smoke", "all"}:
        orchestrator.smoke()
    if normalized in {"audit", "all"}:
        orchestrator.audit()
    if normalized in {"screen", "all"}:
        screen_runs = orchestrator.screen()
        orchestrator.screen_gate(screen_runs)
    if normalized in {"formal", "all"}:
        formal_runs = orchestrator.formal()
        evaluations = orchestrator.formal_evaluations(formal_runs)
        if not dry_run:
            aggregate_formal_results(orchestrator.suite_dir, evaluations)
    return orchestrator.suite_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable v7 E0-E5 experiment protocol"
    )
    parser.add_argument(
        "--stage", default="All",
        choices=("Smoke", "Screen", "Formal", "Audit", "All"),
    )
    parser.add_argument("--parallel-envs", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = run(
        args.stage,
        parallel_envs=args.parallel_envs,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )
    print(f"v7 suite: {suite}")


if __name__ == "__main__":
    main()
