from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon

from configs import load_config, project_path
from environment import RewardVector
from result import aggregate_evaluation_rows
from result.io import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "result/matched_credit_assignment_currentcode_20260809"
)
SEEDS = (11, 23, 37, 53, 71)
SAMPLING_SEEDS = (100011, 100012, 100013)
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPEATS = 10_000
COMMON_INTERVAL = 20
SOURCE_SCHEMA_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = "1.0.0"

ARMS: dict[str, dict[str, Any]] = {
    "baseline": {
        "config": Path("configs/credit_assignment_matched_baseline.json"),
        "gae_lambda": 0.95,
        "forced_action_compression": False,
        "parallel_envs": 10,
        "validation_interval_episodes": 10,
        "smoke_episodes": 10,
    },
    "treatment": {
        "config": Path("configs/credit_assignment_matched_treatment.json"),
        "gae_lambda": 0.995,
        "forced_action_compression": True,
        "parallel_envs": 20,
        "validation_interval_episodes": 20,
        "smoke_episodes": 20,
    },
}

ALLOWED_ARM_DIFFERENCES = {
    "experiment_name",
    "ppo.gae_lambda",
    "training.forced_action_compression",
    "training.parallel_envs",
    "training.smoke_episodes",
    "training.smoke_parallel_envs",
    "training.validation_interval_episodes",
    "training.validation_parallel_envs",
}

PPO_DIAGNOSTICS = (
    "advantage_std",
    "approx_kl",
    "clip_fraction",
    "entropy",
    "value_loss",
)
VALIDATION_METRICS = (
    "completion_rate",
    "truncated_count",
    "mean_flow_time_objective",
    "mean_reconfiguration_cost",
    "mean_worker_load_variance",
    "mean_quality_score",
)
EVALUATION_METRICS = (
    "completion_rate",
    "truncated_count",
    "schedule_violation_count",
    "quality_score",
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
    "relative_heuristic_gap_percent",
)
TRAINING_METRICS = (
    "environment_steps",
    "policy_transitions",
    "forced_actions",
    "forced_action_ratio",
    "mean_policy_steps_per_episode",
    "sampling_seconds",
    "policy_inference_seconds",
    "ppo_update_seconds",
    "wall_clock_seconds",
    *PPO_DIAGNOSTICS,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _number(row.get(field))) is not None
    ]
    return float(statistics.fmean(values)) if values else None


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    observations = [float(value) for value in values]
    if not observations:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
        }
    return {
        "count": len(observations),
        "mean": float(statistics.fmean(observations)),
        "std": (
            float(statistics.stdev(observations))
            if len(observations) > 1
            else 0.0
        ),
        "median": float(statistics.median(observations)),
    }


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    leaves: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        leaves.update(_leaf_values(value[key], path))
    return leaves


def load_matched_configs() -> dict[str, dict[str, Any]]:
    return {
        arm: load_config(PROJECT_ROOT / spec["config"])
        for arm, spec in ARMS.items()
    }


def validate_matched_configs(
    configs: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    configs = load_matched_configs() if configs is None else configs
    if set(configs) != set(ARMS):
        raise ValueError("matched configs must define baseline and treatment")
    leaves = {
        arm: _leaf_values(
            {
                key: value
                for key, value in config.items()
                if key != "_config_path"
            }
        )
        for arm, config in configs.items()
    }
    paths = set(leaves["baseline"]) | set(leaves["treatment"])
    differences = {
        path
        for path in paths
        if leaves["baseline"].get(path) != leaves["treatment"].get(path)
    }
    if differences != ALLOWED_ARM_DIFFERENCES:
        missing = sorted(ALLOWED_ARM_DIFFERENCES - differences)
        unexpected = sorted(differences - ALLOWED_ARM_DIFFERENCES)
        raise ValueError(
            "matched config difference whitelist failed; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for arm, spec in ARMS.items():
        config = configs[arm]
        training = config["training"]
        if int(training["episodes"]) != 1000:
            raise ValueError(f"{arm} must train for 1000 episodes")
        if tuple(int(seed) for seed in config["algorithm_seeds"]) != SEEDS:
            raise ValueError(f"{arm} algorithm seeds must be {list(SEEDS)}")
        if float(config["ppo"]["gamma"]) != 1.0:
            raise ValueError(f"{arm} must use gamma=1.0")
        expected = {
            "gae_lambda": float(config["ppo"]["gae_lambda"]),
            "forced_action_compression": bool(
                training["forced_action_compression"]
            ),
            "parallel_envs": int(training["parallel_envs"]),
            "validation_interval_episodes": int(
                training["validation_interval_episodes"]
            ),
            "smoke_episodes": int(training["smoke_episodes"]),
        }
        for field, actual in expected.items():
            if actual != spec[field]:
                raise ValueError(
                    f"{arm} {field}={actual!r}, expected {spec[field]!r}"
                )
        if int(training["smoke_parallel_envs"]) != spec["parallel_envs"]:
            raise ValueError(f"{arm} smoke parallelism must match its batch")
        if int(training["validation_parallel_envs"]) != spec["parallel_envs"]:
            raise ValueError(
                f"{arm} validation parallelism must match its batch"
            )
    return differences


def _source_files() -> list[Path]:
    allowed_suffixes = {".py", ".json", ".yaml", ".yml", ".txt"}
    paths: set[Path] = set()
    for path in PROJECT_ROOT.glob("*.py"):
        paths.add(path)
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        paths.add(requirements)
    for directory_name in (
        "agent",
        "configs",
        "data",
        "environment",
        "utils",
        "数据",
    ):
        directory = PROJECT_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in allowed_suffixes
                and "__pycache__" not in path.parts
            ):
                paths.add(path)
    result_source = PROJECT_ROOT / "result"
    if result_source.exists():
        for path in result_source.glob("*.py"):
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def source_snapshot() -> dict[str, Any]:
    digest = hashlib.sha256()
    files = []
    for path in _source_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        content = path.read_bytes()
        file_sha = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        files.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": file_sha,
            }
        )
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_sha256": digest.hexdigest(),
        "files": files,
    }


def _git_output(*arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def _environment_versions() -> dict[str, Any]:
    import scipy
    import torch

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


def _archive_source(path: Path, snapshot: dict[str, Any]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in snapshot["files"]:
            relative = str(entry["path"])
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (PROJECT_ROOT / relative).read_bytes())


def initialize_experiment(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        verify_source(manifest)
        validate_matched_configs()
        return manifest
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"experiment root exists without manifest and is not empty: {root}"
        )
    validate_matched_configs()
    snapshot = source_snapshot()
    git_head = str(_git_output("rev-parse", "HEAD")).strip()
    git_status = str(_git_output("status", "--short"))
    git_diff = _git_output("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".", binary=True)
    assert isinstance(git_diff, bytes)
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "study": "matched_current_code_credit_assignment",
        "study_design": {
            "arms": {
                arm: {
                    **{
                        key: value
                        for key, value in spec.items()
                        if key != "config"
                    },
                    "config": spec["config"].as_posix(),
                }
                for arm, spec in ARMS.items()
            },
            "algorithm_seeds": list(SEEDS),
            "episodes_per_run": 1000,
            "sampling_seeds": list(SAMPLING_SEEDS),
            "test_instance_count": 20,
            "common_validation_interval": COMMON_INTERVAL,
            "statistical_unit": "algorithm_seed",
            "primary_endpoint": "greedy_test_quality_score_after_feasibility",
        },
        "source_sha256": snapshot["source_sha256"],
        "source_files": snapshot["files"],
        "git": {
            "head": git_head,
            "dirty": bool(git_status.strip()),
            "status": git_status.splitlines(),
            "workspace_patch_sha256": hashlib.sha256(git_diff).hexdigest(),
        },
        "environment": _environment_versions(),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repeats": BOOTSTRAP_REPEATS,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "provenance").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "evaluations").mkdir(exist_ok=True)
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(root / "provenance" / "source_manifest.json", snapshot)
    (root / "provenance" / "git_head.txt").write_text(
        git_head + "\n", encoding="utf-8"
    )
    (root / "provenance" / "git_status.txt").write_text(
        git_status, encoding="utf-8"
    )
    (root / "provenance" / "workspace.patch").write_bytes(git_diff)
    _write_json_atomic(
        root / "provenance" / "environment.json",
        manifest["environment"],
    )
    _archive_source(root / "provenance" / "source_snapshot.zip", snapshot)
    for arm, config in load_matched_configs().items():
        public = {key: value for key, value in config.items() if key != "_config_path"}
        _write_json_atomic(root / "provenance" / f"{arm}_config.json", public)
    return manifest


def verify_source(manifest: dict[str, Any]) -> str:
    actual = source_snapshot()["source_sha256"]
    expected = str(manifest["source_sha256"])
    if actual != expected:
        raise RuntimeError(
            "source snapshot changed after experiment freeze: "
            f"expected {expected}, observed {actual}; use a new result root"
        )
    return actual


def training_command(
    arm: str,
    seed: int,
    *,
    run_name: str,
    smoke: bool,
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed {seed} is not in {list(SEEDS)}")
    spec = ARMS[arm]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(PROJECT_ROOT / spec["config"]),
    ]
    if smoke:
        command.append("--smoke")
    else:
        command.extend(["--episodes", "1000"])
    command.extend(
        [
            "--algorithm-seed",
            str(seed),
            "--parallel-envs",
            str(spec["parallel_envs"]),
            "--run-name",
            run_name,
        ]
    )
    return command


def planned_runs(
    arms: Sequence[str], seeds: Sequence[int], *, smoke: bool
) -> list[tuple[str, int]]:
    effective_seeds = tuple(seeds)
    if smoke and effective_seeds == SEEDS:
        effective_seeds = (11,)
    return [
        (arm, int(seed))
        for arm in arms
        for seed in effective_seeds
    ]


def next_available_path(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    retry = 0
    while candidate.exists():
        retry += 1
        candidate = root / f"{base_name}_retry{retry}"
    return candidate


def _run_directory_root(arm: str) -> Path:
    config = load_config(PROJECT_ROOT / ARMS[arm]["config"])
    return project_path(config["paths"]["result_root"])


def allocate_run_name(arm: str, seed: int, *, smoke: bool) -> tuple[str, Path]:
    prefix = "matched_credit_smoke" if smoke else "matched_credit"
    suffix = "" if smoke else "_1000"
    base = f"{prefix}_{arm}_seed{seed}{suffix}_20260809"
    root = _run_directory_root(arm)
    path = next_available_path(root, base)
    return path.name, path


def _state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    if path.exists():
        return _read_json(path)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "runs": {},
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _write_json_atomic(root / "state.json", state)


def _reward_component_error(
    row: dict[str, Any], reward_config: dict[str, Any]
) -> float:
    vector = RewardVector(
        flow=float(row.get("reward_flow", 0.0)),
        cost=float(row.get("reward_cost", 0.0)),
        variance=float(row.get("reward_variance", 0.0)),
        completion_progress=float(
            row.get("reward_completion_progress", 0.0)
        ),
        completion_bonus=float(row.get("reward_completion_bonus", 0.0)),
        quality=float(row.get("reward_quality", 0.0)),
        truncation=float(row.get("reward_truncation", 0.0)),
        unfinished=float(row.get("reward_unfinished", 0.0)),
        feasibility_shaping=float(
            row.get("reward_feasibility_shaping", 0.0)
        ),
    )
    reconstructed = vector.scalarize(
        reward_config,
        str(row.get("reward_phase", "legacy")),
    )
    observed = float(row.get("reward_training", row.get("reward", 0.0)))
    return observed - reconstructed


def audit_run(
    run_directory: Path,
    *,
    arm: str,
    seed: int,
    smoke: bool,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    errors: list[str] = []
    required = (
        "config.json",
        "summary.json",
        "train_log.csv",
        "update_log.csv",
        "validation_log.csv",
    )
    missing = [name for name in required if not (run_directory / name).exists()]
    if missing:
        return {
            "arm": arm,
            "seed": seed,
            "smoke": smoke,
            "run_directory": str(run_directory),
            "valid": False,
            "errors": [f"missing artifacts: {missing}"],
        }
    try:
        config = _read_json(run_directory / "config.json")
        summary = _read_json(run_directory / "summary.json")
        train_rows = _read_csv(run_directory / "train_log.csv")
        update_rows = _read_csv(run_directory / "update_log.csv")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "arm": arm,
            "seed": seed,
            "smoke": smoke,
            "run_directory": str(run_directory),
            "valid": False,
            "errors": [f"artifact parse failed: {error}"],
        }
    spec = ARMS[arm]
    expected_episodes = (
        int(spec["smoke_episodes"]) if smoke else 1000
    )
    expected_batch = int(spec["parallel_envs"])
    expected_updates = math.ceil(expected_episodes / expected_batch)
    if int(config.get("seed", -1)) != seed:
        errors.append(f"config seed is {config.get('seed')}, expected {seed}")
    if float(config.get("ppo", {}).get("gae_lambda", math.nan)) != float(
        spec["gae_lambda"]
    ):
        errors.append("GAE lambda does not match the arm")
    if bool(
        config.get("training", {}).get("forced_action_compression")
    ) != bool(spec["forced_action_compression"]):
        errors.append("forced-action compression does not match the arm")
    if int(config.get("training", {}).get("parallel_envs", -1)) != expected_batch:
        errors.append("parallel_envs does not match the arm")
    if len(train_rows) != expected_episodes:
        errors.append(
            f"episode rows={len(train_rows)}, expected {expected_episodes}"
        )
    if len(update_rows) != expected_updates:
        errors.append(f"update rows={len(update_rows)}, expected {expected_updates}")
    if int(summary.get("episodes", -1)) != expected_episodes:
        errors.append("summary episode count is incorrect")
    if int(summary.get("updates", -1)) != expected_updates:
        errors.append("summary update count is incorrect")
    for index, row in enumerate(update_rows):
        expected_count = min(
            expected_batch,
            expected_episodes - index * expected_batch,
        )
        if int(float(row.get("episode_count", -1))) != expected_count:
            errors.append(f"update {index + 1} episode_count is incorrect")
        for field in PPO_DIAGNOSTICS:
            if _number(row.get(field)) is None:
                errors.append(f"update {index + 1} has non-finite {field}")
    environment_steps = 0
    policy_steps = 0
    forced_actions = 0
    reward_errors: list[float] = []
    component_errors: list[float] = []
    reward_config = config.get("reward", {})
    for index, row in enumerate(train_rows):
        try:
            steps = int(float(row["steps"]))
            policy = int(float(row["policy_steps"]))
            forced = int(float(row["forced_actions"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"episode row {index} has invalid step accounting")
            continue
        if steps != policy + forced:
            errors.append(f"episode row {index} violates step identity")
        environment_steps += steps
        policy_steps += policy
        forced_actions += forced
        identity = _number(row.get("reward_identity_error"))
        if identity is None:
            errors.append(f"episode row {index} has invalid reward identity")
        else:
            reward_errors.append(abs(identity))
        try:
            component_errors.append(
                abs(_reward_component_error(row, reward_config))
            )
        except (TypeError, ValueError, KeyError) as error:
            errors.append(
                f"episode row {index} component reconstruction failed: {error}"
            )
    maximum_reward_error = max(reward_errors, default=0.0)
    maximum_component_error = max(component_errors, default=0.0)
    if maximum_reward_error > 1e-8:
        errors.append(f"reward identity error exceeds tolerance: {maximum_reward_error}")
    if maximum_component_error > 1e-8:
        errors.append(
            "reward component identity error exceeds tolerance: "
            f"{maximum_component_error}"
        )
    if int(summary.get("environment_steps", -1)) != environment_steps:
        errors.append("summary environment_steps does not match episode rows")
    if int(summary.get("transitions", -1)) != policy_steps:
        errors.append("summary transitions does not match policy steps")
    if int(summary.get("forced_actions", -1)) != forced_actions:
        errors.append("summary forced_actions does not match episode rows")
    if environment_steps != policy_steps + forced_actions:
        errors.append("run-level step identity failed")
    if not spec["forced_action_compression"] and forced_actions != 0:
        errors.append("baseline unexpectedly reports compressed forced actions")
    checkpoint_sha = None
    if not smoke:
        checkpoint_paths = [
            run_directory / "accepted_checkpoint.pt",
            run_directory / "checkpoint.pt",
            run_directory / "best_checkpoint.pt",
        ]
        if any(not path.exists() for path in checkpoint_paths):
            errors.append("official accepted/checkpoint/best files are incomplete")
        else:
            checkpoint_hashes = {_sha256(path) for path in checkpoint_paths}
            if len(checkpoint_hashes) != 1:
                errors.append("official checkpoint hashes diverge")
            else:
                checkpoint_sha = next(iter(checkpoint_hashes))
                if checkpoint_sha != summary.get("checkpoint_sha256"):
                    errors.append("summary checkpoint hash does not match files")
    if source_sha256 is not None:
        provenance_path = run_directory / "provenance.json"
        if not provenance_path.exists():
            errors.append("run provenance.json is missing")
        else:
            provenance = _read_json(provenance_path)
            if provenance.get("source_sha256") != source_sha256:
                errors.append("run source hash does not match experiment")
    return {
        "arm": arm,
        "seed": seed,
        "smoke": smoke,
        "run_directory": str(run_directory),
        "valid": not errors,
        "errors": errors,
        "episodes": len(train_rows),
        "updates": len(update_rows),
        "environment_steps": environment_steps,
        "policy_steps": policy_steps,
        "forced_actions": forced_actions,
        "forced_action_ratio": (
            forced_actions / environment_steps if environment_steps else 0.0
        ),
        "maximum_reward_identity_error": maximum_reward_error,
        "maximum_reward_component_error": maximum_component_error,
        "checkpoint_sha256": checkpoint_sha,
    }


def _stream_command(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _run_key(arm: str, seed: int, smoke: bool) -> str:
    prefix = "smoke" if smoke else "formal"
    return f"{prefix}:{arm}:seed{seed}"


def _valid_existing_entry(
    entry: dict[str, Any] | None,
    *,
    arm: str,
    seed: int,
    smoke: bool,
    source_sha256: str,
) -> dict[str, Any] | None:
    if not entry or not entry.get("run_directory"):
        return None
    audit = audit_run(
        Path(entry["run_directory"]),
        arm=arm,
        seed=seed,
        smoke=smoke,
        source_sha256=source_sha256,
    )
    return audit if audit["valid"] else None


def execute_training(
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    *,
    arm: str,
    seed: int,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_source(manifest)
    key = _run_key(arm, seed, smoke)
    previous = state["runs"].get(key)
    existing_audit = _valid_existing_entry(
        previous,
        arm=arm,
        seed=seed,
        smoke=smoke,
        source_sha256=str(manifest["source_sha256"]),
    )
    if existing_audit is not None:
        print(f"skip valid completed run {key}: {existing_audit['run_directory']}")
        previous["status"] = "complete"
        previous["audit"] = existing_audit
        _save_state(root, state)
        return previous, existing_audit
    run_name, run_directory = allocate_run_name(arm, seed, smoke=smoke)
    command = training_command(
        arm,
        seed,
        run_name=run_name,
        smoke=smoke,
    )
    attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
    entry = {
        "key": key,
        "arm": arm,
        "seed": seed,
        "smoke": smoke,
        "attempt": attempt,
        "status": "running",
        "started_at_utc": _utc_now(),
        "run_name": run_name,
        "run_directory": str(run_directory.resolve()),
        "command": command,
        "source_sha256": manifest["source_sha256"],
    }
    state["runs"][key] = entry
    _save_state(root, state)
    log_path = root / "logs" / f"{key.replace(':', '_')}_attempt{attempt}.log"
    started = time.perf_counter()
    try:
        exit_code = _stream_command(command, log_path)
    except BaseException:
        entry["status"] = "interrupted"
        entry["finished_at_utc"] = _utc_now()
        entry["wall_clock_seconds"] = time.perf_counter() - started
        _save_state(root, state)
        raise
    entry["finished_at_utc"] = _utc_now()
    entry["wall_clock_seconds"] = time.perf_counter() - started
    entry["exit_code"] = exit_code
    if exit_code != 0:
        entry["status"] = "failed"
        _save_state(root, state)
        raise RuntimeError(f"training failed for {key} with exit code {exit_code}")
    if not run_directory.exists():
        entry["status"] = "failed"
        _save_state(root, state)
        raise RuntimeError(f"training did not create {run_directory}")
    provenance = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_root": str(root.resolve()),
        "source_sha256": manifest["source_sha256"],
        "git_head": manifest["git"]["head"],
        "arm": arm,
        "seed": seed,
        "smoke": smoke,
        "command": command,
        "started_at_utc": entry["started_at_utc"],
        "finished_at_utc": entry["finished_at_utc"],
        "wall_clock_seconds": entry["wall_clock_seconds"],
    }
    _write_json_atomic(run_directory / "provenance.json", provenance)
    audit = audit_run(
        run_directory,
        arm=arm,
        seed=seed,
        smoke=smoke,
        source_sha256=str(manifest["source_sha256"]),
    )
    entry["audit"] = audit
    entry["status"] = "complete" if audit["valid"] else "invalid"
    _save_state(root, state)
    if not audit["valid"]:
        raise RuntimeError(f"audit failed for {key}: {audit['errors']}")
    return entry, audit


def _evaluation_directory(root: Path, arm: str, seed: int) -> Path:
    parent = root / "evaluations" / arm
    parent.mkdir(parents=True, exist_ok=True)
    return next_available_path(parent, f"seed_{seed}")


def _evaluation_is_valid(
    path: Path,
    *,
    source_sha256: str,
    checkpoint_sha256: str,
) -> bool:
    marker = path / "evaluation_manifest.json"
    greedy = path / "greedy" / "instance_metrics.csv"
    sampled = path / "sampled" / "instance_metrics.csv"
    if not marker.exists() or not greedy.exists() or not sampled.exists():
        return False
    try:
        payload = _read_json(marker)
        return (
            payload.get("source_sha256") == source_sha256
            and payload.get("checkpoint_sha256") == checkpoint_sha256
            and len(_read_csv(greedy)) == 20
            and len(_read_csv(sampled)) == 60
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def evaluate_checkpoint(
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    entry: dict[str, Any],
) -> Path:
    verify_source(manifest)
    arm = str(entry["arm"])
    seed = int(entry["seed"])
    run_directory = Path(entry["run_directory"])
    checkpoint = run_directory / "accepted_checkpoint.pt"
    checkpoint_sha = _sha256(checkpoint)
    existing = entry.get("evaluation_directory")
    if existing and _evaluation_is_valid(
        Path(existing),
        source_sha256=str(manifest["source_sha256"]),
        checkpoint_sha256=checkpoint_sha,
    ):
        print(f"skip valid evaluation {arm} seed {seed}: {existing}")
        return Path(existing)
    evaluation_directory = _evaluation_directory(root, arm, seed)
    evaluation_directory.mkdir(parents=True)
    entry["evaluation_directory"] = str(evaluation_directory.resolve())
    entry["evaluation_status"] = "running"
    _save_state(root, state)
    from agent.ppo import PPOAgent, build_actor_critic
    from agent.ppo.parallel import ParallelEpisodeRunner
    from data import load_instance_yaml
    from data.dataset import load_dataset_split
    from environment import AssemblySchedulingEnv
    from eval import evaluate_dataset_parallel

    config = _read_json(run_directory / "config.json")
    config["training"]["validation_parallel_envs"] = 20
    config["training"]["forced_action_compression"] = False
    import torch

    torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    dataset = load_dataset_split(config, "test")
    if len(dataset) != 20:
        raise RuntimeError(f"expected 20 fixed test instances, observed {len(dataset)}")
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    bootstrap = AssemblySchedulingEnv(config).reset(dataset[0].instance)
    network = build_actor_critic(bootstrap, config["network"])
    agent = PPOAgent(network, config["ppo"], device=config["device"])
    agent.load(checkpoint)
    started = time.perf_counter()
    with ParallelEpisodeRunner(
        config=config,
        template=template,
        episode_count=len(dataset),
        worker_count=20,
    ) as runner:
        verify_source(manifest)
        greedy_rows, greedy_metrics = evaluate_dataset_parallel(
            config,
            dataset_name="test",
            ppo_agent=agent,
            runner=runner,
            decode_mode="greedy",
        )
        greedy_metrics.update(
            {
                "algorithm_seed": seed,
                "arm": arm,
                "source_sha256": manifest["source_sha256"],
                "checkpoint_sha256": checkpoint_sha,
            }
        )
        greedy_dir = evaluation_directory / "greedy"
        greedy_dir.mkdir()
        write_csv(greedy_dir / "instance_metrics.csv", greedy_rows)
        write_json(greedy_dir / "metrics.json", greedy_metrics)
        sampled_rows: list[dict[str, Any]] = []
        for sampling_seed in SAMPLING_SEEDS:
            verify_source(manifest)
            rows, metrics = evaluate_dataset_parallel(
                config,
                dataset_name="test",
                ppo_agent=agent,
                runner=runner,
                decode_mode="sampled",
                sampling_seed=sampling_seed,
            )
            rows = [
                {"sampling_seed": sampling_seed, **row}
                for row in rows
            ]
            sampled_rows.extend(rows)
            repeat_dir = evaluation_directory / f"sampled_seed_{sampling_seed}"
            repeat_dir.mkdir()
            write_csv(repeat_dir / "instance_metrics.csv", rows)
            metrics.update(
                {
                    "algorithm_seed": seed,
                    "arm": arm,
                    "sampling_seed": sampling_seed,
                    "source_sha256": manifest["source_sha256"],
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
            write_json(repeat_dir / "metrics.json", metrics)
    sampled_metrics = aggregate_evaluation_rows(
        sampled_rows,
        dataset="test",
        policy="ppo",
        manifest=str(dataset.manifest_path),
    )
    sampled_metrics.update(
        {
            "decode_mode": "sampled",
            "parallel_envs": 20,
            "repeat_count": len(SAMPLING_SEEDS),
            "unique_instance_count": len(dataset),
            "sampling_seeds": list(SAMPLING_SEEDS),
            "algorithm_seed": seed,
            "arm": arm,
            "source_sha256": manifest["source_sha256"],
            "checkpoint_sha256": checkpoint_sha,
        }
    )
    sampled_dir = evaluation_directory / "sampled"
    sampled_dir.mkdir()
    write_csv(sampled_dir / "instance_metrics.csv", sampled_rows)
    write_json(sampled_dir / "metrics.json", sampled_metrics)
    marker = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "arm": arm,
        "algorithm_seed": seed,
        "source_sha256": manifest["source_sha256"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "test_instance_count": 20,
        "greedy_rows": len(greedy_rows),
        "sampled_rows": len(sampled_rows),
        "sampling_seeds": list(SAMPLING_SEEDS),
        "parallel_envs": 20,
        "wall_clock_seconds": time.perf_counter() - started,
        "forced_action_compression": False,
    }
    _write_json_atomic(evaluation_directory / "evaluation_manifest.json", marker)
    entry["evaluation_status"] = "complete"
    entry["evaluation_checkpoint_sha256"] = checkpoint_sha
    _save_state(root, state)
    return evaluation_directory


def _audit_rows(root: Path, manifest: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(state.get("runs", {})):
        entry = state["runs"][key]
        audit = audit_run(
            Path(entry["run_directory"]),
            arm=str(entry["arm"]),
            seed=int(entry["seed"]),
            smoke=bool(entry["smoke"]),
            source_sha256=str(manifest["source_sha256"]),
        )
        entry["audit"] = audit
        entry["status"] = "complete" if audit["valid"] else "invalid"
        rows.append(
            {
                **{key: value for key, value in audit.items() if key != "errors"},
                "errors": " | ".join(audit["errors"]),
            }
        )
    _save_state(root, state)
    write_csv(root / "run_audit.csv", rows)
    return rows


def _evaluation_metric(metrics: dict[str, Any], name: str) -> float:
    if name in {"completion_rate", "truncated_count", "schedule_violation_count"}:
        return float(metrics[name])
    if name == "relative_heuristic_gap_percent":
        return float(metrics["gap_metrics"][name]["mean"])
    return float(metrics["all_instance_metrics"][name]["mean"])


def _seed_metric_rows(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    formal_entries = [
        entry
        for entry in state["runs"].values()
        if not bool(entry["smoke"])
    ]
    for entry in sorted(
        formal_entries,
        key=lambda value: (str(value["arm"]), int(value["seed"])),
    ):
        run_directory = Path(entry["run_directory"])
        summary = _read_json(run_directory / "summary.json")
        update_rows = _read_csv(run_directory / "update_log.csv")
        training = {
            "environment_steps": float(summary["environment_steps"]),
            "policy_transitions": float(summary["transitions"]),
            "forced_actions": float(summary["forced_actions"]),
            "forced_action_ratio": float(summary["forced_action_ratio"]),
            "mean_policy_steps_per_episode": float(
                summary["mean_policy_steps_per_episode"]
            ),
            "sampling_seconds": float(summary["total_sampling_time_seconds"]),
            "policy_inference_seconds": float(
                summary["total_policy_inference_time_seconds"]
            ),
            "ppo_update_seconds": float(
                summary["total_ppo_update_time_seconds"]
            ),
            "wall_clock_seconds": float(entry.get("wall_clock_seconds", 0.0)),
            **{
                field: _mean(update_rows, field)
                for field in PPO_DIAGNOSTICS
            },
        }
        evaluation_directory = Path(entry["evaluation_directory"])
        for mode in ("greedy", "sampled"):
            metrics = _read_json(evaluation_directory / mode / "metrics.json")
            row = {
                "arm": entry["arm"],
                "algorithm_seed": int(entry["seed"]),
                "decode_mode": mode,
                "run_directory": str(run_directory),
                "evaluation_directory": str(evaluation_directory),
                "formal_training_status": summary.get("formal_training_status"),
                **training,
            }
            row.update(
                {
                    name: _evaluation_metric(metrics, name)
                    for name in EVALUATION_METRICS
                }
            )
            rows.append(row)
    return rows


def paired_statistics(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    bootstrap_key: str,
) -> dict[str, Any]:
    if len(baseline) != len(treatment):
        raise ValueError("paired samples must have the same length")
    if not baseline:
        raise ValueError("paired samples must not be empty")
    before = np.asarray(baseline, dtype=np.float64)
    after = np.asarray(treatment, dtype=np.float64)
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("paired samples must be finite")
    differences = after - before
    key_seed = int.from_bytes(
        hashlib.sha256(bootstrap_key.encode("utf-8")).digest()[:8],
        "little",
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED ^ key_seed)
    indices = generator.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPEATS, len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    if np.allclose(differences, 0.0, rtol=0.0, atol=0.0):
        statistic = 0.0
        p_value = 1.0
        method = "all_zero"
    else:
        result = wilcoxon(
            after,
            before,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        method = "scipy_auto"
    relative = np.asarray(
        [
            100.0 * delta / abs(base) if base != 0.0 else math.nan
            for base, delta in zip(before, differences)
        ],
        dtype=np.float64,
    )
    finite_relative = relative[np.isfinite(relative)]
    return {
        "baseline": _summary(before.tolist()),
        "treatment": _summary(after.tolist()),
        "paired_difference_treatment_minus_baseline": {
            **_summary(differences.tolist()),
            "bootstrap_95_ci": [
                float(np.percentile(bootstrap_means, 2.5)),
                float(np.percentile(bootstrap_means, 97.5)),
            ],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "paired_relative_percent": _summary(finite_relative.tolist()),
        "wilcoxon": {
            "statistic": statistic,
            "p_value_two_sided": p_value,
            "method": method,
            "zero_method": "wilcox",
        },
    }


def _paired_outputs(
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = {
        (str(row["arm"]), int(row["algorithm_seed"]), str(row["decode_mode"])): row
        for row in seed_rows
    }
    delta_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {"evaluation": {}, "training": {}}
    for mode in ("greedy", "sampled"):
        aggregate["evaluation"][mode] = {}
        for metric in EVALUATION_METRICS:
            baseline = [
                float(index[("baseline", seed, mode)][metric])
                for seed in SEEDS
            ]
            treatment = [
                float(index[("treatment", seed, mode)][metric])
                for seed in SEEDS
            ]
            aggregate["evaluation"][mode][metric] = paired_statistics(
                baseline,
                treatment,
                bootstrap_key=f"evaluation:{mode}:{metric}",
            )
            for seed, before, after in zip(SEEDS, baseline, treatment):
                delta = after - before
                delta_rows.append(
                    {
                        "category": "evaluation",
                        "decode_mode": mode,
                        "algorithm_seed": seed,
                        "metric": metric,
                        "baseline": before,
                        "treatment": after,
                        "absolute_delta": delta,
                        "relative_delta_percent": (
                            100.0 * delta / abs(before)
                            if before != 0.0
                            else None
                        ),
                    }
                )
    for metric in TRAINING_METRICS:
        baseline = [
            float(index[("baseline", seed, "greedy")][metric])
            for seed in SEEDS
        ]
        treatment = [
            float(index[("treatment", seed, "greedy")][metric])
            for seed in SEEDS
        ]
        aggregate["training"][metric] = paired_statistics(
            baseline,
            treatment,
            bootstrap_key=f"training:{metric}",
        )
        for seed, before, after in zip(SEEDS, baseline, treatment):
            delta = after - before
            delta_rows.append(
                {
                    "category": "training",
                    "decode_mode": "",
                    "algorithm_seed": seed,
                    "metric": metric,
                    "baseline": before,
                    "treatment": after,
                    "absolute_delta": delta,
                    "relative_delta_percent": (
                        100.0 * delta / abs(before)
                        if before != 0.0
                        else None
                    ),
                }
            )
    return delta_rows, aggregate


def _convergence_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    observations: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for entry in state["runs"].values():
        if bool(entry["smoke"]):
            continue
        rows = _read_csv(Path(entry["run_directory"]) / "validation_log.csv")
        for row in rows:
            episode = int(float(row["episode"]))
            if episode % COMMON_INTERVAL == 0:
                observations[(str(entry["arm"]), episode)].append(row)
    output = []
    for (arm, episode), rows in sorted(observations.items()):
        if len(rows) != len(SEEDS):
            raise RuntimeError(
                f"{arm} episode {episode} has {len(rows)} seeds, expected {len(SEEDS)}"
            )
        output_row: dict[str, Any] = {
            "arm": arm,
            "episode": episode,
            "seed_count": len(rows),
        }
        for field in VALIDATION_METRICS:
            values = [float(row[field]) for row in rows]
            summary = _summary(values)
            output_row[f"{field}_mean"] = summary["mean"]
            output_row[f"{field}_std"] = summary["std"]
        output.append(output_row)
    return output


def _format_mean_std(stats: dict[str, Any]) -> str:
    return f"{stats['mean']:.6g} ± {stats['std']:.6g}"


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Matched current-code credit-assignment experiment",
        "",
        f"- Source SHA-256: `{payload['source_sha256']}`",
        "- Paired algorithm seeds: `11, 23, 37, 53, 71`",
        "- Statistical unit: one algorithm seed; test instances and sampled repeats are not independent training replicates.",
        "- Feasibility is interpreted before the greedy test quality score (lower is better).",
        "",
    ]
    labels = {
        "completion_rate": "Completion rate",
        "truncated_count": "Truncated instances",
        "schedule_violation_count": "Schedule violations",
        "quality_score": "Quality score",
        "flow_time_objective": "Flow time",
        "reconfiguration_cost": "Reconfiguration cost",
        "worker_load_variance": "Worker-load variance",
        "relative_heuristic_gap_percent": "Relative heuristic gap (%)",
    }
    for mode in ("greedy", "sampled"):
        lines.extend(
            [
                f"## Held-out test: {mode}",
                "",
                "| Metric | Baseline | Treatment | Δ treatment-baseline (95% bootstrap CI) | Wilcoxon p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in EVALUATION_METRICS:
            stats = payload["statistics"]["evaluation"][mode][metric]
            delta = stats["paired_difference_treatment_minus_baseline"]
            ci = delta["bootstrap_95_ci"]
            lines.append(
                f"| {labels[metric]} | {_format_mean_std(stats['baseline'])} | "
                f"{_format_mean_std(stats['treatment'])} | "
                f"{delta['mean']:.6g} [{ci[0]:.6g}, {ci[1]:.6g}] | "
                f"{stats['wilcoxon']['p_value_two_sided']:.6g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Training efficiency",
            "",
            "| Metric | Baseline | Treatment | Δ treatment-baseline |",
            "|---|---:|---:|---:|",
        ]
    )
    efficiency = (
        "policy_transitions",
        "forced_action_ratio",
        "policy_inference_seconds",
        "sampling_seconds",
        "ppo_update_seconds",
        "wall_clock_seconds",
    )
    for metric in efficiency:
        stats = payload["statistics"]["training"][metric]
        delta = stats["paired_difference_treatment_minus_baseline"]
        lines.append(
            f"| {metric} | {_format_mean_std(stats['baseline'])} | "
            f"{_format_mean_std(stats['treatment'])} | {delta['mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- The baseline validates every 10 episodes and the treatment every 20, so the baseline has twice as many native checkpoint-selection opportunities; convergence comparisons use only common 20-episode checkpoints.",
            "- With five non-zero paired differences, the smallest possible two-sided exact Wilcoxon p-value is 0.0625. Effect sizes, direction consistency, and confidence intervals therefore carry more weight than a p<0.05 threshold.",
            "- Secondary endpoints and sampled decoding are exploratory; no result is declared significant from p-values alone.",
            "",
            "## Historical single-seed appendix (excluded from statistics)",
            "",
            "- `result/runs/m1_seed11_1000_rerun_20260807_125418`",
            "- `result/runs/lambda0995_forcedcompress_epu20_seed11_1000_20260808`",
            "",
        ]
    )
    return "\n".join(lines)


def _require_complete_formal_design(
    state: dict[str, Any], manifest: dict[str, Any]
) -> None:
    for arm in ARMS:
        for seed in SEEDS:
            key = _run_key(arm, seed, False)
            entry = state.get("runs", {}).get(key)
            audit = _valid_existing_entry(
                entry,
                arm=arm,
                seed=seed,
                smoke=False,
                source_sha256=str(manifest["source_sha256"]),
            )
            if audit is None:
                raise RuntimeError(f"formal run is missing or invalid: {key}")
            evaluation = entry.get("evaluation_directory")
            checkpoint_sha = str(audit["checkpoint_sha256"])
            if not evaluation or not _evaluation_is_valid(
                Path(evaluation),
                source_sha256=str(manifest["source_sha256"]),
                checkpoint_sha256=checkpoint_sha,
            ):
                raise RuntimeError(f"evaluation is missing or invalid: {key}")


def aggregate_experiment(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    verify_source(manifest)
    state = _state(root)
    _require_complete_formal_design(state, manifest)
    audit_rows = _audit_rows(root, manifest, state)
    if any(not _bool(row["valid"]) for row in audit_rows if not _bool(row["smoke"])):
        raise RuntimeError("one or more formal audits failed")
    seed_rows = _seed_metric_rows(state)
    if len(seed_rows) != len(ARMS) * len(SEEDS) * 2:
        raise RuntimeError("seed metric table is incomplete")
    delta_rows, statistics_payload = _paired_outputs(seed_rows)
    convergence_rows = _convergence_rows(state)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "source_sha256": manifest["source_sha256"],
        "study_design": manifest["study_design"],
        "statistics": statistics_payload,
        "limitations": {
            "seed_count": len(SEEDS),
            "significance_claim_from_p_value_alone": False,
            "minimum_two_sided_exact_wilcoxon_p_for_five_nonzero_pairs": 0.0625,
            "checkpoint_selection_opportunities": {
                "baseline": 100,
                "treatment": 50,
            },
            "historical_single_seed_results_in_primary_statistics": False,
        },
    }
    write_csv(root / "seed_metrics.csv", seed_rows)
    write_csv(root / "paired_deltas.csv", delta_rows)
    write_csv(root / "convergence_common20.csv", convergence_rows)
    write_json(root / "aggregate.json", payload)
    (root / "report.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    return payload


def run_experiment(
    root: Path,
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    smoke: bool,
) -> None:
    root = root.resolve()
    manifest = initialize_experiment(root)
    state = _state(root)
    for arm, seed in planned_runs(arms, seeds, smoke=smoke):
        entry, _ = execute_training(
            root,
            manifest,
            state,
            arm=arm,
            seed=seed,
            smoke=smoke,
        )
        if not smoke:
            verify_source(manifest)
            evaluate_checkpoint(root, manifest, state, entry)
    _audit_rows(root, manifest, state)
    if not smoke and set(arms) == set(ARMS) and tuple(seeds) == SEEDS:
        aggregate_experiment(root)
        print(f"matched experiment report: {root / 'report.md'}")
    elif not smoke:
        print("partial formal design completed; aggregate after all arms and seeds")
    else:
        print(f"smoke audit: {root / 'run_audit.csv'}")


def audit_experiment(root: Path) -> bool:
    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    verify_source(manifest)
    rows = _audit_rows(root, manifest, _state(root))
    return bool(rows) and all(_bool(row["valid"]) for row in rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze the matched current-code credit-assignment study"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run training and evaluation")
    run_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run_parser.add_argument(
        "--arm",
        dest="arms",
        action="append",
        choices=tuple(ARMS),
        help="run one arm; repeat for both (default: both)",
    )
    run_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
    )
    run_parser.add_argument("--smoke", action="store_true")
    audit_parser = subparsers.add_parser("audit", help="audit existing runs")
    audit_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="rebuild multi-seed outputs"
    )
    aggregate_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        seeds = tuple(int(seed) for seed in args.seeds)
        invalid = sorted(set(seeds) - set(SEEDS))
        if invalid:
            raise SystemExit(f"unsupported algorithm seeds: {invalid}")
        if len(seeds) != len(set(seeds)):
            raise SystemExit("algorithm seeds must be unique")
        arms = tuple(args.arms) if args.arms else tuple(ARMS)
        if len(arms) != len(set(arms)):
            raise SystemExit("arms must be unique")
        if args.smoke and seeds == SEEDS:
            seeds = (11,)
        run_experiment(
            args.root,
            arms=arms,
            seeds=seeds,
            smoke=bool(args.smoke),
        )
    elif args.command == "audit":
        if not audit_experiment(args.root):
            raise SystemExit(1)
    elif args.command == "aggregate":
        aggregate_experiment(args.root)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
