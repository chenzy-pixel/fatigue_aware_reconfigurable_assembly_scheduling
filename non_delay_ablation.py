from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon

from configs import load_config
from configs.config import public_config
from result.io import write_csv, write_json
from train import FORCED_ACTION_COUNT_FIELDS, train


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "configs/credit_assignment_lambda0995_forced_epu20_seed11.json"
DEFAULT_ROOT = PROJECT_ROOT / "result/non_delay_ablation_seed11_600_20260810"
ALGORITHM_SEED = 11
EPISODES = 600
PARALLEL_ENVS = 20
ARMS = {
    "non_delay_on": True,
    "non_delay_off": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def effective_config(non_delay: bool, *, smoke: bool = False) -> dict[str, Any]:
    config = deepcopy(load_config(BASE_CONFIG))
    arm = "non_delay_on" if non_delay else "non_delay_off"
    config["experiment_name"] = f"{arm}_seed11_{EPISODES}"
    config["seed"] = ALGORITHM_SEED
    config["training"]["episodes"] = EPISODES
    config["training"]["parallel_envs"] = PARALLEL_ENVS
    config["training"]["validation_parallel_envs"] = PARALLEL_ENVS
    config["training"]["forced_action_compression"] = True
    config["environment"]["worker_resource_control"][
        "non_delay_worker_dispatch"
    ] = bool(non_delay)
    if smoke:
        config["training"]["smoke_episodes"] = PARALLEL_ENVS
        config["training"]["smoke_parallel_envs"] = PARALLEL_ENVS
    return config


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        if key == "_config_path":
            continue
        child = f"{prefix}.{key}" if prefix else key
        result.update(_leaf_values(value[key], child))
    return result


def validate_design(*, smoke: bool = False) -> set[str]:
    configs = {
        arm: effective_config(value, smoke=smoke)
        for arm, value in ARMS.items()
    }
    leaves = {arm: _leaf_values(config) for arm, config in configs.items()}
    paths = set(leaves["non_delay_on"]) | set(leaves["non_delay_off"])
    differences = {
        path
        for path in paths
        if leaves["non_delay_on"].get(path)
        != leaves["non_delay_off"].get(path)
    }
    expected = {
        "experiment_name",
        "environment.worker_resource_control.non_delay_worker_dispatch",
    }
    if differences != expected:
        raise ValueError(
            "non-delay arms are not matched: "
            f"expected={sorted(expected)}, observed={sorted(differences)}"
        )
    for arm, config in configs.items():
        if int(config["training"]["episodes"]) != EPISODES:
            raise ValueError(f"{arm} must use {EPISODES} episodes")
        if int(config["seed"]) != ALGORITHM_SEED:
            raise ValueError(f"{arm} must use algorithm seed {ALGORITHM_SEED}")
        if not bool(config["training"]["forced_action_compression"]):
            raise ValueError(f"{arm} must keep forced-action compression enabled")
        if float(config["ppo"]["gamma"]) != 1.0:
            raise ValueError(f"{arm} must use gamma=1")
    return differences


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        Path("environment/env.py"),
        Path("agent/ppo/parallel.py"),
        Path("agent/ppo/network.py"),
        Path("train.py"),
        Path("eval.py"),
        Path("result/metrics.py"),
        Path("non_delay_ablation.py"),
        BASE_CONFIG.relative_to(PROJECT_ROOT),
    ):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update((PROJECT_ROOT / relative).read_bytes())
    return digest.hexdigest()


def initialize(root: Path, *, smoke: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest["source_sha256"] != _source_sha256():
            raise RuntimeError("source changed after experiment initialization")
        return manifest
    differences = validate_design(smoke=smoke)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now(),
        "study": "forced_action_classification_and_non_delay_ablation",
        "source_sha256": _source_sha256(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short").splitlines(),
        "base_config": str(BASE_CONFIG),
        "algorithm_seed": ALGORITHM_SEED,
        "episodes_per_arm": PARALLEL_ENVS if smoke else EPISODES,
        "parallel_envs": PARALLEL_ENVS,
        "arms": ARMS,
        "allowed_arm_differences": sorted(differences),
        "fixed_instance_sequence": True,
        "primary_metrics": [
            "completion_rate",
            "flow_time_objective",
            "reconfiguration_cost",
            "worker_load_variance",
            "forced_action_ratio",
            "longest_forced_action_chain",
        ],
    }
    write_json(manifest_path, manifest)
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    for arm, enabled in ARMS.items():
        write_json(
            provenance / f"{arm}_config.json",
            public_config(effective_config(enabled, smoke=smoke)),
        )
    return manifest


def _load_state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    if path.exists():
        return _read_json(path)
    return {"schema_version": "1.0.0", "runs": {}}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    write_json(root / "state.json", state)


def run_arm(root: Path, arm: str, *, smoke: bool) -> Path:
    state = _load_state(root)
    existing = state["runs"].get(arm)
    if existing and existing.get("status") == "complete":
        run_directory = Path(existing["run_directory"])
        if (run_directory / "summary.json").exists():
            print(f"[{arm}] reuse completed run {run_directory}", flush=True)
            return run_directory
    config = effective_config(ARMS[arm], smoke=smoke)
    suffix = "smoke" if smoke else f"{EPISODES}"
    base_run_name = (
        f"non_delay_ablation_{arm}_seed{ALGORITHM_SEED}_{suffix}_20260810"
    )
    run_name = base_run_name
    result_root = Path(config["paths"]["result_root"])
    if not result_root.is_absolute():
        result_root = PROJECT_ROOT / result_root
    retry = 0
    while (result_root / run_name).exists():
        retry += 1
        run_name = f"{base_run_name}_retry{retry}"
    state["runs"][arm] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "run_name": run_name,
    }
    _save_state(root, state)
    print(
        f"[{arm}] training start: seed={ALGORITHM_SEED}, "
        f"episodes={PARALLEL_ENVS if smoke else EPISODES}",
        flush=True,
    )
    run_directory = train(
        config,
        smoke=smoke,
        run_name=run_name,
        online_instances=True,
        algorithm_seed=ALGORITHM_SEED,
        parallel_envs=PARALLEL_ENVS,
        visdom_enabled=False,
    )
    state = _load_state(root)
    state["runs"][arm] = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "run_name": run_name,
        "run_directory": str(run_directory.resolve()),
    }
    _save_state(root, state)
    print(f"[{arm}] training complete: {run_directory}", flush=True)
    return run_directory


def _numbers(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in {None, ""}:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = _numbers(rows, field)
    return float(statistics.fmean(values)) if values else math.nan


def _training_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    environment_steps = sum(int(float(row["steps"])) for row in rows)
    diagnostic_states = sum(
        int(float(row["forced_action_state_count"])) for row in rows
    )
    compressed_forced = sum(int(float(row["forced_actions"])) for row in rows)
    if diagnostic_states != compressed_forced:
        raise RuntimeError(
            "environment forced-state diagnostics disagree with PPO compression: "
            f"diagnostic={diagnostic_states}, compressed={compressed_forced}"
        )
    longest = _numbers(rows, "longest_forced_action_chain")
    counts = {
        name: sum(int(float(row.get(name, 0) or 0)) for row in rows)
        for name in FORCED_ACTION_COUNT_FIELDS
    }
    return {
        "episode_count": len(rows),
        "completion_rate": sum(
            str(row["terminated"]).lower() == "true" for row in rows
        )
        / len(rows),
        "mean_flow_time_objective": _mean(rows, "flow_time_objective"),
        "mean_reconfiguration_cost": _mean(rows, "reconfiguration_cost"),
        "mean_worker_load_variance": _mean(rows, "worker_load_variance"),
        "environment_steps": environment_steps,
        "forced_action_ratio": (
            diagnostic_states / environment_steps if environment_steps else 0.0
        ),
        "mean_episode_longest_forced_action_chain": float(np.mean(longest)),
        "p95_episode_longest_forced_action_chain": float(
            np.percentile(longest, 95)
        ),
        "maximum_forced_action_chain": int(max(longest, default=0)),
        "non_delay_forced_share": (
            counts["forced_worker_pair_non_delay_count"] / diagnostic_states
            if diagnostic_states
            else 0.0
        ),
        **counts,
    }


def _evaluation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("final_checkpoint_evaluation")
    if not isinstance(payload, dict) or not isinstance(payload.get("greedy"), dict):
        return {"available": False}
    greedy = payload["greedy"]
    metrics = greedy["all_instance_metrics"]

    def mean(name: str) -> float | None:
        value = metrics.get(name, {}).get("mean")
        return None if value is None else float(value)

    instance_count = int(greedy["instance_count"])
    forced_mean = mean("forced_action_state_count") or 0.0
    return {
        "available": True,
        "instance_count": instance_count,
        "completion_rate": float(greedy["completion_rate"]),
        "mean_flow_time_objective": mean("flow_time_objective"),
        "mean_reconfiguration_cost": mean("reconfiguration_cost"),
        "mean_worker_load_variance": mean("worker_load_variance"),
        "forced_action_ratio": (
            forced_mean * instance_count / int(greedy["decision_count"])
            if int(greedy["decision_count"])
            else 0.0
        ),
        "mean_longest_forced_action_chain": mean(
            "longest_forced_action_chain"
        ),
        "mean_forced_worker_pair_non_delay_count": mean(
            "forced_worker_pair_non_delay_count"
        ),
    }


def _paired_statistic(
    on_values: list[float],
    off_values: list[float],
) -> dict[str, Any]:
    if len(on_values) != len(off_values) or not on_values:
        raise ValueError("paired samples must be non-empty and aligned")
    differences = np.asarray(off_values) - np.asarray(on_values)
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(off_values, on_values)
    return {
        "count": len(on_values),
        "on_mean": float(np.mean(on_values)),
        "off_mean": float(np.mean(off_values)),
        "off_minus_on_mean": float(np.mean(differences)),
        "off_minus_on_median": float(np.median(differences)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value_two_sided": float(p_value),
    }


def aggregate(root: Path) -> dict[str, Any]:
    state = _load_state(root)
    if set(state.get("runs", {})) != set(ARMS):
        raise RuntimeError("both non-delay arms must complete before aggregation")
    rows_by_arm: dict[str, list[dict[str, str]]] = {}
    comparison_rows = []
    classification_rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_directory = Path(state["runs"][arm]["run_directory"])
        rows = _read_csv(run_directory / "train_log.csv")
        rows_by_arm[arm] = rows
        training = _training_summary(rows)
        evaluation = _evaluation_summary(_read_json(run_directory / "summary.json"))
        summaries[arm] = {"training": training, "evaluation": evaluation}
        comparison_rows.append(
            {
                "arm": arm,
                "non_delay_worker_dispatch": ARMS[arm],
                **{f"training_{key}": value for key, value in training.items()},
                **{f"evaluation_{key}": value for key, value in evaluation.items()},
            }
        )
        for field in FORCED_ACTION_COUNT_FIELDS:
            classification_rows.append(
                {
                    "arm": arm,
                    "classification": field,
                    "count": training[field],
                    "share_of_forced_states": (
                        training[field] / training["forced_action_state_count"]
                        if training["forced_action_state_count"]
                        else 0.0
                    ),
                }
            )

    on_rows = rows_by_arm["non_delay_on"]
    off_rows = rows_by_arm["non_delay_off"]
    on_keys = [(row["episode"], row["instance_seed"]) for row in on_rows]
    off_keys = [(row["episode"], row["instance_seed"]) for row in off_rows]
    if on_keys != off_keys:
        raise RuntimeError("the two arms did not use the same fixed instance sequence")
    paired_fields = (
        "flow_time_objective",
        "reconfiguration_cost",
        "worker_load_variance",
        "forced_action_ratio",
        "longest_forced_action_chain",
        "forced_worker_pair_non_delay_count",
    )
    paired = {
        field: _paired_statistic(
            [float(row[field]) for row in on_rows],
            [float(row[field]) for row in off_rows],
        )
        for field in paired_fields
    }
    write_csv(root / "comparison.csv", comparison_rows)
    write_csv(root / "forced_action_classification.csv", classification_rows)
    write_json(root / "paired_statistics.json", paired)
    write_json(root / "aggregate.json", summaries)
    _write_text(root / "report.md", _report(summaries, paired))
    print(f"aggregate report: {root / 'report.md'}", flush=True)
    return {"arms": summaries, "paired": paired}


def _report(
    summaries: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Forced-action classification and non-delay ablation",
        "",
        f"Algorithm seed: {ALGORITHM_SEED}; episodes per arm: "
        f"{summaries['non_delay_on']['training']['episode_count']}; "
        "both arms use the same episode-indexed instance seeds.",
        "",
        "## Training rollouts",
        "",
        "| Metric | non-delay on | non-delay off | off - on |",
        "|---|---:|---:|---:|",
    ]
    on = summaries["non_delay_on"]["training"]
    off = summaries["non_delay_off"]["training"]
    metrics = (
        ("Completion rate", "completion_rate"),
        ("Mean flow-time objective", "mean_flow_time_objective"),
        ("Mean reconfiguration cost", "mean_reconfiguration_cost"),
        ("Mean worker-load variance", "mean_worker_load_variance"),
        ("Forced-action ratio", "forced_action_ratio"),
        ("P95 episode longest forced chain", "p95_episode_longest_forced_action_chain"),
        ("Maximum forced chain", "maximum_forced_action_chain"),
        ("WORKER unique-pair blocked by non-delay", "forced_worker_pair_non_delay_count"),
    )
    for label, field in metrics:
        on_value = float(on[field])
        off_value = float(off[field])
        lines.append(
            f"| {label} | {on_value:.6g} | {off_value:.6g} | "
            f"{off_value - on_value:+.6g} |"
        )
    lines.extend(
        [
            "",
            "## Paired episode statistics",
            "",
            "Differences are `non-delay off - non-delay on`; minimization metrics "
            "are better when negative.",
            "",
            "| Metric | Mean difference | Median difference | Wilcoxon p |",
            "|---|---:|---:|---:|",
        ]
    )
    for field, payload in paired.items():
        lines.append(
            f"| {field} | {payload['off_minus_on_mean']:+.6g} | "
            f"{payload['off_minus_on_median']:+.6g} | "
            f"{payload['wilcoxon_p_value_two_sided']:.6g} |"
        )
    lines.extend(["", "## Fixed validation", ""])
    eval_on = summaries["non_delay_on"]["evaluation"]
    eval_off = summaries["non_delay_off"]["evaluation"]
    if eval_on.get("available") and eval_off.get("available"):
        lines.extend(
            [
                "| Metric | non-delay on | non-delay off |",
                "|---|---:|---:|",
            ]
        )
        for label, field in (
            ("Completion rate", "completion_rate"),
            ("Mean flow-time objective", "mean_flow_time_objective"),
            ("Mean reconfiguration cost", "mean_reconfiguration_cost"),
            ("Mean worker-load variance", "mean_worker_load_variance"),
            ("Forced-action ratio", "forced_action_ratio"),
            ("Mean longest forced chain", "mean_longest_forced_action_chain"),
        ):
            lines.append(
                f"| {label} | {float(eval_on[field]):.6g} | "
                f"{float(eval_off[field]):.6g} |"
            )
    else:
        lines.append("No formal final checkpoint evaluation was available.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-seed forced-action/non-delay ablation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run_parser.add_argument("--smoke", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "aggregate":
        aggregate(root)
        return
    initialize(root, smoke=args.smoke)
    for arm in ARMS:
        run_arm(root, arm, smoke=args.smoke)
    aggregate(root)


if __name__ == "__main__":
    main()
