from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Any


NUMERIC_UPDATE_FIELDS = (
    "advantage_std",
    "approx_kl",
    "clip_fraction",
    "entropy",
    "value_loss",
)
VALIDATION_FIELDS = (
    "completion_rate",
    "truncated_count",
    "mean_flow_time_objective",
    "mean_reconfiguration_cost",
    "mean_worker_load_variance",
    "mean_quality_score",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := _number(row.get(field))) is not None
    ]
    return fmean(values) if values else None


def _window_means(
    rows: list[dict[str, str]],
    field: str,
) -> dict[str, float | None]:
    window = max(1, len(rows) // 5)
    return {
        "overall": _mean(rows, field),
        "early_20_percent": _mean(rows[:window], field),
        "late_20_percent": _mean(rows[-window:], field),
    }


def _validation_snapshot(row: dict[str, str] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "episode": int(float(row["episode"])),
        **{field: _number(row.get(field)) for field in VALIDATION_FIELDS},
    }


def _best_summary_validation(summary: dict[str, Any]) -> dict[str, Any] | None:
    value = summary.get("best_validation")
    if not isinstance(value, dict):
        return None
    return {
        "episode": value.get("episode"),
        **{field: value.get(field) for field in VALIDATION_FIELDS},
    }


def _run_metrics(run_directory: Path, *, common_interval: int) -> dict[str, Any]:
    summary = _read_json(run_directory / "summary.json")
    config = _read_json(run_directory / "config.json")
    train_rows = _read_csv(run_directory / "train_log.csv")
    update_rows = _read_csv(run_directory / "update_log.csv")
    validation_rows = _read_csv(run_directory / "validation_log.csv")
    common_validation_rows = [
        row
        for row in validation_rows
        if int(float(row["episode"])) % common_interval == 0
    ]
    environment_steps = summary.get("environment_steps")
    if environment_steps is None:
        environment_steps = sum(
            int(float(row.get("steps", 0))) for row in train_rows
        )
    transitions = int(summary.get("transitions", 0))
    forced_actions = summary.get("forced_actions")
    if forced_actions is None:
        forced_actions = max(0, int(environment_steps) - transitions)
    forced_action_ratio = (
        float(forced_actions) / float(environment_steps)
        if float(environment_steps) > 0
        else 0.0
    )
    terminated_count = sum(
        str(row.get("terminated", "")).lower() == "true"
        for row in train_rows
    )
    truncated_count = sum(
        str(row.get("truncated", "")).lower() == "true"
        for row in train_rows
    )
    reward_errors = [
        abs(value)
        for row in train_rows
        if (value := _number(row.get("reward_identity_error"))) is not None
    ]
    return {
        "run_directory": str(run_directory.resolve()),
        "config": {
            "experiment_name": config.get("experiment_name"),
            "seed": config.get("seed"),
            "gamma": config.get("ppo", {}).get("gamma"),
            "gae_lambda": config.get("ppo", {}).get("gae_lambda"),
            "parallel_envs": config.get("training", {}).get("parallel_envs"),
            "forced_action_compression": config.get("training", {}).get(
                "forced_action_compression", False
            ),
        },
        "training": {
            "episodes": int(summary.get("episodes", len(train_rows))),
            "updates": int(summary.get("updates", len(update_rows))),
            "environment_steps": int(environment_steps),
            "policy_transitions": transitions,
            "forced_actions": int(forced_actions),
            "forced_action_ratio": forced_action_ratio,
            "mean_policy_steps_per_episode": (
                transitions / len(train_rows) if train_rows else 0.0
            ),
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "maximum_absolute_reward_identity_error": (
                max(reward_errors) if reward_errors else None
            ),
            "formal_training_status": summary.get("formal_training_status"),
        },
        "timing": {
            "sampling_seconds": summary.get("total_sampling_time_seconds"),
            "policy_inference_seconds": summary.get(
                "total_policy_inference_time_seconds"
            ),
            "ppo_update_seconds": summary.get(
                "total_ppo_update_time_seconds"
            ),
        },
        "ppo_diagnostics": {
            field: _window_means(update_rows, field)
            for field in NUMERIC_UPDATE_FIELDS
        },
        "validation": {
            "common_interval": common_interval,
            "common_checkpoint_count": len(common_validation_rows),
            "final_common_checkpoint": _validation_snapshot(
                common_validation_rows[-1]
                if common_validation_rows
                else None
            ),
            "run_selected_best": _best_summary_validation(summary),
        },
    }


def _delta(treatment: Any, baseline: Any) -> dict[str, float | None]:
    treatment_value = _number(treatment)
    baseline_value = _number(baseline)
    if treatment_value is None or baseline_value is None:
        return {"absolute": None, "percent": None}
    absolute = treatment_value - baseline_value
    percent = (
        100.0 * absolute / abs(baseline_value)
        if baseline_value != 0.0
        else None
    )
    return {"absolute": absolute, "percent": percent}


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def git(*arguments: str, binary: bool = False):
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=not binary,
        ).stdout

    commit = str(git("rev-parse", "HEAD")).strip()
    diff = git("diff", "--binary", "--", ".", binary=True)
    status = str(git("status", "--short"))
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "workspace_sha256": hashlib.sha256(
            diff + status.encode("utf-8")
        ).hexdigest(),
        "status": status.splitlines(),
    }


def _markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    treatment = payload["treatment"]
    rows = []
    fields = (
        ("Episodes", "episodes"),
        ("PPO updates", "updates"),
        ("Environment steps", "environment_steps"),
        ("Policy transitions", "policy_transitions"),
        ("Forced action ratio", "forced_action_ratio"),
        ("Terminated", "terminated_count"),
        ("Truncated", "truncated_count"),
    )
    for label, field in fields:
        before = baseline["training"].get(field)
        after = treatment["training"].get(field)
        difference = _delta(after, before)["absolute"]
        rows.append(f"| {label} | {before} | {after} | {difference} |")
    baseline_validation = baseline["validation"]["final_common_checkpoint"] or {}
    treatment_validation = treatment["validation"]["final_common_checkpoint"] or {}
    validation_rows = []
    for field in VALIDATION_FIELDS:
        before = baseline_validation.get(field)
        after = treatment_validation.get(field)
        difference = _delta(after, before)["absolute"]
        validation_rows.append(
            f"| {field} | {before} | {after} | {difference} |"
        )
    return "\n".join(
        [
            "# Credit-assignment pilot comparison",
            "",
            "> Exploratory single-seed comparison against a historical run. "
            "Code-version differences are a confound; no significance claim is made.",
            "",
            "## Training and compression",
            "",
            "| Metric | Historical baseline | Treatment | Delta |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "## Final validation on common 20-episode checkpoints",
            "",
            "| Metric | Historical baseline | Treatment | Delta |",
            "|---|---:|---:|---:|",
            *validation_rows,
            "",
            "## Configuration",
            "",
            f"- Baseline: `{baseline['config']}`",
            f"- Treatment: `{treatment['config']}`",
            "",
        ]
    )


def compare_runs(
    baseline_directory: Path,
    treatment_directory: Path,
    *,
    common_interval: int = 20,
) -> dict[str, Any]:
    baseline = _run_metrics(
        baseline_directory, common_interval=common_interval
    )
    treatment = _run_metrics(
        treatment_directory, common_interval=common_interval
    )
    baseline_final = baseline["validation"]["final_common_checkpoint"] or {}
    treatment_final = treatment["validation"]["final_common_checkpoint"] or {}
    return {
        "study_design": {
            "type": "exploratory_single_seed_historical_control",
            "seed": 11,
            "common_validation_interval": common_interval,
            "significance_claim": False,
            "caveat": (
                "The historical baseline may use a different code snapshot; "
                "the comparison is not a causal estimate of any one change."
            ),
        },
        "baseline": baseline,
        "treatment": treatment,
        "final_common_validation_deltas": {
            field: _delta(treatment_final.get(field), baseline_final.get(field))
            for field in VALIDATION_FIELDS
        },
        "workspace_provenance": _git_provenance(
            Path(__file__).resolve().parents[1]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the forced-action credit-assignment pilot"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--common-interval", type=int, default=20)
    args = parser.parse_args()
    if args.common_interval <= 0:
        parser.error("--common-interval must be positive")
    payload = compare_runs(
        args.baseline,
        args.treatment,
        common_interval=args.common_interval,
    )
    json_path = args.treatment / "credit_assignment_comparison.json"
    markdown_path = args.treatment / "credit_assignment_comparison.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
