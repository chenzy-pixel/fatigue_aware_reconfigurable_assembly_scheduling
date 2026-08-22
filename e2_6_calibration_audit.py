"""Read-only preregistered E2.6 calibration selection audit.

The auditor intentionally never invokes training/evaluation.  It inspects the
three fixed seed-101 validation runs and, when available, the E2.4 control.
Its only admissible decisions are ``selected``, ``stopped`` and
``guard_pending``; malformed or incomplete inputs are reported as errors
rather than silently treated as a scientific result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PRE_REGISTERED_LAMBDAS = (0.05, 0.10, 0.20)
FULL_GRID_SCOPE = "full_grid_22"
EXPECTED_SEED = 101
EXPECTED_EPISODES = 500
CANONICAL_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class RunAudit:
    coefficient: float | None
    run_directory: Path
    full_grid_rows: tuple[dict[str, str], dict[str, str]]
    base_pass: bool
    failures: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error


def _number(row: Mapping[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"audit row has no finite numeric {name!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"audit row has non-finite {name!r}")
    return value


def _integer(row: Mapping[str, str], name: str) -> int:
    value = _number(row, name)
    if not value.is_integer():
        raise ValueError(f"audit row {name!r} must be integral")
    return int(value)


def _boolean(row: Mapping[str, str], name: str) -> bool:
    value = str(row.get(name, "")).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"audit row has no boolean {name!r}")


def _validate_run_metadata(
    run_directory: Path,
    *,
    coefficient: float | None,
    is_control: bool,
) -> None:
    config = _read_json(run_directory / "config.json")
    summary = _read_json(run_directory / "summary.json")
    if int(config.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError(f"{run_directory}: seed must be {EXPECTED_SEED}")
    if config.get("algorithm_seeds") != [EXPECTED_SEED]:
        raise ValueError(f"{run_directory}: algorithm_seeds must be [101]")
    if int(config.get("training", {}).get("episodes", -1)) != EXPECTED_EPISODES:
        raise ValueError(f"{run_directory}: config must request 500 episodes")
    if int(summary.get("episodes", -1)) != EXPECTED_EPISODES:
        raise ValueError(f"{run_directory}: summary must record 500 episodes")
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping) or not all(
        isinstance(provenance.get(name), str) and provenance[name]
        for name in (
            "source_state_sha256",
            "effective_config_sha256",
            "dataset_manifest_sha256",
        )
    ):
        raise ValueError(f"{run_directory}: missing reproducibility provenance")
    if str(config.get("training", {}).get("validation_split", "")) != "validation":
        raise ValueError(f"{run_directory}: calibration may use validation only")
    if is_control:
        if str(config.get("experiment_name", "")).find("e2_4_calibration_control") < 0:
            raise ValueError(f"{run_directory}: control must be the E2.4 calibration")
        return
    if str(config.get("evaluation", {}).get("result_schema_version", "")) != "4.7.0":
        raise ValueError(f"{run_directory}: E2.6 schema must be 4.7.0")
    auxiliary = config.get("ppo", {}).get(
        "counterfactual_preference_consistency", {}
    )
    if not isinstance(auxiliary, dict) or not bool(auxiliary.get("enabled")):
        raise ValueError(f"{run_directory}: E2.6 auxiliary loss is not enabled")
    observed = float(auxiliary.get("loss_coefficient", math.nan))
    if coefficient is None or not math.isclose(observed, coefficient, abs_tol=1e-12):
        raise ValueError(f"{run_directory}: calibration coefficient does not match manifest")


def _row_failures(row: Mapping[str, str]) -> list[str]:
    failures: list[str] = []
    for name in (
        "all_safe",
        "coverage_pass",
        "controllability_pass",
        "worker_direct_preference_pass",
        "preference_response_pass",
        "low_flow_safety_pass",
        "counterfactual_gate_pass",
    ):
        if not _boolean(row, name):
            failures.append(name)
    if _integer(row, "instance_count") != 20:
        failures.append("instance_count")
    if _integer(row, "preference_count") != 22:
        failures.append("preference_count")
    if _integer(row, "candidate_count") != 440:
        failures.append("candidate_count")
    if _integer(row, "schedule_violation_count") != 0:
        failures.append("schedule_violation_count")
    if _number(row, "completion_rate") < 1.0 - 1e-12:
        failures.append("completion_rate")
    if _integer(row, "low_flow_candidate_count") != 220:
        failures.append("low_flow_candidate_count")
    if _number(row, "low_flow_completion_rate") < 1.0 - 1e-12:
        failures.append("low_flow_completion_rate")
    for name, threshold in (
        ("mean_unique_action_trace_count", 8.0),
        ("mean_unique_objective_count", 8.0),
        ("mean_nondominated_count", 4.0),
    ):
        if _number(row, name) < threshold - 1e-12:
            failures.append(name)
    for name in (
        "preference_response_spearman_flow",
        "preference_response_spearman_cost",
        "preference_response_spearman_variance",
    ):
        if _number(row, name) > -0.05 + 1e-12:
            failures.append(name)
    if _integer(row, "counterfactual_instance_coverage") != 20:
        failures.append("counterfactual_instance_coverage")
    if _number(row, "counterfactual_high_flow_commit_flip_rate") < 0.25 - 1e-12:
        failures.append("counterfactual_high_flow_commit_flip_rate")
    if _integer(row, "counterfactual_low_flow_identity_violation_count") != 0:
        failures.append("counterfactual_low_flow_identity_violation_count")
    if _integer(row, "counterfactual_monotonicity_violation_count") != 0:
        failures.append("counterfactual_monotonicity_violation_count")
    return failures


def _audit_run(
    run_directory: Path,
    *,
    coefficient: float | None,
    is_control: bool,
) -> RunAudit:
    _validate_run_metadata(
        run_directory, coefficient=coefficient, is_control=is_control
    )
    rows = [
        row
        for row in _read_csv(run_directory / "pareto_validation_log.csv")
        if row.get("scope") == FULL_GRID_SCOPE
    ]
    if len(rows) < 2:
        raise ValueError(f"{run_directory}: need two full-grid audit rows")
    final_rows = (rows[-2], rows[-1])
    failures = tuple(
        f"audit_{index + 1}:{name}"
        for index, row in enumerate(final_rows)
        for name in (_row_failures(row) if not is_control else [])
    )
    return RunAudit(
        coefficient=coefficient,
        run_directory=run_directory,
        full_grid_rows=final_rows,
        base_pass=not failures,
        failures=failures,
    )


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    candidate_paths = manifest.get("candidates")
    if not isinstance(candidate_paths, dict):
        raise ValueError("manifest.candidates must be an object")
    parsed_candidates = {float(key): Path(value) for key, value in candidate_paths.items()}
    if set(parsed_candidates) != set(PRE_REGISTERED_LAMBDAS):
        raise ValueError("manifest must contain exactly lambda=0.05, 0.10 and 0.20")
    candidates = [
        _audit_run(
            parsed_candidates[coefficient],
            coefficient=coefficient,
            is_control=False,
        )
        for coefficient in PRE_REGISTERED_LAMBDAS
    ]
    base_passing = [candidate for candidate in candidates if candidate.base_pass]
    control_value = manifest.get("e2_4_control_run")
    control: RunAudit | None = None
    if control_value not in (None, ""):
        control = _audit_run(
            Path(str(control_value)), coefficient=None, is_control=True
        )
    if not base_passing:
        status = "stopped"
        selected = None
        guard_failures: dict[str, list[str]] = {}
    elif control is None:
        status = "guard_pending"
        selected = None
        guard_failures = {}
    else:
        guard_failures = {}
        passing: list[RunAudit] = []
        for candidate in base_passing:
            failures: list[str] = []
            for index, (candidate_row, control_row) in enumerate(
                zip(candidate.full_grid_rows, control.full_grid_rows)
            ):
                candidate_quality = _number(candidate_row, "canonical_quality")
                control_quality = _number(control_row, "canonical_quality")
                if candidate_quality > control_quality * (1.0 + CANONICAL_RELATIVE_TOLERANCE) + 1e-12:
                    failures.append(f"audit_{index + 1}:canonical_quality_guard")
            if failures:
                guard_failures[f"{candidate.coefficient:.2f}"] = failures
            else:
                passing.append(candidate)
        if passing:
            selected_run = min(passing, key=lambda value: float(value.coefficient))
            status = "selected"
            selected = float(selected_run.coefficient)
        else:
            status = "stopped"
            selected = None
    return {
        "protocol": "e2_6_counterfactual_calibration_v1",
        "status": status,
        "selected_loss_coefficient": selected,
        "formal_seed_training_authorized": status == "selected",
        "control_available": control is not None,
        "candidates": [
            {
                "loss_coefficient": candidate.coefficient,
                "run_directory": str(candidate.run_directory),
                "base_pass": candidate.base_pass,
                "failures_before_canonical_guard": list(candidate.failures),
                "final_full_grid_updates": [
                    _integer(row, "update_id")
                    for row in candidate.full_grid_rows
                ],
            }
            for candidate in candidates
        ],
        "canonical_guard_failures": guard_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_manifest(args.manifest)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
