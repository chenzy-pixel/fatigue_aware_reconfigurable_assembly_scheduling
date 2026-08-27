"""Plot raw validation convergence for the three E1 single-objective runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from result.io import write_csv, write_json


OBJECTIVE_FIELDS = {
    "flow": "mean_flow_time_objective",
    "cost": "mean_reconfiguration_cost",
    "variance": "mean_worker_load_variance",
}
OBJECTIVE_LABELS = {
    "flow": "Flow-time objective F",
    "cost": "Reconfiguration cost C",
    "variance": "Worker-load variance V",
}
HARD_GATE_FIELDS = (
    "completion_rate",
    "truncated_count",
    "schedule_violation_count",
)
QUALITY_EXPLORATION_FLOOR = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: Any, *, field: str, row_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"validation row {row_index}: {field} must be numeric"
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"validation row {row_index}: {field} must be finite"
        )
    return number


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_one_hot_config(config: dict[str, Any], objective: str) -> None:
    weights = config.get("reward", {}).get("quality_weights")
    if not isinstance(weights, dict) or set(weights) != set(OBJECTIVE_FIELDS):
        raise ValueError("run config must define flow/cost/variance quality weights")
    expected = {
        name: 1.0 if name == objective else 0.0
        for name in OBJECTIVE_FIELDS
    }
    observed = {name: float(weights[name]) for name in OBJECTIVE_FIELDS}
    if observed != expected:
        raise ValueError(
            f"{objective} run has incompatible quality weights: {observed}"
        )


def load_plot_rows(
    run_directory: str | Path,
    objective: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the raw validation points used by one figure."""

    if objective not in OBJECTIVE_FIELDS:
        raise ValueError(f"unknown objective {objective!r}")
    run_path = Path(run_directory).resolve()
    validation_path = run_path / "validation_log.csv"
    config_path = run_path / "config.json"
    summary_path = run_path / "summary.json"
    for path in (validation_path, config_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _read_json(config_path)
    summary = _read_json(summary_path)
    _validate_one_hot_config(config, objective)
    source_rows = _read_csv(validation_path)
    if not source_rows:
        raise ValueError(f"{validation_path} contains no validation points")

    required = {
        "episode",
        *HARD_GATE_FIELDS,
        *OBJECTIVE_FIELDS.values(),
    }
    missing = required - set(source_rows[0])
    if missing:
        raise ValueError(
            f"{validation_path} is missing required fields: {sorted(missing)}"
        )

    plot_rows: list[dict[str, Any]] = []
    previous_episode = -1
    for row_index, source in enumerate(source_rows, start=2):
        episode_value = _finite_float(
            source["episode"], field="episode", row_index=row_index
        )
        if not episode_value.is_integer():
            raise ValueError(
                f"validation row {row_index}: episode must be an integer"
            )
        episode = int(episode_value)
        if episode <= previous_episode:
            raise ValueError("validation episodes must be strictly increasing")
        previous_episode = episode
        row: dict[str, Any] = {
            "completed_episodes": episode,
            "candidate_phase": source.get("candidate_phase", ""),
            "phase_after_validation": source.get(
                "phase_after_validation", ""
            ),
            "validation_event": source.get("validation_event", ""),
            "exploratory_promotion_event": source.get(
                "exploratory_promotion_event", ""
            ),
            "formal_promotion_event": source.get(
                "formal_promotion_event", ""
            ),
            "window_objective_statistic": source.get(
                "window_objective_statistic", ""
            ),
            "physical_safety_pass": _as_bool(
                source.get("physical_safety_pass", "true")
            ),
        }
        for field in (*HARD_GATE_FIELDS, *OBJECTIVE_FIELDS.values()):
            row[field] = _finite_float(
                source[field], field=field, row_index=row_index
            )
        plot_rows.append(row)
    return plot_rows, summary


def _load_failure_rows(run_directory: str | Path) -> list[dict[str, str]]:
    path = Path(run_directory) / "single_objective_validation_failures.csv"
    return _read_csv(path) if path.is_file() else []


def _phase_markers(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[int | None, int | None]:
    phase = summary.get("training_phase", {})
    if not isinstance(phase, dict):
        phase = {}
    transition = phase.get("phase_transition_episode")
    accepted = phase.get("accepted_quality_episode")
    if transition is None:
        transition = next(
            (
                row["completed_episodes"]
                for row in rows
                if row["validation_event"] == "transition"
            ),
            None,
        )
    if accepted is None:
        accepted = next(
            (
                row["completed_episodes"]
                for row in reversed(rows)
                if row["formal_promotion_event"] == "formal_promoted"
                or row["validation_event"] == "formal_promoted"
            ),
            None,
        )
    return (
        int(transition) if transition is not None else None,
        int(accepted) if accepted is not None else None,
    )


def _exploratory_marker_episodes(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [
        int(row["completed_episodes"])
        for row in rows
        if row["exploratory_promotion_event"] == "exploratory_promoted"
        or row["validation_event"] == "exploratory_promoted"
    ]


def _add_markers(
    axis: plt.Axes,
    transition_episode: int | None,
    accepted_episode: int | None,
    exploratory_episodes: Sequence[int],
) -> None:
    if transition_episode is not None:
        axis.axvline(
            transition_episode,
            color="#9467bd",
            linestyle="--",
            linewidth=1.1,
            label="feasibility -> quality",
        )
    for index, episode in enumerate(exploratory_episodes):
        axis.axvline(
            episode,
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.0,
            label="exploratory window promotion" if index == 0 else None,
        )
    if accepted_episode is not None:
        axis.axvline(
            accepted_episode,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.3,
            label="accepted checkpoint",
        )


def _linear_slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _relative_change(first: float, last: float) -> float | None:
    if math.isclose(first, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    return (last - first) / abs(first)


def analyze_run(
    rows: Sequence[dict[str, Any]],
    objective: str,
    transition_episode: int | None,
    failure_rows: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    """Return transparent diagnostics without smoothing validation values."""

    target_field = OBJECTIVE_FIELDS[objective]
    quality_rows = (
        [
            row
            for row in rows
            if transition_episode is not None
            and row["completed_episodes"] >= transition_episode
        ]
        or list(rows)
    )
    target_values = [float(row[target_field]) for row in quality_rows]
    tail_length = max(2, math.ceil(len(target_values) / 3))
    tail_values = target_values[-tail_length:]
    value_range = max(target_values) - min(target_values)
    tail_slope = _linear_slope(tail_values)
    plateau_candidate = bool(
        len(target_values) >= 6
        and tail_slope is not None
        and abs(tail_slope) * max(1, tail_length - 1)
        <= 0.05 * max(value_range, 1e-12)
    )
    other_diagnostics: dict[str, Any] = {}
    for other, field in OBJECTIVE_FIELDS.items():
        if other == objective:
            continue
        values = [float(row[field]) for row in quality_rows]
        other_diagnostics[other] = {
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
            "relative_change_first_to_last": _relative_change(
                values[0], values[-1]
            ),
        }

    window_statistics = [
        _finite_float(
            row["window_objective_statistic"],
            field="window_objective_statistic",
            row_index=index,
        )
        for index, row in enumerate(quality_rows, start=1)
        if str(row.get("window_objective_statistic", "")).strip()
        not in {"", "None", "nan"}
    ]
    formal_rows = [
        row
        for row in rows
        if row["formal_promotion_event"] == "formal_promoted"
        or row["validation_event"] == "formal_promoted"
    ]
    failure_instance_ids = sorted(
        {
            str(row["instance_id"])
            for row in failure_rows
            if row.get("instance_id")
        }
    )

    return {
        "objective": objective,
        "validation_point_count": len(rows),
        "quality_point_count": len(quality_rows),
        "hard_gates": {
            "all_completion_one": all(
                math.isclose(
                    float(row["completion_rate"]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in quality_rows
            ),
            "all_truncation_zero": all(
                int(row["truncated_count"]) == 0 for row in quality_rows
            ),
            "all_violation_zero": all(
                int(row["schedule_violation_count"]) == 0
                for row in quality_rows
            ),
            "all_physical_safety_pass": all(
                bool(row["physical_safety_pass"]) for row in quality_rows
            ),
        },
        "exploration": {
            "quality_points_at_or_above_95_percent": sum(
                row["completion_rate"] >= QUALITY_EXPLORATION_FLOOR
                for row in quality_rows
            ),
            "quality_point_count": len(quality_rows),
            "exploratory_promotion_count": sum(
                row["exploratory_promotion_event"] == "exploratory_promoted"
                or row["validation_event"] == "exploratory_promoted"
                for row in rows
            ),
            "formal_promotion_count": len(formal_rows),
            "window_statistic_count": len(window_statistics),
            "window_statistic_strict_decrease_count": sum(
                later < earlier
                for earlier, later in zip(
                    window_statistics, window_statistics[1:], strict=False
                )
            ),
            "formal_accepted_current_point_is_100_percent": bool(
                formal_rows
                and math.isclose(
                    formal_rows[-1]["completion_rate"],
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and int(formal_rows[-1]["truncated_count"]) == 0
                and int(formal_rows[-1]["schedule_violation_count"]) == 0
                and bool(formal_rows[-1]["physical_safety_pass"])
            ),
        },
        "failure_details": {
            "failure_row_count": len(failure_rows),
            "instance_ids": failure_instance_ids,
            "truncated_count": sum(
                _as_bool(row.get("truncated", "false")) for row in failure_rows
            ),
            "unfinished_orders": sum(
                _finite_float(
                    row.get("unfinished_orders", 0),
                    field="unfinished_orders",
                    row_index=index,
                )
                for index, row in enumerate(failure_rows, start=1)
            ),
        },
        "target": {
            "field": target_field,
            "first": target_values[0],
            "last": target_values[-1],
            "best": min(target_values),
            "relative_change_first_to_last": _relative_change(
                target_values[0], target_values[-1]
            ),
            "full_slope_per_validation": _linear_slope(target_values),
            "tail_slope_per_validation": tail_slope,
            "non_increasing_step_fraction": (
                sum(
                    later <= earlier
                    for earlier, later in zip(
                        target_values, target_values[1:], strict=False
                    )
                )
                / max(1, len(target_values) - 1)
            ),
            "downward_trend_candidate": bool(
                len(target_values) >= 2
                and target_values[-1] < target_values[0]
                and (_linear_slope(target_values) or 0.0) < 0.0
            ),
            "plateau_candidate": plateau_candidate,
            "plateau_rule": (
                "at least 6 quality points and the last-third fitted change "
                "is at most 5% of the observed quality-phase range"
            ),
        },
        "other_objectives": other_diagnostics,
        "interpretation_note": (
            "Other-objective abnormality needs a domain tolerance; this report "
            "records raw changes and does not invent an acceptance threshold."
        ),
    }


def plot_run(
    run_directory: str | Path,
    objective: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    rows, summary = load_plot_rows(run_directory, objective)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    transition_episode, accepted_episode = _phase_markers(rows, summary)
    exploratory_episodes = _exploratory_marker_episodes(rows)

    data_path = output_path / f"{objective}_convergence_data.csv"
    write_csv(data_path, list(rows))

    episodes = [row["completed_episodes"] for row in rows]
    target_field = OBJECTIVE_FIELDS[objective]
    other_objectives = [name for name in OBJECTIVE_FIELDS if name != objective]
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 12.5), sharex=True)
    panels = axes.ravel()
    series = (
        (
            target_field,
            OBJECTIVE_LABELS[objective],
            "#1f77b4",
        ),
        ("completion_rate", "Completion rate", "#2ca02c"),
        (None, "Truncation / schedule violation", None),
        (
            OBJECTIVE_FIELDS[other_objectives[0]],
            OBJECTIVE_LABELS[other_objectives[0]],
            "#ff7f0e",
        ),
        (
            OBJECTIVE_FIELDS[other_objectives[1]],
            OBJECTIVE_LABELS[other_objectives[1]],
            "#d62728",
        ),
    )
    for index, (field, label, color) in enumerate(series):
        axis = panels[index]
        if field is None:
            axis.plot(
                episodes,
                [row["truncated_count"] for row in rows],
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label="truncated_count",
            )
            axis.plot(
                episodes,
                [row["schedule_violation_count"] for row in rows],
                marker="s",
                linewidth=1.2,
                markersize=3.5,
                label="schedule_violation_count",
            )
            axis.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
        else:
            axis.plot(
                episodes,
                [row[field] for row in rows],
                color=color,
                marker="o",
                linewidth=1.35,
                markersize=3.5,
                label=label,
            )
            if field == "completion_rate":
                axis.axhline(
                    1.0,
                    color="black",
                    linestyle="-",
                    linewidth=0.8,
                    label="target = 1.0",
                )
                axis.axhline(
                    QUALITY_EXPLORATION_FLOOR,
                    color="#ff7f0e",
                    linestyle="--",
                    linewidth=0.9,
                    label="exploration floor = 0.95",
                )
        axis.set_title(label)
        axis.set_ylabel("Raw validation value")
        axis.grid(alpha=0.25)
        _add_markers(
            axis, transition_episode, accepted_episode, exploratory_episodes
        )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=True))
            axis.legend(unique.values(), unique.keys(), fontsize=8)
    panels[5].axis("off")
    for axis in (panels[2], panels[3], panels[4]):
        axis.set_xlabel("Completed episodes")
    figure.suptitle(
        f"E1 single-objective convergence: {objective}", fontsize=15
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    png_path = output_path / f"{objective}_convergence.png"
    pdf_path = output_path / f"{objective}_convergence.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    diagnostics = analyze_run(
        rows,
        objective,
        transition_episode,
        _load_failure_rows(run_directory),
    )
    diagnostics.update(
        {
            "run_directory": str(Path(run_directory).resolve()),
            "phase_transition_episode": transition_episode,
            "accepted_checkpoint_episode": accepted_episode,
            "exploratory_promotion_episodes": exploratory_episodes,
            "plot_data_csv": str(data_path.resolve()),
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
        }
    )
    return diagnostics


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _write_markdown_report(
    output_directory: Path,
    diagnostics: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# E1 single-objective convergence diagnostics",
        "",
        (
            "All values below come from raw `validation_log.csv` points. "
            "The plateau flag is a diagnostic, not an acceptance gate."
        ),
        "",
        "| Objective | ≥95% quality points | Exploratory/formal promotions | "
        "Formal 100% | Target first→last | Downward | Plateau |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in diagnostics:
        gates = report["hard_gates"]
        target = report["target"]
        exploration = report["exploration"]
        lines.append(
            "| {objective} | {qualified}/{quality} | {exploratory}/{formal} | "
            "{formal_100} | {first:.6g}→{last:.6g} ({change}) | {downward} | {plateau} |".format(
                objective=report["objective"],
                qualified=exploration["quality_points_at_or_above_95_percent"],
                quality=exploration["quality_point_count"],
                exploratory=exploration["exploratory_promotion_count"],
                formal=exploration["formal_promotion_count"],
                formal_100=(
                    "yes"
                    if exploration["formal_accepted_current_point_is_100_percent"]
                    else "no"
                ),
                first=target["first"],
                last=target["last"],
                change=_format_percent(target["relative_change_first_to_last"]),
                downward="yes" if target["downward_trend_candidate"] else "no",
                plateau="yes" if target["plateau_candidate"] else "no",
            )
        )
        failures = report["failure_details"]
        lines.append(
            "  Failure rows: {count}; truncated: {truncated}; unfinished orders: "
            "{unfinished:.6g}; instance IDs: {ids}".format(
                count=failures["failure_row_count"],
                truncated=failures["truncated_count"],
                unfinished=failures["unfinished_orders"],
                ids=", ".join(failures["instance_ids"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            (
                "The table uses raw validation points; 95% is an exploratory "
                "floor, while formal promotion remains 100% completion with "
                "zero truncation/violation and physical safety. Other-objective "
                "changes are recorded in "
                "`convergence_diagnostics.json`. Whether a change is abnormal "
                "must be judged against a declared domain tolerance."
            ),
            "",
        ]
    )
    (output_directory / "convergence_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot raw validation convergence for E1 one-hot policies."
    )
    parser.add_argument("--flow-run", type=Path)
    parser.add_argument("--cost-run", type=Path)
    parser.add_argument("--variance-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = {
        "flow": args.flow_run,
        "cost": args.cost_run,
        "variance": args.variance_run,
    }
    selected = {name: path for name, path in runs.items() if path is not None}
    if not selected:
        raise ValueError("at least one run directory must be supplied")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = [
        plot_run(path, objective, args.output_dir)
        for objective, path in selected.items()
    ]
    write_json(
        args.output_dir / "convergence_diagnostics.json",
        {"runs": diagnostics},
    )
    _write_markdown_report(args.output_dir, diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
