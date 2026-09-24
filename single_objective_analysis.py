from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


OBJECTIVES = ("flow", "cost", "variance")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _objective_name(config: dict[str, Any]) -> str:
    weights = tuple(float(value) for value in config["preference"]["quality"]["fixed"])
    mapping = {
        (1.0, 0.0, 0.0): "flow",
        (0.0, 1.0, 0.0): "cost",
        (0.0, 0.0, 1.0): "variance",
    }
    try:
        return mapping[weights]
    except KeyError as error:
        raise ValueError("single-objective analysis requires a one-hot preference") from error


def load_plot_rows(run_directory: str | Path) -> list[dict[str, Any]]:
    run = Path(run_directory)
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(run / "train_log.csv"):
        rows.append(
            {
                "episode": int(raw["episode"]),
                "reward": _number(raw.get("reward")),
                "operation_progress": _number(raw.get("operation_progress")),
                "preference_quality_score": _number(
                    raw.get("preference_quality_score")
                ),
                "flow_time_objective": _number(raw.get("flow_time_objective")),
                "reconfiguration_cost": _number(raw.get("reconfiguration_cost")),
                "worker_load_variance": _number(raw.get("worker_load_variance")),
                "task_succeeded": _bool(raw.get("task_succeeded")),
                "task_failed": _bool(raw.get("task_failed")),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _validation_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "best_episode": None}
    completion = [float(row["completion_rate"]) for row in rows]
    quality = [_number(row.get("preference_balanced_quality_score")) for row in rows]
    return {
        "count": len(rows),
        "episodes": [int(row["episode"]) for row in rows],
        "sampled_completion_rates": completion,
        "preference_balanced_quality_scores": quality,
        "maximum_sampled_completion_rate": max(completion),
        "last_sampled_completion_rate": completion[-1],
    }


def analyze_run(run_directory: str | Path) -> dict[str, Any]:
    run = Path(run_directory)
    config = _read_json(run / "config.json")
    summary = _read_json(run / "summary.json")
    rows = load_plot_rows(run)
    validations = _read_csv(run / "validation_log.csv")
    failed_progress = [
        float(row["operation_progress"])
        for row in rows
        if row["task_failed"] and row["operation_progress"] is not None
    ]
    successful_rows = [row for row in rows if row["task_succeeded"]]
    objective = _objective_name(config)
    objective_field = {
        "flow": "flow_time_objective",
        "cost": "reconfiguration_cost",
        "variance": "worker_load_variance",
    }[objective]
    successful_objectives = [
        float(row[objective_field])
        for row in successful_rows
        if row[objective_field] is not None
    ]
    return {
        "run_directory": str(run.resolve()),
        "objective": objective,
        "reward_mode": config["reward"]["mode"],
        "episode_count": len(rows),
        "training_completion_rate": (
            sum(row["task_succeeded"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_training_reward": _mean(
            [float(row["reward"]) for row in rows if row["reward"] is not None]
        ),
        "mean_successful_objective": _mean(successful_objectives),
        "failure_progress": {
            "count": len(failed_progress),
            "mean": _mean(failed_progress),
            "minimum": min(failed_progress) if failed_progress else None,
            "maximum": max(failed_progress) if failed_progress else None,
        },
        "validation": _validation_summary(validations),
        "checkpoint_selection": summary.get("checkpoint_selection", {}),
        "best_checkpoint": summary.get("best_checkpoint"),
        "last_checkpoint": summary.get("last_checkpoint"),
        "final_sampled": summary.get("final_sampled"),
    }


def plot_run(run_directory: str | Path, output: str | Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    run = Path(run_directory)
    rows = load_plot_rows(run)
    validations = _read_csv(run / "validation_log.csv")
    config = _read_json(run / "config.json")
    objective = _objective_name(config)
    objective_field = {
        "flow": "flow_time_objective",
        "cost": "reconfiguration_cost",
        "variance": "worker_load_variance",
    }[objective]
    destination = Path(output) if output is not None else run / "single_stage_training.png"

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    episodes = [row["episode"] for row in rows]
    axes[0, 0].plot(episodes, [row["reward"] for row in rows], linewidth=0.8)
    axes[0, 0].set(title="Training return", xlabel="Episode", ylabel="ΔP − ΔQ")
    axes[0, 1].plot(
        episodes,
        [row["operation_progress"] for row in rows],
        linewidth=0.8,
        label="terminal progress",
    )
    axes[0, 1].set(title="Terminal operation progress", xlabel="Episode", ylim=(-0.02, 1.02))
    axes[1, 0].scatter(
        episodes,
        [row[objective_field] for row in rows],
        s=6,
        alpha=0.5,
    )
    axes[1, 0].set(title=f"Raw {objective} objective", xlabel="Episode")
    if validations:
        validation_episodes = [int(row["episode"]) for row in validations]
        axes[1, 1].plot(
            validation_episodes,
            [float(row["completion_rate"]) for row in validations],
            marker="o",
            label="sampled",
        )
        axes[1, 1].legend()
    axes[1, 1].set(title="Validation completion", xlabel="Episode", ylim=(-0.02, 1.02))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze single-stage one-hot PPO runs")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="single_objective_analysis.json")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    analyses = [analyze_run(path) for path in args.runs]
    Path(args.output).write_text(
        json.dumps(analyses, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.plots:
        for path in args.runs:
            plot_run(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
