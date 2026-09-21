from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEPARATOR = "-" * 72


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(name))) is not None]
    return sum(values) / len(values) if values else None


def _metric(value: object, digits: int = 3) -> str:
    number = _finite(value)
    return "inf" if value is not None and math.isinf(float(value)) else (
        "n/a" if number is None else f"{number:.{digits}f}"
    )


def _percent(value: object) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{100.0 * number:.1f}%"


def _elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@dataclass
class TrainingConsoleReporter:
    config: Mapping[str, Any]
    total_episodes: int
    parallel_envs: int
    validation_instance_limit: int
    objective_name: str | None
    started_at: float
    writer: Callable[[str], None] = print

    @property
    def run_label(self) -> str:
        return (
            f"single_{self.objective_name}"
            if self.objective_name is not None
            else str(self.config.get("experiment_name", "universal"))
        )

    @property
    def preference_count(self) -> int:
        return 1 if self.objective_name is not None else 66

    def _block(self, *lines: str) -> None:
        for line in lines:
            self.writer(line)

    def start_run(self) -> None:
        formal = self.config["training"]["formal_evaluation"]
        self._block(
            SEPARATOR,
            f"[RUN] {self.run_label} | seed={self.config['seed']} | "
            f"envs={self.parallel_envs} | {self.config['device']}",
            f"episodes={self.total_episodes} | val={self.validation_instance_limit} "
            f"× repeats={int(formal['validation_repeats'])} "
            f"× prefs={self.preference_count} | sampled T={float(formal['temperature']):.1f}",
            f"reward=ΔP-ΔQ | gamma={float(self.config['ppo']['gamma']):.1f} "
            f"| lr={float(self.config['ppo']['learning_rate']):.1e}",
            SEPARATOR,
        )

    def training_update(
        self,
        update_row: Mapping[str, Any],
        episode_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        end = int(update_row["episode_end"]) + 1
        self._block(
            f"[train] ep {end}/{self.total_episodes} | upd {int(update_row['update_id'])} "
            f"| complete {_percent(_mean(episode_rows, 'task_succeeded'))} "
            f"| reward {_metric(_mean(episode_rows, 'reward'))} "
            f"| progress {_metric(_mean(episode_rows, 'operation_progress'))} "
            f"| quality {_metric(_mean(episode_rows, 'preference_quality_score'))}",
            f"        p_loss {_metric(update_row.get('policy_loss'), 4)} "
            f"| v_loss {_metric(update_row.get('value_loss'))} "
            f"| ent {_metric(update_row.get('entropy'), 2)} "
            f"| kl {_metric(update_row.get('approx_kl'), 4)} "
            f"| lr={float(update_row.get('learning_rate', 0.0)):.1e}",
        )

    def validation(
        self,
        row: Mapping[str, Any],
        *,
        selector_state: Mapping[str, Any],
    ) -> None:
        self._block(
            f"[val] ep {int(row['episode'])} | sampled complete "
            f"{_percent(row.get('completion_rate'))} | quality "
            f"{_metric(row.get('preference_balanced_quality_score'), 6)} "
            f"| progress {_metric(row.get('mean_operation_progress'))}",
            f"      greedy complete {_percent(row.get('greedy_completion_rate'))} "
            f"| safety={'PASS' if row.get('physical_safety_pass') else 'FAIL'} "
            f"| event={row.get('checkpoint_event', 'n/a')} "
            f"| best_ep={selector_state.get('best_episode')}",
        )

    def done(
        self,
        *,
        selector_state: Mapping[str, Any],
        run_directory: str | Path,
        best_checkpoint: str | Path | None,
        last_checkpoint: str | Path,
        elapsed_seconds: float | None = None,
    ) -> None:
        elapsed = (
            time.perf_counter() - self.started_at
            if elapsed_seconds is None
            else float(elapsed_seconds)
        )
        lines = [
            SEPARATOR,
            f"[DONE] status={'best_selected' if best_checkpoint else 'no_safe_best'} "
            f"| time={_elapsed(elapsed)}",
        ]
        if best_checkpoint is None:
            lines.extend(("best=none", "checkpoint=none"))
        else:
            lines.extend(
                (
                    f"best complete={_percent(selector_state.get('best_completion_rate'))} "
                    f"| quality={_metric(selector_state.get('best_preference_balanced_quality_score'), 6)} "
                    f"| ep={selector_state.get('best_episode')}",
                    f"checkpoint={Path(best_checkpoint).name}",
                )
            )
        lines.extend(
            (
                f"last={Path(last_checkpoint).name}",
                f"artifacts={Path(run_directory)}",
                SEPARATOR,
            )
        )
        self._block(*lines)


def report_training_failure(
    error: BaseException,
    *,
    run_directory: str | Path | None,
    exit_code: int,
    writer: Callable[[str], None] = print,
) -> None:
    writer(SEPARATOR)
    writer(f"[FAILED] {type(error).__name__}")
    writer(f"message={error}")
    writer(f"exit_code={int(exit_code)}")
    writer(f"artifacts={run_directory if run_directory is not None else 'unavailable'}")
    writer(SEPARATOR)
    writer("")
