from __future__ import annotations

import math
import time
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.formal_preferences import formal_preferences


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
        return len(formal_preferences(self.config, "validation"))

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
            f"formal aggregation={formal.get('sample_aggregation', 'per_sample_v1')}",
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
            f"[val] ep {int(row['episode'])} | {row.get('sample_aggregation', 'sampled')} complete "
            f"{_percent(row.get('completion_rate'))} | quality "
            f"{_metric(row.get('preference_balanced_quality_score'), 6)} "
            f"| progress {_metric(row.get('mean_operation_progress'))}",
            f"      safety={'PASS' if row.get('physical_safety_pass') else 'FAIL'} "
            f"| event={row.get('checkpoint_event', 'n/a')} "
            f"| best_ep={selector_state.get('best_episode')}",
        )
        if "request_count" in row:
            self._block(
                f"      requests={row['request_count']} | deliveries={row['delivery_count']} "
                f"| raw complete={_percent(row.get('raw_overall_completion_rate'))} "
                f"| raw Q={_metric(row.get('raw_preference_balanced_quality_score'), 6)}")

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


class ExperimentConsoleReporter(TrainingConsoleReporter):
    """The established console layout with execution-event phase updates."""

    def __init__(self, config, budget, method, workers, validation_instances, validation_preferences,
                 *, started_at, writer=None):
        super().__init__(config=config,
                         total_episodes=budget['budget_amount'] if budget['budget_unit'] == 'episodes' else 0,
                         parallel_envs=workers, validation_instance_limit=validation_instances,
                         objective_name=None, started_at=started_at,
                         writer=writer if writer is not None else lambda line: print(line, flush=True))
        self.budget, self.method = budget, method
        self.validation_preferences = validation_preferences

    @property
    def run_label(self):
        return str(self.config.get('comparison', {}).get('arm', self.method))

    @property
    def epoch_total(self):
        return math.ceil(self.total_episodes / self.budget['episodes_per_update']) if self.total_episodes else None

    @property
    def validation_repeats(self):
        return 1 if self.method == 'rule_ddqn' else 3

    def start_run(self):
        delivery = 'greedy_single_v1' if self.method == 'rule_ddqn' else 'best_of_3_v1'
        schedule = (f'every {self.budget["validation_interval_epochs"]} epochs + final'
                    if self.total_episodes else 'step fifths + final')
        self._block(SEPARATOR,
                    f'[RUN] {self.run_label} | method={self.method} | seed={self.config["seed"]} '
                    f'| envs={self.parallel_envs} | {self.config["device"]}',
                    f'{self.budget["budget_unit"]}={self.budget["budget_amount"]} '
                    f'| epochs={self.epoch_total or "variable"} | batch={self.budget["episodes_per_update"]}',
                    f'val={self.validation_instance_limit} × repeats={self.validation_repeats} '
                    f'× prefs={self.validation_preferences} | envs={self.config["training"].get("validation_parallel_envs", 20)} '
                    f'| schedule={schedule}',
                    f'formal aggregation={delivery} | lr={float(self.config["ppo"]["learning_rate"]):.1e}',
                    SEPARATOR)

    @contextmanager
    def phase(self, label, *, epoch=None, total=None, unit='episodes'):
        """Print once at phase entry and once on successful or failed exit."""
        started = time.perf_counter()
        state = {'completed': 0, 'active': 0, 'environment_steps': 0}

        def update(values):
            state.update({k: values[k] for k in state if k in values})

        def display(status):
            snapshot = dict(state)
            parts = [f'[{label}] {status}']
            if epoch is not None:
                parts.append(f'epoch {epoch}/{self.epoch_total or "?"}')
            if total is not None:
                parts.append(f'{unit} {snapshot["completed"]}/{total}')
            if label in ('collect', 'val'):
                parts.extend((f'active={snapshot["active"]}', f'steps={snapshot["environment_steps"]}'))
            parts.append(f'elapsed={_elapsed(time.perf_counter()-started)}')
            self._block(' | '.join(parts))

        display('start')
        try:
            yield update
        except BaseException:
            display('failed')
            raise
        else:
            display('done')

    def epoch_update(self, row, episodes):
        metrics = [{**(e.get('metrics') or {}), 'reward': e.get('reward_sum')} for e in episodes]
        adapted = {**row, **row.get('losses', {}), 'episode_end': row['episodes_started']-1}
        if self.method != 'rule_ddqn' and self.total_episodes:
            super().training_update(adapted, metrics)
        else:
            details = (f'        td_loss {_metric(row.get("losses", {}).get("td_loss"), 4)} '
                       f'| epsilon {_metric(row.get("exploration_epsilon"))}' if self.method == 'rule_ddqn'
                       else f'        p_loss {_metric(adapted.get("policy_loss"), 4)} '
                            f'| v_loss {_metric(adapted.get("value_loss"))}')
            self._block(f'[train] ep {row["episodes_started"]}/{self.total_episodes or "?"} '
                        f'| upd {row["update_id"]} | complete {_percent(_mean(metrics, "task_succeeded"))} '
                        f'| reward {_metric(_mean(metrics, "reward"))} '
                        f'| progress {_metric(_mean(metrics, "operation_progress"))} '
                        f'| quality {_metric(_mean(metrics, "preference_quality_score"))}', details)
        self._block(f'        epoch {row["epoch"]}/{self.epoch_total or "?"} '
                    f'| steps={row["training_steps"]} | samples={row["episodes_started"]} '
                    f'| collect={_elapsed(row["collect_seconds"])} | update={_elapsed(row["update_seconds"])} '
                    f'| elapsed={_elapsed(time.perf_counter()-self.started_at)}')

    def validation_row(self, result, episode):
        raw = [r for delivery in result['deliveries'] for r in delivery['raw']]
        rows = result['rows']
        return {'episode': episode,
                'sample_aggregation': 'greedy_single_v1' if self.method == 'rule_ddqn' else 'best_of_3_v1',
                'completion_rate': _mean(rows, 'success'),
                'preference_balanced_quality_score': result['summary'].get('quality'),
                'mean_operation_progress': _mean([r['metrics'] for r in raw], 'operation_progress'),
                'physical_safety_pass': result['summary']['safe'],
                'request_count': len(raw), 'delivery_count': len(rows),
                'raw_overall_completion_rate': _mean(raw, 'success'),
                'raw_preference_balanced_quality_score': _mean(raw, 'quality')}

    def checkpoint_saved(self, epoch, event, *, last_checkpoint, best_checkpoint=None):
        line = f'[checkpoint] epoch {epoch} | event={event} | last={last_checkpoint}'
        if best_checkpoint is not None:
            line += f' | best={best_checkpoint}'
        self._block(line)
