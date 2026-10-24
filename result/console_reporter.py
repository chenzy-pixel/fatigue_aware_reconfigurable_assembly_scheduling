"""Console updates emitted by training execution events."""
from __future__ import annotations
import math
import time
from collections.abc import Callable, Mapping, Sequence
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

class ExperimentConsoleReporter:
    """Compact run, training, validation and completion summaries."""

    def __init__(self, config, budget, method, workers, validation_instances, validation_preferences,
                 *, started_at, writer=None):
        self.config, self.total_episodes = config, budget['budget_amount']
        self.parallel_envs, self.validation_instance_limit = workers, validation_instances
        self.started_at = started_at
        self.writer = writer if writer is not None else lambda line: print(line, flush=True)
        self.budget, self.method = budget, method
        self.validation_preferences = validation_preferences

    @property
    def run_label(self):
        return str(self.config.get('comparison', {}).get('arm', self.method))


    @property
    def validation_repeats(self):
        return 1 if self.method == 'rule_ddqn' else 3

    def start_run(self):
        formal = self.config['training']['formal_evaluation']
        delivery = 'greedy_single_v1' if self.method == 'rule_ddqn' else 'best_of_3_v1'
        decode = 'greedy' if self.method == 'rule_ddqn' else f'sampled T={float(formal["temperature"]):.1f}'
        self._block(
            SEPARATOR,
            f'[RUN] {self.run_label} | seed={self.config["seed"]} | envs={self.parallel_envs} | {self.config["device"]}',
            f'episodes={self.total_episodes} | val={self.validation_instance_limit} '
            f'× repeats={self.validation_repeats} × prefs={self.validation_preferences} | {decode}',
            f'reward=ΔP-ΔQ | gamma={float(self.config["ppo"]["gamma"]):.1f} '
            f'| lr={float(self.config["ppo"]["learning_rate"]):.1e}',
            f'formal aggregation={delivery}',
            SEPARATOR,
        )

    def epoch_update(self, row, episodes):
        metrics = [{**(e.get('metrics') or {}), 'reward': e.get('reward_sum')} for e in episodes]
        adapted = {**row, **row.get('losses', {}), 'episode_end': row['episodes_started']-1}
        if self.method != 'rule_ddqn':
            self.training_update(adapted, metrics)
        else:
            details = (f'        td_loss {_metric(row.get("losses", {}).get("td_loss"), 4)} '
                       f'| epsilon {_metric(row.get("exploration_epsilon"))}' if self.method == 'rule_ddqn'
                       else f'        p_loss {_metric(adapted.get("policy_loss"), 4)} '
                            f'| v_loss {_metric(adapted.get("value_loss"))}')
            self._block(f'[train] ep {row["episodes_started"]}/{self.total_episodes} '
                        f'| upd {row["update_id"]} | complete {_percent(_mean(metrics, "task_succeeded"))} '
                        f'| reward {_metric(_mean(metrics, "reward"))} '
                        f'| progress {_metric(_mean(metrics, "operation_progress"))} '
                        f'| quality {_metric(_mean(metrics, "preference_quality_score"))}', details)

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


    def _block(self, *lines: str) -> None:
        for line in lines:
            self.writer(line)

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
