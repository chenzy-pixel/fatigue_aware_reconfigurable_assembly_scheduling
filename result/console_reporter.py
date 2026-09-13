from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEPARATOR = "=" * 60
SUBSEPARATOR = "-" * 60
OBJECTIVE_FIELDS = {
    "flow": "flow_time_objective",
    "cost": "reconfiguration_cost",
    "variance": "worker_load_variance",
}
VALIDATION_OBJECTIVE_FIELDS = {
    name: f"mean_{field}" for name, field in OBJECTIVE_FIELDS.items()
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Sequence[Mapping[str, Any]], field_name: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := _finite(row.get(field_name))) is not None
    ]
    return sum(values) / len(values) if values else None


def _metric(value: object) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.3f}"


def _short(value: object) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.3g}"


def _percent(value: object) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{100.0 * number:.1f}%"


def _scientific(value: object) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    return f"{number:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def _pass_fail(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"


def _elapsed(value: float) -> str:
    seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _interval(name: str, value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    estimate = _finite(value.get("estimate"))
    lower = _finite(value.get("lower"))
    upper = _finite(value.get("upper"))
    if estimate is None or lower is None or upper is None:
        return None
    return (
        f"{name} {_short(estimate)} "
        f"| 95% CI [{_short(lower)}, {_short(upper)}]"
    )


@dataclass
class TrainingConsoleReporter:
    """Human-readable console view over the persisted training records."""

    config: Mapping[str, Any]
    total_episodes: int
    parallel_envs: int
    validation_instance_limit: int | None
    objective_name: str | None
    started_at: float
    writer: Callable[[str], None] = print
    _phase_headers: set[str] = field(default_factory=set, init=False)

    @property
    def is_single_objective(self) -> bool:
        return self.objective_name in OBJECTIVE_FIELDS

    @property
    def run_label(self) -> str:
        if self.is_single_objective:
            return f"single_{self.objective_name}"
        return str(self.config.get("experiment_name", "training"))

    @property
    def preference_count(self) -> int:
        settings = self.config["training"]["two_stage"].get(
            "pareto_promotion", {}
        )
        return int(settings.get("preference_count", 66))

    @property
    def audit_instance_limit(self) -> int:
        two_stage = self.config["training"]["two_stage"]
        key = (
            "single_objective_promotion"
            if self.is_single_objective
            else "pareto_promotion"
        )
        return int(two_stage.get(key, {}).get("audit_instance_limit", 200))

    def _block(self, *lines: str) -> None:
        for line in lines:
            self.writer(line)
        self.writer("")

    def start_run(self) -> None:
        validation = (
            "all"
            if self.validation_instance_limit is None
            else str(self.validation_instance_limit)
        )
        audit = str(self.audit_instance_limit)
        if not self.is_single_objective:
            validation = f"{validation}×{self.preference_count}"
            audit = f"{audit}×{self.preference_count}"
        self._block(
            SEPARATOR,
            (
                f"[RUN] {self.run_label} | seed={int(self.config['seed'])} "
                f"| envs={self.parallel_envs} | {self.config['device']}"
            ),
            (
                f"episodes={self.total_episodes} | val={validation} "
                f"| audit={audit} "
                f"| lr={_scientific(self.config['ppo']['learning_rate'])}"
            ),
            SEPARATOR,
        )
        self.start_phase("feasibility")

    def start_phase(self, phase: str) -> None:
        if phase in self._phase_headers:
            return
        self._phase_headers.add(phase)
        if phase == "feasibility":
            self._block("[完成率阶段] FEASIBILITY", SUBSEPARATOR)
        elif phase == "quality":
            self._block("[质量阶段] QUALITY", SUBSEPARATOR)
        else:
            raise ValueError(f"unknown training phase {phase!r}")

    def training_update(
        self,
        update: Mapping[str, Any],
        episode_rows: Sequence[Mapping[str, Any]],
        *,
        primary_value: float | None = None,
    ) -> None:
        completed = sum(
            bool(row.get("terminated")) and not bool(row.get("truncated"))
            for row in episode_rows
        )
        completion_rate = completed / len(episode_rows) if episode_rows else 0.0
        primary_name = self.objective_name if self.is_single_objective else "quality"
        primary_field = OBJECTIVE_FIELDS.get(
            str(self.objective_name), "quality_score"
        )
        displayed_primary = (
            primary_value
            if primary_value is not None
            else _mean(episode_rows, primary_field)
        )
        self._block(
            (
                f"[train] ep {int(update['episode_end']) + 1}/"
                f"{self.total_episodes} | upd {int(update['update_id'])}"
            ),
            (
                f"complete {_percent(completion_rate)} "
                f"| reward {_metric(_mean(episode_rows, 'reward'))}"
            ),
            f"{primary_name} {_metric(displayed_primary)}",
            (
                f"p_loss {_short(update.get('policy_loss'))} "
                f"| v_loss {_short(update.get('value_loss'))} "
                f"| ent {_short(update.get('entropy'))} "
                f"| kl {_short(update.get('approx_kl'))} "
                f"| lr {_scientific(update.get('learning_rate'))}"
            ),
        )

    def validation(
        self,
        row: Mapping[str, Any],
        *,
        status: str,
        phase_state: Mapping[str, Any],
        pareto_result: Mapping[str, Any] | None = None,
    ) -> None:
        if pareto_result is not None:
            self._pareto_validation(row, status=status, result=pareto_result)
            return
        objective_order = list(OBJECTIVE_FIELDS)
        if self.is_single_objective:
            objective_order.remove(self.objective_name)  # type: ignore[arg-type]
            objective_order.insert(0, self.objective_name)  # type: ignore[arg-type]
        lines = [
            f"[val] ep {int(row['episode'])} | n={int(row['instance_count'])}",
            f"complete {_percent(row.get('completion_rate'))}",
            " | ".join(
                f"{name} {_metric(row.get(VALIDATION_OBJECTIVE_FIELDS[name]))}"
                for name in objective_order
            ),
        ]
        if str(row.get("candidate_phase")) == "feasibility":
            lines.append(
                "consecutive_success "
                f"{int(row.get('consecutive_completion_successes', 0))}/"
                f"{int(phase_state.get('consecutive_validations_required', 0))}"
            )
        elif self.is_single_objective:
            values = phase_state.get("single_objective_window_values", [])
            count = len(values) if isinstance(values, list) else 0
            size = int(phase_state.get("single_objective_window_size", 0))
            lines.append(f"window {count}/{size}")
            window_median = row.get("window_objective_statistic")
            best_median = phase_state.get("single_objective_candidate_anchor_value")
            best_episode = phase_state.get("single_objective_candidate_episode")
            if window_median is not None or best_median is not None:
                best = _metric(best_median)
                if best_episode is not None:
                    best += f" @ ep{int(best_episode)}"
                lines.append(
                    f"window_med {_metric(window_median)} | best_med {best}"
                )
        lines.append(f"status={status}")
        self._block(*lines)

    def _pareto_validation(
        self,
        row: Mapping[str, Any],
        *,
        status: str,
        result: Mapping[str, Any],
    ) -> None:
        safety = result.get("safety", {})
        endpoints = result.get("endpoints", {})
        endpoint_means = (
            endpoints.get("means", {}) if isinstance(endpoints, Mapping) else {}
        )
        lines = [
            (
                f"[val] ep {int(row['episode'])} "
                f"| n={int(result.get('instance_count', row['instance_count']))} "
                f"× prefs={int(result.get('preference_count', self.preference_count))}"
            ),
            (
                "min_complete "
                f"{_percent(safety.get('minimum_completion_rate') if isinstance(safety, Mapping) else None)}"
            ),
            " | ".join(
                f"{name} {_metric(endpoint_means.get(name) if isinstance(endpoint_means, Mapping) else None)}"
                for name in OBJECTIVE_FIELDS
            ),
        ]
        for name, field_name in (
            ("primary_delta", "primary_delta"),
            ("hv_delta", "hypervolume_delta"),
        ):
            formatted = _interval(name, result.get(field_name))
            if formatted is not None:
                lines.append(formatted)
        lines.append(
            f"gates safety={_pass_fail(result.get('safety_gate_pass'))} "
            f"| endpoints={_pass_fail(result.get('endpoint_gate_pass'))}"
        )
        lines.append(f"status={status}")
        self._block(*lines)

    def transition(self, checkpoint: str | Path) -> None:
        self._block(
            SEPARATOR,
            "[PHASE] 完成率阶段 → 质量阶段",
            f"transition checkpoint={Path(checkpoint).name}",
            SEPARATOR,
        )
        self.start_phase("quality")

    def single_objective_audit(
        self,
        row: Mapping[str, Any],
        *,
        status: str,
        phase_state: Mapping[str, Any],
    ) -> None:
        instance_count = int(row.get("audit_instance_count", 0))
        failed = int(row.get("audit_failed_instance_count", 0))
        objective = str(row.get("single_objective_target", self.objective_name))
        self._block(
            f"[AUDIT] ep {int(row['episode'])} | n={instance_count}",
            (
                f"complete {_percent(row.get('audit_completion_rate'))} "
                f"| failed {failed}/{instance_count}"
            ),
            (
                f"violations {int(row.get('audit_schedule_violation_count', 0))} "
                f"| safety {_pass_fail(row.get('audit_physical_safety_pass'))}"
            ),
            f"{objective} {_metric(row.get('audit_single_objective_value'))}",
            (
                "target "
                f"complete>={_percent(phase_state.get('single_objective_audit_completion_target'))} "
                f"| failed<={int(phase_state.get('single_objective_audit_max_failed_instances', 0))} "
                f"| violations=0 | safety=PASS"
            ),
            f"status={status}",
        )

    def pareto_audit(
        self,
        result: Mapping[str, Any],
        *,
        episode: int,
        status: str,
    ) -> None:
        safety = result.get("safety", {})
        endpoints = result.get("endpoints", {})
        endpoint_means = (
            endpoints.get("means", {}) if isinstance(endpoints, Mapping) else {}
        )
        lines = [
            (
                f"[AUDIT] ep {episode} | n={int(result.get('instance_count', 0))} "
                f"× prefs={int(result.get('preference_count', self.preference_count))}"
            ),
            (
                "min_complete "
                f"{_percent(safety.get('minimum_completion_rate') if isinstance(safety, Mapping) else None)} "
                f"| failed {int(safety.get('failed_instance_count', 0)) if isinstance(safety, Mapping) else 0}"
            ),
            (
                f"violations {int(safety.get('schedule_violation_count', 0)) if isinstance(safety, Mapping) else 0} "
                f"| safety {_pass_fail(result.get('safety_gate_pass'))} "
                f"| endpoints {_pass_fail(result.get('endpoint_gate_pass'))}"
            ),
            " | ".join(
                f"{name} {_metric(endpoint_means.get(name) if isinstance(endpoint_means, Mapping) else None)}"
                for name in OBJECTIVE_FIELDS
            ),
        ]
        for name, field_name in (
            ("primary_delta", "primary_delta"),
            ("hv_delta", "hypervolume_delta"),
        ):
            formatted = _interval(name, result.get(field_name))
            if formatted is not None:
                lines.append(formatted)
        lines.append(f"status={status}")
        self._block(*lines)

    def accepted(
        self,
        *,
        episode: int,
        checkpoint: str | Path,
        phase_state: Mapping[str, Any],
    ) -> None:
        if self.is_single_objective:
            headline = (
                f"[ACCEPTED] {self.objective_name}="
                f"{_metric(phase_state.get('accepted_single_objective_audit_value'))} "
                f"| ep={episode}"
            )
            detail = (
                "window_med="
                f"{_metric(phase_state.get('accepted_single_objective_window_value'))}"
            )
        else:
            headline = f"[ACCEPTED] pareto | ep={episode}"
            detail = "formal_audit=PASS"
        self._block(
            headline,
            detail,
            f"checkpoint={Path(checkpoint).name}",
        )

    def done(
        self,
        *,
        phase_state: Mapping[str, Any],
        run_directory: str | Path,
        accepted_checkpoint: str | Path | None,
        last_checkpoint: str | Path,
        elapsed_seconds: float | None = None,
    ) -> None:
        accepted = bool(phase_state.get("is_formally_accepted"))
        elapsed_seconds = (
            time.perf_counter() - self.started_at
            if elapsed_seconds is None
            else elapsed_seconds
        )
        lines = [
            SEPARATOR,
            f"[DONE] {self.run_label} | seed={int(self.config['seed'])}",
            f"time={_elapsed(elapsed_seconds)}",
            f"status={phase_state.get('formal_training_status', 'unknown')}",
        ]
        if accepted and accepted_checkpoint is not None:
            episode = phase_state.get("accepted_quality_episode")
            if self.is_single_objective:
                lines.append(
                    f"best={_metric(phase_state.get('accepted_single_objective_audit_value'))} "
                    f"@ ep{int(episode)}"
                )
            else:
                lines.append(f"best=pareto @ ep{int(episode)}")
            lines.append(f"checkpoint={Path(accepted_checkpoint).name}")
            lines.append("verified=true")
        else:
            lines.extend(
                (
                    "best=none",
                    "checkpoint=none",
                    f"last={Path(last_checkpoint).name}",
                )
            )
        lines.extend((f"artifacts={Path(run_directory)}", SEPARATOR))
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
