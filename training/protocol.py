"""V8 feasibility, specialist, and preference-conditioned promotion protocols."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


SINGLE_OBJECTIVE_PROMOTION_MODE = "single_objective_guarded_v1"
PARETO_PROMOTION_MODE = "preference_conditioned_pareto_v8"
OBJECTIVES = ("flow", "cost", "variance")


def _objective_name(config: dict) -> str:
    preference = config.get("preference", {})
    quality = preference.get("quality", {}) if isinstance(preference, dict) else {}
    fixed = quality.get("fixed") if isinstance(quality, dict) else None
    if isinstance(fixed, dict):
        weights = {name: float(fixed.get(name, 0.0)) for name in OBJECTIVES}
    elif isinstance(fixed, (list, tuple)) and len(fixed) == len(OBJECTIVES):
        weights = {
            name: float(fixed[index]) for index, name in enumerate(OBJECTIVES)
        }
    else:
        weights = {}
    selected = [
        name
        for name in OBJECTIVES
        if math.isclose(float(weights.get(name, 0.0)), 1.0, abs_tol=1e-12)
    ]
    zeros = all(
        math.isclose(float(weights.get(name, 0.0)), 0.0, abs_tol=1e-12)
        for name in OBJECTIVES
        if name not in selected
    )
    if len(selected) != 1 or not zeros:
        raise ValueError(
            "single-objective promotion requires strictly one-hot "
            "preference.quality.fixed"
        )
    return selected[0]


@dataclass
class TrainingPhaseController:
    enabled: bool = True
    completion_target: float = 1.0
    consecutive_required: int = 3
    quality_completion_floor: float = 0.95
    quality_checkpoint_promotion: str = SINGLE_OBJECTIVE_PROMOTION_MODE
    phase: str = "feasibility"
    consecutive_successes: int = 0
    phase_transition_episode: int | None = None
    accepted_quality_updates: int = 0
    rejected_quality_updates: int = 0
    not_promoted_quality_updates: int = 0
    accepted_quality_score: tuple[float, float, float, float] | None = None
    accepted_quality_episode: int | None = None
    quality_promotion_constraints: dict[str, float] = field(default_factory=dict)
    last_promotion_diagnostics: dict[str, object] = field(default_factory=dict)
    single_objective_name: str | None = None
    accepted_single_objective_value: float | None = None
    single_objective_window_size: int = 5
    single_objective_window_statistic: str = "median"
    single_objective_rollback_below_floor_consecutive: int = 2
    single_objective_candidate_improvement_epsilon: float = 1e-9
    single_objective_audit_instance_limit: int = 200
    single_objective_audit_instance_offset: int = 0
    single_objective_audit_completion_target: float = 0.98
    single_objective_audit_max_failed_instances: int = 4
    single_objective_audit_schedule_violation_target: int = 0
    single_objective_audit_physical_safety_required: bool = True
    single_objective_window_values: list[float] = field(default_factory=list)
    single_objective_window_episodes: list[int] = field(default_factory=list)
    single_objective_candidate_anchor_value: float | None = None
    single_objective_candidate_episode: int | None = None
    single_objective_audit_count: int = 0
    accepted_single_objective_failed_instances: int | None = None
    accepted_single_objective_window_value: float | None = None
    accepted_single_objective_audit_value: float | None = None
    last_single_objective_audit_diagnostics: dict[str, object] = field(
        default_factory=dict
    )
    last_pareto_promotion: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "TrainingPhaseController":
        if str(config["reward"].get("mode")) != "hierarchical_constrained_v1":
            raise ValueError("only hierarchical_constrained_v1 is supported")
        if not math.isclose(float(config["ppo"]["gamma"]), 1.0):
            raise ValueError("V8 telescoping quality training requires ppo.gamma = 1.0")
        settings = config["training"]["two_stage"]
        promotion = str(settings.get("quality_checkpoint_promotion", ""))
        if promotion not in {
            SINGLE_OBJECTIVE_PROMOTION_MODE,
            PARETO_PROMOTION_MODE,
        }:
            raise ValueError(
                "unsupported checkpoint promotion protocol"
            )
        target = float(settings["completion_target"])
        required = int(settings["consecutive_validations"])
        floor = float(settings["quality_completion_floor"])
        if not 0.0 <= target <= 1.0 or not 0.0 <= floor <= 1.0:
            raise ValueError("completion targets must be in [0, 1]")
        if required < 1:
            raise ValueError(
                "consecutive_validations must be positive"
            )
        if promotion == PARETO_PROMOTION_MODE:
            return cls(
                completion_target=target,
                consecutive_required=required,
                quality_completion_floor=floor,
                quality_checkpoint_promotion=promotion,
            )
        if not bool(settings["quality_validate_every_update"]):
            raise ValueError("specialist quality training validates every update")
        raw = settings.get("single_objective_promotion")
        if not isinstance(raw, dict):
            raise ValueError("single_objective_promotion must be an object")
        window_size = int(raw.get("window_size", 0))
        window_statistic = str(raw.get("window_statistic", "")).lower()
        epsilon = float(raw.get("candidate_improvement_epsilon", math.nan))
        rollback_count = int(raw.get("rollback_below_floor_consecutive", 0))
        audit_limit = int(raw.get("audit_instance_limit", 0))
        audit_offset = int(raw.get("audit_instance_offset", 0))
        audit_completion = float(raw.get("audit_completion_target", math.nan))
        audit_max_failed = int(raw.get("audit_max_failed_instances", -1))
        audit_violations = int(raw.get("audit_schedule_violation_target", -1))
        audit_safety = bool(raw.get("audit_physical_safety_required", False))
        if window_size < 1 or window_statistic != "median":
            raise ValueError("single-objective window must be a positive median window")
        if rollback_count < 1 or not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("single-objective rollback/epsilon settings are invalid")
        if (
            audit_limit != 200
            or audit_offset < 0
            or not math.isclose(audit_completion, 0.98, abs_tol=1e-12)
            or audit_max_failed != 4
            or audit_violations != 0
            or not audit_safety
        ):
            raise ValueError(
                "protocol v4 audit requires 200 instances, completion 0.98, "
                "at most 4 failures, zero violations, and physical safety"
            )
        return cls(
            completion_target=target,
            consecutive_required=required,
            quality_completion_floor=floor,
            single_objective_name=_objective_name(config),
            single_objective_window_size=window_size,
            single_objective_window_statistic=window_statistic,
            single_objective_rollback_below_floor_consecutive=rollback_count,
            single_objective_candidate_improvement_epsilon=epsilon,
            single_objective_audit_instance_limit=audit_limit,
            single_objective_audit_instance_offset=audit_offset,
            single_objective_audit_completion_target=audit_completion,
            single_objective_audit_max_failed_instances=audit_max_failed,
            single_objective_audit_schedule_violation_target=audit_violations,
            single_objective_audit_physical_safety_required=audit_safety,
        )

    def should_validate(self, regular_due: bool) -> bool:
        if self.quality_checkpoint_promotion == PARETO_PROMOTION_MODE:
            return bool(regular_due)
        return self.phase == "quality" or regular_due

    def observe_validation(
        self,
        completion_rate: float,
        *,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None = None,
        truncated_count: int = 0,
        schedule_violation_count: int = 0,
        physical_safety_pass: bool = True,
    ) -> str:
        rate = float(completion_rate)
        truncations = int(truncated_count)
        violations = int(schedule_violation_count)
        self.last_promotion_diagnostics = {}
        if self.phase == "feasibility":
            passed = bool(
                rate >= self.completion_target
                and truncations == 0
                and violations == 0
                and physical_safety_pass
            )
            self.consecutive_successes = self.consecutive_successes + 1 if passed else 0
            if self.consecutive_successes < self.consecutive_required:
                return "feasibility"
            self.phase = "quality"
            self.phase_transition_episode = int(completed_episodes)
            self.last_promotion_diagnostics = self._diagnostics(
                "transition", "feasibility_phase_complete", rate, truncations,
                violations, physical_safety_pass, None, False, None,
            )
            return "transition"
        if self.quality_checkpoint_promotion == PARETO_PROMOTION_MODE:
            self.last_promotion_diagnostics = {
                "promotion_mode": PARETO_PROMOTION_MODE,
                "promotion_event": "pareto_evaluation_required",
            }
            return "pareto_evaluation_required"
        return self._observe_single_objective_candidate(
            completion_rate=rate,
            completed_episodes=completed_episodes,
            score=score,
            truncated_count=truncations,
            schedule_violation_count=violations,
            physical_safety_pass=physical_safety_pass,
        )

    def _observe_single_objective_candidate(
        self,
        *,
        completion_rate: float,
        completed_episodes: int,
        score: tuple[float, float, float, float] | None,
        truncated_count: int,
        schedule_violation_count: int,
        physical_safety_pass: bool,
    ) -> str:
        finite = bool(score is not None and len(score) == 4 and math.isfinite(float(score[1])))
        candidate = float(score[1]) if finite and score is not None else None
        completion_pass = completion_rate >= self.quality_completion_floor
        violation_pass = schedule_violation_count == 0
        physical_pass = bool(physical_safety_pass)
        previous_anchor = self.single_objective_candidate_anchor_value
        eligible = completion_pass and violation_pass and physical_pass and finite
        window_stat: float | None = None
        audit_required = False
        if not eligible:
            self.reset_single_objective_window()
            self.rejected_quality_updates += 1
            reason = (
                "completion_below_floor" if not completion_pass else
                "schedule_violation_nonzero" if not violation_pass else
                "physical_safety_failed" if not physical_pass else
                "missing_or_non_finite_objective"
            )
            event = "rejected"
        else:
            self.single_objective_window_values.append(float(candidate))
            self.single_objective_window_episodes.append(int(completed_episodes))
            self.single_objective_window_values = self.single_objective_window_values[-self.single_objective_window_size:]
            self.single_objective_window_episodes = self.single_objective_window_episodes[-self.single_objective_window_size:]
            if len(self.single_objective_window_values) < self.single_objective_window_size:
                event, reason = "window_warmup", "window_warmup"
            else:
                window_stat = float(statistics.median(self.single_objective_window_values))
                audit_required = previous_anchor is None or window_stat < previous_anchor - self.single_objective_candidate_improvement_epsilon
                if audit_required:
                    self.single_objective_candidate_anchor_value = window_stat
                    self.single_objective_candidate_episode = int(completed_episodes)
                    event, reason = "audit_required", "candidate_window_improved"
                else:
                    self.not_promoted_quality_updates += 1
                    event, reason = "not_promoted", "window_not_improved"
        self.last_promotion_diagnostics = self._diagnostics(
            event, reason, completion_rate, truncated_count,
            schedule_violation_count, physical_safety_pass, candidate,
            audit_required, window_stat, previous_anchor,
        )
        return event

    def _diagnostics(
        self, event: str, reason: str, completion_rate: float,
        truncated_count: int, schedule_violation_count: int,
        physical_safety_pass: bool, candidate: float | None,
        audit_required: bool, window_stat: float | None,
        previous_anchor: float | None = None,
    ) -> dict[str, object]:
        return {
            "promotion_mode": self.quality_checkpoint_promotion,
            "promotion_event": event,
            "promotion_decision_reason": reason,
            "promotion_target_objective": self.single_objective_name,
            "promotion_candidate_objective_value": candidate,
            "candidate_anchor_value": self.single_objective_candidate_anchor_value,
            "previous_candidate_anchor_value": previous_anchor,
            "promotion_completion_constraint_pass": completion_rate >= self.quality_completion_floor,
            "promotion_truncation_constraint_pass": truncated_count == 0,
            "promotion_violation_constraint_pass": schedule_violation_count == 0,
            "promotion_physical_safety_constraint_pass": bool(physical_safety_pass),
            "window_size": self.single_objective_window_size,
            "window_statistic": self.single_objective_window_statistic,
            "window_count": len(self.single_objective_window_values),
            "window_objective_values": list(self.single_objective_window_values),
            "window_objective_episodes": list(self.single_objective_window_episodes),
            "window_objective_statistic": window_stat,
            "accepted_window_median": self.accepted_single_objective_window_value,
            "accepted_failed_instance_count": self.accepted_single_objective_failed_instances,
            "audit_required": audit_required,
        }

    def observe_single_objective_audit(
        self, audit: dict[str, object], *, completed_episodes: int,
        window_median: float,
    ) -> str:
        instance_count = int(audit.get("instance_count", 0))
        completed_count = int(audit.get("completed_count", 0))
        failed_count = instance_count - completed_count
        completion_rate = float(audit.get("completion_rate", math.nan))
        violation_count = int(audit.get("schedule_violation_count", 0))
        physical_pass = bool(audit.get("physical_safety_pass", False))
        objective_value = audit.get("single_objective_value")
        objective_pass = objective_value is not None and math.isfinite(float(objective_value))
        completion_pass = bool(
            instance_count == self.single_objective_audit_instance_limit
            and math.isfinite(completion_rate)
            and completion_rate >= self.single_objective_audit_completion_target - 1e-12
            and failed_count <= self.single_objective_audit_max_failed_instances
        )
        violation_pass = violation_count == self.single_objective_audit_schedule_violation_target
        safety_pass = physical_pass if self.single_objective_audit_physical_safety_required else True
        audit_pass = completion_pass and violation_pass and safety_pass and objective_pass
        previous_rank = None if self.accepted_single_objective_failed_instances is None else (
            self.accepted_single_objective_failed_instances,
            float(self.accepted_single_objective_window_value),
        )
        candidate_rank = (failed_count, float(window_median))
        accepted = audit_pass and (previous_rank is None or candidate_rank < previous_rank)
        self.single_objective_audit_count += 1
        if accepted:
            self.accepted_single_objective_failed_instances = failed_count
            self.accepted_single_objective_window_value = float(window_median)
            self.accepted_single_objective_audit_value = float(objective_value)
            self.accepted_single_objective_value = float(window_median)
            self.accepted_quality_episode = int(completed_episodes)
            self.accepted_quality_updates += 1
            event = "accepted"
            reason = "first_audit_pass" if previous_rank is None else "audit_rank_improved"
        elif not audit_pass:
            self.rejected_quality_updates += 1
            event = "audit_rejected"
            reason = (
                "audit_completion_below_98" if not completion_pass else
                "audit_schedule_violation_nonzero" if not violation_pass else
                "audit_physical_safety_failed" if not safety_pass else
                "audit_objective_non_finite"
            )
        else:
            self.not_promoted_quality_updates += 1
            event, reason = "audit_passed_not_accepted", "audit_rank_not_improved"
        self.last_single_objective_audit_diagnostics = {
            "audit_event": event,
            "audit_decision_reason": reason,
            "audit_instance_count": instance_count,
            "audit_completed_count": completed_count,
            "audit_failed_instance_count": failed_count,
            "audit_completion_rate": completion_rate,
            "audit_truncated_count": int(audit.get("truncated_count", 0)),
            "audit_schedule_violation_count": violation_count,
            "audit_physical_safety_pass": physical_pass,
            "audit_completion_pass": completion_pass,
            "audit_violation_pass": violation_pass,
            "audit_safety_pass": safety_pass,
            "audit_objective_pass": objective_pass,
            "audit_single_objective_value": None if objective_value is None else float(objective_value),
            "audit_pass": audit_pass,
            "audit_window_median": float(window_median),
            "audit_candidate_rank": list(candidate_rank),
            "audit_previous_accepted_rank": None if previous_rank is None else list(previous_rank),
            "audit_accepted_rank": None if self.accepted_single_objective_failed_instances is None else [
                self.accepted_single_objective_failed_instances,
                self.accepted_single_objective_window_value,
            ],
            "accepted_checkpoint_episode": self.accepted_quality_episode,
        }
        return event

    def reset_single_objective_window(self) -> None:
        self.single_objective_window_values.clear()
        self.single_objective_window_episodes.clear()

    def observe_pareto_promotion(
        self,
        result: dict[str, object],
        *,
        completed_episodes: int,
        audited: bool,
    ) -> str:
        if self.quality_checkpoint_promotion != PARETO_PROMOTION_MODE:
            raise RuntimeError("pareto promotion is not active")
        self.last_pareto_promotion = dict(result)
        accepted = bool(result.get("accepted", False))
        if accepted and audited:
            self.accepted_quality_episode = int(completed_episodes)
            self.accepted_quality_updates += 1
            return "accepted"
        if accepted:
            return "audit_required"
        self.not_promoted_quality_updates += 1
        return str(result.get("decision", "not_promoted"))

    def observe_sampled_guard(self, *_args: object, **_kwargs: object) -> str:
        raise RuntimeError("sampled preference guards are not part of protocol v4")

    @property
    def is_formally_accepted(self) -> bool:
        """Return whether a complete independent audit accepted a candidate."""

        if self.quality_checkpoint_promotion == PARETO_PROMOTION_MODE:
            return bool(
                self.phase_transition_episode is not None
                and self.accepted_quality_episode is not None
                and self.accepted_quality_updates > 0
            )
        return bool(
            self.phase_transition_episode is not None
            and self.accepted_quality_episode is not None
            and self.accepted_quality_updates > 0
            and self.accepted_single_objective_failed_instances is not None
            and self.accepted_single_objective_window_value is not None
            and self.accepted_single_objective_audit_value is not None
            and self.accepted_single_objective_value is not None
        )

    @property
    def formal_training_status(self) -> str:
        if self.phase_transition_episode is None:
            return "feasibility_not_reached"
        if not self.is_formally_accepted:
            if self.quality_checkpoint_promotion == PARETO_PROMOTION_MODE:
                return "pareto_audit_candidate_not_reached"
            return "single_objective_98_candidate_not_reached"
        return (
            "accepted_v8_pareto_candidate"
            if self.quality_checkpoint_promotion == PARETO_PROMOTION_MODE
            else "accepted_98_experiment_candidate"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "completion_target": self.completion_target,
            "consecutive_validations_required": self.consecutive_required,
            "consecutive_validation_successes": self.consecutive_successes,
            "quality_completion_floor": self.quality_completion_floor,
            "quality_checkpoint_promotion": self.quality_checkpoint_promotion,
            "phase_transition_episode": self.phase_transition_episode,
            "accepted_quality_episode": self.accepted_quality_episode,
            "accepted_quality_updates": self.accepted_quality_updates,
            "rejected_quality_updates": self.rejected_quality_updates,
            "not_promoted_quality_updates": self.not_promoted_quality_updates,
            "formal_training_status": self.formal_training_status,
            "is_formally_accepted": self.is_formally_accepted,
            "single_objective_name": self.single_objective_name,
            "accepted_single_objective_value": self.accepted_single_objective_value,
            "single_objective_window_size": self.single_objective_window_size,
            "single_objective_window_statistic": self.single_objective_window_statistic,
            "single_objective_rollback_below_floor_consecutive": self.single_objective_rollback_below_floor_consecutive,
            "single_objective_candidate_improvement_epsilon": self.single_objective_candidate_improvement_epsilon,
            "single_objective_audit_instance_limit": self.single_objective_audit_instance_limit,
            "single_objective_audit_instance_offset": self.single_objective_audit_instance_offset,
            "single_objective_audit_completion_target": self.single_objective_audit_completion_target,
            "single_objective_audit_max_failed_instances": self.single_objective_audit_max_failed_instances,
            "single_objective_window_values": list(self.single_objective_window_values),
            "single_objective_window_episodes": list(self.single_objective_window_episodes),
            "single_objective_candidate_anchor_value": self.single_objective_candidate_anchor_value,
            "single_objective_candidate_episode": self.single_objective_candidate_episode,
            "single_objective_audit_count": self.single_objective_audit_count,
            "accepted_single_objective_failed_instances": self.accepted_single_objective_failed_instances,
            "accepted_single_objective_window_value": self.accepted_single_objective_window_value,
            "accepted_single_objective_audit_value": self.accepted_single_objective_audit_value,
            "last_promotion_diagnostics": dict(self.last_promotion_diagnostics),
            "last_single_objective_audit_diagnostics": dict(self.last_single_objective_audit_diagnostics),
            "last_pareto_promotion": dict(self.last_pareto_promotion),
        }
