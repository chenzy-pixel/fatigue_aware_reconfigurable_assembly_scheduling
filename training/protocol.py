"""Single-stage validation and checkpoint selection protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


SELECTION_TOLERANCE = 1e-12


def _equal_with_tolerance(first: float, second: float) -> bool:
    if math.isinf(first) or math.isinf(second):
        return first == second
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=SELECTION_TOLERANCE,
    )


@dataclass
class LexicographicCheckpointSelector:
    """Track the safest completion-first, quality-second checkpoint."""

    best_completion_rate: float | None = None
    best_quality_score: float | None = None
    best_episode: int | None = None
    eligible_validations: int = 0
    unsafe_validations: int = 0
    improvement_count: int = 0
    tie_count: int = 0
    last_decision: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "LexicographicCheckpointSelector":
        if str(config["reward"].get("mode")) != (
            "single_stage_progress_quality_failure_v2"
        ):
            raise ValueError(
                "single-stage checkpoint selection requires the single-stage reward"
            )
        if not math.isclose(float(config["ppo"]["gamma"]), 1.0):
            raise ValueError("single-stage telescoping training requires ppo.gamma = 1.0")
        return cls()

    @property
    def has_best(self) -> bool:
        return self.best_episode is not None

    @property
    def best_score(self) -> tuple[float, float] | None:
        if self.best_completion_rate is None or self.best_quality_score is None:
            return None
        return self.best_completion_rate, self.best_quality_score

    def observe(
        self,
        validation: dict,
        *,
        completed_episodes: int,
        physical_safety_pass: bool,
    ) -> str:
        completion_rate = float(validation["completion_rate"])
        quality_score = float(
            validation.get("preference_balanced_quality_score", math.inf)
        )
        if (
            not math.isfinite(completion_rate)
            or completion_rate < 0.0
            or completion_rate > 1.0
        ):
            raise ValueError("validation completion_rate must be in [0, 1]")
        if math.isnan(quality_score) or quality_score < 0.0:
            raise ValueError(
                "preference-balanced quality must be non-negative or +inf"
            )
        schedule_safe = int(validation.get("schedule_violation_count", 0)) == 0
        eligible = bool(schedule_safe and physical_safety_pass)
        previous = self.best_score

        if not eligible:
            self.unsafe_validations += 1
            event = "ineligible"
            reason = (
                "schedule_violation"
                if not schedule_safe
                else "physical_safety_failed"
            )
        else:
            self.eligible_validations += 1
            if previous is None:
                improved = True
                event = "best_initialized"
                reason = "first_safe_checkpoint"
            else:
                previous_completion, previous_quality = previous
                completion_improved = (
                    completion_rate
                    > previous_completion + SELECTION_TOLERANCE
                )
                completion_tied = _equal_with_tolerance(
                    completion_rate, previous_completion
                )
                quality_improved = (
                    completion_tied
                    and quality_score
                    < previous_quality - SELECTION_TOLERANCE
                )
                improved = completion_improved or quality_improved
                if improved:
                    event = "best_improved"
                    reason = (
                        "completion_improved"
                        if completion_improved
                        else "quality_improved_at_equal_completion"
                    )
                elif completion_tied and _equal_with_tolerance(
                    quality_score, previous_quality
                ):
                    self.tie_count += 1
                    event = "tied"
                    reason = "retain_existing_best"
                else:
                    event = "not_improved"
                    reason = "lexicographic_rank_not_improved"
            if improved:
                self.best_completion_rate = completion_rate
                self.best_quality_score = quality_score
                self.best_episode = int(completed_episodes)
                self.improvement_count += 1

        self.last_decision = {
            "checkpoint_event": event,
            "checkpoint_decision_reason": reason,
            "checkpoint_eligible": eligible,
            "candidate_completion_rate": completion_rate,
            "candidate_preference_balanced_quality_score": quality_score,
            "previous_best_score": None if previous is None else list(previous),
            "best_completion_rate": self.best_completion_rate,
            "best_preference_balanced_quality_score": self.best_quality_score,
            "best_episode": self.best_episode,
        }
        return event

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": "single_stage_lexicographic_failure_v2",
            "selection_tolerance": SELECTION_TOLERANCE,
            "has_best": self.has_best,
            "best_completion_rate": self.best_completion_rate,
            "best_preference_balanced_quality_score": self.best_quality_score,
            "best_episode": self.best_episode,
            "eligible_validations": self.eligible_validations,
            "unsafe_validations": self.unsafe_validations,
            "improvement_count": self.improvement_count,
            "tie_count": self.tie_count,
            "last_decision": dict(self.last_decision),
        }
