from __future__ import annotations

import math
from copy import deepcopy

import pytest

from configs import load_config
from configs.config import public_config
from train import (
    LearningRatePlateauController,
    _checkpoint_metadata,
    _failure_progress_summary,
)
from training import LexicographicCheckpointSelector


CONFIGS = {
    "flow": ("configs/e1/single_flow.json", [1.0, 0.0, 0.0]),
    "cost": ("configs/e1/single_cost.json", [0.0, 1.0, 0.0]),
    "variance": ("configs/e1/single_variance.json", [0.0, 0.0, 1.0]),
}


def _validation(
    completion: float,
    quality: float,
    *,
    violations: int = 0,
) -> dict[str, float | int]:
    return {
        "completion_rate": completion,
        "preference_balanced_quality_score": quality,
        "schedule_violation_count": violations,
    }


def test_default_uses_only_single_stage_reward_and_fixed_formal_sampling():
    config = load_config("configs/default.json")
    public = public_config(config)
    assert config["reward"]["mode"] == (
        "single_stage_progress_quality_failure_v2"
    )
    assert config["reward"]["terminal_failure_penalty"] == 1.0
    assert config["ppo"]["gamma"] == 1.0
    assert config["reward"]["feasibility_shaping"]["enabled"] is False
    assert "two_stage" not in config["training"]
    assert config["training"]["formal_evaluation"]["decode_mode"] == "sampled"
    assert "greedy_diagnostic" not in config["training"]["validation_control"]
    assert "audit_seed_offset" not in config["training"]["formal_evaluation"]
    assert not {
        "completion_bonus",
        "truncation_penalty",
        "unfinished_order_penalty",
    }.intersection(config["reward"])
    assert public["runtime_manifest"]["training_protocol"] == (
        "single_stage_lexicographic_failure_v2"
    )


@pytest.mark.parametrize("objective", tuple(CONFIGS))
def test_e1_children_keep_one_hot_preference_and_user_validation_cadence(
    objective: str,
):
    path, preference = CONFIGS[objective]
    child = load_config(path)
    base = load_config("configs/default.json")
    assert child["preference"]["quality"] == {
        "mode": "fixed",
        "fixed": preference,
        "block_size": 20,
        "endpoint_repeats": 2,
        "sobol_count": 14,
    }
    assert child["training"]["validation_interval_episodes"] == 40
    assert child["network"] == base["network"]
    assert child["ppo"] == base["ppo"]


def test_first_safe_checkpoint_initializes_even_with_zero_completion_and_inf_quality():
    selector = LexicographicCheckpointSelector()
    event = selector.observe(
        _validation(0.0, math.inf),
        completed_episodes=40,
        physical_safety_pass=True,
    )
    assert event == "best_initialized"
    assert selector.has_best is True
    assert selector.best_episode == 40
    assert selector.best_score == (0.0, math.inf)


def test_checkpoint_selector_is_completion_first_quality_second_and_ties_keep_best():
    selector = LexicographicCheckpointSelector()
    assert selector.observe(
        _validation(0.5, 0.4), completed_episodes=40, physical_safety_pass=True
    ) == "best_initialized"
    assert selector.observe(
        _validation(0.5, 0.3), completed_episodes=80, physical_safety_pass=True
    ) == "best_improved"
    assert selector.observe(
        _validation(0.5 + 5e-13, 0.3 + 5e-13),
        completed_episodes=120,
        physical_safety_pass=True,
    ) == "tied"
    assert selector.best_episode == 80
    assert selector.observe(
        _validation(0.6, 0.9), completed_episodes=160, physical_safety_pass=True
    ) == "best_improved"
    assert selector.best_score == pytest.approx((0.6, 0.9))


@pytest.mark.parametrize(
    ("physical_safe", "violations"),
    ((False, 0), (True, 1)),
)
def test_unsafe_candidate_never_initializes_best(physical_safe: bool, violations: int):
    selector = LexicographicCheckpointSelector()
    event = selector.observe(
        _validation(1.0, 0.0, violations=violations),
        completed_episodes=40,
        physical_safety_pass=physical_safe,
    )
    assert event == "ineligible"
    assert selector.has_best is False
    assert selector.as_dict()["best_episode"] is None


def test_plateau_decay_does_not_restore_any_checkpoint():
    config = load_config("configs/e1/single_flow.json")
    controller = LearningRatePlateauController.from_config(config)
    controller.patience = 2
    original = controller.learning_rate
    assert controller.observe(False) is False
    assert controller.observe(False) is True
    assert controller.learning_rate == pytest.approx(original * controller.factor)
    assert set(controller.as_dict()) == {
        "learning_rate",
        "minimum_learning_rate",
        "plateau_factor",
        "plateau_patience_validations",
        "stale_validations",
        "learning_rate_decay_count",
    }


def test_checkpoint_metadata_freezes_selection_inputs_and_seed_rule():
    config = load_config("configs/e1/single_flow.json")
    metadata = _checkpoint_metadata(
        config,
        role="best",
        episode=40,
        validation_split="validation",
        validation_instance_limit=2,
        validation=_validation(0.0, math.inf),
    )
    assert metadata["selection_decode_mode"] == "sampled"
    assert metadata["selection_temperature"] == 1.0
    assert metadata["validation_repeat_count"] == 3
    assert metadata["validation_sampling_seeds"] == [100011, 100012, 100013]
    assert metadata["final_test_sampling_seeds"] == [300011, 300012, 300013]
    assert metadata["validation_instance_order"] == [
        "instance_2000000.json",
        "instance_2000001.json",
    ]
    assert metadata["preference_count"] == 1
    assert len(metadata["validation_dataset_manifest"]["sha256"]) == 64
    assert "sha256" in metadata["derived_sampling_seed_rule"]


def test_failure_progress_summary_preserves_partial_completion_distribution():
    rows = [
        {"task_failed": True, "operation_progress": 0.1},
        {"task_failed": True, "operation_progress": 0.75},
        {"task_failed": True, "operation_progress": 0.98},
        {"task_failed": False, "operation_progress": 1.0},
    ]
    summary = _failure_progress_summary(rows)
    assert summary["count"] == 3
    assert summary["median"] == pytest.approx(0.75)
    assert summary["maximum"] == pytest.approx(0.98)
    assert sum(summary["bins"].values()) == 3


def test_latest_only_validation_rejects_stage_configuration():
    config = deepcopy(load_config("configs/e1/single_flow.json"))
    config.pop("runtime_manifest")
    config["training"]["two_stage"] = {}
    from configs import validate_latest_only_config

    with pytest.raises(ValueError, match="training.two_stage"):
        validate_latest_only_config(config)
