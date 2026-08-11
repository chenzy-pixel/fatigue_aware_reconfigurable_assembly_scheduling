from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import matched_credit_assignment as matched
from configs import load_config
from result.io import write_csv, write_json


def _synthetic_smoke_run(tmp_path: Path) -> Path:
    run_directory = tmp_path / "smoke"
    run_directory.mkdir()
    config = deepcopy(
        load_config("configs/credit_assignment_matched_baseline.json")
    )
    config.pop("_config_path", None)
    config["seed"] = 11
    write_json(run_directory / "config.json", config)
    train_rows = []
    for episode in range(10):
        train_rows.append(
            {
                "episode": episode,
                "steps": 1,
                "policy_steps": 1,
                "forced_actions": 0,
                "reward": 0.0,
                "reward_training": 0.0,
                "reward_base": 0.0,
                "expected_reward": 0.0,
                "reward_identity_error": 0.0,
                "reward_phase": "feasibility",
                "reward_flow": 0.0,
                "reward_cost": 0.0,
                "reward_variance": 0.0,
                "reward_completion_progress": 0.0,
                "reward_completion_bonus": 0.0,
                "reward_quality": 0.0,
                "reward_truncation": 0.0,
                "reward_unfinished": 0.0,
                "reward_feasibility_shaping": 0.0,
            }
        )
    update_rows = [
        {
            "update_id": 1,
            "episode_count": 10,
            "advantage_std": 0.1,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "entropy": 0.5,
            "value_loss": 0.1,
        }
    ]
    write_csv(run_directory / "train_log.csv", train_rows)
    write_csv(run_directory / "update_log.csv", update_rows)
    write_csv(run_directory / "validation_log.csv", [])
    write_json(
        run_directory / "summary.json",
        {
            "episodes": 10,
            "updates": 1,
            "environment_steps": 10,
            "transitions": 10,
            "forced_actions": 0,
        },
    )
    write_json(
        run_directory / "provenance.json",
        {"source_sha256": "source"},
    )
    return run_directory


def test_matched_configs_only_have_whitelisted_arm_differences():
    differences = matched.validate_matched_configs()
    assert differences == matched.ALLOWED_ARM_DIFFERENCES
    configs = matched.load_matched_configs()
    assert configs["baseline"]["training"]["episodes"] == 1000
    assert configs["treatment"]["training"]["episodes"] == 1000
    assert configs["baseline"]["ppo"]["gae_lambda"] == 0.95
    assert configs["treatment"]["ppo"]["gae_lambda"] == 0.995


def test_formal_plan_has_ten_unique_sequential_commands():
    plan = matched.planned_runs(tuple(matched.ARMS), matched.SEEDS, smoke=False)
    assert plan == [
        *(('baseline', seed) for seed in matched.SEEDS),
        *(('treatment', seed) for seed in matched.SEEDS),
    ]
    commands = [
        tuple(
            matched.training_command(
                arm,
                seed,
                run_name=f"{arm}_{seed}",
                smoke=False,
            )
        )
        for arm, seed in plan
    ]
    assert len(commands) == len(set(commands)) == 10
    assert all("--episodes" in command for command in commands)


def test_smoke_plan_runs_one_update_per_arm():
    assert matched.planned_runs(
        tuple(matched.ARMS), matched.SEEDS, smoke=True
    ) == [("baseline", 11), ("treatment", 11)]


def test_retry_paths_do_not_overwrite_incomplete_results(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run_retry1").mkdir()
    assert matched.next_available_path(tmp_path, "run") == (
        tmp_path / "run_retry2"
    )


def test_source_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(
        matched,
        "source_snapshot",
        lambda: {"source_sha256": "new"},
    )
    with pytest.raises(RuntimeError, match="source snapshot changed"):
        matched.verify_source({"source_sha256": "frozen"})


def test_smoke_audit_checks_step_and_reward_identities(tmp_path):
    run_directory = _synthetic_smoke_run(tmp_path)
    audit = matched.audit_run(
        run_directory,
        arm="baseline",
        seed=11,
        smoke=True,
        source_sha256="source",
    )
    assert audit["valid"]
    assert audit["environment_steps"] == 10
    assert audit["policy_steps"] == 10
    assert audit["maximum_reward_component_error"] == 0.0

    rows = matched._read_csv(run_directory / "train_log.csv")
    rows[0]["forced_actions"] = "1"
    write_csv(run_directory / "train_log.csv", rows)
    invalid = matched.audit_run(
        run_directory,
        arm="baseline",
        seed=11,
        smoke=True,
        source_sha256="source",
    )
    assert not invalid["valid"]
    assert any("step identity" in error for error in invalid["errors"])


def test_paired_statistics_use_five_seed_level_observations():
    payload = matched.paired_statistics(
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        bootstrap_key="synthetic",
    )
    assert payload["baseline"]["count"] == 5
    assert payload["treatment"]["count"] == 5
    delta = payload["paired_difference_treatment_minus_baseline"]
    assert delta["count"] == 5
    assert delta["mean"] == pytest.approx(1.0)
    assert delta["bootstrap_95_ci"] == pytest.approx([1.0, 1.0])
    assert payload["wilcoxon"]["p_value_two_sided"] == pytest.approx(0.0625)


def test_multi_seed_aggregation_does_not_count_instances_as_replicates():
    rows = []
    for arm in matched.ARMS:
        for seed in matched.SEEDS:
            for mode in ("greedy", "sampled"):
                base = float(seed)
                value = base + (1.0 if arm == "treatment" else 0.0)
                row = {
                    "arm": arm,
                    "algorithm_seed": seed,
                    "decode_mode": mode,
                }
                row.update({metric: value for metric in matched.EVALUATION_METRICS})
                row.update({metric: value for metric in matched.TRAINING_METRICS})
                rows.append(row)
    delta_rows, payload = matched._paired_outputs(rows)
    assert delta_rows
    greedy_quality = payload["evaluation"]["greedy"]["quality_score"]
    assert greedy_quality["baseline"]["count"] == 5
    assert greedy_quality["treatment"]["count"] == 5
    assert greedy_quality[
        "paired_difference_treatment_minus_baseline"
    ]["mean"] == pytest.approx(1.0)
