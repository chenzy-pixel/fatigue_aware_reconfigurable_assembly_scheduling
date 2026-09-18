from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from configs import load_config, validate_latest_only_config
from configs.normalization import (
    apply_normalization_manifest,
    build_normalization_manifest,
    file_sha256,
    load_normalization_manifest,
    write_immutable_manifest,
)
from environment import (
    feasibility_preference_context,
    quality_preference_for_episode,
    simplex_lattice,
)
from result.v8_promotion import (
    _safety_gate,
    compare_preference_conditioned_checkpoints,
    paired_instance_block_bootstrap,
)
from v8_normalization import collect_specialist_audits
from train import (
    _assert_pareto_checkpoint_replay,
    _validate_pareto_validation_protocol,
)


@pytest.mark.parametrize(
    ("path", "preference"),
    (
        ("configs/v8/specialist_flow.json", [1.0, 0.0, 0.0]),
        ("configs/v8/specialist_cost.json", [0.0, 1.0, 0.0]),
        ("configs/v8/specialist_variance.json", [0.0, 0.0, 1.0]),
        ("configs/e1/single_flow.json", [1.0, 0.0, 0.0]),
        ("configs/e1/single_cost.json", [0.0, 1.0, 0.0]),
        ("configs/e1/single_variance.json", [0.0, 0.0, 1.0]),
    ),
)
def test_single_objective_configs_align_feasibility_and_quality_preferences(
    path,
    preference,
):
    config = load_config(path)
    assert config["preference"]["feasibility"] == preference
    assert config["preference"]["quality"]["fixed"] == preference
    assert "quality_weights" not in config["reward"]
    feasibility = feasibility_preference_context(config)
    quality = quality_preference_for_episode(
        config,
        algorithm_seed=11,
        quality_episode_index=0,
    )
    assert feasibility.as_tuple() == tuple(preference)
    assert quality.as_tuple() == tuple(preference)
    assert feasibility.key == quality.key


def test_v8_specialists_use_debug_validation_cadence_and_zero_gate():
    config = load_config("configs/v8/specialist_flow.json")
    assert config["training"]["validation_interval_episodes"] == 20
    assert config["network"]["residual_gate_initial_logit"] == 0.0
    assert config["training"]["two_stage"]["single_objective_promotion"][
        "rollback_completion_drop"
    ] == pytest.approx(3 / 50)


def test_v8_universal_keeps_balanced_feasibility_preference():
    config = load_config("configs/default.json")
    feasibility = feasibility_preference_context(config)
    assert feasibility.as_tuple() == pytest.approx((1.0 / 3.0,) * 3)
    quality_points = [
        quality_preference_for_episode(
            config,
            algorithm_seed=11,
            quality_episode_index=index,
        ).as_tuple()
        for index in range(20)
    ]
    assert quality_points[:6] == [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]


def test_v8_rejects_legacy_reward_quality_weights():
    config = deepcopy(load_config("configs/default.json"))
    config.pop("runtime_manifest")
    config["reward"]["quality_weights"] = {
        "flow": 1.0,
        "cost": 0.0,
        "variance": 0.0,
    }
    with pytest.raises(ValueError, match="reward.quality_weights is not accepted"):
        validate_latest_only_config(config)


def test_normalization_manifest_round_trip_and_hash_guard(tmp_path: Path):
    audit = tmp_path / "audit_manifest.json"
    audit.write_text("{}\n", encoding="utf-8")
    audit_sha = file_sha256(audit)
    rows = []
    for objective_index, objective in enumerate(("flow", "cost", "variance")):
        for seed_index, seed in enumerate((11, 23, 37, 53, 71)):
            checkpoint = tmp_path / f"{objective}_{seed}.pt"
            checkpoint.write_bytes(f"{objective}:{seed}".encode())
            rows.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "raw_objective_mean": 10.0 * (objective_index + 1) + seed_index,
                    "audit_dataset_sha256": audit_sha,
                    "audit_instance_offset": 50,
                    "audit_instance_count": 200,
                }
            )
    manifest = build_normalization_manifest(rows, audit_dataset_path=audit)
    destination = tmp_path / "normalization.json"
    digest = write_immutable_manifest(destination, manifest)
    loaded = load_normalization_manifest(destination, expected_sha256=digest)
    assert loaded["scales"] == {"flow": 12.0, "cost": 22.0, "variance": 32.0}
    config = {
        "objective_scalarizer": {
            "scale_source": "frozen_manifest",
            "normalization_manifest": destination.name,
            "normalization_manifest_sha256": digest,
        },
        "network": {},
        "training": {"two_stage": {"pareto_promotion": {}}},
    }
    apply_normalization_manifest(config, project_root=tmp_path)
    assert config["objective_scalarizer"]["scales"] == loaded["scales"]
    assert config["network"]["normalization_manifest_sha256"] == digest
    assert set(
        config["training"]["two_stage"]["pareto_promotion"][
            "endpoint_prediction_upper_bounds"
        ]
    ) == {"flow", "cost", "variance"}
    with pytest.raises(FileExistsError):
        write_immutable_manifest(destination, manifest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_normalization_manifest(destination, expected_sha256="0" * 64)


def test_collect_specialist_audits_requires_all_v8_checkpoint_provenance(tmp_path: Path):
    audit = tmp_path / "audit_manifest.json"
    audit.write_text("{}\n", encoding="utf-8")
    audit_sha = file_sha256(audit)
    for objective_index, objective in enumerate(("flow", "cost", "variance")):
        for seed_index, seed in enumerate((11, 23, 37, 53, 71)):
            run = tmp_path / f"v8_specialist_{objective}_seed{seed}"
            run.mkdir()
            torch.save(
                {
                    "network_spec": {
                        "policy_head_version": 8,
                        "observation_schema_version": 5,
                        "expert_weight_parameterization": "simplex_softplus_v8",
                    },
                    "metadata": {
                        "single_objective_name": objective,
                        "algorithm_seed": seed,
                        "checkpoint_role": "accepted",
                        "accepted_single_objective_audit_value": (
                            10.0 * (objective_index + 1) + seed_index
                        ),
                        "dataset_manifest_sha256": audit_sha,
                        "single_objective_audit_instance_offset": 50,
                        "single_objective_audit_instance_limit": 200,
                    },
                },
                run / "accepted_checkpoint.pt",
            )
    rows = collect_specialist_audits(tmp_path)
    assert len(rows) == 15
    manifest = build_normalization_manifest(rows, audit_dataset_path=audit)
    assert manifest["scales"] == {"flow": 12.0, "cost": 22.0, "variance": 32.0}


def test_primary_bootstrap_operates_on_paired_instance_blocks():
    candidate = [float(index % 4) for index in range(50)]
    incumbent = [value + 0.5 for value in candidate]
    interval = paired_instance_block_bootstrap(candidate, incumbent)
    assert interval.replicates == 10_000
    assert interval.upper < 0
    with pytest.raises(ValueError, match="10,000"):
        paired_instance_block_bootstrap(candidate, incumbent, replicates=999)


def test_universal_sampled_audit_gates_each_preference_not_any_failure_instance():
    grid = simplex_lattice(10, include=())
    rows = []
    for instance_index in range(200):
        for preference_index, preference in enumerate(grid):
            failed = bool(
                instance_index < 13
                and preference_index == instance_index % len(grid)
            )
            rows.append(
                {
                    "instance_id": f"instance_{instance_index:03d}",
                    "preference_key": f"preference_{preference_index:02d}",
                    "preference": preference.as_dict(),
                    "terminated": not failed,
                    "truncated": failed,
                    "schedule_violation_count": 0,
                    "maximum_worker_fatigue": 0.4,
                    "safe_fatigue_limit": 0.8,
                }
            )
    passed, detail = _safety_gate(rows, minimum_completion=0.98)
    assert passed is True
    assert detail["failed_instance_count"] == 13
    assert detail["failed_instance_count_is_diagnostic"] is True
    assert detail["minimum_completion_rate"] == pytest.approx(199 / 200)

    for row in rows:
        if (
            row["preference_key"] == "preference_00"
            and row["instance_id"] in {
                "instance_013",
                "instance_014",
                "instance_015",
                "instance_016",
                "instance_017",
            }
        ):
            row["terminated"] = False
            row["truncated"] = True
    passed, detail = _safety_gate(rows, minimum_completion=0.98)
    assert passed is False
    assert detail["completion_rate_by_preference"]["preference_00"] < 0.98


def test_universal_checkpoint_replay_requires_identical_seeded_action_traces():
    expected = [
        {
            "instance_id": "instance_001",
            "preference_key": "preference_00",
            "sampling_seed": 200011,
            "derived_sampling_seed": 123,
            "sampling_evaluation_key": "preference_00",
            "sampling_rng_version": "evaluation_unit_sha256_v2",
            "action_trace_sha256": "a" * 64,
            "terminated": True,
            "truncated": False,
            "schedule_violation_count": 0,
            "flow_time_objective": 100.0,
            "reconfiguration_cost": 20.0,
            "worker_load_variance": 3.0,
        }
    ]
    assert len(_assert_pareto_checkpoint_replay(expected, deepcopy(expected))) == 64
    changed = deepcopy(expected)
    changed[0]["action_trace_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="action_trace_sha256"):
        _assert_pareto_checkpoint_replay(expected, changed)


def test_formal_universal_run_requires_frozen_manifest_and_disjoint_audit():
    config = load_config("configs/default.json")
    settings = config["training"]["two_stage"]["pareto_promotion"]
    assert settings["validation_instance_limit"] == 50
    assert settings["audit_instance_offset"] == 50
    assert settings["audit_instance_limit"] == 200
    _validate_pareto_validation_protocol(config, smoke=True, validation_limit=2)
    with pytest.raises(ValueError, match="frozen normalization manifest"):
        _validate_pareto_validation_protocol(config, smoke=False, validation_limit=50)


def test_tiny_nonsignificant_hv_change_keeps_incumbent():
    candidate = []
    incumbent = []
    grid = simplex_lattice(10, include=())
    for instance_index in range(50):
        candidate_flow = 99.999 if instance_index % 2 == 0 else 100.001
        for preference_index, preference in enumerate(grid):
            common = {
                "instance_id": f"instance_{instance_index:03d}",
                "preference": preference.as_dict(),
                "preference_key": f"lambda_{preference_index:02d}",
                "terminated": True,
                "truncated": False,
                "schedule_violation_count": 0,
                "maximum_worker_fatigue": 0.4,
                "safe_fatigue_limit": 0.8,
                "reconfiguration_cost": 100.0,
                "worker_load_variance": 100.0,
                "preference_quality_score": 0.5,
            }
            incumbent.append({**common, "flow_time_objective": 100.0})
            candidate.append({**common, "flow_time_objective": candidate_flow})
    result = compare_preference_conditioned_checkpoints(
        candidate,
        incumbent,
        scales=(1000.0, 1000.0, 1000.0),
        endpoint_prediction_upper_bounds={
            "flow": 1000.0,
            "cost": 1000.0,
            "variance": 1000.0,
        },
    )
    assert result["primary_delta"]["lower"] == pytest.approx(0.0)
    assert result["primary_delta"]["upper"] == pytest.approx(0.0)
    assert result["hypervolume_delta"]["lower"] <= 0.0
    assert result["accepted"] is False
    assert result["decision"] == "keep_incumbent_hv_not_significant"
