from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from configs import load_config
from configs.config import public_config
from configs.runtime import assert_checkpoint_fatigue_mode
from data.dataset import OnlineInstanceDataset
from data.distribution import protocol_hashes
from result.provenance import dataset_manifest_snapshot
from scripts.ablation_protocol import ROLE_CONFIGS, VARIANTS, discover_training_run, protocol_profile
from analysis.ablation_analysis import CONTROL_FIELDS, paired_rows
from train import _rows_are_physically_safe, _rows_satisfy_active_constraints


@pytest.mark.parametrize("role", VARIANTS)
def test_ablations_share_current_parent_budgets_and_benchmark(role):
    config = load_config(ROLE_CONFIGS[role])
    parent = load_config("configs/v8/universal.json" if role in ("no_graph", "shared_head") else ROLE_CONFIGS[role.replace("neutral", "full")])
    assert protocol_hashes(config) == protocol_hashes(parent)
    assert config["reward"] == parent["reward"]
    assert config["reward"]["terminal_failure_penalty"] == 2.0
    assert config["objective_scalarizer"] == parent["objective_scalarizer"]
    assert config["training"] == parent["training"]
    assert config["runtime_manifest"]["fatigue_mode"] == ("neutral" if role.startswith("neutral") else "full")


def test_neutral_reuses_physical_online_cache(tmp_path, fixed_instance, monkeypatch):
    configs = [load_config(ROLE_CONFIGS[role]) for role in ("full_cost", "neutral_cost")]
    for config in configs:
        config["paths"]["training_instances_cache"] = str(tmp_path / "shared")
    full, neutral = [OnlineInstanceDataset(config=c, template=fixed_instance, episode_count=2) for c in configs]
    assert full.cache_directory == neutral.cache_directory
    assert full.generator_environment_precheck_config_hash == neutral.generator_environment_precheck_config_hash
    first = full[0]
    monkeypatch.setattr(neutral.generator, "generate", lambda *args, **kwargs: pytest.fail("cache must be shared"))
    second, hit, _, _ = neutral.get_with_cache_info(0)
    assert hit
    assert first.instance == second.instance
    assert first.metadata == second.metadata


def test_counterfactual_fatigue_does_not_disqualify_neutral_active_constraints():
    row = {"schedule_violation_count": 0, "fatigue_mode": "neutral", "maximum_worker_fatigue": 0.0,
        "fatigue_monitor_peak": 0.9, "safe_fatigue_limit": 0.75}
    assert _rows_satisfy_active_constraints([row])
    assert not _rows_are_physically_safe([row])
    assert not _rows_satisfy_active_constraints([{**row, "fatigue_mode": "full"}])
    assert not _rows_satisfy_active_constraints([{**row, "schedule_violation_count": 1}])


def test_checkpoint_fatigue_guard_covers_old_and_current_metadata():
    full = load_config(ROLE_CONFIGS["full_flow"])
    neutral = load_config(ROLE_CONFIGS["neutral_flow"])
    assert_checkpoint_fatigue_mode({}, full)
    assert_checkpoint_fatigue_mode({"runtime_manifest": {"fatigue_mode": "neutral"}}, neutral)
    with pytest.raises(ValueError, match="fatigue mode"):
        assert_checkpoint_fatigue_mode({}, neutral)
    with pytest.raises(ValueError, match="fatigue mode"):
        assert_checkpoint_fatigue_mode({"runtime_manifest": {"fatigue_mode": "neutral"}}, full)


def _fake_run(root: Path, name: str, config: dict):
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(json.dumps(public_config(config)), encoding="utf-8")
    (run / "best_checkpoint.pt").write_bytes(b"checkpoint fixture")
    manifest = Path(config["paths"]["manifests_root"]) / "validation/manifest.json"
    if not manifest.is_absolute():
        manifest = Path(__file__).resolve().parents[1] / manifest
    summary = {"episodes": config["training"]["episodes"], "checkpoint_selection": {"has_best": True},
        "provenance": {"dataset_manifest_sha256": dataset_manifest_snapshot(manifest)["sha256"],
            "checkpoint_sha256": hashlib.sha256(b"checkpoint fixture").hexdigest()}}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def test_discovery_rejects_wrong_penalty_or_variant(tmp_path):
    current = load_config(ROLE_CONFIGS["full_cost"])
    good = _fake_run(tmp_path, "compatible", current)
    wrong = deepcopy(current)
    wrong["reward"]["terminal_failure_penalty"] = 1.0
    _fake_run(tmp_path, "wrong_penalty", wrong)
    wrong = deepcopy(current)
    wrong["network"]["actor_head_variant"] = "shared_preference"
    _fake_run(tmp_path, "wrong_head", wrong)
    assert discover_training_run("full_cost", 11, tmp_path) == good
    (good / "best_checkpoint.pt").write_bytes(b"corrupt fixture")
    with pytest.raises(FileNotFoundError, match="compatible"):
        discover_training_run("full_cost", 11, tmp_path)


def _cell(success=True, value=10):
    return {**{name: "control" for name in CONTROL_FIELDS}, "instance_id": "i1", "sampling_repeat": "0",
        "preference_key": "p1", "task_succeeded": str(success), "physical_safety_pass": "True", "reconfiguration_cost": value}


def test_pairing_keeps_failures_out_of_objective_deltas():
    first = _cell()
    second = _cell(False, 1)
    cells, summary = paired_rows([first], [second], "cost")
    assert summary["baseline_completed"] == 1 and summary["variant_completed"] == 0
    assert summary["common_success_count"] == 0
    assert "delta_reconfiguration_cost" not in cells[0]


@pytest.mark.parametrize("field", ["terminal_failure_penalty_configured", "derived_sampling_seed", "normalization_manifest_sha256"])
def test_pairing_rejects_control_mismatch(field):
    first, second = _cell(), _cell()
    second[field] = "changed"
    with pytest.raises(ValueError, match=field):
        paired_rows([first], [second], "cost")


def test_pairing_rejects_missing_metadata_and_duplicate_cells():
    first, second = _cell(), _cell()
    first.pop("dataset_manifest_sha256")
    second.pop("dataset_manifest_sha256")
    with pytest.raises(ValueError, match="missing"):
        paired_rows([first], [second], "cost")
    with pytest.raises(ValueError, match="duplicate"):
        paired_rows([_cell(), _cell()], [_cell()], "cost")


@pytest.mark.parametrize("role", ROLE_CONFIGS)
def test_discovery_normalizes_update_default_for_all_nine_roles(tmp_path, role):
    config = load_config(ROLE_CONFIGS[role])
    saved = deepcopy(config)
    saved["training"].setdefault("episodes_per_update", saved["training"]["parallel_envs"])
    assert protocol_profile(config) == protocol_profile(saved)
    run = _fake_run(tmp_path, "current", saved)
    assert discover_training_run(role, 11, tmp_path) == run


@pytest.mark.parametrize("change", ("rho", "initial_checkpoint", "forced_actions"))
def test_discovery_rejects_changed_objective_or_initialization(tmp_path, change):
    config = load_config(ROLE_CONFIGS["full_cost"])
    if change == "rho":
        config["objective_scalarizer"]["rho"] = 0.75
    elif change == "initial_checkpoint":
        config["training"]["initial_checkpoint"] = "warm_start.pt"
    else:
        config["training"]["worker_local_physical_forced_actions"] = False
    _fake_run(tmp_path, "different", config)
    with pytest.raises(FileNotFoundError):
        discover_training_run("full_cost", 11, tmp_path)


def _evaluation_fixture(tmp_path, role):
    import torch
    from configs.formal_preferences import formal_preferences
    from data import load_dataset_split
    from data.dataset import sha256_file
    from data.selection import subset_snapshot
    from result.provenance import effective_config_snapshot
    from result.metrics import evaluation_quality_metric, quality_metric_sha256
    from utils import configured_formal_evaluation_sampling_seeds, derive_evaluation_sampling_seed, SAMPLED_EVALUATION_RNG_VERSION
    from analysis.ablation_analysis import FIELDS

    config = load_config(ROLE_CONFIGS[role])
    training, evaluation = tmp_path / "train", tmp_path / "eval"
    training.mkdir()
    evaluation.mkdir()
    for directory in (training, evaluation):
        (directory / "config.json").write_text(json.dumps(public_config(config)), encoding="utf-8")
    checkpoint = training / "best_checkpoint.pt"
    selectors = {"encoder_variant": config["network"].get("encoder_variant", "hetero_gnn"),
                 "actor_head_variant": config["network"].get("actor_head_variant", "objective_experts"),
                 "fatigue_mode": config["environment"].get("fatigue_mode", "full")}
    torch.save({"network": {"test.weight": torch.zeros(2)},
                "network_spec": {"observation_schema_version": 6, **selectors},
                "metadata": {"algorithm_seed": 11, "effective_config": public_config(config),
                             "runtime_manifest": config["runtime_manifest"]}}, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    summary = {"episodes": config["training"]["episodes"], "checkpoint_selection": {"has_best": True},
               "provenance": {"checkpoint_sha256": checkpoint_hash}}
    (training / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    dataset = load_dataset_split(config, "test")
    indices = list(range(len(dataset)))
    subset = subset_snapshot(dataset, indices, role="evaluation")
    config_hash = effective_config_snapshot(config)["sha256"]
    quality_hash = quality_metric_sha256(evaluation_quality_metric(config))
    seeds = configured_formal_evaluation_sampling_seeds(config, "final_test")
    preferences = formal_preferences(config, "final_test")
    controls = {"algorithm_seed": "11", "generator_version": config["generator"]["version"],
                "dataset_manifest_sha256": subset["dataset_manifest_sha256"], "subset_sha256": subset["subset_sha256"],
                "normalization_manifest_sha256": config["objective_scalarizer"]["normalization_manifest_sha256"],
                "quality_metric_sha256": quality_hash,
                "terminal_failure_penalty_configured": str(config["reward"]["terminal_failure_penalty"]),
                "objective_scalarizer_type": config["objective_scalarizer"]["type"],
                "objective_scalarizer_rho": str(config["objective_scalarizer"]["rho"]),
                "sampling_rng_version": SAMPLED_EVALUATION_RNG_VERSION,
                "checkpoint_sha256": checkpoint_hash, "effective_config_sha256": config_hash,
                "dataset": "test", **selectors,
                **{f"objective_scale_{k}": str(v) for k, v in config["objective_scalarizer"]["scales"].items()},
                **{key: "0" for key in FIELDS}, "task_succeeded": "True", "active_constraint_pass": "True",
                "physical_safety_pass": "True"}
    rows = []
    for record in dataset:
        for repeat, seed in enumerate(seeds):
            for point in preferences:
                key = point.key if role in ("universal", "no_graph", "shared_head") else None
                rows.append({**controls, "instance_id": record.instance.instance_id, "sampling_repeat": str(repeat),
                             "preference_key": point.key, "sampling_seed": str(seed),
                             "derived_sampling_seed": str(derive_evaluation_sampling_seed(seed, record.instance.instance_id, key))})
    metrics = {"dataset": "test", "policy": "ppo", "decode_mode": "sampled", "result_role": "formal_sampled",
               "instance_indices": indices, "sampling_seeds": seeds, "repeat_count": len(seeds),
               "instance_count": len(dataset), "cell_count": len(rows), "unique_instance_count": len(dataset),
               "preference_count": len(preferences), "subset_sha256": subset["subset_sha256"],
               "dataset_manifest_sha256": subset["dataset_manifest_sha256"], "quality_metric_sha256": quality_hash,
               "normalization_manifest_sha256": controls["normalization_manifest_sha256"],
               "provenance": {"checkpoint_sha256": checkpoint_hash, "effective_config_sha256": config_hash,
                              "dataset_manifest_sha256": subset["dataset_manifest_sha256"],
                              "evaluation_subset_sha256": subset["subset_sha256"], "quality_metric_sha256": quality_hash,
                              "normalization_manifest_sha256": controls["normalization_manifest_sha256"]}}
    entry = {"config": ROLE_CONFIGS[role], "training_run": str(training), "evaluation_run": str(evaluation),
             "checkpoint": str(checkpoint)}
    return entry, rows, metrics


@pytest.mark.parametrize("role", ROLE_CONFIGS)
def test_summary_validates_complete_role_sources(tmp_path, role):
    from analysis.ablation_analysis import validate_role_evaluation
    entry, rows, metrics = _evaluation_fixture(tmp_path, role)
    sources = validate_role_evaluation(role, entry, rows, metrics, 11)
    assert sources["checkpoint_sha256"] == metrics["provenance"]["checkpoint_sha256"]


@pytest.mark.parametrize("change, message", (
    ("role", "role config"), ("variant", "fatigue_mode"), ("checkpoint", "checkpoint hash"),
    ("config_hash", "effective config hash"), ("truncated_csv", "evaluation matrix"),
    ("duplicate", "evaluation matrix"), ("missing_monitor", "monitor"),
    ("row_checkpoint", "row checkpoint"), ("derived_seed", "derived sampling seed")))
def test_summary_rejects_mislabeled_or_corrupted_sources(tmp_path, change, message):
    from analysis.ablation_analysis import validate_role_evaluation
    entry, rows, metrics = _evaluation_fixture(tmp_path, "neutral_cost")
    if change == "role":
        entry["config"] = ROLE_CONFIGS["full_cost"]
    elif change == "variant":
        rows[0]["fatigue_mode"] = "full"
    elif change == "checkpoint":
        Path(entry["checkpoint"]).write_bytes(b"changed")
    elif change == "config_hash":
        metrics["provenance"]["effective_config_sha256"] = "changed"
    elif change == "truncated_csv":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "missing_monitor":
        rows[0].pop("fatigue_monitor_peak")
    elif change == "row_checkpoint":
        rows[0]["checkpoint_sha256"] = "changed"
    else:
        rows[0]["derived_sampling_seed"] = "1"
    with pytest.raises(ValueError, match=message):
        validate_role_evaluation("neutral_cost", entry, rows, metrics, 11)


def test_summary_accepts_undefined_optional_metrics_on_failed_rows(tmp_path):
    from analysis.ablation_analysis import validate_role_evaluation
    entry, rows, metrics = _evaluation_fixture(tmp_path, "neutral_cost")
    rows[0]["task_succeeded"] = "False"
    for key in ("mean_interstage_idle_minutes", "completed_reconfigurations_per_operation",
                "fatigue_monitor_over_limit_time_ratio", "fatigue_monitor_over_limit_area_ratio"):
        rows[0][key] = ""
    validate_role_evaluation("neutral_cost", entry, rows, metrics, 11)


def test_summary_exports_five_source_backed_comparisons(tmp_path):
    import csv
    from analysis.ablation_analysis import summarize
    manifest = {"schema": "matched_ablation_evaluation_v1", "status": "complete", "seed": 11, "runs": {}}
    for role in ROLE_CONFIGS:
        directory = tmp_path / role
        directory.mkdir()
        entry, rows, metrics = _evaluation_fixture(directory, role)
        evaluation = Path(entry["evaluation_run"])
        with (evaluation / "instance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        manifest["runs"][role] = entry
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = summarize(path, tmp_path / "report")
    with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 5
    assert all(int(row["baseline_parameters"]) == int(row["variant_parameters"]) == 2 for row in summaries)
    sources = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["sources"]
    assert set(sources) == set(ROLE_CONFIGS)
    assert "descriptive results from one training seed" in (output / "report.md").read_text(encoding="utf-8")
