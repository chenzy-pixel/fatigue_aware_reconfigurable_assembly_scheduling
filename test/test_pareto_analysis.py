from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pareto_analysis import (
    EXPECTED_SAMPLING_SEEDS,
    analyze_audit,
    dominates,
    hypervolume_3d,
    nondominated_indices,
    normalize_objectives,
    vectors_equal,
)


QUALITY_METRIC = {
    "version": "canonical_bounded_quality_v1",
    "flow_scale": 1200.0,
    "cost_scale": 1000.0,
    "variance_scale": 50.0,
    "quality_weights": {"flow": 0.5, "cost": 0.3, "variance": 0.2},
}
MANIFEST_HASH = "manifest-hash"
METRIC_HASH = "metric-hash"


def _audit_payload(*, instance_count: int = 2) -> dict:
    arms = {}
    for arm in ("c0", "e1"):
        arms[arm] = {
            "arm": arm,
            "checkpoint": f"result/runs/{arm}/accepted_checkpoint.pt",
            "checkpoint_metadata": {"algorithm_seed": 11, "seed": 11},
            "instance_count": instance_count,
            "sampled_global_rng_unchanged": True,
            "greedy": {
                "dataset": "validation",
                "instance_count": instance_count,
                "evaluation_schema_version": "4.1.0",
                "dataset_manifest_sha256": MANIFEST_HASH,
                "quality_metric_sha256": METRIC_HASH,
                "quality_metric": deepcopy(QUALITY_METRIC),
            },
            "provenance": {
                "dataset_manifest_sha256": MANIFEST_HASH,
                "quality_metric_sha256": METRIC_HASH,
                "checkpoint_sha256": f"{arm}-checkpoint-hash",
                "network_weights_sha256": f"{arm}-weights-hash",
                "effective_config_sha256": f"{arm}-config-hash",
                "source_state_sha256": "source-hash",
            },
        }
    return {
        "audit_protocol_version": "v7_e1_protocol_v2",
        "result_schema_version": "4.1.0",
        "sampling_seeds": list(EXPECTED_SAMPLING_SEEDS),
        "parallel_envs": [1, 10],
        "all_checks_passed": True,
        "arms": arms,
    }


def _serial_row(
    *,
    arm: str,
    instance_index: int,
    mode: str,
    sampling_seed: int | None,
    objectives: tuple[float, float, float],
    suffix: str,
    terminated: bool = True,
    truncated: bool = False,
    violations: int = 0,
) -> dict[str, object]:
    return {
        "arm": arm,
        "mode": mode,
        "instance_id": f"validation_case_{instance_index}",
        "seed": 2_000_000 + instance_index,
        "pressure_type": "balanced" if instance_index == 0 else "worker_bottleneck",
        "cost_profile": "balanced",
        "terminated": terminated,
        "truncated": truncated,
        "schedule_violation_count": violations,
        "action_trace_sha256": f"trace-{arm}-{instance_index}-{suffix}",
        "sampling_seed": "" if sampling_seed is None else sampling_seed,
        "flow_time_objective": objectives[0],
        "reconfiguration_cost": objectives[1],
        "worker_load_variance": objectives[2],
    }


def _candidate_rows() -> list[dict[str, object]]:
    templates = {
        "c0": (
            (10.0, 10.0, 10.0),
            (9.0, 12.0, 10.0),
            (12.0, 9.0, 9.0),
            (11.0, 11.0, 8.0),
        ),
        "e1": (
            (8.0, 11.0, 10.0),
            (10.0, 8.0, 11.0),
            (9.0, 9.0, 9.0),
            (13.0, 13.0, 13.0),
        ),
    }
    rows: list[dict[str, object]] = []
    for instance_index in range(2):
        offset = float(instance_index)
        for arm in ("c0", "e1"):
            objectives = templates[arm]
            rows.append(
                _serial_row(
                    arm=arm,
                    instance_index=instance_index,
                    mode="greedy_serial",
                    sampling_seed=None,
                    objectives=tuple(value + offset for value in objectives[0]),
                    suffix="greedy",
                )
            )
            for repeat, sampling_seed in enumerate(EXPECTED_SAMPLING_SEEDS, start=1):
                rows.append(
                    _serial_row(
                        arm=arm,
                        instance_index=instance_index,
                        mode="sampled_serial",
                        sampling_seed=sampling_seed,
                        objectives=tuple(
                            value + offset for value in objectives[repeat]
                        ),
                        suffix=f"sampled-{sampling_seed}",
                    )
                )
    parallel_copy = deepcopy(rows[1])
    parallel_copy["mode"] = "sampled_parallel_10"
    rows.append(parallel_copy)
    return rows


def _write_audit(
    root: Path,
    *,
    audit: dict | None = None,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    root.mkdir(parents=True)
    payload = _audit_payload() if audit is None else audit
    records = _candidate_rows() if rows is None else rows
    (root / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (root / "instance_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return root


def test_dominance_equality_and_tolerance() -> None:
    assert dominates((1.0, 2.0, 3.0), (1.0, 2.0, 4.0))
    assert not dominates((1.0, 2.0, 4.0), (1.0, 2.0, 3.0))
    assert vectors_equal((1.0, 2.0, 3.0), (1.0 + 5e-10, 2.0, 3.0))
    assert not dominates((1.0, 2.0, 3.0), (1.0 + 5e-10, 2.0, 3.0))
    assert nondominated_indices([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]) == [0]


def test_bounded_normalization_is_monotone() -> None:
    lower = normalize_objectives((10.0, 20.0, 30.0), (100.0, 100.0, 100.0))
    upper = normalize_objectives((20.0, 40.0, 60.0), (100.0, 100.0, 100.0))
    assert all(0.0 <= value < 1.0 for value in lower + upper)
    assert all(first < second for first, second in zip(lower, upper, strict=True))


def test_exact_hypervolume_for_single_and_overlapping_boxes() -> None:
    assert hypervolume_3d([(0.2, 0.3, 0.4)]) == pytest.approx(0.8 * 0.7 * 0.6)
    assert hypervolume_3d(
        [(0.2, 0.8, 0.8), (0.8, 0.2, 0.8)]
    ) == pytest.approx(0.056)
    assert hypervolume_3d([(0.2, 0.3, 0.4), (0.4, 0.5, 0.6)]) == pytest.approx(
        0.8 * 0.7 * 0.6
    )


def test_analysis_filters_invalid_rows_and_verifies_parallel_copy(tmp_path: Path) -> None:
    rows = _candidate_rows()
    rows.append(
        _serial_row(
            arm="c0",
            instance_index=99,
            mode="sampled_serial",
            sampling_seed=100011,
            objectives=(1.0, 1.0, 1.0),
            suffix="invalid",
            terminated=False,
            truncated=True,
        )
    )
    audit_dir = _write_audit(tmp_path / "audit", rows=rows)
    summary = analyze_audit(audit_dir, tmp_path / "out", render_plots=False)
    assert summary["ingestion"]["parallel_copy_rows_verified_and_excluded"] == 1
    assert summary["ingestion"]["invalid_serial_row_count"] == 1
    assert summary["ingestion"]["invalid_reason_counts"] == {
        "not_terminated": 1,
        "truncated": 1,
    }
    assert summary["candidate_design"]["candidate_count"] == 16


def test_analysis_rejects_manifest_mismatch(tmp_path: Path) -> None:
    audit = _audit_payload()
    audit["arms"]["e1"]["greedy"]["dataset_manifest_sha256"] = "different"
    audit_dir = _write_audit(tmp_path / "audit", audit=audit)
    with pytest.raises(ValueError, match="manifest"):
        analyze_audit(audit_dir, tmp_path / "out", render_plots=False)


def test_analysis_rejects_missing_arm_for_instance(tmp_path: Path) -> None:
    rows = [
        row
        for row in _candidate_rows()
        if not (row["instance_id"] == "validation_case_1" and row["arm"] == "e1")
    ]
    audit_dir = _write_audit(tmp_path / "audit", rows=rows)
    with pytest.raises(ValueError, match="missing arm e1"):
        analyze_audit(audit_dir, tmp_path / "out", render_plots=False)


def test_end_to_end_analysis_writes_tables_report_and_plots(tmp_path: Path) -> None:
    audit_dir = _write_audit(tmp_path / "audit")
    output_dir = tmp_path / "out"
    summary = analyze_audit(audit_dir, output_dir)
    assert summary["candidate_design"] == {
        "instance_count": 2,
        "candidate_count": 16,
        "arm_candidate_counts": {"c0": 8, "e1": 8},
        "serial_modes": ["greedy_serial", "sampled_serial"],
        "aggregate_common_instance_count": 2,
    }
    assert summary["ingestion"]["invalid_serial_row_count"] == 0
    for row in csv.DictReader(
        (output_dir / "instance_summary.csv").open(encoding="utf-8-sig")
    ):
        c0_hv = float(row["c0_hypervolume"])
        e1_hv = float(row["e1_hypervolume"])
        union_hv = float(row["union_hypervolume"])
        assert 0.0 <= c0_hv <= 1.0
        assert 0.0 <= e1_hv <= 1.0
        assert max(c0_hv, e1_hv) <= union_hv <= 1.0
    expected = (
        "candidates.csv",
        "pareto_front.csv",
        "instance_summary.csv",
        "aggregate_candidates.csv",
        "summary.json",
        "report.md",
        "pareto_projections.pdf",
        "pareto_projections.png",
        "hypervolume_comparison.pdf",
        "hypervolume_comparison.png",
    )
    for name in expected:
        path = output_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0
