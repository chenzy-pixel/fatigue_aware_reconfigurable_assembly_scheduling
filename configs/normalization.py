"""Immutable objective scales derived from completed E1 validation runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NORMALIZATION_MANIFEST_SCHEMA = "e1_tail_validation_scales_v1"
OBJECTIVE_FIELDS = {
    "flow": "flow_time_objective",
    "cost": "reconfiguration_cost",
    "variance": "worker_load_variance",
}
SUMMARY_FIELDS = {
    "flow": "mean_flow_time_objective",
    "cost": "mean_reconfiguration_cost",
    "variance": "mean_worker_load_variance",
}
TAIL_EPISODES = (840, 880, 920, 960, 1000)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_normalization_manifest(
    source_runs: Mapping[str, str | Path],
    *,
    validation_dataset_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Recompute five successful-trajectory means per objective and freeze medians."""
    if set(source_runs) != set(OBJECTIVE_FIELDS):
        raise ValueError("source_runs must contain flow, cost, and variance")
    root = Path(project_root).resolve()
    validation = Path(validation_dataset_path).resolve()
    if not validation.is_file():
        raise FileNotFoundError(validation)
    validation_sha = file_sha256(validation)
    validation_files = json.loads(validation.read_text(encoding="utf-8-sig"))["files"][:50]
    expected_instance_seeds = {int(entry["seed"]) for entry in validation_files}
    if len(expected_instance_seeds) != 50:
        raise ValueError("validation manifest must provide 50 distinct fixed instances")
    sources: dict[str, dict[str, Any]] = {}
    scales: dict[str, float] = {}
    for objective, field in OBJECTIVE_FIELDS.items():
        run = Path(source_runs[objective]).resolve()
        config_path = run / "config.json"
        log_path = run / "validation_log.csv"
        rows_path = run / "sampled_validation_instance_metrics.csv"
        summary_path = run / "summary.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(config["seed"]) != 11 or int(config["training"]["episodes"]) != 1000:
            raise ValueError(f"unexpected seed or episode budget: {run}")
        if (
            int(config["training"]["validation_instance_limit"]) != 50
            or int(config["training"]["formal_evaluation"]["validation_repeats"]) != 3
        ):
            raise ValueError(f"source validation protocol is not 50 instances x 3 repeats: {run}")
        if (
            int(summary["episodes"]) != 1000
            or summary["provenance"]["dataset_manifest_sha256"] != validation_sha
        ):
            raise ValueError(f"run did not complete on the expected validation manifest: {run}")
        expected_preference = {
            "flow": [1.0, 0.0, 0.0],
            "cost": [0.0, 1.0, 0.0],
            "variance": [0.0, 0.0, 1.0],
        }[objective]
        if config["preference"]["quality"]["fixed"] != expected_preference:
            raise ValueError(f"wrong objective preference: {run}")
        if (
            objective == "flow"
            and config["network"].get("worker_flow_time_normalization")
            != "candidate_zscore_v1"
        ):
            raise ValueError("flow scale must come from the relative-time run")
        if (
            objective == "flow"
            and float(config["network"].get("worker_flow_time_std_floor", -1))
            != 0.001
        ):
            raise ValueError("flow scale source has an unexpected time standard-deviation floor")
        log = {int(row["episode"]): row for row in _csv_rows(log_path)}
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in _csv_rows(rows_path):
            episode = int(row["validation_episode"])
            if episode in TAIL_EPISODES:
                grouped[episode].append(row)
        means: list[float] = []
        fixed_ids: set[str] | None = None
        for episode in TAIL_EPISODES:
            cells = grouped[episode]
            ids = {row["instance_id"] for row in cells}
            units = {(row["instance_id"], int(row["sampling_repeat"])) for row in cells}
            if len(cells) != 150 or len(ids) != 50 or len(units) != 150:
                raise ValueError(f"expected 50 instances x 3 sampled repeats at episode {episode}: {run}")
            if {int(row["seed"]) for row in cells} != expected_instance_seeds:
                raise ValueError(f"validation rows do not match the fixed manifest instances at episode {episode}: {run}")
            if fixed_ids is None:
                fixed_ids = ids
            elif ids != fixed_ids:
                raise ValueError(f"validation instances changed at episode {episode}: {run}")
            sampling_pairs = {
                (int(row["sampling_repeat"]), int(row["sampling_seed"]))
                for row in cells
            }
            if sampling_pairs != {(0, 100011), (1, 100012), (2, 100013)} or any(
                row["decode_mode"] != "sampled" for row in cells
            ):
                raise ValueError(f"invalid sampled repeats at episode {episode}: {run}")
            successful = [
                row for row in cells
                if row["terminated"] == "True" and row["truncated"] == "False"
            ]
            if not successful:
                raise ValueError(f"no successful validation trajectory at episode {episode}: {run}")
            if (
                int(log[episode]["truncated_count"])
                != sum(row["truncated"] == "True" for row in cells)
                or not math.isclose(
                    len(successful) / len(cells),
                    float(log[episode]["completion_rate"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"validation success statistics disagree with sampled rows at episode {episode}: {run}")
            computed = statistics.mean(float(row[field]) for row in successful)
            recorded = float(log[episode][SUMMARY_FIELDS[objective]])
            if int(log[episode]["instance_count"]) != 150 or not math.isclose(
                computed, recorded, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(f"validation mean disagrees with sampled rows at episode {episode}: {run}")
            means.append(recorded)
        scale = float(statistics.median(means))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid {objective} scale")
        scales[objective] = scale
        sources[objective] = {
            "run": run.relative_to(root).as_posix(),
            "algorithm_seed": 11,
            "validation_episodes": list(TAIL_EPISODES),
            "successful_trajectory_means": means,
            "config_sha256": file_sha256(config_path),
            "validation_log_sha256": file_sha256(log_path),
            "sampled_validation_rows_sha256": file_sha256(rows_path),
            "summary_sha256": file_sha256(summary_path),
        }
    payload: dict[str, Any] = {
        "schema_version": NORMALIZATION_MANIFEST_SCHEMA,
        "formula": "median(mean_success(J_i, episode=e) for e in [840,880,920,960,1000])",
        "bounded_objective": "q_i=J_i/(s_i+J_i)",
        "validation_dataset_manifest": validation.relative_to(root).as_posix(),
        "validation_dataset_manifest_sha256": validation_sha,
        "sources": sources,
        "scales": scales,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def write_immutable_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        dict(manifest), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    return file_sha256(destination)


def load_normalization_manifest(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected
    ):
        raise ValueError("expected normalization manifest SHA256 must be 64 hex digits")
    source = Path(path)
    if file_sha256(source) != expected:
        raise ValueError("normalization manifest SHA256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NORMALIZATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported normalization manifest schema")
    content_sha = payload.pop("content_sha256", None)
    if content_sha != canonical_json_sha256(payload):
        raise ValueError("normalization manifest content hash is invalid")
    payload["content_sha256"] = content_sha
    if (
        set(payload.get("sources", {})) != set(OBJECTIVE_FIELDS)
        or set(payload.get("scales", {})) != set(OBJECTIVE_FIELDS)
    ):
        raise ValueError("normalization manifest sources or scales are incomplete")
    for objective, row in payload["sources"].items():
        means = row.get("successful_trajectory_means")
        if (
            row.get("validation_episodes") != list(TAIL_EPISODES)
            or not isinstance(means, list)
            or len(means) != 5
        ):
            raise ValueError("normalization source has incomplete validation records")
        scale = float(payload["scales"][objective])
        if not math.isfinite(scale) or scale <= 0 or not math.isclose(
            scale, statistics.median(float(v) for v in means), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("normalization scale disagrees with source means")
    return payload


def apply_normalization_manifest(config: dict[str, Any], *, project_root: Path) -> None:
    scalarizer = config.get("objective_scalarizer")
    if not isinstance(scalarizer, dict):
        raise ValueError("config requires objective_scalarizer")
    source = str(scalarizer.get("scale_source", "frozen_manifest"))
    if source != "frozen_manifest":
        raise ValueError(f"unknown objective scalarizer scale_source {source!r}")
    manifest_path = scalarizer.get("normalization_manifest")
    expected = scalarizer.get("normalization_manifest_sha256")
    if not manifest_path or not expected:
        raise ValueError("frozen scales require manifest path and SHA256")
    path = Path(str(manifest_path))
    if not path.is_absolute():
        path = project_root / path
    manifest = load_normalization_manifest(path, expected_sha256=str(expected))
    scalarizer["scales"] = {
        name: float(manifest["scales"][name]) for name in OBJECTIVE_FIELDS
    }
    scalarizer["normalization_manifest"] = str(path.resolve())
    scalarizer["normalization_manifest_sha256"] = str(expected).lower()
    scalarizer["normalization_manifest_content_sha256"] = manifest["content_sha256"]
    network = config.setdefault("network", {})
    if not isinstance(network, dict):
        raise TypeError("network config must be an object")
    network["normalization_manifest_sha256"] = str(expected).lower()
