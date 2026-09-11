from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


NORMALIZATION_MANIFEST_SCHEMA = "v8_objective_normalization_manifest_v1"
OBJECTIVES = ("flow", "cost", "variance")
SPECIALIST_SEEDS = (11, 23, 37, 53, 71)


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


def build_normalization_manifest(
    specialist_rows: Sequence[Mapping[str, Any]],
    *,
    audit_dataset_path: str | Path,
) -> dict[str, Any]:
    """Build the immutable 15-specialist V8 normalization manifest payload."""

    rows = [dict(row) for row in specialist_rows]
    expected = {(objective, seed) for objective in OBJECTIVES for seed in SPECIALIST_SEEDS}
    observed = {(str(row.get("objective")), int(row.get("seed", -1))) for row in rows}
    if len(rows) != 15 or observed != expected:
        raise ValueError("normalization requires exactly 3 objectives x 5 seeds")
    audit_path = Path(audit_dataset_path).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit_sha256 = file_sha256(audit_path)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        checkpoint = Path(str(row["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        raw_mean = float(row["raw_objective_mean"])
        if not math.isfinite(raw_mean) or raw_mean <= 0.0:
            raise ValueError("specialist raw objective means must be finite and positive")
        recorded_audit_sha = row.get("audit_dataset_sha256")
        if recorded_audit_sha is None or str(recorded_audit_sha).lower() != audit_sha256:
            raise ValueError("all specialist audits must use the supplied audit dataset")
        if (
            int(row.get("audit_instance_offset", -1)) != 50
            or int(row.get("audit_instance_count", -1)) != 200
        ):
            raise ValueError(
                "specialist normalization audits require the shared offset=50,count=200 slice"
            )
        normalized_rows.append(
            {
                "objective": str(row["objective"]),
                "seed": int(row["seed"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "raw_objective_mean": raw_mean,
            }
        )
    scales = {
        objective: float(
            statistics.median(
                row["raw_objective_mean"]
                for row in normalized_rows
                if row["objective"] == objective
            )
        )
        for objective in OBJECTIVES
    }
    prediction_upper_bounds = {}
    for objective in OBJECTIVES:
        values = [
            row["raw_objective_mean"]
            for row in normalized_rows
            if row["objective"] == objective
        ]
        mean = statistics.mean(values)
        standard_deviation = statistics.stdev(values)
        # One-sided 95% prediction upper bound for a future seed, df=4.
        prediction_upper_bounds[objective] = float(
            mean + 2.131846786 * standard_deviation * (1.0 + 1.0 / 5.0) ** 0.5
        )
    payload: dict[str, Any] = {
        "schema_version": NORMALIZATION_MANIFEST_SCHEMA,
        "formula": {
            "scale": "median(endpoint_specialist_audit_raw_objective_mean_across_5_seeds)",
            "bounded_objective": "q_i=J_i/(s_i+J_i)",
            "scalarizer": "T=(max_i(lambda_i*q_i)+0.05*sum_i(lambda_i*q_i))/1.05",
        },
        "specialist_seeds": list(SPECIALIST_SEEDS),
        "audit_dataset": str(audit_path),
        "audit_dataset_sha256": audit_sha256,
        "audit_dataset_selection": {
            "instance_offset": 50,
            "instance_count": 200,
        },
        "specialists": sorted(
            normalized_rows, key=lambda row: (OBJECTIVES.index(row["objective"]), row["seed"])
        ),
        "scales": scales,
        "endpoint_prediction_upper_bounds": prediction_upper_bounds,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def write_immutable_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    return file_sha256(destination)


def load_normalization_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path)
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected normalization manifest SHA256 must be a 64-digit hex digest")
    actual = file_sha256(source)
    if actual.lower() != expected_sha256:
        raise ValueError(
            f"normalization manifest SHA256 mismatch: expected={expected_sha256}, actual={actual}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NORMALIZATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported normalization manifest schema")
    content_sha = payload.pop("content_sha256", None)
    if content_sha != canonical_json_sha256(payload):
        raise ValueError("normalization manifest content hash is invalid")
    payload["content_sha256"] = content_sha
    if tuple(payload.get("specialist_seeds", ())) != SPECIALIST_SEEDS:
        raise ValueError("normalization manifest specialist seeds are invalid")
    scales = payload.get("scales")
    if not isinstance(scales, dict) or set(scales) != set(OBJECTIVES):
        raise ValueError("normalization manifest scales are incomplete")
    if any(
        not math.isfinite(float(scales[name])) or float(scales[name]) <= 0.0
        for name in OBJECTIVES
    ):
        raise ValueError("normalization manifest scales must be finite and positive")
    specialists = payload.get("specialists")
    if not isinstance(specialists, list) or len(specialists) != 15:
        raise ValueError("normalization manifest must contain all 15 specialists")
    observed = {
        (str(row.get("objective")), int(row.get("seed", -1)))
        for row in specialists
        if isinstance(row, dict)
    }
    expected = {
        (objective, seed)
        for objective in OBJECTIVES
        for seed in SPECIALIST_SEEDS
    }
    if observed != expected:
        raise ValueError("normalization manifest specialist coverage is invalid")
    return payload


def apply_normalization_manifest(config: dict[str, Any], *, project_root: Path) -> None:
    scalarizer = config.get("objective_scalarizer")
    if not isinstance(scalarizer, dict):
        raise ValueError("V8 config requires objective_scalarizer")
    source = str(scalarizer.get("scale_source", "bootstrap_specialist"))
    if source == "bootstrap_specialist":
        return
    if source != "frozen_manifest":
        raise ValueError(f"unknown objective scalarizer scale_source {source!r}")
    manifest_path = scalarizer.get("normalization_manifest")
    expected = scalarizer.get("normalization_manifest_sha256")
    if not manifest_path or not expected:
        raise ValueError("universal training requires manifest path and expected SHA256")
    path = Path(str(manifest_path))
    if not path.is_absolute():
        path = project_root / path
    manifest = load_normalization_manifest(path, expected_sha256=str(expected))
    scalarizer["scales"] = {
        name: float(manifest["scales"][name]) for name in OBJECTIVES
    }
    scalarizer["normalization_manifest"] = str(path.resolve())
    scalarizer["normalization_manifest_sha256"] = str(expected).lower()
    scalarizer["normalization_manifest_content_sha256"] = manifest["content_sha256"]
    network = config.setdefault("network", {})
    if not isinstance(network, dict):
        raise TypeError("network config must be an object")
    network["normalization_manifest_sha256"] = str(expected).lower()
    two_stage = config.setdefault("training", {}).setdefault("two_stage", {})
    pareto = two_stage.setdefault("pareto_promotion", {})
    pareto["endpoint_prediction_upper_bounds"] = dict(
        manifest["endpoint_prediction_upper_bounds"]
    )
